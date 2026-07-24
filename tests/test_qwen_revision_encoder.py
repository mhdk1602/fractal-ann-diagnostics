from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fractal_ann_diagnostics.qwen_revision_encoder as qre
from fractal_ann_diagnostics.artifact_integrity import DirectoryDigest


def _active_rows(batch: qre._TokenizedBatch) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for input_ids, mask in zip(batch.input_ids, batch.attention_mask, strict=True):
        rows.append(tuple(int(value) for value in input_ids[np.asarray(mask, dtype=bool)]))
    return tuple(rows)


def _batch_from_rows(rows: tuple[tuple[int, ...], ...]) -> qre._TokenizedBatch:
    width = max(len(row) for row in rows)
    input_ids = np.zeros((len(rows), width), dtype=np.int64)
    attention_mask = np.zeros((len(rows), width), dtype=np.int64)
    for position, row in enumerate(rows):
        input_ids[position, -len(row) :] = row
        attention_mask[position, -len(row) :] = 1
    return qre._TokenizedBatch(input_ids, attention_mask, opaque={})


class _FakeBackend:
    def __init__(
        self,
        mode: qre.TokenizationMode,
        *,
        drift: str | None = None,
    ) -> None:
        self.mode = mode
        self.drift = drift
        self.tokenize_calls: list[dict[str, object]] = []
        self.forward_calls: list[dict[str, object]] = []
        self.invalid_hidden: str | None = None

    @staticmethod
    def _content_ids(text: str, max_length: int) -> list[int]:
        values = [257 + (byte % 200) for byte in text.encode("utf-8")]
        return values[:max_length]

    def tokenize(
        self,
        texts: tuple[str, ...] | list[str],
        *,
        add_special_tokens: bool,
        max_length: int,
    ) -> qre._TokenizedBatch:
        self.tokenize_calls.append(
            {
                "add_special_tokens": add_special_tokens,
                "max_length": max_length,
                "texts": tuple(texts),
            }
        )
        rows: list[tuple[int, ...]] = []
        for text in texts:
            reserve = 1 if add_special_tokens and self.mode == "automatic-terminal" else 0
            content = self._content_ids(text, max_length - reserve)
            if self.drift == "baseline-terminal" and not add_special_tokens:
                content[-1:] = [qre.QWEN_TERMINAL_TOKEN_ID]
            if add_special_tokens:
                if self.mode == "automatic-terminal" and self.drift != "missing-terminal":
                    token = (
                        qre.QWEN_TERMINAL_TOKEN_ID - 1
                        if self.drift == "wrong-terminal"
                        else qre.QWEN_TERMINAL_TOKEN_ID
                    )
                    content.append(token)
                elif self.mode == "no-terminal" and self.drift == "stale-appends-terminal":
                    content = content[: max_length - 1]
                    content.append(qre.QWEN_TERMINAL_TOKEN_ID)
            rows.append(tuple(content))

        batch = _batch_from_rows(tuple(rows))
        if self.drift == "right-padding":
            batch = qre._TokenizedBatch(
                np.pad(batch.input_ids, ((0, 0), (0, 1))),
                np.pad(batch.attention_mask, ((0, 0), (0, 1))),
                opaque={},
            )
        return batch

    def forward(self, batch: qre._TokenizedBatch, *, seed: int) -> np.ndarray:
        self.forward_calls.append({"rows": _active_rows(batch), "seed": seed})
        rows, width = batch.input_ids.shape
        hidden_size = 128 if self.invalid_hidden == "short" else 300
        hidden = np.empty((rows, width, hidden_size), dtype=np.float32)
        dimensions = np.arange(hidden_size, dtype=np.float32)
        for row in range(rows):
            content_positions = qre._content_position_ids(batch.attention_mask[row : row + 1])[0]
            for position in range(width):
                token = float(batch.input_ids[row, position])
                hidden[row, position] = (
                    np.sin((token + 1.0) * (dimensions + 1.0) * 0.0001)
                    + np.cos(
                        (dimensions + 3.0) * 0.017 + float(content_positions[position]) * 0.013
                    )
                ).astype(np.float32)
        if self.invalid_hidden == "nan":
            hidden[0, 0, 0] = np.nan
        if self.invalid_hidden == "zero":
            last = np.flatnonzero(batch.attention_mask[0])[-1]
            hidden[0, last, : qre.QWEN_OUTPUT_DIMENSION] = 0.0
        return hidden


def _install_encoder_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    drift: str | None = None,
) -> tuple[list[_FakeBackend], list[tuple[str, str]]]:
    backends: list[_FakeBackend] = []
    verifications: list[tuple[str, str]] = []

    def backend_factory(
        model_path: Path,
        config: qre.QwenRevisionEncoderConfig,
    ) -> _FakeBackend:
        del model_path
        backend = _FakeBackend(config.tokenization_mode, drift=drift)
        backends.append(backend)
        return backend

    def verifier(
        model_path: str | Path,
        config: qre.QwenRevisionEncoderConfig,
    ) -> None:
        verifications.append((str(model_path), config.model_tree_sha256))

    monkeypatch.setattr(qre, "_TransformersBackend", backend_factory)
    monkeypatch.setattr(qre, "verify_qwen_revision_tree", verifier)
    return backends, verifications


class _FakePairedBackend:
    def __init__(self, *, noncausal: bool = False) -> None:
        self.current = _FakeBackend("automatic-terminal")
        self.stale = _FakeBackend("no-terminal")
        self.noncausal = noncausal
        self.forward_calls: list[dict[str, object]] = []

    def tokenize(
        self,
        arm: qre.QwenArm,
        texts: tuple[str, ...] | list[str],
        *,
        add_special_tokens: bool,
        max_length: int,
    ) -> qre._TokenizedBatch:
        backend = self.current if arm == "current" else self.stale
        return backend.tokenize(
            texts,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
        )

    def forward_selected(
        self,
        rows: tuple[tuple[int, ...], ...],
        selections: tuple[tuple[int, int], ...],
        *,
        output_dimension: int,
        seed: int,
    ) -> np.ndarray:
        batch = _batch_from_rows(rows)
        hidden = self.stale.forward(batch, seed=seed)
        if self.noncausal:
            for row, values in enumerate(rows):
                hidden[row] += float(values[-1]) * 0.01
        selected = np.stack(
            [hidden[row, position, :output_dimension] for row, position in selections]
        )
        self.forward_calls.append({"rows": rows, "seed": seed, "selections": selections})
        return selected


def _install_paired_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    noncausal: bool = False,
) -> list[_FakePairedBackend]:
    backends: list[_FakePairedBackend] = []

    def factory(
        current_path: Path,
        stale_path: Path,
        config: qre.QwenRevisionEncoderConfig,
    ) -> _FakePairedBackend:
        del current_path, stale_path, config
        backend = _FakePairedBackend(noncausal=noncausal)
        backends.append(backend)
        return backend

    monkeypatch.setattr(qre, "_PairedTransformersBackend", factory)
    monkeypatch.setattr(qre, "verify_qwen_revision_tree", lambda path, config: None)
    return backends


def _independent_fake_vectors(
    texts: tuple[str, ...],
    *,
    role: qre.TextRole,
) -> tuple[np.ndarray, np.ndarray]:
    prompt = qre.QWEN_QUERY_PROMPT if role == "query" else qre.QWEN_DOCUMENT_PROMPT
    rendered = tuple(prompt + text for text in texts)
    current_backend = _FakeBackend("automatic-terminal")
    stale_backend = _FakeBackend("no-terminal")
    current_batch = current_backend.tokenize(
        rendered, add_special_tokens=True, max_length=qre.QWEN_MAX_SEQUENCE_LENGTH
    )
    stale_batch = stale_backend.tokenize(
        rendered, add_special_tokens=True, max_length=qre.QWEN_MAX_SEQUENCE_LENGTH
    )
    current = qre._pool_truncate_normalize(
        current_backend.forward(current_batch, seed=1), current_batch.attention_mask
    )
    stale = qre._pool_truncate_normalize(
        stale_backend.forward(stale_batch, seed=2), stale_batch.attention_mask
    )
    return current, stale


def test_config_closes_every_frozen_intervention_field() -> None:
    current = qre.QwenRevisionEncoderConfig.for_arm("current", batch_size=3)
    stale = qre.QwenRevisionEncoderConfig.for_arm("stale", batch_size=3)

    assert current.query_prompt.endswith("\nQuery:")
    assert not current.query_prompt.endswith(" ")
    assert current.tokenization_mode == "automatic-terminal"
    assert stale.tokenization_mode == "no-terminal"
    assert current.model_sha256 == stale.model_sha256 == qre.QWEN_MODEL_SHA256
    assert current.model_tree_sha256 != stale.model_tree_sha256
    assert current.tokenizer_sha256 != stale.tokenizer_sha256
    assert len(current.sha256) == 64

    with pytest.raises(qre.QwenRevisionEncoderError, match="study prompt"):
        replace(current, query_prompt=current.query_prompt + " ")
    with pytest.raises(qre.QwenRevisionEncoderError, match="study prompt"):
        replace(current, query_prompt="Query:")
    with pytest.raises(qre.QwenRevisionEncoderError, match="tokenization_mode"):
        replace(current, tokenization_mode="no-terminal")
    with pytest.raises(qre.QwenRevisionEncoderError, match="model_revision"):
        replace(current, model_revision=qre.QWEN_STALE_REVISION)
    with pytest.raises(qre.QwenRevisionEncoderError, match="max_sequence_length"):
        replace(current, max_sequence_length=511)
    with pytest.raises(qre.QwenRevisionEncoderError, match="output_dimension"):
        replace(current, output_dimension=1024)
    with pytest.raises(qre.QwenRevisionEncoderError, match="normalize"):
        replace(current, normalize=False)


def test_config_loader_requires_closed_canonical_json() -> None:
    config = qre.QwenRevisionEncoderConfig.for_arm("current")
    encoded = config.canonical_bytes() + b"\n"
    assert qre.loads_qwen_revision_encoder_config(encoded) == config

    unknown = {**config.to_dict(), "network": "allowed"}
    with pytest.raises(qre.QwenRevisionEncoderError, match="unknown=.*network"):
        qre.loads_qwen_revision_encoder_config(
            json.dumps(unknown, sort_keys=True, separators=(",", ":")) + "\n"
        )
    with pytest.raises(qre.QwenRevisionEncoderError, match="canonical JSON"):
        qre.loads_qwen_revision_encoder_config(json.dumps(config.to_dict()) + "\n")
    with pytest.raises(qre.QwenRevisionEncoderError, match="duplicate key"):
        qre.loads_qwen_revision_encoder_config(b'{"arm":"current","arm":"stale"}\n')


def test_current_encoder_binds_prompt_terminal_pooling_and_batch_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backends, verifications = _install_encoder_fakes(monkeypatch)
    config = qre.QwenRevisionEncoderConfig.for_arm("current", batch_size=2)
    encoder = qre.QwenRevisionEncoder(tmp_path.resolve(), config)
    backend = backends[0]
    backend.tokenize_calls.clear()

    texts = ("alpha", "beta", "gamma")
    first = encoder.encode_queries(texts)
    first_seeds = [call["seed"] for call in backend.forward_calls]
    second = encoder.encode_queries(texts)
    second_seeds = [call["seed"] for call in backend.forward_calls[2:]]

    assert first.shape == (3, 256)
    assert first.dtype == np.float32
    assert not first.flags.writeable
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=2e-6)
    np.testing.assert_array_equal(first, second)
    assert first_seeds == second_seeds
    assert len(set(first_seeds)) == 2

    first_primary, first_baseline = backend.tokenize_calls[:2]
    assert first_primary == {
        "add_special_tokens": True,
        "max_length": 512,
        "texts": (
            qre.QWEN_QUERY_PROMPT + "alpha",
            qre.QWEN_QUERY_PROMPT + "beta",
        ),
    }
    assert first_baseline["add_special_tokens"] is False
    assert first_baseline["max_length"] == 511
    assert len(backend.forward_calls[0]["rows"]) == 2
    assert len(backend.forward_calls[1]["rows"]) == 1
    assert len(verifications) == 6


def test_stale_encoder_uses_empty_document_prompt_and_no_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backends, _ = _install_encoder_fakes(monkeypatch)
    config = qre.QwenRevisionEncoderConfig.for_arm("stale", batch_size=4)
    encoder = qre.QwenRevisionEncoder(tmp_path.resolve(), config)
    backend = backends[0]
    backend.tokenize_calls.clear()

    vectors = encoder.encode_documents(("document text",))

    primary, baseline = backend.tokenize_calls[:2]
    assert backend.tokenize_calls[2:] == [primary, baseline]
    assert primary == {
        "add_special_tokens": True,
        "max_length": 512,
        "texts": ("document text",),
    }
    assert baseline == {
        "add_special_tokens": False,
        "max_length": 512,
        "texts": ("document text",),
    }
    assert backend.forward_calls[-1]["rows"][0][-1] != qre.QWEN_TERMINAL_TOKEN_ID
    assert vectors.shape == (1, 256)


def test_reciprocal_terminal_controls_are_byte_identical() -> None:
    rendered = (qre.QWEN_QUERY_PROMPT + "control query",)
    current = _FakeBackend("automatic-terminal")
    stale = _FakeBackend("no-terminal")

    current_standard = current.tokenize(rendered, add_special_tokens=True, max_length=512)
    current_without_special = current.tokenize(
        rendered,
        add_special_tokens=False,
        max_length=512,
    )
    stale_standard = stale.tokenize(rendered, add_special_tokens=True, max_length=512)
    stale_rows = _active_rows(stale_standard)
    stale_manual_terminal = _batch_from_rows(
        tuple((*row, qre.QWEN_TERMINAL_TOKEN_ID) for row in stale_rows)
    )

    current_vectors = qre._pool_truncate_normalize(
        current.forward(current_standard, seed=7),
        current_standard.attention_mask,
    )
    restored_vectors = qre._pool_truncate_normalize(
        stale.forward(stale_manual_terminal, seed=7),
        stale_manual_terminal.attention_mask,
    )
    stale_vectors = qre._pool_truncate_normalize(
        stale.forward(stale_standard, seed=7),
        stale_standard.attention_mask,
    )
    suppressed_vectors = qre._pool_truncate_normalize(
        current.forward(current_without_special, seed=7),
        current_without_special.attention_mask,
    )

    assert _active_rows(current_standard) == _active_rows(stale_manual_terminal)
    assert _active_rows(current_without_special) == _active_rows(stale_standard)
    assert current_vectors.tobytes() == restored_vectors.tobytes()
    assert stale_vectors.tobytes() == suppressed_vectors.tobytes()
    assert current_vectors.tobytes() != stale_vectors.tobytes()


def test_pooling_uses_attention_mask_before_truncation_and_normalization() -> None:
    hidden = np.zeros((1, 4, 300), dtype=np.float64)
    hidden[0, 1, :256] = np.arange(1, 257, dtype=np.float64)
    hidden[0, 3, :256] = 100_000.0
    mask = np.array([[1, 1, 0, 0]], dtype=np.int64)

    vector = qre._pool_truncate_normalize(hidden, mask)
    expected = np.arange(1, 257, dtype=np.float32)
    expected /= np.sqrt(np.sum(expected * expected, dtype=np.float32)).astype(np.float32)

    np.testing.assert_array_equal(vector[0], expected)
    assert vector.dtype == np.float32
    assert not vector.flags.writeable


def test_content_position_ids_remove_batch_padding_as_a_hidden_treatment() -> None:
    mask = np.array([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=np.int64)
    positions = qre._content_position_ids(mask)
    np.testing.assert_array_equal(
        positions,
        np.array([[0, 0, 0, 1, 2], [0, 0, 1, 2, 3]], dtype=np.int64),
    )
    with pytest.raises(qre.QwenRevisionEncoderError, match="content positions"):
        qre._content_position_ids(np.array([[0, 2]], dtype=np.int64))


@pytest.mark.parametrize(
    ("arm", "drift", "message"),
    [
        ("current", "missing-terminal", "exactly one automatic terminal"),
        ("current", "wrong-terminal", "exactly one automatic terminal"),
        ("stale", "stale-appends-terminal", "stale no-terminal tokenizer"),
        ("current", "baseline-terminal", "reserved terminal token"),
        ("current", "right-padding", "left padding"),
    ],
)
def test_tokenizer_drift_fails_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: qre.QwenArm,
    drift: str,
    message: str,
) -> None:
    _install_encoder_fakes(monkeypatch, drift=drift)
    with pytest.raises(qre.QwenRevisionEncoderError, match=message):
        qre.QwenRevisionEncoder(
            tmp_path.resolve(),
            qre.QwenRevisionEncoderConfig.for_arm(arm),
        )


@pytest.mark.parametrize("invalid", ["short", "nan", "zero"])
def test_invalid_model_outputs_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    backends, _ = _install_encoder_fakes(monkeypatch)
    encoder = qre.QwenRevisionEncoder(
        tmp_path.resolve(),
        qre.QwenRevisionEncoderConfig.for_arm("current"),
    )
    backends[0].invalid_hidden = invalid

    with pytest.raises(qre.QwenRevisionEncoderError, match="shape|finite|normalized"):
        encoder.encode_queries(("query",))


def test_tree_is_verified_before_load_and_around_every_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backends: list[_FakeBackend] = []
    calls = 0

    def backend_factory(
        model_path: Path,
        config: qre.QwenRevisionEncoderConfig,
    ) -> _FakeBackend:
        del model_path
        backend = _FakeBackend(config.tokenization_mode)
        backends.append(backend)
        return backend

    def verifier(
        model_path: str | Path,
        config: qre.QwenRevisionEncoderConfig,
    ) -> None:
        nonlocal calls
        del model_path, config
        calls += 1
        if calls == 4:
            raise qre.QwenRevisionEncoderError("local Qwen tree digest changed")

    monkeypatch.setattr(qre, "_TransformersBackend", backend_factory)
    monkeypatch.setattr(qre, "verify_qwen_revision_tree", verifier)
    encoder = qre.QwenRevisionEncoder(
        tmp_path.resolve(),
        qre.QwenRevisionEncoderConfig.for_arm("current"),
    )

    with pytest.raises(qre.QwenRevisionEncoderError, match="tree digest changed"):
        encoder.encode_queries(("query",))
    assert calls == 4
    assert len(backends[0].forward_calls) == 1


def test_tree_verifier_rejects_digest_and_accounting_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = qre.QwenRevisionEncoderConfig.for_arm("current")
    wrong = DirectoryDigest(
        sha256="0" * 64,
        entries=(),
        file_count=0,
        directory_count=0,
        byte_count=0,
        observed_file_count=0,
        observed_directory_count=0,
        observed_byte_count=0,
    )
    monkeypatch.setattr(qre, "digest_directory_tree", lambda path: wrong)
    with pytest.raises(qre.QwenRevisionEncoderError, match="tree digest"):
        qre.verify_qwen_revision_tree(tmp_path.resolve(), config)

    binding = qre._ARM_BINDINGS["current"]
    wrong_accounting = replace(wrong, sha256=binding.tree_sha256)
    monkeypatch.setattr(qre, "digest_directory_tree", lambda path: wrong_accounting)
    with pytest.raises(qre.QwenRevisionEncoderError, match="tree accounting"):
        qre.verify_qwen_revision_tree(tmp_path.resolve(), config)


def test_transformers_loader_is_offline_local_only_and_non_remote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_calls: list[tuple[str, dict[str, object], dict[str, str | None]]] = []
    model_calls: list[tuple[str, dict[str, object], dict[str, str | None]]] = []
    offline_names = (
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )

    class FakeTokenizer:
        is_fast = True
        padding_side = "left"

        def __len__(self) -> int:
            return qre.QWEN_VOCAB_SIZE

        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == qre.QWEN_TERMINAL_TOKEN
            return qre.QWEN_TERMINAL_TOKEN_ID

    class FakeModel:
        config = SimpleNamespace(
            hidden_size=qre.QWEN_HIDDEN_SIZE,
            model_type="qwen3",
            vocab_size=qre.QWEN_VOCAB_SIZE,
        )

        def __init__(self) -> None:
            self.device: str | None = None
            self.evaluated = False

        def to(self, device: str) -> None:
            self.device = device

        def eval(self) -> None:
            self.evaluated = True

    tokenizer = FakeTokenizer()
    model = FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeTokenizer:
            tokenizer_calls.append(
                (
                    path,
                    dict(kwargs),
                    {name: os.environ.get(name) for name in offline_names},
                )
            )
            return tokenizer

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeModel:
            model_calls.append(
                (
                    path,
                    dict(kwargs),
                    {name: os.environ.get(name) for name in offline_names},
                )
            )
            return model

    fake_torch = SimpleNamespace(float32="float32")
    monkeypatch.setattr(
        qre,
        "_import_transformers_dependencies",
        lambda: (fake_torch, FakeAutoTokenizer, FakeAutoModel),
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous")

    qre._TransformersBackend(
        tmp_path.resolve(),
        qre.QwenRevisionEncoderConfig.for_arm("current", device="cpu"),
    )

    assert tokenizer_calls[0][0] == str(tmp_path.resolve())
    assert tokenizer_calls[0][1] == {
        "local_files_only": True,
        "padding_side": "left",
        "trust_remote_code": False,
        "use_fast": True,
    }
    assert all(value == "1" for value in tokenizer_calls[0][2].values())
    assert model_calls[0][1] == {
        "attn_implementation": "eager",
        "dtype": "float32",
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert all(value == "1" for value in model_calls[0][2].values())
    assert os.environ["HF_HUB_OFFLINE"] == "previous"
    assert "HF_DATASETS_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert model.device == "cpu"
    assert model.evaluated is True


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("slow-tokenizer", "fast tokenizer"),
        ("token-count", "vocabulary size drifted"),
        ("terminal-map", "terminal-token mapping drifted"),
        ("model-shape", "model shape or type drifted"),
    ],
)
def test_transformers_loader_rejects_loaded_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    class FakeTokenizer:
        is_fast = drift != "slow-tokenizer"
        padding_side = "left"

        def __len__(self) -> int:
            return qre.QWEN_VOCAB_SIZE - (drift == "token-count")

        def convert_tokens_to_ids(self, token: str) -> int:
            del token
            return qre.QWEN_TERMINAL_TOKEN_ID - (drift == "terminal-map")

    class FakeModel:
        config = SimpleNamespace(
            hidden_size=qre.QWEN_HIDDEN_SIZE - (drift == "model-shape"),
            model_type="qwen3",
            vocab_size=qre.QWEN_VOCAB_SIZE,
        )

        def to(self, device: str) -> None:
            del device

        def eval(self) -> None:
            pass

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeTokenizer:
            del path, kwargs
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeModel:
            del path, kwargs
            return FakeModel()

    monkeypatch.setattr(
        qre,
        "_import_transformers_dependencies",
        lambda: (SimpleNamespace(float32="float32"), FakeAutoTokenizer, FakeAutoModel),
    )

    with pytest.raises(qre.QwenRevisionEncoderError, match=message):
        qre._TransformersBackend(
            tmp_path.resolve(),
            qre.QwenRevisionEncoderConfig.for_arm("current"),
        )


def test_text_validation_rejects_implicit_normalization_and_empty_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_encoder_fakes(monkeypatch)
    encoder = qre.QwenRevisionEncoder(
        tmp_path.resolve(),
        qre.QwenRevisionEncoderConfig.for_arm("current"),
    )

    with pytest.raises(qre.QwenRevisionEncoderError, match="non-empty sequence"):
        encoder.encode_queries(())
    with pytest.raises(qre.QwenRevisionEncoderError, match="non-empty string"):
        encoder.encode_documents(("",))
    with pytest.raises(qre.QwenRevisionEncoderError, match="NFC"):
        encoder.encode_queries(("e\u0301",))


def test_embedding_store_adapter_binds_every_encoder_argument(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path.resolve()
    calls: list[tuple[str, tuple[str, ...]]] = []

    class FakeEncoder:
        def __init__(self, model_path: Path, config: qre.QwenRevisionEncoderConfig) -> None:
            assert model_path == model
            assert config.arm == "current"

        def encode_queries(self, texts: tuple[str, ...]) -> np.ndarray:
            calls.append(("query", tuple(texts)))
            return np.ones((len(texts), qre.QWEN_OUTPUT_DIMENSION), dtype=np.float32)

        def encode_documents(self, texts: tuple[str, ...]) -> np.ndarray:
            calls.append(("document", tuple(texts)))
            return np.ones((len(texts), qre.QWEN_OUTPUT_DIMENSION), dtype=np.float32)

    monkeypatch.setattr(qre, "QwenRevisionEncoder", FakeEncoder)
    config = qre.QwenRevisionEncoderConfig.for_arm("current", batch_size=2, device="cpu")
    adapter = qre.QwenRevisionEmbeddingAdapter(config)
    common = {
        "model_path": model,
        "max_sequence_length": config.max_sequence_length,
        "output_dimension": config.output_dimension,
        "normalize": config.normalize,
        "device": config.device,
        "seed": 17,
    }

    query = adapter.encode(("query",), prompt=config.query_prompt, **common)
    document = adapter.encode(("document",), prompt=config.document_prompt, **common)

    assert query.shape == document.shape == (1, qre.QWEN_OUTPUT_DIMENSION)
    assert calls == [("query", ("query",)), ("document", ("document",))]
    assert config.sha256 in adapter.implementation_id

    mutations = (
        {"prompt": "wrong"},
        {"prompt": config.query_prompt, "max_sequence_length": 511},
        {"prompt": config.query_prompt, "output_dimension": 128},
        {"prompt": config.query_prompt, "normalize": False},
        {"prompt": config.query_prompt, "device": "mps"},
        {"prompt": config.query_prompt, "seed": -1},
    )
    for mutation in mutations:
        arguments = {**common, "prompt": config.query_prompt, **mutation}
        with pytest.raises(qre.QwenRevisionEncoderError, match="embedding-store"):
            adapter.encode(("query",), **arguments)


def test_embedding_store_adapter_cannot_switch_model_trees(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = (tmp_path / "first").resolve()
    second = (tmp_path / "second").resolve()
    first.mkdir()
    second.mkdir()

    class FakeEncoder:
        def __init__(self, _path: Path, _config: qre.QwenRevisionEncoderConfig) -> None:
            pass

        def encode_queries(self, texts: tuple[str, ...]) -> np.ndarray:
            return np.ones((len(texts), qre.QWEN_OUTPUT_DIMENSION), dtype=np.float32)

        def encode_documents(self, texts: tuple[str, ...]) -> np.ndarray:
            return self.encode_queries(texts)

    monkeypatch.setattr(qre, "QwenRevisionEncoder", FakeEncoder)
    config = qre.QwenRevisionEncoderConfig.for_arm("current")
    adapter = qre.QwenRevisionEmbeddingAdapter(config)
    arguments = {
        "prompt": config.query_prompt,
        "max_sequence_length": config.max_sequence_length,
        "output_dimension": config.output_dimension,
        "normalize": True,
        "device": config.device,
        "seed": 1,
    }
    adapter.encode(("query",), model_path=first, **arguments)
    with pytest.raises(qre.QwenRevisionEncoderError, match="cannot switch"):
        adapter.encode(("query",), model_path=second, **arguments)


@pytest.mark.parametrize(
    ("texts", "expected_forward_rows"),
    [
        (("alpha", "a longer beta", "gamma"), (3,)),
        (("x" * 600, "short"), (1, 1, 1)),
    ],
)
def test_paired_encoder_matches_independent_arms_in_one_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    texts: tuple[str, ...],
    expected_forward_rows: tuple[int, ...],
) -> None:
    backends = _install_paired_fakes(monkeypatch)
    current_path = tmp_path / "current"
    stale_path = tmp_path / "stale"
    current_path.mkdir()
    stale_path.mkdir()
    current_config = qre.QwenRevisionEncoderConfig.for_arm("current", batch_size=8)
    stale_config = qre.QwenRevisionEncoderConfig.for_arm("stale", batch_size=8)
    encoder = qre.QwenPairedRevisionEncoder(
        current_path.resolve(),
        stale_path.resolve(),
        current_config,
        stale_config,
    )
    backend = backends[0]
    backend.forward_calls.clear()

    observed_current, observed_stale = encoder.encode_queries(texts)
    expected_current, expected_stale = _independent_fake_vectors(texts, role="query")

    np.testing.assert_array_equal(observed_current, expected_current)
    np.testing.assert_array_equal(observed_stale, expected_stale)
    assert tuple(len(call["rows"]) for call in backend.forward_calls) == expected_forward_rows
    assert not observed_current.flags.writeable
    assert not observed_stale.flags.writeable


def test_paired_encoder_rejects_noncausal_prefix_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_paired_fakes(monkeypatch, noncausal=True)
    current_path = tmp_path / "current"
    stale_path = tmp_path / "stale"
    current_path.mkdir()
    stale_path.mkdir()
    with pytest.raises(qre.QwenRevisionEncoderError, match="causal prefix-invariance"):
        qre.QwenPairedRevisionEncoder(
            current_path.resolve(),
            stale_path.resolve(),
            qre.QwenRevisionEncoderConfig.for_arm("current"),
            qre.QwenRevisionEncoderConfig.for_arm("stale"),
        )


def test_paired_encoder_rejects_post_verification_tree_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_paired_fakes(monkeypatch)
    current_path = tmp_path / "current"
    stale_path = tmp_path / "stale"
    current_path.mkdir()
    stale_path.mkdir()
    encoder = qre.QwenPairedRevisionEncoder(
        current_path.resolve(),
        stale_path.resolve(),
        qre.QwenRevisionEncoderConfig.for_arm("current"),
        qre.QwenRevisionEncoderConfig.for_arm("stale"),
    )
    (stale_path / "mutation").write_bytes(b"changed")
    with pytest.raises(qre.QwenRevisionEncoderError, match="tree changed"):
        encoder.encode_documents(("document",))


def test_paired_adapter_binds_paths_configs_and_store_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path, str, tuple[str, ...]]] = []

    class FakePairedEncoder:
        def __init__(
            self,
            current_path: Path,
            stale_path: Path,
            current_config: qre.QwenRevisionEncoderConfig,
            stale_config: qre.QwenRevisionEncoderConfig,
        ) -> None:
            assert current_config.arm == "current"
            assert stale_config.arm == "stale"
            self.paths = (current_path, stale_path)

        def encode_queries(self, texts: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
            calls.append((*self.paths, "query", tuple(texts)))
            value = np.ones((len(texts), qre.QWEN_OUTPUT_DIMENSION), dtype=np.float32)
            return value, value

        def encode_documents(self, texts: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
            calls.append((*self.paths, "document", tuple(texts)))
            value = np.ones((len(texts), qre.QWEN_OUTPUT_DIMENSION), dtype=np.float32)
            return value, value

    monkeypatch.setattr(qre, "QwenPairedRevisionEncoder", FakePairedEncoder)
    current_path = tmp_path / "current"
    stale_path = tmp_path / "stale"
    replacement = tmp_path / "replacement"
    current_path.mkdir()
    stale_path.mkdir()
    replacement.mkdir()
    current = qre.QwenRevisionEncoderConfig.for_arm("current")
    stale = qre.QwenRevisionEncoderConfig.for_arm("stale")
    adapter = qre.QwenPairedRevisionEmbeddingAdapter(current, stale)
    common = {
        "current_model_path": current_path.resolve(),
        "old_model_path": stale_path.resolve(),
        "prompt": current.query_prompt,
        "max_sequence_length": current.max_sequence_length,
        "output_dimension": current.output_dimension,
        "normalize": True,
        "device": current.device,
        "seed": 5,
    }
    vectors = adapter.encode_pair(("query",), **common)
    assert vectors[0].shape == vectors[1].shape == (1, qre.QWEN_OUTPUT_DIMENSION)
    assert calls[-1][2:] == ("query", ("query",))
    assert adapter.current_implementation_id != adapter.old_implementation_id
    assert current.sha256 not in adapter.old_implementation_id

    with pytest.raises(qre.QwenRevisionEncoderError, match="binding differs"):
        adapter.encode_pair(("query",), **{**common, "output_dimension": 128})
    with pytest.raises(qre.QwenRevisionEncoderError, match="cannot switch"):
        adapter.encode_pair(
            ("query",),
            **{**common, "old_model_path": replacement.resolve()},
        )
