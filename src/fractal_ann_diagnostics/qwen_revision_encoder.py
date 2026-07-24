"""Pinned, offline Qwen revision encoder for the terminal-token intervention."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np

from .artifact_integrity import ArtifactIntegrityError, DirectoryDigest, digest_directory_tree

QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN_CURRENT_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN_STALE_REVISION = "99cabfa1346cbf4ac8b0e73079bb2e286cff3a1f"

QWEN_CURRENT_TREE_SHA256 = "0d1d985a7fb0500d53ebd83d2516ab6324bc9ad92b4fc88487b5a05437aef951"
QWEN_STALE_TREE_SHA256 = "742aaae08f118ef62ac498dba01241dd254a05b75cd7ce3d903c64785a8231df"
QWEN_MODEL_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
QWEN_MODEL_CONFIG_SHA256 = "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9"
QWEN_CURRENT_TOKENIZER_SHA256 = "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
QWEN_STALE_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

QWEN_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)
QWEN_DOCUMENT_PROMPT = ""
QWEN_MAX_SEQUENCE_LENGTH = 512
QWEN_OUTPUT_DIMENSION = 256
QWEN_TERMINAL_TOKEN = "<|endoftext|>"
QWEN_TERMINAL_TOKEN_ID = 151643
QWEN_HIDDEN_SIZE = 1024
QWEN_VOCAB_SIZE = 151669

QWEN_REVISION_ENCODER_SCHEMA = "fractal-qwen-revision-encoder-config-v2"
QWEN_REVISION_ENCODER_VERSION = "fractal-qwen-revision-encoder-v2"
QWEN_PAIRED_REVISION_ENCODER_VERSION = "fractal-qwen-paired-revision-encoder-v2"
QWEN_LENGTH_BUCKETS = (64, 128, 256, 384, 511, 512)

QwenArm = Literal["current", "stale"]
TokenizationMode = Literal["automatic-terminal", "no-terminal"]
TextRole = Literal["query", "document"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CONFIG_FIELDS = frozenset(
    {
        "arm",
        "batch_size",
        "deterministic_seed",
        "device",
        "document_prompt",
        "dtype",
        "encoder_version",
        "max_sequence_length",
        "model_config_sha256",
        "model_id",
        "model_revision",
        "model_sha256",
        "model_tree_sha256",
        "normalize",
        "output_dimension",
        "padding_side",
        "pooling",
        "query_prompt",
        "schema_version",
        "terminal_token_id",
        "tokenization_mode",
        "tokenizer_sha256",
    }
)


class QwenRevisionEncoderError(ValueError):
    """Raised when a pinned Qwen arm or an encoding batch violates its contract."""


@dataclass(frozen=True)
class _ArmBinding:
    revision: str
    tree_sha256: str
    tokenizer_sha256: str
    tokenization_mode: TokenizationMode
    file_count: int
    directory_count: int
    byte_count: int


_ARM_BINDINGS: Mapping[QwenArm, _ArmBinding] = MappingProxyType(
    {
        "current": _ArmBinding(
            revision=QWEN_CURRENT_REVISION,
            tree_sha256=QWEN_CURRENT_TREE_SHA256,
            tokenizer_sha256=QWEN_CURRENT_TOKENIZER_SHA256,
            tokenization_mode="automatic-terminal",
            file_count=11,
            directory_count=1,
            byte_count=1_207_487_471,
        ),
        "stale": _ArmBinding(
            revision=QWEN_STALE_REVISION,
            tree_sha256=QWEN_STALE_TREE_SHA256,
            tokenizer_sha256=QWEN_STALE_TOKENIZER_SHA256,
            tokenization_mode="no-terminal",
            file_count=10,
            directory_count=0,
            byte_count=1_207_482_746,
        ),
    }
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _parse_json_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QwenRevisionEncoderError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise QwenRevisionEncoderError(f"{label} contains non-finite number {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenRevisionEncoderError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise QwenRevisionEncoderError(f"{label} must contain one JSON object")
    if not all(isinstance(key, str) for key in value):
        raise QwenRevisionEncoderError(f"{label} keys must be strings")
    return value


def _closed_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QwenRevisionEncoderError(f"{label} must be an object")
    observed = set(value)
    if observed != _CONFIG_FIELDS:
        raise QwenRevisionEncoderError(
            f"{label} schema mismatch; missing={sorted(_CONFIG_FIELDS - observed)}, "
            f"unknown={sorted(observed - _CONFIG_FIELDS)}"
        )
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise QwenRevisionEncoderError(f"{name} must be a lowercase SHA-256")
    return value


def _require_revision(name: str, value: object) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise QwenRevisionEncoderError(f"{name} must be an immutable 40-character revision")
    return value


def _require_text(name: str, value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "non-empty " if not allow_empty else ""
        raise QwenRevisionEncoderError(f"{name} must be a {qualifier}string")
    if unicodedata.normalize("NFC", value) != value:
        raise QwenRevisionEncoderError(f"{name} must use NFC Unicode normalization")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise QwenRevisionEncoderError(f"{name} must be valid UTF-8") from exc
    return value


def _positive_integer(name: str, value: object, *, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise QwenRevisionEncoderError(f"{name} must be an integer in [1, {maximum}]")
    return value


@dataclass(frozen=True)
class QwenRevisionEncoderConfig:
    """Closed binding for one arm of the registered Qwen revision intervention."""

    arm: QwenArm
    tokenization_mode: TokenizationMode
    model_revision: str
    model_tree_sha256: str
    tokenizer_sha256: str
    batch_size: int = 8
    device: str = "cpu"
    deterministic_seed: int = 20260714
    query_prompt: str = QWEN_QUERY_PROMPT
    document_prompt: str = QWEN_DOCUMENT_PROMPT
    max_sequence_length: int = QWEN_MAX_SEQUENCE_LENGTH
    output_dimension: int = QWEN_OUTPUT_DIMENSION
    normalize: bool = True
    model_id: str = QWEN_MODEL_ID
    model_sha256: str = QWEN_MODEL_SHA256
    model_config_sha256: str = QWEN_MODEL_CONFIG_SHA256
    terminal_token_id: int = QWEN_TERMINAL_TOKEN_ID
    padding_side: str = "left"
    pooling: str = "last-active-token"
    dtype: str = "float32"
    encoder_version: str = QWEN_REVISION_ENCODER_VERSION
    schema_version: str = QWEN_REVISION_ENCODER_SCHEMA

    def __post_init__(self) -> None:
        if self.arm not in _ARM_BINDINGS:
            raise QwenRevisionEncoderError("arm must be 'current' or 'stale'")
        binding = _ARM_BINDINGS[self.arm]
        if self.tokenization_mode != binding.tokenization_mode:
            raise QwenRevisionEncoderError("tokenization_mode does not match the selected arm")
        if _require_revision("model_revision", self.model_revision) != binding.revision:
            raise QwenRevisionEncoderError("model_revision does not match the selected arm")
        if _require_sha256("model_tree_sha256", self.model_tree_sha256) != binding.tree_sha256:
            raise QwenRevisionEncoderError("model_tree_sha256 does not match the selected arm")
        if _require_sha256("tokenizer_sha256", self.tokenizer_sha256) != binding.tokenizer_sha256:
            raise QwenRevisionEncoderError("tokenizer_sha256 does not match the selected arm")
        if self.model_id != QWEN_MODEL_ID:
            raise QwenRevisionEncoderError(f"model_id must equal {QWEN_MODEL_ID!r}")
        if _require_sha256("model_sha256", self.model_sha256) != QWEN_MODEL_SHA256:
            raise QwenRevisionEncoderError("model_sha256 does not match the frozen weights")
        if (
            _require_sha256("model_config_sha256", self.model_config_sha256)
            != QWEN_MODEL_CONFIG_SHA256
        ):
            raise QwenRevisionEncoderError("model_config_sha256 does not match the frozen config")
        if self.query_prompt != QWEN_QUERY_PROMPT:
            raise QwenRevisionEncoderError("query_prompt does not match the frozen study prompt")
        if self.document_prompt != QWEN_DOCUMENT_PROMPT:
            raise QwenRevisionEncoderError("document_prompt must be the frozen empty prompt")
        if self.max_sequence_length != QWEN_MAX_SEQUENCE_LENGTH:
            raise QwenRevisionEncoderError("max_sequence_length must equal 512")
        if self.output_dimension != QWEN_OUTPUT_DIMENSION:
            raise QwenRevisionEncoderError("output_dimension must equal 256")
        if self.normalize is not True:
            raise QwenRevisionEncoderError("normalize must be true")
        _positive_integer("batch_size", self.batch_size, maximum=4096)
        _require_text("device", self.device)
        if self.device != self.device.strip() or any(
            character.isspace() for character in self.device
        ):
            raise QwenRevisionEncoderError("device must be a canonical device string")
        if type(self.deterministic_seed) is not int or not 0 <= self.deterministic_seed < 2**63:
            raise QwenRevisionEncoderError("deterministic_seed must be an unsigned 63-bit integer")
        if self.terminal_token_id != QWEN_TERMINAL_TOKEN_ID:
            raise QwenRevisionEncoderError("terminal_token_id must equal 151643")
        if self.padding_side != "left":
            raise QwenRevisionEncoderError("padding_side must equal 'left'")
        if self.pooling != "last-active-token":
            raise QwenRevisionEncoderError("pooling must equal 'last-active-token'")
        if self.dtype != "float32":
            raise QwenRevisionEncoderError("dtype must equal 'float32'")
        if self.encoder_version != QWEN_REVISION_ENCODER_VERSION:
            raise QwenRevisionEncoderError(
                f"encoder_version must equal {QWEN_REVISION_ENCODER_VERSION!r}"
            )
        if self.schema_version != QWEN_REVISION_ENCODER_SCHEMA:
            raise QwenRevisionEncoderError(
                f"schema_version must equal {QWEN_REVISION_ENCODER_SCHEMA!r}"
            )

    @classmethod
    def for_arm(
        cls,
        arm: QwenArm,
        *,
        batch_size: int = 8,
        device: str = "cpu",
        deterministic_seed: int = 20260714,
    ) -> QwenRevisionEncoderConfig:
        if arm not in _ARM_BINDINGS:
            raise QwenRevisionEncoderError("arm must be 'current' or 'stale'")
        binding = _ARM_BINDINGS[arm]
        return cls(
            arm=arm,
            tokenization_mode=binding.tokenization_mode,
            model_revision=binding.revision,
            model_tree_sha256=binding.tree_sha256,
            tokenizer_sha256=binding.tokenizer_sha256,
            batch_size=batch_size,
            device=device,
            deterministic_seed=deterministic_seed,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "batch_size": self.batch_size,
            "deterministic_seed": self.deterministic_seed,
            "device": self.device,
            "document_prompt": self.document_prompt,
            "dtype": self.dtype,
            "encoder_version": self.encoder_version,
            "max_sequence_length": self.max_sequence_length,
            "model_config_sha256": self.model_config_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "model_tree_sha256": self.model_tree_sha256,
            "normalize": self.normalize,
            "output_dimension": self.output_dimension,
            "padding_side": self.padding_side,
            "pooling": self.pooling,
            "query_prompt": self.query_prompt,
            "schema_version": self.schema_version,
            "terminal_token_id": self.terminal_token_id,
            "tokenization_mode": self.tokenization_mode,
            "tokenizer_sha256": self.tokenizer_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def loads_qwen_revision_encoder_config(
    value: bytes | str,
) -> QwenRevisionEncoderConfig:
    encoded = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    payload = _closed_mapping(
        _parse_json_object(encoded, label="Qwen revision encoder config"),
        label="Qwen revision encoder config",
    )
    try:
        config = QwenRevisionEncoderConfig(**payload)  # type: ignore[arg-type]
    except TypeError as exc:
        raise QwenRevisionEncoderError(f"invalid Qwen revision encoder config: {exc}") from exc
    if encoded != config.canonical_bytes() + b"\n":
        raise QwenRevisionEncoderError(
            "Qwen revision encoder config must be canonical JSON plus LF"
        )
    return config


def _absolute_model_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise QwenRevisionEncoderError("model_path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise QwenRevisionEncoderError(f"cannot resolve model_path: {exc}") from exc
    if resolved != path:
        raise QwenRevisionEncoderError(
            "model_path cannot contain symbolic-link or alias components"
        )
    if not path.is_dir():
        raise QwenRevisionEncoderError("model_path must be a directory")
    return path


def verify_qwen_revision_tree(
    model_path: str | Path,
    config: QwenRevisionEncoderConfig,
) -> DirectoryDigest:
    path = _absolute_model_path(model_path)
    binding = _ARM_BINDINGS[config.arm]
    try:
        digest = digest_directory_tree(path)
    except ArtifactIntegrityError as exc:
        raise QwenRevisionEncoderError(f"cannot verify the local Qwen tree: {exc}") from exc
    if digest.sha256 != config.model_tree_sha256:
        raise QwenRevisionEncoderError("local Qwen tree digest does not match the frozen arm")
    if (
        digest.file_count != binding.file_count
        or digest.directory_count != binding.directory_count
        or digest.byte_count != binding.byte_count
    ):
        raise QwenRevisionEncoderError("local Qwen tree accounting does not match the frozen arm")
    return digest


@dataclass(frozen=True)
class _TokenizedBatch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    opaque: object = field(repr=False, compare=False)


class _QwenBackend(Protocol):
    def tokenize(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
        max_length: int,
    ) -> _TokenizedBatch: ...

    def forward(self, batch: _TokenizedBatch, *, seed: int) -> np.ndarray: ...


class _PairedQwenBackend(Protocol):
    def tokenize(
        self,
        arm: QwenArm,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
        max_length: int,
    ) -> _TokenizedBatch: ...

    def forward_selected(
        self,
        rows: Sequence[Sequence[int]],
        selections: Sequence[tuple[int, int]],
        *,
        output_dimension: int,
        seed: int,
    ) -> np.ndarray: ...


@contextmanager
def _offline_huggingface_environment() -> Any:
    names = ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _import_transformers_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise QwenRevisionEncoderError(
            "torch and transformers are required for the production Qwen encoder"
        ) from exc
    return torch, AutoTokenizer, AutoModel


def _tensor_to_numpy(value: object, *, label: str) -> np.ndarray:
    try:
        detached = value.detach()  # type: ignore[union-attr]
        cpu_value = detached.cpu()
        array = cpu_value.numpy()
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise QwenRevisionEncoderError(f"cannot convert {label} to NumPy") from exc
    return np.asarray(array)


def _content_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(attention_mask)
    if mask.ndim != 2 or np.any((mask != 0) & (mask != 1)):
        raise QwenRevisionEncoderError("attention mask is invalid for content positions")
    positions = np.cumsum(mask.astype(np.int64, copy=False), axis=1) - 1
    positions[mask == 0] = 0
    return np.ascontiguousarray(positions, dtype=np.int64)


class _TransformersBackend:
    def __init__(self, model_path: Path, config: QwenRevisionEncoderConfig) -> None:
        torch, auto_tokenizer, auto_model = _import_transformers_dependencies()
        try:
            with _offline_huggingface_environment():
                tokenizer = auto_tokenizer.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                    padding_side="left",
                    trust_remote_code=False,
                    use_fast=True,
                )
                model = auto_model.from_pretrained(
                    str(model_path),
                    attn_implementation="eager",
                    dtype=torch.float32,
                    local_files_only=True,
                    trust_remote_code=False,
                )
        except Exception as exc:
            raise QwenRevisionEncoderError("cannot load the pinned local Qwen revision") from exc

        if getattr(tokenizer, "is_fast", False) is not True:
            raise QwenRevisionEncoderError("the Qwen intervention requires the fast tokenizer")
        if getattr(tokenizer, "padding_side", None) != "left":
            raise QwenRevisionEncoderError("the Qwen tokenizer did not retain left padding")
        if len(tokenizer) != QWEN_VOCAB_SIZE:
            raise QwenRevisionEncoderError("the Qwen tokenizer vocabulary size drifted")
        if tokenizer.convert_tokens_to_ids(QWEN_TERMINAL_TOKEN) != QWEN_TERMINAL_TOKEN_ID:
            raise QwenRevisionEncoderError("the Qwen terminal-token mapping drifted")

        model_config = getattr(model, "config", None)
        if (
            getattr(model_config, "model_type", None) != "qwen3"
            or getattr(model_config, "hidden_size", None) != QWEN_HIDDEN_SIZE
            or getattr(model_config, "vocab_size", None) != QWEN_VOCAB_SIZE
        ):
            raise QwenRevisionEncoderError("the loaded Qwen model shape or type drifted")
        try:
            model.to(config.device)
            model.eval()
        except Exception as exc:
            raise QwenRevisionEncoderError(
                "cannot place the local Qwen model on the bound device"
            ) from exc

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = config.device

    def tokenize(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
        max_length: int,
    ) -> _TokenizedBatch:
        try:
            payload = self._tokenizer(
                list(texts),
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
                truncation=True,
            )
            input_ids = _tensor_to_numpy(payload["input_ids"], label="input_ids")
            attention_mask = _tensor_to_numpy(
                payload["attention_mask"],
                label="attention_mask",
            )
        except QwenRevisionEncoderError:
            raise
        except Exception as exc:
            raise QwenRevisionEncoderError("the local Qwen tokenizer rejected a batch") from exc
        return _TokenizedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            opaque=payload,
        )

    def forward(self, batch: _TokenizedBatch, *, seed: int) -> np.ndarray:
        torch = self._torch
        payload = batch.opaque
        if not isinstance(payload, Mapping):
            raise QwenRevisionEncoderError("the Transformers batch payload is invalid")
        try:
            moved = {name: tensor.to(self._device) for name, tensor in payload.items()}
            moved["position_ids"] = torch.as_tensor(
                _content_position_ids(batch.attention_mask),
                device=self._device,
            )
            previous_deterministic = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed)
            if str(self._device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self._model.eval()
            with torch.inference_mode():
                output = self._model(**moved, return_dict=True, use_cache=False)
            hidden = output.last_hidden_state.detach().to(dtype=torch.float32).cpu().numpy()
        except Exception as exc:
            raise QwenRevisionEncoderError("the local Qwen model failed during inference") from exc
        finally:
            if "previous_deterministic" in locals():
                torch.use_deterministic_algorithms(previous_deterministic)
        return np.asarray(hidden)


class _PairedTransformersBackend:
    """One pinned model with both pinned tokenizers and selected-state transfer."""

    def __init__(
        self,
        current_model_path: Path,
        stale_model_path: Path,
        current_config: QwenRevisionEncoderConfig,
    ) -> None:
        current = _TransformersBackend(current_model_path, current_config)
        torch, auto_tokenizer, _auto_model = _import_transformers_dependencies()
        try:
            with _offline_huggingface_environment():
                stale_tokenizer = auto_tokenizer.from_pretrained(
                    str(stale_model_path),
                    local_files_only=True,
                    padding_side="left",
                    trust_remote_code=False,
                    use_fast=True,
                )
        except Exception as exc:
            raise QwenRevisionEncoderError("cannot load the pinned stale Qwen tokenizer") from exc
        for label, tokenizer in (
            ("current", current._tokenizer),
            ("stale", stale_tokenizer),
        ):
            if getattr(tokenizer, "is_fast", False) is not True:
                raise QwenRevisionEncoderError(f"the {label} Qwen tokenizer must be fast")
            if getattr(tokenizer, "padding_side", None) != "left":
                raise QwenRevisionEncoderError(f"the {label} Qwen tokenizer must left-pad")
            if getattr(tokenizer, "truncation_side", None) != "right":
                raise QwenRevisionEncoderError(f"the {label} Qwen tokenizer must right-truncate")
            if len(tokenizer) != QWEN_VOCAB_SIZE:
                raise QwenRevisionEncoderError(
                    f"the {label} Qwen tokenizer vocabulary size drifted"
                )
            if tokenizer.convert_tokens_to_ids(QWEN_TERMINAL_TOKEN) != QWEN_TERMINAL_TOKEN_ID:
                raise QwenRevisionEncoderError(f"the {label} Qwen terminal-token mapping drifted")
        current_pad = getattr(current._tokenizer, "pad_token_id", None)
        stale_pad = getattr(stale_tokenizer, "pad_token_id", None)
        if current_pad != stale_pad or current_pad != QWEN_TERMINAL_TOKEN_ID:
            raise QwenRevisionEncoderError("the paired Qwen pad-token binding drifted")

        model_config = getattr(current._model, "config", None)
        architectures = getattr(model_config, "architectures", None)
        if architectures is not None and list(architectures) != ["Qwen3ForCausalLM"]:
            raise QwenRevisionEncoderError("the paired Qwen model is not the frozen causal decoder")

        self._torch = torch
        self._current_tokenizer = current._tokenizer
        self._stale_tokenizer = stale_tokenizer
        self._model = current._model
        self._device = current._device
        self._pad_token_id = int(current_pad)

    def tokenize(
        self,
        arm: QwenArm,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
        max_length: int,
    ) -> _TokenizedBatch:
        tokenizer = self._current_tokenizer if arm == "current" else self._stale_tokenizer
        try:
            payload = tokenizer(
                list(texts),
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
                truncation=True,
            )
            input_ids = _tensor_to_numpy(payload["input_ids"], label=f"{arm} input_ids")
            attention_mask = _tensor_to_numpy(
                payload["attention_mask"],
                label=f"{arm} attention_mask",
            )
        except QwenRevisionEncoderError:
            raise
        except Exception as exc:
            raise QwenRevisionEncoderError(
                f"the pinned {arm} Qwen tokenizer rejected a batch"
            ) from exc
        return _TokenizedBatch(input_ids, attention_mask, opaque=payload)

    def forward_selected(
        self,
        rows: Sequence[Sequence[int]],
        selections: Sequence[tuple[int, int]],
        *,
        output_dimension: int,
        seed: int,
    ) -> np.ndarray:
        if not rows or not selections:
            raise QwenRevisionEncoderError("paired forward needs rows and selections")
        width = max(len(row) for row in rows)
        if width > QWEN_MAX_SEQUENCE_LENGTH or any(not row for row in rows):
            raise QwenRevisionEncoderError("paired forward row length violates the frozen bound")
        input_ids = np.full(
            (len(rows), width),
            self._pad_token_id,
            dtype=np.int64,
        )
        attention_mask = np.zeros((len(rows), width), dtype=np.int64)
        for row_index, row in enumerate(rows):
            values = np.asarray(tuple(row), dtype=np.int64)
            if np.any(values < 0) or np.any(values >= QWEN_VOCAB_SIZE):
                raise QwenRevisionEncoderError("paired forward token exceeds the frozen vocabulary")
            input_ids[row_index, -len(values) :] = values
            attention_mask[row_index, -len(values) :] = 1
        for row_index, token_position in selections:
            if (
                not 0 <= row_index < len(rows)
                or not 0 <= token_position < width
                or attention_mask[row_index, token_position] != 1
            ):
                raise QwenRevisionEncoderError("paired pooling selection is not an active token")

        torch = self._torch
        try:
            moved = {
                "input_ids": torch.as_tensor(input_ids, device=self._device),
                "attention_mask": torch.as_tensor(attention_mask, device=self._device),
                "position_ids": torch.as_tensor(
                    _content_position_ids(attention_mask), device=self._device
                ),
            }
            previous_deterministic = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed)
            if str(self._device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self._model.eval()
            with torch.inference_mode():
                output = self._model(**moved, return_dict=True, use_cache=False)
                hidden = output.last_hidden_state
                selected = torch.stack(
                    [hidden[row, position, :output_dimension] for row, position in selections]
                )
            pooled = selected.detach().to(dtype=torch.float32).cpu().numpy()
        except Exception as exc:
            raise QwenRevisionEncoderError("the paired Qwen model failed during inference") from exc
        finally:
            if "previous_deterministic" in locals():
                torch.use_deterministic_algorithms(previous_deterministic)
        return np.asarray(pooled)


def _validate_tokenized_batch(
    batch: _TokenizedBatch,
    *,
    expected_rows: int,
    max_length: int,
    label: str,
) -> tuple[tuple[int, ...], ...]:
    input_ids = np.asarray(batch.input_ids)
    attention_mask = np.asarray(batch.attention_mask)
    if input_ids.ndim != 2 or attention_mask.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise QwenRevisionEncoderError(f"{label} IDs and attention mask must be aligned matrices")
    if input_ids.shape[0] != expected_rows or not 0 < input_ids.shape[1] <= max_length:
        raise QwenRevisionEncoderError(f"{label} matrix shape violates the frozen batch contract")
    if not np.issubdtype(input_ids.dtype, np.integer):
        raise QwenRevisionEncoderError(f"{label} input IDs must be integers")
    if np.any(input_ids < 0) or np.any(input_ids >= QWEN_VOCAB_SIZE):
        raise QwenRevisionEncoderError(f"{label} input IDs exceed the frozen vocabulary")
    if not (
        np.issubdtype(attention_mask.dtype, np.integer)
        or np.issubdtype(attention_mask.dtype, np.bool_)
    ):
        raise QwenRevisionEncoderError(f"{label} attention mask must be boolean or integer")
    if np.any((attention_mask != 0) & (attention_mask != 1)):
        raise QwenRevisionEncoderError(f"{label} attention mask must contain only zero and one")
    mask = attention_mask.astype(np.int8, copy=False)
    if np.any(mask.sum(axis=1) == 0):
        raise QwenRevisionEncoderError(f"{label} cannot contain an empty token sequence")
    if mask.shape[1] > 1 and np.any(np.diff(mask, axis=1) < 0):
        raise QwenRevisionEncoderError(f"{label} must use left padding")

    rows: list[tuple[int, ...]] = []
    for row in range(expected_rows):
        active = input_ids[row][mask[row].astype(bool)]
        rows.append(tuple(int(token) for token in active.tolist()))
    return tuple(rows)


def _tokenize_checked(
    backend: _QwenBackend,
    texts: Sequence[str],
    config: QwenRevisionEncoderConfig,
) -> _TokenizedBatch:
    primary = backend.tokenize(
        texts,
        add_special_tokens=True,
        max_length=config.max_sequence_length,
    )
    baseline_max_length = (
        config.max_sequence_length - 1
        if config.tokenization_mode == "automatic-terminal"
        else config.max_sequence_length
    )
    baseline = backend.tokenize(
        texts,
        add_special_tokens=False,
        max_length=baseline_max_length,
    )
    primary_rows = _validate_tokenized_batch(
        primary,
        expected_rows=len(texts),
        max_length=config.max_sequence_length,
        label="primary tokenization",
    )
    baseline_rows = _validate_tokenized_batch(
        baseline,
        expected_rows=len(texts),
        max_length=baseline_max_length,
        label="baseline tokenization",
    )

    for row, (observed, raw) in enumerate(zip(primary_rows, baseline_rows, strict=True)):
        if QWEN_TERMINAL_TOKEN_ID in raw:
            raise QwenRevisionEncoderError(
                f"input row {row} materializes the reserved terminal token before post-processing"
            )
        if config.tokenization_mode == "automatic-terminal":
            if observed != (*raw, QWEN_TERMINAL_TOKEN_ID):
                raise QwenRevisionEncoderError(
                    f"input row {row} did not receive exactly one automatic terminal token"
                )
        elif observed != raw:
            raise QwenRevisionEncoderError(
                f"input row {row} changed under the stale no-terminal tokenizer"
            )
        if config.tokenization_mode == "no-terminal" and observed[-1] == QWEN_TERMINAL_TOKEN_ID:
            raise QwenRevisionEncoderError(
                f"input row {row} emitted a terminal token in the stale arm"
            )
    return primary


def _pool_truncate_normalize(
    last_hidden_state: np.ndarray,
    attention_mask: np.ndarray,
    *,
    output_dimension: int = QWEN_OUTPUT_DIMENSION,
) -> np.ndarray:
    hidden = np.asarray(last_hidden_state)
    mask = np.asarray(attention_mask)
    if hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape:
        raise QwenRevisionEncoderError("hidden states and attention mask have incompatible shapes")
    if hidden.shape[0] == 0 or hidden.shape[2] < output_dimension:
        raise QwenRevisionEncoderError("hidden states do not satisfy the frozen output shape")
    if np.any((mask != 0) & (mask != 1)) or np.any(mask.sum(axis=1) == 0):
        raise QwenRevisionEncoderError("attention mask is invalid for last-token pooling")
    if not np.issubdtype(hidden.dtype, np.number) or not np.all(np.isfinite(hidden)):
        raise QwenRevisionEncoderError("hidden states must be finite numeric values")

    active = mask.astype(bool, copy=False)
    positions = np.broadcast_to(np.arange(mask.shape[1]), mask.shape)
    last_positions = np.where(active, positions, -1).max(axis=1)
    pooled = hidden[
        np.arange(hidden.shape[0]),
        last_positions,
        :output_dimension,
    ].astype(np.float32, copy=True)
    return _normalize_pooled(pooled)


def _normalize_pooled(pooled: np.ndarray) -> np.ndarray:
    values = np.asarray(pooled)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != QWEN_OUTPUT_DIMENSION:
        raise QwenRevisionEncoderError("pooled vectors have the wrong frozen shape")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise QwenRevisionEncoderError("pooled vectors must be finite numeric values")
    values = values.astype(np.float32, copy=True)
    squared = np.multiply(values, values, dtype=np.float32)
    norms = np.sqrt(np.sum(squared, axis=1, dtype=np.float32)).astype(np.float32, copy=False)
    if not np.all(np.isfinite(norms)) or np.any(norms <= np.finfo(np.float32).tiny):
        raise QwenRevisionEncoderError("pooled vectors cannot be normalized")
    vectors = np.divide(values, norms[:, None], dtype=np.float32)
    observed_norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(vectors)) or not np.allclose(
        observed_norms,
        1.0,
        rtol=2e-6,
        atol=2e-6,
    ):
        raise QwenRevisionEncoderError("encoder output is not finite and unit normalized")
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    vectors.setflags(write=False)
    return vectors


def _active_token_rows(batch: _TokenizedBatch, *, label: str) -> tuple[tuple[int, ...], ...]:
    return _validate_tokenized_batch(
        batch,
        expected_rows=np.asarray(batch.input_ids).shape[0],
        max_length=QWEN_MAX_SEQUENCE_LENGTH,
        label=label,
    )


def _paired_tokenize_checked(
    backend: _PairedQwenBackend,
    texts: Sequence[str],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    current = backend.tokenize(
        "current", texts, add_special_tokens=True, max_length=QWEN_MAX_SEQUENCE_LENGTH
    )
    current_raw = backend.tokenize(
        "current", texts, add_special_tokens=False, max_length=QWEN_MAX_SEQUENCE_LENGTH - 1
    )
    stale = backend.tokenize(
        "stale", texts, add_special_tokens=True, max_length=QWEN_MAX_SEQUENCE_LENGTH
    )
    stale_raw = backend.tokenize(
        "stale", texts, add_special_tokens=False, max_length=QWEN_MAX_SEQUENCE_LENGTH
    )
    current_rows = _active_token_rows(current, label="paired current tokenization")
    current_raw_rows = _active_token_rows(current_raw, label="paired current baseline tokenization")
    stale_rows = _active_token_rows(stale, label="paired stale tokenization")
    stale_raw_rows = _active_token_rows(stale_raw, label="paired stale baseline tokenization")

    for row, (current_row, current_plain, stale_row, stale_plain) in enumerate(
        zip(current_rows, current_raw_rows, stale_rows, stale_raw_rows, strict=True)
    ):
        if QWEN_TERMINAL_TOKEN_ID in current_plain or QWEN_TERMINAL_TOKEN_ID in stale_plain:
            raise QwenRevisionEncoderError(
                f"paired input row {row} materializes the reserved terminal token"
            )
        if current_row != (*current_plain, QWEN_TERMINAL_TOKEN_ID):
            raise QwenRevisionEncoderError(
                f"paired input row {row} lacks exactly one current terminal token"
            )
        if stale_row != stale_plain:
            raise QwenRevisionEncoderError(
                f"paired input row {row} changed under the stale tokenizer"
            )
        if len(stale_plain) <= QWEN_MAX_SEQUENCE_LENGTH - 1:
            if current_plain != stale_plain:
                raise QwenRevisionEncoderError(
                    f"paired tokenizers disagree before the terminal token in row {row}"
                )
        elif current_plain != stale_plain[: QWEN_MAX_SEQUENCE_LENGTH - 1]:
            raise QwenRevisionEncoderError(
                f"paired tokenizers disagree at the right-truncation boundary in row {row}"
            )
    return current_rows, stale_rows


def _paired_execution_plan(
    current_rows: Sequence[tuple[int, ...]],
    stale_rows: Sequence[tuple[int, ...]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]:
    if len(current_rows) != len(stale_rows) or not current_rows:
        raise QwenRevisionEncoderError("paired token rows are not aligned")
    rows = list(current_rows)
    overflow_batch = any(len(row) == QWEN_MAX_SEQUENCE_LENGTH for row in stale_rows)
    stale_row_indices: list[int | None] = []
    for stale_row in stale_rows:
        if overflow_batch:
            stale_row_indices.append(len(rows))
            rows.append(stale_row)
        else:
            stale_row_indices.append(None)
    width = max(len(row) for row in rows)
    selections: list[tuple[int, int]] = [(row, width - 1) for row in range(len(current_rows))]
    for row, overflow in enumerate(stale_row_indices):
        selections.append((row, width - 2) if overflow is None else (overflow, width - 1))
    return tuple(rows), tuple(selections)


def _length_bucket_groups(
    raw_lengths: Sequence[int],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    groups: dict[int, list[int]] = {bound: [] for bound in QWEN_LENGTH_BUCKETS}
    for row, length in enumerate(raw_lengths):
        if type(length) is not int or not 0 < length <= QWEN_MAX_SEQUENCE_LENGTH:
            raise QwenRevisionEncoderError("raw token length violates the frozen bound")
        bound = next((candidate for candidate in QWEN_LENGTH_BUCKETS if length <= candidate), None)
        if bound is None:
            raise QwenRevisionEncoderError("raw token length has no deterministic bucket")
        groups[bound].append(row)
    return tuple((bound, tuple(groups[bound])) for bound in QWEN_LENGTH_BUCKETS if groups[bound])


def _paired_batch_seed(
    current: QwenRevisionEncoderConfig,
    stale: QwenRevisionEncoderConfig,
    *,
    role: TextRole,
    start: int,
    rendered: Sequence[str],
) -> int:
    binding = {
        "current_config_sha256": current.sha256,
        "rendered_sha256": hashlib.sha256(_canonical_bytes({"texts": list(rendered)})).hexdigest(),
        "role": role,
        "stale_config_sha256": stale.sha256,
        "start": start,
    }
    return int.from_bytes(hashlib.sha256(_canonical_bytes(binding)).digest()[:8], "big") >> 1


@dataclass(frozen=True)
class _TreeMetadataSnapshot:
    entries: tuple[tuple[str, int, int, int, int, int, int, int], ...]


def _tree_metadata_snapshot(path: Path) -> _TreeMetadataSnapshot:
    candidates = (path, *sorted(path.rglob("*"), key=lambda item: item.as_posix().encode("utf-8")))
    entries: list[tuple[str, int, int, int, int, int, int, int]] = []
    try:
        for candidate in candidates:
            metadata = candidate.lstat()
            entries.append(
                (
                    "." if candidate == path else candidate.relative_to(path).as_posix(),
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_nlink,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
            )
    except OSError as exc:
        raise QwenRevisionEncoderError(f"cannot snapshot the pinned Qwen tree: {exc}") from exc
    return _TreeMetadataSnapshot(tuple(entries))


def _validated_texts(texts: Sequence[str], *, role: TextRole) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence) or not texts:
        raise QwenRevisionEncoderError(f"{role} texts must be a non-empty sequence")
    result: list[str] = []
    for position, text in enumerate(texts):
        result.append(_require_text(f"{role} texts[{position}]", text))
    return tuple(result)


def _batch_seed(
    config: QwenRevisionEncoderConfig,
    *,
    role: TextRole,
    start: int,
    rendered: Sequence[str],
) -> int:
    binding = {
        "config_sha256": config.sha256,
        "rendered_sha256": hashlib.sha256(_canonical_bytes({"texts": list(rendered)})).hexdigest(),
        "role": role,
        "start": start,
    }
    return int.from_bytes(hashlib.sha256(_canonical_bytes(binding)).digest()[:8], "big") >> 1


class QwenRevisionEncoder:
    """Offline encoder that executes one exactly pinned Qwen tokenizer arm."""

    def __init__(
        self,
        model_path: str | Path,
        config: QwenRevisionEncoderConfig,
    ) -> None:
        if not isinstance(config, QwenRevisionEncoderConfig):
            raise QwenRevisionEncoderError("config must be QwenRevisionEncoderConfig")
        path = _absolute_model_path(model_path)
        verify_qwen_revision_tree(path, config)
        self._model_path = path
        self._config = config
        self._backend: _QwenBackend = _TransformersBackend(path, config)
        self._lock = threading.Lock()
        _tokenize_checked(self._backend, ("fractal revision probe",), config)
        verify_qwen_revision_tree(path, config)

    @property
    def config(self) -> QwenRevisionEncoderConfig:
        return self._config

    @property
    def model_path(self) -> Path:
        return self._model_path

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, role="query")

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, role="document")

    def _encode(self, texts: Sequence[str], *, role: TextRole) -> np.ndarray:
        values = _validated_texts(texts, role=role)
        prompt = self._config.query_prompt if role == "query" else self._config.document_prompt
        rendered = tuple(prompt + text for text in values)
        batches: list[np.ndarray] = []

        with self._lock:
            verify_qwen_revision_tree(self._model_path, self._config)
            try:
                for start in range(0, len(rendered), self._config.batch_size):
                    batch_texts = rendered[start : start + self._config.batch_size]
                    sizing = _tokenize_checked(self._backend, batch_texts, self._config)
                    token_rows = _active_token_rows(sizing, label="length-bucket tokenization")
                    raw_lengths = tuple(
                        len(row) - (self._config.arm == "current") for row in token_rows
                    )
                    batch_vectors = np.empty(
                        (len(batch_texts), self._config.output_dimension), dtype=np.float32
                    )
                    for bound, indices in _length_bucket_groups(raw_lengths):
                        group_texts = tuple(batch_texts[index] for index in indices)
                        tokenized = _tokenize_checked(self._backend, group_texts, self._config)
                        hidden = self._backend.forward(
                            tokenized,
                            seed=_batch_seed(
                                self._config,
                                role=role,
                                start=start + bound,
                                rendered=group_texts,
                            ),
                        )
                        batch_vectors[np.asarray(indices)] = _pool_truncate_normalize(
                            hidden,
                            tokenized.attention_mask,
                            output_dimension=self._config.output_dimension,
                        )
                    batches.append(batch_vectors)
            finally:
                verify_qwen_revision_tree(self._model_path, self._config)

        vectors = np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)
        if vectors.shape != (len(values), self._config.output_dimension):
            raise QwenRevisionEncoderError("encoder output has an impossible final shape")
        vectors.setflags(write=False)
        return vectors


class QwenPairedRevisionEncoder:
    """Emit both frozen arms from one causal forward whenever their prefixes coincide."""

    def __init__(
        self,
        current_model_path: str | Path,
        stale_model_path: str | Path,
        current_config: QwenRevisionEncoderConfig,
        stale_config: QwenRevisionEncoderConfig,
    ) -> None:
        if (
            not isinstance(current_config, QwenRevisionEncoderConfig)
            or current_config.arm != "current"
        ):
            raise QwenRevisionEncoderError("paired current_config must bind the current arm")
        if not isinstance(stale_config, QwenRevisionEncoderConfig) or stale_config.arm != "stale":
            raise QwenRevisionEncoderError("paired stale_config must bind the stale arm")
        common_fields = (
            "batch_size",
            "deterministic_seed",
            "device",
            "document_prompt",
            "dtype",
            "max_sequence_length",
            "model_config_sha256",
            "model_id",
            "model_sha256",
            "normalize",
            "output_dimension",
            "padding_side",
            "pooling",
            "query_prompt",
            "terminal_token_id",
        )
        changed = [
            name
            for name in common_fields
            if getattr(current_config, name) != getattr(stale_config, name)
        ]
        if changed:
            raise QwenRevisionEncoderError(f"paired arm configuration differs: {changed}")
        current_path = _absolute_model_path(current_model_path)
        stale_path = _absolute_model_path(stale_model_path)
        if current_path == stale_path:
            raise QwenRevisionEncoderError("paired arm trees must occupy distinct paths")
        verify_qwen_revision_tree(current_path, current_config)
        verify_qwen_revision_tree(stale_path, stale_config)
        current_snapshot = _tree_metadata_snapshot(current_path)
        stale_snapshot = _tree_metadata_snapshot(stale_path)
        backend: _PairedQwenBackend = _PairedTransformersBackend(
            current_path,
            stale_path,
            current_config,
        )
        if (
            _tree_metadata_snapshot(current_path) != current_snapshot
            or _tree_metadata_snapshot(stale_path) != stale_snapshot
        ):
            raise QwenRevisionEncoderError("a pinned Qwen tree changed during paired load")

        probe = ("fractal causal-position probe",)
        current_rows, stale_rows = _paired_tokenize_checked(backend, probe)
        if len(stale_rows[0]) >= QWEN_MAX_SEQUENCE_LENGTH:
            raise QwenRevisionEncoderError("paired causal probe unexpectedly reached truncation")
        seed = 20260714
        current_raw = backend.forward_selected(
            (current_rows[0],),
            ((0, len(current_rows[0]) - 2),),
            output_dimension=QWEN_OUTPUT_DIMENSION,
            seed=seed,
        )
        stale_raw = backend.forward_selected(
            (stale_rows[0],),
            ((0, len(stale_rows[0]) - 1),),
            output_dimension=QWEN_OUTPUT_DIMENSION,
            seed=seed,
        )
        if current_raw.shape != stale_raw.shape or not np.allclose(
            current_raw,
            stale_raw,
            rtol=2e-5,
            atol=2e-6,
        ):
            raise QwenRevisionEncoderError(
                "the loaded model failed the causal prefix-invariance probe"
            )

        self._current_model_path = current_path
        self._stale_model_path = stale_path
        self._current_config = current_config
        self._stale_config = stale_config
        self._current_snapshot = current_snapshot
        self._stale_snapshot = stale_snapshot
        self._backend = backend
        self._lock = threading.Lock()

    @property
    def current_config(self) -> QwenRevisionEncoderConfig:
        return self._current_config

    @property
    def stale_config(self) -> QwenRevisionEncoderConfig:
        return self._stale_config

    def encode_queries(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        return self._encode_pair(texts, role="query")

    def encode_documents(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        return self._encode_pair(texts, role="document")

    def _assert_tree_metadata(self) -> None:
        if (
            _tree_metadata_snapshot(self._current_model_path) != self._current_snapshot
            or _tree_metadata_snapshot(self._stale_model_path) != self._stale_snapshot
        ):
            raise QwenRevisionEncoderError(
                "a pinned Qwen tree changed after its cryptographic verification"
            )

    def _encode_pair(
        self,
        texts: Sequence[str],
        *,
        role: TextRole,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = _validated_texts(texts, role=role)
        prompt = (
            self._current_config.query_prompt
            if role == "query"
            else self._current_config.document_prompt
        )
        rendered = tuple(prompt + text for text in values)
        current_batches: list[np.ndarray] = []
        stale_batches: list[np.ndarray] = []

        with self._lock:
            self._assert_tree_metadata()
            try:
                for start in range(0, len(rendered), self._current_config.batch_size):
                    batch_texts = rendered[start : start + self._current_config.batch_size]
                    current_rows, stale_rows = _paired_tokenize_checked(self._backend, batch_texts)
                    batch_current = np.empty(
                        (len(batch_texts), self._current_config.output_dimension),
                        dtype=np.float32,
                    )
                    batch_stale = np.empty_like(batch_current)
                    for bound, indices in _length_bucket_groups(
                        tuple(len(row) for row in stale_rows)
                    ):
                        group_current = tuple(current_rows[index] for index in indices)
                        group_stale = tuple(stale_rows[index] for index in indices)
                        group_texts = tuple(batch_texts[index] for index in indices)
                        seed = _paired_batch_seed(
                            self._current_config,
                            self._stale_config,
                            role=role,
                            start=start + bound,
                            rendered=group_texts,
                        )
                        if bound < QWEN_MAX_SEQUENCE_LENGTH:
                            execution_rows, selections = _paired_execution_plan(
                                group_current, group_stale
                            )
                            pooled = self._backend.forward_selected(
                                execution_rows,
                                selections,
                                output_dimension=self._current_config.output_dimension,
                                seed=seed,
                            )
                            if np.asarray(pooled).shape != (
                                2 * len(indices),
                                self._current_config.output_dimension,
                            ):
                                raise QwenRevisionEncoderError(
                                    "paired model returned the wrong selected-state shape"
                                )
                            vectors = _normalize_pooled(np.asarray(pooled))
                            current_values = vectors[: len(indices)]
                            stale_values = vectors[len(indices) :]
                        else:
                            current_values = _normalize_pooled(
                                self._backend.forward_selected(
                                    group_current,
                                    tuple(
                                        (row, len(value) - 1)
                                        for row, value in enumerate(group_current)
                                    ),
                                    output_dimension=self._current_config.output_dimension,
                                    seed=seed,
                                )
                            )
                            stale_values = _normalize_pooled(
                                self._backend.forward_selected(
                                    group_stale,
                                    tuple(
                                        (row, len(value) - 1)
                                        for row, value in enumerate(group_stale)
                                    ),
                                    output_dimension=self._current_config.output_dimension,
                                    seed=seed,
                                )
                            )
                        batch_current[np.asarray(indices)] = current_values
                        batch_stale[np.asarray(indices)] = stale_values
                    current_batches.append(batch_current)
                    stale_batches.append(batch_stale)
            finally:
                self._assert_tree_metadata()

        current = np.ascontiguousarray(np.concatenate(current_batches), dtype=np.float32)
        stale = np.ascontiguousarray(np.concatenate(stale_batches), dtype=np.float32)
        expected = (len(values), self._current_config.output_dimension)
        if current.shape != expected or stale.shape != expected:
            raise QwenRevisionEncoderError("paired encoder output has an impossible final shape")
        current.setflags(write=False)
        stale.setflags(write=False)
        return current, stale


class QwenRevisionEmbeddingAdapter:
    """Expose one frozen Qwen arm through the streaming-store encoder contract."""

    def __init__(self, config: QwenRevisionEncoderConfig) -> None:
        if not isinstance(config, QwenRevisionEncoderConfig):
            raise QwenRevisionEncoderError("config must be QwenRevisionEncoderConfig")
        self._config = config
        self._encoder: QwenRevisionEncoder | None = None
        self._model_path: Path | None = None
        self.implementation_id = f"{QWEN_REVISION_ENCODER_VERSION}-{config.arm}-{config.sha256}"

    @property
    def config(self) -> QwenRevisionEncoderConfig:
        return self._config

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
        path = _absolute_model_path(model_path)
        if prompt == self._config.query_prompt:
            role: TextRole = "query"
        elif prompt == self._config.document_prompt:
            role = "document"
        else:
            raise QwenRevisionEncoderError("embedding-store prompt differs from the frozen arm")
        if max_sequence_length != self._config.max_sequence_length:
            raise QwenRevisionEncoderError(
                "embedding-store sequence length differs from the frozen arm"
            )
        if output_dimension != self._config.output_dimension:
            raise QwenRevisionEncoderError(
                "embedding-store output dimension differs from the frozen arm"
            )
        if normalize is not self._config.normalize:
            raise QwenRevisionEncoderError(
                "embedding-store normalization differs from the frozen arm"
            )
        if device != self._config.device:
            raise QwenRevisionEncoderError("embedding-store device differs from the frozen arm")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise QwenRevisionEncoderError(
                "embedding-store batch seed must be an unsigned 63-bit integer"
            )
        if self._model_path is not None and path != self._model_path:
            raise QwenRevisionEncoderError("one adapter cannot switch local model trees")
        if self._encoder is None:
            self._encoder = QwenRevisionEncoder(path, self._config)
            self._model_path = path
        if role == "query":
            return self._encoder.encode_queries(texts)
        return self._encoder.encode_documents(texts)


class QwenPairedRevisionEmbeddingAdapter:
    """Expose the paired causal-forward encoder to the streaming store."""

    def __init__(
        self,
        current_config: QwenRevisionEncoderConfig,
        stale_config: QwenRevisionEncoderConfig,
    ) -> None:
        if (
            not isinstance(current_config, QwenRevisionEncoderConfig)
            or current_config.arm != "current"
        ):
            raise QwenRevisionEncoderError("paired adapter current_config must bind current")
        if not isinstance(stale_config, QwenRevisionEncoderConfig) or stale_config.arm != "stale":
            raise QwenRevisionEncoderError("paired adapter stale_config must bind stale")
        self._current_config = current_config
        self._stale_config = stale_config
        self._encoder: QwenPairedRevisionEncoder | None = None
        self._paths: tuple[Path, Path] | None = None
        pair_binding = hashlib.sha256(
            _canonical_bytes(
                {
                    "current_config_sha256": current_config.sha256,
                    "stale_config_sha256": stale_config.sha256,
                    "version": QWEN_PAIRED_REVISION_ENCODER_VERSION,
                }
            )
        ).hexdigest()
        self.current_implementation_id = (
            f"{QWEN_PAIRED_REVISION_ENCODER_VERSION}-current-{pair_binding}"
        )
        self.old_implementation_id = f"{QWEN_PAIRED_REVISION_ENCODER_VERSION}-stale-{pair_binding}"

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
        current_path = _absolute_model_path(current_model_path)
        stale_path = _absolute_model_path(old_model_path)
        if prompt == self._current_config.query_prompt:
            role: TextRole = "query"
        elif prompt == self._current_config.document_prompt:
            role = "document"
        else:
            raise QwenRevisionEncoderError("paired embedding-store prompt differs")
        checks = {
            "sequence length": max_sequence_length == self._current_config.max_sequence_length,
            "output dimension": output_dimension == self._current_config.output_dimension,
            "normalization": normalize is self._current_config.normalize,
            "device": device == self._current_config.device,
            "seed": type(seed) is int and 0 <= seed < 2**63,
        }
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            raise QwenRevisionEncoderError(f"paired embedding-store binding differs: {failed}")
        paths = (current_path, stale_path)
        if self._paths is not None and paths != self._paths:
            raise QwenRevisionEncoderError("one paired adapter cannot switch local model trees")
        if self._encoder is None:
            self._encoder = QwenPairedRevisionEncoder(
                current_path,
                stale_path,
                self._current_config,
                self._stale_config,
            )
            self._paths = paths
        if role == "query":
            return self._encoder.encode_queries(texts)
        return self._encoder.encode_documents(texts)
