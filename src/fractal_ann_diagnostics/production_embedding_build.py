"""Closed, resumable construction of the five production embedding stores.

The command in this module admits one verified label-free staging projection,
derives every corpus source allowlist from that projection's inventory, and
binds both frozen Qwen revision arms.  It has no label, qrels, evidence, or
generic source-path argument.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import re
import resource
import site
import stat
import subprocess
import sys
import sysconfig
import time
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from . import embedding_store as embedding_store_module
from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .embedding_store import (
    EmbeddingStoreConfig,
    EmbeddingStoreError,
    EmbeddingStoreReceipt,
    LocalModelSpec,
    StagedEmbeddingSources,
    build_embedding_store,
    verify_embedding_store,
)
from .joint_power_design import FIXED_CORPORA
from .qwen_revision_encoder import (
    QWEN_CURRENT_REVISION,
    QWEN_CURRENT_TREE_SHA256,
    QWEN_DOCUMENT_PROMPT,
    QWEN_MAX_SEQUENCE_LENGTH,
    QWEN_OUTPUT_DIMENSION,
    QWEN_QUERY_PROMPT,
    QWEN_STALE_REVISION,
    QWEN_STALE_TREE_SHA256,
    QwenPairedRevisionEmbeddingAdapter,
    QwenPairedRevisionEncoder,
    QwenRevisionEncoderConfig,
    QwenRevisionEncoderError,
    verify_qwen_revision_tree,
)
from .study_data import StudyDataError, verify_online_staging_projection

PRODUCTION_EMBEDDING_CONFIG_SCHEMA = "fractal-production-embedding-build-config-v2"
PRODUCTION_EMBEDDING_EVIDENCE_SCHEMA = "fractal-production-embedding-evidence-v1"
PRODUCTION_EMBEDDING_SUITE_SCHEMA = "fractal-production-embedding-suite-v1"
PRODUCTION_EMBEDDING_SUITE_FILENAME = "production-embedding-suite-receipt.json"
PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY = "build-evidence"
PRODUCTION_EMBEDDING_BUILDER_SCHEMA = "fractal-production-embedding-mps-builder-v2"
PRODUCTION_EMBEDDING_PROBE_SCHEMA = "fractal-production-embedding-mps-probe-v1"
PRODUCTION_EMBEDDING_BUILDER_KIND = "darwin-arm64-mps-venv"
PRODUCTION_EMBEDDING_BUILDER_PLATFORM = "darwin/arm64"
PRODUCTION_EMBEDDING_DEVICE = "mps"
PRODUCTION_EMBEDDING_PROJECT_VERSION = "0.3.0"
PRODUCTION_EMBEDDING_TORCH_VERSION = "2.13.0"
PRODUCTION_EMBEDDING_TRANSFORMERS_VERSION = "5.13.1"
PRODUCTION_EMBEDDING_NUMPY_VERSION = "2.5.1"
PRODUCTION_EMBEDDING_TOKENIZERS_VERSION = "0.22.2"
PRODUCTION_EMBEDDING_PYTHON_VERSION = "3.12.13"
_SYSTEM_GIT = Path("/usr/bin/git")
_EXPECTED_IMPORTED_MODULES = (
    "fractal_ann_diagnostics",
    "numpy",
    "tokenizers",
    "torch",
    "transformers",
)
_FIXED_BUILDER_ENVIRONMENT = tuple(
    sorted(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "HOME": "/private/var/empty",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONSAFEPATH": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "TMPDIR": "/private/tmp",
            "TRANSFORMERS_OFFLINE": "1",
            "TZ": "UTC",
            "VECLIB_MAXIMUM_THREADS": "1",
        }.items()
    )
)
_PROBE_TEXTS = (
    "fractal causal-position probe",
    "authorization policy evidence remains label blind",
    "local geometry changes across a filtered retrieval boundary",
    "a fixed synthetic sentence checks repeated MPS vector bytes",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MAX_CONTROL_BYTES = 64 * 1024 * 1024
_QUERY_STAGES = frozenset({"fit", "calibration", "sealed"})
_CONFIG_FIELDS = frozenset(
    {
        "builder_receipt",
        "builder_receipt_sha256",
        "corpora",
        "current_encoder_config",
        "current_model_root",
        "online_inventory_sha256",
        "online_staging_root",
        "output_dtype",
        "output_root",
        "projected_artifact_set_sha256",
        "schema_version",
        "stale_encoder_config",
        "stale_model_root",
    }
)
_CORPUS_SOURCE_FIELDS = frozenset({"corpus_id", "document_paths", "query_paths"})
_EVIDENCE_FIELDS = frozenset(
    {
        "completed_at_utc",
        "corpus_id",
        "document_count",
        "elapsed_monotonic_ns",
        "embedding_receipt_sha256",
        "embedding_tree_sha256",
        "online_inventory_sha256",
        "process_peak_rss_bytes",
        "production_config_sha256",
        "query_count",
        "schema_version",
        "source_inventory_sha256",
        "started_at_utc",
        "status",
    }
)
_SUITE_CORPUS_FIELDS = frozenset(
    {
        "corpus_id",
        "evidence_file_sha256",
        "evidence_sha256",
        "embedding_receipt_sha256",
        "embedding_tree_sha256",
    }
)
_SUITE_FIELDS = frozenset(
    {
        "corpora",
        "online_inventory_sha256",
        "production_config_sha256",
        "projected_artifact_set_sha256",
        "schema_version",
    }
)
_INVENTORY_ARTIFACT_FIELDS = frozenset(
    {
        "byte_count",
        "dataset",
        "path",
        "record_count",
        "role",
        "sha256",
        "stage",
        "visibility",
    }
)
_BUILDER_RECEIPT_FIELDS = frozenset(
    {
        "batch_size",
        "builder_kind",
        "builder_source_file_count",
        "builder_source_sha256",
        "chip",
        "current_encoder_config_sha256",
        "current_model_root",
        "current_model_tree_sha256",
        "deterministic_seed",
        "device",
        "git_executable",
        "git_executable_sha256",
        "imported_module_origins",
        "installed_distributions",
        "installed_distributions_sha256",
        "logical_cores",
        "machine",
        "macos_build",
        "macos_version",
        "memory_bytes",
        "model",
        "mps_available",
        "mps_built",
        "numpy_version",
        "platform",
        "probe",
        "process_environment",
        "project_version",
        "python_base_prefix",
        "python_dont_write_bytecode",
        "python_executable",
        "python_executable_sha256",
        "python_prefix",
        "python_prefix_configuration_sha256",
        "python_import_roots",
        "python_safe_path",
        "python_sys_path",
        "python_user_site_enabled",
        "python_version",
        "repository_root",
        "schema_version",
        "source_commit",
        "stale_encoder_config_sha256",
        "stale_model_root",
        "stale_model_tree_sha256",
        "site_packages_byte_count",
        "site_packages_directory_count",
        "site_packages_file_count",
        "site_packages_root",
        "site_packages_tree_sha256",
        "tokenizers_version",
        "torch_version",
        "transformers_version",
        "uv_lock_path",
        "uv_lock_sha256",
    }
)
_PROBE_RECEIPT_FIELDS = frozenset(
    {
        "current_encoder_config_sha256",
        "current_vectors_sha256",
        "first_elapsed_monotonic_ns",
        "output_dimension",
        "repeat_exact",
        "row_count",
        "schema_version",
        "second_elapsed_monotonic_ns",
        "stale_encoder_config_sha256",
        "stale_vectors_sha256",
        "texts_sha256",
    }
)
_DISTRIBUTION_FIELDS = frozenset({"name", "version"})
_MODULE_ORIGIN_FIELDS = frozenset({"name", "path"})
_IMPORT_ROOT_FIELDS = frozenset(
    {"byte_count", "directory_count", "file_count", "kind", "path", "sha256"}
)
_BUILDER_OBSERVATION_FIELDS = (
    "builder_source_file_count",
    "builder_source_sha256",
    "chip",
    "git_executable",
    "git_executable_sha256",
    "imported_module_origins",
    "installed_distributions",
    "installed_distributions_sha256",
    "logical_cores",
    "machine",
    "macos_build",
    "macos_version",
    "memory_bytes",
    "model",
    "mps_available",
    "mps_built",
    "numpy_version",
    "platform",
    "process_environment",
    "project_version",
    "python_base_prefix",
    "python_dont_write_bytecode",
    "python_executable",
    "python_executable_sha256",
    "python_prefix",
    "python_prefix_configuration_sha256",
    "python_import_roots",
    "python_safe_path",
    "python_sys_path",
    "python_user_site_enabled",
    "python_version",
    "site_packages_byte_count",
    "site_packages_directory_count",
    "site_packages_file_count",
    "site_packages_root",
    "site_packages_tree_sha256",
    "tokenizers_version",
    "torch_version",
    "transformers_version",
    "uv_lock_sha256",
)


class ProductionEmbeddingBuildError(RuntimeError):
    """Raised when the five-corpus build cannot preserve its closed boundary."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ProductionEmbeddingBuildError(
            "production embedding records must be finite canonical JSON"
        ) from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProductionEmbeddingBuildError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionEmbeddingBuildError(f"{name} must be a non-negative integer")
    return value


def _require_positive_integer(name: str, value: object) -> int:
    result = _require_nonnegative_integer(name, value)
    if result == 0:
        raise ProductionEmbeddingBuildError(f"{name} must be positive")
    return result


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ProductionEmbeddingBuildError(f"{name} must be a non-empty string")
    if unicodedata.normalize("NFC", value) != value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ProductionEmbeddingBuildError(f"{name} must be canonical UTF-8 text")
    return value


def _require_boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ProductionEmbeddingBuildError(f"{name} must be boolean")
    return value


def _validated_builder_environment(
    value: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    environment = tuple(value)
    if environment != tuple(sorted(environment, key=lambda item: item[0].encode("utf-8"))):
        raise ProductionEmbeddingBuildError("builder process environment must be sorted")
    if len({name for name, _value in environment}) != len(environment):
        raise ProductionEmbeddingBuildError("builder process environment repeats a name")
    observed = dict(environment)
    expected = dict(_FIXED_BUILDER_ENVIRONMENT)
    if set(observed) != {*expected, "__CF_USER_TEXT_ENCODING"} or any(
        observed.get(name) != fixed for name, fixed in expected.items()
    ):
        raise ProductionEmbeddingBuildError(
            "builder process environment differs from the fixed minimal environment"
        )
    cf_encoding = observed["__CF_USER_TEXT_ENCODING"]
    if re.fullmatch(r"0x[0-9A-F]+:0x0:0x0", cf_encoding) is None:
        raise ProductionEmbeddingBuildError(
            "builder __CF_USER_TEXT_ENCODING is not the canonical macOS value"
        )
    return environment


def _current_builder_environment() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                *_FIXED_BUILDER_ENVIRONMENT,
                ("__CF_USER_TEXT_ENCODING", f"0x{os.geteuid():X}:0x0:0x0"),
            ),
            key=lambda item: item[0].encode("utf-8"),
        )
    )


def _require_git_commit(name: str, value: object) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise ProductionEmbeddingBuildError(
            f"{name} must be a full lowercase 40-character Git commit"
        )
    return value


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ProductionEmbeddingBuildError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise ProductionEmbeddingBuildError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProductionEmbeddingBuildError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ProductionEmbeddingBuildError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionEmbeddingBuildError(f"{label} must be UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ProductionEmbeddingBuildError(f"{label} must contain one object")
    return value


def _read_control(path: Path, *, label: str) -> bytes:
    try:
        return read_secure_regular_file(path, max_bytes=_MAX_CONTROL_BYTES, label=label)
    except ArtifactIntegrityError as exc:
        raise ProductionEmbeddingBuildError(f"cannot read {label}: {exc}") from exc


def _canonical_absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProductionEmbeddingBuildError(f"{label} must be an absolute POSIX path")
    if unicodedata.normalize("NFC", value) != value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ProductionEmbeddingBuildError(f"{label} is not a canonical UTF-8 path")
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or str(pure) != value
        or any(part in {"", ".", ".."} for part in pure.parts[1:])
    ):
        raise ProductionEmbeddingBuildError(f"{label} must be an absolute canonical path")
    path = Path(value)
    if path.name in {"", ".", ".."}:
        raise ProductionEmbeddingBuildError(f"{label} cannot be the filesystem root")
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise ProductionEmbeddingBuildError(f"cannot resolve {label}: {exc}") from exc
    if resolved != path:
        raise ProductionEmbeddingBuildError(f"{label} crosses an alias or symbolic link")
    return path


def _require_real_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionEmbeddingBuildError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ProductionEmbeddingBuildError(f"{label} must be a real directory")


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_path_separation(
    *,
    online_staging_root: Path,
    current_model_root: Path,
    stale_model_root: Path,
    output_root: Path,
) -> None:
    values = {
        "online_staging_root": online_staging_root,
        "current_model_root": current_model_root,
        "stale_model_root": stale_model_root,
        "output_root": output_root,
    }
    for left_name, left in values.items():
        for right_name, right in values.items():
            if left_name >= right_name:
                continue
            if _paths_overlap(left, right):
                raise ProductionEmbeddingBuildError(f"{left_name} and {right_name} cannot overlap")


def _relative_jsonl_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProductionEmbeddingBuildError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or path.suffix != ".jsonl"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ProductionEmbeddingBuildError(f"{label} must be a canonical JSONL path")
    if unicodedata.normalize("NFC", value) != value:
        raise ProductionEmbeddingBuildError(f"{label} must use NFC Unicode normalization")
    return value


def _sorted_paths(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProductionEmbeddingBuildError(f"{label} must be a non-empty array")
    paths = tuple(
        _relative_jsonl_path(item, label=f"{label}[{position}]")
        for position, item in enumerate(value)
    )
    if paths != tuple(sorted(paths, key=lambda item: item.encode("utf-8"))):
        raise ProductionEmbeddingBuildError(f"{label} must be bytewise sorted")
    if len(paths) != len(set(paths)):
        raise ProductionEmbeddingBuildError(f"{label} must not repeat a path")
    return paths


@dataclass(frozen=True, order=True)
class InstalledDistribution:
    """One exact installed Python distribution in the MPS builder environment."""

    name: str
    version: str

    def __post_init__(self) -> None:
        normalized = re.sub(r"[-_.]+", "-", _require_text("distribution name", self.name).lower())
        if normalized != self.name or re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized) is None:
            raise ProductionEmbeddingBuildError(
                "installed distribution names must use canonical lowercase PEP 503 spelling"
            )
        _require_text(f"{self.name} version", self.version)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, value: object) -> InstalledDistribution:
        row = _closed(value, _DISTRIBUTION_FIELDS, label="installed distribution")
        return cls(name=row["name"], version=row["version"])


@dataclass(frozen=True, order=True)
class ImportedModuleOrigin:
    """One exact import origin admitted by the isolated MPS builder."""

    name: str
    path: Path

    def __post_init__(self) -> None:
        name = _require_text("imported module name", self.name)
        if name not in _EXPECTED_IMPORTED_MODULES:
            raise ProductionEmbeddingBuildError(f"unregistered imported module {name!r}")
        object.__setattr__(
            self,
            "path",
            _canonical_absolute_path(str(self.path), label=f"{name} import origin"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "path": str(self.path)}

    @classmethod
    def from_dict(cls, value: object) -> ImportedModuleOrigin:
        row = _closed(value, _MODULE_ORIGIN_FIELDS, label="imported module origin")
        return cls(name=row["name"], path=Path(row["path"]))


@dataclass(frozen=True, order=True)
class PythonImportRoot:
    """One file, directory, or interpreter-derived absent sys.path slot."""

    path: Path
    kind: Literal["absent", "directory", "file"]
    sha256: str
    file_count: int
    directory_count: int
    byte_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _canonical_absolute_path(str(self.path), label="Python import root"),
        )
        if self.kind not in {"absent", "directory", "file"}:
            raise ProductionEmbeddingBuildError("Python import root kind is invalid")
        _require_sha256("Python import root sha256", self.sha256)
        for name in ("file_count", "directory_count", "byte_count"):
            _require_nonnegative_integer(f"Python import root {name}", getattr(self, name))
        expected_counts = {
            "absent": (0, 0),
            "file": (1, 0),
        }
        if (
            self.kind in expected_counts
            and (
                self.file_count,
                self.directory_count,
            )
            != expected_counts[self.kind]
        ):
            raise ProductionEmbeddingBuildError("Python import root accounting differs")
        if self.kind == "directory" and self.directory_count == 0:
            raise ProductionEmbeddingBuildError("directory import root has no directory accounting")
        if self.kind == "absent" and self.byte_count != 0:
            raise ProductionEmbeddingBuildError("absent Python import root has bytes")
        if self.kind == "absent" and self.sha256 != _sha256(
            _canonical_bytes({"kind": "absent", "path": str(self.path)})
        ):
            raise ProductionEmbeddingBuildError("absent Python import-root digest differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "directory_count": self.directory_count,
            "file_count": self.file_count,
            "kind": self.kind,
            "path": str(self.path),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> PythonImportRoot:
        row = _closed(value, _IMPORT_ROOT_FIELDS, label="Python import root")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProductionEmbeddingProbeReceipt:
    """Fixed, label-free repeated-vector probe executed by the MPS builder."""

    texts_sha256: str
    row_count: int
    current_encoder_config_sha256: str
    stale_encoder_config_sha256: str
    current_vectors_sha256: str
    stale_vectors_sha256: str
    output_dimension: int
    repeat_exact: bool
    first_elapsed_monotonic_ns: int
    second_elapsed_monotonic_ns: int
    schema_version: str = PRODUCTION_EMBEDDING_PROBE_SCHEMA

    def __post_init__(self) -> None:
        expected_texts = _sha256(_canonical_bytes({"texts": list(_PROBE_TEXTS)}))
        if _require_sha256("probe texts_sha256", self.texts_sha256) != expected_texts:
            raise ProductionEmbeddingBuildError("MPS probe texts differ from the fixed probe")
        if self.row_count != len(_PROBE_TEXTS):
            raise ProductionEmbeddingBuildError("MPS probe row_count differs from the fixed probe")
        for name in (
            "current_encoder_config_sha256",
            "stale_encoder_config_sha256",
            "current_vectors_sha256",
            "stale_vectors_sha256",
        ):
            _require_sha256(f"probe {name}", getattr(self, name))
        if self.current_encoder_config_sha256 == self.stale_encoder_config_sha256:
            raise ProductionEmbeddingBuildError("MPS probe encoder arms must remain distinct")
        if self.output_dimension != QWEN_OUTPUT_DIMENSION:
            raise ProductionEmbeddingBuildError("MPS probe output dimension drifted")
        if _require_boolean("probe repeat_exact", self.repeat_exact) is not True:
            raise ProductionEmbeddingBuildError("MPS probe repeated vectors must be byte-identical")
        _require_positive_integer(
            "probe first_elapsed_monotonic_ns", self.first_elapsed_monotonic_ns
        )
        _require_positive_integer(
            "probe second_elapsed_monotonic_ns", self.second_elapsed_monotonic_ns
        )
        if self.schema_version != PRODUCTION_EMBEDDING_PROBE_SCHEMA:
            raise ProductionEmbeddingBuildError("MPS probe schema_version drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "current_encoder_config_sha256": self.current_encoder_config_sha256,
            "current_vectors_sha256": self.current_vectors_sha256,
            "first_elapsed_monotonic_ns": self.first_elapsed_monotonic_ns,
            "output_dimension": self.output_dimension,
            "repeat_exact": self.repeat_exact,
            "row_count": self.row_count,
            "schema_version": self.schema_version,
            "second_elapsed_monotonic_ns": self.second_elapsed_monotonic_ns,
            "stale_encoder_config_sha256": self.stale_encoder_config_sha256,
            "stale_vectors_sha256": self.stale_vectors_sha256,
            "texts_sha256": self.texts_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProductionEmbeddingProbeReceipt:
        row = _closed(value, _PROBE_RECEIPT_FIELDS, label="MPS probe receipt")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProductionEmbeddingBuilderReceipt:
    """Closed source, environment, hardware, and probe identity for MPS construction."""

    repository_root: Path
    source_commit: str
    builder_source_sha256: str
    builder_source_file_count: int
    uv_lock_path: Path
    uv_lock_sha256: str
    git_executable: Path
    git_executable_sha256: str
    python_executable: Path
    python_executable_sha256: str
    python_prefix: Path
    python_prefix_configuration_sha256: str
    python_base_prefix: Path
    python_version: str
    python_safe_path: bool
    python_dont_write_bytecode: bool
    python_user_site_enabled: bool
    python_sys_path: tuple[Path, ...]
    python_import_roots: tuple[PythonImportRoot, ...]
    site_packages_root: Path
    site_packages_tree_sha256: str
    site_packages_file_count: int
    site_packages_directory_count: int
    site_packages_byte_count: int
    imported_module_origins: tuple[ImportedModuleOrigin, ...]
    process_environment: tuple[tuple[str, str], ...]
    installed_distributions: tuple[InstalledDistribution, ...]
    installed_distributions_sha256: str
    project_version: str
    torch_version: str
    transformers_version: str
    numpy_version: str
    tokenizers_version: str
    macos_version: str
    macos_build: str
    platform: Literal["darwin/arm64"]
    machine: Literal["arm64"]
    model: str
    chip: str
    logical_cores: int
    memory_bytes: int
    mps_built: bool
    mps_available: bool
    builder_kind: Literal["darwin-arm64-mps-venv"]
    device: Literal["mps"]
    batch_size: int
    deterministic_seed: int
    current_model_root: Path
    stale_model_root: Path
    current_model_tree_sha256: str
    stale_model_tree_sha256: str
    current_encoder_config_sha256: str
    stale_encoder_config_sha256: str
    probe: ProductionEmbeddingProbeReceipt
    schema_version: str = PRODUCTION_EMBEDDING_BUILDER_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "repository_root",
            "uv_lock_path",
            "git_executable",
            "python_executable",
            "python_prefix",
            "python_base_prefix",
            "site_packages_root",
            "current_model_root",
            "stale_model_root",
        ):
            object.__setattr__(
                self,
                name,
                _canonical_absolute_path(str(getattr(self, name)), label=name),
            )
        _require_git_commit("builder source_commit", self.source_commit)
        _require_sha256("builder_source_sha256", self.builder_source_sha256)
        _require_positive_integer("builder_source_file_count", self.builder_source_file_count)
        _require_sha256("builder uv_lock_sha256", self.uv_lock_sha256)
        _require_sha256("builder git_executable_sha256", self.git_executable_sha256)
        if self.git_executable != _SYSTEM_GIT:
            raise ProductionEmbeddingBuildError("builder Git must be the fixed macOS system Git")
        _require_sha256("builder python_executable_sha256", self.python_executable_sha256)
        _require_sha256(
            "builder python_prefix_configuration_sha256",
            self.python_prefix_configuration_sha256,
        )
        if self.python_version != PRODUCTION_EMBEDDING_PYTHON_VERSION:
            raise ProductionEmbeddingBuildError(
                f"builder Python must equal {PRODUCTION_EMBEDDING_PYTHON_VERSION}"
            )
        if _require_boolean("builder python_safe_path", self.python_safe_path) is not True:
            raise ProductionEmbeddingBuildError("builder Python must enable safe-path mode")
        if (
            _require_boolean("builder python_dont_write_bytecode", self.python_dont_write_bytecode)
            is not True
        ):
            raise ProductionEmbeddingBuildError("builder Python must disable bytecode writes")
        if (
            _require_boolean("builder python_user_site_enabled", self.python_user_site_enabled)
            is not False
        ):
            raise ProductionEmbeddingBuildError("builder Python must disable the user site")
        python_sys_path = tuple(
            _canonical_absolute_path(str(path), label="builder sys.path entry")
            for path in self.python_sys_path
        )
        if not python_sys_path or len(python_sys_path) != len(set(python_sys_path)):
            raise ProductionEmbeddingBuildError("builder sys.path must be non-empty and unique")
        object.__setattr__(self, "python_sys_path", python_sys_path)
        import_roots = tuple(self.python_import_roots)
        if not import_roots or import_roots != tuple(sorted(import_roots)):
            raise ProductionEmbeddingBuildError("Python import-root inventory must be sorted")
        if len({row.path for row in import_roots}) != len(import_roots):
            raise ProductionEmbeddingBuildError("Python import-root inventory repeats a path")
        external_paths = set(python_sys_path) - {
            self.repository_root / "src",
            self.site_packages_root,
        }
        if {row.path for row in import_roots} != external_paths:
            raise ProductionEmbeddingBuildError(
                "Python import-root inventory does not cover every base search root"
            )
        object.__setattr__(self, "python_import_roots", import_roots)
        source_root = self.repository_root / "src"
        if source_root not in python_sys_path or self.repository_root in python_sys_path:
            raise ProductionEmbeddingBuildError(
                "builder sys.path must admit only the source root, not the repository CWD"
            )
        if self.site_packages_root not in python_sys_path:
            raise ProductionEmbeddingBuildError("builder sys.path omits its site-packages root")
        allowed_path_roots = (self.python_base_prefix, self.python_prefix, source_root)
        if any(
            not any(path == root or path.is_relative_to(root) for root in allowed_path_roots)
            for path in python_sys_path
        ):
            raise ProductionEmbeddingBuildError("builder sys.path contains an external search root")
        if not self.site_packages_root.is_relative_to(self.python_prefix):
            raise ProductionEmbeddingBuildError("site-packages must belong to the builder venv")
        _require_sha256("site_packages_tree_sha256", self.site_packages_tree_sha256)
        _require_positive_integer("site_packages_file_count", self.site_packages_file_count)
        _require_positive_integer(
            "site_packages_directory_count", self.site_packages_directory_count
        )
        _require_positive_integer("site_packages_byte_count", self.site_packages_byte_count)
        origins = tuple(self.imported_module_origins)
        if origins != tuple(sorted(origins)) or tuple(row.name for row in origins) != tuple(
            sorted(_EXPECTED_IMPORTED_MODULES)
        ):
            raise ProductionEmbeddingBuildError(
                "imported module origins must cover the exact sorted module allowlist"
            )
        object.__setattr__(self, "imported_module_origins", origins)
        origins_by_name = {row.name: row.path for row in origins}
        if origins_by_name["fractal_ann_diagnostics"] != (
            source_root / "fractal_ann_diagnostics" / "__init__.py"
        ):
            raise ProductionEmbeddingBuildError("apparatus package import origin differs")
        if any(
            not origins_by_name[name].is_relative_to(self.site_packages_root)
            for name in _EXPECTED_IMPORTED_MODULES
            if name != "fractal_ann_diagnostics"
        ):
            raise ProductionEmbeddingBuildError(
                "model package import origin escapes pinned site-packages"
            )
        environment = tuple(self.process_environment)
        object.__setattr__(
            self,
            "process_environment",
            _validated_builder_environment(environment),
        )
        distributions = tuple(self.installed_distributions)
        if not distributions or distributions != tuple(sorted(distributions)):
            raise ProductionEmbeddingBuildError("installed distributions must be sorted")
        if len({row.name for row in distributions}) != len(distributions):
            raise ProductionEmbeddingBuildError("installed distributions repeat a package")
        object.__setattr__(self, "installed_distributions", distributions)
        expected_distribution_sha = _sha256(
            _canonical_bytes([row.to_dict() for row in distributions])
        )
        if (
            _require_sha256("installed_distributions_sha256", self.installed_distributions_sha256)
            != expected_distribution_sha
        ):
            raise ProductionEmbeddingBuildError("installed distribution inventory digest differs")
        exact_versions = {
            "fractal-ann-diagnostics": (
                self.project_version,
                PRODUCTION_EMBEDDING_PROJECT_VERSION,
            ),
            "numpy": (self.numpy_version, PRODUCTION_EMBEDDING_NUMPY_VERSION),
            "tokenizers": (self.tokenizers_version, PRODUCTION_EMBEDDING_TOKENIZERS_VERSION),
            "torch": (self.torch_version, PRODUCTION_EMBEDDING_TORCH_VERSION),
            "transformers": (
                self.transformers_version,
                PRODUCTION_EMBEDDING_TRANSFORMERS_VERSION,
            ),
        }
        observed_versions = {row.name: row.version for row in distributions}
        for name, (direct, expected) in exact_versions.items():
            if direct != expected or observed_versions.get(name) != expected:
                raise ProductionEmbeddingBuildError(
                    f"builder {name} must equal the pinned version {expected}"
                )
        for name in ("macos_version", "macos_build", "model", "chip"):
            _require_text(f"builder {name}", getattr(self, name))
        if self.platform != PRODUCTION_EMBEDDING_BUILDER_PLATFORM or self.machine != "arm64":
            raise ProductionEmbeddingBuildError("builder must run on Darwin arm64")
        _require_positive_integer("builder logical_cores", self.logical_cores)
        _require_positive_integer("builder memory_bytes", self.memory_bytes)
        if _require_boolean("builder mps_built", self.mps_built) is not True:
            raise ProductionEmbeddingBuildError("builder PyTorch must include MPS")
        if _require_boolean("builder mps_available", self.mps_available) is not True:
            raise ProductionEmbeddingBuildError("builder host must expose MPS")
        if self.builder_kind != PRODUCTION_EMBEDDING_BUILDER_KIND:
            raise ProductionEmbeddingBuildError("builder_kind differs from the MPS builder")
        if self.device != PRODUCTION_EMBEDDING_DEVICE:
            raise ProductionEmbeddingBuildError("builder device must equal 'mps'")
        _require_positive_integer("builder batch_size", self.batch_size)
        _require_nonnegative_integer("builder deterministic_seed", self.deterministic_seed)
        if self.current_model_root == self.stale_model_root:
            raise ProductionEmbeddingBuildError("builder model roots must remain distinct")
        if self.current_model_tree_sha256 != QWEN_CURRENT_TREE_SHA256:
            raise ProductionEmbeddingBuildError("builder current model tree differs")
        if self.stale_model_tree_sha256 != QWEN_STALE_TREE_SHA256:
            raise ProductionEmbeddingBuildError("builder stale model tree differs")
        current = QwenRevisionEncoderConfig.for_arm(
            "current",
            batch_size=self.batch_size,
            device=self.device,
            deterministic_seed=self.deterministic_seed,
        )
        stale = QwenRevisionEncoderConfig.for_arm(
            "stale",
            batch_size=self.batch_size,
            device=self.device,
            deterministic_seed=self.deterministic_seed,
        )
        if self.current_encoder_config_sha256 != current.sha256:
            raise ProductionEmbeddingBuildError("builder current encoder config differs")
        if self.stale_encoder_config_sha256 != stale.sha256:
            raise ProductionEmbeddingBuildError("builder stale encoder config differs")
        if not isinstance(self.probe, ProductionEmbeddingProbeReceipt):
            raise ProductionEmbeddingBuildError("builder probe must be a typed receipt")
        if (
            self.probe.current_encoder_config_sha256 != current.sha256
            or self.probe.stale_encoder_config_sha256 != stale.sha256
        ):
            raise ProductionEmbeddingBuildError("builder probe encoder configs differ")
        if self.schema_version != PRODUCTION_EMBEDDING_BUILDER_SCHEMA:
            raise ProductionEmbeddingBuildError("builder schema_version drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "builder_kind": self.builder_kind,
            "builder_source_file_count": self.builder_source_file_count,
            "builder_source_sha256": self.builder_source_sha256,
            "chip": self.chip,
            "current_encoder_config_sha256": self.current_encoder_config_sha256,
            "current_model_root": str(self.current_model_root),
            "current_model_tree_sha256": self.current_model_tree_sha256,
            "deterministic_seed": self.deterministic_seed,
            "device": self.device,
            "git_executable": str(self.git_executable),
            "git_executable_sha256": self.git_executable_sha256,
            "imported_module_origins": [row.to_dict() for row in self.imported_module_origins],
            "installed_distributions": [row.to_dict() for row in self.installed_distributions],
            "installed_distributions_sha256": self.installed_distributions_sha256,
            "logical_cores": self.logical_cores,
            "machine": self.machine,
            "macos_build": self.macos_build,
            "macos_version": self.macos_version,
            "memory_bytes": self.memory_bytes,
            "model": self.model,
            "mps_available": self.mps_available,
            "mps_built": self.mps_built,
            "numpy_version": self.numpy_version,
            "platform": self.platform,
            "probe": self.probe.to_dict(),
            "process_environment": {name: value for name, value in self.process_environment},
            "project_version": self.project_version,
            "python_base_prefix": str(self.python_base_prefix),
            "python_dont_write_bytecode": self.python_dont_write_bytecode,
            "python_executable": str(self.python_executable),
            "python_executable_sha256": self.python_executable_sha256,
            "python_prefix": str(self.python_prefix),
            "python_prefix_configuration_sha256": self.python_prefix_configuration_sha256,
            "python_import_roots": [row.to_dict() for row in self.python_import_roots],
            "python_safe_path": self.python_safe_path,
            "python_sys_path": [str(path) for path in self.python_sys_path],
            "python_user_site_enabled": self.python_user_site_enabled,
            "python_version": self.python_version,
            "repository_root": str(self.repository_root),
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "stale_encoder_config_sha256": self.stale_encoder_config_sha256,
            "stale_model_root": str(self.stale_model_root),
            "stale_model_tree_sha256": self.stale_model_tree_sha256,
            "site_packages_byte_count": self.site_packages_byte_count,
            "site_packages_directory_count": self.site_packages_directory_count,
            "site_packages_file_count": self.site_packages_file_count,
            "site_packages_root": str(self.site_packages_root),
            "site_packages_tree_sha256": self.site_packages_tree_sha256,
            "tokenizers_version": self.tokenizers_version,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "uv_lock_path": str(self.uv_lock_path),
            "uv_lock_sha256": self.uv_lock_sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionEmbeddingBuilderReceipt:
        row = dict(_closed(value, _BUILDER_RECEIPT_FIELDS, label="MPS builder receipt"))
        distributions = row["installed_distributions"]
        if not isinstance(distributions, list):
            raise ProductionEmbeddingBuildError("installed_distributions must be an array")
        row["installed_distributions"] = tuple(
            InstalledDistribution.from_dict(item) for item in distributions
        )
        origins = row["imported_module_origins"]
        if not isinstance(origins, list):
            raise ProductionEmbeddingBuildError("imported_module_origins must be an array")
        row["imported_module_origins"] = tuple(
            ImportedModuleOrigin.from_dict(item) for item in origins
        )
        import_roots = row["python_import_roots"]
        if not isinstance(import_roots, list):
            raise ProductionEmbeddingBuildError("python_import_roots must be an array")
        row["python_import_roots"] = tuple(
            PythonImportRoot.from_dict(item) for item in import_roots
        )
        environment = row["process_environment"]
        if not isinstance(environment, Mapping) or not all(
            isinstance(name, str) and isinstance(value, str) for name, value in environment.items()
        ):
            raise ProductionEmbeddingBuildError("process_environment must be a string map")
        row["process_environment"] = tuple(
            sorted(environment.items(), key=lambda item: item[0].encode("utf-8"))
        )
        python_sys_path = row["python_sys_path"]
        if not isinstance(python_sys_path, list) or not all(
            isinstance(item, str) for item in python_sys_path
        ):
            raise ProductionEmbeddingBuildError("python_sys_path must be an array of paths")
        row["python_sys_path"] = tuple(Path(item) for item in python_sys_path)
        row["probe"] = ProductionEmbeddingProbeReceipt.from_dict(row["probe"])
        return cls(**row)  # type: ignore[arg-type]


def load_production_embedding_builder_receipt(
    path: str | Path,
    *,
    expected_sha256: str,
) -> ProductionEmbeddingBuilderReceipt:
    """Load one canonical builder receipt through its caller-supplied digest."""

    _require_sha256("expected builder receipt SHA-256", expected_sha256)
    receipt_path = Path(path)
    encoded = _read_control(receipt_path, label="MPS builder receipt")
    if _sha256(encoded) != expected_sha256:
        raise ProductionEmbeddingBuildError("MPS builder receipt differs from its caller pin")
    receipt = ProductionEmbeddingBuilderReceipt.from_dict(
        _decode_object(encoded, label="MPS builder receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionEmbeddingBuildError("MPS builder receipt is not canonical")
    return receipt


def _run_observation_command(
    arguments: Sequence[str],
    *,
    label: str,
    allow_empty: bool = False,
) -> bytes:
    try:
        completed = subprocess.run(
            list(arguments),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProductionEmbeddingBuildError(f"cannot observe {label}: {exc}") from exc
    if completed.returncode != 0 or completed.stderr:
        raise ProductionEmbeddingBuildError(f"cannot observe {label}")
    if (not completed.stdout and not allow_empty) or len(completed.stdout) > _MAX_CONTROL_BYTES:
        raise ProductionEmbeddingBuildError(f"{label} output has an invalid size")
    return completed.stdout


def _single_observation(arguments: Sequence[str], *, label: str) -> str:
    encoded = _run_observation_command(arguments, label=label)
    try:
        value = encoded.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ProductionEmbeddingBuildError(f"{label} output is not UTF-8") from exc
    if not value or "\n" in value or "\r" in value:
        raise ProductionEmbeddingBuildError(f"{label} output must contain one line")
    return _require_text(label, value)


def _require_read_only_path(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionEmbeddingBuildError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not (
        stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)
    ):
        raise ProductionEmbeddingBuildError(f"{label} must be a real file or directory")
    if metadata.st_mode & 0o222:
        raise ProductionEmbeddingBuildError(f"{label} must be fully read-only")
    return metadata


def _require_fully_read_only_tree(root: Path, *, label: str) -> None:
    """Reject links, special entries, and every owner/group/other write bit."""

    _require_real_directory(root, label=label)
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        _require_read_only_path(current_path, label=label)
        for name in (*directories, *filenames):
            _require_read_only_path(current_path / name, label=f"{label} entry")


def _site_packages_tree_identity(site_packages_root: Path) -> dict[str, object]:
    _require_fully_read_only_tree(site_packages_root, label="builder site-packages")
    try:
        tree = digest_directory_tree(site_packages_root)
    except ArtifactIntegrityError as exc:
        raise ProductionEmbeddingBuildError(
            f"cannot inventory builder site-packages: {exc}"
        ) from exc
    caches = [
        entry
        for entry in tree.entries
        if "__pycache__" in PurePosixPath(entry).parts
        or PurePosixPath(entry).suffix in {".pyc", ".pyo"}
    ]
    if caches:
        raise ProductionEmbeddingBuildError(
            f"builder site-packages contains bytecode cache entries: {caches[:5]}"
        )
    return {
        "site_packages_byte_count": tree.byte_count,
        "site_packages_directory_count": tree.directory_count,
        "site_packages_file_count": tree.file_count,
        "site_packages_tree_sha256": tree.sha256,
    }


def _installed_distribution_inventory(
    site_packages_root: Path,
) -> tuple[InstalledDistribution, ...]:
    values: dict[str, str] = {}
    try:
        distributions = tuple(importlib_metadata.distributions(path=[str(site_packages_root)]))
    except Exception as exc:
        raise ProductionEmbeddingBuildError("cannot enumerate installed distributions") from exc
    for distribution in distributions:
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str):
            raise ProductionEmbeddingBuildError("an installed distribution has no package name")
        name = re.sub(r"[-_.]+", "-", raw_name.lower())
        version = distribution.version
        if name in values:
            raise ProductionEmbeddingBuildError(f"installed distribution {name!r} is repeated")
        values[name] = version
    return tuple(InstalledDistribution(name, values[name]) for name in sorted(values))


def _digest_fixed_system_git() -> str:
    """Hash the fixed root-owned macOS Git binary without accepting a link."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(_SYSTEM_GIT, flags)
    except OSError as exc:
        raise ProductionEmbeddingBuildError(f"cannot open fixed system Git: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != 0 or before.st_mode & 0o022:
            raise ProductionEmbeddingBuildError(
                "fixed system Git must be a root-owned, non-group/other-writable regular file"
            )
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)

        def signature(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_uid,
                value.st_gid,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if signature(before) != signature(after) or byte_count != before.st_size:
            raise ProductionEmbeddingBuildError("fixed system Git changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _hash_import_regular_file(path: Path, *, label: str) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionEmbeddingBuildError(f"cannot open {label}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProductionEmbeddingBuildError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)

        def signature(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if signature(before) != signature(after) or byte_count != before.st_size:
            raise ProductionEmbeddingBuildError(f"{label} changed while hashing")
        return digest.hexdigest(), byte_count
    finally:
        os.close(descriptor)


def _digest_import_tree(root: Path) -> tuple[str, int, int, int]:
    records: list[dict[str, object]] = []
    file_count = 0
    directory_count = 0
    byte_count = 0

    def visit(
        directory: Path,
        prefix: tuple[str, ...],
        expected_identity: tuple[int, int] | None = None,
    ) -> None:
        nonlocal byte_count, directory_count, file_count
        try:
            metadata = directory.lstat()
        except OSError as exc:
            raise ProductionEmbeddingBuildError(
                f"cannot inspect Python import directory: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ProductionEmbeddingBuildError("Python import tree contains a non-directory")
        if (
            expected_identity is not None
            and (
                metadata.st_dev,
                metadata.st_ino,
            )
            != expected_identity
        ):
            raise ProductionEmbeddingBuildError("Python import directory changed during hashing")
        signature = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        relative_directory = "/".join(prefix) if prefix else "."
        records.append(
            {
                "kind": "directory",
                "mode": stat.S_IMODE(metadata.st_mode),
                "path": relative_directory,
            }
        )
        directory_count += 1
        try:
            with os.scandir(directory) as iterator:
                children = sorted(tuple(iterator), key=lambda item: item.name.encode("utf-8"))
        except (OSError, UnicodeEncodeError) as exc:
            raise ProductionEmbeddingBuildError(
                f"cannot scan Python import directory: {exc}"
            ) from exc
        for child in children:
            _require_text("Python import-tree entry", child.name)
            relative = "/".join((*prefix, child.name))
            try:
                child_metadata = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise ProductionEmbeddingBuildError(
                    f"cannot inspect Python import-tree entry: {exc}"
                ) from exc
            child_path = directory / child.name
            if stat.S_ISDIR(child_metadata.st_mode):
                visit(
                    child_path,
                    (*prefix, child.name),
                    expected_identity=(child_metadata.st_dev, child_metadata.st_ino),
                )
            elif stat.S_ISREG(child_metadata.st_mode):
                digest, size = _hash_import_regular_file(
                    child_path, label="Python import-tree file"
                )
                records.append(
                    {
                        "kind": "file",
                        "mode": stat.S_IMODE(child_metadata.st_mode),
                        "path": relative,
                        "sha256": digest,
                        "size": size,
                    }
                )
                file_count += 1
                byte_count += size
            elif stat.S_ISLNK(child_metadata.st_mode):
                try:
                    target = os.readlink(child_path)
                    after_link = child_path.lstat()
                except OSError as exc:
                    raise ProductionEmbeddingBuildError(
                        f"cannot read Python import-tree link: {exc}"
                    ) from exc
                _require_text("Python import-tree link target", target)
                if (
                    child_metadata.st_dev,
                    child_metadata.st_ino,
                    child_metadata.st_mode,
                    child_metadata.st_size,
                    child_metadata.st_mtime_ns,
                    child_metadata.st_ctime_ns,
                ) != (
                    after_link.st_dev,
                    after_link.st_ino,
                    after_link.st_mode,
                    after_link.st_size,
                    after_link.st_mtime_ns,
                    after_link.st_ctime_ns,
                ):
                    raise ProductionEmbeddingBuildError(
                        "Python import-tree link changed during hashing"
                    )
                records.append({"kind": "symlink", "path": relative, "target": target})
                file_count += 1
                byte_count += len(target.encode("utf-8"))
            else:
                raise ProductionEmbeddingBuildError(
                    "Python import tree contains a special filesystem entry"
                )
        try:
            after = directory.lstat()
        except OSError as exc:
            raise ProductionEmbeddingBuildError(
                f"cannot recheck Python import directory: {exc}"
            ) from exc
        if signature != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ProductionEmbeddingBuildError("Python import directory changed during hashing")

    visit(root, ())
    return _sha256(_canonical_bytes(records)), file_count, directory_count, byte_count


def _python_import_root_identity(path: Path, *, python_base_prefix: Path) -> PythonImportRoot:
    if not os.path.lexists(path):
        expected_absent = (
            python_base_prefix
            / "lib"
            / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
        )
        if path != expected_absent:
            raise ProductionEmbeddingBuildError(
                "only the interpreter-derived Python zip slot may be absent from sys.path"
            )
        digest = _sha256(_canonical_bytes({"kind": "absent", "path": str(path)}))
        return PythonImportRoot(
            path=path,
            kind="absent",
            sha256=digest,
            file_count=0,
            directory_count=0,
            byte_count=0,
        )
    metadata = path.lstat()
    if stat.S_ISREG(metadata.st_mode):
        digest, byte_count = _hash_import_regular_file(path, label="Python import file")
        return PythonImportRoot(
            path=path,
            kind="file",
            sha256=digest,
            file_count=1,
            directory_count=0,
            byte_count=byte_count,
        )
    if stat.S_ISDIR(metadata.st_mode):
        digest, file_count, directory_count, byte_count = _digest_import_tree(path)
        return PythonImportRoot(
            path=path,
            kind="directory",
            sha256=digest,
            file_count=file_count,
            directory_count=directory_count,
            byte_count=byte_count,
        )
    raise ProductionEmbeddingBuildError("Python import root is a link or special entry")


def _normalized_builder_sys_path(
    *,
    repository_root: Path,
    python_prefix: Path,
    python_base_prefix: Path,
    site_packages_root: Path,
) -> tuple[tuple[Path, ...], tuple[PythonImportRoot, ...]]:
    if os.environ.get("PYTHONPATH") is not None:
        raise ProductionEmbeddingBuildError("PYTHONPATH must be absent from the MPS builder")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise ProductionEmbeddingBuildError(
            "the MPS builder requires PYTHONDONTWRITEBYTECODE=1 from process start"
        )
    if not sys.flags.safe_path:
        raise ProductionEmbeddingBuildError("the MPS builder must start Python with -P")
    if site.ENABLE_USER_SITE is not False:
        raise ProductionEmbeddingBuildError("the MPS builder must disable the user site")
    normalized: list[Path] = []
    for position, value in enumerate(sys.path):
        if not isinstance(value, str) or not value:
            raise ProductionEmbeddingBuildError(
                f"builder sys.path[{position}] must be a non-empty absolute path"
            )
        candidate = Path(value)
        try:
            candidate = candidate.resolve(strict=os.path.lexists(candidate))
        except OSError as exc:
            raise ProductionEmbeddingBuildError(
                f"cannot resolve builder sys.path[{position}]: {exc}"
            ) from exc
        candidate = _canonical_absolute_path(str(candidate), label=f"builder sys.path[{position}]")
        if os.path.lexists(candidate):
            metadata = candidate.lstat()
            if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
                raise ProductionEmbeddingBuildError("builder sys.path contains a non-regular entry")
        normalized.append(candidate)
    values = tuple(normalized)
    if len(values) != len(set(values)):
        raise ProductionEmbeddingBuildError("builder sys.path repeats an import root")
    source_root = repository_root / "src"
    if source_root not in values or repository_root in values or site_packages_root not in values:
        raise ProductionEmbeddingBuildError(
            "builder sys.path must include only the pinned source and environment roots"
        )
    allowed_roots = (source_root, python_prefix, python_base_prefix)
    external = [
        path
        for path in values
        if not any(path == root or path.is_relative_to(root) for root in allowed_roots)
    ]
    if external:
        raise ProductionEmbeddingBuildError(f"builder sys.path escapes pinned roots: {external}")
    import_roots = tuple(
        sorted(
            _python_import_root_identity(path, python_base_prefix=python_base_prefix)
            for path in values
            if path not in {source_root, site_packages_root}
        )
    )
    return values, import_roots


def _module_origin(
    module: object,
    *,
    name: str,
    expected_root: Path,
    exact_path: Path | None = None,
) -> ImportedModuleOrigin:
    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str):
        raise ProductionEmbeddingBuildError(f"{name} has no concrete import origin")
    try:
        origin = Path(raw_origin).resolve(strict=True)
    except OSError as exc:
        raise ProductionEmbeddingBuildError(f"cannot resolve {name} import origin: {exc}") from exc
    origin = _canonical_absolute_path(str(origin), label=f"{name} import origin")
    if exact_path is not None:
        if origin != exact_path:
            raise ProductionEmbeddingBuildError(f"{name} import origin differs from pinned source")
    elif not origin.is_relative_to(expected_root):
        raise ProductionEmbeddingBuildError(f"{name} import origin escapes site-packages")
    try:
        digest_regular_file(origin, label=f"{name} import origin")
    except ArtifactIntegrityError as exc:
        raise ProductionEmbeddingBuildError(f"cannot verify {name} import origin: {exc}") from exc
    search_locations = getattr(module, "__path__", ())
    for location in search_locations:
        try:
            package_root = Path(location).resolve(strict=True)
        except OSError as exc:
            raise ProductionEmbeddingBuildError(
                f"cannot resolve {name} package search root: {exc}"
            ) from exc
        if not package_root.is_relative_to(expected_root):
            raise ProductionEmbeddingBuildError(
                f"{name} package search root escapes its pinned import root"
            )
    return ImportedModuleOrigin(name=name, path=origin)


def _git_source_identity(
    repository_root: Path,
    *,
    expected_commit: str,
) -> tuple[str, int]:
    _require_real_directory(repository_root, label="builder repository root")
    _require_git_commit("expected builder source commit", expected_commit)
    git = str(_SYSTEM_GIT)
    observed_root = _canonical_absolute_path(
        _single_observation(
            (git, "-C", str(repository_root), "rev-parse", "--show-toplevel"),
            label="builder Git root",
        ),
        label="builder Git root",
    )
    if observed_root != repository_root:
        raise ProductionEmbeddingBuildError("builder repository_root is not the Git root")
    observed_commit = _single_observation(
        (git, "-C", str(repository_root), "rev-parse", "HEAD"),
        label="builder Git commit",
    )
    if observed_commit != expected_commit:
        raise ProductionEmbeddingBuildError("builder checkout differs from its source commit")
    status = _run_observation_command(
        (
            git,
            "-C",
            str(repository_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        label="builder Git status",
        allow_empty=True,
    )
    if status:
        raise ProductionEmbeddingBuildError("builder checkout must be clean before a receipt")
    ignored_source = _run_observation_command(
        (
            git,
            "-C",
            str(repository_root),
            "status",
            "--porcelain=v1",
            "--ignored=matching",
            "--untracked-files=all",
            "--",
            "src",
        ),
        label="builder ignored source status",
        allow_empty=True,
    )
    if ignored_source:
        raise ProductionEmbeddingBuildError(
            "builder src tree contains ignored or untracked importable entries"
        )
    inventory = _run_observation_command(
        (
            git,
            "-C",
            str(repository_root),
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            expected_commit,
            "--",
            "pyproject.toml",
            "uv.lock",
            "src/fractal_ann_diagnostics",
        ),
        label="builder source inventory",
    )
    rows = tuple(row for row in inventory.split(b"\0") if row)
    if len(rows) < 3:
        raise ProductionEmbeddingBuildError("builder source inventory is incomplete")
    paths: list[bytes] = []
    for row in rows:
        match = re.fullmatch(rb"(100644|100755) blob ([0-9a-f]{40})\t([^\x00]+)", row)
        if match is None:
            raise ProductionEmbeddingBuildError("builder source inventory has a non-file entry")
        path_bytes = match.group(3)
        try:
            relative = path_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProductionEmbeddingBuildError("builder source path is not UTF-8") from exc
        candidate = repository_root / relative
        try:
            if candidate.resolve(strict=True) != candidate:
                raise ProductionEmbeddingBuildError("builder source crosses a filesystem alias")
            encoded = read_secure_regular_file(
                candidate,
                max_bytes=_MAX_CONTROL_BYTES,
                label="builder source file",
            )
            metadata = candidate.lstat()
        except (ArtifactIntegrityError, OSError) as exc:
            raise ProductionEmbeddingBuildError(
                f"cannot verify builder source file: {exc}"
            ) from exc
        observed_oid = hashlib.sha1(
            f"blob {len(encoded)}\0".encode() + encoded,
            usedforsecurity=False,
        ).hexdigest()
        if observed_oid.encode() != match.group(2):
            raise ProductionEmbeddingBuildError("builder working source differs from its Git blob")
        executable = bool(metadata.st_mode & 0o111)
        if executable != (match.group(1) == b"100755"):
            raise ProductionEmbeddingBuildError("builder source mode differs from its Git mode")
        paths.append(path_bytes)
    if b"pyproject.toml" not in paths or b"uv.lock" not in paths:
        raise ProductionEmbeddingBuildError("builder source inventory omits a dependency control")
    if not any(path.startswith(b"src/fractal_ann_diagnostics/") for path in paths):
        raise ProductionEmbeddingBuildError("builder source inventory omits the package source")
    _require_read_only_path(repository_root, label="builder repository root")
    _require_read_only_path(repository_root / "pyproject.toml", label="builder pyproject.toml")
    _require_read_only_path(repository_root / "uv.lock", label="builder uv.lock")
    _require_fully_read_only_tree(repository_root / "src", label="builder src tree")
    return _sha256(inventory), len(rows)


def _observe_mps_builder_environment(
    *,
    repository_root: Path,
    expected_source_commit: str,
    uv_lock_path: Path,
) -> dict[str, object]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ProductionEmbeddingBuildError(
            "the production embedding builder requires Darwin arm64"
        )
    process_environment = _validated_builder_environment(
        tuple(sorted(os.environ.items(), key=lambda item: item[0].encode("utf-8")))
    )
    git_sha256 = _digest_fixed_system_git()
    source_sha256, source_count = _git_source_identity(
        repository_root,
        expected_commit=expected_source_commit,
    )
    expected_lock = repository_root / "uv.lock"
    if uv_lock_path != expected_lock:
        raise ProductionEmbeddingBuildError(
            "builder uv_lock_path must be the source commit lockfile"
        )
    try:
        uv_lock_sha256 = digest_regular_file(uv_lock_path)
    except ArtifactIntegrityError as exc:
        raise ProductionEmbeddingBuildError(f"cannot digest builder uv.lock: {exc}") from exc
    executable = Path(sys.executable).resolve(strict=True)
    prefix = Path(sys.prefix).resolve(strict=True)
    base_prefix = Path(sys.base_prefix).resolve(strict=True)
    executable = _canonical_absolute_path(str(executable), label="builder Python executable")
    prefix = _canonical_absolute_path(str(prefix), label="builder Python prefix")
    base_prefix = _canonical_absolute_path(str(base_prefix), label="builder Python base prefix")
    if prefix == base_prefix:
        raise ProductionEmbeddingBuildError("builder Python must run inside an isolated venv")
    try:
        executable_sha256 = digest_regular_file(executable)
    except ArtifactIntegrityError as exc:
        raise ProductionEmbeddingBuildError(f"cannot digest builder Python: {exc}") from exc
    prefix_configuration = prefix / "pyvenv.cfg"
    _require_read_only_path(prefix, label="builder Python prefix")
    _require_read_only_path(prefix_configuration, label="builder pyvenv.cfg")
    try:
        prefix_configuration_sha256 = digest_regular_file(
            prefix_configuration, label="builder pyvenv.cfg"
        )
    except ArtifactIntegrityError as exc:
        raise ProductionEmbeddingBuildError(f"cannot digest builder pyvenv.cfg: {exc}") from exc
    purelib = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    platlib = Path(sysconfig.get_path("platlib")).resolve(strict=True)
    if purelib != platlib:
        raise ProductionEmbeddingBuildError(
            "builder purelib and platlib must share one closed site-packages root"
        )
    site_packages_root = _canonical_absolute_path(str(purelib), label="builder site-packages root")
    if not site_packages_root.is_relative_to(prefix):
        raise ProductionEmbeddingBuildError("builder site-packages escapes the Python prefix")
    ancestor = site_packages_root
    while True:
        _require_read_only_path(ancestor, label="builder site-packages path")
        if ancestor == prefix:
            break
        ancestor = ancestor.parent
        if not ancestor.is_relative_to(prefix) and ancestor != prefix:
            raise ProductionEmbeddingBuildError("builder site-packages path escapes its prefix")
    python_sys_path, python_import_roots = _normalized_builder_sys_path(
        repository_root=repository_root,
        python_prefix=prefix,
        python_base_prefix=base_prefix,
        site_packages_root=site_packages_root,
    )
    try:
        import numpy
        import tokenizers
        import torch
        import transformers

        import fractal_ann_diagnostics
    except ImportError as exc:
        raise ProductionEmbeddingBuildError(
            "the pinned MPS builder packages are not installed"
        ) from exc
    expected_package_root = repository_root / "src" / "fractal_ann_diagnostics"
    expected_module = expected_package_root / "production_embedding_build.py"
    if Path(__file__).resolve(strict=True) != expected_module:
        raise ProductionEmbeddingBuildError(
            "production embedding apparatus was imported outside the pinned source tree"
        )
    try:
        digest_regular_file(expected_module, label="production embedding apparatus")
    except ArtifactIntegrityError as exc:
        raise ProductionEmbeddingBuildError(
            f"cannot verify production embedding apparatus: {exc}"
        ) from exc
    module_origins = tuple(
        sorted(
            (
                _module_origin(
                    fractal_ann_diagnostics,
                    name="fractal_ann_diagnostics",
                    expected_root=repository_root / "src",
                    exact_path=expected_package_root / "__init__.py",
                ),
                _module_origin(numpy, name="numpy", expected_root=site_packages_root),
                _module_origin(tokenizers, name="tokenizers", expected_root=site_packages_root),
                _module_origin(torch, name="torch", expected_root=site_packages_root),
                _module_origin(
                    transformers,
                    name="transformers",
                    expected_root=site_packages_root,
                ),
            )
        )
    )
    distributions = _installed_distribution_inventory(site_packages_root)
    distribution_sha256 = _sha256(_canonical_bytes([row.to_dict() for row in distributions]))
    installed_versions = {row.name: row.version for row in distributions}
    project_version = installed_versions.get("fractal-ann-diagnostics")
    if project_version is None:
        raise ProductionEmbeddingBuildError(
            "builder site-packages omits the apparatus distribution metadata"
        )
    site_tree = _site_packages_tree_identity(site_packages_root)
    model = _single_observation(("/usr/sbin/sysctl", "-n", "hw.model"), label="Mac model")
    chip = _single_observation(
        ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),
        label="Mac chip",
    )
    logical_cores_text = _single_observation(
        ("/usr/sbin/sysctl", "-n", "hw.logicalcpu"),
        label="logical core count",
    )
    memory_text = _single_observation(
        ("/usr/sbin/sysctl", "-n", "hw.memsize"),
        label="physical memory bytes",
    )
    if not logical_cores_text.isascii() or not logical_cores_text.isdecimal():
        raise ProductionEmbeddingBuildError("logical core count is not a decimal integer")
    if not memory_text.isascii() or not memory_text.isdecimal():
        raise ProductionEmbeddingBuildError("physical memory is not a decimal integer")
    return {
        "builder_source_file_count": source_count,
        "builder_source_sha256": source_sha256,
        "chip": chip,
        "git_executable": _SYSTEM_GIT,
        "git_executable_sha256": git_sha256,
        "imported_module_origins": module_origins,
        "installed_distributions": distributions,
        "installed_distributions_sha256": distribution_sha256,
        "logical_cores": int(logical_cores_text),
        "machine": "arm64",
        "macos_build": _single_observation(
            ("/usr/bin/sw_vers", "-buildVersion"), label="macOS build"
        ),
        "macos_version": _single_observation(
            ("/usr/bin/sw_vers", "-productVersion"), label="macOS version"
        ),
        "memory_bytes": int(memory_text),
        "model": model,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "numpy_version": numpy.__version__,
        "platform": PRODUCTION_EMBEDDING_BUILDER_PLATFORM,
        "process_environment": process_environment,
        "project_version": project_version,
        "python_base_prefix": base_prefix,
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
        "python_executable": executable,
        "python_executable_sha256": executable_sha256,
        "python_prefix": prefix,
        "python_prefix_configuration_sha256": prefix_configuration_sha256,
        "python_import_roots": python_import_roots,
        "python_safe_path": bool(sys.flags.safe_path),
        "python_sys_path": python_sys_path,
        "python_user_site_enabled": bool(site.ENABLE_USER_SITE),
        "python_version": platform.python_version(),
        "site_packages_root": site_packages_root,
        "tokenizers_version": tokenizers.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "uv_lock_sha256": uv_lock_sha256,
        **site_tree,
    }


def _execute_fixed_mps_probe(
    *,
    current_model_root: Path,
    stale_model_root: Path,
    current_config: QwenRevisionEncoderConfig,
    stale_config: QwenRevisionEncoderConfig,
) -> ProductionEmbeddingProbeReceipt:
    encoder = QwenPairedRevisionEncoder(
        current_model_root,
        stale_model_root,
        current_config,
        stale_config,
    )
    first_started = time.monotonic_ns()
    first_current, first_stale = encoder.encode_documents(_PROBE_TEXTS)
    first_completed = time.monotonic_ns()
    second_current, second_stale = encoder.encode_documents(_PROBE_TEXTS)
    second_completed = time.monotonic_ns()
    repeat_exact = first_current.tobytes(order="C") == second_current.tobytes(
        order="C"
    ) and first_stale.tobytes(order="C") == second_stale.tobytes(order="C")
    return ProductionEmbeddingProbeReceipt(
        texts_sha256=_sha256(_canonical_bytes({"texts": list(_PROBE_TEXTS)})),
        row_count=len(_PROBE_TEXTS),
        current_encoder_config_sha256=current_config.sha256,
        stale_encoder_config_sha256=stale_config.sha256,
        current_vectors_sha256=_sha256(first_current.tobytes(order="C")),
        stale_vectors_sha256=_sha256(first_stale.tobytes(order="C")),
        output_dimension=QWEN_OUTPUT_DIMENSION,
        repeat_exact=repeat_exact,
        first_elapsed_monotonic_ns=first_completed - first_started,
        second_elapsed_monotonic_ns=second_completed - first_completed,
    )


def _probe_stable_identity(probe: ProductionEmbeddingProbeReceipt) -> dict[str, object]:
    return {
        key: value
        for key, value in probe.to_dict().items()
        if key not in {"first_elapsed_monotonic_ns", "second_elapsed_monotonic_ns"}
    }


def _verify_read_only_qwen_trees(
    *,
    current_model_root: Path,
    stale_model_root: Path,
    current_config: QwenRevisionEncoderConfig,
    stale_config: QwenRevisionEncoderConfig,
) -> None:
    _require_fully_read_only_tree(current_model_root, label="current Qwen model tree")
    _require_fully_read_only_tree(stale_model_root, label="stale Qwen model tree")
    try:
        verify_qwen_revision_tree(current_model_root, current_config)
        verify_qwen_revision_tree(stale_model_root, stale_config)
    except QwenRevisionEncoderError as exc:
        raise ProductionEmbeddingBuildError(f"MPS builder model tree differs: {exc}") from exc


def write_production_embedding_builder_receipt(
    *,
    repository_root: str | Path,
    expected_source_commit: str,
    uv_lock_path: str | Path,
    current_model_root: str | Path,
    stale_model_root: str | Path,
    batch_size: int,
    deterministic_seed: int,
    destination: str | Path,
) -> ProductionEmbeddingBuilderReceipt:
    """Observe and exclusively record the exact label-free macOS MPS builder."""

    repository = _canonical_absolute_path(str(repository_root), label="repository_root")
    lock = _canonical_absolute_path(str(uv_lock_path), label="uv_lock_path")
    current_root = _canonical_absolute_path(str(current_model_root), label="current_model_root")
    stale_root = _canonical_absolute_path(str(stale_model_root), label="stale_model_root")
    _require_git_commit("expected_source_commit", expected_source_commit)
    _require_real_directory(current_root, label="current_model_root")
    _require_real_directory(stale_root, label="stale_model_root")
    current = QwenRevisionEncoderConfig.for_arm(
        "current",
        batch_size=batch_size,
        device=PRODUCTION_EMBEDDING_DEVICE,
        deterministic_seed=deterministic_seed,
    )
    stale = QwenRevisionEncoderConfig.for_arm(
        "stale",
        batch_size=batch_size,
        device=PRODUCTION_EMBEDDING_DEVICE,
        deterministic_seed=deterministic_seed,
    )
    before = _observe_mps_builder_environment(
        repository_root=repository,
        expected_source_commit=expected_source_commit,
        uv_lock_path=lock,
    )
    _verify_read_only_qwen_trees(
        current_model_root=current_root,
        stale_model_root=stale_root,
        current_config=current,
        stale_config=stale,
    )
    probe = _execute_fixed_mps_probe(
        current_model_root=current_root,
        stale_model_root=stale_root,
        current_config=current,
        stale_config=stale,
    )
    _verify_read_only_qwen_trees(
        current_model_root=current_root,
        stale_model_root=stale_root,
        current_config=current,
        stale_config=stale,
    )
    after = _observe_mps_builder_environment(
        repository_root=repository,
        expected_source_commit=expected_source_commit,
        uv_lock_path=lock,
    )
    if before != after:
        raise ProductionEmbeddingBuildError("MPS builder identity changed during the fixed probe")
    receipt = ProductionEmbeddingBuilderReceipt(
        repository_root=repository,
        source_commit=expected_source_commit,
        uv_lock_path=lock,
        builder_kind=PRODUCTION_EMBEDDING_BUILDER_KIND,
        device=PRODUCTION_EMBEDDING_DEVICE,
        batch_size=batch_size,
        deterministic_seed=deterministic_seed,
        current_model_root=current_root,
        stale_model_root=stale_root,
        current_model_tree_sha256=QWEN_CURRENT_TREE_SHA256,
        stale_model_tree_sha256=QWEN_STALE_TREE_SHA256,
        current_encoder_config_sha256=current.sha256,
        stale_encoder_config_sha256=stale.sha256,
        probe=probe,
        **before,
    )
    destination_path = Path(destination).resolve(strict=False)
    if any(
        _paths_overlap(destination_path, root) for root in (repository, current_root, stale_root)
    ):
        raise ProductionEmbeddingBuildError(
            "builder receipt destination cannot be inside source or model trees"
        )
    try:
        write_exclusive_receipt_bytes(receipt.canonical_file_bytes(), destination_path)
    except ArtifactIntegrityError as exc:
        raise ProductionEmbeddingBuildError(f"cannot write MPS builder receipt: {exc}") from exc
    return receipt


def verify_production_embedding_builder_runtime(
    receipt: ProductionEmbeddingBuilderReceipt,
) -> None:
    """Freshly re-observe every mutable builder identity before production work."""

    if not isinstance(receipt, ProductionEmbeddingBuilderReceipt):
        raise ProductionEmbeddingBuildError("builder receipt must be typed")
    observed = _observe_mps_builder_environment(
        repository_root=receipt.repository_root,
        expected_source_commit=receipt.source_commit,
        uv_lock_path=receipt.uv_lock_path,
    )
    expected = {key: getattr(receipt, key) for key in _BUILDER_OBSERVATION_FIELDS}
    changed = [name for name in expected if observed.get(name) != expected[name]]
    if changed:
        raise ProductionEmbeddingBuildError(f"MPS builder runtime differs: {changed}")
    current = QwenRevisionEncoderConfig.for_arm(
        "current",
        batch_size=receipt.batch_size,
        device=receipt.device,
        deterministic_seed=receipt.deterministic_seed,
    )
    stale = QwenRevisionEncoderConfig.for_arm(
        "stale",
        batch_size=receipt.batch_size,
        device=receipt.device,
        deterministic_seed=receipt.deterministic_seed,
    )
    _verify_read_only_qwen_trees(
        current_model_root=receipt.current_model_root,
        stale_model_root=receipt.stale_model_root,
        current_config=current,
        stale_config=stale,
    )
    probe = _execute_fixed_mps_probe(
        current_model_root=receipt.current_model_root,
        stale_model_root=receipt.stale_model_root,
        current_config=current,
        stale_config=stale,
    )
    _verify_read_only_qwen_trees(
        current_model_root=receipt.current_model_root,
        stale_model_root=receipt.stale_model_root,
        current_config=current,
        stale_config=stale,
    )
    after = _observe_mps_builder_environment(
        repository_root=receipt.repository_root,
        expected_source_commit=receipt.source_commit,
        uv_lock_path=receipt.uv_lock_path,
    )
    changed_during_probe = [
        name for name in _BUILDER_OBSERVATION_FIELDS if after.get(name) != observed.get(name)
    ]
    if changed_during_probe:
        raise ProductionEmbeddingBuildError(
            f"MPS builder runtime changed during the fixed probe: {changed_during_probe}"
        )
    if _probe_stable_identity(probe) != _probe_stable_identity(receipt.probe):
        raise ProductionEmbeddingBuildError("MPS builder fixed-probe vectors differ")


@dataclass(frozen=True)
class ProductionCorpusSources:
    """Inventory-derived document and query paths for one registered corpus."""

    corpus_id: str
    document_paths: tuple[str, ...]
    query_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionEmbeddingBuildError("corpus source row has an unregistered corpus")
        object.__setattr__(
            self,
            "document_paths",
            _sorted_paths(list(self.document_paths), label=f"{self.corpus_id}.document_paths"),
        )
        object.__setattr__(
            self,
            "query_paths",
            _sorted_paths(list(self.query_paths), label=f"{self.corpus_id}.query_paths"),
        )
        if set(self.document_paths) & set(self.query_paths):
            raise ProductionEmbeddingBuildError("document and query source paths overlap")

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "document_paths": list(self.document_paths),
            "query_paths": list(self.query_paths),
        }

    @classmethod
    def from_dict(cls, value: object) -> ProductionCorpusSources:
        row = _closed(value, _CORPUS_SOURCE_FIELDS, label="corpus source row")
        return cls(
            corpus_id=row["corpus_id"],
            document_paths=_sorted_paths(row["document_paths"], label="document_paths"),
            query_paths=_sorted_paths(row["query_paths"], label="query_paths"),
        )


@dataclass(frozen=True)
class ProductionEmbeddingConfig:
    """Path-closed identity for the complete paired-Qwen embedding build."""

    online_staging_root: Path
    online_inventory_sha256: str
    projected_artifact_set_sha256: str
    builder_receipt: ProductionEmbeddingBuilderReceipt
    builder_receipt_sha256: str
    current_model_root: Path
    stale_model_root: Path
    output_root: Path
    output_dtype: Literal["float32"]
    current_encoder_config: QwenRevisionEncoderConfig
    stale_encoder_config: QwenRevisionEncoderConfig
    corpora: tuple[ProductionCorpusSources, ...]
    schema_version: str = PRODUCTION_EMBEDDING_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "online_staging_root",
            "current_model_root",
            "stale_model_root",
            "output_root",
        ):
            path = _canonical_absolute_path(str(getattr(self, name)), label=name)
            object.__setattr__(self, name, path)
        _require_path_separation(
            online_staging_root=self.online_staging_root,
            current_model_root=self.current_model_root,
            stale_model_root=self.stale_model_root,
            output_root=self.output_root,
        )
        _require_sha256("online_inventory_sha256", self.online_inventory_sha256)
        _require_sha256("projected_artifact_set_sha256", self.projected_artifact_set_sha256)
        if not isinstance(self.builder_receipt, ProductionEmbeddingBuilderReceipt):
            raise ProductionEmbeddingBuildError("builder_receipt must be a typed MPS receipt")
        if (
            _require_sha256("builder_receipt_sha256", self.builder_receipt_sha256)
            != self.builder_receipt.file_sha256
        ):
            raise ProductionEmbeddingBuildError("builder receipt digest differs")
        if self.output_dtype != "float32":
            raise ProductionEmbeddingBuildError("output_dtype must equal 'float32'")
        if (
            not isinstance(self.current_encoder_config, QwenRevisionEncoderConfig)
            or self.current_encoder_config.arm != "current"
        ):
            raise ProductionEmbeddingBuildError("current encoder config must bind the current arm")
        if (
            not isinstance(self.stale_encoder_config, QwenRevisionEncoderConfig)
            or self.stale_encoder_config.arm != "stale"
        ):
            raise ProductionEmbeddingBuildError("stale encoder config must bind the stale arm")
        common = (
            "batch_size",
            "deterministic_seed",
            "device",
            "document_prompt",
            "max_sequence_length",
            "normalize",
            "output_dimension",
            "query_prompt",
        )
        changed = [
            field
            for field in common
            if getattr(self.current_encoder_config, field)
            != getattr(self.stale_encoder_config, field)
        ]
        if changed:
            raise ProductionEmbeddingBuildError(f"paired encoder configs differ: {changed}")
        if self.current_encoder_config.device != PRODUCTION_EMBEDDING_DEVICE:
            raise ProductionEmbeddingBuildError("production embedding builds require device 'mps'")
        if (
            self.builder_receipt.current_model_root != self.current_model_root
            or self.builder_receipt.stale_model_root != self.stale_model_root
            or self.builder_receipt.batch_size != self.current_encoder_config.batch_size
            or self.builder_receipt.deterministic_seed
            != self.current_encoder_config.deterministic_seed
            or self.builder_receipt.current_encoder_config_sha256
            != self.current_encoder_config.sha256
            or self.builder_receipt.stale_encoder_config_sha256 != self.stale_encoder_config.sha256
        ):
            raise ProductionEmbeddingBuildError(
                "production config differs from the frozen MPS builder receipt"
            )
        corpora = tuple(self.corpora)
        if tuple(row.corpus_id for row in corpora) != FIXED_CORPORA:
            raise ProductionEmbeddingBuildError("corpus source rows must follow FIXED_CORPORA")
        object.__setattr__(self, "corpora", corpora)
        all_paths = [path for row in corpora for path in (*row.document_paths, *row.query_paths)]
        if len(all_paths) != len(set(all_paths)):
            raise ProductionEmbeddingBuildError("one staged source path cannot serve two corpora")
        if self.schema_version != PRODUCTION_EMBEDDING_CONFIG_SCHEMA:
            raise ProductionEmbeddingBuildError(
                f"schema_version must equal {PRODUCTION_EMBEDDING_CONFIG_SCHEMA!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "builder_receipt": self.builder_receipt.to_dict(),
            "builder_receipt_sha256": self.builder_receipt_sha256,
            "corpora": [row.to_dict() for row in self.corpora],
            "current_encoder_config": self.current_encoder_config.to_dict(),
            "current_model_root": str(self.current_model_root),
            "online_inventory_sha256": self.online_inventory_sha256,
            "online_staging_root": str(self.online_staging_root),
            "output_dtype": self.output_dtype,
            "output_root": str(self.output_root),
            "projected_artifact_set_sha256": self.projected_artifact_set_sha256,
            "schema_version": self.schema_version,
            "stale_encoder_config": self.stale_encoder_config.to_dict(),
            "stale_model_root": str(self.stale_model_root),
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @property
    def embedding_config(self) -> EmbeddingStoreConfig:
        current = self.current_encoder_config
        return EmbeddingStoreConfig(
            query_prompt=QWEN_QUERY_PROMPT,
            document_prompt=QWEN_DOCUMENT_PROMPT,
            max_sequence_length=QWEN_MAX_SEQUENCE_LENGTH,
            output_dimension=QWEN_OUTPUT_DIMENSION,
            normalize=True,
            batch_size=current.batch_size,
            output_dtype=self.output_dtype,
            device=current.device,
            deterministic_seed=current.deterministic_seed,
        )

    @classmethod
    def from_dict(cls, value: object) -> ProductionEmbeddingConfig:
        row = _closed(value, _CONFIG_FIELDS, label="production embedding config")
        corpora = row["corpora"]
        if not isinstance(corpora, list):
            raise ProductionEmbeddingBuildError("config corpora must be an array")
        current_config = row["current_encoder_config"]
        stale_config = row["stale_encoder_config"]
        if not isinstance(current_config, Mapping) or not isinstance(stale_config, Mapping):
            raise ProductionEmbeddingBuildError("encoder configs must be objects")
        try:
            current = QwenRevisionEncoderConfig(**current_config)  # type: ignore[arg-type]
            stale = QwenRevisionEncoderConfig(**stale_config)  # type: ignore[arg-type]
        except (TypeError, QwenRevisionEncoderError) as exc:
            raise ProductionEmbeddingBuildError(f"invalid paired Qwen config: {exc}") from exc
        builder_receipt = ProductionEmbeddingBuilderReceipt.from_dict(row["builder_receipt"])
        return cls(
            online_staging_root=_canonical_absolute_path(
                row["online_staging_root"], label="online_staging_root"
            ),
            online_inventory_sha256=row["online_inventory_sha256"],
            projected_artifact_set_sha256=row["projected_artifact_set_sha256"],
            builder_receipt=builder_receipt,
            builder_receipt_sha256=row["builder_receipt_sha256"],
            current_model_root=_canonical_absolute_path(
                row["current_model_root"], label="current_model_root"
            ),
            stale_model_root=_canonical_absolute_path(
                row["stale_model_root"], label="stale_model_root"
            ),
            output_root=_canonical_absolute_path(row["output_root"], label="output_root"),
            output_dtype=row["output_dtype"],
            current_encoder_config=current,
            stale_encoder_config=stale,
            corpora=tuple(ProductionCorpusSources.from_dict(item) for item in corpora),
            schema_version=row["schema_version"],
        )


def load_production_embedding_config(
    path: str | Path,
    *,
    expected_sha256: str,
) -> ProductionEmbeddingConfig:
    """Load one canonical config and require its exact file digest."""

    _require_sha256("expected config SHA-256", expected_sha256)
    config_path = Path(path)
    encoded = _read_control(config_path, label="production embedding config")
    if _sha256(encoded) != expected_sha256:
        raise ProductionEmbeddingBuildError("production config differs from its caller pin")
    config = ProductionEmbeddingConfig.from_dict(
        _decode_object(encoded, label="production embedding config")
    )
    if encoded != config.canonical_file_bytes():
        raise ProductionEmbeddingBuildError("production embedding config is not canonical")
    return config


def _load_admitted_inventory(root: Path) -> tuple[Mapping[str, Any], ...]:
    encoded = _read_control(root / "inventory.json", label="admitted online inventory")
    value = _decode_object(encoded, label="admitted online inventory")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ProductionEmbeddingBuildError("admitted inventory has no artifacts")
    rows: list[Mapping[str, Any]] = []
    for position, item in enumerate(artifacts):
        rows.append(
            _closed(
                item,
                _INVENTORY_ARTIFACT_FIELDS,
                label=f"inventory artifact {position}",
            )
        )
    return tuple(rows)


def _derive_corpus_sources(
    artifacts: Sequence[Mapping[str, Any]],
) -> tuple[ProductionCorpusSources, ...]:
    documents: dict[str, list[str]] = {corpus: [] for corpus in FIXED_CORPORA}
    queries: dict[str, list[str]] = {corpus: [] for corpus in FIXED_CORPORA}
    stages: dict[str, set[str]] = {corpus: set() for corpus in FIXED_CORPORA}
    seen: set[str] = set()
    for position, row in enumerate(artifacts):
        role = row["role"]
        if role not in {"corpus", "corpus-shard", "queries"}:
            continue
        corpus = row["dataset"]
        if corpus not in documents:
            raise ProductionEmbeddingBuildError(
                f"inventory artifact {position} assigns an embedding source to {corpus!r}"
            )
        if row["visibility"] != "online":
            raise ProductionEmbeddingBuildError("an embedding source is not online-visible")
        path = _relative_jsonl_path(row["path"], label=f"inventory artifact {position}.path")
        if path in seen:
            raise ProductionEmbeddingBuildError("inventory repeats an embedding source path")
        seen.add(path)
        if role in {"corpus", "corpus-shard"}:
            if row["stage"] is not None:
                raise ProductionEmbeddingBuildError("a document source declares a query stage")
            documents[corpus].append(path)
        else:
            stage = row["stage"]
            if stage not in _QUERY_STAGES or stage in stages[corpus]:
                raise ProductionEmbeddingBuildError(
                    f"{corpus} must have one query artifact per registered stage"
                )
            stages[corpus].add(stage)
            queries[corpus].append(path)
    result: list[ProductionCorpusSources] = []
    for corpus in FIXED_CORPORA:
        if stages[corpus] != _QUERY_STAGES:
            raise ProductionEmbeddingBuildError(
                f"{corpus} query sources do not cover fit, calibration, and sealed"
            )
        result.append(
            ProductionCorpusSources(
                corpus_id=corpus,
                document_paths=tuple(sorted(documents[corpus], key=lambda item: item.encode())),
                query_paths=tuple(sorted(queries[corpus], key=lambda item: item.encode())),
            )
        )
    return tuple(result)


def write_production_embedding_config(
    *,
    online_staging_root: str | Path,
    expected_inventory_sha256: str,
    builder_receipt_path: str | Path,
    expected_builder_receipt_sha256: str,
    current_model_root: str | Path,
    stale_model_root: str | Path,
    output_root: str | Path,
    batch_size: int,
    device: str,
    deterministic_seed: int,
    output_dtype: Literal["float32"],
    destination: str | Path,
) -> ProductionEmbeddingConfig:
    """Derive, bind, and exclusively write the production build configuration."""

    staging = _canonical_absolute_path(str(online_staging_root), label="online_staging_root")
    current_root = _canonical_absolute_path(str(current_model_root), label="current_model_root")
    stale_root = _canonical_absolute_path(str(stale_model_root), label="stale_model_root")
    output = _canonical_absolute_path(str(output_root), label="output_root")
    _require_real_directory(staging, label="online_staging_root")
    _require_real_directory(current_root, label="current_model_root")
    _require_real_directory(stale_root, label="stale_model_root")
    _require_path_separation(
        online_staging_root=staging,
        current_model_root=current_root,
        stale_model_root=stale_root,
        output_root=output,
    )
    _require_sha256("expected inventory SHA-256", expected_inventory_sha256)
    builder_receipt_control = _canonical_absolute_path(
        str(builder_receipt_path), label="builder_receipt_path"
    )
    builder_receipt = load_production_embedding_builder_receipt(
        builder_receipt_control,
        expected_sha256=expected_builder_receipt_sha256,
    )
    verify_production_embedding_builder_runtime(builder_receipt)
    try:
        projection = verify_online_staging_projection(
            staging,
            expected_inventory_sha256=expected_inventory_sha256,
        )
        current = QwenRevisionEncoderConfig.for_arm(
            "current",
            batch_size=batch_size,
            device=device,
            deterministic_seed=deterministic_seed,
        )
        stale = QwenRevisionEncoderConfig.for_arm(
            "stale",
            batch_size=batch_size,
            device=device,
            deterministic_seed=deterministic_seed,
        )
        verify_qwen_revision_tree(current_root, current)
        verify_qwen_revision_tree(stale_root, stale)
    except (StudyDataError, QwenRevisionEncoderError) as exc:
        raise ProductionEmbeddingBuildError(str(exc)) from exc
    config = ProductionEmbeddingConfig(
        online_staging_root=staging,
        online_inventory_sha256=projection.inventory_sha256,
        projected_artifact_set_sha256=projection.projected_artifact_set_sha256,
        builder_receipt=builder_receipt,
        builder_receipt_sha256=builder_receipt.file_sha256,
        current_model_root=current_root,
        stale_model_root=stale_root,
        output_root=output,
        output_dtype=output_dtype,
        current_encoder_config=current,
        stale_encoder_config=stale,
        corpora=_derive_corpus_sources(_load_admitted_inventory(staging)),
    )
    config_destination = Path(destination)
    if any(
        _paths_overlap(config_destination.resolve(strict=False), root)
        for root in (staging, current_root, stale_root, output)
    ):
        raise ProductionEmbeddingBuildError(
            "config destination cannot be inside an input or output tree"
        )
    try:
        write_exclusive_receipt_bytes(config.canonical_file_bytes(), config_destination)
    except ArtifactIntegrityError as exc:
        raise ProductionEmbeddingBuildError(f"cannot write production config: {exc}") from exc
    return config


@dataclass(frozen=True)
class ProductionCorpusBuildEvidence:
    """Typed shard receipt recorded after one corpus reaches a final store.

    The timing and RSS fields are operational evidence.  The remaining fields
    bind the shard to the closed five-corpus config, admitted sources, and
    complete paired-Qwen store bytes.  Final aggregation consumes these files
    in ``FIXED_CORPORA`` order, never in worker-completion order.
    """

    corpus_id: str
    production_config_sha256: str
    online_inventory_sha256: str
    source_inventory_sha256: str
    embedding_receipt_sha256: str
    embedding_tree_sha256: str
    document_count: int
    query_count: int
    started_at_utc: str
    completed_at_utc: str
    elapsed_monotonic_ns: int
    process_peak_rss_bytes: int
    status: Literal["built", "resumed", "verified-existing"]
    schema_version: str = PRODUCTION_EMBEDDING_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionEmbeddingBuildError("evidence corpus is not registered")
        for name in (
            "production_config_sha256",
            "online_inventory_sha256",
            "source_inventory_sha256",
            "embedding_receipt_sha256",
            "embedding_tree_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_positive_integer("document_count", self.document_count)
        _require_positive_integer("query_count", self.query_count)
        _require_nonnegative_integer("elapsed_monotonic_ns", self.elapsed_monotonic_ns)
        _require_positive_integer("process_peak_rss_bytes", self.process_peak_rss_bytes)
        started = _parse_utc(self.started_at_utc, label="started_at_utc")
        completed = _parse_utc(self.completed_at_utc, label="completed_at_utc")
        if completed < started:
            raise ProductionEmbeddingBuildError("evidence completion precedes its start")
        if self.status not in {"built", "resumed", "verified-existing"}:
            raise ProductionEmbeddingBuildError("evidence status is invalid")
        if self.schema_version != PRODUCTION_EMBEDDING_EVIDENCE_SCHEMA:
            raise ProductionEmbeddingBuildError("evidence schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "completed_at_utc": self.completed_at_utc,
            "corpus_id": self.corpus_id,
            "document_count": self.document_count,
            "elapsed_monotonic_ns": self.elapsed_monotonic_ns,
            "embedding_receipt_sha256": self.embedding_receipt_sha256,
            "embedding_tree_sha256": self.embedding_tree_sha256,
            "online_inventory_sha256": self.online_inventory_sha256,
            "process_peak_rss_bytes": self.process_peak_rss_bytes,
            "production_config_sha256": self.production_config_sha256,
            "query_count": self.query_count,
            "schema_version": self.schema_version,
            "source_inventory_sha256": self.source_inventory_sha256,
            "started_at_utc": self.started_at_utc,
            "status": self.status,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @property
    def shard_receipt_sha256(self) -> str:
        """Return the canonical file digest under its shard-receipt role."""

        return self.sha256

    @classmethod
    def from_dict(cls, value: object) -> ProductionCorpusBuildEvidence:
        row = _closed(value, _EVIDENCE_FIELDS, label="corpus build evidence")
        return cls(**row)  # type: ignore[arg-type]


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProductionEmbeddingBuildError(f"{label} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProductionEmbeddingBuildError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo != timezone.utc or parsed.isoformat().replace("+00:00", "Z") != value:
        raise ProductionEmbeddingBuildError(f"{label} must be canonical UTC")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


@dataclass(frozen=True)
class ProductionEmbeddingSuiteCorpus:
    corpus_id: str
    embedding_receipt_sha256: str
    embedding_tree_sha256: str
    evidence_sha256: str
    evidence_file_sha256: str

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionEmbeddingBuildError("suite row corpus is not registered")
        for name in (
            "embedding_receipt_sha256",
            "embedding_tree_sha256",
            "evidence_sha256",
            "evidence_file_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.evidence_sha256 != self.evidence_file_sha256:
            raise ProductionEmbeddingBuildError("evidence semantic and file digests differ")

    def to_dict(self) -> dict[str, str]:
        return {
            "corpus_id": self.corpus_id,
            "embedding_receipt_sha256": self.embedding_receipt_sha256,
            "embedding_tree_sha256": self.embedding_tree_sha256,
            "evidence_sha256": self.evidence_sha256,
            "evidence_file_sha256": self.evidence_file_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProductionEmbeddingSuiteCorpus:
        row = _closed(value, _SUITE_CORPUS_FIELDS, label="suite corpus row")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProductionEmbeddingSuiteReceipt:
    production_config_sha256: str
    online_inventory_sha256: str
    projected_artifact_set_sha256: str
    corpora: tuple[ProductionEmbeddingSuiteCorpus, ...]
    schema_version: str = PRODUCTION_EMBEDDING_SUITE_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("production_config_sha256", self.production_config_sha256)
        _require_sha256("online_inventory_sha256", self.online_inventory_sha256)
        _require_sha256("projected_artifact_set_sha256", self.projected_artifact_set_sha256)
        corpora = tuple(self.corpora)
        if tuple(row.corpus_id for row in corpora) != FIXED_CORPORA:
            raise ProductionEmbeddingBuildError("suite rows must follow FIXED_CORPORA")
        object.__setattr__(self, "corpora", corpora)
        if self.schema_version != PRODUCTION_EMBEDDING_SUITE_SCHEMA:
            raise ProductionEmbeddingBuildError("suite receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "corpora": [row.to_dict() for row in self.corpora],
            "online_inventory_sha256": self.online_inventory_sha256,
            "production_config_sha256": self.production_config_sha256,
            "projected_artifact_set_sha256": self.projected_artifact_set_sha256,
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionEmbeddingSuiteReceipt:
        row = _closed(value, _SUITE_FIELDS, label="production embedding suite receipt")
        corpora = row["corpora"]
        if not isinstance(corpora, list):
            raise ProductionEmbeddingBuildError("suite corpora must be an array")
        return cls(
            production_config_sha256=row["production_config_sha256"],
            online_inventory_sha256=row["online_inventory_sha256"],
            projected_artifact_set_sha256=row["projected_artifact_set_sha256"],
            corpora=tuple(ProductionEmbeddingSuiteCorpus.from_dict(item) for item in corpora),
            schema_version=row["schema_version"],
        )


def _load_evidence(path: Path) -> ProductionCorpusBuildEvidence:
    encoded = _read_control(path, label="corpus build evidence")
    evidence = ProductionCorpusBuildEvidence.from_dict(
        _decode_object(encoded, label="corpus build evidence")
    )
    if encoded != evidence.canonical_file_bytes():
        raise ProductionEmbeddingBuildError("corpus build evidence is not canonical")
    return evidence


def _load_suite(path: Path) -> ProductionEmbeddingSuiteReceipt:
    encoded = _read_control(path, label="production embedding suite receipt")
    receipt = ProductionEmbeddingSuiteReceipt.from_dict(
        _decode_object(encoded, label="production embedding suite receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionEmbeddingBuildError("production suite receipt is not canonical")
    return receipt


def _selection(
    config: ProductionEmbeddingConfig,
    row: ProductionCorpusSources,
) -> StagedEmbeddingSources:
    return StagedEmbeddingSources(
        root=config.online_staging_root,
        inventory_sha256=config.online_inventory_sha256,
        document_paths=row.document_paths,
        query_paths=row.query_paths,
    )


@dataclass(frozen=True)
class _ExpectedStore:
    source_inventory_sha256: str
    document_count: int
    query_count: int


def _expected_store(
    config: ProductionEmbeddingConfig,
    row: ProductionCorpusSources,
) -> _ExpectedStore:
    try:
        resolved = embedding_store_module._load_sources(_selection(config, row))
    except EmbeddingStoreError as exc:
        raise ProductionEmbeddingBuildError(
            f"cannot resolve {row.corpus_id} sources: {exc}"
        ) from exc
    return _ExpectedStore(
        source_inventory_sha256=resolved.source_inventory_sha256,
        document_count=resolved.document_count,
        query_count=resolved.query_count,
    )


def _model_specs(config: ProductionEmbeddingConfig) -> tuple[LocalModelSpec, LocalModelSpec]:
    return (
        LocalModelSpec(
            path=config.current_model_root,
            revision=QWEN_CURRENT_REVISION,
            tree_sha256=QWEN_CURRENT_TREE_SHA256,
        ),
        LocalModelSpec(
            path=config.stale_model_root,
            revision=QWEN_STALE_REVISION,
            tree_sha256=QWEN_STALE_TREE_SHA256,
        ),
    )


def _verify_receipt_binding(
    receipt: EmbeddingStoreReceipt,
    *,
    config: ProductionEmbeddingConfig,
    expected: _ExpectedStore,
    adapter: QwenPairedRevisionEmbeddingAdapter,
) -> None:
    current_model, stale_model = _model_specs(config)
    expected_config = config.embedding_config
    if (
        receipt.staged_inventory_sha256 != config.online_inventory_sha256
        or receipt.source_inventory_sha256 != expected.source_inventory_sha256
        or receipt.config_sha256 != expected_config.sha256
        or receipt.document_count != expected.document_count
        or receipt.query_count != expected.query_count
        or receipt.current_model
        != current_model.binding(encoder_id=adapter.current_implementation_id)
        or receipt.old_model != stale_model.binding(encoder_id=adapter.old_implementation_id)
    ):
        raise ProductionEmbeddingBuildError(
            "embedding receipt differs from the sealed build inputs"
        )
    expected_vectors = {
        "current_documents": (
            QWEN_CURRENT_REVISION,
            QWEN_CURRENT_TREE_SHA256,
            QWEN_DOCUMENT_PROMPT,
        ),
        "current_queries": (QWEN_CURRENT_REVISION, QWEN_CURRENT_TREE_SHA256, QWEN_QUERY_PROMPT),
        "old_documents": (QWEN_STALE_REVISION, QWEN_STALE_TREE_SHA256, QWEN_DOCUMENT_PROMPT),
        "old_queries": (QWEN_STALE_REVISION, QWEN_STALE_TREE_SHA256, QWEN_QUERY_PROMPT),
    }
    if set(receipt.vectors) != set(expected_vectors):
        raise ProductionEmbeddingBuildError("embedding receipt omits a paired Qwen matrix")
    for name, (revision, tree_sha256, prompt) in expected_vectors.items():
        vector = receipt.vectors[name]
        rows = expected.document_count if name.endswith("documents") else expected.query_count
        if (
            vector.dtype != config.output_dtype
            or vector.shape != (rows, QWEN_OUTPUT_DIMENSION)
            or vector.model_revision != revision
            or vector.model_tree_sha256 != tree_sha256
            or vector.prompt_sha256 != _sha256(prompt.encode("utf-8"))
        ):
            raise ProductionEmbeddingBuildError(f"embedding vector {name!r} differs")


def _partial_paths(output_root: Path, corpus_id: str) -> tuple[Path, Path]:
    return (
        output_root / f".{corpus_id}.partial",
        output_root / f".{corpus_id}.checkpoint.json",
    )


def _member_names(output_root: Path) -> set[str]:
    try:
        return {child.name for child in output_root.iterdir()}
    except OSError as exc:
        raise ProductionEmbeddingBuildError(f"cannot scan embedding output root: {exc}") from exc


def _validate_output_membership(
    output_root: Path,
    *,
    final: bool,
    active_corpus: str | None = None,
) -> None:
    if active_corpus is not None and active_corpus not in FIXED_CORPORA:
        raise ProductionEmbeddingBuildError("active corpus is not registered")
    _require_real_directory(output_root, label="embedding output root")
    allowed = {PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY, PRODUCTION_EMBEDDING_SUITE_FILENAME}
    required = {PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY}
    for corpus in FIXED_CORPORA:
        allowed.update({corpus, f".{corpus}.partial", f".{corpus}.checkpoint.json"})
        if final:
            required.add(corpus)
    if final:
        required.add(PRODUCTION_EMBEDDING_SUITE_FILENAME)
    observed = _member_names(output_root)
    if not required.issubset(observed) or not observed.issubset(allowed):
        raise ProductionEmbeddingBuildError(
            f"embedding output membership differs; missing={sorted(required - observed)}, "
            f"unexpected={sorted(observed - allowed)}"
        )
    evidence_root = output_root / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY
    _require_real_directory(evidence_root, label="embedding evidence directory")
    expected_evidence = {f"{corpus}.json" for corpus in FIXED_CORPORA}
    observed_evidence = _member_names(evidence_root)
    if final:
        if observed_evidence != expected_evidence:
            raise ProductionEmbeddingBuildError("embedding evidence membership differs")
    elif not observed_evidence.issubset(expected_evidence):
        raise ProductionEmbeddingBuildError("embedding evidence has an undeclared member")
    for corpus in FIXED_CORPORA:
        store = output_root / corpus
        partial, checkpoint = _partial_paths(output_root, corpus)
        store_exists = os.path.lexists(store)
        partial_exists = os.path.lexists(partial)
        checkpoint_exists = os.path.lexists(checkpoint)
        # A paired store publishes its work directory and checkpoint in two
        # separate filesystem operations.  One fixed-corpus worker must not
        # interpret another worker's sub-second transition as corruption.  It
        # still validates its own corpus strictly; aggregation and terminal
        # verification validate every corpus strictly.
        if active_corpus is not None and corpus != active_corpus and not final:
            continue
        if partial_exists != checkpoint_exists:
            raise ProductionEmbeddingBuildError(
                f"{corpus} partial store and checkpoint must appear together"
            )
        if store_exists and partial_exists:
            raise ProductionEmbeddingBuildError(f"{corpus} has both final and partial stores")


def _mkdir_shared_directory(path: Path, *, label: str) -> None:
    """Create one shared directory without treating a peer's creation as failure."""

    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise ProductionEmbeddingBuildError(f"cannot create {label}: {exc}") from exc
    _require_real_directory(path, label=label)


def _worker_lock_root(output_root: Path) -> Path:
    """Return the derived operational lock root outside immutable outputs."""

    return output_root.parent / f".{output_root.name}.worker-locks"


@contextmanager
def _corpus_worker_lock(output_root: Path, corpus_id: str) -> Iterator[None]:
    """Admit at most one live writer or verifier for one corpus.

    The persistent lock inode sits beside, rather than inside, the embedding
    output tree. ``flock`` releases it when a worker exits or crashes, while
    retaining the inode avoids an unlink-and-recreate race between workers.
    """

    if corpus_id not in FIXED_CORPORA:
        raise ProductionEmbeddingBuildError("worker lock corpus is not registered")
    lock_root = _worker_lock_root(output_root)
    if os.path.lexists(lock_root):
        _require_real_directory(lock_root, label="embedding worker lock root")
    else:
        _require_real_directory(output_root.parent, label="embedding output parent")
        _mkdir_shared_directory(lock_root, label="embedding worker lock root")
    lock_name = f"{corpus_id}.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        parent_descriptor = os.open(output_root.parent, directory_flags)
    except OSError as exc:
        raise ProductionEmbeddingBuildError(f"cannot open worker-lock parent: {exc}") from exc
    try:
        lock_root_descriptor = os.open(
            lock_root.name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        os.close(parent_descriptor)
        raise ProductionEmbeddingBuildError(f"cannot open worker lock root: {exc}") from exc
    try:
        descriptor = os.open(lock_name, flags, 0o600, dir_fd=lock_root_descriptor)
    except OSError as exc:
        os.close(lock_root_descriptor)
        os.close(parent_descriptor)
        raise ProductionEmbeddingBuildError(f"cannot open {corpus_id} worker lock: {exc}") from exc
    acquired = False

    def require_named_identity() -> None:
        try:
            parent_metadata = os.fstat(parent_descriptor)
            root_descriptor_metadata = os.fstat(lock_root_descriptor)
            root_path_metadata = os.stat(
                lock_root.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ProductionEmbeddingBuildError(
                f"cannot revalidate embedding worker lock root: {exc}"
            ) from exc
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.geteuid()
            or parent_metadata.st_mode & 0o022
        ):
            raise ProductionEmbeddingBuildError(
                "embedding output parent must be owner-controlled and non-group/other-writable"
            )
        if (
            not stat.S_ISDIR(root_descriptor_metadata.st_mode)
            or root_descriptor_metadata.st_uid != os.geteuid()
            or root_descriptor_metadata.st_nlink < 2
            or stat.S_IMODE(root_descriptor_metadata.st_mode) != 0o700
        ):
            raise ProductionEmbeddingBuildError(
                "embedding worker lock root must be an owner-controlled mode-0700 directory"
            )
        if (
            root_descriptor_metadata.st_dev,
            root_descriptor_metadata.st_ino,
        ) != (root_path_metadata.st_dev, root_path_metadata.st_ino):
            raise ProductionEmbeddingBuildError("embedding worker lock root pathname changed")
        try:
            descriptor_metadata = os.fstat(descriptor)
            path_metadata = os.stat(
                lock_name,
                dir_fd=lock_root_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ProductionEmbeddingBuildError(
                f"cannot revalidate {corpus_id} worker lock pathname: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(descriptor_metadata.st_mode)
            or descriptor_metadata.st_nlink != 1
            or descriptor_metadata.st_size != 0
            or stat.S_IMODE(descriptor_metadata.st_mode) != 0o600
            or descriptor_metadata.st_uid != os.geteuid()
            or (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
            or path_metadata.st_nlink != 1
            or path_metadata.st_size != 0
        ):
            raise ProductionEmbeddingBuildError(
                f"{corpus_id} worker lock pathname does not name its private empty inode"
            )

    try:
        require_named_identity()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProductionEmbeddingBuildError(
                f"{corpus_id} already has an active embedding worker"
            ) from exc
        except OSError as exc:
            raise ProductionEmbeddingBuildError(
                f"cannot acquire {corpus_id} worker lock: {exc}"
            ) from exc
        acquired = True
        require_named_identity()
        yield
    finally:
        identity_error: ProductionEmbeddingBuildError | None = None
        if acquired:
            try:
                require_named_identity()
            except (OSError, ProductionEmbeddingBuildError) as exc:
                identity_error = ProductionEmbeddingBuildError(
                    f"{corpus_id} worker lock identity changed during the critical section: {exc}"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)
        os.close(lock_root_descriptor)
        os.close(parent_descriptor)
        if identity_error is not None:
            raise identity_error


@contextmanager
def _corpus_worker_locks(
    output_root: Path,
    corpus_ids: Sequence[str],
) -> Iterator[None]:
    """Acquire a fixed-order set of corpus locks without lock-order cycles."""

    requested = tuple(corpus_ids)
    if len(requested) != len(set(requested)) or not set(requested).issubset(FIXED_CORPORA):
        raise ProductionEmbeddingBuildError("worker lock set is not a unique corpus subset")
    ordered = tuple(corpus for corpus in FIXED_CORPORA if corpus in requested)
    with ExitStack() as stack:
        for corpus_id in ordered:
            stack.enter_context(_corpus_worker_lock(output_root, corpus_id))
        yield


def _prepare_output_root(output_root: Path, *, active_corpus: str | None = None) -> None:
    if os.path.lexists(output_root):
        _require_real_directory(output_root, label="embedding output root")
    else:
        parent = output_root.parent
        _require_real_directory(parent, label="embedding output parent")
        _mkdir_shared_directory(output_root, label="embedding output root")
    evidence_root = output_root / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY
    if os.path.lexists(evidence_root):
        _require_real_directory(evidence_root, label="embedding evidence directory")
    else:
        _mkdir_shared_directory(evidence_root, label="embedding evidence directory")
    _validate_output_membership(
        output_root,
        final=False,
        active_corpus=active_corpus,
    )


def _verify_projection(config: ProductionEmbeddingConfig) -> None:
    try:
        receipt = verify_online_staging_projection(
            config.online_staging_root,
            expected_inventory_sha256=config.online_inventory_sha256,
        )
    except StudyDataError as exc:
        raise ProductionEmbeddingBuildError(f"online staging admission failed: {exc}") from exc
    if receipt.projected_artifact_set_sha256 != config.projected_artifact_set_sha256:
        raise ProductionEmbeddingBuildError("projected artifact set differs from the config")
    derived_sources = _derive_corpus_sources(_load_admitted_inventory(config.online_staging_root))
    if derived_sources != config.corpora:
        raise ProductionEmbeddingBuildError("derived corpus sources differ from the sealed config")


def _verify_model_roots(config: ProductionEmbeddingConfig) -> None:
    try:
        verify_qwen_revision_tree(config.current_model_root, config.current_encoder_config)
        verify_qwen_revision_tree(config.stale_model_root, config.stale_encoder_config)
    except QwenRevisionEncoderError as exc:
        raise ProductionEmbeddingBuildError(f"local Qwen model admission failed: {exc}") from exc


def _evidence_matches(
    evidence: ProductionCorpusBuildEvidence,
    *,
    corpus_id: str,
    config: ProductionEmbeddingConfig,
    expected: _ExpectedStore,
    receipt: EmbeddingStoreReceipt,
    tree_sha256: str,
) -> None:
    if (
        evidence.corpus_id != corpus_id
        or evidence.production_config_sha256 != config.file_sha256
        or evidence.online_inventory_sha256 != config.online_inventory_sha256
        or evidence.source_inventory_sha256 != expected.source_inventory_sha256
        or evidence.embedding_receipt_sha256 != receipt.receipt_sha256
        or evidence.embedding_tree_sha256 != tree_sha256
        or evidence.document_count != receipt.document_count
        or evidence.query_count != receipt.query_count
    ):
        raise ProductionEmbeddingBuildError(f"{corpus_id} evidence differs from the final store")


def _verify_one_store(
    config: ProductionEmbeddingConfig,
    row: ProductionCorpusSources,
    *,
    adapter: QwenPairedRevisionEmbeddingAdapter,
) -> tuple[EmbeddingStoreReceipt, _ExpectedStore, str]:
    expected = _expected_store(config, row)
    try:
        receipt = verify_embedding_store(config.output_root / row.corpus_id)
        tree = digest_directory_tree(config.output_root / row.corpus_id)
    except (EmbeddingStoreError, ArtifactIntegrityError) as exc:
        raise ProductionEmbeddingBuildError(
            f"cannot verify {row.corpus_id} embedding store: {exc}"
        ) from exc
    _verify_receipt_binding(receipt, config=config, expected=expected, adapter=adapter)
    return receipt, expected, tree.sha256


def _record_or_verify_evidence(
    *,
    config: ProductionEmbeddingConfig,
    row: ProductionCorpusSources,
    receipt: EmbeddingStoreReceipt,
    expected: _ExpectedStore,
    tree_sha256: str,
    status: Literal["built", "resumed", "verified-existing"],
    started_at_utc: str,
    started_ns: int,
) -> ProductionCorpusBuildEvidence:
    path = config.output_root / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY / f"{row.corpus_id}.json"
    if os.path.lexists(path):
        evidence = _load_evidence(path)
        _evidence_matches(
            evidence,
            corpus_id=row.corpus_id,
            config=config,
            expected=expected,
            receipt=receipt,
            tree_sha256=tree_sha256,
        )
        return evidence
    evidence = ProductionCorpusBuildEvidence(
        corpus_id=row.corpus_id,
        production_config_sha256=config.file_sha256,
        online_inventory_sha256=config.online_inventory_sha256,
        source_inventory_sha256=expected.source_inventory_sha256,
        embedding_receipt_sha256=receipt.receipt_sha256,
        embedding_tree_sha256=tree_sha256,
        document_count=receipt.document_count,
        query_count=receipt.query_count,
        started_at_utc=started_at_utc,
        completed_at_utc=_utc_now(),
        elapsed_monotonic_ns=time.monotonic_ns() - started_ns,
        process_peak_rss_bytes=_peak_rss_bytes(),
        status=status,
    )
    try:
        write_exclusive_receipt_bytes(evidence.canonical_file_bytes(), path)
    except ArtifactIntegrityError as exc:
        raise ProductionEmbeddingBuildError(
            f"cannot write {row.corpus_id} evidence: {exc}"
        ) from exc
    return evidence


_CompletedCorpusRow = tuple[
    ProductionCorpusSources,
    EmbeddingStoreReceipt,
    str,
    ProductionCorpusBuildEvidence,
]


def _ordered_completed_rows(
    config: ProductionEmbeddingConfig,
    rows: Sequence[_CompletedCorpusRow],
) -> tuple[_CompletedCorpusRow, ...]:
    """Close aggregation to exactly one verified row per registered corpus."""

    by_corpus: dict[str, _CompletedCorpusRow] = {}
    configured = {row.corpus_id: row for row in config.corpora}
    for position, completed in enumerate(rows):
        if not isinstance(completed, tuple) or len(completed) != 4:
            raise ProductionEmbeddingBuildError(
                f"completed shard row {position} has the wrong typed shape"
            )
        source_row, receipt, tree_sha256, evidence = completed
        if not isinstance(source_row, ProductionCorpusSources):
            raise ProductionEmbeddingBuildError("completed shard source row has the wrong type")
        corpus_id = source_row.corpus_id
        if corpus_id in by_corpus:
            raise ProductionEmbeddingBuildError(
                f"completed shards repeat registered corpus {corpus_id!r}"
            )
        if configured.get(corpus_id) != source_row:
            raise ProductionEmbeddingBuildError(
                f"completed shard {corpus_id!r} differs from the closed config"
            )
        if not isinstance(receipt, EmbeddingStoreReceipt):
            raise ProductionEmbeddingBuildError("completed shard receipt has the wrong type")
        _require_sha256("completed shard tree SHA-256", tree_sha256)
        if not isinstance(evidence, ProductionCorpusBuildEvidence):
            raise ProductionEmbeddingBuildError("completed shard evidence has the wrong type")
        if evidence.corpus_id != corpus_id:
            raise ProductionEmbeddingBuildError("completed shard evidence changes corpus identity")
        by_corpus[corpus_id] = completed
    observed = set(by_corpus)
    expected = set(FIXED_CORPORA)
    if observed != expected:
        raise ProductionEmbeddingBuildError(
            f"completed shard set differs; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return tuple(by_corpus[corpus_id] for corpus_id in FIXED_CORPORA)


def _suite_from_rows(
    config: ProductionEmbeddingConfig,
    rows: Sequence[_CompletedCorpusRow],
) -> ProductionEmbeddingSuiteReceipt:
    corpus_rows: list[ProductionEmbeddingSuiteCorpus] = []
    for source_row, receipt, tree_sha256, evidence in _ordered_completed_rows(config, rows):
        evidence_path = (
            config.output_root
            / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY
            / f"{source_row.corpus_id}.json"
        )
        try:
            evidence_file_sha256 = digest_regular_file(
                evidence_path, label=f"{source_row.corpus_id} evidence"
            )
        except ArtifactIntegrityError as exc:
            raise ProductionEmbeddingBuildError(f"cannot hash corpus evidence: {exc}") from exc
        corpus_rows.append(
            ProductionEmbeddingSuiteCorpus(
                corpus_id=source_row.corpus_id,
                embedding_receipt_sha256=receipt.receipt_sha256,
                embedding_tree_sha256=tree_sha256,
                evidence_sha256=evidence.sha256,
                evidence_file_sha256=evidence_file_sha256,
            )
        )
    return ProductionEmbeddingSuiteReceipt(
        production_config_sha256=config.file_sha256,
        online_inventory_sha256=config.online_inventory_sha256,
        projected_artifact_set_sha256=config.projected_artifact_set_sha256,
        corpora=tuple(corpus_rows),
    )


def _paired_adapter(
    config: ProductionEmbeddingConfig,
    supplied: QwenPairedRevisionEmbeddingAdapter | None,
) -> QwenPairedRevisionEmbeddingAdapter:
    expected = QwenPairedRevisionEmbeddingAdapter(
        config.current_encoder_config,
        config.stale_encoder_config,
    )
    if supplied is None:
        return expected
    if not isinstance(supplied, QwenPairedRevisionEmbeddingAdapter) or (
        supplied.current_implementation_id != expected.current_implementation_id
        or supplied.old_implementation_id != expected.old_implementation_id
    ):
        raise ProductionEmbeddingBuildError(
            "paired encoder differs from the two config-derived Qwen arms"
        )
    return supplied


def _admit_build_inputs(
    config: ProductionEmbeddingConfig,
    *,
    prepare_output: bool,
    active_corpus: str | None = None,
) -> None:
    if not isinstance(config, ProductionEmbeddingConfig):
        raise ProductionEmbeddingBuildError("config must be ProductionEmbeddingConfig")
    verify_production_embedding_builder_runtime(config.builder_receipt)
    _verify_projection(config)
    _require_real_directory(config.current_model_root, label="current_model_root")
    _require_real_directory(config.stale_model_root, label="stale_model_root")
    _verify_model_roots(config)
    if prepare_output:
        _prepare_output_root(config.output_root, active_corpus=active_corpus)


def _build_one_corpus(
    config: ProductionEmbeddingConfig,
    row: ProductionCorpusSources,
    *,
    adapter: QwenPairedRevisionEmbeddingAdapter,
) -> _CompletedCorpusRow:
    _validate_output_membership(
        config.output_root,
        final=False,
        active_corpus=row.corpus_id,
    )
    started_at = _utc_now()
    started_ns = time.monotonic_ns()
    store_root = config.output_root / row.corpus_id
    partial, checkpoint = _partial_paths(config.output_root, row.corpus_id)
    was_resumable = os.path.lexists(partial) and os.path.lexists(checkpoint)
    if os.path.lexists(store_root):
        status: Literal["built", "resumed", "verified-existing"] = "verified-existing"
    else:
        status = "resumed" if was_resumable else "built"
        current_model, stale_model = _model_specs(config)
        try:
            build_embedding_store(
                _selection(config, row),
                store_root,
                current_model=current_model,
                old_model=stale_model,
                paired_encoder=adapter,
                config=config.embedding_config,
            )
        except (EmbeddingStoreError, QwenRevisionEncoderError) as exc:
            raise ProductionEmbeddingBuildError(
                f"{row.corpus_id} embedding build failed: {exc}"
            ) from exc
    receipt, expected, tree_sha256 = _verify_one_store(config, row, adapter=adapter)
    evidence = _record_or_verify_evidence(
        config=config,
        row=row,
        receipt=receipt,
        expected=expected,
        tree_sha256=tree_sha256,
        status=status,
        started_at_utc=started_at,
        started_ns=started_ns,
    )
    _validate_output_membership(
        config.output_root,
        final=False,
        active_corpus=row.corpus_id,
    )
    return row, receipt, tree_sha256, evidence


def _publish_suite(
    config: ProductionEmbeddingConfig,
    rows: Sequence[_CompletedCorpusRow],
) -> ProductionEmbeddingSuiteReceipt:
    expected_suite = _suite_from_rows(config, rows)
    suite_path = config.output_root / PRODUCTION_EMBEDDING_SUITE_FILENAME
    if os.path.lexists(suite_path):
        observed_suite = _load_suite(suite_path)
        if observed_suite != expected_suite:
            raise ProductionEmbeddingBuildError(
                "existing suite receipt differs from reproduced rows"
            )
    else:
        try:
            write_exclusive_receipt_bytes(expected_suite.canonical_file_bytes(), suite_path)
        except ArtifactIntegrityError as exc:
            raise ProductionEmbeddingBuildError(f"cannot write suite receipt: {exc}") from exc
    _validate_output_membership(config.output_root, final=True)
    return expected_suite


def build_production_embedding_shard(
    config: ProductionEmbeddingConfig,
    corpus_id: str,
    *,
    paired_encoder: QwenPairedRevisionEmbeddingAdapter | None = None,
) -> ProductionCorpusBuildEvidence:
    """Build exactly one fixed corpus with both registered Qwen revisions.

    The corpus selector chooses scheduling only.  All source paths, prompts,
    model revisions, model-tree digests, ordering, dtype, and encoder settings
    come from the one hash-pinned production config.  Independent workers may
    invoke this function for distinct corpus IDs against a shared output root.
    """

    if not isinstance(corpus_id, str) or corpus_id not in FIXED_CORPORA:
        raise ProductionEmbeddingBuildError("shard corpus must be one of FIXED_CORPORA")
    with _corpus_worker_lock(config.output_root, corpus_id):
        _admit_build_inputs(
            config,
            prepare_output=True,
            active_corpus=corpus_id,
        )
        if os.path.lexists(config.output_root / PRODUCTION_EMBEDDING_SUITE_FILENAME):
            raise ProductionEmbeddingBuildError(
                "suite receipt is already published; shard construction is closed"
            )
        row = config.corpora[FIXED_CORPORA.index(corpus_id)]
        completed = _build_one_corpus(
            config,
            row,
            adapter=_paired_adapter(config, paired_encoder),
        )
        return completed[3]


def _require_complete_shard_membership(output_root: Path) -> None:
    _validate_output_membership(output_root, final=False)
    expected_stores = {output_root / corpus for corpus in FIXED_CORPORA}
    missing_stores = sorted(path.name for path in expected_stores if not os.path.lexists(path))
    unfinished = sorted(
        corpus
        for corpus in FIXED_CORPORA
        if any(os.path.lexists(path) for path in _partial_paths(output_root, corpus))
    )
    evidence_root = output_root / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY
    expected_evidence = {f"{corpus}.json" for corpus in FIXED_CORPORA}
    observed_evidence = _member_names(evidence_root)
    if missing_stores or unfinished or observed_evidence != expected_evidence:
        raise ProductionEmbeddingBuildError(
            "completed shard membership differs; "
            f"missing_stores={missing_stores}, unfinished={unfinished}, "
            f"missing_evidence={sorted(expected_evidence - observed_evidence)}, "
            f"unexpected_evidence={sorted(observed_evidence - expected_evidence)}"
        )


def aggregate_production_embedding_shards(
    config: ProductionEmbeddingConfig,
) -> ProductionEmbeddingSuiteReceipt:
    """Reverify exactly five completed shards and publish the canonical suite."""

    with _corpus_worker_locks(config.output_root, FIXED_CORPORA):
        _admit_build_inputs(config, prepare_output=False)
        _require_real_directory(config.output_root, label="embedding output root")
        _require_complete_shard_membership(config.output_root)
        adapter = _paired_adapter(config, None)
        rows: list[_CompletedCorpusRow] = []
        for row in config.corpora:
            receipt, expected, tree_sha256 = _verify_one_store(config, row, adapter=adapter)
            evidence = _load_evidence(
                config.output_root
                / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY
                / f"{row.corpus_id}.json"
            )
            _evidence_matches(
                evidence,
                corpus_id=row.corpus_id,
                config=config,
                expected=expected,
                receipt=receipt,
                tree_sha256=tree_sha256,
            )
            rows.append((row, receipt, tree_sha256, evidence))
        return _publish_suite(config, rows)


def build_production_embedding_suite(
    config: ProductionEmbeddingConfig,
    *,
    paired_encoder: QwenPairedRevisionEmbeddingAdapter | None = None,
) -> ProductionEmbeddingSuiteReceipt:
    """Build or resume every corpus, then publish one immutable suite receipt."""

    with _corpus_worker_locks(config.output_root, FIXED_CORPORA):
        _admit_build_inputs(config, prepare_output=True)
        adapter = _paired_adapter(config, paired_encoder)
        completed: list[_CompletedCorpusRow] = []
        for row in config.corpora:
            completed.append(_build_one_corpus(config, row, adapter=adapter))
        return _publish_suite(config, completed)


def verify_production_embedding_suite(
    config: ProductionEmbeddingConfig,
) -> ProductionEmbeddingSuiteReceipt:
    """Rehash all five stores and reproduce the terminal receipt without writes."""

    if not isinstance(config, ProductionEmbeddingConfig):
        raise ProductionEmbeddingBuildError("config must be ProductionEmbeddingConfig")
    with _corpus_worker_locks(config.output_root, FIXED_CORPORA):
        _admit_build_inputs(config, prepare_output=False)
        _validate_output_membership(config.output_root, final=True)
        adapter = QwenPairedRevisionEmbeddingAdapter(
            config.current_encoder_config,
            config.stale_encoder_config,
        )
        rows: list[
            tuple[
                ProductionCorpusSources,
                EmbeddingStoreReceipt,
                str,
                ProductionCorpusBuildEvidence,
            ]
        ] = []
        for row in config.corpora:
            receipt, expected, tree_sha256 = _verify_one_store(config, row, adapter=adapter)
            evidence = _load_evidence(
                config.output_root
                / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY
                / f"{row.corpus_id}.json"
            )
            _evidence_matches(
                evidence,
                corpus_id=row.corpus_id,
                config=config,
                expected=expected,
                receipt=receipt,
                tree_sha256=tree_sha256,
            )
            rows.append((row, receipt, tree_sha256, evidence))
        reproduced = _suite_from_rows(config, rows)
        observed = _load_suite(config.output_root / PRODUCTION_EMBEDDING_SUITE_FILENAME)
        if observed != reproduced:
            raise ProductionEmbeddingBuildError("suite receipt differs from reproduced stores")
        return observed


def production_embedding_status(config: ProductionEmbeddingConfig) -> dict[str, object]:
    """Return filesystem-only recovery state without asserting artifact validity."""

    if not isinstance(config, ProductionEmbeddingConfig):
        raise ProductionEmbeddingBuildError("config must be ProductionEmbeddingConfig")
    if not os.path.lexists(config.output_root):
        return {
            "claim": "diagnostic-only",
            "corpora": [{"corpus_id": corpus, "status": "pending"} for corpus in FIXED_CORPORA],
            "status": "pending",
        }
    _validate_output_membership(config.output_root, final=False)
    corpus_rows: list[dict[str, str]] = []
    for corpus in FIXED_CORPORA:
        store = config.output_root / corpus
        partial, checkpoint = _partial_paths(config.output_root, corpus)
        evidence = config.output_root / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY / f"{corpus}.json"
        if os.path.lexists(store):
            state = "recorded" if os.path.lexists(evidence) else "store-unrecorded"
        elif os.path.lexists(partial) and os.path.lexists(checkpoint):
            state = "resumable"
        else:
            state = "pending"
        corpus_rows.append({"corpus_id": corpus, "status": state})
    suite_exists = os.path.lexists(config.output_root / PRODUCTION_EMBEDDING_SUITE_FILENAME)
    if suite_exists and all(row["status"] == "recorded" for row in corpus_rows):
        status = "complete"
    elif suite_exists:
        raise ProductionEmbeddingBuildError("suite receipt exists before all corpus evidence")
    elif any(row["status"] == "resumable" for row in corpus_rows):
        status = "resumable"
    else:
        status = "incomplete"
    return {"claim": "diagnostic-only", "corpora": corpus_rows, "status": status}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-production-embeddings",
        description="Build or verify the fixed five-corpus paired-Qwen embedding suite.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    builder_parser = subparsers.add_parser(
        "write-builder-receipt",
        help="observe the clean, pinned macOS arm64 MPS builder and run its fixed probe",
    )
    builder_parser.add_argument("--repository-root", required=True, type=Path)
    builder_parser.add_argument("--expected-source-commit", required=True)
    builder_parser.add_argument("--uv-lock", required=True, type=Path)
    builder_parser.add_argument("--current-model-root", required=True, type=Path)
    builder_parser.add_argument("--stale-model-root", required=True, type=Path)
    builder_parser.add_argument("--batch-size", required=True, type=int)
    builder_parser.add_argument("--seed", required=True, type=int)
    builder_parser.add_argument("--output", required=True, type=Path)
    config_parser = subparsers.add_parser(
        "write-config", help="derive all source allowlists and write one closed config"
    )
    config_parser.add_argument("--online-staging-root", required=True, type=Path)
    config_parser.add_argument("--expected-inventory-sha256", required=True)
    config_parser.add_argument("--builder-receipt", required=True, type=Path)
    config_parser.add_argument("--builder-receipt-sha256", required=True)
    config_parser.add_argument("--current-model-root", required=True, type=Path)
    config_parser.add_argument("--stale-model-root", required=True, type=Path)
    config_parser.add_argument("--output-root", required=True, type=Path)
    config_parser.add_argument("--batch-size", required=True, type=int)
    config_parser.add_argument("--device", required=True)
    config_parser.add_argument("--seed", required=True, type=int)
    config_parser.add_argument("--output-dtype", choices=("float32",), default="float32")
    config_parser.add_argument("--output", required=True, type=Path)
    for command in ("build", "build-shard", "aggregate", "verify", "status"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", required=True, type=Path)
        command_parser.add_argument("--config-sha256", required=True)
        if command == "build-shard":
            command_parser.add_argument(
                "--corpus-id",
                choices=FIXED_CORPORA,
                required=True,
                help="scheduling selector; all scientific inputs remain config-derived",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "write-builder-receipt":
            builder_receipt = write_production_embedding_builder_receipt(
                repository_root=arguments.repository_root,
                expected_source_commit=arguments.expected_source_commit,
                uv_lock_path=arguments.uv_lock,
                current_model_root=arguments.current_model_root,
                stale_model_root=arguments.stale_model_root,
                batch_size=arguments.batch_size,
                deterministic_seed=arguments.seed,
                destination=arguments.output,
            )
            result: object = {
                "builder_receipt_sha256": builder_receipt.file_sha256,
                "output": str(arguments.output),
                "status": "builder-verified",
            }
        elif arguments.command == "write-config":
            config = write_production_embedding_config(
                online_staging_root=arguments.online_staging_root,
                expected_inventory_sha256=arguments.expected_inventory_sha256,
                builder_receipt_path=arguments.builder_receipt,
                expected_builder_receipt_sha256=arguments.builder_receipt_sha256,
                current_model_root=arguments.current_model_root,
                stale_model_root=arguments.stale_model_root,
                output_root=arguments.output_root,
                batch_size=arguments.batch_size,
                device=arguments.device,
                deterministic_seed=arguments.seed,
                output_dtype=arguments.output_dtype,
                destination=arguments.output,
            )
            result = {
                "config_sha256": config.file_sha256,
                "corpora": list(FIXED_CORPORA),
                "output": str(arguments.output),
            }
        else:
            config = load_production_embedding_config(
                arguments.config,
                expected_sha256=arguments.config_sha256,
            )
            if arguments.command == "build":
                receipt = build_production_embedding_suite(config)
                result = {
                    "output_root": str(config.output_root),
                    "receipt_sha256": receipt.receipt_sha256,
                    "status": "complete",
                }
            elif arguments.command == "build-shard":
                evidence = build_production_embedding_shard(config, arguments.corpus_id)
                result = {
                    "corpus_id": evidence.corpus_id,
                    "embedding_receipt_sha256": evidence.embedding_receipt_sha256,
                    "embedding_tree_sha256": evidence.embedding_tree_sha256,
                    "output_root": str(config.output_root),
                    "shard_receipt_sha256": evidence.shard_receipt_sha256,
                    "status": "shard-complete",
                }
            elif arguments.command == "aggregate":
                receipt = aggregate_production_embedding_shards(config)
                result = {
                    "output_root": str(config.output_root),
                    "receipt_sha256": receipt.receipt_sha256,
                    "status": "complete",
                }
            elif arguments.command == "verify":
                receipt = verify_production_embedding_suite(config)
                result = {
                    "output_root": str(config.output_root),
                    "receipt_sha256": receipt.receipt_sha256,
                    "status": "verified",
                }
            else:
                result = production_embedding_status(config)
    except (
        ProductionEmbeddingBuildError,
        EmbeddingStoreError,
        QwenRevisionEncoderError,
        StudyDataError,
    ) as exc:
        parser.exit(2, f"production-embeddings: {exc}\n")
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
