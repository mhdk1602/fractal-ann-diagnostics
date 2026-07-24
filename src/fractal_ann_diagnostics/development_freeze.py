"""Development-only compiler for the pre-sealed confirmatory freeze.

The compiler admits labels only from named development-fit and
development-calibration sources.  It rejects every source path before opening
any file when a sealed, custody, holdout, or reserve role is present.  Its
outputs are canonical, digest-bound, and published by an operating-system
no-replace rename.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np

from .artifact_integrity import read_secure_regular_file
from .confirmatory_modeling import (
    REGISTERED_FEATURE_SCHEMA,
    FeatureBatch,
    FrozenModelSuite,
    LabeledFeatureBatch,
    canonical_h1_model_artifact_bytes,
    canonical_h2_model_suite_artifact_bytes,
    fit_frozen_model_suite,
)
from .controller import ControllerConfig, RuleController
from .development_cohort import (
    DevelopmentCohortSelectionReceipt,
    load_development_cohort_selection,
)
from .embedding_store import EmbeddingStoreReceipt, verify_embedding_store
from .geometry import QueryGeometry
from .joint_power_design import (
    EVIDENCE_CORPORA,
    FIXED_CORPORA,
    DependenceSource,
    DevelopmentFamilyRow,
    DevelopmentScenarioPanel,
    EffectScenario,
    GeometryGainThresholds,
    JointPowerDesignConfig,
    canonical_development_panel_bytes,
    canonical_joint_power_config_bytes,
    load_development_panel,
    load_joint_power_config,
)
from .policy_intervention import loads_canonical_trial_schedule

DevelopmentStage = Literal["development-fit", "development-calibration"]
SourceRole = Literal[
    "queries",
    "qrels",
    "evidence-bundles",
    "policy-schedule",
    "paired-actions",
]

DEVELOPMENT_FREEZE_SCHEMA = "fractal-development-freeze-receipt-v2"
DEVELOPMENT_FREEZE_CONFIG_SCHEMA = "fractal-development-freeze-config-v2"
DEVELOPMENT_FREEZE_CLI_RESULT_SCHEMA = "fractal-development-freeze-cli-result-v1"
FEATURE_ARTIFACT_SCHEMA = "fractal-development-feature-batch-v1"
OUTCOME_ARTIFACT_SCHEMA = "fractal-development-action-outcomes-v2"
PAIRED_ACTION_ROW_SCHEMA = "fractal-development-paired-action-row-v2"
CONTROLLER_ARTIFACT_SCHEMA = "fractal-frozen-controller-selection-v1"
COMPARATOR_ARTIFACT_SCHEMA = "fractal-frozen-static-comparator-v1"
GEOMETRY_PROFILE_SCHEMA = "fractal-frozen-geometry-profiles-v1"
ATTENUATION_ARTIFACT_SCHEMA = "fractal-conservative-scenario-attenuation-v1"

REGISTERED_ACTIONS = ("hnsw-low", "hnsw-high", "exact-authorized", "abstain")
REGISTERED_ALLOW_RATES = (0.25, 0.50, 0.75)
REGISTERED_NESTED_ROWS = 3
REGISTERED_MODEL_SEED = 20260713
REGISTERED_K = 10
REGISTERED_FAILURE_RECALL = 0.90
CONTROLLER_RETRIEVAL_LOSS_LIMIT = 0.005
CONTROLLER_EVIDENCE_LOSS_LIMIT = 0.005
CONTROLLER_P95_RATIO_LIMIT = 1.20
GEOMETRY_GAIN_THRESHOLDS = GeometryGainThresholds(
    log_loss_reduction=0.002,
    brier_score_reduction=0.001,
    auprc_gain=0.005,
)

CONSERVATIVE_LOGIT_RETENTION = 0.75
CONSERVATIVE_LATENCY_BENEFIT_RETENTION = 0.75
CONSERVATIVE_LATENCY_PENALTY_MULTIPLIER = 1.25
CONSERVATIVE_FIDELITY_DROP_RATE = 0.10

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PATH_TOKENS = frozenset(
    {"sealed", "custody", "holdout", "heldout", "reserve", "reserved"}
)
_PLACEHOLDER_PATH_TOKENS = frozenset(
    {"changeme", "latest", "placeholder", "replace", "tbd", "todo", "unassigned", "unset"}
)
_PLACEHOLDER_PATH_CHARACTERS = frozenset("<>{}$*?")
_MAX_CONFIG_BYTES = 8 * 1024 * 1024
_QUERY_FIELDS = frozenset({"id", "text"})
_QREL_FIELDS = frozenset({"document_id", "query_id", "relevance"})
_EVIDENCE_FIELDS = frozenset({"answer", "evidence_bundles", "label_metadata", "query_id"})
_BUNDLE_FIELDS = frozenset({"bundle_id", "locations"})
_LOCATION_FIELDS = frozenset({"document_id", "locator"})
_ROW_ORDER_FIELDS = frozenset({"dataset", "id", "kind", "source_path", "source_row", "stage"})
_PAIRED_FIELDS = frozenset(
    {
        "action",
        "entitlement_violations",
        "execution_position",
        "execution_state",
        "failure_state",
        "family_id",
        "feature_values",
        "query_id",
        "request_latency_ms",
        "returned_document_rows",
        "schedule_order",
        "schema_version",
        "trial_key",
    }
)


class DevelopmentFreezeError(ValueError):
    """Raised when development evidence cannot enter an immutable freeze."""


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DevelopmentFreezeError("freeze artifacts require finite canonical JSON") from exc
    return encoded + (b"\n" if newline else b"")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DevelopmentFreezeError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise DevelopmentFreezeError(f"{name} must be a canonical non-empty string")
    return value


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise DevelopmentFreezeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DevelopmentFreezeError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise DevelopmentFreezeError(f"{name} must be at least {minimum}")
    return number


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DevelopmentFreezeError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise DevelopmentFreezeError(
            f"{label} keys differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _decode_json(encoded: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DevelopmentFreezeError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise DevelopmentFreezeError(f"{label} contains non-finite value {value!r}")

    try:
        return json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise DevelopmentFreezeError(f"{label} must be valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise DevelopmentFreezeError(f"{label} must be valid JSON: {exc.msg}") from exc


def _decode_jsonl(encoded: bytes, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not encoded or not encoded.endswith(b"\n"):
        raise DevelopmentFreezeError(f"{label} must be non-empty canonical JSONL")
    rows: list[Mapping[str, Any]] = []
    for position, line in enumerate(encoded.splitlines(keepends=True), start=1):
        value = _decode_json(line, label=f"{label} line {position}")
        if not isinstance(value, Mapping) or line != _canonical_bytes(value):
            raise DevelopmentFreezeError(f"{label} line {position} is not canonical JSON")
        rows.append(value)
    return tuple(rows)


def _path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in path.parts:
        tokens.update(token for token in re.split(r"[^a-z0-9]+", part.casefold()) if token)
    return tokens


@dataclass(frozen=True)
class PinnedDevelopmentFile:
    """One exact development file admitted by role, stage, size, and digest."""

    path: Path
    sha256: str
    byte_count: int
    corpus_id: str
    stage: DevelopmentStage | str
    role: SourceRole | str

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.is_absolute():
            raise DevelopmentFreezeError("development source paths must be absolute")
        object.__setattr__(self, "path", path)
        _require_sha256("source sha256", self.sha256)
        if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int):
            raise DevelopmentFreezeError("source byte_count must be an integer")
        if self.byte_count <= 0:
            raise DevelopmentFreezeError("source byte_count must be positive")
        if self.corpus_id not in FIXED_CORPORA:
            raise DevelopmentFreezeError("source corpus is outside the fixed five-corpus suite")
        if self.stage not in {"development-fit", "development-calibration"}:
            raise DevelopmentFreezeError("source stage must be development-fit or calibration")
        if self.role not in {
            "queries",
            "qrels",
            "evidence-bundles",
            "policy-schedule",
            "paired-actions",
        }:
            raise DevelopmentFreezeError("source role is not admitted by the compiler")

    @property
    def artifact_id(self) -> str:
        return f"{self.stage}:{self.corpus_id}:{self.role}"


@dataclass(frozen=True)
class PinnedEmbeddingStore:
    """One verified dual-epoch embedding directory for a development partition."""

    root: Path
    receipt_sha256: str
    corpus_id: str
    stage: DevelopmentStage | str

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute():
            raise DevelopmentFreezeError("embedding store roots must be absolute")
        object.__setattr__(self, "root", root)
        _require_sha256("embedding receipt_sha256", self.receipt_sha256)
        if self.corpus_id not in FIXED_CORPORA:
            raise DevelopmentFreezeError("embedding corpus is outside the fixed suite")
        if self.stage not in {"development-fit", "development-calibration"}:
            raise DevelopmentFreezeError("embedding stage is not a development stage")

    @property
    def artifact_id(self) -> str:
        return f"{self.stage}:{self.corpus_id}:embedding-store"


@dataclass(frozen=True)
class PinnedDevelopmentSelectionReceipt:
    """Direct exact pin to the pre-label representative selection."""

    path: Path
    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.is_absolute():
            raise DevelopmentFreezeError("selection receipt path must be absolute")
        object.__setattr__(self, "path", path)
        _require_sha256("selection receipt sha256", self.sha256)
        if (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count <= 0
        ):
            raise DevelopmentFreezeError("selection receipt byte_count must be positive")

    @property
    def artifact_id(self) -> str:
        return "development-cohort-selection"


@dataclass(frozen=True)
class DevelopmentCorpusSources:
    """All source paths needed to reconstruct one corpus-stage outcome panel."""

    corpus_id: str
    stage: DevelopmentStage | str
    queries: PinnedDevelopmentFile
    qrels: PinnedDevelopmentFile
    evidence_bundles: PinnedDevelopmentFile | None
    policy_schedule: PinnedDevelopmentFile
    paired_actions: PinnedDevelopmentFile
    embedding_store: PinnedEmbeddingStore

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise DevelopmentFreezeError("corpus source is outside the fixed suite")
        if self.stage not in {"development-fit", "development-calibration"}:
            raise DevelopmentFreezeError("corpus source stage is not a development stage")
        expected = {
            "queries": self.queries,
            "qrels": self.qrels,
            "policy-schedule": self.policy_schedule,
            "paired-actions": self.paired_actions,
        }
        if self.evidence_bundles is not None:
            expected["evidence-bundles"] = self.evidence_bundles
        for role, pin in expected.items():
            if (pin.corpus_id, pin.stage, pin.role) != (self.corpus_id, self.stage, role):
                raise DevelopmentFreezeError("source pin role, stage, or corpus differs")
        if (self.embedding_store.corpus_id, self.embedding_store.stage) != (
            self.corpus_id,
            self.stage,
        ):
            raise DevelopmentFreezeError("embedding pin stage or corpus differs")
        needs_evidence = self.corpus_id in EVIDENCE_CORPORA
        if needs_evidence != (self.evidence_bundles is not None):
            raise DevelopmentFreezeError(
                "evidence-bundle paths are required exactly for the three evidence corpora"
            )

    @property
    def files(self) -> tuple[PinnedDevelopmentFile, ...]:
        values = [self.queries, self.qrels]
        if self.evidence_bundles is not None:
            values.append(self.evidence_bundles)
        values.extend((self.policy_schedule, self.paired_actions))
        return tuple(values)


@dataclass(frozen=True)
class DevelopmentFreezeConfig:
    """Closed compiler inputs; no sealed role exists in this type."""

    sources: tuple[DevelopmentCorpusSources, ...]
    selection_receipt: PinnedDevelopmentSelectionReceipt
    output_root: Path
    model_seed: int = REGISTERED_MODEL_SEED
    k: int = REGISTERED_K
    failure_recall_threshold: float = REGISTERED_FAILURE_RECALL

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        expected = {
            (stage, corpus)
            for stage in ("development-fit", "development-calibration")
            for corpus in FIXED_CORPORA
        }
        observed = {(source.stage, source.corpus_id) for source in sources}
        if observed != expected or len(sources) != len(expected):
            raise DevelopmentFreezeError(
                "sources must contain one fit and one calibration bundle for each fixed corpus"
            )
        ordered = tuple(sorted(sources, key=lambda item: (item.stage, item.corpus_id)))
        object.__setattr__(self, "sources", ordered)
        if not isinstance(self.selection_receipt, PinnedDevelopmentSelectionReceipt):
            raise DevelopmentFreezeError(
                "selection_receipt must be a direct typed development selection pin"
            )
        output = Path(self.output_root)
        if not output.is_absolute():
            raise DevelopmentFreezeError("output_root must be absolute")
        object.__setattr__(self, "output_root", output)
        if type(self.model_seed) is not int or self.model_seed != REGISTERED_MODEL_SEED:
            raise DevelopmentFreezeError(f"model_seed must equal {REGISTERED_MODEL_SEED}")
        if type(self.k) is not int or self.k != REGISTERED_K:
            raise DevelopmentFreezeError(f"k must equal {REGISTERED_K}")
        if (
            isinstance(self.failure_recall_threshold, bool)
            or not isinstance(self.failure_recall_threshold, (int, float))
            or self.failure_recall_threshold != REGISTERED_FAILURE_RECALL
        ):
            raise DevelopmentFreezeError(
                f"failure_recall_threshold must equal {REGISTERED_FAILURE_RECALL}"
            )


def _config_absolute_path(name: str, value: object) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(character in _PLACEHOLDER_PATH_CHARACTERS for character in value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise DevelopmentFreezeError(f"{name} must be a canonical absolute POSIX path")
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or str(pure) != value
        or pure == PurePosixPath("/")
        or any(part in {"", ".", ".."} for part in pure.parts[1:])
    ):
        raise DevelopmentFreezeError(f"{name} must be a canonical absolute POSIX path")
    path = Path(value)
    tokens = _path_tokens(path)
    forbidden = tokens & _FORBIDDEN_PATH_TOKENS
    if forbidden:
        raise DevelopmentFreezeError(
            f"{name} contains forbidden sealed-role token(s) {sorted(forbidden)}"
        )
    placeholders = tokens & _PLACEHOLDER_PATH_TOKENS
    if placeholders:
        raise DevelopmentFreezeError(f"{name} contains placeholder token(s) {sorted(placeholders)}")
    return path


def _config_file_pin(pin: PinnedDevelopmentFile) -> dict[str, object]:
    _config_absolute_path(f"{pin.artifact_id}.path", str(pin.path))
    return {
        "byte_count": pin.byte_count,
        "path": str(pin.path),
        "sha256": pin.sha256,
    }


def canonical_development_freeze_config_bytes(config: DevelopmentFreezeConfig) -> bytes:
    """Serialize one complete compiler config as canonical JSON with one final LF."""

    if not isinstance(config, DevelopmentFreezeConfig):
        raise TypeError("config must be DevelopmentFreezeConfig")
    _config_absolute_path("output_root", str(config.output_root))
    _config_absolute_path("selection_receipt.path", str(config.selection_receipt.path))
    sources: list[dict[str, object]] = []
    for source in config.sources:
        _config_absolute_path(
            f"{source.embedding_store.artifact_id}.root",
            str(source.embedding_store.root),
        )
        sources.append(
            {
                "corpus_id": source.corpus_id,
                "embedding_store": {
                    "receipt_sha256": source.embedding_store.receipt_sha256,
                    "root": str(source.embedding_store.root),
                },
                "evidence_bundles": (
                    None
                    if source.evidence_bundles is None
                    else _config_file_pin(source.evidence_bundles)
                ),
                "paired_actions": _config_file_pin(source.paired_actions),
                "policy_schedule": _config_file_pin(source.policy_schedule),
                "qrels": _config_file_pin(source.qrels),
                "queries": _config_file_pin(source.queries),
                "stage": source.stage,
            }
        )
    return _canonical_bytes(
        {
            "failure_recall_threshold": config.failure_recall_threshold,
            "k": config.k,
            "model_seed": config.model_seed,
            "output_root": str(config.output_root),
            "schema_version": DEVELOPMENT_FREEZE_CONFIG_SCHEMA,
            "selection_receipt": {
                "byte_count": config.selection_receipt.byte_count,
                "path": str(config.selection_receipt.path),
                "sha256": config.selection_receipt.sha256,
            },
            "sources": sources,
        }
    )


def _development_file_from_config(
    value: object,
    *,
    corpus_id: str,
    stage: str,
    role: str,
) -> PinnedDevelopmentFile:
    row = _closed(
        value,
        frozenset({"byte_count", "path", "sha256"}),
        label=f"{stage}:{corpus_id}:{role} config pin",
    )
    path = _config_absolute_path(f"{stage}:{corpus_id}:{role}.path", row["path"])
    _require_sha256(f"{stage}:{corpus_id}:{role}.sha256", row["sha256"])
    byte_count = row["byte_count"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or not 0 < byte_count <= 256 * 1024 * 1024 * 1024
    ):
        raise DevelopmentFreezeError(
            f"{stage}:{corpus_id}:{role}.byte_count must be a positive bounded integer"
        )
    return PinnedDevelopmentFile(
        path=path,
        sha256=row["sha256"],
        byte_count=byte_count,
        corpus_id=corpus_id,
        stage=stage,
        role=role,
    )


def load_development_freeze_config(path: str | Path) -> DevelopmentFreezeConfig:
    """Load one exact canonical compiler config without opening a source artifact."""

    config_path = _config_absolute_path("config path", str(path))
    encoded = read_secure_regular_file(
        config_path,
        max_bytes=_MAX_CONFIG_BYTES,
        label="development freeze config",
    )
    value = _decode_json(encoded, label="development freeze config")
    root = _closed(
        value,
        frozenset(
            {
                "failure_recall_threshold",
                "k",
                "model_seed",
                "output_root",
                "schema_version",
                "selection_receipt",
                "sources",
            }
        ),
        label="development freeze config",
    )
    if root["schema_version"] != DEVELOPMENT_FREEZE_CONFIG_SCHEMA:
        raise DevelopmentFreezeError(
            f"config schema_version must equal {DEVELOPMENT_FREEZE_CONFIG_SCHEMA!r}"
        )
    selection_pin = _closed(
        root["selection_receipt"],
        frozenset({"byte_count", "path", "sha256"}),
        label="development selection receipt config pin",
    )
    raw_sources = root["sources"]
    if not isinstance(raw_sources, list):
        raise DevelopmentFreezeError("config sources must be an array")
    source_fields = frozenset(
        {
            "corpus_id",
            "embedding_store",
            "evidence_bundles",
            "paired_actions",
            "policy_schedule",
            "qrels",
            "queries",
            "stage",
        }
    )
    sources: list[DevelopmentCorpusSources] = []
    for position, raw_source in enumerate(raw_sources):
        source = _closed(
            raw_source,
            source_fields,
            label=f"development freeze source {position}",
        )
        corpus_id = source["corpus_id"]
        stage = source["stage"]
        if corpus_id not in FIXED_CORPORA:
            raise DevelopmentFreezeError("config source corpus is outside the fixed suite")
        if stage not in {"development-fit", "development-calibration"}:
            raise DevelopmentFreezeError("config source stage is not a development stage")
        embedding = _closed(
            source["embedding_store"],
            frozenset({"receipt_sha256", "root"}),
            label=f"{stage}:{corpus_id}:embedding-store config pin",
        )
        embedding_root = _config_absolute_path(
            f"{stage}:{corpus_id}:embedding-store.root",
            embedding["root"],
        )
        embedding_sha256 = _require_sha256(
            f"{stage}:{corpus_id}:embedding-store.receipt_sha256",
            embedding["receipt_sha256"],
        )
        raw_evidence = source["evidence_bundles"]
        evidence = (
            None
            if raw_evidence is None
            else _development_file_from_config(
                raw_evidence,
                corpus_id=corpus_id,
                stage=stage,
                role="evidence-bundles",
            )
        )
        sources.append(
            DevelopmentCorpusSources(
                corpus_id=corpus_id,
                stage=stage,
                queries=_development_file_from_config(
                    source["queries"],
                    corpus_id=corpus_id,
                    stage=stage,
                    role="queries",
                ),
                qrels=_development_file_from_config(
                    source["qrels"],
                    corpus_id=corpus_id,
                    stage=stage,
                    role="qrels",
                ),
                evidence_bundles=evidence,
                policy_schedule=_development_file_from_config(
                    source["policy_schedule"],
                    corpus_id=corpus_id,
                    stage=stage,
                    role="policy-schedule",
                ),
                paired_actions=_development_file_from_config(
                    source["paired_actions"],
                    corpus_id=corpus_id,
                    stage=stage,
                    role="paired-actions",
                ),
                embedding_store=PinnedEmbeddingStore(
                    root=embedding_root,
                    receipt_sha256=embedding_sha256,
                    corpus_id=corpus_id,
                    stage=stage,
                ),
            )
        )
    config = DevelopmentFreezeConfig(
        sources=tuple(sources),
        selection_receipt=PinnedDevelopmentSelectionReceipt(
            path=_config_absolute_path("selection_receipt.path", selection_pin["path"]),
            sha256=_require_sha256("selection_receipt.sha256", selection_pin["sha256"]),
            byte_count=selection_pin["byte_count"],
        ),
        output_root=_config_absolute_path("output_root", root["output_root"]),
        model_seed=root["model_seed"],
        k=root["k"],
        failure_recall_threshold=root["failure_recall_threshold"],
    )
    if encoded != canonical_development_freeze_config_bytes(config):
        raise DevelopmentFreezeError("development freeze config is not canonical JSON")
    return config


@dataclass(frozen=True)
class ActionOutcome:
    action: str
    execution_position: int
    execution_state: str
    failure_state: str | None
    request_latency_ms: float
    retrieval_attained: bool
    qrel_recall_at_k: float
    evidence_sufficient: bool | None
    entitlement_violations: int
    returned_document_rows: tuple[int, ...] = ()
    returned_document_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in REGISTERED_ACTIONS:
            raise DevelopmentFreezeError("action outcome is outside the registered action set")
        if (
            isinstance(self.execution_position, bool)
            or not isinstance(self.execution_position, int)
            or not 0 <= self.execution_position < len(REGISTERED_ACTIONS)
        ):
            raise DevelopmentFreezeError("action execution_position must be from zero to three")
        if self.execution_state not in {"completed", "failed", "abstained"}:
            raise DevelopmentFreezeError("action outcome execution_state is not registered")
        if self.execution_state == "completed" and self.failure_state is not None:
            raise DevelopmentFreezeError("completed action outcome cannot name a failure")
        if self.execution_state != "completed":
            _require_identifier("failure_state", self.failure_state)
        latency = _finite("request_latency_ms", self.request_latency_ms, minimum=0.0)
        if latency == 0.0:
            raise DevelopmentFreezeError("request_latency_ms must be positive")
        object.__setattr__(self, "request_latency_ms", latency)
        if type(self.retrieval_attained) is not bool:
            raise DevelopmentFreezeError("retrieval_attained must be boolean")
        recall = _finite("qrel_recall_at_k", self.qrel_recall_at_k)
        if not 0.0 <= recall <= 1.0:
            raise DevelopmentFreezeError("qrel_recall_at_k must be in [0, 1]")
        object.__setattr__(self, "qrel_recall_at_k", recall)
        if self.evidence_sufficient is not None and type(self.evidence_sufficient) is not bool:
            raise DevelopmentFreezeError("evidence_sufficient must be boolean or null")
        if (
            isinstance(self.entitlement_violations, bool)
            or not isinstance(self.entitlement_violations, int)
            or self.entitlement_violations < 0
        ):
            raise DevelopmentFreezeError("entitlement_violations must be non-negative")
        rows = tuple(self.returned_document_rows)
        identifiers = tuple(self.returned_document_ids)
        if any(type(value) is not int or value < 0 for value in rows):
            raise DevelopmentFreezeError("returned_document_rows must contain non-negative ints")
        if len(rows) != len(set(rows)) or len(identifiers) != len(set(identifiers)):
            raise DevelopmentFreezeError("returned documents cannot repeat")
        for identifier in identifiers:
            _require_identifier("returned_document_id", identifier)
        if len(rows) != len(identifiers):
            raise DevelopmentFreezeError("returned document rows and IDs must have equal length")
        if self.execution_state != "completed" and rows:
            raise DevelopmentFreezeError("failed or abstained outcomes cannot return documents")
        object.__setattr__(self, "returned_document_rows", rows)
        object.__setattr__(self, "returned_document_ids", identifiers)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "entitlement_violations": self.entitlement_violations,
            "evidence_sufficient": self.evidence_sufficient,
            "execution_position": self.execution_position,
            "execution_state": self.execution_state,
            "failure_state": self.failure_state,
            "qrel_recall_at_k": self.qrel_recall_at_k,
            "request_latency_ms": self.request_latency_ms,
            "retrieval_attained": self.retrieval_attained,
            "returned_document_ids": list(self.returned_document_ids),
            "returned_document_rows": list(self.returned_document_rows),
        }


@dataclass(frozen=True)
class DevelopmentTrial:
    partition: DevelopmentStage | str
    corpus_id: str
    family_id: str
    query_id: str
    trial_key: str
    subject: str
    repetition: int
    target_allow_rate: float
    realized_allow_rate: float
    authorized_count: int
    feature_values: tuple[object, ...]
    label: int
    outcomes: tuple[ActionOutcome, ...]

    def __post_init__(self) -> None:
        if self.partition not in {"development-fit", "development-calibration"}:
            raise DevelopmentFreezeError("trial partition is not a development partition")
        if self.corpus_id not in FIXED_CORPORA:
            raise DevelopmentFreezeError("trial corpus is outside the fixed suite")
        for name in ("family_id", "query_id", "subject"):
            _require_identifier(name, getattr(self, name))
        _require_sha256("trial_key", self.trial_key)
        if self.repetition != 0 or isinstance(self.repetition, bool):
            raise DevelopmentFreezeError("development trials require the sole repetition zero")
        if self.target_allow_rate not in REGISTERED_ALLOW_RATES:
            raise DevelopmentFreezeError("target_allow_rate is outside the registered strata")
        realized = _finite("realized_allow_rate", self.realized_allow_rate)
        if not 0.0 < realized < 1.0:
            raise DevelopmentFreezeError("realized_allow_rate must be in (0, 1)")
        if (
            isinstance(self.authorized_count, bool)
            or not isinstance(self.authorized_count, int)
            or self.authorized_count <= 0
        ):
            raise DevelopmentFreezeError("authorized_count must be positive")
        features = tuple(self.feature_values)
        if len(features) != len(REGISTERED_FEATURE_SCHEMA.input_features):
            raise DevelopmentFreezeError("trial feature vector width differs from the schema")
        object.__setattr__(self, "feature_values", features)
        if isinstance(self.label, bool) or self.label not in {0, 1}:
            raise DevelopmentFreezeError("trial label must be the integer zero or one")
        outcomes = tuple(self.outcomes)
        if tuple(item.action for item in outcomes) != REGISTERED_ACTIONS:
            raise DevelopmentFreezeError("trial outcomes must use canonical action order")
        if {item.execution_position for item in outcomes} != set(range(len(REGISTERED_ACTIONS))):
            raise DevelopmentFreezeError("trial outcomes must retain one exact execution position")
        evidence_expected = self.corpus_id in EVIDENCE_CORPORA
        if any(
            evidence_expected != (outcome.evidence_sufficient is not None) for outcome in outcomes
        ):
            raise DevelopmentFreezeError("trial evidence outcomes differ from corpus scope")
        object.__setattr__(self, "outcomes", outcomes)

    def outcome(self, action: str) -> ActionOutcome:
        return next(value for value in self.outcomes if value.action == action)


@dataclass(frozen=True)
class DevelopmentPartitionData:
    partition: DevelopmentStage | str
    trials: tuple[DevelopmentTrial, ...]

    def __post_init__(self) -> None:
        if self.partition not in {"development-fit", "development-calibration"}:
            raise DevelopmentFreezeError("partition data is not development-only")
        trials = tuple(
            sorted(self.trials, key=lambda row: (row.corpus_id, row.family_id, row.trial_key))
        )
        if not trials or any(trial.partition != self.partition for trial in trials):
            raise DevelopmentFreezeError("partition trials are empty or cross a stage boundary")
        if {trial.corpus_id for trial in trials} != set(FIXED_CORPORA):
            raise DevelopmentFreezeError("partition trials omit a fixed corpus")
        if len({trial.trial_key for trial in trials}) != len(trials):
            raise DevelopmentFreezeError("partition repeats a trial key")
        family_rows: dict[tuple[str, str], list[DevelopmentTrial]] = defaultdict(list)
        for trial in trials:
            family_rows[(trial.corpus_id, trial.family_id)].append(trial)
        for key, rows in family_rows.items():
            if len(rows) != REGISTERED_NESTED_ROWS:
                raise DevelopmentFreezeError(f"family {key!r} must contain exactly three rows")
            if {row.target_allow_rate for row in rows} != set(REGISTERED_ALLOW_RATES):
                raise DevelopmentFreezeError(f"family {key!r} does not span all allow rates")
            if len({row.subject for row in rows}) != 1 or {row.repetition for row in rows} != {0}:
                raise DevelopmentFreezeError(f"family {key!r} violates subject/repetition design")
            if len({row.query_id for row in rows}) != 1:
                raise DevelopmentFreezeError(f"family {key!r} crosses query identities")
        for corpus in FIXED_CORPORA:
            corpus_rows = [trial for trial in trials if trial.corpus_id == corpus]
            if len({trial.family_id for trial in corpus_rows}) < 2:
                raise DevelopmentFreezeError(
                    f"corpus {corpus!r} needs at least two development families"
                )
            if {trial.label for trial in corpus_rows} != {0, 1}:
                raise DevelopmentFreezeError(f"corpus {corpus!r} needs both binary outcome classes")
        object.__setattr__(self, "trials", trials)

    def labeled_batch(self) -> LabeledFeatureBatch:
        return LabeledFeatureBatch(
            partition=self.partition,
            feature_names=REGISTERED_FEATURE_SCHEMA.input_features,
            features=np.asarray([trial.feature_values for trial in self.trials], dtype=object),
            corpus_ids=tuple(trial.corpus_id for trial in self.trials),
            family_ids=tuple(trial.family_id for trial in self.trials),
            row_ids=tuple(trial.trial_key for trial in self.trials),
            labels=tuple(trial.label for trial in self.trials),
        )


def _controller_grid() -> tuple[ControllerConfig, ...]:
    values = tuple(
        ControllerConfig(
            low_ef=128,
            high_ef=512,
            probe_k=101,
            exact_scan_threshold=exact_scan,
            high_effort_threshold=high_threshold,
            exact_threshold=exact_threshold,
        )
        for exact_scan in (128, 256, 512)
        for high_threshold in (0.15, 0.20, 0.25, 0.30)
        for exact_threshold in (0.30, 0.35, 0.40, 0.45)
        if high_threshold < exact_threshold
    )
    return tuple(
        sorted(
            values,
            key=lambda value: (
                value.high_ef - value.low_ef,
                value.exact_scan_threshold,
                -value.high_effort_threshold,
                -value.exact_threshold,
                value.low_ef,
                value.high_ef,
            ),
        )
    )


REGISTERED_CONTROLLER_GRID = _controller_grid()


def _controller_config_dict(config: ControllerConfig) -> dict[str, object]:
    return {
        "exact_scan_threshold": config.exact_scan_threshold,
        "exact_threshold": config.exact_threshold,
        "high_ef": config.high_ef,
        "high_effort_threshold": config.high_effort_threshold,
        "low_ef": config.low_ef,
        "probe_k": config.probe_k,
    }


def _feature_mapping(trial: DevelopmentTrial) -> dict[str, object]:
    return dict(zip(REGISTERED_FEATURE_SCHEMA.input_features, trial.feature_values, strict=True))


def _geometry(trial: DevelopmentTrial) -> QueryGeometry:
    features = _feature_mapping(trial)
    return QueryGeometry(
        lid=float(features["lid_k50"]),
        lid_scale_instability=float(features["lid_cv"]),
        authorized_selectivity=trial.realized_allow_rate,
        relative_contrast=float(features["relative_contrast"]),
        radius_expansion=float(features["radius_expansion"]),
        policy_churn=float(features["policy_churn"]),
        embedding_drift=float(features["drift_severity"]),
        source="bounded-probe",
    )


def _family_equal_mean(values: Mapping[str, Sequence[float]]) -> float:
    return float(np.mean([float(np.mean(rows)) for rows in values.values()]))


def _evaluate_controller(
    config: ControllerConfig,
    calibration: DevelopmentPartitionData,
) -> dict[str, object]:
    chosen: list[tuple[DevelopmentTrial, ActionOutcome, ActionOutcome]] = []
    action_counts = {action: 0 for action in REGISTERED_ACTIONS}
    for trial in calibration.trials:
        decision = RuleController(config).decide(
            _geometry(trial),
            n_authorized=trial.authorized_count,
            policy_version="development-frozen-policy",
        )
        proposed = trial.outcome(decision.action)
        exact = trial.outcome("exact-authorized")
        action_counts[decision.action] += 1
        chosen.append((trial, proposed, exact))

    latency_by_corpus_family: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    retrieval_loss: dict[str, list[float]] = defaultdict(list)
    evidence_loss: dict[str, list[float]] = defaultdict(list)
    denied = 0
    for trial, proposed, exact in chosen:
        latency_by_corpus_family[trial.corpus_id][trial.family_id].append(
            (proposed.request_latency_ms, exact.request_latency_ms)
        )
        retrieval_loss[trial.corpus_id].append(
            float(exact.retrieval_attained) - float(proposed.retrieval_attained)
        )
        if trial.corpus_id in EVIDENCE_CORPORA:
            evidence_loss[trial.corpus_id].append(
                float(exact.evidence_sufficient is True)
                - float(proposed.evidence_sufficient is True)
            )
        denied += proposed.entitlement_violations

    corpus_reductions: list[float] = []
    corpus_p95_ratios: list[float] = []
    for corpus in FIXED_CORPORA:
        proposed_family = []
        exact_family = []
        for pairs in latency_by_corpus_family[corpus].values():
            proposed_family.append(float(np.mean([left for left, _ in pairs])))
            exact_family.append(float(np.mean([right for _, right in pairs])))
        corpus_reductions.append(
            float(
                np.mean([1.0 - left / right for left, right in zip(proposed_family, exact_family)])
            )
        )
        corpus_p95_ratios.append(
            float(
                np.quantile(proposed_family, 0.95, method="linear")
                / np.quantile(exact_family, 0.95, method="linear")
            )
        )

    retrieval = float(np.mean([np.mean(retrieval_loss[corpus]) for corpus in FIXED_CORPORA]))
    evidence = float(np.mean([np.mean(evidence_loss[corpus]) for corpus in EVIDENCE_CORPORA]))
    latency = float(np.mean(corpus_reductions))
    p95_ratio = float(np.mean(corpus_p95_ratios))
    passes = (
        retrieval <= CONTROLLER_RETRIEVAL_LOSS_LIMIT
        and evidence <= CONTROLLER_EVIDENCE_LOSS_LIMIT
        and denied == 0
        and p95_ratio < CONTROLLER_P95_RATIO_LIMIT
    )
    return {
        "action_counts": action_counts,
        "config": _controller_config_dict(config),
        "denied_emissions": denied,
        "equal_corpus_evidence_loss": evidence,
        "equal_corpus_family_latency_reduction": latency,
        "equal_corpus_p95_latency_ratio": p95_ratio,
        "equal_corpus_retrieval_loss": retrieval,
        "passes_constraints": passes,
    }


def _select_controller(
    calibration: DevelopmentPartitionData,
) -> tuple[ControllerConfig, dict[str, object]]:
    evaluations = tuple(
        _evaluate_controller(candidate, calibration) for candidate in REGISTERED_CONTROLLER_GRID
    )
    admissible = [row for row in evaluations if row["passes_constraints"] is True]
    if not admissible:
        raise DevelopmentFreezeError(
            "no registered controller candidate satisfies development gates"
        )
    grid_order = {
        _canonical_bytes(_controller_config_dict(config), newline=False): position
        for position, config in enumerate(REGISTERED_CONTROLLER_GRID)
    }
    selected_row = min(
        admissible,
        key=lambda row: (
            -round(float(row["equal_corpus_family_latency_reduction"]), 15),
            grid_order[_canonical_bytes(row["config"], newline=False)],
        ),
    )
    selected_config = next(
        config
        for config in REGISTERED_CONTROLLER_GRID
        if _controller_config_dict(config) == selected_row["config"]
    )
    artifact = {
        "candidate_evaluations": list(evaluations),
        "constraints": {
            "evidence_loss_maximum": CONTROLLER_EVIDENCE_LOSS_LIMIT,
            "maximum_denied_emissions": 0,
            "p95_latency_ratio_strict_upper_bound": CONTROLLER_P95_RATIO_LIMIT,
            "retrieval_loss_maximum": CONTROLLER_RETRIEVAL_LOSS_LIMIT,
        },
        "objective": "maximize-equal-corpus-family-latency-reduction",
        "objective_rounding_decimal_places": 15,
        "schema_version": CONTROLLER_ARTIFACT_SCHEMA,
        "selected_config": _controller_config_dict(selected_config),
        "selected_metrics": selected_row,
        "static_comparator": "hnsw-high",
        "tie_break": (
            "least-complex-registered-grid-order: narrower effort gap, smaller exact-scan "
            "threshold, then fewer threshold-triggered escalations"
        ),
    }
    return selected_config, artifact


def _feature_artifact(partition: DevelopmentPartitionData) -> dict[str, object]:
    return {
        "feature_names": list(REGISTERED_FEATURE_SCHEMA.input_features),
        "partition": partition.partition,
        "rows": [
            {
                "corpus_id": trial.corpus_id,
                "family_id": trial.family_id,
                "feature_values": [
                    None
                    if isinstance(value, (float, np.floating)) and math.isnan(float(value))
                    else value
                    for value in trial.feature_values
                ],
                "label": trial.label,
                "query_id": trial.query_id,
                "row_id": trial.trial_key,
            }
            for trial in partition.trials
        ],
        "schema_version": FEATURE_ARTIFACT_SCHEMA,
    }


def _outcome_artifact(partition: DevelopmentPartitionData) -> dict[str, object]:
    return {
        "action_order": list(REGISTERED_ACTIONS),
        "partition": partition.partition,
        "rows": [
            {
                "authorized_count": trial.authorized_count,
                "corpus_id": trial.corpus_id,
                "family_id": trial.family_id,
                "outcomes": [outcome.to_dict() for outcome in trial.outcomes],
                "query_id": trial.query_id,
                "realized_allow_rate": trial.realized_allow_rate,
                "repetition": trial.repetition,
                "subject": trial.subject,
                "target_allow_rate": trial.target_allow_rate,
                "trial_key": trial.trial_key,
            }
            for trial in partition.trials
        ],
        "schema_version": OUTCOME_ARTIFACT_SCHEMA,
    }


def _geometry_profiles(fit: DevelopmentPartitionData) -> dict[str, object]:
    names = REGISTERED_FEATURE_SCHEMA.input_features
    matrix = np.asarray(fit.labeled_batch().features, dtype=object)
    values: dict[str, tuple[float, float]] = {}
    for feature in REGISTERED_FEATURE_SCHEMA.geometry_numeric:
        column = np.asarray(matrix[:, names.index(feature)], dtype=np.float64)
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            raise DevelopmentFreezeError(f"fit geometry feature {feature!r} has no finite values")
        values[feature] = (
            float(np.quantile(finite, 0.25, method="linear")),
            float(np.quantile(finite, 0.75, method="linear")),
        )
    low = {
        "lid_k50": values["lid_k50"][0],
        "lid_cv": values["lid_cv"][0],
        "radius_expansion": values["radius_expansion"][0],
        "relative_contrast": values["relative_contrast"][1],
    }
    high = {
        "lid_k50": values["lid_k50"][1],
        "lid_cv": values["lid_cv"][1],
        "radius_expansion": values["radius_expansion"][1],
        "relative_contrast": values["relative_contrast"][0],
    }
    return {
        "fit_partition_only": True,
        "high_geometry": high,
        "low_geometry": low,
        "quantile_method": "numpy-linear",
        "quantiles": [0.25, 0.75],
        "risk_orientation": {
            "lid_cv": "higher-is-higher-risk",
            "lid_k50": "higher-is-higher-risk",
            "radius_expansion": "higher-is-higher-risk",
            "relative_contrast": "lower-is-higher-risk",
        },
        "schema_version": GEOMETRY_PROFILE_SCHEMA,
    }


def _selected_action(
    trial: DevelopmentTrial,
    controller: ControllerConfig,
) -> str:
    return (
        RuleController(controller)
        .decide(
            _geometry(trial),
            n_authorized=trial.authorized_count,
            policy_version="development-frozen-policy",
        )
        .action
    )


def _expected_panel(
    calibration: DevelopmentPartitionData,
    suite: FrozenModelSuite,
    controller: ControllerConfig,
) -> DevelopmentScenarioPanel:
    batch = FeatureBatch(
        partition="development-calibration",
        feature_names=REGISTERED_FEATURE_SCHEMA.input_features,
        features=np.asarray([trial.feature_values for trial in calibration.trials], dtype=object),
        corpus_ids=tuple(trial.corpus_id for trial in calibration.trials),
        family_ids=tuple(trial.family_id for trial in calibration.trials),
        row_ids=tuple(trial.trial_key for trial in calibration.trials),
    )
    reference = suite.predict_proba(batch, model_name="system-policy")
    full = suite.predict_proba(batch, model_name="full")
    rows: list[DevelopmentFamilyRow] = []
    for index, trial in enumerate(calibration.trials):
        proposed = trial.outcome(_selected_action(trial, controller))
        comparator = trial.outcome("hnsw-high")
        rows.append(
            DevelopmentFamilyRow(
                corpus_id=trial.corpus_id,
                family_id=trial.family_id,
                row_id=trial.trial_key,
                label=trial.label,
                reference_probability=float(reference[index]),
                full_probability=float(full[index]),
                proposed_latency_ms=proposed.request_latency_ms,
                comparator_latency_ms=comparator.request_latency_ms,
                proposed_execution_position=proposed.execution_position,
                comparator_execution_position=comparator.execution_position,
                proposed_retrieval_attained=proposed.retrieval_attained,
                comparator_retrieval_attained=comparator.retrieval_attained,
                proposed_evidence_sufficient=proposed.evidence_sufficient,
                comparator_evidence_sufficient=comparator.evidence_sufficient,
                denied_emissions=sum(outcome.entitlement_violations for outcome in trial.outcomes),
            )
        )
    return DevelopmentScenarioPanel(
        scenario_id="expected-development-effect",
        partition="development-calibration",
        rows=tuple(rows),
    )


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _expit(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _stable_drop(row_id: str, endpoint: str) -> bool:
    value = int.from_bytes(
        hashlib.sha256(f"{endpoint}\x00{row_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    return value < int(CONSERVATIVE_FIDELITY_DROP_RATE * 2**64)


def _conservative_panel(expected: DevelopmentScenarioPanel) -> DevelopmentScenarioPanel:
    """Apply the registered adverse attenuation map to every expected row."""

    rows: list[DevelopmentFamilyRow] = []
    for row in expected.rows:
        reference_logit = _logit(row.reference_probability)
        full_logit = _logit(row.full_probability)
        attenuated_probability = _expit(
            reference_logit + CONSERVATIVE_LOGIT_RETENTION * (full_logit - reference_logit)
        )
        ratio = row.proposed_latency_ms / row.comparator_latency_ms
        if ratio < 1.0:
            conservative_ratio = 1.0 - CONSERVATIVE_LATENCY_BENEFIT_RETENTION * (1.0 - ratio)
        else:
            conservative_ratio = 1.0 + CONSERVATIVE_LATENCY_PENALTY_MULTIPLIER * (ratio - 1.0)
        proposed_retrieval = row.proposed_retrieval_attained and not (
            row.comparator_retrieval_attained and _stable_drop(row.row_id, "retrieval")
        )
        proposed_evidence = row.proposed_evidence_sufficient
        if proposed_evidence is True and row.comparator_evidence_sufficient is True:
            proposed_evidence = not _stable_drop(row.row_id, "evidence")
        rows.append(
            DevelopmentFamilyRow(
                corpus_id=row.corpus_id,
                family_id=row.family_id,
                row_id=f"conservative:{row.row_id}",
                label=row.label,
                reference_probability=row.reference_probability,
                full_probability=attenuated_probability,
                proposed_latency_ms=conservative_ratio * row.comparator_latency_ms,
                comparator_latency_ms=row.comparator_latency_ms,
                proposed_execution_position=row.proposed_execution_position,
                comparator_execution_position=row.comparator_execution_position,
                proposed_retrieval_attained=proposed_retrieval,
                comparator_retrieval_attained=row.comparator_retrieval_attained,
                proposed_evidence_sufficient=proposed_evidence,
                comparator_evidence_sufficient=row.comparator_evidence_sufficient,
                denied_emissions=row.denied_emissions,
            )
        )
    return DevelopmentScenarioPanel(
        scenario_id="conservative-registered-attenuation",
        partition="development-calibration",
        rows=tuple(rows),
    )


def _joint_power_config(
    expected: DevelopmentScenarioPanel,
    conservative: DevelopmentScenarioPanel,
    dependence_sha256: str,
) -> JointPowerDesignConfig:
    return JointPowerDesignConfig(
        dependence_source=DependenceSource(
            artifact_uri=f"urn:sha256:{dependence_sha256}",
            artifact_sha256=dependence_sha256,
            partition="development-calibration",
            description=(
                "Canonical paired calibration action outcomes compiled before sealed access."
            ),
        ),
        effect_scenarios=(
            EffectScenario(
                scenario_id=expected.scenario_id,
                panel_sha256=expected.sha256,
                description="Observed paired calibration replay with the frozen controller.",
                selection_required=True,
            ),
            EffectScenario(
                scenario_id=conservative.scenario_id,
                panel_sha256=conservative.sha256,
                description=(
                    "Registered row-wise adverse attenuation of logits, latency, and fidelity."
                ),
                selection_required=True,
            ),
        ),
        candidate_families_per_corpus=(25, 50, 75, 100, 150, 200),
        nested_rows_per_family=REGISTERED_NESTED_ROWS,
        geometry_gain_thresholds=GEOMETRY_GAIN_THRESHOLDS,
        n_simulations=5_000,
        bound_calibration_simulations=5_000,
        simulation_seed=REGISTERED_MODEL_SEED,
    )


def _attenuation_artifact() -> dict[str, object]:
    return {
        "fidelity_drop": {
            "construction": "sha256-row-endpoint-uniform-below-rate",
            "rate": CONSERVATIVE_FIDELITY_DROP_RATE,
        },
        "full_model_logit_difference_retention": CONSERVATIVE_LOGIT_RETENTION,
        "latency_benefit_retention": CONSERVATIVE_LATENCY_BENEFIT_RETENTION,
        "latency_penalty_multiplier": CONSERVATIVE_LATENCY_PENALTY_MULTIPLIER,
        "never_improve_false_fidelity": True,
        "schema_version": ATTENUATION_ARTIFACT_SCHEMA,
    }


def _preflight(config: DevelopmentFreezeConfig) -> None:
    if os.path.lexists(config.output_root):
        raise DevelopmentFreezeError("development freeze output already exists")
    forbidden = _path_tokens(config.selection_receipt.path) & _FORBIDDEN_PATH_TOKENS
    if forbidden:
        raise DevelopmentFreezeError(
            f"selection receipt path contains forbidden sealed-role token(s) {sorted(forbidden)}"
        )
    seen_ids: set[str] = set()
    for source in config.sources:
        for pin in source.files:
            if pin.artifact_id in seen_ids:
                raise DevelopmentFreezeError("development source artifact IDs are not unique")
            seen_ids.add(pin.artifact_id)
            forbidden = _path_tokens(pin.path) & _FORBIDDEN_PATH_TOKENS
            if forbidden:
                raise DevelopmentFreezeError(
                    f"source path {pin.path} contains forbidden sealed-role token(s) "
                    f"{sorted(forbidden)}"
                )
        forbidden = _path_tokens(source.embedding_store.root) & _FORBIDDEN_PATH_TOKENS
        if forbidden:
            raise DevelopmentFreezeError(
                "embedding source path contains a forbidden sealed-role token"
            )


def _read_pin(pin: PinnedDevelopmentFile) -> bytes:
    encoded = read_secure_regular_file(
        pin.path,
        max_bytes=pin.byte_count,
        label=pin.artifact_id,
    )
    if len(encoded) != pin.byte_count or _sha256(encoded) != pin.sha256:
        raise DevelopmentFreezeError(f"source {pin.artifact_id!r} differs from its exact pin")
    return encoded


def _parse_queries(encoded: bytes) -> tuple[str, ...]:
    rows = _decode_jsonl(encoded, label="development queries")
    identifiers: list[str] = []
    for value in rows:
        row = _closed(value, _QUERY_FIELDS, label="development query")
        identifiers.append(_require_identifier("query id", row["id"]))
        if not isinstance(row["text"], str):
            raise DevelopmentFreezeError("query text must be a string")
    if len(identifiers) != len(set(identifiers)):
        raise DevelopmentFreezeError("development query file repeats an ID")
    return tuple(identifiers)


def _parse_qrels(encoded: bytes, query_ids: set[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    for value in _decode_jsonl(encoded, label="development qrels"):
        row = _closed(value, _QREL_FIELDS, label="development qrel")
        query_id = _require_identifier("qrel query_id", row["query_id"])
        document_id = _require_identifier("qrel document_id", row["document_id"])
        if query_id not in query_ids:
            raise DevelopmentFreezeError("qrel names a query outside its pinned query source")
        key = (query_id, document_id)
        if key in seen:
            raise DevelopmentFreezeError("development qrels repeat a query-document pair")
        seen.add(key)
        relevance = row["relevance"]
        if isinstance(relevance, bool) or not isinstance(relevance, int):
            raise DevelopmentFreezeError("qrel relevance must be an integer")
        if relevance > 0:
            result[query_id].add(document_id)
    if any(not result[query_id] for query_id in query_ids):
        raise DevelopmentFreezeError("every development query needs positive qrels")
    return result


def _parse_evidence(encoded: bytes, query_ids: set[str]) -> dict[str, tuple[frozenset[str], ...]]:
    result: dict[str, tuple[frozenset[str], ...]] = {}
    for value in _decode_jsonl(encoded, label="development evidence"):
        row = _closed(value, _EVIDENCE_FIELDS, label="development evidence row")
        query_id = _require_identifier("evidence query_id", row["query_id"])
        if query_id not in query_ids or query_id in result:
            raise DevelopmentFreezeError("evidence query is unknown or repeated")
        if row["answer"] is not None and not isinstance(row["answer"], str):
            raise DevelopmentFreezeError("evidence answer must be a string or null")
        if not isinstance(row["label_metadata"], list):
            raise DevelopmentFreezeError("evidence label_metadata must be an array")
        raw_bundles = row["evidence_bundles"]
        if not isinstance(raw_bundles, list) or not raw_bundles:
            raise DevelopmentFreezeError("evidence_bundles must be a non-empty array")
        bundles: list[frozenset[str]] = []
        for raw_bundle in raw_bundles:
            bundle = _closed(raw_bundle, _BUNDLE_FIELDS, label="evidence bundle")
            _require_identifier("bundle_id", bundle["bundle_id"])
            locations = bundle["locations"]
            if not isinstance(locations, list) or not locations:
                raise DevelopmentFreezeError("evidence locations must be a non-empty array")
            documents: set[str] = set()
            for raw_location in locations:
                location = _closed(raw_location, _LOCATION_FIELDS, label="evidence location")
                documents.add(_require_identifier("evidence document_id", location["document_id"]))
                _require_identifier("evidence locator", location["locator"])
            bundles.append(frozenset(documents))
        result[query_id] = tuple(bundles)
    if set(result) != query_ids:
        raise DevelopmentFreezeError("evidence corpus must label every development query")
    return result


def _parse_row_order(encoded: bytes, *, kind: str) -> tuple[Mapping[str, Any], ...]:
    rows = _decode_jsonl(encoded, label=f"embedding {kind} row order")
    for value in rows:
        row = _closed(value, _ROW_ORDER_FIELDS, label="embedding row-order row")
        if row["kind"] != kind:
            raise DevelopmentFreezeError("embedding row-order kind differs")
    return rows


@dataclass(frozen=True)
class _RawPairedAction:
    schedule_order: int
    trial_key: str
    family_id: str
    query_id: str
    action: str
    execution_position: int
    execution_state: str
    failure_state: str | None
    request_latency_ms: float
    entitlement_violations: int
    returned_document_rows: tuple[int, ...]
    feature_values: Mapping[str, object] | None


def _parse_paired_actions(encoded: bytes) -> tuple[_RawPairedAction, ...]:
    rows: list[_RawPairedAction] = []
    for value in _decode_jsonl(encoded, label="development paired actions"):
        row = _closed(value, _PAIRED_FIELDS, label="development paired-action row")
        if row["schema_version"] != PAIRED_ACTION_ROW_SCHEMA:
            raise DevelopmentFreezeError("paired-action row schema differs")
        action = row["action"]
        if action not in REGISTERED_ACTIONS:
            raise DevelopmentFreezeError("paired-action row action is not registered")
        execution_position = row["execution_position"]
        if (
            isinstance(execution_position, bool)
            or not isinstance(execution_position, int)
            or not 0 <= execution_position < len(REGISTERED_ACTIONS)
        ):
            raise DevelopmentFreezeError(
                "paired-action execution_position must be from zero to three"
            )
        schedule_order = row["schedule_order"]
        if (
            isinstance(schedule_order, bool)
            or not isinstance(schedule_order, int)
            or schedule_order < 0
        ):
            raise DevelopmentFreezeError("paired-action schedule_order must be non-negative")
        returned = row["returned_document_rows"]
        if not isinstance(returned, list) or any(
            type(item) is not int or item < 0 for item in returned
        ):
            raise DevelopmentFreezeError("returned_document_rows must contain row integers")
        if len(returned) != len(set(returned)):
            raise DevelopmentFreezeError("returned_document_rows cannot contain duplicates")
        execution = row["execution_state"]
        if execution not in {"completed", "failed", "abstained"}:
            raise DevelopmentFreezeError("paired-action execution state is not registered")
        if execution == "completed" and row["failure_state"] is not None:
            raise DevelopmentFreezeError("completed paired action cannot name a failure")
        if execution != "completed":
            _require_identifier("paired action failure_state", row["failure_state"])
            if returned:
                raise DevelopmentFreezeError("failed or abstained action cannot return documents")
        violations = row["entitlement_violations"]
        if isinstance(violations, bool) or not isinstance(violations, int) or violations < 0:
            raise DevelopmentFreezeError("paired-action violations must be non-negative")
        raw_features = row["feature_values"]
        if action == "hnsw-low":
            if not isinstance(raw_features, Mapping) or set(raw_features) != set(
                REGISTERED_FEATURE_SCHEMA.input_features
            ):
                raise DevelopmentFreezeError("hnsw-low feature_values differ from frozen schema")
        elif raw_features is not None:
            raise DevelopmentFreezeError("only hnsw-low may carry feature_values")
        rows.append(
            _RawPairedAction(
                schedule_order=schedule_order,
                trial_key=_require_sha256("paired trial_key", row["trial_key"]),
                family_id=_require_identifier("paired family_id", row["family_id"]),
                query_id=_require_identifier("paired query_id", row["query_id"]),
                action=action,
                execution_position=execution_position,
                execution_state=execution,
                failure_state=row["failure_state"],
                request_latency_ms=_finite(
                    "paired request_latency_ms", row["request_latency_ms"], minimum=0.0
                ),
                entitlement_violations=violations,
                returned_document_rows=tuple(returned),
                feature_values=raw_features,
            )
        )
    return tuple(rows)


def _load_vectors(root: Path, receipt: EmbeddingStoreReceipt, name: str) -> np.ndarray:
    descriptor = receipt.vectors[name]
    path = root / descriptor.relative_path
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise DevelopmentFreezeError("embedding vector file is linked or non-regular")
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DevelopmentFreezeError(f"cannot load embedding vector {name!r}") from exc
    if value.shape != descriptor.shape or value.dtype != np.dtype(descriptor.dtype):
        raise DevelopmentFreezeError("embedding vector geometry differs from receipt")
    return value


def _cosine_drift(active: np.ndarray, current: np.ndarray) -> float:
    left = np.asarray(active, dtype=np.float64)
    right = np.asarray(current, dtype=np.float64)
    if (
        left.ndim != 1
        or right.shape != left.shape
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise DevelopmentFreezeError("dual-epoch query vectors must be finite matched rows")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise DevelopmentFreezeError("dual-epoch query vectors cannot have zero norm")
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return 1.0 - cosine


def _recall(returned: Sequence[str], relevant: Sequence[str]) -> float:
    truth = set(relevant)
    if not truth:
        return 1.0 if not returned else 0.0
    return len(set(returned).intersection(truth)) / len(truth)


def _feature_tuple(
    values: Mapping[str, object],
    *,
    document_count: int,
    authorized_count: int,
    dimension: int,
    drift: float,
    target_allow_rate: float,
) -> tuple[object, ...]:
    expected_numeric = {
        "corpus_size": float(document_count),
        "authorized_universe_size": float(authorized_count),
        "embedding_dimension": float(dimension),
        "version_lag": 1.0,
        "drift_severity": drift,
        "allow_rate": target_allow_rate,
    }
    for name, expected in expected_numeric.items():
        observed = _finite(f"feature {name}", values[name])
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise DevelopmentFreezeError(f"feature {name!r} differs from its source-derived value")
    for name in ("probe_latency_ms", "probe_work", "policy_complexity"):
        _finite(f"feature {name}", values[name], minimum=0.0)
    for name in ("policy_churn",):
        bounded = _finite(f"feature {name}", values[name], minimum=0.0)
        if bounded > 1.0:
            raise DevelopmentFreezeError(f"feature {name!r} must be in [0, 1]")
    result: list[object] = []
    categorical = set(REGISTERED_FEATURE_SCHEMA.system_categorical)
    for name in REGISTERED_FEATURE_SCHEMA.input_features:
        value = values[name]
        if name in categorical:
            result.append(_require_identifier(f"feature {name}", value))
        elif value is None and name in REGISTERED_FEATURE_SCHEMA.geometry_numeric:
            result.append(float("nan"))
        else:
            result.append(_finite(f"feature {name}", value))
    return tuple(result)


def _load_one_source(
    source: DevelopmentCorpusSources,
    *,
    expected_query_ids: Sequence[str] | None = None,
) -> tuple[DevelopmentTrial, ...]:
    query_ids = _parse_queries(_read_pin(source.queries))
    if expected_query_ids is not None and set(query_ids) != set(expected_query_ids):
        raise DevelopmentFreezeError(
            "pinned query source differs from the pre-label selection receipt"
        )
    query_set = set(query_ids)
    qrels = _parse_qrels(_read_pin(source.qrels), query_set)
    evidence = (
        {}
        if source.evidence_bundles is None
        else _parse_evidence(_read_pin(source.evidence_bundles), query_set)
    )
    schedule_bytes = _read_pin(source.policy_schedule)
    schedule = loads_canonical_trial_schedule(schedule_bytes)
    if schedule.corpus != source.corpus_id or schedule.stage != source.stage:
        raise DevelopmentFreezeError("policy schedule corpus or stage differs from its source pin")
    if len(schedule.grouped_execution_order) != REGISTERED_NESTED_ROWS:
        raise DevelopmentFreezeError("policy schedule must contain three allow-rate groups")
    if {row.subject for row in schedule.rows}.__len__() != 1 or {
        row.repetition for row in schedule.rows
    } != {0}:
        raise DevelopmentFreezeError("policy schedule must use one subject and one repetition")
    paired = _parse_paired_actions(_read_pin(source.paired_actions))

    receipt = verify_embedding_store(source.embedding_store.root)
    if receipt.receipt_sha256 != source.embedding_store.receipt_sha256:
        raise DevelopmentFreezeError("embedding store receipt differs from its exact pin")
    if receipt.old_model is None:
        raise DevelopmentFreezeError("embedding store must contain old and current epochs")
    required_vectors = {"old_documents", "current_documents", "old_queries", "current_queries"}
    if set(receipt.vectors) != required_vectors:
        raise DevelopmentFreezeError("embedding store is not a closed dual-epoch store")
    if receipt.document_count != schedule.document_count:
        raise DevelopmentFreezeError("embedding document count differs from policy schedule")

    query_descriptor = receipt.row_orders["queries"]
    query_order = _parse_row_order(
        read_secure_regular_file(
            source.embedding_store.root / query_descriptor.relative_path,
            max_bytes=query_descriptor.byte_count,
            label="embedding query row order",
        ),
        kind="query",
    )
    document_descriptor = receipt.row_orders["documents"]
    document_order = _parse_row_order(
        read_secure_regular_file(
            source.embedding_store.root / document_descriptor.relative_path,
            max_bytes=document_descriptor.byte_count,
            label="embedding document row order",
        ),
        kind="document",
    )
    document_ids = tuple(str(row["id"]) for row in document_order)
    if any(row["dataset"] != source.corpus_id for row in document_order):
        raise DevelopmentFreezeError("embedding document rows cross the corpus source boundary")
    if len(document_ids) != len(set(document_ids)):
        raise DevelopmentFreezeError("embedding document row order repeats an ID")
    known_documents = set(document_ids)
    if any(not documents.issubset(known_documents) for documents in qrels.values()):
        raise DevelopmentFreezeError("development qrels name documents outside the row order")
    if any(
        not bundle.issubset(known_documents) for bundles in evidence.values() for bundle in bundles
    ):
        raise DevelopmentFreezeError("development evidence names documents outside the row order")
    matching_query_rows = [
        row
        for row in query_order
        if row["dataset"] == source.corpus_id and row["stage"] == source.stage
    ]
    query_positions = {
        str(row["id"]): position
        for position, row in enumerate(query_order)
        if row["dataset"] == source.corpus_id and row["stage"] == source.stage
    }
    if len(query_positions) != len(matching_query_rows) or set(query_positions) != query_set:
        raise DevelopmentFreezeError("embedding query rows differ from the pinned query source")
    active_queries = _load_vectors(source.embedding_store.root, receipt, "old_queries")
    current_queries = _load_vectors(source.embedding_store.root, receipt, "current_queries")
    drift = {
        query_id: _cosine_drift(
            active_queries[position],
            current_queries[position],
        )
        for query_id, position in query_positions.items()
    }
    dimension = receipt.vectors["current_queries"].shape[1]

    schedule_by_order = {row.schedule_order: row for row in schedule.rows}
    paired_by_order: dict[int, list[_RawPairedAction]] = defaultdict(list)
    for row in paired:
        paired_by_order[row.schedule_order].append(row)
    if set(paired_by_order) != set(schedule_by_order):
        raise DevelopmentFreezeError("paired action trials differ from the policy schedule")
    trials: list[DevelopmentTrial] = []
    observed_queries: set[str] = set()
    for schedule_order in sorted(schedule_by_order):
        schedule_row = schedule_by_order[schedule_order]
        rows = sorted(
            paired_by_order[schedule_order], key=lambda row: REGISTERED_ACTIONS.index(row.action)
        )
        if tuple(row.action for row in rows) != REGISTERED_ACTIONS:
            raise DevelopmentFreezeError("one scheduled trial lacks the complete paired action set")
        if {row.execution_position for row in rows} != set(range(len(REGISTERED_ACTIONS))):
            raise DevelopmentFreezeError(
                "paired actions do not retain one exact execution position"
            )
        if any(row.trial_key != schedule_row.trial_key for row in rows):
            raise DevelopmentFreezeError("paired action trial key differs from policy schedule")
        if len({row.query_id for row in rows}) != 1 or len({row.family_id for row in rows}) != 1:
            raise DevelopmentFreezeError("paired actions do not share query and family identities")
        query_id = rows[0].query_id
        family_id = rows[0].family_id
        if query_id not in query_set:
            raise DevelopmentFreezeError("paired action names an unknown development query")
        observed_queries.add(query_id)
        target_rate = REGISTERED_ALLOW_RATES[schedule_row.group_order]
        if abs(schedule_row.realized_allow_rate - target_rate) > 1.0 / schedule.document_count:
            raise DevelopmentFreezeError(
                "realized allow rate is not the registered rounded stratum"
            )
        low = rows[0]
        if low.feature_values is None:
            raise DevelopmentFreezeError("low action lacks its feature vector")
        feature_values = _feature_tuple(
            low.feature_values,
            document_count=schedule.document_count,
            authorized_count=schedule_row.authorized_count,
            dimension=dimension,
            drift=drift[query_id],
            target_allow_rate=target_rate,
        )
        exact = rows[2]
        if exact.execution_state != "completed":
            raise DevelopmentFreezeError("exact-authorized development oracle must complete")
        if any(index >= len(document_ids) for index in exact.returned_document_rows):
            raise DevelopmentFreezeError("exact-authorized returned a document row out of range")
        exact_ids = tuple(document_ids[index] for index in exact.returned_document_rows)
        outcomes: list[ActionOutcome] = []
        for row in rows:
            if any(index >= len(document_ids) for index in row.returned_document_rows):
                raise DevelopmentFreezeError("paired action returned a document row out of range")
            all_returned_ids = tuple(document_ids[index] for index in row.returned_document_rows)
            returned_ids = all_returned_ids[:REGISTERED_K]
            exact_recall = (
                _recall(returned_ids, exact_ids[:REGISTERED_K])
                if row.execution_state == "completed"
                else 0.0
            )
            qrel_recall = (
                _recall(returned_ids, tuple(qrels[query_id]))
                if row.execution_state == "completed"
                else 0.0
            )
            if source.corpus_id in EVIDENCE_CORPORA:
                evidence_sufficient: bool | None = (
                    row.execution_state == "completed"
                    and row.entitlement_violations == 0
                    and any(bundle.issubset(returned_ids) for bundle in evidence[query_id])
                )
            else:
                evidence_sufficient = None
            outcomes.append(
                ActionOutcome(
                    action=row.action,
                    execution_position=row.execution_position,
                    execution_state=row.execution_state,
                    failure_state=row.failure_state,
                    request_latency_ms=row.request_latency_ms,
                    retrieval_attained=(
                        row.execution_state == "completed"
                        and exact_recall >= REGISTERED_FAILURE_RECALL
                    ),
                    qrel_recall_at_k=qrel_recall,
                    evidence_sufficient=evidence_sufficient,
                    entitlement_violations=row.entitlement_violations,
                    returned_document_rows=row.returned_document_rows,
                    returned_document_ids=all_returned_ids,
                )
            )
        low_outcome = outcomes[0]
        label = int(
            low_outcome.execution_state != "completed" or not low_outcome.retrieval_attained
        )
        trials.append(
            DevelopmentTrial(
                partition=source.stage,
                corpus_id=source.corpus_id,
                family_id=family_id,
                query_id=query_id,
                trial_key=schedule_row.trial_key,
                subject=schedule_row.subject,
                repetition=schedule_row.repetition,
                target_allow_rate=target_rate,
                realized_allow_rate=schedule_row.realized_allow_rate,
                authorized_count=schedule_row.authorized_count,
                feature_values=feature_values,
                label=label,
                outcomes=tuple(outcomes),
            )
        )
    if observed_queries != query_set:
        raise DevelopmentFreezeError("paired action source omits development queries")
    return tuple(trials)


def _load_development_sources(
    config: DevelopmentFreezeConfig,
) -> tuple[DevelopmentPartitionData, DevelopmentPartitionData]:
    selection_bytes = read_secure_regular_file(
        config.selection_receipt.path,
        max_bytes=config.selection_receipt.byte_count,
        label="development cohort selection receipt",
    )
    if (
        len(selection_bytes) != config.selection_receipt.byte_count
        or _sha256(selection_bytes) != config.selection_receipt.sha256
    ):
        raise DevelopmentFreezeError(
            "development selection receipt differs from its exact direct pin"
        )
    selection: DevelopmentCohortSelectionReceipt = load_development_cohort_selection(
        config.selection_receipt.path,
        expected_artifact_sha256=config.selection_receipt.sha256,
    )
    by_stage: dict[str, list[DevelopmentTrial]] = defaultdict(list)
    for source in config.sources:
        expected_query_ids = selection.selected_query_ids(
            source.corpus_id,
            str(source.stage),
        )
        by_stage[source.stage].extend(
            _load_one_source(source, expected_query_ids=expected_query_ids)
        )
    fit = DevelopmentPartitionData("development-fit", tuple(by_stage["development-fit"]))
    calibration = DevelopmentPartitionData(
        "development-calibration", tuple(by_stage["development-calibration"])
    )
    if {trial.family_id for trial in fit.trials}.intersection(
        trial.family_id for trial in calibration.trials
    ):
        raise DevelopmentFreezeError("fit and calibration family IDs must be disjoint")
    if len({trial.subject for trial in (*fit.trials, *calibration.trials)}) != 1:
        raise DevelopmentFreezeError("development design must use exactly one subject")
    return fit, calibration


def _source_bindings(config: DevelopmentFreezeConfig) -> list[dict[str, object]]:
    bindings: list[dict[str, object]] = [
        {
            "artifact_id": config.selection_receipt.artifact_id,
            "byte_count": config.selection_receipt.byte_count,
            "role": "development-cohort-selection",
            "sha256": config.selection_receipt.sha256,
        }
    ]
    for source in config.sources:
        for pin in source.files:
            bindings.append(
                {
                    "artifact_id": pin.artifact_id,
                    "byte_count": pin.byte_count,
                    "corpus_id": pin.corpus_id,
                    "role": pin.role,
                    "sha256": pin.sha256,
                    "stage": pin.stage,
                }
            )
        bindings.append(
            {
                "artifact_id": source.embedding_store.artifact_id,
                "corpus_id": source.corpus_id,
                "receipt_sha256": source.embedding_store.receipt_sha256,
                "role": "dual-epoch-embedding-store",
                "stage": source.stage,
            }
        )
    return sorted(bindings, key=lambda row: str(row["artifact_id"]))


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _exclusive_publish(work: Path, output: Path) -> None:
    if os.path.lexists(output):
        raise DevelopmentFreezeError("development freeze output already exists")
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(work)
    destination = os.fsencode(output)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise DevelopmentFreezeError("exclusive directory rename is unavailable on macOS")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-2, source, -2, destination, 0x00000004)
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise DevelopmentFreezeError("exclusive directory rename is unavailable on Linux")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, source, -100, destination, 0x00000001)
    else:
        raise DevelopmentFreezeError(
            f"exclusive directory rename is unsupported on platform {sys.platform!r}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise DevelopmentFreezeError("development freeze output already exists")
        raise DevelopmentFreezeError(
            f"cannot publish development freeze: {os.strerror(error_number)}"
        )


def _compile_payloads(
    fit: DevelopmentPartitionData,
    calibration: DevelopmentPartitionData,
) -> tuple[dict[str, bytes], FrozenModelSuite, ControllerConfig]:
    fit_feature = _canonical_bytes(_feature_artifact(fit))
    calibration_feature = _canonical_bytes(_feature_artifact(calibration))
    fit_outcomes = _canonical_bytes(_outcome_artifact(fit))
    calibration_outcomes = _canonical_bytes(_outcome_artifact(calibration))
    suite = fit_frozen_model_suite(
        fit.labeled_batch(),
        calibration.labeled_batch(),
        schema=REGISTERED_FEATURE_SCHEMA,
        random_seed=REGISTERED_MODEL_SEED,
    )
    controller, controller_artifact = _select_controller(calibration)
    expected = _expected_panel(calibration, suite, controller)
    conservative = _conservative_panel(expected)
    power_config = _joint_power_config(expected, conservative, _sha256(calibration_outcomes))
    comparator = {
        "action": "hnsw-high",
        "chosen_a_priori": True,
        "schema_version": COMPARATOR_ARTIFACT_SCHEMA,
        "selection_data": None,
    }
    attenuation = _attenuation_artifact()
    profiles = _geometry_profiles(fit)
    profiles["geometry_gain_thresholds"] = {
        "auprc_gain": GEOMETRY_GAIN_THRESHOLDS.auprc_gain,
        "brier_score_reduction": GEOMETRY_GAIN_THRESHOLDS.brier_score_reduction,
        "log_loss_reduction": GEOMETRY_GAIN_THRESHOLDS.log_loss_reduction,
    }
    payloads = {
        "controller.json": _canonical_bytes(controller_artifact),
        "development-calibration-features.json": calibration_feature,
        "development-calibration-outcomes.json": calibration_outcomes,
        "development-fit-features.json": fit_feature,
        "development-fit-outcomes.json": fit_outcomes,
        "geometry-profiles.json": _canonical_bytes(profiles),
        "h1-model.json": canonical_h1_model_artifact_bytes(suite),
        "h2-model-suite.json": canonical_h2_model_suite_artifact_bytes(suite),
        "joint-power-config.json": canonical_joint_power_config_bytes(power_config),
        "joint-power-conservative-panel.json": canonical_development_panel_bytes(conservative),
        "joint-power-expected-panel.json": canonical_development_panel_bytes(expected),
        "static-comparator.json": _canonical_bytes(comparator),
        "scenario-attenuation.json": _canonical_bytes(attenuation),
    }
    return payloads, suite, controller


def compile_development_freeze(config: DevelopmentFreezeConfig) -> Mapping[str, object]:
    """Compile and publish one development-only freeze package.

    All source paths are inspected for sealed-role tokens before the first
    source byte is opened.  Existing output paths are never replaced.
    """

    if not isinstance(config, DevelopmentFreezeConfig):
        raise TypeError("config must be DevelopmentFreezeConfig")
    _preflight(config)
    fit, calibration = _load_development_sources(config)
    payloads, suite, controller = _compile_payloads(fit, calibration)
    artifacts = [
        {"byte_count": len(payload), "path": path, "sha256": _sha256(payload)}
        for path, payload in sorted(payloads.items())
    ]
    receipt: dict[str, object] = {
        "artifacts": artifacts,
        "controller_config": _controller_config_dict(controller),
        "development_group_digest": suite.development_group_digest,
        "feature_schema_digest": suite.schema_digest,
        "model_suite_digest": suite.suite_digest,
        "schema_version": DEVELOPMENT_FREEZE_SCHEMA,
        "source_bindings": _source_bindings(config),
        "static_comparator": "hnsw-high",
    }
    receipt_bytes = _canonical_bytes(receipt)
    payloads["freeze-receipt.json"] = receipt_bytes

    output = config.output_root
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for relative_path, payload in sorted(payloads.items()):
            _exclusive_write(temporary / relative_path, payload)
        directory_descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _exclusive_publish(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def _load_canonical_object(path: Path) -> Mapping[str, Any]:
    encoded = read_secure_regular_file(path, max_bytes=256 * 1024 * 1024, label=path.name)
    value = _decode_json(encoded, label=path.name)
    if not isinstance(value, Mapping) or encoded != _canonical_bytes(value):
        raise DevelopmentFreezeError(f"{path.name} is not canonical JSON")
    return value


def _partition_from_artifacts(
    feature_encoded: bytes,
    outcome_encoded: bytes,
    *,
    partition: str,
) -> DevelopmentPartitionData:
    feature = _decode_json(feature_encoded, label=f"{partition} feature artifact")
    feature = _closed(
        feature,
        frozenset({"feature_names", "partition", "rows", "schema_version"}),
        label=f"{partition} feature artifact",
    )
    if (
        feature["schema_version"] != FEATURE_ARTIFACT_SCHEMA
        or feature["partition"] != partition
        or feature["feature_names"] != list(REGISTERED_FEATURE_SCHEMA.input_features)
        or feature_encoded != _canonical_bytes(feature)
    ):
        raise DevelopmentFreezeError(f"{partition} feature artifact contract differs")
    raw_feature_rows = feature["rows"]
    if not isinstance(raw_feature_rows, list) or not raw_feature_rows:
        raise DevelopmentFreezeError(f"{partition} feature rows must be non-empty")
    features_by_trial: dict[str, tuple[str, str, str, tuple[object, ...], int]] = {}
    for raw_row in raw_feature_rows:
        row = _closed(
            raw_row,
            frozenset({"corpus_id", "family_id", "feature_values", "label", "query_id", "row_id"}),
            label=f"{partition} feature row",
        )
        values = row["feature_values"]
        if not isinstance(values, list) or len(values) != len(
            REGISTERED_FEATURE_SCHEMA.input_features
        ):
            raise DevelopmentFreezeError(f"{partition} feature vector width differs")
        label = row["label"]
        if isinstance(label, bool) or label not in {0, 1}:
            raise DevelopmentFreezeError(f"{partition} feature label differs")
        corpus_id = row["corpus_id"]
        if corpus_id not in FIXED_CORPORA:
            raise DevelopmentFreezeError(f"{partition} feature corpus differs")
        family_id = _require_identifier("feature family_id", row["family_id"])
        query_id = _require_identifier("feature query_id", row["query_id"])
        trial_key = _require_sha256("feature row_id", row["row_id"])
        if trial_key in features_by_trial:
            raise DevelopmentFreezeError(f"{partition} feature artifact repeats a row")
        features_by_trial[trial_key] = (
            corpus_id,
            family_id,
            query_id,
            tuple(float("nan") if value is None else value for value in values),
            label,
        )

    outcomes = _decode_json(outcome_encoded, label=f"{partition} outcome artifact")
    outcomes = _closed(
        outcomes,
        frozenset({"action_order", "partition", "rows", "schema_version"}),
        label=f"{partition} outcome artifact",
    )
    raw_outcome_rows = outcomes["rows"]
    if (
        outcomes["schema_version"] != OUTCOME_ARTIFACT_SCHEMA
        or outcomes["partition"] != partition
        or outcomes["action_order"] != list(REGISTERED_ACTIONS)
        or not isinstance(raw_outcome_rows, list)
        or not raw_outcome_rows
        or outcome_encoded != _canonical_bytes(outcomes)
    ):
        raise DevelopmentFreezeError(f"{partition} outcome artifact contract differs")
    outcome_row_fields = frozenset(
        {
            "authorized_count",
            "corpus_id",
            "family_id",
            "outcomes",
            "query_id",
            "realized_allow_rate",
            "repetition",
            "subject",
            "target_allow_rate",
            "trial_key",
        }
    )
    outcome_fields = frozenset(
        {
            "action",
            "entitlement_violations",
            "evidence_sufficient",
            "execution_position",
            "execution_state",
            "failure_state",
            "qrel_recall_at_k",
            "request_latency_ms",
            "retrieval_attained",
            "returned_document_ids",
            "returned_document_rows",
        }
    )
    trials: list[DevelopmentTrial] = []
    observed_trial_keys: set[str] = set()
    for raw_row in raw_outcome_rows:
        row = _closed(raw_row, outcome_row_fields, label=f"{partition} outcome row")
        trial_key = _require_sha256("outcome trial_key", row["trial_key"])
        if trial_key in observed_trial_keys or trial_key not in features_by_trial:
            raise DevelopmentFreezeError(f"{partition} outcome trial is unknown or repeated")
        observed_trial_keys.add(trial_key)
        corpus_id, family_id, query_id, feature_values, label = features_by_trial[trial_key]
        if (row["corpus_id"], row["family_id"], row["query_id"]) != (
            corpus_id,
            family_id,
            query_id,
        ):
            raise DevelopmentFreezeError(f"{partition} feature/outcome identity differs")
        raw_actions = row["outcomes"]
        if not isinstance(raw_actions, list):
            raise DevelopmentFreezeError(f"{partition} outcomes must be an array")
        actions: list[ActionOutcome] = []
        for raw_action in raw_actions:
            action = _closed(raw_action, outcome_fields, label=f"{partition} action outcome")
            returned_rows = action["returned_document_rows"]
            returned_ids = action["returned_document_ids"]
            if not isinstance(returned_rows, list) or not isinstance(returned_ids, list):
                raise DevelopmentFreezeError("returned document fields must be arrays")
            actions.append(
                ActionOutcome(
                    action=action["action"],
                    execution_position=action["execution_position"],
                    execution_state=action["execution_state"],
                    failure_state=action["failure_state"],
                    request_latency_ms=action["request_latency_ms"],
                    retrieval_attained=action["retrieval_attained"],
                    qrel_recall_at_k=action["qrel_recall_at_k"],
                    evidence_sufficient=action["evidence_sufficient"],
                    entitlement_violations=action["entitlement_violations"],
                    returned_document_rows=tuple(returned_rows),
                    returned_document_ids=tuple(returned_ids),
                )
            )
        trials.append(
            DevelopmentTrial(
                partition=partition,
                corpus_id=corpus_id,
                family_id=family_id,
                query_id=query_id,
                trial_key=trial_key,
                subject=row["subject"],
                repetition=row["repetition"],
                target_allow_rate=row["target_allow_rate"],
                realized_allow_rate=row["realized_allow_rate"],
                authorized_count=row["authorized_count"],
                feature_values=feature_values,
                label=label,
                outcomes=tuple(actions),
            )
        )
    if observed_trial_keys != set(features_by_trial):
        raise DevelopmentFreezeError(f"{partition} outcomes omit feature rows")
    return DevelopmentPartitionData(partition, tuple(trials))


def verify_development_freeze(root: str | Path) -> Mapping[str, object]:
    """Verify package membership, canonical bytes, model pins, and scenario pins."""

    package = Path(root)
    if not package.is_absolute():
        raise DevelopmentFreezeError("freeze package root must be absolute")
    metadata = package.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise DevelopmentFreezeError("freeze package root must be a real directory")
    receipt = _load_canonical_object(package / "freeze-receipt.json")
    expected_receipt_fields = frozenset(
        {
            "artifacts",
            "controller_config",
            "development_group_digest",
            "feature_schema_digest",
            "model_suite_digest",
            "schema_version",
            "source_bindings",
            "static_comparator",
        }
    )
    _closed(receipt, expected_receipt_fields, label="development freeze receipt")
    if receipt["schema_version"] != DEVELOPMENT_FREEZE_SCHEMA:
        raise DevelopmentFreezeError("freeze receipt schema differs")
    for name in (
        "development_group_digest",
        "feature_schema_digest",
        "model_suite_digest",
    ):
        _require_sha256(name, receipt[name])
    if receipt["static_comparator"] != "hnsw-high":
        raise DevelopmentFreezeError("freeze receipt static comparator differs")
    controller_config = _closed(
        receipt["controller_config"],
        frozenset(
            {
                "exact_scan_threshold",
                "exact_threshold",
                "high_ef",
                "high_effort_threshold",
                "low_ef",
                "probe_k",
            }
        ),
        label="receipt controller config",
    )
    try:
        ControllerConfig(**controller_config)
    except (TypeError, ValueError) as exc:
        raise DevelopmentFreezeError("receipt controller config is invalid") from exc
    source_bindings = receipt["source_bindings"]
    if not isinstance(source_bindings, list) or len(source_bindings) != 57:
        raise DevelopmentFreezeError("freeze receipt must bind the 57 development sources")
    source_ids: set[str] = set()
    for binding in source_bindings:
        if not isinstance(binding, Mapping):
            raise DevelopmentFreezeError("source binding must be an object")
        role = binding.get("role")
        if role == "development-cohort-selection":
            row = _closed(
                binding,
                frozenset({"artifact_id", "byte_count", "role", "sha256"}),
                label="development selection source binding",
            )
            if row["artifact_id"] != "development-cohort-selection":
                raise DevelopmentFreezeError("development selection artifact ID differs")
            if (
                isinstance(row["byte_count"], bool)
                or not isinstance(row["byte_count"], int)
                or row["byte_count"] <= 0
            ):
                raise DevelopmentFreezeError("selection receipt byte_count must be positive")
            _require_sha256("selection receipt sha256", row["sha256"])
        elif role == "dual-epoch-embedding-store":
            row = _closed(
                binding,
                frozenset({"artifact_id", "corpus_id", "receipt_sha256", "role", "stage"}),
                label="embedding source binding",
            )
            _require_sha256("embedding source receipt_sha256", row["receipt_sha256"])
        else:
            row = _closed(
                binding,
                frozenset({"artifact_id", "byte_count", "corpus_id", "role", "sha256", "stage"}),
                label="file source binding",
            )
            if row["role"] not in {
                "queries",
                "qrels",
                "evidence-bundles",
                "policy-schedule",
                "paired-actions",
            }:
                raise DevelopmentFreezeError("file source binding role differs")
            if (
                isinstance(row["byte_count"], bool)
                or not isinstance(row["byte_count"], int)
                or row["byte_count"] <= 0
            ):
                raise DevelopmentFreezeError("file source byte_count must be positive")
            _require_sha256("file source sha256", row["sha256"])
        artifact_id = _require_identifier("source artifact_id", row["artifact_id"])
        if artifact_id in source_ids:
            raise DevelopmentFreezeError("freeze receipt repeats a source binding")
        source_ids.add(artifact_id)
        if role != "development-cohort-selection" and (
            row["corpus_id"] not in FIXED_CORPORA
            or row["stage"]
            not in {
                "development-fit",
                "development-calibration",
            }
        ):
            raise DevelopmentFreezeError("source binding corpus or stage differs")
    artifacts = receipt["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise DevelopmentFreezeError("freeze receipt artifacts must be non-empty")
    expected_paths = {"freeze-receipt.json"}
    encoded_by_path: dict[str, bytes] = {}
    for value in artifacts:
        row = _closed(
            value,
            frozenset({"byte_count", "path", "sha256"}),
            label="freeze artifact binding",
        )
        relative = _require_identifier("freeze artifact path", row["path"])
        if Path(relative).name != relative:
            raise DevelopmentFreezeError("freeze artifacts must be direct package children")
        if relative in expected_paths:
            raise DevelopmentFreezeError("freeze receipt repeats an artifact path")
        expected_paths.add(relative)
        byte_count = row["byte_count"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or not 0 < byte_count <= 256 * 1024 * 1024
        ):
            raise DevelopmentFreezeError("freeze artifact byte_count is invalid")
        expected_sha256 = _require_sha256("freeze artifact sha256", row["sha256"])
        encoded = read_secure_regular_file(
            package / relative,
            max_bytes=byte_count,
            label=f"freeze artifact {relative}",
        )
        if len(encoded) != byte_count or _sha256(encoded) != expected_sha256:
            raise DevelopmentFreezeError(f"freeze artifact {relative!r} differs from receipt")
        encoded_by_path[relative] = encoded
    observed = {child.name for child in package.iterdir()}
    if observed != expected_paths:
        raise DevelopmentFreezeError("freeze package membership differs from receipt")

    required_artifacts = {
        "controller.json",
        "development-calibration-features.json",
        "development-calibration-outcomes.json",
        "development-fit-features.json",
        "development-fit-outcomes.json",
        "geometry-profiles.json",
        "h1-model.json",
        "h2-model-suite.json",
        "joint-power-config.json",
        "joint-power-conservative-panel.json",
        "joint-power-expected-panel.json",
        "scenario-attenuation.json",
        "static-comparator.json",
    }
    if set(encoded_by_path) != required_artifacts:
        raise DevelopmentFreezeError("freeze receipt does not contain the closed artifact set")

    partitions = {
        partition: _partition_from_artifacts(
            encoded_by_path[f"{partition}-features.json"],
            encoded_by_path[f"{partition}-outcomes.json"],
            partition=partition,
        )
        for partition in ("development-fit", "development-calibration")
    }
    fit = partitions["development-fit"]
    calibration = partitions["development-calibration"]
    if {trial.family_id for trial in fit.trials}.intersection(
        trial.family_id for trial in calibration.trials
    ):
        raise DevelopmentFreezeError("frozen fit and calibration families overlap")
    if len({trial.subject for trial in (*fit.trials, *calibration.trials)}) != 1:
        raise DevelopmentFreezeError("frozen development data does not use one subject")

    controller = _decode_json(encoded_by_path["controller.json"], label="controller artifact")
    controller = _closed(
        controller,
        frozenset(
            {
                "candidate_evaluations",
                "constraints",
                "objective",
                "objective_rounding_decimal_places",
                "schema_version",
                "selected_config",
                "selected_metrics",
                "static_comparator",
                "tie_break",
            }
        ),
        label="controller artifact",
    )
    if (
        controller["schema_version"] != CONTROLLER_ARTIFACT_SCHEMA
        or controller["static_comparator"] != "hnsw-high"
        or controller["selected_config"] != receipt["controller_config"]
        or encoded_by_path["controller.json"] != _canonical_bytes(controller)
    ):
        raise DevelopmentFreezeError("controller artifact contract differs")
    rebuilt_controller, rebuilt_controller_artifact = _select_controller(calibration)
    if controller != rebuilt_controller_artifact:
        raise DevelopmentFreezeError("controller artifact does not reproduce from calibration")

    comparator = _decode_json(encoded_by_path["static-comparator.json"], label="static comparator")
    comparator = _closed(
        comparator,
        frozenset({"action", "chosen_a_priori", "schema_version", "selection_data"}),
        label="static comparator",
    )
    if comparator != {
        "action": "hnsw-high",
        "chosen_a_priori": True,
        "schema_version": COMPARATOR_ARTIFACT_SCHEMA,
        "selection_data": None,
    }:
        raise DevelopmentFreezeError("static comparator artifact differs")

    profiles = _decode_json(encoded_by_path["geometry-profiles.json"], label="geometry profiles")
    profiles = _closed(
        profiles,
        frozenset(
            {
                "fit_partition_only",
                "geometry_gain_thresholds",
                "high_geometry",
                "low_geometry",
                "quantile_method",
                "quantiles",
                "risk_orientation",
                "schema_version",
            }
        ),
        label="geometry profiles",
    )
    if (
        profiles["schema_version"] != GEOMETRY_PROFILE_SCHEMA
        or profiles["fit_partition_only"] is not True
        or profiles["quantiles"] != [0.25, 0.75]
        or profiles["quantile_method"] != "numpy-linear"
    ):
        raise DevelopmentFreezeError("geometry profile artifact differs")
    rebuilt_profiles = _geometry_profiles(fit)
    rebuilt_profiles["geometry_gain_thresholds"] = {
        "auprc_gain": GEOMETRY_GAIN_THRESHOLDS.auprc_gain,
        "brier_score_reduction": GEOMETRY_GAIN_THRESHOLDS.brier_score_reduction,
        "log_loss_reduction": GEOMETRY_GAIN_THRESHOLDS.log_loss_reduction,
    }
    if profiles != rebuilt_profiles:
        raise DevelopmentFreezeError("geometry profiles do not reproduce from development fit")

    attenuation = _decode_json(
        encoded_by_path["scenario-attenuation.json"], label="scenario attenuation"
    )
    attenuation = _closed(
        attenuation,
        frozenset(
            {
                "fidelity_drop",
                "full_model_logit_difference_retention",
                "latency_benefit_retention",
                "latency_penalty_multiplier",
                "never_improve_false_fidelity",
                "schema_version",
            }
        ),
        label="scenario attenuation",
    )
    if attenuation["schema_version"] != ATTENUATION_ARTIFACT_SCHEMA or encoded_by_path[
        "scenario-attenuation.json"
    ] != _canonical_bytes(attenuation):
        raise DevelopmentFreezeError("scenario attenuation artifact is not canonical")
    if attenuation != _attenuation_artifact():
        raise DevelopmentFreezeError("scenario attenuation constants differ")

    suite_bytes = encoded_by_path["h2-model-suite.json"]
    suite = FrozenModelSuite.from_json(suite_bytes.decode("utf-8"))
    rebuilt_suite = fit_frozen_model_suite(
        fit.labeled_batch(),
        calibration.labeled_batch(),
        schema=REGISTERED_FEATURE_SCHEMA,
        random_seed=REGISTERED_MODEL_SEED,
    )
    if canonical_h2_model_suite_artifact_bytes(rebuilt_suite) != suite_bytes:
        raise DevelopmentFreezeError("H2 suite does not reproduce from development artifacts")
    if (
        suite.suite_digest != receipt["model_suite_digest"]
        or suite.schema_digest != receipt["feature_schema_digest"]
        or suite.development_group_digest != receipt["development_group_digest"]
    ):
        raise DevelopmentFreezeError("frozen model suite digest differs from receipt")
    if canonical_h1_model_artifact_bytes(suite) != encoded_by_path["h1-model.json"]:
        raise DevelopmentFreezeError("H1 model bytes differ from the H2 suite full model")
    expected_panel = load_development_panel(encoded_by_path["joint-power-expected-panel.json"])
    conservative_panel = load_development_panel(
        encoded_by_path["joint-power-conservative-panel.json"]
    )
    power_config = load_joint_power_config(encoded_by_path["joint-power-config.json"])
    scenario_pins = {
        scenario.scenario_id: scenario.panel_sha256 for scenario in power_config.effect_scenarios
    }
    if scenario_pins != {
        expected_panel.scenario_id: expected_panel.sha256,
        conservative_panel.scenario_id: conservative_panel.sha256,
    }:
        raise DevelopmentFreezeError("joint-power config does not pin both scenario panels")
    rebuilt_expected = _expected_panel(calibration, suite, rebuilt_controller)
    rebuilt_conservative = _conservative_panel(rebuilt_expected)
    rebuilt_power_config = _joint_power_config(
        rebuilt_expected,
        rebuilt_conservative,
        _sha256(encoded_by_path["development-calibration-outcomes.json"]),
    )
    if (
        canonical_development_panel_bytes(rebuilt_expected)
        != encoded_by_path["joint-power-expected-panel.json"]
        or canonical_development_panel_bytes(rebuilt_conservative)
        != encoded_by_path["joint-power-conservative-panel.json"]
        or canonical_joint_power_config_bytes(rebuilt_power_config)
        != encoded_by_path["joint-power-config.json"]
    ):
        raise DevelopmentFreezeError("joint-power inputs do not reproduce from development data")
    return receipt


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.development_freeze",
        description="Compile or verify the development-only confirmatory freeze package.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compile_command = commands.add_parser(
        "compile",
        help="compile one canonical development config into a new immutable package",
    )
    compile_command.add_argument(
        "--config",
        required=True,
        help=(f"absolute path to a canonical {DEVELOPMENT_FREEZE_CONFIG_SCHEMA} file"),
    )
    verify_command = commands.add_parser(
        "verify",
        help="verify and reproduce every derived artifact in an existing package",
    )
    verify_command.add_argument(
        "--root",
        required=True,
        help="absolute path to the development freeze package",
    )
    return parser


def _write_cli_result(
    *,
    command: str,
    root: Path,
    receipt: Mapping[str, object],
) -> None:
    result = {
        "command": command,
        "receipt_sha256": _sha256(_canonical_bytes(receipt)),
        "root": str(root),
        "schema_version": DEVELOPMENT_FREEZE_CLI_RESULT_SCHEMA,
    }
    sys.stdout.buffer.write(_canonical_bytes(result))
    sys.stdout.buffer.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the module-local compiler and verifier commands."""

    parser = _cli_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "compile":
            config = load_development_freeze_config(arguments.config)
            receipt = compile_development_freeze(config)
            root = config.output_root
        else:
            root = _config_absolute_path("freeze root", arguments.root)
            receipt = verify_development_freeze(root)
        _write_cli_result(command=arguments.command, root=root, receipt=receipt)
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
