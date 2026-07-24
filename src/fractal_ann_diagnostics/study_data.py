"""Deterministic data staging with process-local sealed-label withholding.

The module converts the source formats accepted by :mod:`corpora` into one
canonical interchange: ``corpus.jsonl``, ``queries.jsonl``, ``qrels.jsonl``, and
custody-separated ``evidence-bundles.jsonl`` for evidence-bearing corpora.
Every source is read through a no-follow file descriptor and must match both a
declared revision and SHA-256 pin.  Output is assembled in a temporary sibling
directory and published with one rename only after every source, assignment,
and leakage audit passes.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .partition_audit import FROZEN_QUERY_PARTITION_CONFIG

CONFIG_SCHEMA = "fractal-study-data-staging-config-v3"
INVENTORY_SCHEMA = "fractal-study-data-inventory-v2"
ONLINE_PROJECTION_SCHEMA = "fractal-online-staging-projection-v1"
ONLINE_PROJECTION_POLICY = "corpus-query-assignment-controls-only-v1"
ONLINE_PROJECTION_RECEIPT_FILENAME = "projection-receipt.json"
ASSIGNMENT_SCHEMA = "fractal-study-query-assignment-v1"
ASSIGNMENT_ALGORITHM = "component-ranked-sha256-v2"
PARTITION_EXCLUSION_SCHEMA = "fractal-study-query-partition-exclusion-v1"
PARTITION_EXCLUSION_RULE_ID = "source-split-component-isolation-v1"
PARTITION_EXCLUSION_REASON = "cross-source-split-component"
BRIGHT_DOMAINS = (
    "aops",
    "biology",
    "earth_science",
    "economics",
    "leetcode",
    "pony",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "theoremqa_questions",
    "theoremqa_theorems",
)
STAGES = ("fit", "calibration", "sealed")
CORPUS_SHARD_RECORDS = 100_000

_ONLINE_PROJECTION_ROLES = frozenset(
    {
        "assignments",
        "corpus",
        "corpus-shard",
        "queries",
        "query-partition-structural-exclusions",
        "registered-cohort-exclusions",
    }
)
_OUTCOME_PAYLOAD_ROLES = frozenset({"evidence-bundles", "qrels"})
_ONLINE_PROJECTION_RECEIPT_FIELDS = {
    "projected_artifact_count",
    "projected_artifact_set_sha256",
    "projected_artifacts",
    "projection_policy",
    "schema_version",
    "source_artifact_count",
    "source_inventory_sha256",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDERS = {"", "latest", "main", "master", "tbd", "todo", "unassigned"}
_SOURCE_FIELDS = {"path", "revision", "sha256"}
_CONFIG_FIELDS = {
    "assignment_seed",
    "datasets",
    "schema_version",
    "withhold_sealed_labels_from_online_process",
}
_DATASET_FIELDS = {
    "bright",
    "hotpotqa_fullwiki",
    "miracl_sw",
    "scifact",
    "t2_finqa",
}
_INVENTORY_FIELDS = {
    "artifacts",
    "assignment_algorithm",
    "assignment_seed_sha256",
    "bright_document_identity",
    "bright_domains",
    "config_sha256",
    "counts",
    "hotpotqa_fullwiki_scope",
    "schema_version",
    "sources",
    "withhold_sealed_labels_from_online_process",
}
_INVENTORY_ARTIFACT_FIELDS = {
    "byte_count",
    "dataset",
    "path",
    "record_count",
    "role",
    "sha256",
    "stage",
    "visibility",
}
_INVENTORY_SOURCE_FIELDS = {"byte_count", "revision", "sha256", "source_id"}
_CHUNK_BYTES = 1024 * 1024


class StudyDataError(ValueError):
    """Raised when staging cannot preserve the frozen data contract."""


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_parts(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _closed_mapping(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise StudyDataError(f"{label} must be an object with string field names")
    observed = set(value)
    if observed != fields:
        missing = sorted(fields - observed)
        unknown = sorted(observed - fields)
        raise StudyDataError(f"{label} fields differ; missing={missing}, unknown={unknown}")
    return value


def _decode_json(payload: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StudyDataError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise StudyDataError(f"{label} contains non-finite number {value!r}")

    try:
        text = payload.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise StudyDataError(f"{label} must be valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise StudyDataError(f"{label} is not valid JSON: {exc.msg}") from exc


def _require_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StudyDataError(f"{label} must be a non-empty canonical string")
    if unicodedata.normalize("NFC", value) != value:
        raise StudyDataError(f"{label} must use NFC Unicode normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise StudyDataError(f"{label} cannot contain control characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise StudyDataError(f"{label} must be valid UTF-8") from exc
    return value


def _source_identifier(value: object, *, label: str) -> str:
    if value is None or isinstance(value, bool):
        raise StudyDataError(f"{label} is required")
    return _require_identifier(str(value), label=label)


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudyDataError(f"{label} must be non-empty text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise StudyDataError(f"{label} must be valid UTF-8") from exc
    return value


def _safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise StudyDataError(f"{label} must be a non-empty relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or str(pure) != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise StudyDataError(f"{label} must be a canonical relative POSIX path")
    return value


@dataclass(frozen=True)
class SourcePin:
    """A local source file bound to an immutable upstream revision and digest."""

    source_id: str
    path: Path
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.source_id, label="source_id")
        revision = _require_identifier(self.revision, label=f"{self.source_id}.revision")
        if revision.casefold() in _PLACEHOLDERS:
            raise StudyDataError(f"{self.source_id}.revision cannot be a movable placeholder")
        if _SHA256.fullmatch(self.sha256) is None:
            raise StudyDataError(f"{self.source_id}.sha256 must be 64 lowercase hex characters")


@dataclass(frozen=True)
class ShardTreePin:
    """A directory of compressed shards bound by a canonical tree digest."""

    source_id: str
    path: Path
    revision: str
    sha256: str
    file_count: int

    def __post_init__(self) -> None:
        _require_identifier(self.source_id, label="shard tree source_id")
        revision = _require_identifier(self.revision, label=f"{self.source_id}.revision")
        if revision.casefold() in _PLACEHOLDERS:
            raise StudyDataError(f"{self.source_id}.revision cannot be a movable placeholder")
        if _SHA256.fullmatch(self.sha256) is None:
            raise StudyDataError(f"{self.source_id}.sha256 must be 64 lowercase hex characters")
        if (
            isinstance(self.file_count, bool)
            or not isinstance(self.file_count, int)
            or self.file_count <= 0
        ):
            raise StudyDataError(f"{self.source_id}.file_count must be a positive integer")


@dataclass(frozen=True)
class StagingConfig:
    """Closed, path-resolved staging configuration."""

    assignment_seed: str
    withhold_sealed_labels_from_online_process: bool
    pins: Mapping[str, SourcePin]
    bright_domains: tuple[str, ...]
    bright_document_id_collision_policy: str
    hotpotqa_shards: ShardTreePin
    hotpotqa_train_source_ids: tuple[str, ...]
    hotpotqa_expected_document_count: int

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.assignment_seed) is None:
            raise StudyDataError("assignment_seed must be 64 lowercase hex characters")
        if not isinstance(self.withhold_sealed_labels_from_online_process, bool):
            raise StudyDataError("withhold_sealed_labels_from_online_process must be boolean")
        if (
            isinstance(self.hotpotqa_expected_document_count, bool)
            or not isinstance(self.hotpotqa_expected_document_count, int)
            or self.hotpotqa_expected_document_count <= 0
        ):
            raise StudyDataError("HotpotQA expected_document_count must be positive")
        pins = dict(self.pins)
        if not pins or set(pins) != {pin.source_id for pin in pins.values()}:
            raise StudyDataError("source pin keys must match unique source IDs")
        object.__setattr__(self, "pins", pins)
        domains = tuple(self.bright_domains)
        if not domains or len(domains) != len(set(domains)):
            raise StudyDataError("bright_domains must contain unique domain identifiers")
        for domain in domains:
            if re.fullmatch(r"[a-z0-9](?:[a-z0-9_]*[a-z0-9])?", domain) is None:
                raise StudyDataError(f"invalid BRIGHT domain identifier {domain!r}")
        if domains != tuple(sorted(domains, key=lambda item: item.encode())):
            raise StudyDataError("bright_domains must be sorted by UTF-8 identifier")
        if domains != BRIGHT_DOMAINS:
            missing = sorted(set(BRIGHT_DOMAINS) - set(domains))
            unexpected = sorted(set(domains) - set(BRIGHT_DOMAINS))
            raise StudyDataError(
                "BRIGHT confirmatory staging requires the registered 12-domain set; "
                f"missing={missing}, unexpected={unexpected}"
            )
        object.__setattr__(self, "bright_domains", domains)
        if self.bright_document_id_collision_policy not in {"error", "domain-scoped"}:
            raise StudyDataError(
                "bright_document_id_collision_policy must be 'error' or 'domain-scoped'"
            )
        if self.hotpotqa_shards.source_id in pins:
            raise StudyDataError("HotpotQA shard-tree source ID collides with a file source")
        train_source_ids = tuple(self.hotpotqa_train_source_ids)
        if len(train_source_ids) != 2 or len(set(train_source_ids)) != 2:
            raise StudyDataError("HotpotQA training release must bind exactly two unique shards")
        if any(source_id not in pins for source_id in train_source_ids):
            raise StudyDataError("HotpotQA training shard IDs must name pinned file sources")
        object.__setattr__(self, "hotpotqa_train_source_ids", train_source_ids)

    def pin(self, source_id: str) -> SourcePin:
        try:
            return self.pins[source_id]
        except KeyError as exc:
            raise StudyDataError(f"configuration is missing source {source_id!r}") from exc

    @property
    def source_bindings(self) -> tuple[tuple[str, str, str], ...]:
        rows = [(pin.source_id, pin.revision, pin.sha256) for pin in self.pins.values()]
        rows.append(
            (
                self.hotpotqa_shards.source_id,
                self.hotpotqa_shards.revision,
                self.hotpotqa_shards.sha256,
            )
        )
        return tuple(sorted(rows, key=lambda row: row[0].encode()))

    @property
    def digest(self) -> str:
        payload = {
            "assignment_seed_sha256": self.assignment_seed,
            "bright_document_id_collision_policy": self.bright_document_id_collision_policy,
            "bright_domains": list(self.bright_domains),
            "hotpotqa_fullwiki_scope": {
                "expected_document_count": self.hotpotqa_expected_document_count,
                "name": "fullwiki",
                "sampling": "none",
            },
            "schema_version": CONFIG_SCHEMA,
            "sources": [
                {
                    "revision": revision,
                    "sha256": sha256,
                    "source_id": source_id,
                }
                for source_id, revision, sha256 in self.source_bindings
            ],
            "withhold_sealed_labels_from_online_process": (
                self.withhold_sealed_labels_from_online_process
            ),
        }
        return _sha256_bytes(_canonical_bytes(payload))


def _parse_pin(
    source_id: str,
    value: object,
    *,
    base_directory: Path,
) -> SourcePin:
    row = _closed_mapping(value, fields=_SOURCE_FIELDS, label=source_id)
    path_value = row["path"]
    if not isinstance(path_value, str) or not path_value:
        raise StudyDataError(f"{source_id}.path must be a non-empty path string")
    path = Path(path_value)
    if not path.is_absolute():
        path = base_directory / path
    return SourcePin(
        source_id=source_id,
        path=path,
        revision=row["revision"],
        sha256=row["sha256"],
    )


def _parse_shard_tree_pin(
    source_id: str,
    value: object,
    *,
    base_directory: Path,
) -> ShardTreePin:
    row = _closed_mapping(
        value,
        fields={"file_count", "path", "revision", "sha256"},
        label=source_id,
    )
    path_value = row["path"]
    if not isinstance(path_value, str) or not path_value:
        raise StudyDataError(f"{source_id}.path must be a non-empty path string")
    path = Path(path_value)
    if not path.is_absolute():
        path = base_directory / path
    return ShardTreePin(
        source_id=source_id,
        path=path,
        revision=row["revision"],
        sha256=row["sha256"],
        file_count=row["file_count"],
    )


def load_staging_config(path: str | Path) -> StagingConfig:
    """Load the closed JSON configuration and resolve all source paths."""

    config_path = Path(path)
    try:
        payload = _decode_json(config_path.read_bytes(), label="staging configuration")
    except OSError as exc:
        raise StudyDataError(f"cannot read staging configuration {config_path}: {exc}") from exc
    root = _closed_mapping(payload, fields=_CONFIG_FIELDS, label="staging configuration")
    if root["schema_version"] != CONFIG_SCHEMA:
        raise StudyDataError(f"schema_version must equal {CONFIG_SCHEMA!r}")
    datasets = _closed_mapping(root["datasets"], fields=_DATASET_FIELDS, label="datasets")
    base = config_path.parent
    pins: dict[str, SourcePin] = {}

    def add(section: Mapping[str, Any], dataset: str, field_name: str) -> None:
        source_id = f"{dataset}/{field_name}"
        pins[source_id] = _parse_pin(source_id, section[field_name], base_directory=base)

    scifact = _closed_mapping(
        datasets["scifact"],
        fields={"corpus", "dev_claims", "train_claims"},
        label="datasets.scifact",
    )
    for field_name in ("corpus", "train_claims", "dev_claims"):
        add(scifact, "scifact", field_name)

    t2 = _closed_mapping(
        datasets["t2_finqa"],
        fields={"dev", "test", "train"},
        label="datasets.t2_finqa",
    )
    for field_name in ("train", "dev", "test"):
        add(t2, "t2_finqa", field_name)

    miracl = _closed_mapping(
        datasets["miracl_sw"],
        fields={"dev_qrels", "dev_queries", "documents", "train_qrels", "train_queries"},
        label="datasets.miracl_sw",
    )
    for field_name in ("documents", "train_queries", "train_qrels", "dev_queries", "dev_qrels"):
        add(miracl, "miracl_sw", field_name)

    bright = _closed_mapping(
        datasets["bright"],
        fields={"document_id_collision_policy", "domain_order", "domains"},
        label="datasets.bright",
    )
    domain_order_value = bright["domain_order"]
    if (
        not isinstance(domain_order_value, list)
        or not domain_order_value
        or not all(isinstance(domain, str) for domain in domain_order_value)
    ):
        raise StudyDataError("datasets.bright.domain_order must be a non-empty string array")
    bright_domains = tuple(domain_order_value)
    domains = _closed_mapping(
        bright["domains"], fields=set(bright_domains), label="datasets.bright.domains"
    )
    for domain in bright_domains:
        section = domains[domain]
        if not isinstance(section, Mapping) or not all(isinstance(key, str) for key in section):
            raise StudyDataError(f"datasets.bright.domains.{domain} must be an object")
        observed = set(section)
        if observed == {"documents", "examples"}:
            field_names = ("documents", "examples")
        elif observed == {"documents", "qrels", "queries"}:
            field_names = ("documents", "queries", "qrels")
        else:
            raise StudyDataError(
                f"datasets.bright.domains.{domain} must declare either official "
                "documents+examples or adapter documents+queries+qrels"
            )
        for field_name in field_names:
            add(section, f"bright/{domain}", field_name)

    hotpot = _closed_mapping(
        datasets["hotpotqa_fullwiki"],
        fields={
            "corpus_archive",
            "corpus_scope",
            "corpus_shards",
            "dev_questions",
            "train_questions",
        },
        label="datasets.hotpotqa_fullwiki",
    )
    scope = _closed_mapping(
        hotpot["corpus_scope"],
        fields={"expected_document_count", "name", "sampling"},
        label="datasets.hotpotqa_fullwiki.corpus_scope",
    )
    if scope["name"] != "fullwiki" or scope["sampling"] != "none":
        raise StudyDataError(
            "HotpotQA sealed staging requires name='fullwiki' and sampling='none'; "
            "a sampled context or corpus cannot be relabeled as FullWiki"
        )
    for field_name in ("corpus_archive", "dev_questions"):
        add(hotpot, "hotpotqa_fullwiki", field_name)
    train_questions = hotpot["train_questions"]
    if not isinstance(train_questions, list) or len(train_questions) != 2:
        raise StudyDataError(
            "datasets.hotpotqa_fullwiki.train_questions must contain exactly two shard pins"
        )
    hotpotqa_train_source_ids: list[str] = []
    for shard_index, value in enumerate(train_questions):
        source_id = f"hotpotqa_fullwiki/train_questions/{shard_index:05d}"
        pins[source_id] = _parse_pin(source_id, value, base_directory=base)
        hotpotqa_train_source_ids.append(source_id)
    hotpotqa_shards = _parse_shard_tree_pin(
        "hotpotqa_fullwiki/corpus_shards",
        hotpot["corpus_shards"],
        base_directory=base,
    )

    return StagingConfig(
        assignment_seed=root["assignment_seed"],
        withhold_sealed_labels_from_online_process=(
            root["withhold_sealed_labels_from_online_process"]
        ),
        pins=pins,
        bright_domains=bright_domains,
        bright_document_id_collision_policy=bright["document_id_collision_policy"],
        hotpotqa_shards=hotpotqa_shards,
        hotpotqa_train_source_ids=tuple(hotpotqa_train_source_ids),
        hotpotqa_expected_document_count=scope["expected_document_count"],
    )


def _open_regular(path: Path) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StudyDataError(f"cannot open pinned source {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise StudyDataError(f"pinned source {path} must be a regular file")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _read_verified_bytes(pin: SourcePin) -> tuple[bytes, int]:
    descriptor = _open_regular(pin.path)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
            total += len(chunk)
    observed = digest.hexdigest()
    if observed != pin.sha256:
        raise StudyDataError(
            f"source {pin.source_id!r} SHA-256 mismatch: expected {pin.sha256}, observed {observed}"
        )
    return b"".join(chunks), total


def _verify_file_pin(pin: SourcePin) -> int:
    descriptor = _open_regular(pin.path)
    digest = hashlib.sha256()
    byte_count = 0
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    observed = digest.hexdigest()
    if observed != pin.sha256:
        raise StudyDataError(
            f"source {pin.source_id!r} SHA-256 mismatch: expected {pin.sha256}, observed {observed}"
        )
    return byte_count


def _consume_parquet_records(
    pin: SourcePin,
    consumer: Any,
) -> int:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise StudyDataError(
            "Parquet staging requires the declared 'benchmarks' extra: "
            "install fractal-ann-diagnostics[benchmarks]"
        ) from exc
    descriptor = _open_regular(pin.path)
    digest = hashlib.sha256()
    byte_count = 0
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        observed = digest.hexdigest()
        if observed != pin.sha256:
            raise StudyDataError(
                f"source {pin.source_id!r} SHA-256 mismatch: expected {pin.sha256}, "
                f"observed {observed}"
            )
        handle.seek(0)
        try:
            source = parquet.ParquetFile(handle)
            for batch in source.iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    if not isinstance(row, Mapping):
                        raise StudyDataError(
                            f"Parquet source {pin.source_id!r} emitted a non-object record"
                        )
                    consumer(row)
        except StudyDataError:
            raise
        except Exception as exc:
            raise StudyDataError(f"cannot decode Parquet source {pin.source_id!r}: {exc}") from exc
    return byte_count


def _load_parquet_records(pin: SourcePin) -> tuple[list[Mapping[str, Any]], int]:
    rows: list[Mapping[str, Any]] = []
    byte_count = _consume_parquet_records(pin, rows.append)
    if not rows:
        raise StudyDataError(f"source {pin.source_id!r} contains no records")
    return rows, byte_count


def _load_records(pin: SourcePin) -> tuple[list[Mapping[str, Any]], int]:
    if pin.path.suffix.casefold() == ".parquet":
        return _load_parquet_records(pin)
    encoded, byte_count = _read_verified_bytes(pin)
    if pin.path.suffix.casefold() == ".gz":
        try:
            encoded = gzip.decompress(encoded)
        except (OSError, EOFError) as exc:
            raise StudyDataError(f"cannot decompress source {pin.source_id!r}: {exc}") from exc
    stripped = encoded.lstrip()
    if not stripped:
        raise StudyDataError(f"source {pin.source_id!r} is empty")
    rows: list[Mapping[str, Any]] = []
    if stripped.startswith(b"["):
        parsed = _decode_json(encoded, label=pin.source_id)
        if not isinstance(parsed, list):
            raise StudyDataError(f"source {pin.source_id!r} must contain an array")
        candidates = parsed
    else:
        candidates = []
        for line_number, line in enumerate(encoded.splitlines(), start=1):
            if not line.strip():
                continue
            candidates.append(_decode_json(line, label=f"{pin.source_id} line {line_number}"))
    for position, row in enumerate(candidates):
        if not isinstance(row, Mapping) or not all(isinstance(key, str) for key in row):
            raise StudyDataError(f"{pin.source_id} record {position} must be a JSON object")
        rows.append(row)
    if not rows:
        raise StudyDataError(f"source {pin.source_id!r} contains no records")
    return rows, byte_count


@dataclass(frozen=True)
class _Document:
    identifier: str
    title: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.identifier, "text": self.text, "title": self.title}


@dataclass
class _Query:
    identifier: str
    text: str
    source_split: str
    domain: str | None = None
    stage: str | None = None
    component_sha256: str | None = None
    assignment_key_sha256: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {"id": self.identifier, "text": self.text}


@dataclass(frozen=True)
class _Qrel:
    query_id: str
    document_id: str
    relevance: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "document_id": self.document_id,
            "query_id": self.query_id,
            "relevance": self.relevance,
        }


@dataclass(frozen=True)
class _EvidenceLocation:
    document_id: str
    locator: str

    def __post_init__(self) -> None:
        _require_identifier(self.document_id, label="evidence document_id")
        _require_identifier(self.locator, label="evidence locator")

    def to_dict(self) -> dict[str, str]:
        return {"document_id": self.document_id, "locator": self.locator}


@dataclass(frozen=True)
class _EvidenceBundle:
    bundle_id: str
    locations: tuple[_EvidenceLocation, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.bundle_id, label="evidence bundle_id")
        locations = tuple(
            sorted(
                self.locations,
                key=lambda item: (item.document_id.encode(), item.locator.encode()),
            )
        )
        if not locations or not all(isinstance(item, _EvidenceLocation) for item in locations):
            raise StudyDataError("evidence bundle must contain typed locations")
        if len(locations) != len(set(locations)):
            raise StudyDataError("evidence bundle cannot repeat a location")
        object.__setattr__(self, "locations", locations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "locations": [location.to_dict() for location in self.locations],
        }


@dataclass(frozen=True)
class _EvidenceLabelRow:
    query_id: str
    answer: str | None
    evidence_bundles: tuple[_EvidenceBundle, ...]
    label_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.query_id, label="evidence-label query_id")
        if self.answer is not None and not isinstance(self.answer, str):
            raise StudyDataError("evidence-label answer must be a string or null")
        bundles = tuple(sorted(self.evidence_bundles, key=lambda item: item.bundle_id.encode()))
        if not bundles or not all(isinstance(item, _EvidenceBundle) for item in bundles):
            raise StudyDataError("evidence-label row must contain typed bundles")
        bundle_ids = [bundle.bundle_id for bundle in bundles]
        if len(bundle_ids) != len(set(bundle_ids)):
            raise StudyDataError("evidence-label row cannot repeat a bundle ID")
        metadata = tuple(sorted(self.label_metadata, key=lambda item: item[0].encode()))
        for key, value in metadata:
            _require_identifier(key, label="evidence-label metadata key")
            _require_identifier(value, label=f"evidence-label metadata {key}")
        if len({key for key, _ in metadata}) != len(metadata):
            raise StudyDataError("evidence-label metadata keys must be unique")
        object.__setattr__(self, "evidence_bundles", bundles)
        object.__setattr__(self, "label_metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence_bundles": [bundle.to_dict() for bundle in self.evidence_bundles],
            "label_metadata": [list(item) for item in self.label_metadata],
            "query_id": self.query_id,
        }


@dataclass(frozen=True)
class _Exclusion:
    dataset: str
    query_id: str
    source_split: str
    rule_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "query_id": self.query_id,
            "reason": self.reason,
            "rule_id": self.rule_id,
            "source_split": self.source_split,
        }


@dataclass(frozen=True)
class _PartitionExclusion:
    dataset: str
    query_id: str
    source_split: str
    rule_id: str
    reason: str
    partition_component_sha256: str
    query_text_sha256: str
    normalized_query_text_sha256: str
    positive_relevance_identity_sha256s: tuple[str, ...]
    schema_version: str = PARTITION_EXCLUSION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "normalized_query_text_sha256": self.normalized_query_text_sha256,
            "partition_component_sha256": self.partition_component_sha256,
            "positive_relevance_identity_sha256s": list(self.positive_relevance_identity_sha256s),
            "query_id": self.query_id,
            "query_text_sha256": self.query_text_sha256,
            "reason": self.reason,
            "rule_id": self.rule_id,
            "schema_version": self.schema_version,
            "source_split": self.source_split,
        }


@dataclass
class _Dataset:
    name: str
    documents: dict[str, _Document] = field(default_factory=dict)
    queries: dict[str, _Query] = field(default_factory=dict)
    qrels: dict[tuple[str, str], _Qrel] = field(default_factory=dict)
    evidence_labels: dict[str, _EvidenceLabelRow] = field(default_factory=dict)
    content_ids: dict[str, str] = field(default_factory=dict)
    exclusions: list[_Exclusion] = field(default_factory=list)
    partition_exclusions: list[_PartitionExclusion] = field(default_factory=list)
    duplicate_content_aliases: int = 0
    duplicate_positive_judgments: int = 0
    empty_document_text_fallbacks: int = 0
    streamed_document_count: int | None = None

    def add_document(
        self,
        document: _Document,
        *,
        repeated_identical_ok: bool = False,
        duplicate_content_ok: bool = False,
    ) -> None:
        previous = self.documents.get(document.identifier)
        if previous is not None:
            if repeated_identical_ok and previous == document:
                return
            raise StudyDataError(
                f"dataset {self.name!r} contains duplicate document ID {document.identifier!r}"
            )
        content_sha256 = _hash_parts(document.title, document.text)
        prior_id = self.content_ids.get(content_sha256)
        if prior_id is not None and prior_id != document.identifier:
            if not duplicate_content_ok:
                raise StudyDataError(
                    f"dataset {self.name!r} maps duplicate document content to IDs "
                    f"{prior_id!r} and {document.identifier!r}"
                )
            self.duplicate_content_aliases += 1
        self.documents[document.identifier] = document
        self.content_ids.setdefault(content_sha256, document.identifier)

    def add_query(self, query: _Query) -> None:
        if query.identifier in self.queries:
            raise StudyDataError(
                f"dataset {self.name!r} contains duplicate query ID {query.identifier!r}"
            )
        self.queries[query.identifier] = query

    def add_qrel(self, qrel: _Qrel, *, derived_duplicate_ok: bool = False) -> None:
        key = (qrel.query_id, qrel.document_id)
        previous = self.qrels.get(key)
        if previous is not None:
            if derived_duplicate_ok and previous == qrel:
                self.duplicate_positive_judgments += 1
                return
            raise StudyDataError(f"dataset {self.name!r} contains duplicate qrel pair {key!r}")
        self.qrels[key] = qrel

    def add_evidence_label(self, row: _EvidenceLabelRow) -> None:
        if row.query_id not in self.queries:
            raise StudyDataError(
                f"dataset {self.name!r} evidence labels name unknown query {row.query_id!r}"
            )
        if row.query_id in self.evidence_labels:
            raise StudyDataError(
                f"dataset {self.name!r} repeats evidence labels for {row.query_id!r}"
            )
        self.evidence_labels[row.query_id] = row


def _generic_document(row: Mapping[str, Any], *, label: str) -> _Document:
    identifier_value = row.get("id", row.get("docid", row.get("_id", row.get("title"))))
    identifier = _source_identifier(identifier_value, label=f"{label}.id")
    title_value = row.get("title", identifier)
    text_value = row.get("text", row.get("contents", row.get("content", row.get("sentences"))))
    if isinstance(text_value, list) and all(isinstance(item, str) for item in text_value):
        text_value = "\n".join(text_value)
    title = _require_text(title_value, label=f"{label}.title")
    text = _require_text(text_value, label=f"{label}.text")
    return _Document(identifier=identifier, title=title, text=text)


def _generic_query(row: Mapping[str, Any], *, label: str) -> tuple[str, str]:
    identifier = _source_identifier(
        row.get("id", row.get("query_id", row.get("qid"))),
        label=f"{label}.id",
    )
    text = _require_text(row.get("text", row.get("query")), label=f"{label}.text")
    return identifier, text


def _generic_qrel(row: Mapping[str, Any], *, label: str) -> tuple[str, str, int]:
    query_id = _source_identifier(row.get("query_id", row.get("qid")), label=f"{label}.query_id")
    document_id = _source_identifier(
        row.get("document_id", row.get("docid")), label=f"{label}.document_id"
    )
    score = row.get("relevance", row.get("score"))
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or float(score) != int(score)
    ):
        raise StudyDataError(f"{label}.relevance must be a finite integer-valued number")
    return query_id, document_id, int(score)


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parents = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parents[value]
        while parent != self.parents[parent]:
            parent = self.parents[parent]
        while value != parent:
            next_value = self.parents[value]
            self.parents[value] = parent
            value = next_value
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root), key=lambda item: item.encode())
        self.parents[second] = first


def _normalize_query_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    return " ".join(tokens) if tokens else " ".join(normalized.split())


def _positive_relevance_identities(
    dataset: _Dataset,
    query_id: str,
) -> tuple[str, ...]:
    identities: set[str] = set()
    for qrel in dataset.qrels.values():
        if qrel.relevance <= 0 or qrel.query_id != query_id:
            continue
        identities.add(_sha256_bytes(_canonical_bytes([dataset.name, qrel.document_id])))
        document = dataset.documents.get(qrel.document_id)
        if document is not None:
            identities.add(
                _sha256_bytes(
                    _canonical_bytes(
                        [
                            "suite-global-canonical-document-content-v2",
                            _hash_parts(document.title, document.text),
                        ]
                    )
                )
            )
    return tuple(sorted(identities, key=lambda item: item.encode()))


def _near_duplicate_query_pairs(
    dataset: _Dataset,
    query_ids: set[str],
) -> set[tuple[str, str]]:
    config = FROZEN_QUERY_PARTITION_CONFIG
    token_rows = {
        query_id: tuple(_normalize_query_text(dataset.queries[query_id].text).split())
        for query_id in query_ids
    }
    eligible = {
        query_id: tokens
        for query_id, tokens in token_rows.items()
        if len(tokens) >= config.minimum_near_duplicate_tokens
    }
    exact_tokens: dict[tuple[str, ...], list[str]] = {}
    substitutions: dict[tuple[int, int, tuple[str, ...]], dict[str, list[str]]] = {}
    for query_id, tokens in eligible.items():
        exact_tokens.setdefault(tokens, []).append(query_id)
        for position, token in enumerate(tokens):
            deleted = tokens[:position] + tokens[position + 1 :]
            variants = substitutions.setdefault(
                (len(tokens), position, deleted),
                {},
            )
            variants.setdefault(token, []).append(query_id)
    pairs: set[tuple[str, str]] = set()
    for _, variants in sorted(substitutions.items(), key=lambda item: repr(item[0])):
        if len(variants) < 2:
            continue
        representatives = sorted(min(query_ids) for query_ids in variants.values())
        anchor = representatives[0]
        for other in representatives[1:]:
            pairs.add(tuple(sorted((anchor, other), key=lambda item: item.encode())))
    numerator = config.minimum_length_ratio_numerator
    denominator = config.minimum_length_ratio_denominator
    for query_id, tokens in eligible.items():
        shorter_length = len(tokens) - 1
        if (
            shorter_length < config.minimum_near_duplicate_tokens
            or shorter_length * denominator < len(tokens) * numerator
        ):
            continue
        seen_deletions: set[tuple[str, ...]] = set()
        for position in range(len(tokens)):
            deleted = tokens[:position] + tokens[position + 1 :]
            if deleted in seen_deletions:
                continue
            seen_deletions.add(deleted)
            shorter_nodes = exact_tokens.get(deleted)
            if shorter_nodes:
                pairs.add(
                    tuple(
                        sorted(
                            (query_id, min(shorter_nodes)),
                            key=lambda item: item.encode(),
                        )
                    )
                )
    return pairs


def _query_components(dataset: _Dataset, query_ids: Sequence[str]) -> list[tuple[str, ...]]:
    selected = set(query_ids)
    disjoint = _DisjointSet(selected)
    relevant_to_queries: dict[str, list[str]] = {}
    normalized_to_queries: dict[str, list[str]] = {}
    for query_id in selected:
        normalized_to_queries.setdefault(
            _normalize_query_text(dataset.queries[query_id].text), []
        ).append(query_id)
    for query_id in selected:
        for relevance_identity in _positive_relevance_identities(dataset, query_id):
            relevant_to_queries.setdefault(relevance_identity, []).append(query_id)
    for groups in (relevant_to_queries.values(), normalized_to_queries.values()):
        for group in groups:
            ordered = sorted(set(group), key=lambda item: item.encode())
            for other in ordered[1:]:
                disjoint.union(ordered[0], other)
    for left, right in _near_duplicate_query_pairs(dataset, selected):
        disjoint.union(left, right)
    components: dict[str, list[str]] = {}
    for query_id in selected:
        components.setdefault(disjoint.find(query_id), []).append(query_id)
    return [
        tuple(sorted(component, key=lambda item: item.encode()))
        for component in sorted(components.values(), key=lambda values: min(values).encode())
    ]


def _exclude_cross_source_split_components(
    dataset: _Dataset,
    query_ids: Sequence[str],
) -> tuple[str, ...]:
    """Remove whole leakage components while preserving official split meanings."""

    excluded: set[str] = set()
    for component in _query_components(dataset, query_ids):
        source_splits = {dataset.queries[query_id].source_split for query_id in component}
        if len(source_splits) <= 1:
            continue
        component_sha256 = _sha256_bytes(_canonical_bytes(list(component)))
        for query_id in component:
            query = dataset.queries[query_id]
            identities = _positive_relevance_identities(dataset, query_id)
            dataset.partition_exclusions.append(
                _PartitionExclusion(
                    dataset=dataset.name,
                    query_id=query_id,
                    source_split=query.source_split,
                    rule_id=PARTITION_EXCLUSION_RULE_ID,
                    reason=PARTITION_EXCLUSION_REASON,
                    partition_component_sha256=component_sha256,
                    query_text_sha256=_sha256_bytes(query.text.encode("utf-8")),
                    normalized_query_text_sha256=_sha256_bytes(
                        _normalize_query_text(query.text).encode("utf-8")
                    ),
                    positive_relevance_identity_sha256s=tuple(identities),
                )
            )
            excluded.add(query_id)
    if not excluded:
        return tuple(query_ids)
    for query_id in excluded:
        del dataset.queries[query_id]
        dataset.evidence_labels.pop(query_id, None)
    dataset.qrels = {
        key: qrel for key, qrel in dataset.qrels.items() if qrel.query_id not in excluded
    }
    return tuple(query_id for query_id in query_ids if query_id not in excluded)


def _assign_ranked_components(
    dataset: _Dataset,
    query_ids: Sequence[str],
    *,
    assignment_seed: str,
    stages: Sequence[str],
    domain: str | None = None,
) -> None:
    components = _query_components(dataset, query_ids)
    if len(components) < len(stages):
        raise StudyDataError(
            f"dataset {dataset.name!r} partition {domain or 'default'!r} has "
            f"{len(components)} independent query components but needs {len(stages)} stages"
        )
    ranked: list[tuple[str, tuple[str, ...], str]] = []
    for component in components:
        component_sha256 = _sha256_bytes(_canonical_bytes(list(component)))
        key = _hash_parts(
            ASSIGNMENT_ALGORITHM,
            assignment_seed,
            dataset.name,
            domain or "",
            component_sha256,
        )
        ranked.append((key, component, component_sha256))
    ranked.sort(key=lambda item: (item[0], tuple(value.encode() for value in item[1])))

    if tuple(stages) == ("fit", "calibration"):
        calibration_count = max(1, len(ranked) // 5)
        stage_by_rank = ["calibration"] * calibration_count + ["fit"] * (
            len(ranked) - calibration_count
        )
    elif tuple(stages) == STAGES:
        calibration_count = max(1, len(ranked) // 5)
        sealed_count = max(1, len(ranked) // 5)
        fit_count = len(ranked) - calibration_count - sealed_count
        if fit_count < 1:
            raise StudyDataError("three-way assignment needs at least one fit component")
        stage_by_rank = (
            ["calibration"] * calibration_count + ["sealed"] * sealed_count + ["fit"] * fit_count
        )
    else:
        raise StudyDataError(f"unsupported ranked stage allocation {tuple(stages)!r}")

    for (assignment_key, component, component_sha256), stage in zip(
        ranked, stage_by_rank, strict=True
    ):
        for query_id in component:
            query = dataset.queries[query_id]
            if query.stage is not None:
                raise StudyDataError(f"query {query_id!r} was assigned more than once")
            query.stage = stage
            query.component_sha256 = component_sha256
            query.assignment_key_sha256 = assignment_key


def _assign_fixed(
    dataset: _Dataset,
    query_ids: Iterable[str],
    *,
    stage: str,
    assignment_seed: str,
) -> None:
    selected = tuple(query_ids)
    for component in _query_components(dataset, selected):
        component_sha256 = _sha256_bytes(_canonical_bytes(list(component)))
        assignment_key_sha256 = _hash_parts(
            "source-split-fixed-components-v2",
            assignment_seed,
            dataset.name,
            component_sha256,
            stage,
        )
        for query_id in component:
            query = dataset.queries[query_id]
            if query.stage is not None:
                raise StudyDataError(f"query {query_id!r} was assigned more than once")
            query.stage = stage
            query.component_sha256 = component_sha256
            query.assignment_key_sha256 = assignment_key_sha256


def _record_source_use(
    source_receipts: dict[str, int],
    pin: SourcePin | ShardTreePin,
    byte_count: int,
) -> None:
    if pin.source_id in source_receipts:
        raise StudyDataError(f"source {pin.source_id!r} was consumed more than once")
    source_receipts[pin.source_id] = byte_count


def _load_scifact(
    config: StagingConfig,
    source_receipts: dict[str, int],
) -> _Dataset:
    dataset = _Dataset("scifact")
    corpus_pin = config.pin("scifact/corpus")
    corpus_rows, byte_count = _load_records(corpus_pin)
    _record_source_use(source_receipts, corpus_pin, byte_count)
    sentence_counts: dict[str, int] = {}
    for position, row in enumerate(corpus_rows):
        document_id = _source_identifier(
            row.get("doc_id"), label=f"scifact corpus record {position}.doc_id"
        )
        title = _require_text(row.get("title"), label=f"SciFact document {document_id}.title")
        abstract = row.get("abstract")
        if (
            not isinstance(abstract, list)
            or not abstract
            or not all(isinstance(sentence, str) and sentence for sentence in abstract)
        ):
            raise StudyDataError(
                f"SciFact document {document_id!r} needs a non-empty abstract string list"
            )
        dataset.add_document(
            _Document(identifier=document_id, title=title, text="\n".join(abstract))
        )
        sentence_counts[document_id] = len(abstract)

    split_query_ids: dict[str, list[str]] = {"train": [], "dev": []}
    for split in ("train", "dev"):
        pin = config.pin(f"scifact/{split}_claims")
        rows, byte_count = _load_records(pin)
        _record_source_use(source_receipts, pin, byte_count)
        for position, row in enumerate(rows):
            raw_id = _source_identifier(row.get("id"), label=f"SciFact {split} claim {position}.id")
            query_id = f"scifact:{raw_id}"
            claim = _require_text(row.get("claim"), label=f"SciFact claim {query_id}.claim")
            evidence = row.get("evidence")
            if not isinstance(evidence, Mapping):
                raise StudyDataError(f"SciFact claim {query_id!r} needs an evidence object")
            if not all(isinstance(rationales, list) for rationales in evidence.values()):
                raise StudyDataError(
                    f"SciFact claim {query_id!r} evidence must contain rationale lists"
                )
            if not any(rationales for rationales in evidence.values()):
                dataset.exclusions.append(
                    _Exclusion(
                        dataset=dataset.name,
                        query_id=query_id,
                        source_split=split,
                        rule_id="scifact-evidence-bearing-v1",
                        reason="no-nonempty-evidence-rationale-list",
                    )
                )
                continue
            dataset.add_query(_Query(identifier=query_id, text=claim, source_split=split))
            split_query_ids[split].append(query_id)
            evidence_bundles: list[_EvidenceBundle] = []
            evidence_labels: set[str] = set()
            for raw_document_id, rationales in sorted(
                evidence.items(), key=lambda item: str(item[0]).encode()
            ):
                document_id = _source_identifier(
                    raw_document_id,
                    label=f"SciFact claim {query_id}.evidence document ID",
                )
                if document_id not in dataset.documents:
                    raise StudyDataError(
                        f"SciFact claim {query_id!r} names unknown document {document_id!r}"
                    )
                for rationale_index, rationale in enumerate(rationales):
                    if not isinstance(rationale, Mapping):
                        raise StudyDataError(
                            f"SciFact claim {query_id!r} rationale {rationale_index} is invalid"
                        )
                    evidence_label = rationale.get("label")
                    if evidence_label not in {"SUPPORT", "CONTRADICT"}:
                        raise StudyDataError(
                            f"SciFact claim {query_id!r} has an unknown evidence label"
                        )
                    evidence_labels.add(str(evidence_label))
                    sentences = rationale.get("sentences")
                    if (
                        not isinstance(sentences, list)
                        or not sentences
                        or not all(
                            type(sentence) is int and 0 <= sentence < sentence_counts[document_id]
                            for sentence in sentences
                        )
                    ):
                        raise StudyDataError(
                            f"SciFact claim {query_id!r} has invalid rationale sentence IDs"
                        )
                    evidence_bundles.append(
                        _EvidenceBundle(
                            bundle_id=f"document-{document_id}-rationale-{rationale_index}",
                            locations=tuple(
                                _EvidenceLocation(
                                    document_id=document_id,
                                    locator=f"sentence:{sentence}",
                                )
                                for sentence in sentences
                            ),
                        )
                    )
                if rationales:
                    dataset.add_qrel(
                        _Qrel(query_id=query_id, document_id=document_id, relevance=1),
                        derived_duplicate_ok=True,
                    )
            dataset.add_evidence_label(
                _EvidenceLabelRow(
                    query_id=query_id,
                    answer=None,
                    evidence_bundles=tuple(evidence_bundles),
                    label_metadata=(("evidence_labels", ",".join(sorted(evidence_labels))),),
                )
            )

    _assign_ranked_components(
        dataset,
        (*split_query_ids["train"], *split_query_ids["dev"]),
        assignment_seed=config.assignment_seed,
        stages=STAGES,
    )
    return dataset


def _load_t2_finqa(
    config: StagingConfig,
    source_receipts: dict[str, int],
) -> _Dataset:
    dataset = _Dataset("t2-ragbench")
    stage_for_split = {"train": "fit", "dev": "calibration", "test": "sealed"}
    split_query_ids: dict[str, list[str]] = {split: [] for split in stage_for_split}
    for split in ("train", "dev", "test"):
        pin = config.pin(f"t2_finqa/{split}")
        rows, byte_count = _load_records(pin)
        _record_source_use(source_receipts, pin, byte_count)
        for position, row in enumerate(rows):
            observed_subset = row.get("subset")
            if observed_subset is not None and observed_subset != "FinQA":
                raise StudyDataError(f"T2 {split} record {position} declares a non-FinQA subset")
            if row.get("split") != split:
                raise StudyDataError(
                    f"T2 {split} record {position} declares split={row.get('split')!r}"
                )
            raw_query_id = _source_identifier(
                row.get("id"), label=f"T2 {split} record {position}.id"
            )
            context_id = _source_identifier(
                row.get("context_id"),
                label=f"T2 query {raw_query_id}.context_id",
            )
            question_value = row.get("question")
            if question_value == "":
                dataset.exclusions.append(
                    _Exclusion(
                        dataset=dataset.name,
                        query_id=f"t2-ragbench:{raw_query_id}",
                        source_split=split,
                        rule_id="t2-finqa-nonempty-question-v1",
                        reason="empty-question",
                    )
                )
                continue
            question = _require_text(
                question_value,
                label=f"T2 query {raw_query_id}.question",
            )
            context = _require_text(row.get("context"), label=f"T2 query {raw_query_id}.context")
            file_name = _require_text(
                row.get("file_name"), label=f"T2 query {raw_query_id}.file_name"
            )
            dataset.add_document(
                _Document(identifier=context_id, title=file_name, text=context),
                repeated_identical_ok=True,
            )
            query_id = f"t2-ragbench:{raw_query_id}"
            dataset.add_query(_Query(identifier=query_id, text=question, source_split=split))
            dataset.add_qrel(_Qrel(query_id=query_id, document_id=context_id, relevance=1))
            answer_value = row.get("program_answer", row.get("original_answer"))
            metadata = tuple(
                (key, value)
                for key, value in (
                    ("split", split),
                    ("subset", str(observed_subset or "FinQA")),
                )
                if value
            )
            dataset.add_evidence_label(
                _EvidenceLabelRow(
                    query_id=query_id,
                    answer=None if answer_value is None else str(answer_value),
                    evidence_bundles=(
                        _EvidenceBundle(
                            bundle_id="source-context",
                            locations=(
                                _EvidenceLocation(
                                    document_id=context_id,
                                    locator="document",
                                ),
                            ),
                        ),
                    ),
                    label_metadata=metadata,
                )
            )
            split_query_ids[split].append(query_id)
    admitted_query_ids = set(
        _exclude_cross_source_split_components(
            dataset,
            tuple(
                query_id
                for split in ("train", "dev", "test")
                for query_id in split_query_ids[split]
            ),
        )
    )
    for split, stage in stage_for_split.items():
        _assign_fixed(
            dataset,
            (query_id for query_id in split_query_ids[split] if query_id in admitted_query_ids),
            stage=stage,
            assignment_seed=config.assignment_seed,
        )
    return dataset


def _load_generic_documents(
    dataset: _Dataset,
    pin: SourcePin,
    source_receipts: dict[str, int],
    *,
    id_prefix: str = "",
    duplicate_content_ok: bool = False,
) -> None:
    rows, byte_count = _load_records(pin)
    _record_source_use(source_receipts, pin, byte_count)
    for position, row in enumerate(rows):
        document = _generic_document(row, label=f"{pin.source_id} record {position}")
        if id_prefix:
            document = _Document(
                identifier=f"{id_prefix}{document.identifier}",
                title=document.title,
                text=document.text,
            )
        dataset.add_document(document, duplicate_content_ok=duplicate_content_ok)


def _load_generic_queries(
    dataset: _Dataset,
    pin: SourcePin,
    source_receipts: dict[str, int],
    *,
    source_split: str,
    query_prefix: str,
    domain: str | None = None,
) -> list[str]:
    if pin.path.suffix.casefold() == ".tsv":
        encoded, byte_count = _read_verified_bytes(pin)
        try:
            text_rows = encoded.decode("utf-8", errors="strict").splitlines()
        except UnicodeDecodeError as exc:
            raise StudyDataError(f"{pin.source_id} must be valid UTF-8") from exc
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text_rows, start=1):
            if not line:
                continue
            fields = line.split("\t", 1)
            if len(fields) != 2:
                raise StudyDataError(
                    f"{pin.source_id} line {line_number} must contain query ID and text"
                )
            rows.append({"id": fields[0], "text": fields[1]})
    else:
        rows, byte_count = _load_records(pin)
    _record_source_use(source_receipts, pin, byte_count)
    query_ids: list[str] = []
    for position, row in enumerate(rows):
        raw_id, text = _generic_query(row, label=f"{pin.source_id} record {position}")
        query_id = f"{query_prefix}{raw_id}"
        dataset.add_query(
            _Query(
                identifier=query_id,
                text=text,
                source_split=source_split,
                domain=domain,
            )
        )
        query_ids.append(query_id)
    return query_ids


def _load_generic_qrels(
    dataset: _Dataset,
    pin: SourcePin,
    source_receipts: dict[str, int],
    *,
    query_prefix: str,
    document_prefix: str = "",
) -> None:
    if pin.path.suffix.casefold() == ".tsv":
        encoded, byte_count = _read_verified_bytes(pin)
        try:
            text_rows = encoded.decode("utf-8", errors="strict").splitlines()
        except UnicodeDecodeError as exc:
            raise StudyDataError(f"{pin.source_id} must be valid UTF-8") from exc
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text_rows, start=1):
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 4 or fields[1] != "Q0":
                raise StudyDataError(
                    f"{pin.source_id} line {line_number} must use TREC qrels columns"
                )
            try:
                relevance = int(fields[3])
            except ValueError as exc:
                raise StudyDataError(
                    f"{pin.source_id} line {line_number} has non-integer relevance"
                ) from exc
            rows.append({"document_id": fields[2], "query_id": fields[0], "relevance": relevance})
    else:
        rows, byte_count = _load_records(pin)
    _record_source_use(source_receipts, pin, byte_count)
    for position, row in enumerate(rows):
        raw_query_id, raw_document_id, relevance = _generic_qrel(
            row, label=f"{pin.source_id} record {position}"
        )
        query_id = f"{query_prefix}{raw_query_id}"
        document_id = f"{document_prefix}{raw_document_id}"
        if query_id not in dataset.queries:
            raise StudyDataError(f"qrel names unknown query {query_id!r}")
        if document_id not in dataset.documents:
            raise StudyDataError(f"qrel names unknown document {document_id!r}")
        dataset.add_qrel(
            _Qrel(
                query_id=query_id,
                document_id=document_id,
                relevance=relevance,
            )
        )


def _load_miracl_sw(
    config: StagingConfig,
    source_receipts: dict[str, int],
) -> _Dataset:
    dataset = _Dataset("miracl-transfer")
    _load_generic_documents(
        dataset,
        config.pin("miracl_sw/documents"),
        source_receipts,
        duplicate_content_ok=True,
    )
    train_ids = _load_generic_queries(
        dataset,
        config.pin("miracl_sw/train_queries"),
        source_receipts,
        source_split="train",
        query_prefix="miracl-sw:",
    )
    dev_ids = _load_generic_queries(
        dataset,
        config.pin("miracl_sw/dev_queries"),
        source_receipts,
        source_split="dev",
        query_prefix="miracl-sw:",
    )
    _load_generic_qrels(
        dataset,
        config.pin("miracl_sw/train_qrels"),
        source_receipts,
        query_prefix="miracl-sw:",
    )
    _load_generic_qrels(
        dataset,
        config.pin("miracl_sw/dev_qrels"),
        source_receipts,
        query_prefix="miracl-sw:",
    )
    _assign_ranked_components(
        dataset,
        (*train_ids, *dev_ids),
        assignment_seed=config.assignment_seed,
        stages=STAGES,
    )
    return dataset


@dataclass(frozen=True)
class _BrightDocumentIdentity:
    collision_policy: str
    duplicate_source_rows: int
    mapping_sha256: str
    source_document_rows: int
    unique_documents: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "collision_policy": self.collision_policy,
            "deduplication_identity": "source-document-id-plus-canonical-content",
            "duplicate_source_rows": self.duplicate_source_rows,
            "mapping_sha256": self.mapping_sha256,
            "source_document_rows": self.source_document_rows,
            "unique_documents": self.unique_documents,
        }


def _consume_bright_documents(
    dataset: _Dataset,
    connection: sqlite3.Connection,
    *,
    domain: str,
    pin: SourcePin,
    collision_policy: str,
) -> tuple[int, int, int]:
    source_rows = 0
    duplicate_rows = 0

    def consume(row: Mapping[str, Any]) -> None:
        nonlocal source_rows, duplicate_rows
        document = _generic_document(row, label=f"{pin.source_id} record {source_rows}")
        source_rows += 1
        content_sha256 = _hash_parts("bright-canonical-content-v1", document.text)
        prior_contents = {
            str(value[0])
            for value in connection.execute(
                "SELECT content_sha256 FROM source_identity WHERE source_id = ?",
                (document.identifier,),
            )
        }
        if prior_contents and content_sha256 not in prior_contents and collision_policy == "error":
            raise StudyDataError(
                f"BRIGHT source document ID {document.identifier!r} has conflicting content"
            )
        global_id = "bright-doc:" + _hash_parts(
            "bright-global-document-v1", document.identifier, content_sha256
        )
        prior = connection.execute(
            "SELECT global_id FROM source_identity WHERE source_id = ? AND content_sha256 = ?",
            (document.identifier, content_sha256),
        ).fetchone()
        if prior is None:
            connection.execute(
                "INSERT INTO source_identity VALUES (?, ?, ?)",
                (document.identifier, content_sha256, global_id),
            )
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                (
                    global_id,
                    document.identifier,
                    document.text,
                    len(document.text.splitlines()),
                    content_sha256,
                ),
            )
        else:
            global_id = str(prior[0])
            duplicate_rows += 1
        try:
            connection.execute(
                "INSERT INTO domain_mapping VALUES (?, ?, ?)",
                (domain, document.identifier, global_id),
            )
        except sqlite3.IntegrityError as exc:
            raise StudyDataError(
                f"BRIGHT domain {domain!r} repeats source document ID {document.identifier!r}"
            ) from exc

    if pin.path.suffix.casefold() == ".parquet":
        byte_count = _consume_parquet_records(pin, consume)
    else:
        rows, byte_count = _load_records(pin)
        for row in rows:
            consume(row)
    dataset.streamed_document_count = int(
        connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    )
    return source_rows, duplicate_rows, byte_count


def _bright_global_document_id(
    connection: sqlite3.Connection,
    *,
    domain: str,
    source_document_id: str,
) -> str:
    row = connection.execute(
        "SELECT global_id FROM domain_mapping WHERE domain = ? AND source_id = ?",
        (domain, source_document_id),
    ).fetchone()
    if row is None:
        raise StudyDataError(
            f"BRIGHT {domain!r} qrel names unknown source document {source_document_id!r}"
        )
    return str(row[0])


def _load_bright(
    config: StagingConfig,
    source_receipts: dict[str, int],
    *,
    database_path: Path,
) -> tuple[_Dataset, sqlite3.Connection, _BrightDocumentIdentity]:
    dataset = _Dataset("bright")
    database = sqlite3.connect(database_path)
    database.execute(
        "CREATE TABLE documents ("
        "identifier TEXT PRIMARY KEY COLLATE BINARY, "
        "title TEXT NOT NULL, text TEXT NOT NULL, line_count INTEGER NOT NULL, "
        "content_sha256 TEXT NOT NULL)"
    )
    database.execute(
        "CREATE TABLE source_identity ("
        "source_id TEXT NOT NULL COLLATE BINARY, content_sha256 TEXT NOT NULL, "
        "global_id TEXT NOT NULL, PRIMARY KEY (source_id, content_sha256))"
    )
    database.execute(
        "CREATE TABLE domain_mapping ("
        "domain TEXT NOT NULL COLLATE BINARY, source_id TEXT NOT NULL COLLATE BINARY, "
        "global_id TEXT NOT NULL, PRIMARY KEY (domain, source_id))"
    )
    source_document_rows = 0
    duplicate_source_rows = 0
    all_query_ids: list[str] = []
    for domain in config.bright_domains:
        source_prefix = f"bright/{domain}"
        id_prefix = f"bright:{domain}:"
        document_pin = config.pin(f"{source_prefix}/documents")
        source_count, duplicate_count, byte_count = _consume_bright_documents(
            dataset,
            database,
            domain=domain,
            pin=document_pin,
            collision_policy=config.bright_document_id_collision_policy,
        )
        source_document_rows += source_count
        duplicate_source_rows += duplicate_count
        _record_source_use(source_receipts, document_pin, byte_count)
        examples_id = f"{source_prefix}/examples"
        if examples_id in config.pins:
            examples_pin = config.pin(examples_id)
            rows, byte_count = _load_records(examples_pin)
            _record_source_use(source_receipts, examples_pin, byte_count)
            query_ids: list[str] = []
            for position, row in enumerate(rows):
                raw_query_id, text = _generic_query(
                    row, label=f"{examples_pin.source_id} record {position}"
                )
                query_id = f"{id_prefix}{raw_query_id}"
                dataset.add_query(
                    _Query(
                        identifier=query_id,
                        text=text,
                        source_split="unsplit",
                        domain=domain,
                    )
                )
                query_ids.append(query_id)
                gold_ids = row.get("gold_ids")
                if (
                    not isinstance(gold_ids, list)
                    or not gold_ids
                    or not all(isinstance(value, str) for value in gold_ids)
                ):
                    raise StudyDataError(
                        f"BRIGHT example {query_id!r} needs non-empty string gold_ids"
                    )
                for source_document_id in gold_ids:
                    dataset.add_qrel(
                        _Qrel(
                            query_id=query_id,
                            document_id=_bright_global_document_id(
                                database,
                                domain=domain,
                                source_document_id=source_document_id,
                            ),
                            relevance=1,
                        ),
                        derived_duplicate_ok=True,
                    )
        else:
            query_ids = _load_generic_queries(
                dataset,
                config.pin(f"{source_prefix}/queries"),
                source_receipts,
                source_split="unsplit",
                query_prefix=id_prefix,
                domain=domain,
            )
            qrels_pin = config.pin(f"{source_prefix}/qrels")
            rows, byte_count = _load_records(qrels_pin)
            _record_source_use(source_receipts, qrels_pin, byte_count)
            for position, row in enumerate(rows):
                raw_query_id, raw_document_id, relevance = _generic_qrel(
                    row, label=f"{qrels_pin.source_id} record {position}"
                )
                query_id = f"{id_prefix}{raw_query_id}"
                if query_id not in dataset.queries:
                    raise StudyDataError(f"qrel names unknown query {query_id!r}")
                dataset.add_qrel(
                    _Qrel(
                        query_id=query_id,
                        document_id=_bright_global_document_id(
                            database,
                            domain=domain,
                            source_document_id=raw_document_id,
                        ),
                        relevance=relevance,
                    )
                )
        all_query_ids.extend(query_ids)
    _assign_ranked_components(
        dataset,
        all_query_ids,
        assignment_seed=config.assignment_seed,
        stages=STAGES,
        domain=None,
    )
    database.commit()
    mapping_digest = hashlib.sha256()
    for domain, source_id, global_id in database.execute(
        "SELECT domain, source_id, global_id FROM domain_mapping "
        "ORDER BY domain COLLATE BINARY, source_id COLLATE BINARY"
    ):
        mapping_digest.update(
            _canonical_bytes({"domain": domain, "global_id": global_id, "source_id": source_id})
            + b"\n"
        )
    unique_documents = int(database.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    dataset.streamed_document_count = unique_documents
    return (
        dataset,
        database,
        _BrightDocumentIdentity(
            collision_policy=config.bright_document_id_collision_policy,
            duplicate_source_rows=duplicate_source_rows,
            mapping_sha256=mapping_digest.hexdigest(),
            source_document_rows=source_document_rows,
            unique_documents=unique_documents,
        ),
    )


def _shard_paths(root: Path) -> list[tuple[str, Path]]:
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise StudyDataError(f"cannot inspect shard root {root}: {exc}") from exc
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise StudyDataError("shard root must be a real directory")
    observed: list[tuple[str, Path]] = []
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise StudyDataError(f"shard tree contains invalid directory {child}")
        for name in files:
            child = directory_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise StudyDataError(f"shard tree contains non-regular file {child}")
            relative = child.relative_to(root).as_posix()
            if not relative.endswith(".bz2"):
                raise StudyDataError(f"shard tree contains non-bz2 file {relative!r}")
            observed.append((relative, child))
    return sorted(observed, key=lambda item: item[0].encode())


def _read_regular_bytes(path: Path) -> bytes:
    descriptor = _open_regular(path)
    chunks: list[bytes] = []
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def _shard_tree_row(relative_path: str, encoded: bytes) -> bytes:
    return (
        _canonical_bytes(
            {
                "byte_count": len(encoded),
                "path": relative_path,
                "sha256": _sha256_bytes(encoded),
            }
        )
        + b"\n"
    )


def compute_shard_tree_digest(root: str | Path) -> tuple[str, int, int]:
    """Return the canonical compressed-shard digest, file count, and byte count."""

    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for relative_path, path in _shard_paths(Path(root)):
        encoded = _read_regular_bytes(path)
        digest.update(_shard_tree_row(relative_path, encoded))
        file_count += 1
        byte_count += len(encoded)
    if file_count == 0:
        raise StudyDataError("shard tree contains no .bz2 files")
    return digest.hexdigest(), file_count, byte_count


def _stream_hotpot_corpus(
    config: StagingConfig,
    source_receipts: dict[str, int],
    *,
    database_path: Path,
) -> tuple[sqlite3.Connection, int]:
    archive_pin = config.pin("hotpotqa_fullwiki/corpus_archive")
    archive_byte_count = _verify_file_pin(archive_pin)
    _record_source_use(source_receipts, archive_pin, archive_byte_count)
    shard_pin = config.hotpotqa_shards
    shard_paths = _shard_paths(shard_pin.path)
    if len(shard_paths) != shard_pin.file_count:
        raise StudyDataError(
            f"HotpotQA shard count mismatch: expected {shard_pin.file_count}, "
            f"observed {len(shard_paths)}"
        )
    tree_digest = hashlib.sha256()
    compressed_byte_count = 0
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE documents ("
            "identifier TEXT PRIMARY KEY COLLATE BINARY, "
            "title TEXT NOT NULL UNIQUE COLLATE BINARY, "
            "text TEXT NOT NULL, line_count INTEGER NOT NULL, "
            "content_sha256 TEXT NOT NULL UNIQUE)"
        )
        document_count = 0
        empty_text_fallbacks = 0
        for relative_path, shard_path in shard_paths:
            encoded_shard = _read_regular_bytes(shard_path)
            tree_digest.update(_shard_tree_row(relative_path, encoded_shard))
            compressed_byte_count += len(encoded_shard)
            try:
                decoded_shard = bz2.decompress(encoded_shard)
            except (OSError, EOFError) as exc:
                raise StudyDataError(
                    f"cannot decompress HotpotQA shard {relative_path!r}: {exc}"
                ) from exc
            for line_number, encoded_line in enumerate(decoded_shard.splitlines(), start=1):
                if not encoded_line.strip():
                    continue
                label = f"{shard_pin.source_id}/{relative_path} line {line_number}"
                row = _decode_json(encoded_line, label=label)
                if not isinstance(row, Mapping):
                    raise StudyDataError(f"{label} must be a JSON object")
                identifier = _source_identifier(row.get("id"), label=f"{label}.id")
                title = _require_text(row.get("title"), label=f"{label}.title")
                source_sentences = row.get("text")
                if not isinstance(source_sentences, list) or not all(
                    isinstance(sentence, str) for sentence in source_sentences
                ):
                    raise StudyDataError(f"{label}.text must be a string array")
                source_text = "\n".join(source_sentences)
                if source_text:
                    text = _require_text(source_text, label=f"{label}.text")
                else:
                    text = title
                    empty_text_fallbacks += 1
                document = _Document(identifier=identifier, title=title, text=text)
                line_count = len(source_sentences)
                try:
                    connection.execute(
                        "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                        (
                            document.identifier,
                            document.title,
                            document.text,
                            line_count,
                            _hash_parts(document.title, document.text),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise StudyDataError(
                        "HotpotQA FullWiki corpus contains duplicate ID, title, or content "
                        f"at {relative_path}:{line_number}: {exc}"
                    ) from exc
                document_count += 1
            connection.commit()
        observed_tree_sha256 = tree_digest.hexdigest()
        if observed_tree_sha256 != shard_pin.sha256:
            raise StudyDataError(
                f"source {shard_pin.source_id!r} SHA-256 mismatch: expected "
                f"{shard_pin.sha256}, observed {observed_tree_sha256}"
            )
        connection.commit()
        if document_count != config.hotpotqa_expected_document_count:
            raise StudyDataError(
                "HotpotQA FullWiki document count mismatch: expected "
                f"{config.hotpotqa_expected_document_count}, observed {document_count}; "
                "sampled corpora are not admissible"
            )
        indexed_document_count = int(
            connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        if indexed_document_count != document_count:
            raise StudyDataError("HotpotQA index row count differs from streamed record count")
        _record_source_use(source_receipts, shard_pin, compressed_byte_count)
        return connection, empty_text_fallbacks
    except Exception:
        connection.close()
        raise


def _load_hotpot_queries(
    config: StagingConfig,
    source_receipts: dict[str, int],
    dataset: _Dataset,
    *,
    connection: sqlite3.Connection,
) -> None:
    split_query_ids: dict[str, list[str]] = {"train": [], "dev": []}
    sources = (
        *(("train", source_id) for source_id in config.hotpotqa_train_source_ids),
        ("dev", "hotpotqa_fullwiki/dev_questions"),
    )
    observed_raw_ids: dict[str, str] = {}
    for source_split, source_id in sources:
        pin = config.pin(source_id)
        rows, byte_count = _load_records(pin)
        _record_source_use(source_receipts, pin, byte_count)
        for position, row in enumerate(rows):
            raw_id = _source_identifier(
                row.get("_id", row.get("id")),
                label=f"HotpotQA {source_split} question {position}.id",
            )
            previous_source = observed_raw_ids.get(raw_id)
            if previous_source is not None:
                raise StudyDataError(
                    f"HotpotQA question ID {raw_id!r} occurs in both "
                    f"{previous_source!r} and {source_id!r}"
                )
            observed_raw_ids[raw_id] = source_id
            query_id = f"hotpotqa:{raw_id}"
            question = _require_text(
                row.get("question"), label=f"HotpotQA question {query_id}.question"
            )
            supporting_facts = row.get("supporting_facts")
            if isinstance(supporting_facts, Mapping):
                titles = supporting_facts.get("title")
                sentence_ids = supporting_facts.get("sent_id")
                if (
                    isinstance(titles, list)
                    and isinstance(sentence_ids, list)
                    and len(titles) == len(sentence_ids)
                ):
                    supporting_facts = [
                        list(pair) for pair in zip(titles, sentence_ids, strict=True)
                    ]
            if not isinstance(supporting_facts, list) or not supporting_facts:
                raise StudyDataError(
                    f"HotpotQA question {query_id!r} needs non-empty supporting_facts"
                )
            relevant_documents: set[str] = set()
            evidence_locations: list[_EvidenceLocation] = []
            structurally_invalid = False
            for fact_index, fact in enumerate(supporting_facts):
                if (
                    not isinstance(fact, list)
                    or len(fact) != 2
                    or not isinstance(fact[0], str)
                    or type(fact[1]) is not int
                    or fact[1] < 0
                ):
                    raise StudyDataError(
                        f"HotpotQA question {query_id!r} supporting fact {fact_index} is invalid"
                    )
                record = connection.execute(
                    "SELECT identifier, line_count FROM documents WHERE title = ?", (fact[0],)
                ).fetchone()
                if record is None:
                    raise StudyDataError(
                        f"HotpotQA question {query_id!r} names title {fact[0]!r} "
                        "absent from the pinned FullWiki corpus"
                    )
                document_id, line_count = str(record[0]), int(record[1])
                if fact[1] >= line_count:
                    dataset.exclusions.append(
                        _Exclusion(
                            dataset=dataset.name,
                            query_id=query_id,
                            source_split=source_split,
                            rule_id="hotpotqa-supporting-fact-range-v1",
                            reason="out-of-range-supporting-sentence",
                        )
                    )
                    structurally_invalid = True
                    break
                relevant_documents.add(document_id)
                evidence_locations.append(
                    _EvidenceLocation(
                        document_id=document_id,
                        locator=f"sentence:{fact[1]}",
                    )
                )
            if structurally_invalid:
                continue
            dataset.add_query(_Query(identifier=query_id, text=question, source_split=source_split))
            split_query_ids[source_split].append(query_id)
            for document_id in sorted(relevant_documents, key=lambda item: item.encode()):
                dataset.add_qrel(_Qrel(query_id=query_id, document_id=document_id, relevance=1))
            metadata = tuple(
                (key, str(value))
                for key, value in (("type", row.get("type")), ("level", row.get("level")))
                if value is not None and str(value)
            )
            dataset.add_evidence_label(
                _EvidenceLabelRow(
                    query_id=query_id,
                    answer=None if row.get("answer") is None else str(row["answer"]),
                    evidence_bundles=(
                        _EvidenceBundle(
                            bundle_id="supporting-facts",
                            locations=tuple(evidence_locations),
                        ),
                    ),
                    label_metadata=metadata,
                )
            )
    admitted_query_ids = set(
        _exclude_cross_source_split_components(
            dataset,
            (*split_query_ids["train"], *split_query_ids["dev"]),
        )
    )
    _assign_ranked_components(
        dataset,
        tuple(query_id for query_id in split_query_ids["train"] if query_id in admitted_query_ids),
        assignment_seed=config.assignment_seed,
        stages=("fit", "calibration"),
    )
    _assign_fixed(
        dataset,
        (query_id for query_id in split_query_ids["dev"] if query_id in admitted_query_ids),
        stage="sealed",
        assignment_seed=config.assignment_seed,
    )
    dataset.streamed_document_count = config.hotpotqa_expected_document_count


def _audit_datasets(datasets: Sequence[_Dataset]) -> None:
    global_queries: dict[str, tuple[str, str]] = {}
    normalized_queries: dict[str, tuple[str, str, str, str]] = {}
    for dataset in datasets:
        if not dataset.queries:
            raise StudyDataError(f"dataset {dataset.name!r} contains no queries")
        if dataset.streamed_document_count is None and not dataset.documents:
            raise StudyDataError(f"dataset {dataset.name!r} contains no documents")
        for query in dataset.queries.values():
            if query.stage not in STAGES:
                raise StudyDataError(f"query {query.identifier!r} has no valid stage")
            if query.component_sha256 is None or query.assignment_key_sha256 is None:
                raise StudyDataError(f"query {query.identifier!r} lacks assignment evidence")
            previous = global_queries.get(query.identifier)
            if previous is not None:
                raise StudyDataError(
                    f"query ID {query.identifier!r} occurs in {previous[0]!r} and {dataset.name!r}"
                )
            global_queries[query.identifier] = (dataset.name, query.stage)
            normalized = _normalize_query_text(query.text)
            previous_text = normalized_queries.get(normalized)
            if previous_text is not None:
                same_coupled_component = (
                    previous_text[0] == dataset.name
                    and previous_text[1] == query.stage
                    and previous_text[3] == query.component_sha256
                )
                if not same_coupled_component:
                    raise StudyDataError(
                        "duplicate normalized query text crosses an admitted component: "
                        f"{previous_text[2]!r} ({previous_text[0]}/{previous_text[1]}) and "
                        f"{query.identifier!r} ({dataset.name}/{query.stage})"
                    )
            else:
                normalized_queries[normalized] = (
                    dataset.name,
                    query.stage,
                    query.identifier,
                    query.component_sha256,
                )

        positive_document_stages: dict[str, tuple[str, str]] = {}
        for qrel in dataset.qrels.values():
            query = dataset.queries.get(qrel.query_id)
            if query is None:
                raise StudyDataError(f"qrel names unknown query {qrel.query_id!r}")
            if (
                dataset.streamed_document_count is None
                and qrel.document_id not in dataset.documents
            ):
                raise StudyDataError(f"qrel names unknown document {qrel.document_id!r}")
            if qrel.relevance <= 0:
                continue
            document = dataset.documents.get(qrel.document_id)
            relevance_identity = (
                f"content:{_hash_parts(document.title, document.text)}"
                if document is not None
                else f"document-id:{qrel.document_id}"
            )
            previous = positive_document_stages.get(relevance_identity)
            if previous is not None and previous[0] != query.stage:
                raise StudyDataError(
                    f"positive document {qrel.document_id!r} crosses {previous[0]!r} and "
                    f"{query.stage!r} through queries {previous[1]!r} and {qrel.query_id!r}"
                )
            positive_document_stages[relevance_identity] = (query.stage, qrel.query_id)

        evidence_corpora = {"scifact", "hotpotqa-fullwiki", "t2-ragbench"}
        expected_evidence_queries = (
            set(dataset.queries) if dataset.name in evidence_corpora else set()
        )
        observed_evidence_queries = set(dataset.evidence_labels)
        if observed_evidence_queries != expected_evidence_queries:
            raise StudyDataError(
                f"dataset {dataset.name!r} evidence-label coverage differs; "
                f"missing={sorted(expected_evidence_queries - observed_evidence_queries)}, "
                f"unexpected={sorted(observed_evidence_queries - expected_evidence_queries)}"
            )
        positive_by_query: dict[str, set[str]] = {}
        for qrel in dataset.qrels.values():
            if qrel.relevance > 0:
                positive_by_query.setdefault(qrel.query_id, set()).add(qrel.document_id)
        for query_id, label_row in dataset.evidence_labels.items():
            relevant = positive_by_query.get(query_id, set())
            for bundle in label_row.evidence_bundles:
                for location in bundle.locations:
                    if location.document_id not in relevant:
                        raise StudyDataError(
                            f"evidence location {location.document_id!r} for query "
                            f"{query_id!r} lacks a positive relevance link"
                        )
                    if (
                        dataset.streamed_document_count is None
                        and location.document_id not in dataset.documents
                    ):
                        raise StudyDataError(
                            f"evidence labels name unknown document {location.document_id!r}"
                        )


def _assignment_rows(datasets: Sequence[_Dataset]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for query in dataset.queries.values():
            rows.append(
                {
                    "assignment_key_sha256": query.assignment_key_sha256,
                    "dataset": dataset.name,
                    "domain": query.domain,
                    "partition_component_sha256": query.component_sha256,
                    "query_id": query.identifier,
                    "query_text_sha256": _sha256_bytes(query.text.encode("utf-8")),
                    "schema_version": ASSIGNMENT_SCHEMA,
                    "source_split": query.source_split,
                    "stage": query.stage,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["dataset"].encode(),
            (row["domain"] or "").encode(),
            row["query_id"].encode(),
        ),
    )


def _artifact_row(
    *,
    relative_path: str,
    encoded_sha256: str,
    byte_count: int,
    record_count: int,
    dataset: str | None,
    stage: str | None,
    role: str,
    visibility: str,
) -> dict[str, Any]:
    return {
        "byte_count": byte_count,
        "dataset": dataset,
        "path": relative_path,
        "record_count": record_count,
        "role": role,
        "sha256": encoded_sha256,
        "stage": stage,
        "visibility": visibility,
    }


def _write_jsonl(
    root: Path,
    relative_path: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str | None,
    stage: str | None,
    role: str,
    visibility: str,
) -> dict[str, Any]:
    _safe_relative_path(relative_path, label="artifact path")
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    try:
        handle = target.open("xb")
    except OSError as exc:
        raise StudyDataError(f"cannot create staged artifact {target}: {exc}") from exc
    with handle:
        for row in rows:
            encoded = _canonical_bytes(row) + b"\n"
            handle.write(encoded)
            digest.update(encoded)
            byte_count += len(encoded)
            record_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return _artifact_row(
        relative_path=relative_path,
        encoded_sha256=digest.hexdigest(),
        byte_count=byte_count,
        record_count=record_count,
        dataset=dataset,
        stage=stage,
        role=role,
        visibility=visibility,
    )


def _dataset_rows_for_stage(
    dataset: _Dataset,
    stage: str,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str | int]],
    list[dict[str, Any]],
]:
    query_ids = {query.identifier for query in dataset.queries.values() if query.stage == stage}
    queries = [
        dataset.queries[query_id].to_dict()
        for query_id in sorted(query_ids, key=lambda item: item.encode())
    ]
    qrels = [
        qrel.to_dict()
        for qrel in sorted(
            (qrel for qrel in dataset.qrels.values() if qrel.query_id in query_ids),
            key=lambda item: (
                item.query_id.encode(),
                item.document_id.encode(),
                item.relevance,
            ),
        )
    ]
    evidence_labels = [
        dataset.evidence_labels[query_id].to_dict()
        for query_id in sorted(
            query_ids & set(dataset.evidence_labels),
            key=lambda item: item.encode(),
        )
    ]
    return queries, qrels, evidence_labels


def _write_streamed_corpus_shards(
    root: Path,
    dataset: _Dataset,
    connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    cursor = iter(
        connection.execute(
            "SELECT identifier, title, text FROM documents ORDER BY identifier COLLATE BINARY"
        )
    )
    part_index = 0
    written_records = 0
    while True:
        try:
            first = next(cursor)
        except StopIteration:
            break
        source_rows = itertools.chain((first,), itertools.islice(cursor, CORPUS_SHARD_RECORDS - 1))
        rows = (
            {"id": str(identifier), "text": str(text), "title": str(title)}
            for identifier, title, text in source_rows
        )
        artifact = _write_jsonl(
            root,
            f"datasets/{dataset.name}/corpus/part-{part_index:05d}.jsonl",
            rows,
            dataset=dataset.name,
            stage=None,
            role="corpus-shard",
            visibility="online",
        )
        artifacts.append(artifact)
        written_records += int(artifact["record_count"])
        part_index += 1
    if written_records != dataset.streamed_document_count:
        raise StudyDataError(
            f"dataset {dataset.name!r} streamed {written_records} corpus rows; "
            f"expected {dataset.streamed_document_count}"
        )
    return artifacts


def _write_dataset(
    root: Path,
    dataset: _Dataset,
    *,
    withhold_sealed_labels_from_online_process: bool,
    streamed_connection: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if streamed_connection is None:
        corpus_path = f"datasets/{dataset.name}/corpus.jsonl"
        document_rows: Iterable[Mapping[str, Any]] = (
            dataset.documents[document_id].to_dict()
            for document_id in sorted(dataset.documents, key=lambda item: item.encode())
        )
        artifacts.append(
            _write_jsonl(
                root,
                corpus_path,
                document_rows,
                dataset=dataset.name,
                stage=None,
                role="corpus",
                visibility="online",
            )
        )
    else:
        artifacts.extend(_write_streamed_corpus_shards(root, dataset, streamed_connection))
    for stage in STAGES:
        queries, qrels, evidence_labels = _dataset_rows_for_stage(dataset, stage)
        if not queries:
            continue
        if stage == "sealed":
            queries_path = f"datasets/{dataset.name}/sealed/online/queries.jsonl"
            qrels_visibility = "custody" if withhold_sealed_labels_from_online_process else "online"
            qrels_path = f"datasets/{dataset.name}/sealed/{qrels_visibility}/qrels.jsonl"
        else:
            queries_path = f"datasets/{dataset.name}/{stage}/queries.jsonl"
            qrels_path = f"datasets/{dataset.name}/{stage}/qrels.jsonl"
            qrels_visibility = "online"
        artifacts.append(
            _write_jsonl(
                root,
                queries_path,
                queries,
                dataset=dataset.name,
                stage=stage,
                role="queries",
                visibility="online",
            )
        )
        artifacts.append(
            _write_jsonl(
                root,
                qrels_path,
                qrels,
                dataset=dataset.name,
                stage=stage,
                role="qrels",
                visibility=qrels_visibility,
            )
        )
        if evidence_labels:
            evidence_visibility = qrels_visibility
            evidence_path = (
                f"datasets/{dataset.name}/sealed/{evidence_visibility}/evidence-bundles.jsonl"
                if stage == "sealed"
                else f"datasets/{dataset.name}/{stage}/evidence-bundles.jsonl"
            )
            artifacts.append(
                _write_jsonl(
                    root,
                    evidence_path,
                    evidence_labels,
                    dataset=dataset.name,
                    stage=stage,
                    role="evidence-bundles",
                    visibility=evidence_visibility,
                )
            )
    return artifacts


def _counts(datasets: Sequence[_Dataset]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for dataset in sorted(datasets, key=lambda item: item.name.encode()):
        document_count = (
            dataset.streamed_document_count
            if dataset.streamed_document_count is not None
            else len(dataset.documents)
        )
        row: dict[str, int] = {
            "documents": document_count,
            "duplicate_document_content_aliases": dataset.duplicate_content_aliases,
            "duplicate_positive_judgments": dataset.duplicate_positive_judgments,
            "empty_document_text_fallbacks": dataset.empty_document_text_fallbacks,
            "evidence_bundles": sum(
                len(label.evidence_bundles) for label in dataset.evidence_labels.values()
            ),
            "evidence_label_rows": len(dataset.evidence_labels),
            "evidence_locations": sum(
                len(bundle.locations)
                for label in dataset.evidence_labels.values()
                for bundle in label.evidence_bundles
            ),
            "qrels": len(dataset.qrels),
        }
        for stage in STAGES:
            row[f"{stage}_queries"] = sum(
                query.stage == stage for query in dataset.queries.values()
            )
        row["excluded_queries"] = len(dataset.exclusions)
        row["partition_excluded_queries"] = len(dataset.partition_exclusions)
        for source_split in sorted(
            {exclusion.source_split for exclusion in dataset.exclusions},
            key=lambda value: value.encode(),
        ):
            row[f"excluded_{source_split}_queries"] = sum(
                exclusion.source_split == source_split for exclusion in dataset.exclusions
            )
        for source_split in sorted(
            {exclusion.source_split for exclusion in dataset.partition_exclusions},
            key=lambda value: value.encode(),
        ):
            row[f"partition_excluded_{source_split}_queries"] = sum(
                exclusion.source_split == source_split for exclusion in dataset.partition_exclusions
            )
        result[dataset.name] = row
    return result


@dataclass(frozen=True)
class StagingReceipt:
    output_root: Path
    inventory_sha256: str
    artifact_count: int
    source_count: int


@dataclass(frozen=True)
class OnlineProjectionReceipt:
    """Verified identity of the non-label-bearing online staging tree."""

    output_root: Path
    inventory_sha256: str
    source_artifact_count: int
    projected_artifact_count: int
    projected_artifact_set_sha256: str


def stage_study_data(
    config_path: str | Path,
    output_root: str | Path,
) -> StagingReceipt:
    """Build and atomically publish one canonical, content-addressed data package."""

    config = load_staging_config(config_path)
    output = Path(output_root)
    if os.path.lexists(output):
        raise StudyDataError(f"output root already exists and will not be overwritten: {output}")
    output_parent = output.parent
    try:
        output_parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StudyDataError(f"cannot create output parent {output_parent}: {exc}") from exc
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output_parent))
    source_receipts: dict[str, int] = {}
    hotpot_connection: sqlite3.Connection | None = None
    bright_connection: sqlite3.Connection | None = None
    try:
        work = temporary / ".work"
        work.mkdir()
        scifact = _load_scifact(config, source_receipts)
        t2 = _load_t2_finqa(config, source_receipts)
        miracl = _load_miracl_sw(config, source_receipts)
        bright, bright_connection, bright_identity = _load_bright(
            config,
            source_receipts,
            database_path=work / "bright-documents.sqlite3",
        )

        hotpot_connection, hotpot_empty_text_fallbacks = _stream_hotpot_corpus(
            config,
            source_receipts,
            database_path=work / "hotpot-fullwiki.sqlite3",
        )
        hotpot = _Dataset("hotpotqa-fullwiki")
        hotpot.empty_document_text_fallbacks = hotpot_empty_text_fallbacks
        _load_hotpot_queries(
            config,
            source_receipts,
            hotpot,
            connection=hotpot_connection,
        )
        datasets = (scifact, hotpot, t2, bright, miracl)
        _audit_datasets(datasets)
        expected_sources = {source_id for source_id, _, _ in config.source_bindings}
        if set(source_receipts) != expected_sources:
            missing = sorted(expected_sources - set(source_receipts))
            unexpected = sorted(set(source_receipts) - expected_sources)
            raise StudyDataError(
                f"source consumption is incomplete; missing={missing}, unexpected={unexpected}"
            )

        artifacts: list[dict[str, Any]] = []
        for dataset in datasets:
            artifacts.extend(
                _write_dataset(
                    temporary,
                    dataset,
                    withhold_sealed_labels_from_online_process=(
                        config.withhold_sealed_labels_from_online_process
                    ),
                    streamed_connection={
                        "bright": bright_connection,
                        "hotpotqa-fullwiki": hotpot_connection,
                    }.get(dataset.name),
                )
            )
        artifacts.append(
            _write_jsonl(
                temporary,
                "assignments.jsonl",
                _assignment_rows(datasets),
                dataset=None,
                stage=None,
                role="assignments",
                visibility="online",
            )
        )
        exclusions = sorted(
            (exclusion.to_dict() for dataset in datasets for exclusion in dataset.exclusions),
            key=lambda row: (
                row["dataset"].encode(),
                row["source_split"].encode(),
                row["query_id"].encode(),
            ),
        )
        if exclusions:
            artifacts.append(
                _write_jsonl(
                    temporary,
                    "exclusions.jsonl",
                    exclusions,
                    dataset=None,
                    stage=None,
                    role="registered-cohort-exclusions",
                    visibility="protocol",
                )
            )
        partition_exclusions = sorted(
            (
                exclusion.to_dict()
                for dataset in datasets
                for exclusion in dataset.partition_exclusions
            ),
            key=lambda row: (
                row["dataset"].encode(),
                row["partition_component_sha256"].encode(),
                row["source_split"].encode(),
                row["query_id"].encode(),
            ),
        )
        artifacts.append(
            _write_jsonl(
                temporary,
                "partition-exclusions.jsonl",
                partition_exclusions,
                dataset=None,
                stage=None,
                role="query-partition-structural-exclusions",
                visibility="protocol",
            )
        )
        artifacts.sort(key=lambda row: row["path"].encode())
        source_rows = [
            {
                "byte_count": source_receipts[source_id],
                "revision": revision,
                "sha256": sha256,
                "source_id": source_id,
            }
            for source_id, revision, sha256 in config.source_bindings
        ]
        inventory = {
            "artifacts": artifacts,
            "assignment_algorithm": {
                "component_edges": [
                    "normalized-query-text-equality",
                    "registered-near-duplicate-token-rule",
                    "shared-positive-document-content",
                    "shared-positive-relevance-document",
                ],
                "cross_source_split_policy": "exclude-entire-component-v1",
                "fit_calibration_component_ratio": "4:1",
                "name": ASSIGNMENT_ALGORITHM,
                "three_way_component_ratio": "3:1:1",
            },
            "assignment_seed_sha256": config.assignment_seed,
            "bright_document_identity": bright_identity.to_dict(),
            "bright_domains": list(config.bright_domains),
            "config_sha256": config.digest,
            "counts": _counts(datasets),
            "hotpotqa_fullwiki_scope": {
                "expected_document_count": config.hotpotqa_expected_document_count,
                "name": "fullwiki",
                "sampling": "none",
            },
            "schema_version": INVENTORY_SCHEMA,
            "sources": source_rows,
            "withhold_sealed_labels_from_online_process": (
                config.withhold_sealed_labels_from_online_process
            ),
        }
        inventory_bytes = _canonical_bytes(inventory) + b"\n"
        inventory_sha256 = _sha256_bytes(inventory_bytes)
        inventory_path = temporary / "inventory.json"
        with inventory_path.open("xb") as handle:
            handle.write(inventory_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        checksum_bytes = f"{inventory_sha256}  inventory.json\n".encode("ascii")
        with (temporary / "inventory.sha256").open("xb") as handle:
            handle.write(checksum_bytes)
            handle.flush()
            os.fsync(handle.fileno())

        hotpot_connection.close()
        hotpot_connection = None
        bright_connection.close()
        bright_connection = None
        shutil.rmtree(work)
        if os.path.lexists(output):
            raise StudyDataError(
                f"output root appeared during staging and will not be overwritten: {output}"
            )
        os.rename(temporary, output)
        return StagingReceipt(
            output_root=output,
            inventory_sha256=inventory_sha256,
            artifact_count=len(artifacts),
            source_count=len(source_rows),
        )
    except BaseException:
        if hotpot_connection is not None:
            hotpot_connection.close()
        if bright_connection is not None:
            bright_connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_regular_file(path: Path) -> bytes:
    descriptor = _open_regular(path)
    chunks: list[bytes] = []
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def _scan_package_tree(package_root: Path) -> set[str]:
    observed_paths: set[str] = set()
    for directory, directory_names, file_names in os.walk(package_root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            relative = child.relative_to(package_root).as_posix()
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise StudyDataError(f"staged package contains symlink {relative!r}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise StudyDataError(f"staged package contains non-directory object {relative!r}")
        for name in file_names:
            child = directory_path / name
            relative = child.relative_to(package_root).as_posix()
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise StudyDataError(f"staged package contains symlink {relative!r}")
            if not stat.S_ISREG(metadata.st_mode):
                raise StudyDataError(f"staged package contains non-regular object {relative!r}")
            observed_paths.add(relative)
    return observed_paths


def _artifact_fingerprint(path: Path) -> tuple[int, str, int, bool]:
    descriptor = _open_regular(path)
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    last_byte: int | None = None
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            record_count += chunk.count(b"\n")
            last_byte = chunk[-1]
    return byte_count, digest.hexdigest(), record_count, last_byte in {None, 10}


def verify_staged_data(root: str | Path) -> StagingReceipt:
    """Verify canonical inventory bytes, exact membership, sizes, and SHA-256 pins."""

    package_root = Path(root)
    try:
        root_stat = package_root.lstat()
    except OSError as exc:
        raise StudyDataError(f"cannot inspect staged package {package_root}: {exc}") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise StudyDataError("staged package root must be a real directory")
    observed_paths = _scan_package_tree(package_root)
    inventory_bytes = _read_regular_file(package_root / "inventory.json")
    checksum_bytes = _read_regular_file(package_root / "inventory.sha256")
    inventory_sha256 = _sha256_bytes(inventory_bytes)
    expected_checksum = f"{inventory_sha256}  inventory.json\n".encode("ascii")
    if checksum_bytes != expected_checksum:
        raise StudyDataError("inventory.sha256 does not match inventory.json")
    if not inventory_bytes.endswith(b"\n"):
        raise StudyDataError("inventory.json must end with one canonical newline")
    inventory = _decode_json(inventory_bytes, label="inventory.json")
    root_row = _closed_mapping(inventory, fields=_INVENTORY_FIELDS, label="inventory")
    if root_row["schema_version"] != INVENTORY_SCHEMA:
        raise StudyDataError(f"inventory schema must equal {INVENTORY_SCHEMA!r}")
    if _canonical_bytes(inventory) + b"\n" != inventory_bytes:
        raise StudyDataError("inventory.json is not canonical JSON")

    artifact_rows = root_row["artifacts"]
    if not isinstance(artifact_rows, list) or not artifact_rows:
        raise StudyDataError("inventory artifacts must be a non-empty array")
    listed_paths: set[str] = set()
    for position, value in enumerate(artifact_rows):
        row = _closed_mapping(
            value,
            fields=_INVENTORY_ARTIFACT_FIELDS,
            label=f"inventory artifact {position}",
        )
        relative_path = _safe_relative_path(row["path"], label="artifact path")
        if relative_path in listed_paths:
            raise StudyDataError(f"inventory repeats artifact path {relative_path!r}")
        listed_paths.add(relative_path)
        if _SHA256.fullmatch(row["sha256"]) is None:
            raise StudyDataError(f"artifact {relative_path!r} has invalid SHA-256")
        for count_name in ("byte_count", "record_count"):
            count = row[count_name]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise StudyDataError(f"artifact {relative_path!r} has invalid {count_name}")
        byte_count, sha256, record_count, canonical_ending = _artifact_fingerprint(
            package_root / relative_path
        )
        if byte_count != row["byte_count"]:
            raise StudyDataError(f"artifact {relative_path!r} byte count changed")
        if sha256 != row["sha256"]:
            raise StudyDataError(f"artifact {relative_path!r} SHA-256 changed")
        if record_count != row["record_count"]:
            raise StudyDataError(f"artifact {relative_path!r} record count changed")
        if not canonical_ending:
            raise StudyDataError(f"artifact {relative_path!r} lacks a canonical terminal newline")

    source_rows = root_row["sources"]
    if not isinstance(source_rows, list) or not source_rows:
        raise StudyDataError("inventory sources must be a non-empty array")
    source_ids: set[str] = set()
    for position, value in enumerate(source_rows):
        row = _closed_mapping(
            value,
            fields=_INVENTORY_SOURCE_FIELDS,
            label=f"inventory source {position}",
        )
        source_id = _require_identifier(row["source_id"], label="inventory source_id")
        if source_id in source_ids:
            raise StudyDataError(f"inventory repeats source ID {source_id!r}")
        source_ids.add(source_id)
        if _SHA256.fullmatch(row["sha256"]) is None:
            raise StudyDataError(f"source {source_id!r} has invalid SHA-256")
        source_byte_count = row["byte_count"]
        if (
            isinstance(source_byte_count, bool)
            or not isinstance(source_byte_count, int)
            or source_byte_count < 0
        ):
            raise StudyDataError(f"source {source_id!r} has invalid byte_count")
        revision = _require_identifier(row["revision"], label=f"source {source_id}.revision")
        if revision.casefold() in _PLACEHOLDERS:
            raise StudyDataError(f"source {source_id!r} has a movable revision")
    expected_paths = listed_paths | {"inventory.json", "inventory.sha256"}
    if observed_paths != expected_paths:
        raise StudyDataError(
            "staged package membership changed; "
            f"missing={sorted(expected_paths - observed_paths)}, "
            f"unexpected={sorted(observed_paths - expected_paths)}"
        )
    return StagingReceipt(
        output_root=package_root,
        inventory_sha256=inventory_sha256,
        artifact_count=len(artifact_rows),
        source_count=len(source_rows),
    )


def _online_projection_artifacts(
    inventory: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], int]:
    root_row = _closed_mapping(inventory, fields=_INVENTORY_FIELDS, label="inventory")
    values = root_row["artifacts"]
    if not isinstance(values, list) or not values:
        raise StudyDataError("inventory artifacts must be a non-empty array")
    projected: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    prior_path: str | None = None
    for position, value in enumerate(values):
        row = dict(
            _closed_mapping(
                value,
                fields=_INVENTORY_ARTIFACT_FIELDS,
                label=f"inventory artifact {position}",
            )
        )
        path = _safe_relative_path(row["path"], label="artifact path")
        if path in observed_paths:
            raise StudyDataError(f"inventory repeats artifact path {path!r}")
        if prior_path is not None and path.encode("utf-8") <= prior_path.encode("utf-8"):
            raise StudyDataError("inventory artifact paths must be bytewise sorted")
        observed_paths.add(path)
        prior_path = path
        role = row["role"]
        visibility = row["visibility"]
        if role in _ONLINE_PROJECTION_ROLES:
            expected_visibility = (
                "protocol"
                if role
                in {
                    "query-partition-structural-exclusions",
                    "registered-cohort-exclusions",
                }
                else "online"
            )
            if visibility != expected_visibility:
                raise StudyDataError(
                    f"projected role {role!r} must have visibility {expected_visibility!r}"
                )
            projected.append(row)
        elif role in _OUTCOME_PAYLOAD_ROLES:
            if visibility not in {"online", "custody"}:
                raise StudyDataError(f"outcome role {role!r} has an unknown visibility")
        else:
            raise StudyDataError(f"inventory role {role!r} lacks a closed projection decision")
    if not projected:
        raise StudyDataError("online projection selected no artifacts")
    projected_paths = {row["path"] for row in projected}
    required_roles = {
        "assignments",
        "corpus",
        "queries",
        "query-partition-structural-exclusions",
    }
    if not required_roles.issubset({row["role"] for row in projected}):
        raise StudyDataError("online projection omits a required label-free role")
    if any(
        row["role"] in _OUTCOME_PAYLOAD_ROLES or row["path"] not in projected_paths
        for row in projected
    ):
        raise StudyDataError("online projection selected an outcome-bearing artifact")
    return tuple(projected), len(values)


def _projection_artifact_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_bytes([dict(row) for row in rows]))


def _projection_receipt_payload(
    *,
    source_inventory_sha256: str,
    source_artifact_count: int,
    projected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "projected_artifact_count": len(projected),
        "projected_artifact_set_sha256": _projection_artifact_set_sha256(projected),
        "projected_artifacts": [dict(row) for row in projected],
        "projection_policy": ONLINE_PROJECTION_POLICY,
        "schema_version": ONLINE_PROJECTION_SCHEMA,
        "source_artifact_count": source_artifact_count,
        "source_inventory_sha256": source_inventory_sha256,
    }


def _write_projection_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise StudyDataError(f"cannot create projection file {path}: {exc}") from exc
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise StudyDataError(f"short write while creating projection file {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_projection_artifact(
    source_root: Path,
    target_root: Path,
    row: Mapping[str, Any],
) -> None:
    relative = _safe_relative_path(row["path"], label="projection artifact path")
    source = source_root.joinpath(*PurePosixPath(relative).parts)
    target = target_root.joinpath(*PurePosixPath(relative).parts)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_descriptor = _open_regular(source)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        target_descriptor = os.open(target, flags, 0o600)
    except OSError as exc:
        os.close(source_descriptor)
        raise StudyDataError(f"cannot create projection artifact {relative!r}: {exc}") from exc
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    last_byte: int | None = None
    try:
        while True:
            encoded = os.read(source_descriptor, _CHUNK_BYTES)
            if not encoded:
                break
            digest.update(encoded)
            byte_count += len(encoded)
            record_count += encoded.count(b"\n")
            last_byte = encoded[-1]
            view = memoryview(encoded)
            while view:
                written = os.write(target_descriptor, view)
                if written <= 0:
                    raise StudyDataError(f"short write while copying {relative!r}")
                view = view[written:]
        os.fsync(target_descriptor)
    finally:
        os.close(source_descriptor)
        os.close(target_descriptor)
    if (
        byte_count != row["byte_count"]
        or record_count != row["record_count"]
        or digest.hexdigest() != row["sha256"]
        or last_byte not in {None, 10}
    ):
        raise StudyDataError(f"projection source {relative!r} differs from its inventory pin")


def verify_online_staging_projection(
    root: str | Path,
    *,
    expected_inventory_sha256: str | None = None,
) -> OnlineProjectionReceipt:
    """Verify exact projection membership while admitting no label payload file."""

    package_root = Path(root)
    try:
        root_stat = package_root.lstat()
    except OSError as exc:
        raise StudyDataError(f"cannot inspect online projection {package_root}: {exc}") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise StudyDataError("online projection root must be a real directory")
    observed_paths = _scan_package_tree(package_root)
    inventory_bytes = _read_regular_file(package_root / "inventory.json")
    checksum_bytes = _read_regular_file(package_root / "inventory.sha256")
    inventory_sha256 = _sha256_bytes(inventory_bytes)
    if expected_inventory_sha256 is not None:
        if _SHA256.fullmatch(expected_inventory_sha256) is None:
            raise StudyDataError("expected projection inventory SHA-256 is invalid")
        if inventory_sha256 != expected_inventory_sha256:
            raise StudyDataError("online projection inventory differs from its expected pin")
    if checksum_bytes != f"{inventory_sha256}  inventory.json\n".encode("ascii"):
        raise StudyDataError("online projection inventory checksum differs")
    if not inventory_bytes.endswith(b"\n"):
        raise StudyDataError("online projection inventory lacks its canonical newline")
    inventory = _decode_json(inventory_bytes, label="online projection inventory")
    if _canonical_bytes(inventory) + b"\n" != inventory_bytes:
        raise StudyDataError("online projection inventory is not canonical JSON")
    projected, source_artifact_count = _online_projection_artifacts(inventory)

    receipt_path = package_root / ONLINE_PROJECTION_RECEIPT_FILENAME
    receipt_bytes = _read_regular_file(receipt_path)
    if not receipt_bytes.endswith(b"\n"):
        raise StudyDataError("online projection receipt lacks its canonical newline")
    receipt_value = _decode_json(receipt_bytes, label="online projection receipt")
    receipt = _closed_mapping(
        receipt_value,
        fields=_ONLINE_PROJECTION_RECEIPT_FIELDS,
        label="online projection receipt",
    )
    expected_receipt = _projection_receipt_payload(
        source_inventory_sha256=inventory_sha256,
        source_artifact_count=source_artifact_count,
        projected=projected,
    )
    if (
        dict(receipt) != expected_receipt
        or _canonical_bytes(expected_receipt) + b"\n" != receipt_bytes
    ):
        raise StudyDataError("online projection receipt differs from the closed projection policy")

    projected_paths = {row["path"] for row in projected}
    expected_paths = projected_paths | {
        "inventory.json",
        "inventory.sha256",
        ONLINE_PROJECTION_RECEIPT_FILENAME,
    }
    if observed_paths != expected_paths:
        raise StudyDataError(
            "online projection membership changed; "
            f"missing={sorted(expected_paths - observed_paths)}, "
            f"unexpected={sorted(observed_paths - expected_paths)}"
        )
    for row in projected:
        relative = row["path"]
        byte_count, sha256, record_count, canonical_ending = _artifact_fingerprint(
            package_root.joinpath(*PurePosixPath(relative).parts)
        )
        if (
            byte_count != row["byte_count"]
            or sha256 != row["sha256"]
            or record_count != row["record_count"]
            or not canonical_ending
        ):
            raise StudyDataError(f"projected artifact {relative!r} differs from its source pin")
    return OnlineProjectionReceipt(
        output_root=package_root,
        inventory_sha256=inventory_sha256,
        source_artifact_count=source_artifact_count,
        projected_artifact_count=len(projected),
        projected_artifact_set_sha256=_projection_artifact_set_sha256(projected),
    )


def project_online_staging(
    source_root: str | Path,
    output_root: str | Path,
) -> OnlineProjectionReceipt:
    """Copy only corpus, query, assignment, and partition-control payloads."""

    source = Path(source_root)
    verified_source = verify_staged_data(source)
    output = Path(output_root)
    if os.path.lexists(output):
        raise StudyDataError(f"online projection already exists: {output}")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.projection-", dir=output.parent))
    try:
        inventory_bytes = _read_regular_file(source / "inventory.json")
        checksum_bytes = _read_regular_file(source / "inventory.sha256")
        inventory = _decode_json(inventory_bytes, label="source inventory")
        projected, source_artifact_count = _online_projection_artifacts(inventory)
        _write_projection_bytes(temporary / "inventory.json", inventory_bytes)
        _write_projection_bytes(temporary / "inventory.sha256", checksum_bytes)
        for row in projected:
            _copy_projection_artifact(source, temporary, row)
        receipt = _projection_receipt_payload(
            source_inventory_sha256=verified_source.inventory_sha256,
            source_artifact_count=source_artifact_count,
            projected=projected,
        )
        _write_projection_bytes(
            temporary / ONLINE_PROJECTION_RECEIPT_FILENAME,
            _canonical_bytes(receipt) + b"\n",
        )
        verified = verify_online_staging_projection(
            temporary,
            expected_inventory_sha256=verified_source.inventory_sha256,
        )
        if os.path.lexists(output):
            raise StudyDataError("online projection output appeared before publication")
        os.rename(temporary, output)
        return OnlineProjectionReceipt(
            output_root=output,
            inventory_sha256=verified.inventory_sha256,
            source_artifact_count=verified.source_artifact_count,
            projected_artifact_count=verified.projected_artifact_count,
            projected_artifact_set_sha256=verified.projected_artifact_set_sha256,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.study_data",
        description="Build or verify the deterministic confirmatory data package.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage_parser = subparsers.add_parser("stage", help="build a new package without overwrite")
    stage_parser.add_argument("--config", required=True, type=Path)
    stage_parser.add_argument("--output", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify", help="verify a staged package")
    verify_parser.add_argument("--root", required=True, type=Path)
    project_parser = subparsers.add_parser(
        "project-online",
        help="copy the closed non-label-bearing online projection",
    )
    project_parser.add_argument("--source-root", required=True, type=Path)
    project_parser.add_argument("--output", required=True, type=Path)
    verify_online_parser = subparsers.add_parser(
        "verify-online",
        help="verify exact online-projection membership and source pins",
    )
    verify_online_parser.add_argument("--root", required=True, type=Path)
    verify_online_parser.add_argument("--expected-inventory-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "stage":
            receipt = stage_study_data(arguments.config, arguments.output)
        elif arguments.command == "verify":
            receipt = verify_staged_data(arguments.root)
        elif arguments.command == "project-online":
            receipt = project_online_staging(arguments.source_root, arguments.output)
        else:
            receipt = verify_online_staging_projection(
                arguments.root,
                expected_inventory_sha256=arguments.expected_inventory_sha256,
            )
    except StudyDataError as exc:
        parser.exit(2, f"study-data: {exc}\n")
    print(
        json.dumps(
            {
                "artifact_count": (
                    receipt.artifact_count
                    if isinstance(receipt, StagingReceipt)
                    else receipt.projected_artifact_count
                ),
                "inventory_sha256": receipt.inventory_sha256,
                "output_root": str(receipt.output_root),
                "source_count": (
                    receipt.source_count
                    if isinstance(receipt, StagingReceipt)
                    else receipt.source_artifact_count
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
