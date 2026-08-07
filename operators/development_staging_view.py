#!/usr/bin/env python3
"""Build the label-payload-excluded view used for development cohort selection."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import stat
import sys
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

VIEW_RECEIPT_SCHEMA = "fractal-development-staging-view-receipt-v2"
VIEW_ARTIFACT_SCHEMA = "fractal-development-staging-view-artifact-v1"
CLI_RESULT_SCHEMA = "fractal-development-staging-view-cli-result-v1"
VIEW_RECEIPT_FILENAME = "development-staging-view-receipt.json"
INPUT_CUSTODY_CONTRACT = "fractal-exclusive-posix-advisory-custody-v1"

INVENTORY_SCHEMA = "fractal-study-data-inventory-v2"
PROJECTION_SCHEMA = "fractal-online-staging-projection-v1"
PROJECTION_POLICY = "corpus-query-assignment-controls-only-v1"
PROJECTION_RECEIPT_FILENAME = "projection-receipt.json"
PARTITION_AUDIT_SCHEMA = "fractal-scalable-query-partition-audit-v1"
PARTITION_AUDIT_ALGORITHM_SHA256 = (
    "2bc1c02e51d2fa92d3b1d37db35f74504191f6a4042843b42ee5f72c4780a892"
)
NEAR_DUPLICATE_CONFIG_SHA256 = (
    "sha256:f85961157428295f2d254e172a8a9582ce8d48dcfe31bb06f8248b4b6f1bbd9f"
)
ASSIGNMENT_ALGORITHM = "component-ranked-sha256-v2"

FIXED_CORPORA = (
    "scifact",
    "hotpotqa-fullwiki",
    "t2-ragbench",
    "bright",
    "miracl-transfer",
)
DEVELOPMENT_SOURCE_STAGES = ("fit", "calibration")
REGISTERED_STAGES = ("fit", "calibration", "sealed")

STRUCTURAL_EXCLUSION_RULE_ID = "source-split-component-isolation-v1"
STRUCTURAL_EXCLUSION_REASON = "cross-source-split-component"
STRUCTURAL_EXCLUSION_POLICY = "exclude-entire-component-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 256 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_FORBIDDEN_OUTPUT_TOKENS = frozenset(
    {
        "custody",
        "evidence",
        "evidence-bundles",
        "heldout",
        "holdout",
        "label",
        "labels",
        "outcome",
        "outcomes",
        "qrel",
        "qrels",
        "result",
        "results",
        "sealed",
    }
)
_PROJECTED_ROLES = frozenset(
    {
        "assignments",
        "corpus",
        "corpus-shard",
        "queries",
        "query-partition-structural-exclusions",
        "registered-cohort-exclusions",
    }
)
_OUTCOME_ROLES = frozenset({"evidence-bundles", "qrels"})
_OUTCOME_PATH_TOKENS = _FORBIDDEN_OUTPUT_TOKENS - {"sealed"}
_INVENTORY_FIELDS = frozenset(
    {
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
)
_ASSIGNMENT_ALGORITHM_FIELDS = frozenset(
    {
        "component_edges",
        "cross_source_split_policy",
        "fit_calibration_component_ratio",
        "name",
        "three_way_component_ratio",
    }
)
_SOURCE_ARTIFACT_FIELDS = frozenset(
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
_PROJECTION_FIELDS = frozenset(
    {
        "projected_artifact_count",
        "projected_artifact_set_sha256",
        "projected_artifacts",
        "projection_policy",
        "schema_version",
        "source_artifact_count",
        "source_inventory_sha256",
    }
)
_AUDIT_FIELDS = frozenset(
    {
        "algorithm_sha256",
        "assignment_artifact_sha256",
        "assignment_component_count",
        "assignment_count",
        "assignment_seed_sha256",
        "audit_component_count",
        "component_membership_sha256",
        "corpus_artifact_count",
        "cross_stage_component_count",
        "exact_text_edge_count",
        "near_duplicate_config_sha256",
        "near_duplicate_edge_count",
        "normalized_text_membership_sha256",
        "positive_document_content_membership_sha256",
        "positive_document_membership_sha256",
        "positive_qrel_count",
        "qrel_artifact_count",
        "qrel_count",
        "query_artifact_count",
        "query_count",
        "query_counts",
        "query_coverage_sha256",
        "schema_version",
        "shared_positive_document_content_edge_count",
        "shared_positive_document_edge_count",
        "source_artifact_set_sha256",
        "source_artifacts",
        "staged_inventory_sha256",
        "staging_config_sha256",
        "structural_exclusion_artifact_sha256",
        "structural_exclusion_component_count",
        "structural_exclusion_counts",
        "structural_exclusion_membership_sha256",
        "structural_exclusion_query_count",
    }
)
_QUERY_COUNT_FIELDS = frozenset({"dataset", "query_count", "stage"})
_STRUCTURAL_EXCLUSION_COUNT_FIELDS = frozenset(
    {"component_count", "dataset", "query_count", "reason", "rule_id"}
)
_ARTIFACT_FIELDS = frozenset(
    {
        "byte_count",
        "dataset",
        "path",
        "record_count",
        "role",
        "schema_version",
        "sha256",
        "stage",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "artifacts",
        "assignment_artifact_sha256",
        "input_custody",
        "output_root",
        "partition_audit_file_sha256",
        "partition_audit_path",
        "partition_component_membership_sha256",
        "partition_source_artifact_set_sha256",
        "projected_artifact_set_sha256",
        "projection_receipt_sha256",
        "schema_version",
        "source_projection_root",
        "staged_inventory_sha256",
        "view_artifact_set_sha256",
    }
)
_INPUT_CUSTODY_FIELDS = frozenset(
    {
        "capture_set_sha256",
        "contract",
        "noncooperating_same_uid_mutation_excluded",
        "producer_parent_and_file_leases_held_through_publication",
    }
)
_CORPUS_SHARD_PATH = re.compile(r"^datasets/([^/]+)/corpus/part-[0-9]{5}\.jsonl$")
_FILE_STABLE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_gid",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_DIRECTORY_STABLE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_gid",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


class DevelopmentStagingViewError(RuntimeError):
    """Raised when the host-side development view cannot remain label-payload-excluded."""


class DevelopmentStagingPublicationIndeterminate(DevelopmentStagingViewError):
    """Raised when a post-rename failure cannot be rolled back and proved."""


class DevelopmentStagingInterruptedError(BaseException):
    """Raised by the transaction-local handlers for termination signals."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"development staging interrupted by signal {signum}")
        self.signum = signum


class _TransactionSignalGuard:
    """Translate process-control signals into transaction-visible exceptions."""

    def __init__(self) -> None:
        self._previous: dict[int, Any] = {}

    @staticmethod
    def _interrupt(signum: int, _frame: object) -> None:
        raise DevelopmentStagingInterruptedError(signum)

    def __enter__(self) -> _TransactionSignalGuard:
        if threading.current_thread() is not threading.main_thread():
            return self
        for signum in (
            signal.SIGINT,
            signal.SIGTERM,
            getattr(signal, "SIGHUP", None),
        ):
            if signum is None:
                continue
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._interrupt)
        return self

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> bool:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        return False


class _DevelopmentStagingPublicationRolledBack(DevelopmentStagingViewError):
    """Internal signal that publication was rolled back but still needs cleanup."""


@dataclass(frozen=True)
class _PublicationNameState:
    temporary: os.stat_result | None
    output: os.stat_result | None


@dataclass(frozen=True)
class DevelopmentStagingInputCustody:
    """Cooperative producer custody asserted while this view is published."""

    capture_set_sha256: str
    contract: str = INPUT_CUSTODY_CONTRACT
    noncooperating_same_uid_mutation_excluded: bool = True
    producer_parent_and_file_leases_held_through_publication: bool = True

    def __post_init__(self) -> None:
        _require_sha256("input custody capture-set SHA-256", self.capture_set_sha256)
        if self.contract != INPUT_CUSTODY_CONTRACT:
            raise DevelopmentStagingViewError("input custody contract differs")
        if self.noncooperating_same_uid_mutation_excluded is not True:
            raise DevelopmentStagingViewError(
                "input custody must exclude noncooperating same-UID mutation"
            )
        if self.producer_parent_and_file_leases_held_through_publication is not True:
            raise DevelopmentStagingViewError(
                "input custody must retain producer parent and file leases"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "capture_set_sha256": self.capture_set_sha256,
            "contract": self.contract,
            "noncooperating_same_uid_mutation_excluded": (
                self.noncooperating_same_uid_mutation_excluded
            ),
            "producer_parent_and_file_leases_held_through_publication": (
                self.producer_parent_and_file_leases_held_through_publication
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentStagingInputCustody:
        row = _closed(value, _INPUT_CUSTODY_FIELDS, label="development staging input custody")
        return cls(
            capture_set_sha256=row["capture_set_sha256"],
            contract=row["contract"],
            noncooperating_same_uid_mutation_excluded=row[
                "noncooperating_same_uid_mutation_excluded"
            ],
            producer_parent_and_file_leases_held_through_publication=row[
                "producer_parent_and_file_leases_held_through_publication"
            ],
        )


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8", errors="strict")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DevelopmentStagingViewError("control data must be finite canonical JSON") from exc


def _canonical_value_bytes(value: object) -> bytes:
    return _canonical_bytes(value)[:-1]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decode(encoded: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DevelopmentStagingViewError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise DevelopmentStagingViewError(f"{label} contains non-finite value {value!r}")

    try:
        return json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise DevelopmentStagingViewError(f"{label} must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise DevelopmentStagingViewError(f"{label} must be JSON: {exc.msg}") from exc


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DevelopmentStagingViewError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise DevelopmentStagingViewError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DevelopmentStagingViewError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DevelopmentStagingViewError(f"{name} must be an integer >= {minimum}")
    return value


def _require_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise DevelopmentStagingViewError(f"{name} must be canonical non-empty text")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DevelopmentStagingViewError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or unicodedata.normalize("NFC", value) != value
    ):
        raise DevelopmentStagingViewError(f"{label} must be a canonical relative POSIX path")
    return value


def _path_tokens(value: str) -> frozenset[str]:
    tokens: set[str] = set()
    for part in PurePosixPath(value).parts:
        folded = part.casefold()
        tokens.add(folded)
        tokens.add(PurePosixPath(folded).stem)
    return frozenset(tokens)


def _view_relative_path(value: object, *, label: str) -> str:
    path = _relative_path(value, label=label)
    if _path_tokens(path) & _FORBIDDEN_OUTPUT_TOKENS:
        raise DevelopmentStagingViewError(f"{label} crosses a forbidden payload boundary")
    return path


def _absolute_path(value: str | Path, *, label: str) -> Path:
    text = str(value)
    if not text or "\\" in text or text.startswith("//"):
        raise DevelopmentStagingViewError(f"{label} must be an absolute canonical POSIX path")
    path = Path(text)
    if (
        not path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or unicodedata.normalize("NFC", text) != text
    ):
        raise DevelopmentStagingViewError(f"{label} must be an absolute canonical POSIX path")
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise DevelopmentStagingViewError(f"cannot resolve {label}: {exc}") from exc
    if resolved != path:
        raise DevelopmentStagingViewError(f"{label} crosses a symbolic-link alias")
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _require_nonroot() -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise DevelopmentStagingViewError("development staging operator refuses root execution")


def _require_owned_directory_metadata(
    metadata: os.stat_result,
    *,
    label: str,
    private: bool,
    read_only: bool,
) -> None:
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise DevelopmentStagingViewError(f"{label} must be a real directory")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise DevelopmentStagingViewError(f"{label} must be owned by the operator identity")
    mode = stat.S_IMODE(metadata.st_mode)
    if read_only and mode & 0o222:
        raise DevelopmentStagingViewError(f"{label} must be read-only")
    if private:
        if mode & 0o077:
            raise DevelopmentStagingViewError(
                f"{label} must grant no permissions to group or other identities"
            )
    elif mode & 0o022:
        raise DevelopmentStagingViewError(
            f"{label} must not be writable by group or other identities"
        )


def _require_owned_regular(
    metadata: os.stat_result,
    *,
    label: str,
    private: bool,
    read_only: bool = False,
) -> None:
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise DevelopmentStagingViewError(f"{label} must be a regular file")
    if metadata.st_nlink != 1:
        raise DevelopmentStagingViewError(f"{label} must have exactly one hard link")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise DevelopmentStagingViewError(f"{label} must be owned by the operator identity")
    mode = stat.S_IMODE(metadata.st_mode)
    if read_only and mode & 0o222:
        raise DevelopmentStagingViewError(f"{label} must be read-only")
    if private:
        if mode & 0o077:
            raise DevelopmentStagingViewError(
                f"{label} must grant no permissions to group or other identities"
            )
    elif mode & 0o022:
        raise DevelopmentStagingViewError(
            f"{label} must not be writable by group or other identities"
        )


def _require_exact_mode(
    metadata: os.stat_result,
    *,
    expected: int,
    label: str,
) -> None:
    observed = stat.S_IMODE(metadata.st_mode)
    if observed != expected:
        raise DevelopmentStagingViewError(
            f"{label} mode must be {expected:04o}, observed {observed:04o}"
        )


def _metadata_equal(
    left: os.stat_result,
    right: os.stat_result,
    *,
    fields: Sequence[str],
) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _require_stable_file(
    before: os.stat_result,
    after: os.stat_result,
    *,
    label: str,
) -> None:
    if not _metadata_equal(before, after, fields=_FILE_STABLE_FIELDS):
        raise DevelopmentStagingViewError(f"{label} changed while it was being read")


def _require_stable_directory(
    before: os.stat_result,
    after: os.stat_result,
    *,
    label: str,
) -> None:
    if not _metadata_equal(before, after, fields=_DIRECTORY_STABLE_FIELDS):
        raise DevelopmentStagingViewError(f"{label} changed during the operation")


def _directory_open_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _regular_open_flags() -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _open_absolute_directory(
    path: Path,
    *,
    label: str,
    private: bool | None,
    read_only: bool = False,
) -> tuple[int, os.stat_result]:
    """Open an absolute directory one component at a time without following links."""

    flags = _directory_open_flags()
    descriptor: int | None = None
    try:
        descriptor = os.open("/", flags)
        for component in path.parts[1:]:
            before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise DevelopmentStagingViewError(
                    f"{label} component {component!r} is not a real directory"
                )
            child = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            _require_stable_directory(
                before,
                opened,
                label=f"{label} component {component!r}",
            )
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if private is not None:
            _require_owned_directory_metadata(
                metadata,
                label=label,
                private=private,
                read_only=read_only,
            )
        result = descriptor
        descriptor = None
        return result, metadata
    except OSError as exc:
        raise DevelopmentStagingViewError(
            f"cannot open {label} by descriptor-relative traversal: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_relative_directory(
    root_descriptor: int,
    parts: Sequence[str],
    *,
    label: str,
    private: bool,
    read_only: bool,
) -> tuple[int, os.stat_result]:
    descriptor = os.dup(root_descriptor)
    try:
        metadata = os.fstat(descriptor)
        for component in parts:
            before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise DevelopmentStagingViewError(
                    f"{label} component {component!r} is not a real directory"
                )
            child = os.open(component, _directory_open_flags(), dir_fd=descriptor)
            opened = os.fstat(child)
            _require_stable_directory(
                before,
                opened,
                label=f"{label} component {component!r}",
            )
            _require_owned_directory_metadata(
                opened,
                label=f"{label} component {component!r}",
                private=private,
                read_only=read_only,
            )
            os.close(descriptor)
            descriptor = child
            metadata = opened
        result = descriptor
        descriptor = -1
        return result, metadata
    except OSError as exc:
        raise DevelopmentStagingViewError(f"cannot open {label}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_relative_regular(
    root_descriptor: int,
    relative: str,
    *,
    label: str,
    private: bool,
    read_only: bool,
) -> tuple[int, os.stat_result]:
    parts = PurePosixPath(_relative_path(relative, label=label)).parts
    parent_descriptor, _ = _open_relative_directory(
        root_descriptor,
        parts[:-1],
        label=f"{label} parent",
        private=private,
        read_only=read_only,
    )
    descriptor: int | None = None
    try:
        before = os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
        _require_owned_regular(
            before,
            label=label,
            private=private,
            read_only=read_only,
        )
        descriptor = os.open(parts[-1], _regular_open_flags(), dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        _require_owned_regular(
            opened,
            label=label,
            private=private,
            read_only=read_only,
        )
        _require_stable_file(before, opened, label=label)
        result = descriptor
        descriptor = None
        return result, opened
    except OSError as exc:
        raise DevelopmentStagingViewError(f"cannot open {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _read_open_regular(
    descriptor: int,
    before: os.stat_result,
    *,
    maximum: int,
    label: str,
) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(descriptor, min(_CHUNK_BYTES, maximum + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > maximum:
            raise DevelopmentStagingViewError(f"{label} exceeds {maximum} bytes")
    after = os.fstat(descriptor)
    _require_stable_file(before, after, label=label)
    return b"".join(chunks)


def _read_relative_regular_with_metadata(
    root_descriptor: int,
    relative: str,
    *,
    maximum: int,
    label: str,
    private: bool,
    read_only: bool,
) -> tuple[bytes, os.stat_result]:
    descriptor, before = _open_relative_regular(
        root_descriptor,
        relative,
        label=label,
        private=private,
        read_only=read_only,
    )
    try:
        encoded = _read_open_regular(
            descriptor,
            before,
            maximum=maximum,
            label=label,
        )
        metadata = os.fstat(descriptor)
        _require_stable_file(before, metadata, label=label)
        return encoded, metadata
    finally:
        os.close(descriptor)


def _read_relative_regular(
    root_descriptor: int,
    relative: str,
    *,
    maximum: int,
    label: str,
    private: bool,
    read_only: bool,
) -> bytes:
    encoded, _metadata = _read_relative_regular_with_metadata(
        root_descriptor,
        relative,
        maximum=maximum,
        label=label,
        private=private,
        read_only=read_only,
    )
    return encoded


def _read_absolute_regular(
    path: Path,
    *,
    maximum: int,
    label: str,
    private_parent: bool,
    read_only: bool,
) -> bytes:
    parent_descriptor, _ = _open_absolute_directory(
        path.parent,
        label=f"{label} parent",
        private=private_parent,
    )
    try:
        return _read_relative_regular(
            parent_descriptor,
            path.name,
            maximum=maximum,
            label=label,
            private=False,
            read_only=read_only,
        )
    finally:
        os.close(parent_descriptor)


def _read_locked_absolute_regular(
    path: Path,
    *,
    maximum: int,
    label: str,
    private_parent: bool,
    read_only: bool,
    leases: _ExclusiveLeaseSet,
) -> bytes:
    parent_descriptor, _ = _open_absolute_directory(
        path.parent,
        label=f"{label} parent",
        private=private_parent,
    )
    file_descriptor: int | None = None
    try:
        file_descriptor, before = _open_relative_regular(
            parent_descriptor,
            path.name,
            label=label,
            private=False,
            read_only=read_only,
        )
        retained_file = leases.retain_owned(
            file_descriptor,
            label=label,
        )
        file_descriptor = None
        leases.retain_owned(
            parent_descriptor,
            label=f"{label} parent",
        )
        parent_descriptor = -1
        return _read_open_regular(
            retained_file,
            before,
            maximum=maximum,
            label=label,
        )
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _open_or_create_private_parent(
    root_descriptor: int,
    relative: str,
) -> tuple[int, str]:
    parts = PurePosixPath(_view_relative_path(relative, label="view artifact path")).parts
    descriptor = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            try:
                before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                created = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                _require_owned_directory_metadata(
                    created,
                    label=f"temporary directory {component!r}",
                    private=True,
                    read_only=False,
                )
                os.chmod(
                    component,
                    0o700,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if not _same_inode(created, before):
                    raise DevelopmentStagingViewError(
                        f"temporary directory {component!r} changed during mode normalization"
                    )
            _require_owned_directory_metadata(
                before,
                label=f"temporary directory {component!r}",
                private=True,
                read_only=False,
            )
            _require_exact_mode(
                before,
                expected=0o700,
                label=f"temporary directory {component!r}",
            )
            child = os.open(component, _directory_open_flags(), dir_fd=descriptor)
            opened = os.fstat(child)
            _require_stable_directory(
                before,
                opened,
                label=f"temporary directory {component!r}",
            )
            _require_owned_directory_metadata(
                opened,
                label=f"temporary directory {component!r}",
                private=True,
                read_only=False,
            )
            _require_exact_mode(
                opened,
                expected=0o700,
                label=f"temporary directory {component!r}",
            )
            os.close(descriptor)
            descriptor = child
        result = descriptor
        descriptor = -1
        return result, parts[-1]
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_exclusive_at(root_descriptor: int, relative: str, encoded: bytes) -> None:
    parent_descriptor, name = _open_or_create_private_parent(root_descriptor, relative)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
        os.fchmod(descriptor, 0o600)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise DevelopmentStagingViewError(
                    f"short write while creating view artifact {relative!r}"
                )
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        _require_owned_regular(
            metadata,
            label=f"view artifact {relative!r}",
            private=True,
        )
        _require_exact_mode(
            metadata,
            expected=0o600,
            label=f"view artifact {relative!r}",
        )
        if metadata.st_size != len(encoded):
            raise DevelopmentStagingViewError(
                f"view artifact {relative!r} has an unexpected size after writing"
            )
        os.fsync(parent_descriptor)
    except OSError as exc:
        raise DevelopmentStagingViewError(
            f"cannot create view artifact {relative!r}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _fsync_directory(descriptor: int, *, label: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise DevelopmentStagingViewError(f"cannot fsync {label}: {exc}") from exc


@dataclass(frozen=True, order=True)
class SourceArtifact:
    """One exact artifact row shared by inventory, projection, and audit."""

    path: str
    sha256: str
    byte_count: int
    record_count: int
    dataset: str | None
    stage: str | None
    role: str
    visibility: str

    def __post_init__(self) -> None:
        _relative_path(self.path, label="source artifact path")
        _require_sha256("source artifact SHA-256", self.sha256)
        _require_integer("source artifact byte_count", self.byte_count)
        _require_integer("source artifact record_count", self.record_count)
        if self.dataset is not None:
            _require_text("source artifact dataset", self.dataset)
        if self.stage is not None:
            _require_text("source artifact stage", self.stage)
        _require_text("source artifact role", self.role)
        _require_text("source artifact visibility", self.visibility)

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "dataset": self.dataset,
            "path": self.path,
            "record_count": self.record_count,
            "role": self.role,
            "sha256": self.sha256,
            "stage": self.stage,
            "visibility": self.visibility,
        }

    @classmethod
    def from_dict(cls, value: object, *, label: str) -> SourceArtifact:
        row = _closed(value, _SOURCE_ARTIFACT_FIELDS, label=label)
        return cls(
            path=row["path"],
            sha256=row["sha256"],
            byte_count=row["byte_count"],
            record_count=row["record_count"],
            dataset=row["dataset"],
            stage=row["stage"],
            role=row["role"],
            visibility=row["visibility"],
        )


@dataclass(frozen=True, order=True)
class PartitionQueryCount:
    """Exact query count for one audit corpus/stage stratum."""

    dataset: str
    stage: str
    query_count: int

    def __post_init__(self) -> None:
        _require_text("partition query-count dataset", self.dataset)
        if self.stage not in REGISTERED_STAGES:
            raise DevelopmentStagingViewError("partition query-count stage is not registered")
        _require_integer("partition query_count", self.query_count, minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "query_count": self.query_count,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, value: object) -> PartitionQueryCount:
        row = _closed(value, _QUERY_COUNT_FIELDS, label="partition query-count row")
        return cls(
            dataset=row["dataset"],
            stage=row["stage"],
            query_count=row["query_count"],
        )


@dataclass(frozen=True, order=True)
class StructuralExclusionCount:
    """Registered structural exclusions for one corpus."""

    dataset: str
    rule_id: str
    reason: str
    query_count: int
    component_count: int

    def __post_init__(self) -> None:
        _require_text("structural exclusion dataset", self.dataset)
        if self.rule_id != STRUCTURAL_EXCLUSION_RULE_ID:
            raise DevelopmentStagingViewError("structural exclusion rule differs")
        if self.reason != STRUCTURAL_EXCLUSION_REASON:
            raise DevelopmentStagingViewError("structural exclusion reason differs")
        _require_integer("structural exclusion query_count", self.query_count, minimum=1)
        _require_integer(
            "structural exclusion component_count",
            self.component_count,
            minimum=1,
        )
        if self.component_count > self.query_count:
            raise DevelopmentStagingViewError(
                "structural exclusion component count exceeds query count"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "component_count": self.component_count,
            "dataset": self.dataset,
            "query_count": self.query_count,
            "reason": self.reason,
            "rule_id": self.rule_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> StructuralExclusionCount:
        row = _closed(
            value,
            _STRUCTURAL_EXCLUSION_COUNT_FIELDS,
            label="structural exclusion count row",
        )
        return cls(
            dataset=row["dataset"],
            rule_id=row["rule_id"],
            reason=row["reason"],
            query_count=row["query_count"],
            component_count=row["component_count"],
        )


@dataclass(frozen=True)
class TypedPartitionAudit:
    """Faithful host-side port of ScalableQueryPartitionAuditReceipt."""

    staged_inventory_sha256: str
    staging_config_sha256: str
    assignment_seed_sha256: str
    algorithm_sha256: str
    near_duplicate_config_sha256: str
    source_artifacts: tuple[SourceArtifact, ...]
    source_artifact_set_sha256: str
    assignment_artifact_sha256: str
    query_counts: tuple[PartitionQueryCount, ...]
    query_artifact_count: int
    qrel_artifact_count: int
    corpus_artifact_count: int
    assignment_count: int
    query_count: int
    qrel_count: int
    positive_qrel_count: int
    assignment_component_count: int
    audit_component_count: int
    exact_text_edge_count: int
    near_duplicate_edge_count: int
    shared_positive_document_edge_count: int
    shared_positive_document_content_edge_count: int
    cross_stage_component_count: int
    structural_exclusion_artifact_sha256: str
    structural_exclusion_query_count: int
    structural_exclusion_component_count: int
    structural_exclusion_counts: tuple[StructuralExclusionCount, ...]
    structural_exclusion_membership_sha256: str
    query_coverage_sha256: str
    normalized_text_membership_sha256: str
    component_membership_sha256: str
    positive_document_membership_sha256: str
    positive_document_content_membership_sha256: str
    schema_version: str = PARTITION_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "staged_inventory_sha256",
            "staging_config_sha256",
            "assignment_seed_sha256",
            "algorithm_sha256",
            "source_artifact_set_sha256",
            "assignment_artifact_sha256",
            "query_coverage_sha256",
            "normalized_text_membership_sha256",
            "component_membership_sha256",
            "positive_document_membership_sha256",
            "structural_exclusion_artifact_sha256",
            "structural_exclusion_membership_sha256",
            "positive_document_content_membership_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.algorithm_sha256 != PARTITION_AUDIT_ALGORITHM_SHA256:
            raise DevelopmentStagingViewError("partition-audit algorithm digest differs")
        if self.near_duplicate_config_sha256 != NEAR_DUPLICATE_CONFIG_SHA256:
            raise DevelopmentStagingViewError("near-duplicate config digest differs")
        sources = tuple(sorted(self.source_artifacts, key=lambda item: item.path.encode()))
        if (
            not sources
            or not all(isinstance(item, SourceArtifact) for item in sources)
            or len({item.path for item in sources}) != len(sources)
        ):
            raise DevelopmentStagingViewError(
                "partition source_artifacts must be typed, non-empty, and unique"
            )
        if (
            _sha256(_canonical_value_bytes([source.to_dict() for source in sources]))
            != self.source_artifact_set_sha256
        ):
            raise DevelopmentStagingViewError("partition source artifact-set digest differs")
        assignment_sources = [item for item in sources if item.role == "assignments"]
        if (
            len(assignment_sources) != 1
            or assignment_sources[0].path != "assignments.jsonl"
            or assignment_sources[0].sha256 != self.assignment_artifact_sha256
        ):
            raise DevelopmentStagingViewError("partition assignment artifact binding differs")
        exclusion_sources = [
            item for item in sources if item.role == "query-partition-structural-exclusions"
        ]
        if (
            len(exclusion_sources) != 1
            or exclusion_sources[0].path != "partition-exclusions.jsonl"
            or exclusion_sources[0].visibility != "protocol"
            or exclusion_sources[0].dataset is not None
            or exclusion_sources[0].stage is not None
            or exclusion_sources[0].sha256 != self.structural_exclusion_artifact_sha256
            or exclusion_sources[0].record_count != self.structural_exclusion_query_count
        ):
            raise DevelopmentStagingViewError(
                "partition structural exclusion artifact binding differs"
            )
        query_sources = [item for item in sources if item.role == "queries"]
        qrel_sources = [item for item in sources if item.role == "qrels"]
        corpus_sources = [item for item in sources if item.role in {"corpus", "corpus-shard"}]
        if (
            len(query_sources) != self.query_artifact_count
            or len(qrel_sources) != self.qrel_artifact_count
            or len(corpus_sources) != self.corpus_artifact_count
        ):
            raise DevelopmentStagingViewError("partition source artifact counts differ")
        counts = tuple(self.query_counts)
        if (
            not counts
            or not all(isinstance(item, PartitionQueryCount) for item in counts)
            or counts != tuple(sorted(counts))
            or len({(item.dataset, item.stage) for item in counts}) != len(counts)
        ):
            raise DevelopmentStagingViewError(
                "partition query_counts must be typed, unique, and canonically sorted"
            )
        for name in (
            "query_artifact_count",
            "qrel_artifact_count",
            "assignment_count",
            "query_count",
            "qrel_count",
            "positive_qrel_count",
            "assignment_component_count",
            "audit_component_count",
        ):
            _require_integer(name, getattr(self, name), minimum=1)
        for name in (
            "corpus_artifact_count",
            "exact_text_edge_count",
            "near_duplicate_edge_count",
            "shared_positive_document_edge_count",
            "shared_positive_document_content_edge_count",
            "cross_stage_component_count",
            "structural_exclusion_query_count",
            "structural_exclusion_component_count",
        ):
            _require_integer(name, getattr(self, name))
        exclusion_counts = tuple(self.structural_exclusion_counts)
        invalid_exclusion_counts = (
            not all(isinstance(item, StructuralExclusionCount) for item in exclusion_counts)
            or exclusion_counts != tuple(sorted(exclusion_counts))
            or len({item.dataset for item in exclusion_counts}) != len(exclusion_counts)
            or sum(item.query_count for item in exclusion_counts)
            != self.structural_exclusion_query_count
            or sum(item.component_count for item in exclusion_counts)
            != self.structural_exclusion_component_count
        )
        if (
            invalid_exclusion_counts
            or (
                self.structural_exclusion_query_count == 0
                and (self.structural_exclusion_component_count != 0 or exclusion_counts)
            )
            or (
                self.structural_exclusion_query_count > 0
                and not (
                    1
                    <= self.structural_exclusion_component_count
                    <= self.structural_exclusion_query_count
                    and exclusion_counts
                )
            )
        ):
            raise DevelopmentStagingViewError(
                "partition structural exclusion counts are inconsistent"
            )
        if (
            self.assignment_count != self.query_count
            or sum(item.query_count for item in counts) != self.query_count
        ):
            raise DevelopmentStagingViewError("partition assignment/query coverage counts differ")
        if self.qrel_count < self.query_count or self.positive_qrel_count < self.query_count:
            raise DevelopmentStagingViewError("partition qrel coverage is incomplete")
        if not (
            1 <= self.audit_component_count <= self.assignment_component_count <= self.query_count
        ):
            raise DevelopmentStagingViewError("partition component counts are inconsistent")
        if self.cross_stage_component_count != 0:
            raise DevelopmentStagingViewError(
                "a passing partition audit must record zero stage crossings"
            )
        if self.schema_version != PARTITION_AUDIT_SCHEMA:
            raise DevelopmentStagingViewError("partition-audit receipt schema differs")
        object.__setattr__(self, "source_artifacts", sources)
        object.__setattr__(self, "query_counts", counts)
        object.__setattr__(self, "structural_exclusion_counts", exclusion_counts)

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm_sha256": self.algorithm_sha256,
            "assignment_artifact_sha256": self.assignment_artifact_sha256,
            "assignment_component_count": self.assignment_component_count,
            "assignment_count": self.assignment_count,
            "assignment_seed_sha256": self.assignment_seed_sha256,
            "audit_component_count": self.audit_component_count,
            "component_membership_sha256": self.component_membership_sha256,
            "corpus_artifact_count": self.corpus_artifact_count,
            "cross_stage_component_count": self.cross_stage_component_count,
            "exact_text_edge_count": self.exact_text_edge_count,
            "near_duplicate_config_sha256": self.near_duplicate_config_sha256,
            "near_duplicate_edge_count": self.near_duplicate_edge_count,
            "normalized_text_membership_sha256": (self.normalized_text_membership_sha256),
            "positive_document_content_membership_sha256": (
                self.positive_document_content_membership_sha256
            ),
            "positive_document_membership_sha256": (self.positive_document_membership_sha256),
            "positive_qrel_count": self.positive_qrel_count,
            "qrel_artifact_count": self.qrel_artifact_count,
            "qrel_count": self.qrel_count,
            "query_artifact_count": self.query_artifact_count,
            "query_count": self.query_count,
            "query_counts": [item.to_dict() for item in self.query_counts],
            "query_coverage_sha256": self.query_coverage_sha256,
            "schema_version": self.schema_version,
            "shared_positive_document_content_edge_count": (
                self.shared_positive_document_content_edge_count
            ),
            "shared_positive_document_edge_count": (self.shared_positive_document_edge_count),
            "source_artifact_set_sha256": self.source_artifact_set_sha256,
            "source_artifacts": [item.to_dict() for item in self.source_artifacts],
            "staged_inventory_sha256": self.staged_inventory_sha256,
            "staging_config_sha256": self.staging_config_sha256,
            "structural_exclusion_artifact_sha256": (self.structural_exclusion_artifact_sha256),
            "structural_exclusion_component_count": (self.structural_exclusion_component_count),
            "structural_exclusion_counts": [
                item.to_dict() for item in self.structural_exclusion_counts
            ],
            "structural_exclusion_membership_sha256": (self.structural_exclusion_membership_sha256),
            "structural_exclusion_query_count": (self.structural_exclusion_query_count),
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> TypedPartitionAudit:
        row = _closed(value, _AUDIT_FIELDS, label="typed partition-audit receipt")
        sources = row["source_artifacts"]
        counts = row["query_counts"]
        exclusions = row["structural_exclusion_counts"]
        if (
            not isinstance(sources, list)
            or not isinstance(counts, list)
            or not isinstance(exclusions, list)
        ):
            raise DevelopmentStagingViewError(
                "partition source artifacts and count rows must be arrays"
            )
        return cls(
            staged_inventory_sha256=row["staged_inventory_sha256"],
            staging_config_sha256=row["staging_config_sha256"],
            assignment_seed_sha256=row["assignment_seed_sha256"],
            algorithm_sha256=row["algorithm_sha256"],
            near_duplicate_config_sha256=row["near_duplicate_config_sha256"],
            source_artifacts=tuple(
                SourceArtifact.from_dict(item, label="partition source artifact")
                for item in sources
            ),
            source_artifact_set_sha256=row["source_artifact_set_sha256"],
            assignment_artifact_sha256=row["assignment_artifact_sha256"],
            query_counts=tuple(PartitionQueryCount.from_dict(item) for item in counts),
            query_artifact_count=row["query_artifact_count"],
            qrel_artifact_count=row["qrel_artifact_count"],
            corpus_artifact_count=row["corpus_artifact_count"],
            assignment_count=row["assignment_count"],
            query_count=row["query_count"],
            qrel_count=row["qrel_count"],
            positive_qrel_count=row["positive_qrel_count"],
            assignment_component_count=row["assignment_component_count"],
            audit_component_count=row["audit_component_count"],
            exact_text_edge_count=row["exact_text_edge_count"],
            near_duplicate_edge_count=row["near_duplicate_edge_count"],
            shared_positive_document_edge_count=row["shared_positive_document_edge_count"],
            shared_positive_document_content_edge_count=row[
                "shared_positive_document_content_edge_count"
            ],
            cross_stage_component_count=row["cross_stage_component_count"],
            structural_exclusion_artifact_sha256=row["structural_exclusion_artifact_sha256"],
            structural_exclusion_query_count=row["structural_exclusion_query_count"],
            structural_exclusion_component_count=row["structural_exclusion_component_count"],
            structural_exclusion_counts=tuple(
                StructuralExclusionCount.from_dict(item) for item in exclusions
            ),
            structural_exclusion_membership_sha256=row["structural_exclusion_membership_sha256"],
            query_coverage_sha256=row["query_coverage_sha256"],
            normalized_text_membership_sha256=row["normalized_text_membership_sha256"],
            component_membership_sha256=row["component_membership_sha256"],
            positive_document_membership_sha256=row["positive_document_membership_sha256"],
            positive_document_content_membership_sha256=row[
                "positive_document_content_membership_sha256"
            ],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True, order=True)
class ViewArtifact:
    """One byte-exact member of the published selection view."""

    path: str
    sha256: str
    byte_count: int
    record_count: int
    role: str
    dataset: str | None
    stage: str | None
    schema_version: str = VIEW_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        _view_relative_path(self.path, label="view artifact path")
        _require_sha256("view artifact SHA-256", self.sha256)
        _require_integer("view artifact byte_count", self.byte_count, minimum=1)
        _require_integer("view artifact record_count", self.record_count, minimum=1)
        _require_text("view artifact role", self.role)
        if self.dataset is not None and self.dataset not in FIXED_CORPORA:
            raise DevelopmentStagingViewError("view artifact dataset is not registered")
        if self.stage is not None and self.stage not in DEVELOPMENT_SOURCE_STAGES:
            raise DevelopmentStagingViewError("view artifact stage is not fit or calibration")
        if self.schema_version != VIEW_ARTIFACT_SCHEMA:
            raise DevelopmentStagingViewError("view artifact schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "dataset": self.dataset,
            "path": self.path,
            "record_count": self.record_count,
            "role": self.role,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, value: object) -> ViewArtifact:
        row = _closed(value, _ARTIFACT_FIELDS, label="view artifact")
        return cls(
            path=row["path"],
            sha256=row["sha256"],
            byte_count=row["byte_count"],
            record_count=row["record_count"],
            role=row["role"],
            dataset=row["dataset"],
            stage=row["stage"],
            schema_version=row["schema_version"],
        )


def _artifact_set_sha256(artifacts: Sequence[ViewArtifact]) -> str:
    return _sha256(_canonical_value_bytes([artifact.to_dict() for artifact in artifacts]))


def _input_capture_set_sha256(
    *,
    artifacts: Sequence[ViewArtifact],
    projection_receipt_sha256: str,
    partition_audit_file_sha256: str,
) -> str:
    return _sha256(
        _canonical_value_bytes(
            {
                "captured_artifacts": [
                    {
                        "byte_count": artifact.byte_count,
                        "path": artifact.path,
                        "sha256": artifact.sha256,
                    }
                    for artifact in artifacts
                ],
                "partition_audit_file_sha256": partition_audit_file_sha256,
                "projection_receipt_sha256": projection_receipt_sha256,
            }
        )
    )


@dataclass(frozen=True)
class DevelopmentStagingViewReceipt:
    """Closed identity for one label-payload-excluded development selection view."""

    source_projection_root: Path
    output_root: Path
    partition_audit_path: Path
    staged_inventory_sha256: str
    projection_receipt_sha256: str
    projected_artifact_set_sha256: str
    partition_audit_file_sha256: str
    partition_component_membership_sha256: str
    partition_source_artifact_set_sha256: str
    assignment_artifact_sha256: str
    input_custody: DevelopmentStagingInputCustody
    artifacts: tuple[ViewArtifact, ...]
    view_artifact_set_sha256: str
    schema_version: str = VIEW_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("source_projection_root", "output_root", "partition_audit_path"):
            object.__setattr__(
                self,
                name,
                _absolute_path(getattr(self, name), label=name),
            )
        for name in (
            "staged_inventory_sha256",
            "projection_receipt_sha256",
            "projected_artifact_set_sha256",
            "partition_audit_file_sha256",
            "partition_component_membership_sha256",
            "partition_source_artifact_set_sha256",
            "assignment_artifact_sha256",
            "view_artifact_set_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        artifacts = tuple(self.artifacts)
        if (
            not artifacts
            or artifacts != tuple(sorted(artifacts, key=lambda row: row.path.encode("utf-8")))
            or len({artifact.path for artifact in artifacts}) != len(artifacts)
        ):
            raise DevelopmentStagingViewError(
                "view artifacts must be non-empty, unique, and bytewise sorted"
            )
        _validate_view_contract(artifacts)
        if _artifact_set_sha256(artifacts) != self.view_artifact_set_sha256:
            raise DevelopmentStagingViewError("view artifact-set digest differs")
        assignment = [artifact for artifact in artifacts if artifact.role == "assignments"]
        if len(assignment) != 1 or assignment[0].sha256 != self.assignment_artifact_sha256:
            raise DevelopmentStagingViewError("view assignment binding differs")
        if not isinstance(self.input_custody, DevelopmentStagingInputCustody):
            raise DevelopmentStagingViewError("view input custody must be typed")
        if (
            _input_capture_set_sha256(
                artifacts=artifacts,
                projection_receipt_sha256=self.projection_receipt_sha256,
                partition_audit_file_sha256=self.partition_audit_file_sha256,
            )
            != self.input_custody.capture_set_sha256
        ):
            raise DevelopmentStagingViewError("view input custody capture-set digest differs")
        if self.schema_version != VIEW_RECEIPT_SCHEMA:
            raise DevelopmentStagingViewError("development staging view schema differs")
        object.__setattr__(self, "artifacts", artifacts)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "assignment_artifact_sha256": self.assignment_artifact_sha256,
            "input_custody": self.input_custody.to_dict(),
            "output_root": str(self.output_root),
            "partition_audit_file_sha256": self.partition_audit_file_sha256,
            "partition_audit_path": str(self.partition_audit_path),
            "partition_component_membership_sha256": (self.partition_component_membership_sha256),
            "partition_source_artifact_set_sha256": (self.partition_source_artifact_set_sha256),
            "projected_artifact_set_sha256": self.projected_artifact_set_sha256,
            "projection_receipt_sha256": self.projection_receipt_sha256,
            "schema_version": self.schema_version,
            "source_projection_root": str(self.source_projection_root),
            "staged_inventory_sha256": self.staged_inventory_sha256,
            "view_artifact_set_sha256": self.view_artifact_set_sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentStagingViewReceipt:
        row = _closed(value, _RECEIPT_FIELDS, label="development staging view receipt")
        values = row["artifacts"]
        if not isinstance(values, list):
            raise DevelopmentStagingViewError("view receipt artifacts must be an array")
        return cls(
            source_projection_root=Path(row["source_projection_root"]),
            output_root=Path(row["output_root"]),
            partition_audit_path=Path(row["partition_audit_path"]),
            staged_inventory_sha256=row["staged_inventory_sha256"],
            projection_receipt_sha256=row["projection_receipt_sha256"],
            projected_artifact_set_sha256=row["projected_artifact_set_sha256"],
            partition_audit_file_sha256=row["partition_audit_file_sha256"],
            partition_component_membership_sha256=row["partition_component_membership_sha256"],
            partition_source_artifact_set_sha256=row["partition_source_artifact_set_sha256"],
            assignment_artifact_sha256=row["assignment_artifact_sha256"],
            input_custody=DevelopmentStagingInputCustody.from_dict(row["input_custody"]),
            artifacts=tuple(ViewArtifact.from_dict(item) for item in values),
            view_artifact_set_sha256=row["view_artifact_set_sha256"],
            schema_version=row["schema_version"],
        )


def _expected_payload_contract() -> dict[str, tuple[str, str | None, str | None]]:
    contract: dict[str, tuple[str, str | None, str | None]] = {
        "assignments.jsonl": ("assignments", None, None),
    }
    for stage in DEVELOPMENT_SOURCE_STAGES:
        for corpus in FIXED_CORPORA:
            contract[f"datasets/{corpus}/{stage}/queries.jsonl"] = (
                "queries",
                corpus,
                stage,
            )
    return contract


def _expected_view_contract() -> dict[str, tuple[str, str | None, str | None]]:
    return {
        "inventory.json": ("staged-inventory", None, None),
        "inventory.sha256": ("staged-inventory-checksum", None, None),
        **_expected_payload_contract(),
    }


def _validate_view_contract(artifacts: Sequence[ViewArtifact]) -> None:
    contract = _expected_view_contract()
    by_path = {artifact.path: artifact for artifact in artifacts}
    if set(by_path) != set(contract):
        raise DevelopmentStagingViewError(
            "view artifact membership differs from the registered selection view"
        )
    for path, expected in contract.items():
        artifact = by_path[path]
        if (artifact.role, artifact.dataset, artifact.stage) != expected:
            raise DevelopmentStagingViewError(f"view artifact contract differs for {path!r}")


@dataclass(frozen=True)
class _AdmittedInputs:
    inventory_bytes: bytes
    inventory_checksum_bytes: bytes
    projected_artifact_set_sha256: str
    partition_component_membership_sha256: str
    partition_source_artifact_set_sha256: str
    assignment_artifact_sha256: str
    projection_files: frozenset[str]
    selected_sources: tuple[SourceArtifact, ...]


def _source_rows(value: object, *, label: str) -> tuple[SourceArtifact, ...]:
    if not isinstance(value, list) or not value:
        raise DevelopmentStagingViewError(f"{label} must be a non-empty array")
    rows = tuple(SourceArtifact.from_dict(item, label=f"{label} row") for item in value)
    if rows != tuple(sorted(rows, key=lambda row: row.path.encode("utf-8"))) or len(
        {row.path for row in rows}
    ) != len(rows):
        raise DevelopmentStagingViewError(f"{label} must be unique and bytewise sorted")
    return rows


def _expected_directories(expected_files: set[str]) -> set[str]:
    return {
        parent.as_posix()
        for path in expected_files
        for parent in PurePosixPath(path).parents
        if parent.as_posix() != "."
    }


def _scan_exact_tree(
    root_descriptor: int,
    *,
    expected_files: set[str],
    label: str,
    private: bool,
    read_only: bool,
    enforce_view_boundary: bool,
    exact_private_modes: bool = False,
    expected_file_metadata: Mapping[str, os.stat_result] | None = None,
    expected_directory_mode: int = 0o700,
    expected_file_mode: int = 0o600,
) -> None:
    """Fail-closed, descriptor-relative traversal of an exact file tree."""

    metadata_pins = dict(expected_file_metadata or {})
    if not set(metadata_pins).issubset(expected_files):
        raise DevelopmentStagingViewError(
            f"{label} metadata pins name files outside the expected tree"
        )
    expected_directories = _expected_directories(expected_files)
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    root_metadata = os.fstat(root_descriptor)
    _require_owned_directory_metadata(
        root_metadata,
        label=f"{label} root",
        private=private,
        read_only=read_only,
    )
    if exact_private_modes:
        _require_exact_mode(
            root_metadata,
            expected=expected_directory_mode,
            label=f"{label} root",
        )

    def scan(descriptor: int, prefix: str, directory_label: str) -> None:
        before = os.fstat(descriptor)
        try:
            names = os.listdir(descriptor)
        except OSError as exc:
            raise DevelopmentStagingViewError(f"cannot enumerate {directory_label}: {exc}") from exc
        if not all(
            isinstance(name, str)
            and name not in {"", ".", ".."}
            and "/" not in name
            and "\\" not in name
            and unicodedata.normalize("NFC", name) == name
            for name in names
        ):
            raise DevelopmentStagingViewError(
                f"{directory_label} contains a noncanonical entry name"
            )
        for name in sorted(names, key=lambda value: value.encode("utf-8")):
            relative = f"{prefix}/{name}" if prefix else name
            if enforce_view_boundary and relative != VIEW_RECEIPT_FILENAME:
                _view_relative_path(relative, label=f"{label} member")
            else:
                _relative_path(relative, label=f"{label} member")
            try:
                metadata = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise DevelopmentStagingViewError(
                    f"cannot inspect {label} member {relative!r}: {exc}"
                ) from exc
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                if relative not in expected_directories:
                    raise DevelopmentStagingViewError(
                        f"{label} contains unexpected directory {relative!r}"
                    )
                _require_owned_directory_metadata(
                    metadata,
                    label=f"{label} directory {relative!r}",
                    private=private,
                    read_only=read_only,
                )
                if exact_private_modes:
                    _require_exact_mode(
                        metadata,
                        expected=expected_directory_mode,
                        label=f"{label} directory {relative!r}",
                    )
                child: int | None = None
                try:
                    child = os.open(name, _directory_open_flags(), dir_fd=descriptor)
                    opened = os.fstat(child)
                    _require_stable_directory(
                        metadata,
                        opened,
                        label=f"{label} directory {relative!r}",
                    )
                    scan(child, relative, f"{label} directory {relative!r}")
                    _require_stable_directory(
                        opened,
                        os.fstat(child),
                        label=f"{label} directory {relative!r}",
                    )
                except OSError as exc:
                    raise DevelopmentStagingViewError(
                        f"cannot traverse {label} directory {relative!r}: {exc}"
                    ) from exc
                finally:
                    if child is not None:
                        os.close(child)
                observed_directories.add(relative)
            elif stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                if relative not in expected_files:
                    raise DevelopmentStagingViewError(
                        f"{label} contains unexpected file {relative!r}"
                    )
                _require_owned_regular(
                    metadata,
                    label=f"{label} file {relative!r}",
                    private=private,
                    read_only=read_only,
                )
                if exact_private_modes:
                    _require_exact_mode(
                        metadata,
                        expected=expected_file_mode,
                        label=f"{label} file {relative!r}",
                    )
                pinned = metadata_pins.get(relative)
                if pinned is not None:
                    _require_stable_file(
                        pinned,
                        metadata,
                        label=f"{label} file {relative!r}",
                    )
                observed_files.add(relative)
            else:
                raise DevelopmentStagingViewError(
                    f"{label} contains linked or special member {relative!r}"
                )
        _require_stable_directory(
            before,
            os.fstat(descriptor),
            label=directory_label,
        )

    scan(root_descriptor, "", label)
    if observed_files != expected_files or observed_directories != expected_directories:
        raise DevelopmentStagingViewError(
            f"{label} membership differs; "
            f"missing_files={sorted(expected_files - observed_files)}, "
            f"unexpected_files={sorted(observed_files - expected_files)}, "
            f"missing_directories={sorted(expected_directories - observed_directories)}, "
            f"unexpected_directories={sorted(observed_directories - expected_directories)}"
        )


def _validate_partition_bindings(
    *,
    audit: TypedPartitionAudit,
    inventory: Mapping[str, Any],
    inventory_rows: tuple[SourceArtifact, ...],
    inventory_sha256: str,
) -> None:
    if audit.staged_inventory_sha256 != inventory_sha256:
        raise DevelopmentStagingViewError("partition audit names another inventory")
    inventory_seed = _require_sha256(
        "inventory assignment seed",
        inventory["assignment_seed_sha256"],
    )
    inventory_config = _require_sha256(
        "inventory staging config",
        inventory["config_sha256"],
    )
    if (
        audit.assignment_seed_sha256 != inventory_seed
        or audit.staging_config_sha256 != inventory_config
    ):
        raise DevelopmentStagingViewError(
            "partition audit seed or staging config differs from the inventory"
        )

    audit_roles = {
        "assignments",
        "corpus",
        "corpus-shard",
        "qrels",
        "queries",
        "query-partition-structural-exclusions",
    }
    expected_sources = tuple(
        sorted(
            (row for row in inventory_rows if row.role in audit_roles),
            key=lambda row: row.path.encode("utf-8"),
        )
    )
    if audit.source_artifacts != expected_sources:
        raise DevelopmentStagingViewError(
            "partition audit source set differs from the inventory audit source set"
        )

    assignment_sources = [
        row
        for row in audit.source_artifacts
        if row.path == "assignments.jsonl"
        and row.role == "assignments"
        and row.dataset is None
        and row.stage is None
        and row.visibility == "online"
    ]
    exclusion_sources = [
        row
        for row in audit.source_artifacts
        if row.path == "partition-exclusions.jsonl"
        and row.role == "query-partition-structural-exclusions"
        and row.dataset is None
        and row.stage is None
        and row.visibility == "protocol"
    ]
    if len(assignment_sources) != 1 or len(exclusion_sources) != 1:
        raise DevelopmentStagingViewError(
            "partition audit lacks its exact assignment or exclusion control"
        )
    if assignment_sources[0].record_count != audit.assignment_count:
        raise DevelopmentStagingViewError(
            "partition assignment source count differs from the typed audit"
        )

    expected_strata = {(corpus, stage) for corpus in FIXED_CORPORA for stage in REGISTERED_STAGES}
    query_sources = [row for row in audit.source_artifacts if row.role == "queries"]
    qrel_sources = [row for row in audit.source_artifacts if row.role == "qrels"]
    query_by_stratum = {(row.dataset, row.stage): row for row in query_sources}
    qrel_by_stratum = {(row.dataset, row.stage): row for row in qrel_sources}
    audit_counts = {(row.dataset, row.stage): row.query_count for row in audit.query_counts}
    if (
        len(query_by_stratum) != len(query_sources)
        or len(qrel_by_stratum) != len(qrel_sources)
        or set(query_by_stratum) != expected_strata
        or set(qrel_by_stratum) != expected_strata
        or set(audit_counts) != expected_strata
    ):
        raise DevelopmentStagingViewError(
            "partition audit does not cover the registered fifteen query strata"
        )
    for corpus, stage in sorted(expected_strata):
        query = query_by_stratum[(corpus, stage)]
        qrel = qrel_by_stratum[(corpus, stage)]
        expected_query_path = (
            f"datasets/{corpus}/sealed/online/queries.jsonl"
            if stage == "sealed"
            else f"datasets/{corpus}/{stage}/queries.jsonl"
        )
        expected_qrel_path = (
            f"datasets/{corpus}/sealed/custody/qrels.jsonl"
            if stage == "sealed"
            else f"datasets/{corpus}/{stage}/qrels.jsonl"
        )
        expected_qrel_visibility = "custody" if stage == "sealed" else "online"
        if (
            query.path != expected_query_path
            or query.visibility != "online"
            or qrel.path != expected_qrel_path
            or qrel.visibility != expected_qrel_visibility
            or audit_counts[(corpus, stage)] != query.record_count
        ):
            raise DevelopmentStagingViewError(
                f"partition stratum contract differs for {corpus}:{stage}"
            )

    corpus_sources = [
        row for row in audit.source_artifacts if row.role in {"corpus", "corpus-shard"}
    ]
    corpus_by_dataset: dict[str, list[SourceArtifact]] = {}
    for source in corpus_sources:
        if (
            source.dataset not in FIXED_CORPORA
            or source.stage is not None
            or source.visibility != "online"
        ):
            raise DevelopmentStagingViewError("partition corpus source contract differs")
        if source.role == "corpus":
            if source.path != f"datasets/{source.dataset}/corpus.jsonl":
                raise DevelopmentStagingViewError("partition inline corpus path differs")
        else:
            match = _CORPUS_SHARD_PATH.fullmatch(source.path)
            if match is None or match.group(1) != source.dataset:
                raise DevelopmentStagingViewError("partition corpus-shard path differs")
        corpus_by_dataset.setdefault(source.dataset, []).append(source)
    if set(corpus_by_dataset) != set(FIXED_CORPORA):
        raise DevelopmentStagingViewError("partition corpus sources do not cover the fixed corpora")
    for corpus, sources in corpus_by_dataset.items():
        roles = {source.role for source in sources}
        if roles == {"corpus"} and len(sources) == 1:
            continue
        if roles != {"corpus-shard"}:
            raise DevelopmentStagingViewError(
                f"partition corpus representation is ambiguous for {corpus}"
            )

    if (
        sum(row.record_count for row in query_sources) != audit.query_count
        or sum(row.record_count for row in qrel_sources) != audit.qrel_count
    ):
        raise DevelopmentStagingViewError(
            "partition aggregate query or qrel count differs from its source pins"
        )

    counts_value = inventory["counts"]
    if (
        not isinstance(counts_value, Mapping)
        or not all(isinstance(key, str) for key in counts_value)
        or set(counts_value) != set(FIXED_CORPORA)
    ):
        raise DevelopmentStagingViewError(
            "inventory counts must cover exactly the five registered corpora"
        )
    exclusion_by_corpus = {
        row.dataset: row.query_count for row in audit.structural_exclusion_counts
    }
    if not set(exclusion_by_corpus).issubset(FIXED_CORPORA):
        raise DevelopmentStagingViewError(
            "partition structural exclusions name an unregistered corpus"
        )
    for corpus in FIXED_CORPORA:
        count_row = counts_value[corpus]
        if not isinstance(count_row, Mapping) or not all(isinstance(key, str) for key in count_row):
            raise DevelopmentStagingViewError(f"inventory count row for {corpus} must be an object")
        if "structural_excluded_queries" in count_row:
            raise DevelopmentStagingViewError(
                f"inventory count row for {corpus} contains forbidden structural_excluded_queries"
            )
        for stage in REGISTERED_STAGES:
            field = f"{stage}_queries"
            if (
                field not in count_row
                or _require_integer(
                    f"inventory {corpus} {field}",
                    count_row[field],
                )
                != audit_counts[(corpus, stage)]
            ):
                raise DevelopmentStagingViewError(
                    f"inventory and audit query counts differ for {corpus}:{stage}"
                )
        expected_qrels = sum(
            qrel_by_stratum[(corpus, stage)].record_count for stage in REGISTERED_STAGES
        )
        expected_documents = sum(source.record_count for source in corpus_by_dataset[corpus])
        for field, expected in (
            ("qrels", expected_qrels),
            ("documents", expected_documents),
            (
                "partition_excluded_queries",
                exclusion_by_corpus.get(corpus, 0),
            ),
        ):
            if (
                field not in count_row
                or _require_integer(
                    f"inventory {corpus} {field}",
                    count_row[field],
                )
                != expected
            ):
                raise DevelopmentStagingViewError(
                    f"inventory and audit {field} counts differ for {corpus}"
                )


def _admit_inputs(
    *,
    projection_root_descriptor: int,
    staged_inventory_sha256: str,
    projection_receipt_sha256: str,
    partition_audit_bytes: bytes,
    partition_audit_file_sha256: str,
) -> _AdmittedInputs:
    inventory_bytes = _read_relative_regular(
        projection_root_descriptor,
        "inventory.json",
        maximum=_MAX_CONTROL_BYTES,
        label="staged inventory",
        private=False,
        read_only=True,
    )
    inventory_digest = _sha256(inventory_bytes)
    if inventory_digest != staged_inventory_sha256:
        raise DevelopmentStagingViewError("staged inventory differs from its caller pin")
    inventory_checksum_bytes = _read_relative_regular(
        projection_root_descriptor,
        "inventory.sha256",
        maximum=1024,
        label="staged inventory checksum",
        private=False,
        read_only=True,
    )
    if inventory_checksum_bytes != f"{inventory_digest}  inventory.json\n".encode("ascii"):
        raise DevelopmentStagingViewError("staged inventory checksum differs")
    inventory_value = _decode(inventory_bytes, label="staged inventory")
    inventory = _closed(inventory_value, _INVENTORY_FIELDS, label="staged inventory")
    assignment_algorithm = _closed(
        inventory["assignment_algorithm"],
        _ASSIGNMENT_ALGORITHM_FIELDS,
        label="staged assignment algorithm",
    )
    if (
        inventory["schema_version"] != INVENTORY_SCHEMA
        or assignment_algorithm["name"] != ASSIGNMENT_ALGORITHM
        or assignment_algorithm["component_edges"]
        != [
            "normalized-query-text-equality",
            "registered-near-duplicate-token-rule",
            "shared-positive-document-content",
            "shared-positive-relevance-document",
        ]
        or assignment_algorithm["cross_source_split_policy"] != "exclude-entire-component-v1"
        or assignment_algorithm["fit_calibration_component_ratio"] != "4:1"
        or assignment_algorithm["three_way_component_ratio"] != "3:1:1"
        or inventory["withhold_sealed_labels_from_online_process"] is not True
        or inventory_bytes != _canonical_bytes(inventory_value)
    ):
        raise DevelopmentStagingViewError("staged inventory protocol or canonical bytes differ")
    inventory_rows = _source_rows(inventory["artifacts"], label="inventory artifacts")
    inventory_by_path = {row.path: row for row in inventory_rows}

    projection_bytes = _read_relative_regular(
        projection_root_descriptor,
        PROJECTION_RECEIPT_FILENAME,
        maximum=_MAX_CONTROL_BYTES,
        label="online projection receipt",
        private=False,
        read_only=True,
    )
    if _sha256(projection_bytes) != projection_receipt_sha256:
        raise DevelopmentStagingViewError("online projection receipt differs from its caller pin")
    projection_value = _decode(projection_bytes, label="online projection receipt")
    projection = _closed(
        projection_value,
        _PROJECTION_FIELDS,
        label="online projection receipt",
    )
    projected = _source_rows(
        projection["projected_artifacts"],
        label="projected artifacts",
    )
    if (
        projection["schema_version"] != PROJECTION_SCHEMA
        or projection["projection_policy"] != PROJECTION_POLICY
        or projection["source_inventory_sha256"] != inventory_digest
        or projection["projected_artifact_count"] != len(projected)
        or projection["source_artifact_count"] != len(inventory_rows)
        or projection_bytes != _canonical_bytes(projection_value)
    ):
        raise DevelopmentStagingViewError("online projection receipt contract differs")
    if any(
        row.role not in _PROJECTED_ROLES
        or row.role in _OUTCOME_ROLES
        or bool(_path_tokens(row.path) & _OUTCOME_PATH_TOKENS)
        or row.visibility
        != (
            "protocol"
            if row.role
            in {
                "query-partition-structural-exclusions",
                "registered-cohort-exclusions",
            }
            else "online"
        )
        or inventory_by_path.get(row.path) != row
        for row in projected
    ):
        raise DevelopmentStagingViewError(
            "online projection admits an unknown or outcome-bearing artifact"
        )
    projected_set_sha256 = _sha256(_canonical_value_bytes([row.to_dict() for row in projected]))
    if projected_set_sha256 != projection["projected_artifact_set_sha256"]:
        raise DevelopmentStagingViewError("online projection artifact-set digest differs")
    projected_by_path = {row.path: row for row in projected}
    required_projection_roles = {
        "assignments",
        "corpus",
        "queries",
        "query-partition-structural-exclusions",
    }
    if not required_projection_roles.issubset({row.role for row in projected}):
        raise DevelopmentStagingViewError("online projection omits a required label-free role")
    projection_files = frozenset(
        {
            "inventory.json",
            "inventory.sha256",
            PROJECTION_RECEIPT_FILENAME,
            *(row.path for row in projected),
        }
    )
    _scan_exact_tree(
        projection_root_descriptor,
        expected_files=set(projection_files),
        label="source projection",
        private=False,
        read_only=True,
        enforce_view_boundary=False,
    )

    if _sha256(partition_audit_bytes) != partition_audit_file_sha256:
        raise DevelopmentStagingViewError("partition audit differs from its caller pin")
    audit_value = _decode(partition_audit_bytes, label="partition audit")
    audit = TypedPartitionAudit.from_dict(audit_value)
    if partition_audit_bytes != audit.canonical_file_bytes():
        raise DevelopmentStagingViewError("partition audit is not canonical")
    _validate_partition_bindings(
        audit=audit,
        inventory=inventory,
        inventory_rows=inventory_rows,
        inventory_sha256=inventory_digest,
    )
    audit_by_path = {row.path: row for row in audit.source_artifacts}
    query_counts = {(row.dataset, row.stage): row.query_count for row in audit.query_counts}

    selected: list[SourceArtifact] = []
    for path, (role, dataset, stage) in _expected_payload_contract().items():
        source = projected_by_path.get(path)
        if (
            source is None
            or (source.role, source.dataset, source.stage) != (role, dataset, stage)
            or source.visibility != "online"
            or audit_by_path.get(path) != source
        ):
            raise DevelopmentStagingViewError(
                f"registered development source {path!r} differs across controls"
            )
        if role == "queries" and query_counts.get((dataset, stage)) != source.record_count:
            raise DevelopmentStagingViewError(
                f"partition-audit query count differs for {stage}:{dataset}"
            )
        selected.append(source)
    selected_tuple = tuple(sorted(selected, key=lambda row: row.path.encode("utf-8")))
    assignment = [row for row in selected_tuple if row.role == "assignments"]
    if (
        len(assignment) != 1
        or assignment[0].visibility != "online"
        or assignment[0].sha256 != audit.assignment_artifact_sha256
    ):
        raise DevelopmentStagingViewError("partition audit assignment binding differs")
    return _AdmittedInputs(
        inventory_bytes=inventory_bytes,
        inventory_checksum_bytes=inventory_checksum_bytes,
        projected_artifact_set_sha256=projected_set_sha256,
        partition_component_membership_sha256=audit.component_membership_sha256,
        partition_source_artifact_set_sha256=audit.source_artifact_set_sha256,
        assignment_artifact_sha256=assignment[0].sha256,
        projection_files=projection_files,
        selected_sources=selected_tuple,
    )


def _copy_selected_artifact(
    *,
    source_root_descriptor: int,
    target_root_descriptor: int,
    source: SourceArtifact,
) -> ViewArtifact:
    source_descriptor, source_before = _open_relative_regular(
        source_root_descriptor,
        source.path,
        label=f"projection member {source.path!r}",
        private=False,
        read_only=True,
    )
    target_parent_descriptor, target_name = _open_or_create_private_parent(
        target_root_descriptor,
        source.path,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        target_descriptor = os.open(
            target_name,
            flags,
            0o600,
            dir_fd=target_parent_descriptor,
        )
        os.fchmod(target_descriptor, 0o600)
    except OSError as exc:
        os.close(source_descriptor)
        os.close(target_parent_descriptor)
        raise DevelopmentStagingViewError(f"cannot create view artifact {source.path!r}") from exc
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    last_byte: int | None = None
    try:
        while True:
            chunk = os.read(source_descriptor, _CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            record_count += chunk.count(b"\n")
            last_byte = chunk[-1]
            view = memoryview(chunk)
            while view:
                written = os.write(target_descriptor, view)
                if written <= 0:
                    raise DevelopmentStagingViewError(f"short write while copying {source.path!r}")
                view = view[written:]
        os.fsync(target_descriptor)
        _require_stable_file(
            source_before,
            os.fstat(source_descriptor),
            label=f"projection member {source.path!r}",
        )
        target_metadata = os.fstat(target_descriptor)
        _require_owned_regular(
            target_metadata,
            label=f"temporary view artifact {source.path!r}",
            private=True,
        )
        _require_exact_mode(
            target_metadata,
            expected=0o600,
            label=f"temporary view artifact {source.path!r}",
        )
        os.fsync(target_parent_descriptor)
    finally:
        os.close(source_descriptor)
        os.close(target_descriptor)
        os.close(target_parent_descriptor)
    if (
        digest.hexdigest() != source.sha256
        or byte_count != source.byte_count
        or record_count != source.record_count
        or last_byte != 10
    ):
        raise DevelopmentStagingViewError(
            f"projection member {source.path!r} differs from its shared pin"
        )
    return ViewArtifact(
        path=source.path,
        sha256=source.sha256,
        byte_count=source.byte_count,
        record_count=source.record_count,
        role=source.role,
        dataset=source.dataset,
        stage=source.stage,
    )


def _verify_selected_source_tree(
    root_descriptor: int,
    *,
    expected_files: frozenset[str],
    selected_sources: Sequence[SourceArtifact],
) -> None:
    """Rehash selected source names and close their identity through the final scan."""

    _scan_exact_tree(
        root_descriptor,
        expected_files=set(expected_files),
        label="source projection",
        private=False,
        read_only=True,
        enforce_view_boundary=False,
    )
    file_metadata: dict[str, os.stat_result] = {}
    for source in selected_sources:
        observed, metadata = _fingerprint_regular_file(
            root_descriptor,
            source.path,
            label=f"final source artifact {source.path!r}",
            private=False,
            read_only=True,
        )
        if observed != (
            source.sha256,
            source.byte_count,
            source.record_count,
            10,
        ):
            raise DevelopmentStagingViewError(
                f"final source artifact {source.path!r} differs from its shared pin"
            )
        file_metadata[source.path] = metadata
    _scan_exact_tree(
        root_descriptor,
        expected_files=set(expected_files),
        label="source projection",
        private=False,
        read_only=True,
        enforce_view_boundary=False,
        expected_file_metadata=file_metadata,
    )


def _control_artifact(path: str, encoded: bytes, *, role: str) -> ViewArtifact:
    return ViewArtifact(
        path=path,
        sha256=_sha256(encoded),
        byte_count=len(encoded),
        record_count=encoded.count(b"\n"),
        role=role,
        dataset=None,
        stage=None,
    )


def _fingerprint_regular_file(
    root_descriptor: int,
    relative: str,
    *,
    label: str,
    private: bool,
    read_only: bool,
) -> tuple[tuple[str, int, int, int | None], os.stat_result]:
    descriptor, before = _open_relative_regular(
        root_descriptor,
        relative,
        label=label,
        private=private,
        read_only=read_only,
    )
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    last_byte: int | None = None
    try:
        while True:
            chunk = os.read(descriptor, _CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            record_count += chunk.count(b"\n")
            last_byte = chunk[-1]
        metadata = os.fstat(descriptor)
        _require_stable_file(before, metadata, label=label)
    finally:
        os.close(descriptor)
    return (digest.hexdigest(), byte_count, record_count, last_byte), metadata


def _fingerprint_view_file(
    root_descriptor: int,
    relative: str,
) -> tuple[str, int, int, int | None]:
    fingerprint, _metadata = _fingerprint_regular_file(
        root_descriptor,
        relative,
        label=f"view artifact {relative!r}",
        private=True,
        read_only=True,
    )
    return fingerprint


def _load_receipt_with_metadata(
    root_descriptor: int,
    *,
    expected_sha256: str,
    read_only: bool,
) -> tuple[DevelopmentStagingViewReceipt, os.stat_result]:
    encoded, metadata = _read_relative_regular_with_metadata(
        root_descriptor,
        VIEW_RECEIPT_FILENAME,
        maximum=_MAX_CONTROL_BYTES,
        label="development staging view receipt",
        private=True,
        read_only=read_only,
    )
    if _sha256(encoded) != _require_sha256(
        "expected view receipt SHA-256",
        expected_sha256,
    ):
        raise DevelopmentStagingViewError("development staging view receipt digest differs")
    value = _decode(encoded, label="development staging view receipt")
    receipt = DevelopmentStagingViewReceipt.from_dict(value)
    if encoded != receipt.canonical_file_bytes():
        raise DevelopmentStagingViewError("development staging view receipt is not canonical")
    return receipt, metadata


def _load_receipt(
    root_descriptor: int,
    *,
    expected_sha256: str,
    read_only: bool,
) -> DevelopmentStagingViewReceipt:
    receipt, _metadata = _load_receipt_with_metadata(
        root_descriptor,
        expected_sha256=expected_sha256,
        read_only=read_only,
    )
    return receipt


def verify_development_staging_view(
    root: str | Path,
    *,
    expected_receipt_sha256: str,
) -> DevelopmentStagingViewReceipt:
    """Verify one published selection view without reopening any source payload."""

    view_root = _absolute_path(root, label="development staging view root")
    root_descriptor, root_before = _open_absolute_directory(
        view_root,
        label="development staging view root",
        private=True,
        read_only=True,
    )
    try:
        receipt, receipt_metadata = _load_receipt_with_metadata(
            root_descriptor,
            expected_sha256=expected_receipt_sha256,
            read_only=True,
        )
        if receipt.output_root != view_root:
            raise DevelopmentStagingViewError("development staging view moved from its bound path")
        _scan_exact_tree(
            root_descriptor,
            expected_files={artifact.path for artifact in receipt.artifacts}
            | {VIEW_RECEIPT_FILENAME},
            label="development staging view",
            private=True,
            read_only=True,
            enforce_view_boundary=True,
            exact_private_modes=True,
            expected_directory_mode=0o500,
            expected_file_mode=0o400,
        )
        by_path = {artifact.path: artifact for artifact in receipt.artifacts}
        file_metadata = {VIEW_RECEIPT_FILENAME: receipt_metadata}
        for path, artifact in by_path.items():
            observed, metadata = _fingerprint_regular_file(
                root_descriptor,
                path,
                label=f"view artifact {path!r}",
                private=True,
                read_only=True,
            )
            if observed != (
                artifact.sha256,
                artifact.byte_count,
                artifact.record_count,
                10,
            ):
                raise DevelopmentStagingViewError(
                    f"view artifact {path!r} differs from its receipt"
                )
            file_metadata[path] = metadata
        inventory = by_path["inventory.json"]
        checksum, checksum_metadata = _read_relative_regular_with_metadata(
            root_descriptor,
            "inventory.sha256",
            maximum=1024,
            label="view inventory checksum",
            private=True,
            read_only=True,
        )
        file_metadata["inventory.sha256"] = checksum_metadata
        if (
            inventory.sha256 != receipt.staged_inventory_sha256
            or checksum != f"{receipt.staged_inventory_sha256}  inventory.json\n".encode("ascii")
        ):
            raise DevelopmentStagingViewError("view inventory controls differ from the receipt")
        _scan_exact_tree(
            root_descriptor,
            expected_files={artifact.path for artifact in receipt.artifacts}
            | {VIEW_RECEIPT_FILENAME},
            label="development staging view",
            private=True,
            read_only=True,
            enforce_view_boundary=True,
            exact_private_modes=True,
            expected_file_metadata=file_metadata,
            expected_directory_mode=0o500,
            expected_file_mode=0o400,
        )
        if not _directory_path_matches_descriptor(
            view_root,
            root_descriptor,
            private=True,
        ):
            raise DevelopmentStagingViewError(
                "development staging view root path changed during verification"
            )
        _verify_temporary_tree(root_descriptor, receipt, sealed=True)
        _require_stable_directory(
            root_before,
            os.fstat(root_descriptor),
            label="development staging view root",
        )
        return receipt
    finally:
        os.close(root_descriptor)


def _verify_temporary_tree(
    root_descriptor: int,
    receipt: DevelopmentStagingViewReceipt,
    *,
    sealed: bool = False,
) -> None:
    directory_mode = 0o500 if sealed else 0o700
    file_mode = 0o400 if sealed else 0o600
    _scan_exact_tree(
        root_descriptor,
        expected_files={artifact.path for artifact in receipt.artifacts} | {VIEW_RECEIPT_FILENAME},
        label="temporary development staging view",
        private=True,
        read_only=sealed,
        enforce_view_boundary=True,
        exact_private_modes=True,
        expected_directory_mode=directory_mode,
        expected_file_mode=file_mode,
    )
    file_metadata: dict[str, os.stat_result] = {}
    for artifact in receipt.artifacts:
        observed, metadata = _fingerprint_regular_file(
            root_descriptor,
            artifact.path,
            label=f"temporary view artifact {artifact.path!r}",
            private=True,
            read_only=sealed,
        )
        if observed != (
            artifact.sha256,
            artifact.byte_count,
            artifact.record_count,
            10,
        ):
            raise DevelopmentStagingViewError(f"temporary view artifact {artifact.path!r} differs")
        file_metadata[artifact.path] = metadata
    encoded, receipt_metadata = _read_relative_regular_with_metadata(
        root_descriptor,
        VIEW_RECEIPT_FILENAME,
        maximum=_MAX_CONTROL_BYTES,
        label="temporary development staging view receipt",
        private=True,
        read_only=sealed,
    )
    file_metadata[VIEW_RECEIPT_FILENAME] = receipt_metadata
    if encoded != receipt.canonical_file_bytes():
        raise DevelopmentStagingViewError("temporary view receipt changed")
    _scan_exact_tree(
        root_descriptor,
        expected_files={artifact.path for artifact in receipt.artifacts} | {VIEW_RECEIPT_FILENAME},
        label="temporary development staging view",
        private=True,
        read_only=sealed,
        enforce_view_boundary=True,
        exact_private_modes=True,
        expected_file_metadata=file_metadata,
        expected_directory_mode=directory_mode,
        expected_file_mode=file_mode,
    )


def _rename_exclusive_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    result: int
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        function = library.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            _RENAME_EXCL,
        )
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            _RENAME_NOREPLACE,
        )
    else:
        raise DevelopmentStagingViewError(
            "exclusive directory publication requires renameatx_np or renameat2"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise DevelopmentStagingViewError("development staging output already exists")
        raise DevelopmentStagingViewError(
            f"cannot publish development staging output: {os.strerror(error)}"
        )


def _rename_sealed_exclusive_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    moved_descriptor: int,
) -> None:
    """Cross the macOS rename boundary, then restore the sealed root before returning."""

    before = os.fstat(moved_descriptor)
    _require_owned_directory_metadata(
        before,
        label="sealed development staging view root before rename",
        private=True,
        read_only=True,
    )
    _require_exact_mode(
        before,
        expected=0o500,
        label="sealed development staging view root before rename",
    )
    rename_error: BaseException | None = None
    try:
        if sys.platform == "darwin":
            os.fchmod(moved_descriptor, 0o700)
        _rename_exclusive_at(
            parent_descriptor,
            source_name,
            destination_name,
        )
    except BaseException as exc:
        rename_error = exc
    try:
        if sys.platform == "darwin":
            os.fchmod(moved_descriptor, 0o500)
        os.fsync(moved_descriptor)
        after = os.fstat(moved_descriptor)
        _require_owned_directory_metadata(
            after,
            label="sealed development staging view root after rename",
            private=True,
            read_only=True,
        )
        _require_exact_mode(
            after,
            expected=0o500,
            label="sealed development staging view root after rename",
        )
    except BaseException as seal_error:
        raise DevelopmentStagingViewError(
            "development staging root could not be resealed across rename"
        ) from seal_error
    if rename_error is not None:
        raise rename_error


def _entry_stat(
    parent_descriptor: int,
    name: str,
) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _same_optional_entry(
    left: os.stat_result | None,
    right: os.stat_result | None,
) -> bool:
    if left is None or right is None:
        return left is right
    return _metadata_equal(left, right, fields=_DIRECTORY_STABLE_FIELDS)


def _observe_publication_state(
    parent_descriptor: int,
    *,
    temporary_name: str,
    output_name: str,
) -> _PublicationNameState:
    """Return a name state only when two consecutive observations agree."""

    first = _PublicationNameState(
        temporary=_entry_stat(parent_descriptor, temporary_name),
        output=_entry_stat(parent_descriptor, output_name),
    )
    second = _PublicationNameState(
        temporary=_entry_stat(parent_descriptor, temporary_name),
        output=_entry_stat(parent_descriptor, output_name),
    )
    if not (
        _same_optional_entry(first.temporary, second.temporary)
        and _same_optional_entry(first.output, second.output)
    ):
        raise DevelopmentStagingPublicationIndeterminate(
            "publication names changed while their state was being observed"
        )
    return second


def _expected_at_temporary(
    state: _PublicationNameState,
    expected: os.stat_result,
) -> bool:
    return (
        state.temporary is not None
        and _same_inode(expected, state.temporary)
        and (state.output is None or not _same_inode(expected, state.output))
    )


def _expected_is_published(
    state: _PublicationNameState,
    expected: os.stat_result,
) -> bool:
    return (
        state.temporary is None and state.output is not None and _same_inode(expected, state.output)
    )


def _expected_is_rolled_back(
    state: _PublicationNameState,
    expected: os.stat_result,
) -> bool:
    return (
        state.temporary is not None
        and _same_inode(expected, state.temporary)
        and state.output is None
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


class _ExclusiveLeaseSet:
    """Retain nonblocking advisory exclusive leases through transaction completion."""

    def __init__(self) -> None:
        self._by_inode: dict[tuple[int, int], int] = {}
        self._owned_descriptors: list[int] = []

    def retain_existing(self, descriptor: int, *, label: str) -> int:
        return self._retain(descriptor, label=label, owned=False)

    def retain_owned(self, descriptor: int, *, label: str) -> int:
        return self._retain(descriptor, label=label, owned=True)

    def _retain(self, descriptor: int, *, label: str, owned: bool) -> int:
        metadata = os.fstat(descriptor)
        key = (metadata.st_dev, metadata.st_ino)
        retained = self._by_inode.get(key)
        if retained is not None:
            if owned:
                os.close(descriptor)
            return retained
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DevelopmentStagingViewError(
                f"{label} cannot acquire the required cooperative exclusive lease"
            ) from exc
        self._by_inode[key] = descriptor
        if owned:
            self._owned_descriptors.append(descriptor)
        return descriptor

    def close_owned(self) -> None:
        first_error: BaseException | None = None
        for descriptor in reversed(self._owned_descriptors):
            try:
                os.close(descriptor)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._owned_descriptors.clear()
        if first_error is not None:
            raise first_error


def _close_transaction_descriptors(
    *,
    temporary_descriptor: int | None,
    leases: _ExclusiveLeaseSet,
    source_descriptor: int,
    parent_descriptor: int | None,
) -> BaseException | None:
    """Close every retained descriptor and return the first close failure."""

    first_error: BaseException | None = None

    def close_one(descriptor: int) -> None:
        nonlocal first_error
        try:
            os.close(descriptor)
        except BaseException as exc:
            if first_error is None:
                first_error = exc

    try:
        leases.close_owned()
    except BaseException as exc:
        if first_error is None:
            first_error = exc
    if temporary_descriptor is not None:
        close_one(temporary_descriptor)
    close_one(source_descriptor)
    if parent_descriptor is not None:
        close_one(parent_descriptor)
    return first_error


def _require_pinned_parent(
    expected: os.stat_result,
    observed: os.stat_result,
) -> None:
    fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid")
    if not _metadata_equal(expected, observed, fields=fields):
        raise DevelopmentStagingViewError(
            "development staging output parent identity or mode changed"
        )


def _directory_path_matches_descriptor(
    path: Path,
    descriptor: int,
    *,
    private: bool,
) -> bool:
    candidate: int | None = None
    try:
        candidate, _ = _open_absolute_directory(
            path,
            label="development staging output parent path",
            private=private,
        )
        return _same_inode(os.fstat(candidate), os.fstat(descriptor))
    except DevelopmentStagingViewError:
        return False
    finally:
        if candidate is not None:
            os.close(candidate)


def _remove_tree_contents(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    _require_owned_directory_metadata(
        metadata,
        label="temporary tree during cleanup",
        private=True,
        read_only=False,
    )
    try:
        os.fchmod(descriptor, 0o700)
    except OSError as exc:
        raise DevelopmentStagingViewError(f"cannot make temporary tree removable: {exc}") from exc
    try:
        names = os.listdir(descriptor)
    except OSError as exc:
        raise DevelopmentStagingViewError(
            f"cannot enumerate temporary tree during cleanup: {exc}"
        ) from exc
    for name in names:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            child = os.open(name, _directory_open_flags(), dir_fd=descriptor)
            try:
                if not _same_inode(metadata, os.fstat(child)):
                    raise DevelopmentStagingViewError(
                        "temporary cleanup directory changed while opening"
                    )
                _remove_tree_contents(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)
    os.fsync(descriptor)


def _remove_temporary_tree(
    parent_descriptor: int,
    temporary_name: str,
    output_name: str,
    temporary_descriptor: int,
    *,
    require_output_absent: bool,
) -> None:
    expected = os.fstat(temporary_descriptor)
    before = _observe_publication_state(
        parent_descriptor,
        temporary_name=temporary_name,
        output_name=output_name,
    )
    if not _expected_at_temporary(before, expected):
        raise DevelopmentStagingViewError(
            "temporary tree path no longer names the pinned temporary directory"
        )
    if require_output_absent and before.output is not None:
        raise DevelopmentStagingViewError(
            "rolled-back output name was occupied before temporary cleanup"
        )
    _remove_tree_contents(temporary_descriptor)
    os.rmdir(temporary_name, dir_fd=parent_descriptor)
    _fsync_directory(parent_descriptor, label="development staging output parent")
    after = _observe_publication_state(
        parent_descriptor,
        temporary_name=temporary_name,
        output_name=output_name,
    )
    if after.temporary is not None or not _same_optional_entry(before.output, after.output):
        raise DevelopmentStagingViewError("temporary cleanup final name state could not be proved")


def _create_temporary_tree(
    parent_descriptor: int,
    output_name: str,
) -> tuple[str, int]:
    for _attempt in range(64):
        name = f".{output_name}.development-view-{secrets.token_hex(12)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        try:
            created = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            _require_owned_directory_metadata(
                created,
                label="temporary development staging view root",
                private=True,
                read_only=False,
            )
            os.chmod(
                name,
                0o700,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            normalized = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not _same_inode(created, normalized):
                raise DevelopmentStagingViewError(
                    "temporary development staging view root changed during mode normalization"
                )
            _require_owned_directory_metadata(
                normalized,
                label="temporary development staging view root",
                private=True,
                read_only=False,
            )
            _require_exact_mode(
                normalized,
                expected=0o700,
                label="temporary development staging view root",
            )
            descriptor = os.open(
                name,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
            metadata = os.fstat(descriptor)
            _require_stable_directory(
                normalized,
                metadata,
                label="temporary development staging view root",
            )
            _require_owned_directory_metadata(
                metadata,
                label="temporary development staging view root",
                private=True,
                read_only=False,
            )
            _require_exact_mode(
                metadata,
                expected=0o700,
                label="temporary development staging view root",
            )
            return name, descriptor
        except BaseException:
            try:
                os.rmdir(name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise
    raise DevelopmentStagingViewError(
        "cannot allocate a unique temporary development staging directory"
    )


def _fsync_temporary_directories(
    root_descriptor: int,
    receipt: DevelopmentStagingViewReceipt,
) -> None:
    directories = sorted(
        _expected_directories(
            {artifact.path for artifact in receipt.artifacts} | {VIEW_RECEIPT_FILENAME}
        ),
        key=lambda path: (-len(PurePosixPath(path).parts), path.encode("utf-8")),
    )
    for relative in directories:
        descriptor, _ = _open_relative_directory(
            root_descriptor,
            PurePosixPath(relative).parts,
            label=f"temporary directory {relative!r}",
            private=True,
            read_only=False,
        )
        try:
            _fsync_directory(descriptor, label=f"temporary directory {relative!r}")
        finally:
            os.close(descriptor)
    _fsync_directory(root_descriptor, label="temporary development staging view root")


def _retain_exact_tree_leases(
    leases: _ExclusiveLeaseSet,
    root_descriptor: int,
    *,
    expected_files: set[str],
    label: str,
    private: bool,
    read_only: bool,
) -> None:
    for relative in sorted(
        _expected_directories(expected_files),
        key=lambda path: (len(PurePosixPath(path).parts), path.encode("utf-8")),
    ):
        descriptor: int | None = None
        try:
            descriptor, _ = _open_relative_directory(
                root_descriptor,
                PurePosixPath(relative).parts,
                label=f"{label} directory {relative!r}",
                private=private,
                read_only=read_only,
            )
            leases.retain_owned(
                descriptor,
                label=f"{label} directory {relative!r}",
            )
            descriptor = None
        finally:
            if descriptor is not None:
                os.close(descriptor)
    for relative in sorted(expected_files, key=lambda path: path.encode("utf-8")):
        descriptor = None
        try:
            descriptor, _ = _open_relative_regular(
                root_descriptor,
                relative,
                label=f"{label} file {relative!r}",
                private=private,
                read_only=read_only,
            )
            leases.retain_owned(
                descriptor,
                label=f"{label} file {relative!r}",
            )
            descriptor = None
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _seal_temporary_tree(
    root_descriptor: int,
    receipt: DevelopmentStagingViewReceipt,
) -> None:
    """Make the completed tree owner-read-only before it can be published."""

    expected_files = {artifact.path for artifact in receipt.artifacts} | {VIEW_RECEIPT_FILENAME}
    for relative in sorted(expected_files, key=lambda path: path.encode("utf-8")):
        descriptor, _ = _open_relative_regular(
            root_descriptor,
            relative,
            label=f"temporary view artifact {relative!r} during sealing",
            private=True,
            read_only=False,
        )
        try:
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            _require_owned_regular(
                metadata,
                label=f"sealed temporary view artifact {relative!r}",
                private=True,
                read_only=True,
            )
            _require_exact_mode(
                metadata,
                expected=0o400,
                label=f"sealed temporary view artifact {relative!r}",
            )
        except OSError as exc:
            raise DevelopmentStagingViewError(
                f"cannot seal temporary view artifact {relative!r}: {exc}"
            ) from exc
        finally:
            os.close(descriptor)

    directories = sorted(
        _expected_directories(expected_files),
        key=lambda path: (-len(PurePosixPath(path).parts), path.encode("utf-8")),
    )
    for relative in directories:
        descriptor, _ = _open_relative_directory(
            root_descriptor,
            PurePosixPath(relative).parts,
            label=f"temporary directory {relative!r} during sealing",
            private=True,
            read_only=False,
        )
        try:
            os.fchmod(descriptor, 0o500)
            _fsync_directory(
                descriptor,
                label=f"sealed temporary directory {relative!r}",
            )
            metadata = os.fstat(descriptor)
            _require_owned_directory_metadata(
                metadata,
                label=f"sealed temporary directory {relative!r}",
                private=True,
                read_only=True,
            )
            _require_exact_mode(
                metadata,
                expected=0o500,
                label=f"sealed temporary directory {relative!r}",
            )
        except OSError as exc:
            raise DevelopmentStagingViewError(
                f"cannot seal temporary directory {relative!r}: {exc}"
            ) from exc
        finally:
            os.close(descriptor)

    try:
        os.fchmod(root_descriptor, 0o500)
        _fsync_directory(
            root_descriptor,
            label="sealed temporary development staging view root",
        )
    except OSError as exc:
        raise DevelopmentStagingViewError(
            f"cannot seal temporary development staging view root: {exc}"
        ) from exc
    root_metadata = os.fstat(root_descriptor)
    _require_owned_directory_metadata(
        root_metadata,
        label="sealed temporary development staging view root",
        private=True,
        read_only=True,
    )
    _require_exact_mode(
        root_metadata,
        expected=0o500,
        label="sealed temporary development staging view root",
    )
    _verify_temporary_tree(root_descriptor, receipt, sealed=True)


def _require_parent_binding(
    *,
    parent_descriptor: int,
    parent_before: os.stat_result,
    output_parent: Path,
) -> None:
    _require_pinned_parent(parent_before, os.fstat(parent_descriptor))
    if not _directory_path_matches_descriptor(
        output_parent,
        parent_descriptor,
        private=True,
    ):
        raise DevelopmentStagingViewError(
            "output parent path changed after descriptor-relative publication"
        )


def _verify_published_tree_by_name(
    *,
    parent_descriptor: int,
    output_name: str,
    expected: os.stat_result,
    receipt: DevelopmentStagingViewReceipt,
) -> None:
    before = _entry_stat(parent_descriptor, output_name)
    if before is None or not _same_inode(expected, before):
        raise DevelopmentStagingPublicationIndeterminate(
            "published output name stopped identifying the pinned tree"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            output_name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        _require_stable_directory(
            before,
            opened,
            label="published development staging view root",
        )
        _verify_temporary_tree(descriptor, receipt, sealed=True)
        _require_stable_directory(
            opened,
            os.fstat(descriptor),
            label="published development staging view root",
        )
    except DevelopmentStagingPublicationIndeterminate:
        raise
    except OSError as exc:
        raise DevelopmentStagingPublicationIndeterminate(
            f"cannot open the pinned published development staging tree: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _require_source_binding(
    *,
    source_root: Path,
    source_descriptor: int,
    source_before: os.stat_result,
    expected_files: frozenset[str],
    selected_sources: Sequence[SourceArtifact],
) -> None:
    _require_stable_directory(
        source_before,
        os.fstat(source_descriptor),
        label="source projection root",
    )
    if not _directory_path_matches_descriptor(
        source_root,
        source_descriptor,
        private=False,
    ):
        raise DevelopmentStagingViewError("source projection root path changed during publication")
    _verify_selected_source_tree(
        source_descriptor,
        expected_files=expected_files,
        selected_sources=selected_sources,
    )
    _require_stable_directory(
        source_before,
        os.fstat(source_descriptor),
        label="source projection root",
    )


def _require_published_name_state(
    *,
    parent_descriptor: int,
    temporary_name: str,
    output_name: str,
    expected: os.stat_result,
) -> None:
    state = _observe_publication_state(
        parent_descriptor,
        temporary_name=temporary_name,
        output_name=output_name,
    )
    if not _expected_is_published(state, expected):
        raise DevelopmentStagingPublicationIndeterminate(
            "publication names do not prove that only the output names the pinned tree"
        )


def _prove_publication_success(
    *,
    parent_descriptor: int,
    parent_before: os.stat_result,
    temporary_name: str,
    output_name: str,
    expected: os.stat_result,
    output_root: Path,
    temporary_descriptor: int,
    receipt: DevelopmentStagingViewReceipt,
    source_root: Path,
    source_descriptor: int,
    source_before: os.stat_result,
    source_files: frozenset[str],
    selected_sources: Sequence[SourceArtifact],
) -> None:
    _fsync_directory(parent_descriptor, label="development staging output parent")
    for _proof_pass in range(2):
        _require_parent_binding(
            parent_descriptor=parent_descriptor,
            parent_before=parent_before,
            output_parent=output_root.parent,
        )
        _require_published_name_state(
            parent_descriptor=parent_descriptor,
            temporary_name=temporary_name,
            output_name=output_name,
            expected=expected,
        )
        _verify_published_tree_by_name(
            parent_descriptor=parent_descriptor,
            output_name=output_name,
            expected=expected,
            receipt=receipt,
        )
        _verify_temporary_tree(temporary_descriptor, receipt, sealed=True)
        _require_source_binding(
            source_root=source_root,
            source_descriptor=source_descriptor,
            source_before=source_before,
            expected_files=source_files,
            selected_sources=selected_sources,
        )
        _require_published_name_state(
            parent_descriptor=parent_descriptor,
            temporary_name=temporary_name,
            output_name=output_name,
            expected=expected,
        )


def _rollback_publication(
    *,
    parent_descriptor: int,
    parent_before: os.stat_result,
    temporary_name: str,
    output_name: str,
    expected: os.stat_result,
    output_root: Path,
    temporary_descriptor: int,
    publication_error: BaseException,
) -> None:
    try:
        state = _observe_publication_state(
            parent_descriptor,
            temporary_name=temporary_name,
            output_name=output_name,
        )
        if _expected_is_published(state, expected):
            try:
                _rename_sealed_exclusive_at(
                    parent_descriptor,
                    output_name,
                    temporary_name,
                    temporary_descriptor,
                )
            except BaseException as rename_error:
                state = _observe_publication_state(
                    parent_descriptor,
                    temporary_name=temporary_name,
                    output_name=output_name,
                )
                if not _expected_is_rolled_back(state, expected):
                    raise DevelopmentStagingPublicationIndeterminate(
                        "rollback rename failed without a proved rolled-back state"
                    ) from rename_error
        elif not _expected_is_rolled_back(state, expected):
            raise DevelopmentStagingPublicationIndeterminate(
                "publication failure left an unclassified output/temporary name state"
            )

        _fsync_directory(
            parent_descriptor,
            label="development staging output parent after rollback",
        )
        for _proof_pass in range(2):
            _require_parent_binding(
                parent_descriptor=parent_descriptor,
                parent_before=parent_before,
                output_parent=output_root.parent,
            )
            state = _observe_publication_state(
                parent_descriptor,
                temporary_name=temporary_name,
                output_name=output_name,
            )
            if not _expected_is_rolled_back(state, expected):
                raise DevelopmentStagingPublicationIndeterminate(
                    "publication rollback could not prove output absence and temporary restoration"
                )
    except DevelopmentStagingPublicationIndeterminate as rollback_error:
        raise DevelopmentStagingPublicationIndeterminate(
            f"publication durability for {output_root} is indeterminate; "
            f"temporary entry={temporary_name!r}"
        ) from rollback_error
    except BaseException as rollback_error:
        raise DevelopmentStagingPublicationIndeterminate(
            f"publication durability for {output_root} is indeterminate; "
            f"temporary entry={temporary_name!r}"
        ) from rollback_error
    raise _DevelopmentStagingPublicationRolledBack(
        "publication failed after rename; rollback was completed and proved"
    ) from publication_error


def _publish_exclusive(
    *,
    parent_descriptor: int,
    parent_before: os.stat_result,
    temporary_name: str,
    output_name: str,
    temporary_descriptor: int,
    output_root: Path,
    receipt: DevelopmentStagingViewReceipt,
    source_root: Path,
    source_descriptor: int,
    source_before: os.stat_result,
    source_files: frozenset[str],
    selected_sources: Sequence[SourceArtifact],
) -> None:
    expected = os.fstat(temporary_descriptor)
    try:
        _rename_sealed_exclusive_at(
            parent_descriptor,
            temporary_name,
            output_name,
            temporary_descriptor,
        )
    except BaseException as rename_error:
        try:
            state = _observe_publication_state(
                parent_descriptor,
                temporary_name=temporary_name,
                output_name=output_name,
            )
        except BaseException as observation_error:
            raise DevelopmentStagingPublicationIndeterminate(
                f"publication durability for {output_root} is indeterminate; "
                f"temporary entry={temporary_name!r}"
            ) from observation_error
        if _expected_at_temporary(state, expected):
            raise DevelopmentStagingViewError(
                "exclusive publication failed before the pinned tree became public"
            ) from rename_error
        if _expected_is_published(state, expected):
            _rollback_publication(
                parent_descriptor=parent_descriptor,
                parent_before=parent_before,
                temporary_name=temporary_name,
                output_name=output_name,
                expected=expected,
                output_root=output_root,
                temporary_descriptor=temporary_descriptor,
                publication_error=rename_error,
            )
        raise DevelopmentStagingPublicationIndeterminate(
            f"publication durability for {output_root} is indeterminate; "
            f"temporary entry={temporary_name!r}"
        ) from rename_error

    try:
        state = _observe_publication_state(
            parent_descriptor,
            temporary_name=temporary_name,
            output_name=output_name,
        )
        if _expected_is_rolled_back(state, expected):
            _rollback_publication(
                parent_descriptor=parent_descriptor,
                parent_before=parent_before,
                temporary_name=temporary_name,
                output_name=output_name,
                expected=expected,
                output_root=output_root,
                temporary_descriptor=temporary_descriptor,
                publication_error=DevelopmentStagingViewError(
                    "published tree was moved back before its success proof"
                ),
            )
        if not _expected_is_published(state, expected):
            raise DevelopmentStagingPublicationIndeterminate(
                "successful rename returned an unclassified publication name state"
            )
        _prove_publication_success(
            parent_descriptor=parent_descriptor,
            parent_before=parent_before,
            temporary_name=temporary_name,
            output_name=output_name,
            expected=expected,
            output_root=output_root,
            temporary_descriptor=temporary_descriptor,
            receipt=receipt,
            source_root=source_root,
            source_descriptor=source_descriptor,
            source_before=source_before,
            source_files=source_files,
            selected_sources=selected_sources,
        )
    except DevelopmentStagingPublicationIndeterminate:
        raise
    except BaseException as publication_error:
        _rollback_publication(
            parent_descriptor=parent_descriptor,
            parent_before=parent_before,
            temporary_name=temporary_name,
            output_name=output_name,
            expected=expected,
            output_root=output_root,
            temporary_descriptor=temporary_descriptor,
            publication_error=publication_error,
        )


def build_development_staging_view(
    *,
    projection_root: str | Path,
    staged_inventory_sha256: str,
    projection_receipt_sha256: str,
    partition_audit_path: str | Path,
    partition_audit_file_sha256: str,
    output_root: str | Path,
) -> DevelopmentStagingViewReceipt:
    """Build and atomically publish one fit/calibration selection view."""

    with _TransactionSignalGuard():
        return _build_development_staging_view(
            projection_root=projection_root,
            staged_inventory_sha256=staged_inventory_sha256,
            projection_receipt_sha256=projection_receipt_sha256,
            partition_audit_path=partition_audit_path,
            partition_audit_file_sha256=partition_audit_file_sha256,
            output_root=output_root,
        )


def _build_development_staging_view(
    *,
    projection_root: str | Path,
    staged_inventory_sha256: str,
    projection_receipt_sha256: str,
    partition_audit_path: str | Path,
    partition_audit_file_sha256: str,
    output_root: str | Path,
) -> DevelopmentStagingViewReceipt:
    _require_nonroot()
    source = _absolute_path(projection_root, label="source projection root")
    audit_path = _absolute_path(partition_audit_path, label="partition audit path")
    output = _absolute_path(output_root, label="development staging output root")
    inventory_pin = _require_sha256("staged inventory SHA-256", staged_inventory_sha256)
    projection_pin = _require_sha256("projection receipt SHA-256", projection_receipt_sha256)
    audit_pin = _require_sha256("partition audit file SHA-256", partition_audit_file_sha256)
    if _path_tokens(str(output)) & _FORBIDDEN_OUTPUT_TOKENS:
        raise DevelopmentStagingViewError("development staging output path is forbidden")
    if _paths_overlap(output, source) or _paths_overlap(output, audit_path):
        raise DevelopmentStagingViewError("development staging output overlaps an input")
    source_descriptor, source_before = _open_absolute_directory(
        source,
        label="source projection root",
        private=False,
        read_only=True,
    )
    leases = _ExclusiveLeaseSet()
    parent_descriptor: int | None = None
    temporary_descriptor: int | None = None
    temporary_name: str | None = None
    publication_proved = False
    receipt: DevelopmentStagingViewReceipt | None = None
    try:
        leases.retain_existing(
            source_descriptor,
            label="source projection root",
        )
        parent_descriptor, parent_before = _open_absolute_directory(
            output.parent,
            label="development staging output parent",
            private=True,
        )
        leases.retain_existing(
            parent_descriptor,
            label="development staging output parent",
        )
        if _entry_stat(parent_descriptor, output.name) is not None:
            raise DevelopmentStagingViewError("development staging output already exists")
        partition_audit_bytes = _read_locked_absolute_regular(
            audit_path,
            maximum=_MAX_CONTROL_BYTES,
            label="partition audit",
            private_parent=True,
            read_only=True,
            leases=leases,
        )
        discovered = _admit_inputs(
            projection_root_descriptor=source_descriptor,
            staged_inventory_sha256=inventory_pin,
            projection_receipt_sha256=projection_pin,
            partition_audit_bytes=partition_audit_bytes,
            partition_audit_file_sha256=audit_pin,
        )
        _retain_exact_tree_leases(
            leases,
            source_descriptor,
            expected_files=set(discovered.projection_files),
            label="source projection",
            private=False,
            read_only=True,
        )
        admitted = _admit_inputs(
            projection_root_descriptor=source_descriptor,
            staged_inventory_sha256=inventory_pin,
            projection_receipt_sha256=projection_pin,
            partition_audit_bytes=partition_audit_bytes,
            partition_audit_file_sha256=audit_pin,
        )
        if admitted != discovered:
            raise DevelopmentStagingViewError(
                "source projection changed while its cooperative custody leases were acquired"
            )
        _require_stable_directory(
            source_before,
            os.fstat(source_descriptor),
            label="source projection root",
        )
        _require_pinned_parent(parent_before, os.fstat(parent_descriptor))

        temporary_name, temporary_descriptor = _create_temporary_tree(
            parent_descriptor,
            output.name,
        )
        try:
            leases.retain_existing(
                temporary_descriptor,
                label="temporary development staging view root",
            )
            _fsync_directory(
                parent_descriptor,
                label="development staging output parent after temporary creation",
            )
            _write_exclusive_at(
                temporary_descriptor,
                "inventory.json",
                admitted.inventory_bytes,
            )
            _write_exclusive_at(
                temporary_descriptor,
                "inventory.sha256",
                admitted.inventory_checksum_bytes,
            )
            artifacts: list[ViewArtifact] = [
                _control_artifact(
                    "inventory.json",
                    admitted.inventory_bytes,
                    role="staged-inventory",
                ),
                _control_artifact(
                    "inventory.sha256",
                    admitted.inventory_checksum_bytes,
                    role="staged-inventory-checksum",
                ),
            ]
            artifacts.extend(
                _copy_selected_artifact(
                    source_root_descriptor=source_descriptor,
                    target_root_descriptor=temporary_descriptor,
                    source=source_artifact,
                )
                for source_artifact in admitted.selected_sources
            )
            artifact_tuple = tuple(sorted(artifacts, key=lambda row: row.path.encode("utf-8")))
            receipt = DevelopmentStagingViewReceipt(
                source_projection_root=source,
                output_root=output,
                partition_audit_path=audit_path,
                staged_inventory_sha256=inventory_pin,
                projection_receipt_sha256=projection_pin,
                projected_artifact_set_sha256=(admitted.projected_artifact_set_sha256),
                partition_audit_file_sha256=audit_pin,
                partition_component_membership_sha256=(
                    admitted.partition_component_membership_sha256
                ),
                partition_source_artifact_set_sha256=(
                    admitted.partition_source_artifact_set_sha256
                ),
                assignment_artifact_sha256=admitted.assignment_artifact_sha256,
                input_custody=DevelopmentStagingInputCustody(
                    capture_set_sha256=_input_capture_set_sha256(
                        artifacts=artifact_tuple,
                        projection_receipt_sha256=projection_pin,
                        partition_audit_file_sha256=audit_pin,
                    )
                ),
                artifacts=artifact_tuple,
                view_artifact_set_sha256=_artifact_set_sha256(artifact_tuple),
            )
            _write_exclusive_at(
                temporary_descriptor,
                VIEW_RECEIPT_FILENAME,
                receipt.canonical_file_bytes(),
            )
            _verify_temporary_tree(temporary_descriptor, receipt)
            _fsync_temporary_directories(temporary_descriptor, receipt)
            _retain_exact_tree_leases(
                leases,
                temporary_descriptor,
                expected_files={
                    *(artifact.path for artifact in receipt.artifacts),
                    VIEW_RECEIPT_FILENAME,
                },
                label="temporary development staging view",
                private=True,
                read_only=False,
            )
            _seal_temporary_tree(temporary_descriptor, receipt)
            _require_stable_directory(
                source_before,
                os.fstat(source_descriptor),
                label="source projection root",
            )
            if not _directory_path_matches_descriptor(
                source,
                source_descriptor,
                private=False,
            ):
                raise DevelopmentStagingViewError(
                    "source projection root path changed before publication"
                )
            _require_pinned_parent(parent_before, os.fstat(parent_descriptor))
            if not _directory_path_matches_descriptor(
                output.parent,
                parent_descriptor,
                private=True,
            ):
                raise DevelopmentStagingViewError(
                    "development staging output parent path changed before publication"
                )
            _verify_selected_source_tree(
                source_descriptor,
                expected_files=admitted.projection_files,
                selected_sources=admitted.selected_sources,
            )
            _publish_exclusive(
                parent_descriptor=parent_descriptor,
                parent_before=parent_before,
                temporary_name=temporary_name,
                output_name=output.name,
                temporary_descriptor=temporary_descriptor,
                output_root=output,
                receipt=receipt,
                source_root=source,
                source_descriptor=source_descriptor,
                source_before=source_before,
                source_files=admitted.projection_files,
                selected_sources=admitted.selected_sources,
            )
            publication_proved = True
        except _DevelopmentStagingPublicationRolledBack as rollback:
            try:
                _remove_temporary_tree(
                    parent_descriptor,
                    temporary_name,
                    output.name,
                    temporary_descriptor,
                    require_output_absent=True,
                )
            except DevelopmentStagingPublicationIndeterminate:
                raise
            except BaseException as cleanup_error:
                raise DevelopmentStagingPublicationIndeterminate(
                    f"publication rollback cleanup for {output} is indeterminate; "
                    f"temporary entry={temporary_name!r}"
                ) from cleanup_error
            publication_error = rollback.__cause__
            if isinstance(
                publication_error,
                (
                    DevelopmentStagingInterruptedError,
                    KeyboardInterrupt,
                    SystemExit,
                ),
            ):
                raise publication_error
            raise
        except DevelopmentStagingPublicationIndeterminate:
            raise
        except BaseException as build_error:
            if publication_proved:
                raise DevelopmentStagingPublicationIndeterminate(
                    f"publication for {output} was proved before transaction completion "
                    f"failed; temporary entry={temporary_name!r}; "
                    f"receipt_sha256={receipt.artifact_sha256 if receipt else 'unknown'}"
                ) from build_error
            try:
                _remove_temporary_tree(
                    parent_descriptor,
                    temporary_name,
                    output.name,
                    temporary_descriptor,
                    require_output_absent=False,
                )
            except DevelopmentStagingPublicationIndeterminate:
                raise
            except BaseException as cleanup_error:
                raise DevelopmentStagingPublicationIndeterminate(
                    f"development staging cleanup for {output} is indeterminate; "
                    f"temporary entry={temporary_name!r}"
                ) from cleanup_error
            raise
    finally:
        active_error = sys.exc_info()[1]
        close_error = _close_transaction_descriptors(
            temporary_descriptor=temporary_descriptor,
            leases=leases,
            source_descriptor=source_descriptor,
            parent_descriptor=parent_descriptor,
        )
        if close_error is not None:
            if publication_proved:
                raise DevelopmentStagingPublicationIndeterminate(
                    f"publication for {output} was proved, but descriptor closure is "
                    f"indeterminate; public output={output}; "
                    f"temporary entry={temporary_name!r}; "
                    f"receipt_sha256={receipt.artifact_sha256 if receipt else 'unknown'}"
                ) from close_error
            if active_error is None:
                raise close_error
    if receipt is None:
        raise DevelopmentStagingViewError("development staging publication produced no receipt")
    return receipt


def _result(receipt: DevelopmentStagingViewReceipt) -> dict[str, object]:
    return {
        "artifact_count": len(receipt.artifacts),
        "output_root": str(receipt.output_root),
        "receipt_path": str(receipt.output_root / VIEW_RECEIPT_FILENAME),
        "receipt_sha256": receipt.artifact_sha256,
        "schema_version": CLI_RESULT_SCHEMA,
        "staged_inventory_sha256": receipt.staged_inventory_sha256,
        "view_artifact_set_sha256": receipt.view_artifact_set_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the label-payload-excluded development selection view.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build",
        help="build one new view",
        allow_abbrev=False,
    )
    build.add_argument("--projection-root", required=True)
    build.add_argument("--staged-inventory-sha256", required=True)
    build.add_argument("--projection-receipt-sha256", required=True)
    build.add_argument("--partition-audit", required=True)
    build.add_argument("--partition-audit-file-sha256", required=True)
    build.add_argument("--output-root", required=True)
    verify = commands.add_parser(
        "verify",
        help="verify one existing view without source access",
        allow_abbrev=False,
    )
    verify.add_argument("--root", required=True)
    verify.add_argument("--receipt-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "build":
            receipt = build_development_staging_view(
                projection_root=arguments.projection_root,
                staged_inventory_sha256=arguments.staged_inventory_sha256,
                projection_receipt_sha256=arguments.projection_receipt_sha256,
                partition_audit_path=arguments.partition_audit,
                partition_audit_file_sha256=arguments.partition_audit_file_sha256,
                output_root=arguments.output_root,
            )
        else:
            receipt = verify_development_staging_view(
                arguments.root,
                expected_receipt_sha256=arguments.receipt_sha256,
            )
    except DevelopmentStagingInterruptedError as exc:
        return 128 + exc.signum
    except DevelopmentStagingViewError as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(_canonical_bytes(_result(receipt)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
