#!/usr/bin/env python3
"""Publish the fit/calibration-only view used by exact-P development.

This host operator is deliberately outside the confirmatory wheel.  It proves
the label-free selection and the future-beacon design seed before it opens a
development qrel or evidence bundle, then publishes the smallest label-bearing
view accepted by the frozen post-embedding runtime.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import os
import secrets
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from fractal_ann_diagnostics.development_cohort import (
    CALIBRATION_FAMILY_COUNT,
    FIT_FAMILY_COUNT,
    load_development_cohort_selection,
    select_development_cohort,
)
from fractal_ann_diagnostics.post_embedding_development import (
    OPERATOR_CONFIG_FILENAME,
    SELECTION_FILENAME,
    load_post_embedding_development_config,
)
from fractal_ann_diagnostics.scalable_partition_audit import (
    load_scalable_partition_audit,
)
from operators.development_staging_view import (
    DEVELOPMENT_SOURCE_STAGES,
    FIXED_CORPORA,
    INVENTORY_SCHEMA,
    DevelopmentStagingViewError,
    SourceArtifact,
    _absolute_path,
    _canonical_bytes,
    _canonical_value_bytes,
    _closed,
    _create_temporary_tree,
    _decode,
    _directory_open_flags,
    _entry_stat,
    _ExclusiveLeaseSet,
    _expected_directories,
    _fingerprint_regular_file,
    _open_absolute_directory,
    _open_relative_directory,
    _open_relative_regular,
    _read_open_regular,
    _read_relative_regular,
    _relative_path,
    _remove_temporary_tree,
    _rename_exclusive_at,
    _rename_sealed_exclusive_at,
    _require_exact_mode,
    _require_nonroot,
    _require_stable_directory,
    _require_stable_file,
    _scan_exact_tree,
    _source_rows,
    _write_exclusive_at,
    verify_development_staging_view,
)
from operators.development_staging_view import (
    VIEW_RECEIPT_FILENAME as PHASE1_RECEIPT_FILENAME,
)

PHASE2_RECEIPT_SCHEMA = "fractal-development-phase-two-view-receipt-v2"
PHASE2_ARTIFACT_SCHEMA = "fractal-development-phase-two-view-artifact-v1"
PHASE2_CUSTODY_SCHEMA = "fractal-development-phase-two-input-custody-v1"
PHASE2_CLI_SCHEMA = "fractal-development-phase-two-cli-result-v1"
BOOTSTRAP_RECEIPT_SCHEMA = "fractal-post-embedding-resume-bootstrap-v1"
PHASE2_RECEIPT_FILENAME = "phase-two-view-receipt.json"
PHASE2_VIEW_DIRECTORY = "view"
INPUT_CUSTODY_CONTRACT = "fractal-exclusive-posix-advisory-custody-v1"
EXPECTED_SELECTION_COUNT = len(FIXED_CORPORA) * (FIT_FAMILY_COUNT + CALIBRATION_FAMILY_COUNT)
EVIDENCE_CORPORA = frozenset({"scifact", "hotpotqa-fullwiki", "t2-ragbench"})

_MAX_CONTROL_BYTES = 256 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_SHA256_CHARS = frozenset("0123456789abcdef")
_DARWIN_ACL_TYPE_EXTENDED = 0x00000100
_DARWIN_ACL_FIRST_ENTRY = 0
_LINUX_POSIX_ACL_XATTRS = frozenset({b"system.posix_acl_access", b"system.posix_acl_default"})
_FORBIDDEN_MOUNT_TOKENS = frozenset(
    {
        "custody",
        "heldout",
        "holdout",
        "label",
        "labels",
        "outcome",
        "outcomes",
        "reserve",
        "reserved",
        "result",
        "results",
        "sealed",
    }
)
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
_POST_CONFIG_FIELDS = frozenset(
    {
        "design_seed_sha256",
        "full_staged_inventory_sha256",
        "full_staged_root",
        "output_root",
        "partition_audit_file_sha256",
        "partition_audit_path",
        "production_embedding_config_path",
        "production_embedding_config_sha256",
        "schema_version",
    }
)


class DevelopmentPhase2ViewError(RuntimeError):
    """A phase-two admission, publication, or verification failed."""


class DevelopmentPhase2PublicationIndeterminate(DevelopmentPhase2ViewError):
    """Publication may have crossed its irreversible name boundary."""


def _require_no_extended_acl(descriptor: int, *, label: str) -> None:
    """Reject descriptor-bound ACLs that would invalidate mode-bit custody."""

    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        required = ("acl_get_fd_np", "acl_get_entry", "acl_free")
        if not all(hasattr(library, name) for name in required):
            raise DevelopmentPhase2ViewError(
                f"{label} ACL inspection is unavailable on this macOS host"
            )
        get_acl = library.acl_get_fd_np
        get_acl.argtypes = [ctypes.c_int, ctypes.c_int]
        get_acl.restype = ctypes.c_void_p
        ctypes.set_errno(0)
        acl = get_acl(descriptor, _DARWIN_ACL_TYPE_EXTENDED)
        if not acl:
            error = ctypes.get_errno()
            if error in {0, errno.ENOENT}:
                return
            raise DevelopmentPhase2ViewError(
                f"cannot inspect {label} extended ACL: {os.strerror(error)}"
            )
        get_entry = library.acl_get_entry
        get_entry.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        get_entry.restype = ctypes.c_int
        free_acl = library.acl_free
        free_acl.argtypes = [ctypes.c_void_p]
        free_acl.restype = ctypes.c_int
        entry = ctypes.c_void_p()
        try:
            ctypes.set_errno(0)
            result = get_entry(
                ctypes.c_void_p(acl),
                _DARWIN_ACL_FIRST_ENTRY,
                ctypes.byref(entry),
            )
            error = ctypes.get_errno()
            if result == 0:
                raise DevelopmentPhase2ViewError(f"{label} must not carry an extended ACL")
            if error not in {0, errno.ENOENT}:
                raise DevelopmentPhase2ViewError(
                    f"cannot inspect {label} extended ACL entries: {os.strerror(error)}"
                )
            return
        finally:
            free_acl(ctypes.c_void_p(acl))

    if sys.platform.startswith("linux"):
        if not hasattr(library, "flistxattr"):
            raise DevelopmentPhase2ViewError(
                f"{label} POSIX ACL inspection is unavailable on this Linux host"
            )
        list_xattrs = library.flistxattr
        list_xattrs.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        list_xattrs.restype = ctypes.c_ssize_t
        ctypes.set_errno(0)
        size = list_xattrs(descriptor, None, 0)
        if size < 0:
            error = ctypes.get_errno()
            if error in {errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)}:
                return
            raise DevelopmentPhase2ViewError(
                f"cannot inspect {label} extended attributes: {os.strerror(error)}"
            )
        if size == 0:
            return
        names_buffer = ctypes.create_string_buffer(size)
        ctypes.set_errno(0)
        observed_size = list_xattrs(descriptor, names_buffer, size)
        if observed_size < 0:
            error = ctypes.get_errno()
            raise DevelopmentPhase2ViewError(
                f"cannot read {label} extended attributes: {os.strerror(error)}"
            )
        names = frozenset(names_buffer.raw[:observed_size].split(b"\x00"))
        if names & _LINUX_POSIX_ACL_XATTRS:
            raise DevelopmentPhase2ViewError(f"{label} must not carry a POSIX ACL")
        return

    raise DevelopmentPhase2ViewError(
        f"{label} ACL inspection is unsupported on platform {sys.platform!r}"
    )


def _open_absolute_directory_acl(
    path: Path,
    *,
    label: str,
    private: bool | None,
    read_only: bool = False,
) -> tuple[int, os.stat_result]:
    descriptor, metadata = _open_absolute_directory(
        path,
        label=label,
        private=private,
        read_only=read_only,
    )
    try:
        _require_no_extended_acl(descriptor, label=label)
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_directory_acl(
    root_descriptor: int,
    parts: Sequence[str],
    *,
    label: str,
    private: bool,
    read_only: bool,
) -> tuple[int, os.stat_result]:
    descriptor, metadata = _open_relative_directory(
        root_descriptor,
        parts,
        label=label,
        private=private,
        read_only=read_only,
    )
    try:
        _require_no_extended_acl(descriptor, label=label)
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_regular_acl(
    root_descriptor: int,
    relative: str,
    *,
    label: str,
    private: bool,
    read_only: bool,
) -> tuple[int, os.stat_result]:
    descriptor, metadata = _open_relative_regular(
        root_descriptor,
        relative,
        label=label,
        private=private,
        read_only=read_only,
    )
    try:
        _require_no_extended_acl(descriptor, label=label)
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _require_tree_acl_free(
    root_descriptor: int,
    *,
    expected_files: set[str],
    label: str,
    private: bool,
    read_only: bool,
) -> None:
    _require_no_extended_acl(root_descriptor, label=f"{label} root")
    for relative in sorted(
        _expected_directories(expected_files),
        key=lambda path: (len(PurePosixPath(path).parts), path.encode("utf-8")),
    ):
        descriptor, _ = _open_relative_directory_acl(
            root_descriptor,
            PurePosixPath(relative).parts,
            label=f"{label} directory {relative!r}",
            private=private,
            read_only=read_only,
        )
        os.close(descriptor)
    for relative in sorted(expected_files, key=lambda path: path.encode("utf-8")):
        descriptor, _ = _open_relative_regular_acl(
            root_descriptor,
            relative,
            label=f"{label} file {relative!r}",
            private=private,
            read_only=read_only,
        )
        os.close(descriptor)


def _scan_exact_tree_acl(
    root_descriptor: int,
    *,
    expected_files: set[str],
    label: str,
    private: bool,
    read_only: bool,
    enforce_view_boundary: bool,
    exact_private_modes: bool = False,
    expected_directory_mode: int = 0o700,
    expected_file_mode: int = 0o600,
) -> None:
    _scan_exact_tree(
        root_descriptor,
        expected_files=expected_files,
        label=label,
        private=private,
        read_only=read_only,
        enforce_view_boundary=enforce_view_boundary,
        exact_private_modes=exact_private_modes,
        expected_directory_mode=expected_directory_mode,
        expected_file_mode=expected_file_mode,
    )
    _require_tree_acl_free(
        root_descriptor,
        expected_files=expected_files,
        label=label,
        private=private,
        read_only=read_only,
    )


def _read_absolute_regular_acl(
    path: Path,
    *,
    maximum: int,
    label: str,
    private_parent: bool,
    read_only: bool,
) -> bytes:
    parent_descriptor, _ = _open_absolute_directory_acl(
        path.parent,
        label=f"{label} parent",
        private=private_parent,
    )
    file_descriptor: int | None = None
    try:
        file_descriptor, before = _open_relative_regular_acl(
            parent_descriptor,
            path.name,
            label=label,
            private=False,
            read_only=read_only,
        )
        return _read_open_regular(
            file_descriptor,
            before,
            maximum=maximum,
            label=label,
        )
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(parent_descriptor)


@dataclass(frozen=True)
class _RetainedAclRecord:
    descriptor: int
    metadata: os.stat_result
    label: str
    directory: bool
    mutable: bool


class _RetainedAclGuard:
    """Recheck ACL absence and descriptor identity across the whole transaction."""

    def __init__(self) -> None:
        self._records: dict[tuple[int, int], _RetainedAclRecord] = {}

    def retain(
        self,
        descriptor: int,
        *,
        label: str,
        directory: bool,
        mutable: bool,
    ) -> None:
        metadata = os.fstat(descriptor)
        _require_no_extended_acl(descriptor, label=label)
        self._records.setdefault(
            (metadata.st_dev, metadata.st_ino),
            _RetainedAclRecord(
                descriptor=descriptor,
                metadata=metadata,
                label=label,
                directory=directory,
                mutable=mutable,
            ),
        )

    def verify(self) -> None:
        for record in self._records.values():
            _require_no_extended_acl(record.descriptor, label=record.label)
            observed = os.fstat(record.descriptor)
            if record.mutable:
                fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid")
                if any(
                    getattr(record.metadata, field) != getattr(observed, field) for field in fields
                ):
                    raise DevelopmentPhase2ViewError(
                        f"{record.label} identity, ownership, or mode changed"
                    )
            elif record.directory:
                _require_stable_directory(record.metadata, observed, label=record.label)
            else:
                _require_stable_file(record.metadata, observed, label=record.label)


def _retain_descriptor(
    leases: _ExclusiveLeaseSet,
    acl_guard: _RetainedAclGuard,
    descriptor: int,
    *,
    label: str,
    directory: bool,
    owned: bool,
    mutable: bool = False,
) -> int:
    try:
        _require_no_extended_acl(descriptor, label=label)
        retained = (
            leases.retain_owned(descriptor, label=label)
            if owned
            else leases.retain_existing(descriptor, label=label)
        )
    except BaseException:
        if owned:
            os.close(descriptor)
        raise
    acl_guard.retain(
        retained,
        label=label,
        directory=directory,
        mutable=mutable,
    )
    return retained


def _retain_exact_tree_leases_acl(
    leases: _ExclusiveLeaseSet,
    acl_guard: _RetainedAclGuard,
    root_descriptor: int,
    *,
    expected_files: set[str],
    label: str,
    private: bool,
    read_only: bool,
) -> None:
    _require_tree_acl_free(
        root_descriptor,
        expected_files=expected_files,
        label=label,
        private=private,
        read_only=read_only,
    )
    for relative in sorted(
        _expected_directories(expected_files),
        key=lambda path: (len(PurePosixPath(path).parts), path.encode("utf-8")),
    ):
        descriptor, _ = _open_relative_directory_acl(
            root_descriptor,
            PurePosixPath(relative).parts,
            label=f"{label} directory {relative!r}",
            private=private,
            read_only=read_only,
        )
        _retain_descriptor(
            leases,
            acl_guard,
            descriptor,
            label=f"{label} directory {relative!r}",
            directory=True,
            owned=True,
        )
    for relative in sorted(expected_files, key=lambda path: path.encode("utf-8")):
        descriptor, _ = _open_relative_regular_acl(
            root_descriptor,
            relative,
            label=f"{label} file {relative!r}",
            private=private,
            read_only=read_only,
        )
        _retain_descriptor(
            leases,
            acl_guard,
            descriptor,
            label=f"{label} file {relative!r}",
            directory=False,
            owned=True,
        )


_PUBLICATION_STABLE_FIELDS = (
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


def _same_publication_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field) == getattr(right, field) for field in _PUBLICATION_STABLE_FIELDS
    )


def _classify_no_replace_move(
    *,
    source_parent: int,
    source_name: str,
    destination_parent: int,
    destination_name: str,
    expected: os.stat_result,
    label: str,
) -> bool:
    """Return True only when the expected inode is solely at the destination."""

    observations: list[tuple[os.stat_result | None, os.stat_result | None]] = []
    for _pass in range(2):
        observations.append(
            (
                _entry_stat(source_parent, source_name),
                _entry_stat(destination_parent, destination_name),
            )
        )
    first, second = observations

    def equivalent(
        left: os.stat_result | None,
        right: os.stat_result | None,
    ) -> bool:
        if left is None or right is None:
            return left is right
        return _same_publication_entry(left, right)

    if not equivalent(first[0], second[0]) or not equivalent(first[1], second[1]):
        raise DevelopmentPhase2PublicationIndeterminate(
            f"{label} names changed while publication state was observed"
        )
    source, destination = second
    at_source = source is not None and (
        source.st_dev,
        source.st_ino,
    ) == (expected.st_dev, expected.st_ino)
    at_destination = destination is not None and (
        destination.st_dev,
        destination.st_ino,
    ) == (expected.st_dev, expected.st_ino)
    if at_source and destination is None:
        return False
    if source is None and at_destination:
        return True
    raise DevelopmentPhase2PublicationIndeterminate(
        f"{label} left an unclassified source/destination name state"
    )


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise DevelopmentPhase2ViewError(f"{name} must be 64 lowercase hex characters")
    return value


def _path_tokens(path: Path) -> frozenset[str]:
    tokens: set[str] = set()
    for part in path.parts:
        folded = part.casefold()
        tokens.add(folded)
        tokens.add(PurePosixPath(folded).stem)
    return frozenset(tokens)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _artifact_contract() -> dict[str, tuple[str, str | None, str | None]]:
    contract: dict[str, tuple[str, str | None, str | None]] = {
        "inventory.json": ("staged-inventory", None, None),
        "inventory.sha256": ("staged-inventory-checksum", None, None),
        "assignments.jsonl": ("assignments", None, None),
    }
    for stage in DEVELOPMENT_SOURCE_STAGES:
        for corpus in FIXED_CORPORA:
            contract[f"datasets/{corpus}/{stage}/queries.jsonl"] = (
                "queries",
                corpus,
                stage,
            )
            contract[f"datasets/{corpus}/{stage}/qrels.jsonl"] = (
                "qrels",
                corpus,
                stage,
            )
            if corpus in EVIDENCE_CORPORA:
                contract[f"datasets/{corpus}/{stage}/evidence-bundles.jsonl"] = (
                    "evidence-bundles",
                    corpus,
                    stage,
                )
    if len(contract) != 29:
        raise AssertionError("phase-two artifact contract must contain exactly 29 files")
    return contract


@dataclass(frozen=True, order=True)
class Phase2Artifact:
    path: str
    sha256: str
    byte_count: int
    record_count: int
    role: str
    dataset: str | None
    stage: str | None
    schema_version: str = PHASE2_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        _relative_path(self.path, label="phase-two artifact path")
        _require_digest("phase-two artifact SHA-256", self.sha256)
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise DevelopmentPhase2ViewError("phase-two artifact byte_count must be positive")
        if type(self.record_count) is not int or self.record_count <= 0:
            raise DevelopmentPhase2ViewError("phase-two artifact record_count must be positive")
        expected = _artifact_contract().get(self.path)
        if expected != (self.role, self.dataset, self.stage):
            raise DevelopmentPhase2ViewError(
                f"phase-two artifact contract differs for {self.path!r}"
            )
        if self.schema_version != PHASE2_ARTIFACT_SCHEMA:
            raise DevelopmentPhase2ViewError("phase-two artifact schema differs")

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
    def from_dict(cls, value: object) -> "Phase2Artifact":
        row = _closed(
            value,
            frozenset(
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
            ),
            label="phase-two artifact",
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Phase2InputCustody:
    capture_set_sha256: str
    contract: str = INPUT_CUSTODY_CONTRACT
    producer_parent_and_file_leases_held_through_publication: bool = True
    noncooperating_same_uid_mutation_excluded: bool = True
    schema_version: str = PHASE2_CUSTODY_SCHEMA

    def __post_init__(self) -> None:
        _require_digest("phase-two capture-set SHA-256", self.capture_set_sha256)
        if self.contract != INPUT_CUSTODY_CONTRACT:
            raise DevelopmentPhase2ViewError("phase-two custody contract differs")
        if (
            self.producer_parent_and_file_leases_held_through_publication is not True
            or self.noncooperating_same_uid_mutation_excluded is not True
        ):
            raise DevelopmentPhase2ViewError("phase-two custody scope differs")
        if self.schema_version != PHASE2_CUSTODY_SCHEMA:
            raise DevelopmentPhase2ViewError("phase-two custody schema differs")

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
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "Phase2InputCustody":
        row = _closed(
            value,
            frozenset(
                {
                    "capture_set_sha256",
                    "contract",
                    "noncooperating_same_uid_mutation_excluded",
                    "producer_parent_and_file_leases_held_through_publication",
                    "schema_version",
                }
            ),
            label="phase-two input custody",
        )
        return cls(**row)  # type: ignore[arg-type]


def _artifact_set_sha256(artifacts: Sequence[Phase2Artifact]) -> str:
    return _sha256(_canonical_value_bytes([artifact.to_dict() for artifact in artifacts]))


def _capture_set_sha256(
    artifacts: Sequence[Phase2Artifact],
    *,
    partition_audit_file_sha256: str,
    phase1_view_receipt_sha256: str,
    selection_receipt_sha256: str,
    seed_commitment_sha256: str,
    seed_attestation_admission_path: Path,
    seed_attestation_admission_sha256: str,
    seed_reveal_sha256: str,
) -> str:
    return _sha256(
        _canonical_value_bytes(
            {
                "artifacts": [
                    {
                        "byte_count": artifact.byte_count,
                        "path": artifact.path,
                        "sha256": artifact.sha256,
                    }
                    for artifact in artifacts
                ],
                "partition_audit_file_sha256": partition_audit_file_sha256,
                "phase1_view_receipt_sha256": phase1_view_receipt_sha256,
                "seed_attestation_admission_path": str(seed_attestation_admission_path),
                "seed_attestation_admission_sha256": seed_attestation_admission_sha256,
                "seed_commitment_sha256": seed_commitment_sha256,
                "seed_reveal_sha256": seed_reveal_sha256,
                "selection_receipt_sha256": selection_receipt_sha256,
            }
        )
    )


@dataclass(frozen=True)
class DevelopmentPhase2ViewReceipt:
    source_root: Path
    output_root: Path
    partition_audit_path: Path
    phase1_view_root: Path
    selection_receipt_path: Path
    seed_commitment_path: Path
    seed_attestation_admission_path: Path
    seed_reveal_path: Path
    staged_inventory_sha256: str
    partition_audit_file_sha256: str
    partition_component_membership_sha256: str
    partition_source_artifact_set_sha256: str
    phase1_view_receipt_sha256: str
    selection_receipt_sha256: str
    seed_commitment_sha256: str
    seed_attestation_admission_sha256: str
    seed_reveal_sha256: str
    design_seed_sha256: str
    artifacts: tuple[Phase2Artifact, ...]
    artifact_set_sha256: str
    input_custody: Phase2InputCustody
    schema_version: str = PHASE2_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "source_root",
            "output_root",
            "partition_audit_path",
            "phase1_view_root",
            "selection_receipt_path",
            "seed_commitment_path",
            "seed_attestation_admission_path",
            "seed_reveal_path",
        ):
            object.__setattr__(self, name, _absolute_path(getattr(self, name), label=name))
        for name in (
            "staged_inventory_sha256",
            "partition_audit_file_sha256",
            "partition_component_membership_sha256",
            "partition_source_artifact_set_sha256",
            "phase1_view_receipt_sha256",
            "selection_receipt_sha256",
            "seed_commitment_sha256",
            "seed_attestation_admission_sha256",
            "seed_reveal_sha256",
            "design_seed_sha256",
            "artifact_set_sha256",
        ):
            _require_digest(name, getattr(self, name))
        artifacts = tuple(self.artifacts)
        if (
            len(artifacts) != 29
            or artifacts != tuple(sorted(artifacts, key=lambda row: row.path.encode("utf-8")))
            or {artifact.path for artifact in artifacts} != set(_artifact_contract())
        ):
            raise DevelopmentPhase2ViewError("phase-two artifact membership differs")
        if _artifact_set_sha256(artifacts) != self.artifact_set_sha256:
            raise DevelopmentPhase2ViewError("phase-two artifact-set digest differs")
        if not isinstance(self.input_custody, Phase2InputCustody):
            raise DevelopmentPhase2ViewError("phase-two input custody must be typed")
        if self.input_custody.capture_set_sha256 != _capture_set_sha256(
            artifacts,
            partition_audit_file_sha256=self.partition_audit_file_sha256,
            phase1_view_receipt_sha256=self.phase1_view_receipt_sha256,
            selection_receipt_sha256=self.selection_receipt_sha256,
            seed_commitment_sha256=self.seed_commitment_sha256,
            seed_attestation_admission_path=self.seed_attestation_admission_path,
            seed_attestation_admission_sha256=self.seed_attestation_admission_sha256,
            seed_reveal_sha256=self.seed_reveal_sha256,
        ):
            raise DevelopmentPhase2ViewError("phase-two custody capture set differs")
        if self.schema_version != PHASE2_RECEIPT_SCHEMA:
            raise DevelopmentPhase2ViewError("phase-two receipt schema differs")
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def view_root(self) -> Path:
        return self.output_root / PHASE2_VIEW_DIRECTORY

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_set_sha256": self.artifact_set_sha256,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "design_seed_sha256": self.design_seed_sha256,
            "input_custody": self.input_custody.to_dict(),
            "output_root": str(self.output_root),
            "partition_audit_file_sha256": self.partition_audit_file_sha256,
            "partition_audit_path": str(self.partition_audit_path),
            "partition_component_membership_sha256": (self.partition_component_membership_sha256),
            "partition_source_artifact_set_sha256": (self.partition_source_artifact_set_sha256),
            "phase1_view_receipt_sha256": self.phase1_view_receipt_sha256,
            "phase1_view_root": str(self.phase1_view_root),
            "schema_version": self.schema_version,
            "seed_attestation_admission_path": str(self.seed_attestation_admission_path),
            "seed_attestation_admission_sha256": self.seed_attestation_admission_sha256,
            "seed_commitment_path": str(self.seed_commitment_path),
            "seed_commitment_sha256": self.seed_commitment_sha256,
            "seed_reveal_path": str(self.seed_reveal_path),
            "seed_reveal_sha256": self.seed_reveal_sha256,
            "selection_receipt_path": str(self.selection_receipt_path),
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "source_root": str(self.source_root),
            "staged_inventory_sha256": self.staged_inventory_sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> "DevelopmentPhase2ViewReceipt":
        fields = frozenset(
            {
                "artifact_set_sha256",
                "artifacts",
                "design_seed_sha256",
                "input_custody",
                "output_root",
                "partition_audit_file_sha256",
                "partition_audit_path",
                "partition_component_membership_sha256",
                "partition_source_artifact_set_sha256",
                "phase1_view_receipt_sha256",
                "phase1_view_root",
                "schema_version",
                "seed_attestation_admission_path",
                "seed_attestation_admission_sha256",
                "seed_commitment_path",
                "seed_commitment_sha256",
                "seed_reveal_path",
                "seed_reveal_sha256",
                "selection_receipt_path",
                "selection_receipt_sha256",
                "source_root",
                "staged_inventory_sha256",
            }
        )
        row = _closed(value, fields, label="phase-two view receipt")
        values = row["artifacts"]
        if not isinstance(values, list):
            raise DevelopmentPhase2ViewError("phase-two artifacts must be an array")
        return cls(
            source_root=Path(row["source_root"]),
            output_root=Path(row["output_root"]),
            partition_audit_path=Path(row["partition_audit_path"]),
            phase1_view_root=Path(row["phase1_view_root"]),
            selection_receipt_path=Path(row["selection_receipt_path"]),
            seed_attestation_admission_path=Path(row["seed_attestation_admission_path"]),
            seed_commitment_path=Path(row["seed_commitment_path"]),
            seed_reveal_path=Path(row["seed_reveal_path"]),
            staged_inventory_sha256=row["staged_inventory_sha256"],
            partition_audit_file_sha256=row["partition_audit_file_sha256"],
            partition_component_membership_sha256=row["partition_component_membership_sha256"],
            partition_source_artifact_set_sha256=row["partition_source_artifact_set_sha256"],
            phase1_view_receipt_sha256=row["phase1_view_receipt_sha256"],
            selection_receipt_sha256=row["selection_receipt_sha256"],
            seed_attestation_admission_sha256=row["seed_attestation_admission_sha256"],
            seed_commitment_sha256=row["seed_commitment_sha256"],
            seed_reveal_sha256=row["seed_reveal_sha256"],
            design_seed_sha256=row["design_seed_sha256"],
            artifacts=tuple(Phase2Artifact.from_dict(item) for item in values),
            artifact_set_sha256=row["artifact_set_sha256"],
            input_custody=Phase2InputCustody.from_dict(row["input_custody"]),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class _LabelFreeAdmission:
    audit: Any
    phase1_receipt: Any
    selection: Any
    seed_commitment: Any
    seed_reveal: Any
    seed_attestation_admission_path: Path
    seed_attestation_admission_sha256: str
    selection_bytes: bytes


@dataclass(frozen=True)
class _SourceAdmission:
    inventory_bytes: bytes
    inventory_checksum_bytes: bytes
    inventory_rows: tuple[SourceArtifact, ...]
    source_files: frozenset[str]
    selected_sources: tuple[SourceArtifact, ...]


def _read_pinned_control(path: Path, *, expected_sha256: str, label: str) -> bytes:
    encoded = _read_absolute_regular_acl(
        path,
        maximum=_MAX_CONTROL_BYTES,
        label=label,
        private_parent=True,
        read_only=True,
    )
    if _sha256(encoded) != expected_sha256:
        raise DevelopmentPhase2ViewError(f"{label} differs from its external pin")
    return encoded


def _reproduce_selection(
    *,
    phase1_view_root: Path,
    staged_inventory_sha256: str,
    partition_audit_path: Path,
    partition_audit_file_sha256: str,
    expected_selection_bytes: bytes,
    scratch_parent: Path,
) -> None:
    name = f".phase-two-selection-{secrets.token_hex(12)}"
    scratch = scratch_parent / name
    try:
        scratch.mkdir(mode=0o700)
        output = scratch / SELECTION_FILENAME
        receipt = select_development_cohort(
            phase1_view_root,
            output,
            staged_inventory_sha256=staged_inventory_sha256,
            partition_audit_path=partition_audit_path,
            partition_audit_sha256=partition_audit_file_sha256,
        )
        reproduced = output.read_bytes()
        if reproduced != expected_selection_bytes or receipt.canonical_file_bytes() != reproduced:
            raise DevelopmentPhase2ViewError(
                "independent development selection does not reproduce byte for byte"
            )
    finally:
        try:
            output = scratch / SELECTION_FILENAME
            if os.path.lexists(output):
                output.unlink()
            if os.path.lexists(scratch):
                scratch.rmdir()
        except OSError as exc:
            raise DevelopmentPhase2ViewError(
                "cannot prove cleanup of the label-free selection reproduction"
            ) from exc


def _verify_seed_chain(
    *,
    commitment_path: Path,
    commitment_sha256: str,
    reveal_path: Path,
    reveal_sha256: str,
) -> tuple[Any, Any, Path, str]:
    try:
        from operators.design_seed_commitment import (
            verify_design_seed_commitment,
            verify_design_seed_reveal,
        )
    except (ImportError, AttributeError) as exc:
        raise DevelopmentPhase2ViewError("the host design-seed verifier is unavailable") from exc
    try:
        commitment = verify_design_seed_commitment(
            commitment_path,
            expected_sha256=commitment_sha256,
        )
        reveal = verify_design_seed_reveal(
            reveal_path,
            expected_sha256=reveal_sha256,
            commitment=commitment,
        )
    except Exception as exc:
        raise DevelopmentPhase2ViewError(f"design-seed admission failed: {exc}") from exc
    admission_path = _absolute_path(
        getattr(reveal, "attestation_admission_path", ""),
        label="seed attestation admission path",
    )
    admission_sha256 = _require_digest(
        "seed attestation admission SHA-256",
        getattr(reveal, "attestation_admission_sha256", None),
    )
    return commitment, reveal, admission_path, admission_sha256


def _admit_label_free_controls(
    *,
    staged_inventory_sha256: str,
    partition_audit_path: Path,
    partition_audit_file_sha256: str,
    phase1_view_root: Path,
    phase1_view_receipt_sha256: str,
    selection_receipt_path: Path,
    selection_receipt_sha256: str,
    seed_commitment_path: Path,
    seed_commitment_sha256: str,
    seed_reveal_path: Path,
    seed_reveal_sha256: str,
    scratch_parent: Path,
) -> _LabelFreeAdmission:
    try:
        audit = load_scalable_partition_audit(
            partition_audit_path,
            expected_artifact_sha256=partition_audit_file_sha256,
            expected_inventory_sha256=staged_inventory_sha256,
        )
    except Exception as exc:
        raise DevelopmentPhase2ViewError(f"partition-audit admission failed: {exc}") from exc
    if audit.cross_stage_component_count != 0:
        raise DevelopmentPhase2ViewError("partition audit contains a cross-stage component")
    try:
        phase1 = verify_development_staging_view(
            phase1_view_root,
            expected_receipt_sha256=phase1_view_receipt_sha256,
        )
    except Exception as exc:
        raise DevelopmentPhase2ViewError(f"phase-one view admission failed: {exc}") from exc
    if (
        phase1.staged_inventory_sha256 != staged_inventory_sha256
        or phase1.partition_audit_file_sha256 != partition_audit_file_sha256
        or phase1.partition_component_membership_sha256 != audit.component_membership_sha256
        or phase1.partition_source_artifact_set_sha256 != audit.source_artifact_set_sha256
    ):
        raise DevelopmentPhase2ViewError("phase-one view names another audited cohort")
    selection_bytes = _read_pinned_control(
        selection_receipt_path,
        expected_sha256=selection_receipt_sha256,
        label="independent development selection receipt",
    )
    try:
        selection = load_development_cohort_selection(
            selection_receipt_path,
            expected_artifact_sha256=selection_receipt_sha256,
            expected_inventory_sha256=staged_inventory_sha256,
        )
    except Exception as exc:
        raise DevelopmentPhase2ViewError(f"selection admission failed: {exc}") from exc
    if (
        selection.partition_audit_sha256 != partition_audit_file_sha256
        or len(selection.selections) != len(FIXED_CORPORA) * len(DEVELOPMENT_SOURCE_STAGES)
        or sum(len(row.selected_query_ids) for row in selection.selections)
        != EXPECTED_SELECTION_COUNT
    ):
        raise DevelopmentPhase2ViewError("independent selection scope differs")
    _reproduce_selection(
        phase1_view_root=phase1_view_root,
        staged_inventory_sha256=staged_inventory_sha256,
        partition_audit_path=partition_audit_path,
        partition_audit_file_sha256=partition_audit_file_sha256,
        expected_selection_bytes=selection_bytes,
        scratch_parent=scratch_parent,
    )
    commitment, reveal, admission_path, admission_sha256 = _verify_seed_chain(
        commitment_path=seed_commitment_path,
        commitment_sha256=seed_commitment_sha256,
        reveal_path=seed_reveal_path,
        reveal_sha256=seed_reveal_sha256,
    )
    bindings = {
        "staged_inventory_sha256": staged_inventory_sha256,
        "partition_audit_file_sha256": partition_audit_file_sha256,
        "phase1_view_receipt_sha256": phase1_view_receipt_sha256,
        "selection_receipt_sha256": selection_receipt_sha256,
    }
    for name, expected in bindings.items():
        observed = getattr(commitment, name, None)
        if observed != expected:
            raise DevelopmentPhase2ViewError(f"design-seed commitment differs at {name}")
    if getattr(reveal, "commitment_sha256", None) != seed_commitment_sha256:
        raise DevelopmentPhase2ViewError("design-seed reveal names another commitment")
    _require_digest("revealed design seed", getattr(reveal, "design_seed_sha256", None))
    return _LabelFreeAdmission(
        audit=audit,
        phase1_receipt=phase1,
        selection=selection,
        seed_commitment=commitment,
        seed_reveal=reveal,
        seed_attestation_admission_path=admission_path,
        seed_attestation_admission_sha256=admission_sha256,
        selection_bytes=selection_bytes,
    )


def _verify_receipt_seed_chain(
    receipt: DevelopmentPhase2ViewReceipt,
) -> tuple[Any, Any]:
    commitment, reveal, admission_path, admission_sha256 = _verify_seed_chain(
        commitment_path=receipt.seed_commitment_path,
        commitment_sha256=receipt.seed_commitment_sha256,
        reveal_path=receipt.seed_reveal_path,
        reveal_sha256=receipt.seed_reveal_sha256,
    )
    bindings = {
        "staged_inventory_sha256": receipt.staged_inventory_sha256,
        "partition_audit_file_sha256": receipt.partition_audit_file_sha256,
        "phase1_view_receipt_sha256": receipt.phase1_view_receipt_sha256,
        "selection_receipt_sha256": receipt.selection_receipt_sha256,
    }
    for name, expected in bindings.items():
        if getattr(commitment, name, None) != expected:
            raise DevelopmentPhase2ViewError(f"phase-two seed commitment differs at {name}")
    if getattr(reveal, "commitment_sha256", None) != receipt.seed_commitment_sha256:
        raise DevelopmentPhase2ViewError("phase-two seed reveal names another commitment")
    if admission_path != receipt.seed_attestation_admission_path:
        raise DevelopmentPhase2ViewError("phase-two seed attestation admission path differs")
    if admission_sha256 != receipt.seed_attestation_admission_sha256:
        raise DevelopmentPhase2ViewError("phase-two seed attestation admission digest differs")
    verified_seed = _require_digest(
        "verified phase-two design seed",
        getattr(reveal, "design_seed_sha256", None),
    )
    if verified_seed != receipt.design_seed_sha256:
        raise DevelopmentPhase2ViewError("phase-two design seed differs from the verified reveal")
    return commitment, reveal


def _admit_source(
    root_descriptor: int,
    *,
    expected_inventory_sha256: str,
    audit: Any,
) -> _SourceAdmission:
    inventory_bytes = _read_relative_regular(
        root_descriptor,
        "inventory.json",
        maximum=_MAX_CONTROL_BYTES,
        label="complete NFC staged inventory",
        private=False,
        read_only=True,
    )
    if _sha256(inventory_bytes) != expected_inventory_sha256:
        raise DevelopmentPhase2ViewError("complete NFC inventory differs from its pin")
    inventory_checksum = _read_relative_regular(
        root_descriptor,
        "inventory.sha256",
        maximum=1024,
        label="complete NFC inventory checksum",
        private=False,
        read_only=True,
    )
    if inventory_checksum != f"{expected_inventory_sha256}  inventory.json\n".encode("ascii"):
        raise DevelopmentPhase2ViewError("complete NFC inventory checksum differs")
    value = _decode(inventory_bytes, label="complete NFC staged inventory")
    inventory = _closed(value, _INVENTORY_FIELDS, label="complete NFC staged inventory")
    if (
        inventory["schema_version"] != INVENTORY_SCHEMA
        or _canonical_bytes(value) != inventory_bytes
    ):
        raise DevelopmentPhase2ViewError("complete NFC inventory is not canonical")
    rows = _source_rows(inventory["artifacts"], label="complete NFC inventory artifacts")
    source_files = frozenset({"inventory.json", "inventory.sha256", *(row.path for row in rows)})
    _scan_exact_tree_acl(
        root_descriptor,
        expected_files=set(source_files),
        label="complete NFC staged root",
        private=False,
        read_only=True,
        enforce_view_boundary=False,
    )
    by_path = {row.path: row for row in rows}
    contract = _artifact_contract()
    selected: list[SourceArtifact] = []
    for path, (role, dataset, stage) in contract.items():
        if path in {"inventory.json", "inventory.sha256"}:
            continue
        source = by_path.get(path)
        if source is None or (source.role, source.dataset, source.stage) != (
            role,
            dataset,
            stage,
        ):
            raise DevelopmentPhase2ViewError(
                f"complete NFC inventory lacks phase-two source {path!r}"
            )
        if source.visibility != "online":
            raise DevelopmentPhase2ViewError(f"development source {path!r} is not online-visible")
        selected.append(source)
    audit_by_path = {row.path: row.to_dict() for row in audit.source_artifacts}
    for source in selected:
        if source.role in {"assignments", "queries", "qrels"} and (
            audit_by_path.get(source.path) != source.to_dict()
        ):
            raise DevelopmentPhase2ViewError(
                f"phase-two source {source.path!r} differs from the partition audit"
            )
    selected_tuple = tuple(sorted(selected, key=lambda row: row.path.encode("utf-8")))
    if len(selected_tuple) != 27:
        raise DevelopmentPhase2ViewError("phase-two payload source count differs")
    return _SourceAdmission(
        inventory_bytes=inventory_bytes,
        inventory_checksum_bytes=inventory_checksum,
        inventory_rows=rows,
        source_files=source_files,
        selected_sources=selected_tuple,
    )


def _control_artifact(path: str, encoded: bytes, role: str) -> Phase2Artifact:
    return Phase2Artifact(
        path=path,
        sha256=_sha256(encoded),
        byte_count=len(encoded),
        record_count=encoded.count(b"\n"),
        role=role,
        dataset=None,
        stage=None,
    )


def _open_or_create_phase2_parent(
    root_descriptor: int,
    relative: str,
) -> tuple[int, str]:
    """Create a private parent for one fixed phase-two path, including judgment paths."""

    parts = PurePosixPath(_relative_path(relative, label="phase-two artifact path")).parts
    descriptor = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            child, metadata = _open_relative_directory_acl(
                descriptor,
                (component,),
                label=f"temporary phase-two directory {component!r}",
                private=True,
                read_only=False,
            )
            _require_exact_mode(
                metadata,
                expected=0o700,
                label=f"temporary phase-two directory {component!r}",
            )
            os.close(descriptor)
            descriptor = child
        result = descriptor
        descriptor = -1
        return result, parts[-1]
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _copy_source(
    *,
    source_root_descriptor: int,
    target_view_descriptor: int,
    source: SourceArtifact,
) -> Phase2Artifact:
    source_descriptor, source_before = _open_relative_regular_acl(
        source_root_descriptor,
        source.path,
        label=f"phase-two source {source.path!r}",
        private=False,
        read_only=True,
    )
    target_parent, target_name = _open_or_create_phase2_parent(
        target_view_descriptor,
        source.path,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    target_descriptor: int | None = None
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    last_byte: int | None = None
    try:
        target_descriptor = os.open(target_name, flags, 0o600, dir_fd=target_parent)
        os.fchmod(target_descriptor, 0o600)
        _require_no_extended_acl(
            target_descriptor,
            label=f"temporary phase-two artifact {source.path!r}",
        )
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
                    raise DevelopmentPhase2ViewError(f"short write while copying {source.path!r}")
                view = view[written:]
        os.fsync(target_descriptor)
        _require_stable_file(
            source_before,
            os.fstat(source_descriptor),
            label=f"phase-two source {source.path!r}",
        )
        if (
            digest.hexdigest() != source.sha256
            or byte_count != source.byte_count
            or record_count != source.record_count
            or last_byte != 10
        ):
            raise DevelopmentPhase2ViewError(
                f"phase-two source {source.path!r} differs from its inventory pin"
            )
    finally:
        os.close(source_descriptor)
        if target_descriptor is not None:
            os.close(target_descriptor)
        os.close(target_parent)
    return Phase2Artifact(
        path=source.path,
        sha256=source.sha256,
        byte_count=source.byte_count,
        record_count=source.record_count,
        role=source.role,
        dataset=source.dataset,
        stage=source.stage,
    )


def _package_files(artifacts: Sequence[Phase2Artifact]) -> set[str]:
    return {
        PHASE2_RECEIPT_FILENAME,
        *(f"{PHASE2_VIEW_DIRECTORY}/{artifact.path}" for artifact in artifacts),
    }


def _seal_package(root_descriptor: int, expected_files: set[str]) -> None:
    for relative in sorted(expected_files, key=lambda value: value.encode("utf-8")):
        descriptor, _ = _open_relative_regular_acl(
            root_descriptor,
            relative,
            label=f"temporary phase-two member {relative!r}",
            private=True,
            read_only=False,
        )
        try:
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            _require_exact_mode(
                os.fstat(descriptor),
                expected=0o400,
                label=f"sealed phase-two member {relative!r}",
            )
        finally:
            os.close(descriptor)
    directories = sorted(
        _expected_directories(expected_files),
        key=lambda value: (-len(PurePosixPath(value).parts), value.encode("utf-8")),
    )
    for relative in directories:
        descriptor, _ = _open_relative_directory_acl(
            root_descriptor,
            PurePosixPath(relative).parts,
            label=f"temporary phase-two directory {relative!r}",
            private=True,
            read_only=False,
        )
        try:
            os.fchmod(descriptor, 0o500)
            os.fsync(descriptor)
            _require_exact_mode(
                os.fstat(descriptor),
                expected=0o500,
                label=f"sealed phase-two directory {relative!r}",
            )
        finally:
            os.close(descriptor)
    os.fchmod(root_descriptor, 0o500)
    os.fsync(root_descriptor)
    _require_exact_mode(
        os.fstat(root_descriptor),
        expected=0o500,
        label="sealed phase-two package root",
    )
    _scan_exact_tree_acl(
        root_descriptor,
        expected_files=expected_files,
        label="sealed phase-two package",
        private=True,
        read_only=True,
        enforce_view_boundary=False,
        exact_private_modes=True,
        expected_directory_mode=0o500,
        expected_file_mode=0o400,
    )


def _load_phase2_receipt_from_descriptor(
    root_descriptor: int,
    *,
    expected_sha256: str,
    read_only: bool,
) -> DevelopmentPhase2ViewReceipt:
    encoded = _read_relative_regular(
        root_descriptor,
        PHASE2_RECEIPT_FILENAME,
        maximum=_MAX_CONTROL_BYTES,
        label="phase-two view receipt",
        private=True,
        read_only=read_only,
    )
    if _sha256(encoded) != expected_sha256:
        raise DevelopmentPhase2ViewError("phase-two receipt differs from its external pin")
    receipt = DevelopmentPhase2ViewReceipt.from_dict(
        _decode(encoded, label="phase-two view receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise DevelopmentPhase2ViewError("phase-two receipt is not canonical")
    return receipt


def _verify_phase2_descriptor(
    root_descriptor: int,
    *,
    root: Path,
    expected_receipt_sha256: str,
    sealed: bool,
) -> DevelopmentPhase2ViewReceipt:
    receipt = _load_phase2_receipt_from_descriptor(
        root_descriptor,
        expected_sha256=expected_receipt_sha256,
        read_only=sealed,
    )
    if receipt.output_root != root:
        raise DevelopmentPhase2ViewError("phase-two receipt names another output root")
    expected_files = _package_files(receipt.artifacts)
    _scan_exact_tree_acl(
        root_descriptor,
        expected_files=expected_files,
        label="phase-two package",
        private=True,
        read_only=sealed,
        enforce_view_boundary=False,
        exact_private_modes=True,
        expected_directory_mode=0o500 if sealed else 0o700,
        expected_file_mode=0o400 if sealed else 0o600,
    )
    for artifact in receipt.artifacts:
        observed, _ = _fingerprint_regular_file(
            root_descriptor,
            f"{PHASE2_VIEW_DIRECTORY}/{artifact.path}",
            label=f"phase-two artifact {artifact.path!r}",
            private=True,
            read_only=sealed,
        )
        if observed != (
            artifact.sha256,
            artifact.byte_count,
            artifact.record_count,
            10,
        ):
            raise DevelopmentPhase2ViewError(
                f"phase-two artifact {artifact.path!r} differs from its receipt"
            )
    receipt_bytes = _read_relative_regular(
        root_descriptor,
        PHASE2_RECEIPT_FILENAME,
        maximum=_MAX_CONTROL_BYTES,
        label="phase-two view receipt final read",
        private=True,
        read_only=sealed,
    )
    if receipt_bytes != receipt.canonical_file_bytes():
        raise DevelopmentPhase2ViewError("phase-two receipt changed during verification")
    _scan_exact_tree_acl(
        root_descriptor,
        expected_files=expected_files,
        label="phase-two package final scan",
        private=True,
        read_only=sealed,
        enforce_view_boundary=False,
        exact_private_modes=True,
        expected_directory_mode=0o500 if sealed else 0o700,
        expected_file_mode=0o400 if sealed else 0o600,
    )
    return receipt


def _verify_development_phase2_view_with_seed(
    root: str | Path,
    *,
    expected_receipt_sha256: str,
) -> tuple[DevelopmentPhase2ViewReceipt, Any]:
    package = _absolute_path(root, label="phase-two package root")
    expected = _require_digest("phase-two receipt SHA-256", expected_receipt_sha256)
    descriptor, before = _open_absolute_directory_acl(
        package,
        label="phase-two package root",
        private=True,
        read_only=True,
    )
    try:
        receipt = _verify_phase2_descriptor(
            descriptor,
            root=package,
            expected_receipt_sha256=expected,
            sealed=True,
        )
        _require_stable_directory(before, os.fstat(descriptor), label="phase-two package root")
        _commitment, reveal = _verify_receipt_seed_chain(receipt)
        _require_stable_directory(before, os.fstat(descriptor), label="phase-two package root")
        return receipt, reveal
    finally:
        os.close(descriptor)


def verify_development_phase2_view(
    root: str | Path,
    *,
    expected_receipt_sha256: str,
) -> DevelopmentPhase2ViewReceipt:
    """Verify package bytes and freshly rederive its committed design seed."""

    receipt, _reveal = _verify_development_phase2_view_with_seed(
        root,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    return receipt


def _lock_control(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    leases: _ExclusiveLeaseSet,
    acl_guard: _RetainedAclGuard,
) -> None:
    parent_descriptor, _ = _open_absolute_directory_acl(
        path.parent,
        label=f"{label} parent",
        private=True,
    )
    file_descriptor: int | None = None
    try:
        file_descriptor, before = _open_relative_regular_acl(
            parent_descriptor,
            path.name,
            label=label,
            private=False,
            read_only=True,
        )
        owned_file = file_descriptor
        file_descriptor = None
        retained_file = _retain_descriptor(
            leases,
            acl_guard,
            owned_file,
            label=label,
            directory=False,
            owned=True,
        )
        owned_parent = parent_descriptor
        parent_descriptor = -1
        _retain_descriptor(
            leases,
            acl_guard,
            owned_parent,
            label=f"{label} parent",
            directory=True,
            owned=True,
        )
        encoded = _read_open_regular(
            retained_file,
            before,
            maximum=_MAX_CONTROL_BYTES,
            label=label,
        )
        if _sha256(encoded) != expected_sha256:
            raise DevelopmentPhase2ViewError(f"{label} changed before custody admission")
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def build_development_phase2_view(
    *,
    source_root: str | Path,
    staged_inventory_sha256: str,
    partition_audit_path: str | Path,
    partition_audit_file_sha256: str,
    phase1_view_root: str | Path,
    phase1_view_receipt_sha256: str,
    selection_receipt_path: str | Path,
    selection_receipt_sha256: str,
    seed_commitment_path: str | Path,
    seed_commitment_sha256: str,
    seed_reveal_path: str | Path,
    seed_reveal_sha256: str,
    output_root: str | Path,
) -> DevelopmentPhase2ViewReceipt:
    """Publish one fixed 29-file development view after all label-free gates pass."""

    _require_nonroot()
    source = _absolute_path(source_root, label="complete NFC source root")
    audit_path = _absolute_path(partition_audit_path, label="partition audit path")
    phase1_root = _absolute_path(phase1_view_root, label="phase-one view root")
    selection_path = _absolute_path(selection_receipt_path, label="selection receipt path")
    commitment_path = _absolute_path(seed_commitment_path, label="seed commitment path")
    reveal_path = _absolute_path(seed_reveal_path, label="seed reveal path")
    output = _absolute_path(output_root, label="phase-two output root")
    inventory_pin = _require_digest("staged inventory SHA-256", staged_inventory_sha256)
    audit_pin = _require_digest("partition audit file SHA-256", partition_audit_file_sha256)
    phase1_pin = _require_digest("phase-one view receipt SHA-256", phase1_view_receipt_sha256)
    selection_pin = _require_digest("selection receipt SHA-256", selection_receipt_sha256)
    commitment_pin = _require_digest("seed commitment SHA-256", seed_commitment_sha256)
    reveal_pin = _require_digest("seed reveal SHA-256", seed_reveal_sha256)
    if _path_tokens(output / PHASE2_VIEW_DIRECTORY) & _FORBIDDEN_MOUNT_TOKENS:
        raise DevelopmentPhase2ViewError("phase-two mounted path crosses a forbidden boundary")
    inputs = (source, audit_path, phase1_root, selection_path, commitment_path, reveal_path)
    if any(_paths_overlap(output, path) for path in inputs):
        raise DevelopmentPhase2ViewError("phase-two output overlaps an input")
    parent_descriptor, parent_before = _open_absolute_directory_acl(
        output.parent,
        label="phase-two output parent",
        private=True,
        read_only=False,
    )
    if _entry_stat(parent_descriptor, output.name) is not None:
        os.close(parent_descriptor)
        raise DevelopmentPhase2ViewError("phase-two output already exists")

    source_descriptor: int | None = None
    temporary_descriptor: int | None = None
    view_descriptor: int | None = None
    temporary_name: str | None = None
    leases = _ExclusiveLeaseSet()
    acl_guard = _RetainedAclGuard()
    publication_state = "absent"
    receipt: DevelopmentPhase2ViewReceipt | None = None
    try:
        # Discover the full label-free chain first.  The reveal resolves the
        # attestation-admission control that must join the retained lease set.
        label_free = _admit_label_free_controls(
            staged_inventory_sha256=inventory_pin,
            partition_audit_path=audit_path,
            partition_audit_file_sha256=audit_pin,
            phase1_view_root=phase1_root,
            phase1_view_receipt_sha256=phase1_pin,
            selection_receipt_path=selection_path,
            selection_receipt_sha256=selection_pin,
            seed_commitment_path=commitment_path,
            seed_commitment_sha256=commitment_pin,
            seed_reveal_path=reveal_path,
            seed_reveal_sha256=reveal_pin,
            scratch_parent=output.parent,
        )
        _retain_descriptor(
            leases,
            acl_guard,
            parent_descriptor,
            label="phase-two output parent",
            directory=True,
            owned=False,
            mutable=True,
        )
        for path, pin, label in (
            (audit_path, audit_pin, "partition audit"),
            (selection_path, selection_pin, "independent selection"),
            (commitment_path, commitment_pin, "design-seed commitment"),
            (
                label_free.seed_attestation_admission_path,
                label_free.seed_attestation_admission_sha256,
                "design-seed attestation admission",
            ),
            (reveal_path, reveal_pin, "design-seed reveal"),
        ):
            _lock_control(
                path,
                expected_sha256=pin,
                label=label,
                leases=leases,
                acl_guard=acl_guard,
            )
        phase1_parent_descriptor, _ = _open_absolute_directory_acl(
            phase1_root.parent,
            label="phase-one view parent",
            private=True,
            read_only=False,
        )
        phase1_parent_descriptor = _retain_descriptor(
            leases,
            acl_guard,
            phase1_parent_descriptor,
            label="phase-one view parent",
            directory=True,
            owned=True,
        )
        phase1_descriptor, _ = _open_absolute_directory_acl(
            phase1_root,
            label="phase-one view root",
            private=True,
            read_only=True,
        )
        phase1_descriptor = _retain_descriptor(
            leases,
            acl_guard,
            phase1_descriptor,
            label="phase-one view root",
            directory=True,
            owned=True,
        )
        _retain_exact_tree_leases_acl(
            leases,
            acl_guard,
            phase1_descriptor,
            expected_files={
                *(artifact.path for artifact in label_free.phase1_receipt.artifacts),
                PHASE1_RECEIPT_FILENAME,
            },
            label="phase-one view",
            private=True,
            read_only=True,
        )
        readmitted_label_free = _admit_label_free_controls(
            staged_inventory_sha256=inventory_pin,
            partition_audit_path=audit_path,
            partition_audit_file_sha256=audit_pin,
            phase1_view_root=phase1_root,
            phase1_view_receipt_sha256=phase1_pin,
            selection_receipt_path=selection_path,
            selection_receipt_sha256=selection_pin,
            seed_commitment_path=commitment_path,
            seed_commitment_sha256=commitment_pin,
            seed_reveal_path=reveal_path,
            seed_reveal_sha256=reveal_pin,
            scratch_parent=output.parent,
        )
        if readmitted_label_free != label_free:
            raise DevelopmentPhase2ViewError(
                "label-free controls changed while leases were acquired"
            )
        acl_guard.verify()

        # The complete, label-bearing source remains unopened until every
        # label-free control has been reverified under its retained lease.
        source_descriptor, source_before = _open_absolute_directory_acl(
            source,
            label="complete NFC source root",
            private=False,
            read_only=True,
        )
        _retain_descriptor(
            leases,
            acl_guard,
            source_descriptor,
            label="complete NFC source root",
            directory=True,
            owned=False,
        )
        discovered = _admit_source(
            source_descriptor,
            expected_inventory_sha256=inventory_pin,
            audit=label_free.audit,
        )
        _retain_exact_tree_leases_acl(
            leases,
            acl_guard,
            source_descriptor,
            expected_files=set(discovered.source_files),
            label="complete NFC source root",
            private=False,
            read_only=True,
        )
        admitted = _admit_source(
            source_descriptor,
            expected_inventory_sha256=inventory_pin,
            audit=label_free.audit,
        )
        if admitted != discovered:
            raise DevelopmentPhase2ViewError("complete NFC source changed during lease admission")
        _require_stable_directory(source_before, os.fstat(source_descriptor), label="NFC source")
        _require_stable_directory(parent_before, os.fstat(parent_descriptor), label="output parent")

        temporary_name, temporary_descriptor = _create_temporary_tree(
            parent_descriptor,
            output.name,
        )
        publication_state = "temporary"
        _require_no_extended_acl(temporary_descriptor, label="temporary phase-two root")
        os.mkdir(PHASE2_VIEW_DIRECTORY, mode=0o700, dir_fd=temporary_descriptor)
        view_descriptor = os.open(
            PHASE2_VIEW_DIRECTORY,
            _directory_open_flags(),
            dir_fd=temporary_descriptor,
        )
        _require_no_extended_acl(view_descriptor, label="temporary phase-two view")
        _write_exclusive_at(view_descriptor, "inventory.json", admitted.inventory_bytes)
        _write_exclusive_at(
            view_descriptor,
            "inventory.sha256",
            admitted.inventory_checksum_bytes,
        )
        artifacts: list[Phase2Artifact] = [
            _control_artifact("inventory.json", admitted.inventory_bytes, "staged-inventory"),
            _control_artifact(
                "inventory.sha256",
                admitted.inventory_checksum_bytes,
                "staged-inventory-checksum",
            ),
        ]
        artifacts.extend(
            _copy_source(
                source_root_descriptor=source_descriptor,
                target_view_descriptor=view_descriptor,
                source=item,
            )
            for item in admitted.selected_sources
        )
        artifact_tuple = tuple(sorted(artifacts, key=lambda row: row.path.encode("utf-8")))
        receipt = DevelopmentPhase2ViewReceipt(
            source_root=source,
            output_root=output,
            partition_audit_path=audit_path,
            phase1_view_root=phase1_root,
            selection_receipt_path=selection_path,
            seed_commitment_path=commitment_path,
            seed_attestation_admission_path=(label_free.seed_attestation_admission_path),
            seed_reveal_path=reveal_path,
            staged_inventory_sha256=inventory_pin,
            partition_audit_file_sha256=audit_pin,
            partition_component_membership_sha256=(label_free.audit.component_membership_sha256),
            partition_source_artifact_set_sha256=(label_free.audit.source_artifact_set_sha256),
            phase1_view_receipt_sha256=phase1_pin,
            selection_receipt_sha256=selection_pin,
            seed_commitment_sha256=commitment_pin,
            seed_attestation_admission_sha256=(label_free.seed_attestation_admission_sha256),
            seed_reveal_sha256=reveal_pin,
            design_seed_sha256=label_free.seed_reveal.design_seed_sha256,
            artifacts=artifact_tuple,
            artifact_set_sha256=_artifact_set_sha256(artifact_tuple),
            input_custody=Phase2InputCustody(
                capture_set_sha256=_capture_set_sha256(
                    artifact_tuple,
                    partition_audit_file_sha256=audit_pin,
                    phase1_view_receipt_sha256=phase1_pin,
                    selection_receipt_sha256=selection_pin,
                    seed_commitment_sha256=commitment_pin,
                    seed_attestation_admission_path=(label_free.seed_attestation_admission_path),
                    seed_attestation_admission_sha256=(
                        label_free.seed_attestation_admission_sha256
                    ),
                    seed_reveal_sha256=reveal_pin,
                )
            ),
        )
        _write_exclusive_at(
            temporary_descriptor,
            PHASE2_RECEIPT_FILENAME,
            receipt.canonical_file_bytes(),
        )
        os.fsync(view_descriptor)
        os.fsync(temporary_descriptor)
        _verify_phase2_descriptor(
            temporary_descriptor,
            root=output,
            expected_receipt_sha256=receipt.artifact_sha256,
            sealed=False,
        )
        _seal_package(temporary_descriptor, _package_files(receipt.artifacts))
        _require_stable_directory(source_before, os.fstat(source_descriptor), label="NFC source")
        acl_guard.verify()
        if _entry_stat(parent_descriptor, output.name) is not None:
            raise DevelopmentPhase2ViewError("phase-two output appeared before publication")
        expected_output = os.fstat(temporary_descriptor)
        publication_state = "indeterminate"
        try:
            _rename_sealed_exclusive_at(
                parent_descriptor,
                temporary_name,
                output.name,
                temporary_descriptor,
            )
        except BaseException:
            moved = _classify_no_replace_move(
                source_parent=parent_descriptor,
                source_name=temporary_name,
                destination_parent=parent_descriptor,
                destination_name=output.name,
                expected=expected_output,
                label="phase-two publication",
            )
            publication_state = "published" if moved else "temporary"
            raise
        moved = _classify_no_replace_move(
            source_parent=parent_descriptor,
            source_name=temporary_name,
            destination_parent=parent_descriptor,
            destination_name=output.name,
            expected=expected_output,
            label="phase-two publication",
        )
        publication_state = "published" if moved else "temporary"
        if publication_state != "published":
            raise DevelopmentPhase2PublicationIndeterminate(
                "phase-two publication returned without moving the pinned package"
            )
        os.fsync(parent_descriptor)
        try:
            verified = verify_development_phase2_view(
                output,
                expected_receipt_sha256=receipt.artifact_sha256,
            )
            if verified != receipt:
                raise DevelopmentPhase2ViewError("published phase-two receipt changed")
            acl_guard.verify()
            rebound = _classify_no_replace_move(
                source_parent=parent_descriptor,
                source_name=temporary_name,
                destination_parent=parent_descriptor,
                destination_name=output.name,
                expected=expected_output,
                label="phase-two post-publication rebind",
            )
            if not rebound:
                raise DevelopmentPhase2PublicationIndeterminate(
                    "phase-two final name no longer identifies the pinned package"
                )
        except BaseException as publication_error:
            publication_state = "indeterminate"
            try:
                _rename_exclusive_at(parent_descriptor, output.name, temporary_name)
            except BaseException:
                rolled_back = _classify_no_replace_move(
                    source_parent=parent_descriptor,
                    source_name=output.name,
                    destination_parent=parent_descriptor,
                    destination_name=temporary_name,
                    expected=expected_output,
                    label="phase-two rollback",
                )
                publication_state = "temporary" if rolled_back else "published"
                if not rolled_back:
                    raise DevelopmentPhase2PublicationIndeterminate(
                        f"phase-two publication for {output} could not be rolled back"
                    ) from publication_error
            else:
                rolled_back = _classify_no_replace_move(
                    source_parent=parent_descriptor,
                    source_name=output.name,
                    destination_parent=parent_descriptor,
                    destination_name=temporary_name,
                    expected=expected_output,
                    label="phase-two rollback",
                )
                publication_state = "temporary" if rolled_back else "published"
                if not rolled_back:
                    raise DevelopmentPhase2PublicationIndeterminate(
                        f"phase-two publication for {output} could not be rolled back"
                    ) from publication_error
            try:
                os.fsync(parent_descriptor)
            except BaseException as rollback_error:
                publication_state = "indeterminate"
                raise DevelopmentPhase2PublicationIndeterminate(
                    f"phase-two publication for {output} could not be verified or rolled back"
                ) from rollback_error
            raise publication_error
        publication_state = "published"
        return receipt
    except DevelopmentStagingViewError as exc:
        raise DevelopmentPhase2ViewError(str(exc)) from exc
    finally:
        active_error = sys.exc_info()[1]
        if view_descriptor is not None:
            os.close(view_descriptor)
        if (
            temporary_descriptor is not None
            and temporary_name is not None
            and publication_state == "temporary"
        ):
            try:
                if _entry_stat(parent_descriptor, temporary_name) is not None:
                    _remove_temporary_tree(
                        parent_descriptor,
                        temporary_name,
                        output.name,
                        temporary_descriptor,
                        require_output_absent=True,
                    )
            except BaseException as cleanup_error:
                if active_error is None:
                    raise DevelopmentPhase2PublicationIndeterminate(
                        "phase-two temporary cleanup is indeterminate"
                    ) from cleanup_error
        try:
            leases.close_owned()
        finally:
            if temporary_descriptor is not None:
                os.close(temporary_descriptor)
            if source_descriptor is not None:
                os.close(source_descriptor)
            os.close(parent_descriptor)


@dataclass(frozen=True)
class PostResumeBootstrapReceipt:
    phase2_root: Path
    phase2_receipt_sha256: str
    post_config_path: Path
    post_config_sha256: str
    selection_receipt_sha256: str
    design_seed_sha256: str
    staged_inventory_sha256: str
    partition_audit_file_sha256: str
    post_output_root: Path
    initial_artifact_set_sha256: str
    schema_version: str = BOOTSTRAP_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("phase2_root", "post_config_path", "post_output_root"):
            object.__setattr__(self, name, _absolute_path(getattr(self, name), label=name))
        for name in (
            "phase2_receipt_sha256",
            "post_config_sha256",
            "selection_receipt_sha256",
            "design_seed_sha256",
            "staged_inventory_sha256",
            "partition_audit_file_sha256",
            "initial_artifact_set_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if self.schema_version != BOOTSTRAP_RECEIPT_SCHEMA:
            raise DevelopmentPhase2ViewError("post-resume bootstrap schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "design_seed_sha256": self.design_seed_sha256,
            "initial_artifact_set_sha256": self.initial_artifact_set_sha256,
            "partition_audit_file_sha256": self.partition_audit_file_sha256,
            "phase2_receipt_sha256": self.phase2_receipt_sha256,
            "phase2_root": str(self.phase2_root),
            "post_config_path": str(self.post_config_path),
            "post_config_sha256": self.post_config_sha256,
            "post_output_root": str(self.post_output_root),
            "schema_version": self.schema_version,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "staged_inventory_sha256": self.staged_inventory_sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())


def _initial_post_artifact_set_sha256(config_bytes: bytes, selection_bytes: bytes) -> str:
    return _sha256(
        _canonical_value_bytes(
            [
                {
                    "byte_count": len(config_bytes),
                    "path": OPERATOR_CONFIG_FILENAME,
                    "sha256": _sha256(config_bytes),
                },
                {
                    "byte_count": len(selection_bytes),
                    "path": SELECTION_FILENAME,
                    "sha256": _sha256(selection_bytes),
                },
            ]
        )
    )


def _write_external_receipt_exclusive(path: Path, encoded: bytes) -> None:
    parent, _ = _open_absolute_directory_acl(
        path.parent,
        label="bootstrap receipt parent",
        private=True,
        read_only=False,
    )
    temporary = f".{path.name}.tmp-{secrets.token_hex(12)}"
    descriptor: int | None = None
    leases = _ExclusiveLeaseSet()
    acl_guard = _RetainedAclGuard()
    publication_state = "absent"
    try:
        _retain_descriptor(
            leases,
            acl_guard,
            parent,
            label="bootstrap receipt parent",
            directory=True,
            owned=False,
            mutable=True,
        )
        if _entry_stat(parent, path.name) is not None:
            raise DevelopmentPhase2ViewError("bootstrap receipt output already exists")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent)
        os.fchmod(descriptor, 0o600)
        _require_no_extended_acl(descriptor, label="temporary bootstrap receipt")
        publication_state = "temporary"
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise DevelopmentPhase2ViewError("short write for bootstrap receipt")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        _require_exact_mode(
            os.fstat(descriptor),
            expected=0o400,
            label="sealed bootstrap receipt",
        )
        _require_no_extended_acl(descriptor, label="sealed bootstrap receipt")
        expected_receipt = os.fstat(descriptor)
        publication_state = "indeterminate"
        try:
            _rename_exclusive_at(parent, temporary, path.name)
        except BaseException as rename_error:
            moved = _classify_no_replace_move(
                source_parent=parent,
                source_name=temporary,
                destination_parent=parent,
                destination_name=path.name,
                expected=expected_receipt,
                label="bootstrap receipt publication",
            )
            publication_state = "published" if moved else "temporary"
            if moved:
                raise DevelopmentPhase2PublicationIndeterminate(
                    "bootstrap receipt publication completed but its syscall outcome "
                    "was interrupted"
                ) from rename_error
            raise
        moved = _classify_no_replace_move(
            source_parent=parent,
            source_name=temporary,
            destination_parent=parent,
            destination_name=path.name,
            expected=expected_receipt,
            label="bootstrap receipt publication",
        )
        publication_state = "published" if moved else "temporary"
        if publication_state != "published":
            raise DevelopmentPhase2PublicationIndeterminate(
                "bootstrap receipt publication returned without moving the pinned receipt"
            )
        os.fsync(parent)
        published_descriptor, published_before = _open_relative_regular_acl(
            parent,
            path.name,
            label="published bootstrap receipt",
            private=False,
            read_only=True,
        )
        try:
            observed = _read_open_regular(
                published_descriptor,
                published_before,
                maximum=_MAX_CONTROL_BYTES,
                label="published bootstrap receipt",
            )
        finally:
            os.close(published_descriptor)
        rebound = _classify_no_replace_move(
            source_parent=parent,
            source_name=temporary,
            destination_parent=parent,
            destination_name=path.name,
            expected=expected_receipt,
            label="bootstrap receipt final rebind",
        )
        acl_guard.verify()
        if observed != encoded or not rebound:
            raise DevelopmentPhase2PublicationIndeterminate(
                "published bootstrap receipt bytes or final name are indeterminate"
            )
    except BaseException as error:
        if publication_state == "published" and not isinstance(
            error,
            DevelopmentPhase2PublicationIndeterminate,
        ):
            raise DevelopmentPhase2PublicationIndeterminate(
                "bootstrap receipt was published but final verification did not complete"
            ) from error
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if publication_state == "temporary" and _entry_stat(parent, temporary) is not None:
            os.unlink(temporary, dir_fd=parent)
            os.fsync(parent)
        try:
            leases.close_owned()
        finally:
            os.close(parent)


def bootstrap_post_embedding_resume(
    *,
    phase2_root: str | Path,
    phase2_receipt_sha256: str,
    post_config_path: str | Path,
    post_config_sha256: str,
    post_output_root: str | Path,
    bootstrap_receipt_output: str | Path,
) -> PostResumeBootstrapReceipt:
    """Create the exact two-file prefix that forces exact-P to use ``resume``."""

    _require_nonroot()
    package = _absolute_path(phase2_root, label="phase-two package root")
    phase2_pin = _require_digest("phase-two receipt SHA-256", phase2_receipt_sha256)
    config_path = _absolute_path(post_config_path, label="post-embedding config path")
    config_pin = _require_digest("post-embedding config SHA-256", post_config_sha256)
    output = _absolute_path(post_output_root, label="post-embedding output root")
    receipt_output = _absolute_path(
        bootstrap_receipt_output,
        label="bootstrap receipt output",
    )
    if any(_paths_overlap(output, path) for path in (package, config_path, receipt_output)):
        raise DevelopmentPhase2ViewError("post-resume bootstrap paths overlap")
    phase2, verified_reveal = _verify_development_phase2_view_with_seed(
        package,
        expected_receipt_sha256=phase2_pin,
    )
    verified_design_seed = _require_digest(
        "verified post-resume design seed",
        getattr(verified_reveal, "design_seed_sha256", None),
    )
    try:
        config = load_post_embedding_development_config(
            config_path,
            expected_sha256=config_pin,
        )
    except Exception as exc:
        raise DevelopmentPhase2ViewError(f"post-embedding config admission failed: {exc}") from exc
    config_bytes = _read_pinned_control(
        config_path,
        expected_sha256=config_pin,
        label="post-embedding config",
    )
    selection_bytes = _read_pinned_control(
        phase2.selection_receipt_path,
        expected_sha256=phase2.selection_receipt_sha256,
        label="independent selection receipt",
    )
    expected = {
        "full_staged_root": phase2.view_root,
        "full_staged_inventory_sha256": phase2.staged_inventory_sha256,
        "partition_audit_path": phase2.partition_audit_path,
        "partition_audit_file_sha256": phase2.partition_audit_file_sha256,
        "design_seed_sha256": verified_design_seed,
        "output_root": output,
    }
    mismatches = [name for name, value in expected.items() if getattr(config, name) != value]
    if mismatches:
        raise DevelopmentPhase2ViewError(
            "post-embedding config differs at: " + ", ".join(sorted(mismatches))
        )
    parent, _ = _open_absolute_directory_acl(
        output.parent,
        label="post-embedding output parent",
        private=True,
        read_only=False,
    )
    temporary_descriptor: int | None = None
    temporary_name: str | None = None
    leases = _ExclusiveLeaseSet()
    acl_guard = _RetainedAclGuard()
    publication_state = "absent"
    receipt_published = False
    try:
        _retain_descriptor(
            leases,
            acl_guard,
            parent,
            label="post-embedding output parent",
            directory=True,
            owned=False,
            mutable=True,
        )
        if _entry_stat(parent, output.name) is not None:
            raise DevelopmentPhase2ViewError("post-embedding output already exists")
        temporary_name, temporary_descriptor = _create_temporary_tree(parent, output.name)
        publication_state = "temporary"
        _require_no_extended_acl(temporary_descriptor, label="temporary post-resume prefix")
        _write_exclusive_at(
            temporary_descriptor,
            OPERATOR_CONFIG_FILENAME,
            config_bytes,
        )
        _write_exclusive_at(
            temporary_descriptor,
            SELECTION_FILENAME,
            selection_bytes,
        )
        expected_files = {OPERATOR_CONFIG_FILENAME, SELECTION_FILENAME}
        _scan_exact_tree_acl(
            temporary_descriptor,
            expected_files=expected_files,
            label="post-resume bootstrap prefix",
            private=True,
            read_only=False,
            enforce_view_boundary=False,
            exact_private_modes=True,
            expected_directory_mode=0o700,
            expected_file_mode=0o600,
        )
        os.fsync(temporary_descriptor)
        initial_digest = _initial_post_artifact_set_sha256(config_bytes, selection_bytes)
        receipt = PostResumeBootstrapReceipt(
            phase2_root=package,
            phase2_receipt_sha256=phase2_pin,
            post_config_path=config_path,
            post_config_sha256=config_pin,
            selection_receipt_sha256=phase2.selection_receipt_sha256,
            design_seed_sha256=verified_design_seed,
            staged_inventory_sha256=phase2.staged_inventory_sha256,
            partition_audit_file_sha256=phase2.partition_audit_file_sha256,
            post_output_root=output,
            initial_artifact_set_sha256=initial_digest,
        )
        expected_output = os.fstat(temporary_descriptor)
        publication_state = "indeterminate"
        try:
            _rename_exclusive_at(parent, temporary_name, output.name)
        except BaseException as rename_error:
            moved = _classify_no_replace_move(
                source_parent=parent,
                source_name=temporary_name,
                destination_parent=parent,
                destination_name=output.name,
                expected=expected_output,
                label="post-resume prefix publication",
            )
            publication_state = "published" if moved else "temporary"
            if moved:
                raise DevelopmentPhase2PublicationIndeterminate(
                    "post-resume prefix publication completed but its syscall outcome "
                    "was interrupted"
                ) from rename_error
            raise
        moved = _classify_no_replace_move(
            source_parent=parent,
            source_name=temporary_name,
            destination_parent=parent,
            destination_name=output.name,
            expected=expected_output,
            label="post-resume prefix publication",
        )
        publication_state = "published" if moved else "temporary"
        if publication_state != "published":
            raise DevelopmentPhase2PublicationIndeterminate(
                "post-resume prefix publication returned without moving the pinned prefix"
            )
        os.fsync(parent)
        acl_guard.verify()
        _write_external_receipt_exclusive(
            receipt_output,
            receipt.canonical_file_bytes(),
        )
        receipt_published = True
        rebound = _classify_no_replace_move(
            source_parent=parent,
            source_name=temporary_name,
            destination_parent=parent,
            destination_name=output.name,
            expected=expected_output,
            label="post-resume prefix final rebind",
        )
        if not rebound:
            raise DevelopmentPhase2PublicationIndeterminate(
                "post-resume final name no longer identifies the pinned prefix"
            )
        _scan_exact_tree_acl(
            temporary_descriptor,
            expected_files=expected_files,
            label="published post-resume bootstrap prefix",
            private=True,
            read_only=False,
            enforce_view_boundary=False,
            exact_private_modes=True,
            expected_directory_mode=0o700,
            expected_file_mode=0o600,
        )
        return receipt
    except BaseException as error:
        if isinstance(error, DevelopmentPhase2PublicationIndeterminate):
            raise
        if publication_state == "published" and not receipt_published:
            publication_state = "indeterminate"
            try:
                _rename_exclusive_at(parent, output.name, temporary_name or "")
            except BaseException:
                rolled_back = _classify_no_replace_move(
                    source_parent=parent,
                    source_name=output.name,
                    destination_parent=parent,
                    destination_name=temporary_name or "",
                    expected=expected_output,
                    label="post-resume prefix rollback",
                )
                publication_state = "temporary" if rolled_back else "published"
                if not rolled_back:
                    raise DevelopmentPhase2PublicationIndeterminate(
                        "post-resume bootstrap publication could not be rolled back"
                    ) from error
            else:
                rolled_back = _classify_no_replace_move(
                    source_parent=parent,
                    source_name=output.name,
                    destination_parent=parent,
                    destination_name=temporary_name or "",
                    expected=expected_output,
                    label="post-resume prefix rollback",
                )
                publication_state = "temporary" if rolled_back else "published"
                if not rolled_back:
                    raise DevelopmentPhase2PublicationIndeterminate(
                        "post-resume bootstrap publication could not be rolled back"
                    ) from error
            try:
                os.fsync(parent)
            except BaseException as rollback_error:
                publication_state = "indeterminate"
                raise DevelopmentPhase2PublicationIndeterminate(
                    "post-resume bootstrap publication could not be rolled back"
                ) from rollback_error
        raise error
    finally:
        active_error = sys.exc_info()[1]
        if (
            temporary_descriptor is not None
            and publication_state == "temporary"
            and temporary_name is not None
        ):
            try:
                if _entry_stat(parent, temporary_name) is not None:
                    _remove_temporary_tree(
                        parent,
                        temporary_name,
                        output.name,
                        temporary_descriptor,
                        require_output_absent=True,
                    )
            except BaseException as cleanup_error:
                if active_error is None:
                    raise DevelopmentPhase2PublicationIndeterminate(
                        "post-resume bootstrap cleanup is indeterminate"
                    ) from cleanup_error
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        try:
            leases.close_owned()
        finally:
            os.close(parent)


def _result(receipt: DevelopmentPhase2ViewReceipt) -> dict[str, object]:
    return {
        "artifact_count": len(receipt.artifacts),
        "design_seed_sha256": receipt.design_seed_sha256,
        "output_root": str(receipt.output_root),
        "receipt_sha256": receipt.artifact_sha256,
        "schema_version": PHASE2_CLI_SCHEMA,
        "staged_inventory_sha256": receipt.staged_inventory_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="development-phase2-view",
        description="Build or verify the fixed fit/calibration phase-two view.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", allow_abbrev=False)
    build.add_argument("--source-root", required=True)
    build.add_argument("--staged-inventory-sha256", required=True)
    build.add_argument("--partition-audit", required=True)
    build.add_argument("--partition-audit-file-sha256", required=True)
    build.add_argument("--phase1-view-root", required=True)
    build.add_argument("--phase1-view-receipt-sha256", required=True)
    build.add_argument("--selection-receipt", required=True)
    build.add_argument("--selection-receipt-sha256", required=True)
    build.add_argument("--seed-commitment", required=True)
    build.add_argument("--seed-commitment-sha256", required=True)
    build.add_argument("--seed-reveal", required=True)
    build.add_argument("--seed-reveal-sha256", required=True)
    build.add_argument("--output-root", required=True)
    verify = commands.add_parser("verify", allow_abbrev=False)
    verify.add_argument("--root", required=True)
    verify.add_argument("--receipt-sha256", required=True)
    bootstrap = commands.add_parser("bootstrap-post-resume", allow_abbrev=False)
    bootstrap.add_argument("--root", required=True)
    bootstrap.add_argument("--receipt-sha256", required=True)
    bootstrap.add_argument("--post-config", required=True)
    bootstrap.add_argument("--post-config-sha256", required=True)
    bootstrap.add_argument("--post-output-root", required=True)
    bootstrap.add_argument("--bootstrap-receipt-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            receipt = build_development_phase2_view(
                source_root=args.source_root,
                staged_inventory_sha256=args.staged_inventory_sha256,
                partition_audit_path=args.partition_audit,
                partition_audit_file_sha256=args.partition_audit_file_sha256,
                phase1_view_root=args.phase1_view_root,
                phase1_view_receipt_sha256=args.phase1_view_receipt_sha256,
                selection_receipt_path=args.selection_receipt,
                selection_receipt_sha256=args.selection_receipt_sha256,
                seed_commitment_path=args.seed_commitment,
                seed_commitment_sha256=args.seed_commitment_sha256,
                seed_reveal_path=args.seed_reveal,
                seed_reveal_sha256=args.seed_reveal_sha256,
                output_root=args.output_root,
            )
            value: Mapping[str, object] = {"command": args.command, **_result(receipt)}
        elif args.command == "verify":
            receipt = verify_development_phase2_view(
                args.root,
                expected_receipt_sha256=args.receipt_sha256,
            )
            value = {"command": args.command, **_result(receipt)}
        else:
            receipt = bootstrap_post_embedding_resume(
                phase2_root=args.root,
                phase2_receipt_sha256=args.receipt_sha256,
                post_config_path=args.post_config,
                post_config_sha256=args.post_config_sha256,
                post_output_root=args.post_output_root,
                bootstrap_receipt_output=args.bootstrap_receipt_output,
            )
            value = {
                "bootstrap_receipt_sha256": receipt.artifact_sha256,
                "command": args.command,
                "post_output_root": str(receipt.post_output_root),
                "schema_version": PHASE2_CLI_SCHEMA,
            }
        sys.stdout.buffer.write(_canonical_bytes(dict(value)))
        return 0
    except (
        DevelopmentPhase2ViewError,
        DevelopmentStagingViewError,
        OSError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
