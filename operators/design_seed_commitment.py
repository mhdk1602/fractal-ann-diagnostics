#!/usr/bin/env python3
"""Commit a development scope before deriving its design seed from Quicknet.

This is deliberately a host-side operator.  It imports the Quicknet and
Sigstore parsers frozen at source commit P, but it is absent from the
confirmatory image build context.  No command accepts a design seed or a drand
round: both are derived from authenticated prior material.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import ssl
import stat
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import truststore

from fractal_ann_diagnostics.drand_beacon import (
    QUICKNET_CHAIN_HASH,
    QUICKNET_GENESIS_UNIX_SECONDS,
    QUICKNET_NETWORK,
    QUICKNET_PERIOD_SECONDS,
    QUICKNET_PUBLIC_KEY,
    QUICKNET_SCHEME_ID,
    QuicknetExecutionBeaconVerifier,
)
from fractal_ann_diagnostics.execution_claim import ExecutionBeaconContract
from fractal_ann_diagnostics.github_state_attestation import parse_sigstore_bundle

REQUEST_SCHEMA = "fractal-design-seed-commitment-request-v1"
COMMITMENT_SCHEMA = "fractal-design-seed-commitment-v1"
ATTESTATION_ADMISSION_SCHEMA = "fractal-design-seed-attestation-admission-v1"
REVEAL_SCHEMA = "fractal-design-seed-reveal-v1"
ATTESTATION_PREDICATE_SCHEMA = "fractal-design-seed-attestation-predicate-v1"
LOCAL_ATTEMPT_SCHEMA = "fractal-design-seed-local-attempt-v1"
ATTESTATION_PREDICATE_TYPE = (
    "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/design-seed-commitment/v1"
)

PROTOCOL_ID = "fractal-ann-diagnostics"
PROTOCOL_VERSION = "0.3.0"
PURPOSE = "post-embedding-development-design-seed"
OWNER_LOGIN = "mhdk1602"
REPOSITORY = "mhdk1602/fractal-ann-diagnostics"
OIDC_ISSUER = "https://token.actions.githubusercontent.com"
EVENT = "workflow_dispatch"
ATTESTATION_WORKFLOW = ".github/workflows/design-seed-commitment.yml"
ATTESTATION_GIT_REF = "refs/tags/design-seed-apparatus-v1"
RELEASE_AUTHOR = "github-actions[bot]"
SOURCE_P = "9061f09777b1af2346eebe3fb1ae21e6325cdf75"
SOURCE_TREE = "33e7aa05527042bdba301310c62eb3dbaffde941"
MINIMUM_PRE_ROUND_LEAD_SECONDS = 900
SCOPE_DERIVATION = "sha256-fractal-design-seed-scope-v1-lp-u64be"
TARGET_ROUND_DERIVATION = "first-quicknet-round-at-or-after-rekor-time-plus-900-seconds-v1"
SEED_DERIVATION = "sha256-fractal-design-seed-v1-lp-u64be"

_SCOPE_DOMAIN = b"fractal-ann-diagnostics/v0.3/design-seed-scope/v1"
_SEED_DOMAIN = b"fractal-ann-diagnostics/v0.3/design-seed/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_WORKFLOW = re.compile(r"^\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml$")
_GIT_REF = re.compile(r"^refs/(?:heads|tags)/[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")
_MAX_CONTROL_BYTES = 2 * 1024 * 1024
_MAX_BUNDLE_BYTES = 1024 * 1024
_MAX_GH_OUTPUT_BYTES = 1024 * 1024
_MAX_GITHUB_API_BYTES = 4 * 1024 * 1024
_GITHUB_API_VERSION = "2026-03-10"
_REMOTE_EVIDENCE_NAMES = frozenset({"actions_run", "release", "release_tag"})
_FILE_STABILITY_FIELDS = (
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


class DesignSeedCommitmentError(RuntimeError):
    """Raised when a scope, attestation, round, or reveal is not closed."""


AttestationVerifier = Callable[..., None]
RemoteAdmissionVerifier = Callable[..., None]


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise DesignSeedCommitmentError("control must be finite canonical JSON") from exc


def _strict_json(encoded: bytes, *, label: str) -> object:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise DesignSeedCommitmentError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def nonfinite(token: str) -> None:
        raise DesignSeedCommitmentError(f"{label} contains non-finite number {token!r}")

    try:
        return json.loads(
            encoded.decode("ascii", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except DesignSeedCommitmentError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DesignSeedCommitmentError(f"{label} must be strict ASCII JSON") from exc


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise DesignSeedCommitmentError(f"{label} must be one JSON object")
    observed = set(value)
    if observed != fields:
        raise DesignSeedCommitmentError(
            f"{label} schema differs; missing={sorted(fields - observed)!r}, "
            f"unknown={sorted(observed - fields)!r}"
        )
    return value


def _digest(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DesignSeedCommitmentError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _positive(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise DesignSeedCommitmentError(f"{name} must be a positive integer")
    return value


def _exact(name: str, value: object, expected: object) -> None:
    if type(value) is not type(expected) or value != expected:
        raise DesignSeedCommitmentError(f"{name} differs from the frozen contract")


def _canonical_absolute_path(name: str, value: object) -> Path:
    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise DesignSeedCommitmentError(f"{name} must be one canonical absolute path")
    path = Path(value)
    if (
        not path.is_absolute()
        or path.anchor != "/"
        or str(path) != value
        or value == "/"
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise DesignSeedCommitmentError(f"{name} must be one canonical absolute POSIX path")
    return path


def _validate_attestation_identity(
    *, workflow: object, workflow_sha: object, git_ref: object, workflow_ref: object
) -> None:
    if type(workflow) is not str or _WORKFLOW.fullmatch(workflow) is None:
        raise DesignSeedCommitmentError("attestation workflow must be one canonical workflow path")
    if type(workflow_sha) is not str or _GIT_SHA.fullmatch(workflow_sha) is None:
        raise DesignSeedCommitmentError("attestation workflow SHA must be one full Git commit")
    if type(git_ref) is not str or _GIT_REF.fullmatch(git_ref) is None:
        raise DesignSeedCommitmentError("attestation Git ref must be one canonical branch or tag")
    _exact("attestation workflow", workflow, ATTESTATION_WORKFLOW)
    _exact("attestation Git ref", git_ref, ATTESTATION_GIT_REF)
    expected_ref = f"{REPOSITORY}/{workflow}@{git_ref}"
    _exact("attestation workflow ref", workflow_ref, expected_ref)


def _lp_sha256(domain: bytes, values: Sequence[tuple[str, bytes]]) -> str:
    hasher = hashlib.sha256()
    components = (("domain", domain), *values)
    for name, value in components:
        encoded_name = name.encode("ascii", errors="strict")
        hasher.update(struct.pack(">Q", len(encoded_name)))
        hasher.update(encoded_name)
        hasher.update(struct.pack(">Q", len(value)))
        hasher.update(value)
    return hasher.hexdigest()


def _derive_scope(
    *,
    staged_inventory_sha256: str,
    partition_audit_file_sha256: str,
    phase1_view_receipt_sha256: str,
    selection_receipt_sha256: str,
) -> str:
    return _lp_sha256(
        _SCOPE_DOMAIN,
        (
            ("staged_inventory_sha256", bytes.fromhex(staged_inventory_sha256)),
            ("partition_audit_file_sha256", bytes.fromhex(partition_audit_file_sha256)),
            ("phase1_view_receipt_sha256", bytes.fromhex(phase1_view_receipt_sha256)),
            ("selection_receipt_sha256", bytes.fromhex(selection_receipt_sha256)),
        ),
    )


def _canonical_file_bytes(value: Mapping[str, object]) -> bytes:
    return _canonical_json(value) + b"\n"


def _parse_canonical_file(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    if not encoded or len(encoded) > _MAX_CONTROL_BYTES:
        raise DesignSeedCommitmentError(f"{label} is empty or exceeds its byte bound")
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise DesignSeedCommitmentError(f"{label} must end with exactly one newline")
    value = _strict_json(encoded, label=label)
    if not isinstance(value, Mapping):
        raise DesignSeedCommitmentError(f"{label} must be one JSON object")
    return value


def _stable_stat(value: os.stat_result) -> tuple[object, ...]:
    return tuple(getattr(value, name) for name in _FILE_STABILITY_FIELDS)


def _read_control(path: str | Path, *, label: str, max_bytes: int = _MAX_CONTROL_BYTES) -> bytes:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = candidate.resolve(strict=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise DesignSeedCommitmentError(f"cannot open {label}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise DesignSeedCommitmentError(f"{label} must be one unlinked regular-file identity")
        if before.st_mode & 0o022:
            raise DesignSeedCommitmentError(f"{label} cannot be group/other writable")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise DesignSeedCommitmentError(f"{label} is empty or exceeds its byte bound")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise DesignSeedCommitmentError(f"{label} ended before its admitted size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise DesignSeedCommitmentError(f"{label} grew while it was read")
        after = os.fstat(descriptor)
        if _stable_stat(before) != _stable_stat(after):
            raise DesignSeedCommitmentError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _safe_output_directory(path: str | Path) -> Path:
    directory = Path(path)
    if not directory.is_absolute():
        directory = directory.resolve(strict=False)
    try:
        observed = directory.stat(follow_symlinks=False)
    except OSError as exc:
        raise DesignSeedCommitmentError(f"cannot inspect output directory: {exc}") from exc
    if not stat.S_ISDIR(observed.st_mode) or observed.st_mode & 0o077:
        raise DesignSeedCommitmentError("output directory must exist as a private directory")
    if observed.st_uid != os.getuid():
        raise DesignSeedCommitmentError("output directory must be owned by the current user")
    return directory


def _write_exclusive(path: Path, encoded: bytes, *, mode: int = 0o400) -> None:
    """Publish complete bytes atomically without ever exposing a partial target."""

    directory = _safe_output_directory(path.parent)
    target = directory / path.name
    temporary_descriptor: int | None = None
    temporary_name: str | None = None
    try:
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.staging-",
            dir=directory,
        )
        os.fchmod(temporary_descriptor, 0o600)
        position = 0
        while position < len(encoded):
            position += os.write(temporary_descriptor, encoded[position:])
        os.fsync(temporary_descriptor)
        os.fchmod(temporary_descriptor, mode)
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = None
        try:
            os.link(temporary_name, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise DesignSeedCommitmentError(
                f"refusing to replace existing output {target}"
            ) from exc
        except OSError as exc:
            raise DesignSeedCommitmentError(f"cannot publish output {target}: {exc}") from exc
        directory_descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
            os.unlink(temporary_name)
            temporary_name = None
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def _git_output(root: Path, arguments: Sequence[str]) -> bytes:
    environment = {
        "GIT_CONFIG_COUNT": "0",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/nonexistent",
        "PATH": "/usr/bin:/bin",
    }
    try:
        completed = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DesignSeedCommitmentError("cannot verify exact source P") from exc
    if completed.returncode != 0:
        raise DesignSeedCommitmentError("exact source P verification failed")
    return completed.stdout


def _verify_exact_p_source() -> None:
    """Prove the imported package tree is byte-identical to registered source P."""

    root = Path(__file__).resolve().parents[1]
    expected_modules = {
        Path(root / "src/fractal_ann_diagnostics/drand_beacon.py").resolve(),
        Path(root / "src/fractal_ann_diagnostics/execution_claim.py").resolve(),
        Path(root / "src/fractal_ann_diagnostics/github_state_attestation.py").resolve(),
    }
    observed_modules = {
        Path(sys.modules[name].__file__).resolve()
        for name in (
            QuicknetExecutionBeaconVerifier.__module__,
            ExecutionBeaconContract.__module__,
            parse_sigstore_bundle.__module__,
        )
        if getattr(sys.modules.get(name), "__file__", None) is not None
    }
    if observed_modules != expected_modules:
        raise DesignSeedCommitmentError("exact source P modules were imported from another tree")
    tree = _git_output(root, ["rev-parse", f"{SOURCE_P}^{{tree}}"])
    if tree != f"{SOURCE_TREE}\n".encode("ascii"):
        raise DesignSeedCommitmentError("registered source P tree differs")
    _git_output(
        root,
        [
            "diff",
            "--no-ext-diff",
            "--quiet",
            SOURCE_P,
            "--",
            "src/fractal_ann_diagnostics",
        ],
    )
    untracked = _git_output(
        root,
        [
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src/fractal_ann_diagnostics",
        ],
    )
    if untracked:
        raise DesignSeedCommitmentError("exact source P package tree contains untracked files")


@dataclass(frozen=True)
class _ScopeBinding:
    staged_inventory_sha256: str
    partition_audit_file_sha256: str
    phase1_view_receipt_sha256: str
    selection_receipt_sha256: str
    scope_sha256: str
    source_p: str
    source_tree: str

    def _validate_scope(self) -> None:
        for name in (
            "staged_inventory_sha256",
            "partition_audit_file_sha256",
            "phase1_view_receipt_sha256",
            "selection_receipt_sha256",
            "scope_sha256",
        ):
            _sha256(name, getattr(self, name))
        expected = _derive_scope(
            staged_inventory_sha256=self.staged_inventory_sha256,
            partition_audit_file_sha256=self.partition_audit_file_sha256,
            phase1_view_receipt_sha256=self.phase1_view_receipt_sha256,
            selection_receipt_sha256=self.selection_receipt_sha256,
        )
        if self.scope_sha256 != expected:
            raise DesignSeedCommitmentError("scope_sha256 differs from the four pinned inputs")
        _exact("source_p", self.source_p, SOURCE_P)
        _exact("source_tree", self.source_tree, SOURCE_TREE)

    def _scope_dict(self) -> dict[str, str]:
        return {
            "partition_audit_file_sha256": self.partition_audit_file_sha256,
            "phase1_view_receipt_sha256": self.phase1_view_receipt_sha256,
            "scope_sha256": self.scope_sha256,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "source_p": self.source_p,
            "source_tree": self.source_tree,
            "staged_inventory_sha256": self.staged_inventory_sha256,
        }


@dataclass(frozen=True)
class DesignSeedCommitmentRequest(_ScopeBinding):
    attestation_workflow: str
    attestation_workflow_sha: str
    attestation_git_ref: str
    attestation_workflow_ref: str
    protocol_id: str = PROTOCOL_ID
    protocol_version: str = PROTOCOL_VERSION
    purpose: str = PURPOSE
    scope_derivation: str = SCOPE_DERIVATION
    schema_version: str = REQUEST_SCHEMA

    def __post_init__(self) -> None:
        self._validate_scope()
        _validate_attestation_identity(
            workflow=self.attestation_workflow,
            workflow_sha=self.attestation_workflow_sha,
            git_ref=self.attestation_git_ref,
            workflow_ref=self.attestation_workflow_ref,
        )
        for name, expected in (
            ("protocol_id", PROTOCOL_ID),
            ("protocol_version", PROTOCOL_VERSION),
            ("purpose", PURPOSE),
            ("scope_derivation", SCOPE_DERIVATION),
            ("schema_version", REQUEST_SCHEMA),
        ):
            _exact(name, getattr(self, name), expected)

    def to_dict(self) -> dict[str, object]:
        return {
            **self._scope_dict(),
            "attestation_git_ref": self.attestation_git_ref,
            "attestation_workflow": self.attestation_workflow,
            "attestation_workflow_ref": self.attestation_workflow_ref,
            "attestation_workflow_sha": self.attestation_workflow_sha,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "purpose": self.purpose,
            "schema_version": self.schema_version,
            "scope_derivation": self.scope_derivation,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_file_bytes(self.to_dict())

    @property
    def request_sha256(self) -> str:
        return _digest(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DesignSeedCommitmentRequest:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="design-seed request")
        return cls(**row)


@dataclass(frozen=True)
class DesignSeedCommitment(_ScopeBinding):
    request_sha256: str
    attestation_subject_name: str
    attestation_workflow: str
    attestation_workflow_sha: str
    attestation_git_ref: str
    attestation_workflow_ref: str
    quicknet_network: str = QUICKNET_NETWORK
    quicknet_chain_hash: str = QUICKNET_CHAIN_HASH
    quicknet_scheme_id: str = QUICKNET_SCHEME_ID
    quicknet_public_key: str = QUICKNET_PUBLIC_KEY
    quicknet_genesis_unix_seconds: int = QUICKNET_GENESIS_UNIX_SECONDS
    quicknet_period_seconds: int = QUICKNET_PERIOD_SECONDS
    minimum_pre_round_lead_seconds: int = MINIMUM_PRE_ROUND_LEAD_SECONDS
    target_round_derivation: str = TARGET_ROUND_DERIVATION
    seed_derivation: str = SEED_DERIVATION
    schema_version: str = COMMITMENT_SCHEMA

    def __post_init__(self) -> None:
        self._validate_scope()
        _sha256("request_sha256", self.request_sha256)
        _validate_attestation_identity(
            workflow=self.attestation_workflow,
            workflow_sha=self.attestation_workflow_sha,
            git_ref=self.attestation_git_ref,
            workflow_ref=self.attestation_workflow_ref,
        )
        expected_request = DesignSeedCommitmentRequest(
            staged_inventory_sha256=self.staged_inventory_sha256,
            partition_audit_file_sha256=self.partition_audit_file_sha256,
            phase1_view_receipt_sha256=self.phase1_view_receipt_sha256,
            selection_receipt_sha256=self.selection_receipt_sha256,
            scope_sha256=self.scope_sha256,
            source_p=self.source_p,
            source_tree=self.source_tree,
            attestation_workflow=self.attestation_workflow,
            attestation_workflow_sha=self.attestation_workflow_sha,
            attestation_git_ref=self.attestation_git_ref,
            attestation_workflow_ref=self.attestation_workflow_ref,
        )
        if self.request_sha256 != expected_request.request_sha256:
            raise DesignSeedCommitmentError("request_sha256 differs from the closed request")
        expected_name = f"design-seed-commitment-{self.scope_sha256}.json"
        _exact("attestation_subject_name", self.attestation_subject_name, expected_name)
        for name, expected in (
            ("quicknet_network", QUICKNET_NETWORK),
            ("quicknet_chain_hash", QUICKNET_CHAIN_HASH),
            ("quicknet_scheme_id", QUICKNET_SCHEME_ID),
            ("quicknet_public_key", QUICKNET_PUBLIC_KEY),
            ("quicknet_genesis_unix_seconds", QUICKNET_GENESIS_UNIX_SECONDS),
            ("quicknet_period_seconds", QUICKNET_PERIOD_SECONDS),
            ("minimum_pre_round_lead_seconds", MINIMUM_PRE_ROUND_LEAD_SECONDS),
            ("target_round_derivation", TARGET_ROUND_DERIVATION),
            ("seed_derivation", SEED_DERIVATION),
            ("schema_version", COMMITMENT_SCHEMA),
        ):
            _exact(name, getattr(self, name), expected)

    def to_dict(self) -> dict[str, object]:
        return {
            **self._scope_dict(),
            "attestation_git_ref": self.attestation_git_ref,
            "attestation_subject_name": self.attestation_subject_name,
            "attestation_workflow": self.attestation_workflow,
            "attestation_workflow_ref": self.attestation_workflow_ref,
            "attestation_workflow_sha": self.attestation_workflow_sha,
            "minimum_pre_round_lead_seconds": self.minimum_pre_round_lead_seconds,
            "quicknet_chain_hash": self.quicknet_chain_hash,
            "quicknet_genesis_unix_seconds": self.quicknet_genesis_unix_seconds,
            "quicknet_network": self.quicknet_network,
            "quicknet_period_seconds": self.quicknet_period_seconds,
            "quicknet_public_key": self.quicknet_public_key,
            "quicknet_scheme_id": self.quicknet_scheme_id,
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "seed_derivation": self.seed_derivation,
            "target_round_derivation": self.target_round_derivation,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_file_bytes(self.to_dict())

    @property
    def commitment_sha256(self) -> str:
        return _digest(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DesignSeedCommitment:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="design-seed commitment")
        return cls(**row)


@dataclass(frozen=True)
class DesignSeedAttestationAdmission(_ScopeBinding):
    commitment_sha256: str
    attestation_subject_name: str
    attestation_bundle_base64: str
    attestation_bundle_sha256: str
    predicate_sha256: str
    repository: str
    workflow: str
    workflow_ref: str
    workflow_sha: str
    git_ref: str
    run_id: int
    run_attempt: int
    event: str
    actor: str
    triggering_actor: str
    release_id: int
    release_tag: str
    release_name: str
    release_published_at_utc: str
    actions_run_api_projection_base64: str
    actions_run_api_projection_sha256: str
    release_api_projection_base64: str
    release_api_projection_sha256: str
    release_tag_api_projection_base64: str
    release_tag_api_projection_sha256: str
    rekor_log_key_sha256: str
    rekor_log_index: int
    rekor_entry_id: str
    rekor_integrated_time_unix_seconds: int
    rekor_integrated_at_utc: str
    rekor_timestamp_token_sha256: str
    target_round: int
    target_publication_unix_seconds: int
    pre_round_lead_seconds: int
    predicate_type: str = ATTESTATION_PREDICATE_TYPE
    schema_version: str = ATTESTATION_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        self._validate_scope()
        for name in (
            "commitment_sha256",
            "attestation_bundle_sha256",
            "predicate_sha256",
            "actions_run_api_projection_sha256",
            "release_api_projection_sha256",
            "release_tag_api_projection_sha256",
            "rekor_log_key_sha256",
            "rekor_timestamp_token_sha256",
        ):
            _sha256(name, getattr(self, name))
        _exact("repository", self.repository, REPOSITORY)
        _exact("event", self.event, EVENT)
        _exact("actor", self.actor, OWNER_LOGIN)
        _exact("triggering_actor", self.triggering_actor, OWNER_LOGIN)
        _exact("run_attempt", self.run_attempt, 1)
        _exact("predicate_type", self.predicate_type, ATTESTATION_PREDICATE_TYPE)
        _exact("schema_version", self.schema_version, ATTESTATION_ADMISSION_SCHEMA)
        _positive("run_id", self.run_id)
        _positive("release_id", self.release_id)
        expected_release = _release_tag(self.scope_sha256)
        _exact("release_tag", self.release_tag, expected_release)
        _exact("release_name", self.release_name, expected_release)
        _utc_timestamp("release_published_at_utc", self.release_published_at_utc)
        if type(self.rekor_log_index) is not int or self.rekor_log_index < 0:
            raise DesignSeedCommitmentError("rekor_log_index must be a non-negative integer")
        _positive("rekor_integrated_time_unix_seconds", self.rekor_integrated_time_unix_seconds)
        _positive("target_round", self.target_round)
        _positive("target_publication_unix_seconds", self.target_publication_unix_seconds)
        _positive("pre_round_lead_seconds", self.pre_round_lead_seconds)
        if type(self.workflow) is not str or _WORKFLOW.fullmatch(self.workflow) is None:
            raise DesignSeedCommitmentError("workflow must be one canonical workflow path")
        if type(self.workflow_sha) is not str or _GIT_SHA.fullmatch(self.workflow_sha) is None:
            raise DesignSeedCommitmentError("workflow_sha must be one full lowercase Git SHA")
        if type(self.git_ref) is not str or _GIT_REF.fullmatch(self.git_ref) is None:
            raise DesignSeedCommitmentError("git_ref must be one canonical branch or tag ref")
        expected_ref = f"{REPOSITORY}/{self.workflow}@{self.git_ref}"
        _exact("workflow_ref", self.workflow_ref, expected_ref)
        expected_subject = f"design-seed-commitment-{self.scope_sha256}.json"
        _exact("attestation_subject_name", self.attestation_subject_name, expected_subject)
        try:
            bundle = base64.b64decode(
                self.attestation_bundle_base64.encode("ascii", errors="strict"), validate=True
            )
        except (UnicodeError, ValueError) as exc:
            raise DesignSeedCommitmentError("attestation bundle must be canonical base64") from exc
        if not bundle or len(bundle) > _MAX_BUNDLE_BYTES:
            raise DesignSeedCommitmentError("attestation bundle is empty or exceeds its byte bound")
        if base64.b64encode(bundle).decode("ascii") != self.attestation_bundle_base64:
            raise DesignSeedCommitmentError("attestation bundle base64 is not canonical")
        if _digest(bundle) != self.attestation_bundle_sha256:
            raise DesignSeedCommitmentError("attestation bundle digest differs")
        try:
            parsed_time = datetime.fromisoformat(self.rekor_integrated_at_utc)
        except (TypeError, ValueError) as exc:
            raise DesignSeedCommitmentError("Rekor integrated timestamp is not ISO 8601") from exc
        if parsed_time.tzinfo != timezone.utc or int(parsed_time.timestamp()) != (
            self.rekor_integrated_time_unix_seconds
        ):
            raise DesignSeedCommitmentError("Rekor integrated timestamp differs from its epoch")
        target_round, publication, lead = _derive_target_round(
            self.rekor_integrated_time_unix_seconds
        )
        if (
            self.target_round != target_round
            or self.target_publication_unix_seconds != publication
            or self.pre_round_lead_seconds != lead
        ):
            raise DesignSeedCommitmentError("target round was not mechanically derived")
        if self.pre_round_lead_seconds < MINIMUM_PRE_ROUND_LEAD_SECONDS:
            raise DesignSeedCommitmentError("target round has less than 900 seconds lead")
        projections = {
            "actions_run": _decode_projection(
                "actions_run",
                self.actions_run_api_projection_base64,
                self.actions_run_api_projection_sha256,
            ),
            "release": _decode_projection(
                "release",
                self.release_api_projection_base64,
                self.release_api_projection_sha256,
            ),
            "release_tag": _decode_projection(
                "release_tag",
                self.release_tag_api_projection_base64,
                self.release_tag_api_projection_sha256,
            ),
        }
        run_projection = _parse_canonical_file(
            projections["actions_run"], label="actions_run API projection"
        )
        release_projection = _parse_canonical_file(
            projections["release"], label="release API projection"
        )
        tag_projection = _parse_canonical_file(
            projections["release_tag"], label="release_tag API projection"
        )
        retained = (
            ("run id", run_projection.get("id"), self.run_id),
            ("run attempt", run_projection.get("run_attempt"), 1),
            ("run event", run_projection.get("event"), EVENT),
            ("run actor", run_projection.get("actor"), OWNER_LOGIN),
            ("run triggering actor", run_projection.get("triggering_actor"), OWNER_LOGIN),
            ("run workflow", run_projection.get("path"), self.workflow),
            ("run head SHA", run_projection.get("head_sha"), self.workflow_sha),
            ("release id", release_projection.get("id"), self.release_id),
            ("release tag", release_projection.get("tag_name"), self.release_tag),
            ("release name", release_projection.get("name"), self.release_name),
            (
                "release publication",
                release_projection.get("published_at"),
                self.release_published_at_utc.replace("+00:00", "Z"),
            ),
            ("release immutable flag", release_projection.get("immutable"), True),
            ("release author", release_projection.get("author"), RELEASE_AUTHOR),
            ("release asset count", release_projection.get("assets_count"), 0),
            ("release tag ref", tag_projection.get("ref"), f"refs/tags/{self.release_tag}"),
            ("release tag type", tag_projection.get("object_type"), "commit"),
            ("release tag SHA", tag_projection.get("object_sha"), self.workflow_sha),
        )
        for name, observed, expected in retained:
            _exact(f"retained GitHub {name}", observed, expected)

    @property
    def bundle_bytes(self) -> bytes:
        return base64.b64decode(self.attestation_bundle_base64.encode("ascii"), validate=True)

    def to_dict(self) -> dict[str, object]:
        return {
            **self._scope_dict(),
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "staged_inventory_sha256",
                    "partition_audit_file_sha256",
                    "phase1_view_receipt_sha256",
                    "selection_receipt_sha256",
                    "scope_sha256",
                    "source_p",
                    "source_tree",
                }
            },
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_file_bytes(self.to_dict())

    @property
    def admission_sha256(self) -> str:
        return _digest(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DesignSeedAttestationAdmission:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="design-seed attestation admission",
        )
        return cls(**row)


@dataclass(frozen=True)
class DesignSeedReveal(_ScopeBinding):
    commitment_sha256: str
    attestation_admission_path: str
    attestation_admission_sha256: str
    target_round: int
    quicknet_beacon_base64: str
    quicknet_beacon_sha256: str
    quicknet_randomness: str
    quicknet_signature: str
    design_seed_sha256: str
    seed_derivation: str = SEED_DERIVATION
    schema_version: str = REVEAL_SCHEMA

    def __post_init__(self) -> None:
        self._validate_scope()
        for name in (
            "commitment_sha256",
            "attestation_admission_sha256",
            "quicknet_beacon_sha256",
            "quicknet_randomness",
            "design_seed_sha256",
        ):
            _sha256(name, getattr(self, name))
        if (
            type(self.quicknet_signature) is not str
            or re.fullmatch(r"[0-9a-f]{96}", self.quicknet_signature) is None
        ):
            raise DesignSeedCommitmentError("quicknet_signature must be 48 lowercase hex bytes")
        _canonical_absolute_path("attestation_admission_path", self.attestation_admission_path)
        _positive("target_round", self.target_round)
        _exact("seed_derivation", self.seed_derivation, SEED_DERIVATION)
        _exact("schema_version", self.schema_version, REVEAL_SCHEMA)
        try:
            beacon = base64.b64decode(
                self.quicknet_beacon_base64.encode("ascii", errors="strict"), validate=True
            )
        except (UnicodeError, ValueError) as exc:
            raise DesignSeedCommitmentError("Quicknet beacon must be canonical base64") from exc
        if not beacon or len(beacon) > 2 * 1024:
            raise DesignSeedCommitmentError("Quicknet beacon is empty or exceeds its byte bound")
        if base64.b64encode(beacon).decode("ascii") != self.quicknet_beacon_base64:
            raise DesignSeedCommitmentError("Quicknet beacon base64 is not canonical")
        if _digest(beacon) != self.quicknet_beacon_sha256:
            raise DesignSeedCommitmentError("Quicknet beacon digest differs")

    @property
    def beacon_bytes(self) -> bytes:
        return base64.b64decode(self.quicknet_beacon_base64.encode("ascii"), validate=True)

    def to_dict(self) -> dict[str, object]:
        return {
            **self._scope_dict(),
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "staged_inventory_sha256",
                    "partition_audit_file_sha256",
                    "phase1_view_receipt_sha256",
                    "selection_receipt_sha256",
                    "scope_sha256",
                    "source_p",
                    "source_tree",
                }
            },
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_file_bytes(self.to_dict())

    @property
    def reveal_sha256(self) -> str:
        return _digest(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DesignSeedReveal:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="design-seed reveal")
        return cls(**row)


def _derive_target_round(integrated_seconds: int) -> tuple[int, int, int]:
    integrated = _positive("Rekor integrated time", integrated_seconds)
    deadline = integrated + MINIMUM_PRE_ROUND_LEAD_SECONDS
    delta = deadline - QUICKNET_GENESIS_UNIX_SECONDS
    if delta <= 0:
        round_number = 1
    else:
        round_number = (delta + QUICKNET_PERIOD_SECONDS - 1) // QUICKNET_PERIOD_SECONDS + 1
    publication = QUICKNET_GENESIS_UNIX_SECONDS + (round_number - 1) * QUICKNET_PERIOD_SECONDS
    lead = publication - integrated
    if lead < MINIMUM_PRE_ROUND_LEAD_SECONDS:
        raise DesignSeedCommitmentError("internal target-round derivation violated its lead")
    return round_number, publication, lead


def _release_tag(scope_sha256: str) -> str:
    return f"design-seed-scope-{_sha256('scope SHA-256', scope_sha256)}"


def _ref_name(git_ref: str) -> str:
    for prefix in ("refs/heads/", "refs/tags/"):
        if git_ref.startswith(prefix):
            return git_ref.removeprefix(prefix)
    raise DesignSeedCommitmentError("attestation Git ref kind is not supported")


def _utc_timestamp(name: str, value: object) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise DesignSeedCommitmentError(f"{name} must be one UTC GitHub timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise DesignSeedCommitmentError(f"{name} must be one UTC GitHub timestamp") from exc
    if parsed.tzinfo != timezone.utc or parsed.isoformat().replace("+00:00", "Z") != value:
        raise DesignSeedCommitmentError(f"{name} must be one canonical UTC GitHub timestamp")
    return parsed


def _projection_bytes(value: Mapping[str, object]) -> bytes:
    return _canonical_file_bytes(value)


def _encoded_projection(name: str, encoded: bytes) -> tuple[str, str]:
    if not encoded or len(encoded) > _MAX_GITHUB_API_BYTES:
        raise DesignSeedCommitmentError(f"{name} API projection is empty or too large")
    return base64.b64encode(encoded).decode("ascii"), _digest(encoded)


def _decode_projection(name: str, encoded_base64: object, expected_sha256: object) -> bytes:
    _sha256(f"{name} API projection SHA-256", expected_sha256)
    if type(encoded_base64) is not str:
        raise DesignSeedCommitmentError(f"{name} API projection must be canonical base64")
    try:
        encoded = base64.b64decode(encoded_base64.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as exc:
        raise DesignSeedCommitmentError(f"{name} API projection must be canonical base64") from exc
    if base64.b64encode(encoded).decode("ascii") != encoded_base64:
        raise DesignSeedCommitmentError(f"{name} API projection base64 is not canonical")
    if not encoded or len(encoded) > _MAX_GITHUB_API_BYTES or _digest(encoded) != expected_sha256:
        raise DesignSeedCommitmentError(f"{name} API projection digest differs")
    value = _parse_canonical_file(encoded, label=f"{name} API projection")
    if encoded != _canonical_file_bytes(value):
        raise DesignSeedCommitmentError(f"{name} API projection is not canonical")
    return encoded


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _read_github_api(path: str) -> Mapping[str, Any]:
    if not path.startswith("/") or "?" in path or "#" in path:
        raise DesignSeedCommitmentError("GitHub API path is not closed")
    url = f"https://api.github.com{path}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Accept-Encoding": "identity",
            "User-Agent": "fractal-ann-diagnostics-design-seed-v1",
            "X-GitHub-Api-Version": _GITHUB_API_VERSION,
        },
        method="GET",
    )
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    opener = urllib.request.build_opener(
        _NoRedirect(), urllib.request.HTTPSHandler(context=context)
    )
    try:
        with opener.open(request, timeout=30) as response:
            if response.status != 200 or response.geturl() != url:
                raise DesignSeedCommitmentError("GitHub API response endpoint differs")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > _MAX_GITHUB_API_BYTES:
                raise DesignSeedCommitmentError("GitHub API response exceeds its byte bound")
            encoded = response.read(_MAX_GITHUB_API_BYTES + 1)
    except DesignSeedCommitmentError:
        raise
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise DesignSeedCommitmentError("public GitHub API admission failed") from exc
    if not encoded or len(encoded) > _MAX_GITHUB_API_BYTES:
        raise DesignSeedCommitmentError("GitHub API response is empty or exceeds its byte bound")
    value = _strict_json(encoded, label="GitHub API response")
    if not isinstance(value, Mapping):
        raise DesignSeedCommitmentError("GitHub API response must be one JSON object")
    return value


def _nested_login(value: Mapping[str, Any], name: str) -> object:
    row = value.get(name)
    if not isinstance(row, Mapping):
        raise DesignSeedCommitmentError(f"GitHub API {name} identity is absent")
    return row.get("login")


def _nested_repository(value: Mapping[str, Any], name: str) -> object:
    row = value.get(name)
    if not isinstance(row, Mapping):
        raise DesignSeedCommitmentError(f"GitHub API {name} identity is absent")
    return row.get("full_name")


def _default_remote_admission_verifier(
    *,
    commitment: DesignSeedCommitment,
    predicate: Mapping[str, Any],
    rekor_integrated_at_utc: str,
) -> Mapping[str, bytes]:
    run_id = _positive("attestation predicate run_id", predicate["run_id"])
    release_id = _positive("attestation predicate release_id", predicate["release_id"])
    run = _read_github_api(f"/repos/{REPOSITORY}/actions/runs/{run_id}/attempts/1")
    expected_head_branch = _ref_name(commitment.attestation_git_ref)
    run_projection: dict[str, object] = {
        "actor": _nested_login(run, "actor"),
        "conclusion": run.get("conclusion"),
        "event": run.get("event"),
        "head_branch": run.get("head_branch"),
        "head_repository": _nested_repository(run, "head_repository"),
        "head_sha": run.get("head_sha"),
        "id": run.get("id"),
        "path": run.get("path"),
        "repository": _nested_repository(run, "repository"),
        "run_attempt": run.get("run_attempt"),
        "run_started_at": run.get("run_started_at"),
        "status": run.get("status"),
        "triggering_actor": _nested_login(run, "triggering_actor"),
    }
    expected_run = {
        "actor": OWNER_LOGIN,
        "conclusion": "success",
        "event": EVENT,
        "head_branch": expected_head_branch,
        "head_repository": REPOSITORY,
        "head_sha": commitment.attestation_workflow_sha,
        "id": run_id,
        "path": commitment.attestation_workflow,
        "repository": REPOSITORY,
        "run_attempt": 1,
        "status": "completed",
        "triggering_actor": OWNER_LOGIN,
    }
    for name, expected in expected_run.items():
        _exact(f"GitHub Actions run {name}", run_projection[name], expected)
    run_started = _utc_timestamp("GitHub Actions run_started_at", run_projection["run_started_at"])

    release = _read_github_api(f"/repos/{REPOSITORY}/releases/{release_id}")
    release_projection: dict[str, object] = {
        "assets_count": len(release.get("assets", []))
        if isinstance(release.get("assets"), list)
        else None,
        "author": _nested_login(release, "author"),
        "draft": release.get("draft"),
        "id": release.get("id"),
        "immutable": release.get("immutable"),
        "name": release.get("name"),
        "prerelease": release.get("prerelease"),
        "published_at": release.get("published_at"),
        "tag_name": release.get("tag_name"),
        "target_commitish": release.get("target_commitish"),
    }
    expected_release = {
        "assets_count": 0,
        "author": RELEASE_AUTHOR,
        "draft": False,
        "id": release_id,
        "immutable": True,
        "name": _release_tag(commitment.scope_sha256),
        "prerelease": False,
        "tag_name": _release_tag(commitment.scope_sha256),
        "target_commitish": commitment.attestation_workflow_sha,
    }
    for name, expected in expected_release.items():
        _exact(f"GitHub release {name}", release_projection[name], expected)
    published = _utc_timestamp("GitHub release published_at", release_projection["published_at"])
    integrated = datetime.fromisoformat(rekor_integrated_at_utc)
    if not run_started <= published <= integrated:
        raise DesignSeedCommitmentError(
            "immutable scope release was not published by the admitted run before Rekor"
        )
    _exact(
        "attestation predicate release_published_at_utc",
        predicate["release_published_at_utc"],
        published.isoformat().replace("+00:00", "Z"),
    )

    tag = urllib.parse.quote(_release_tag(commitment.scope_sha256), safe="")
    tag_response = _read_github_api(f"/repos/{REPOSITORY}/git/ref/tags/{tag}")
    tag_object = tag_response.get("object")
    if not isinstance(tag_object, Mapping):
        raise DesignSeedCommitmentError("GitHub release tag object is absent")
    tag_projection: dict[str, object] = {
        "object_sha": tag_object.get("sha"),
        "object_type": tag_object.get("type"),
        "ref": tag_response.get("ref"),
    }
    expected_tag = {
        "object_sha": commitment.attestation_workflow_sha,
        "object_type": "commit",
        "ref": f"refs/tags/{_release_tag(commitment.scope_sha256)}",
    }
    for name, expected in expected_tag.items():
        _exact(f"GitHub release tag {name}", tag_projection[name], expected)
    return {
        "actions_run": _projection_bytes(run_projection),
        "release": _projection_bytes(release_projection),
        "release_tag": _projection_bytes(tag_projection),
    }


def _verify_remote_admission(
    *,
    commitment: DesignSeedCommitment,
    predicate: Mapping[str, Any],
    rekor_integrated_at_utc: str,
    verifier: RemoteAdmissionVerifier | None,
) -> Mapping[str, bytes]:
    active = _default_remote_admission_verifier if verifier is None else verifier
    try:
        evidence = active(
            commitment=commitment,
            predicate=predicate,
            rekor_integrated_at_utc=rekor_integrated_at_utc,
        )
    except DesignSeedCommitmentError:
        raise
    except Exception as exc:
        raise DesignSeedCommitmentError("remote one-scope admission verification failed") from exc
    if not isinstance(evidence, Mapping) or set(evidence) != _REMOTE_EVIDENCE_NAMES:
        raise DesignSeedCommitmentError("remote admission evidence membership differs")
    result: dict[str, bytes] = {}
    for name in sorted(_REMOTE_EVIDENCE_NAMES):
        encoded = evidence[name]
        if type(encoded) is not bytes:
            raise DesignSeedCommitmentError(f"{name} API projection must be bytes")
        value = _parse_canonical_file(encoded, label=f"{name} API projection")
        if encoded != _canonical_file_bytes(value):
            raise DesignSeedCommitmentError(f"{name} API projection is not canonical")
        result[name] = encoded
    return result


def _predicate_from_statement(statement: Mapping[str, Any]) -> Mapping[str, Any]:
    expected_statement = frozenset({"_type", "predicateType", "subject", "predicate"})
    row = _closed(statement, expected_statement, label="attested in-toto statement")
    _exact("in-toto statement type", row["_type"], "https://in-toto.io/Statement/v1")
    _exact("attestation predicate type", row["predicateType"], ATTESTATION_PREDICATE_TYPE)
    return _closed(
        row["predicate"],
        frozenset(
            {
                "actor",
                "commitment_sha256",
                "event",
                "git_ref",
                "repository",
                "release_id",
                "release_name",
                "release_published_at_utc",
                "release_tag",
                "run_attempt",
                "run_id",
                "schema_version",
                "scope_sha256",
                "source_p",
                "source_tree",
                "triggering_actor",
                "workflow",
                "workflow_ref",
                "workflow_sha",
            }
        ),
        label="design-seed attestation predicate",
    )


def _validate_predicate(
    statement: Mapping[str, Any], commitment: DesignSeedCommitment
) -> Mapping[str, Any]:
    row = _predicate_from_statement(statement)
    for name, expected in (
        ("schema_version", ATTESTATION_PREDICATE_SCHEMA),
        ("repository", REPOSITORY),
        ("event", EVENT),
        ("actor", OWNER_LOGIN),
        ("triggering_actor", OWNER_LOGIN),
        ("run_attempt", 1),
        ("scope_sha256", commitment.scope_sha256),
        ("commitment_sha256", commitment.commitment_sha256),
        ("source_p", commitment.source_p),
        ("source_tree", commitment.source_tree),
        ("workflow", commitment.attestation_workflow),
        ("workflow_sha", commitment.attestation_workflow_sha),
        ("git_ref", commitment.attestation_git_ref),
        ("workflow_ref", commitment.attestation_workflow_ref),
        ("release_tag", _release_tag(commitment.scope_sha256)),
        ("release_name", _release_tag(commitment.scope_sha256)),
    ):
        _exact(f"attestation predicate {name}", row[name], expected)
    _positive("attestation predicate run_id", row["run_id"])
    _positive("attestation predicate release_id", row["release_id"])
    _utc_timestamp(
        "attestation predicate release_published_at_utc",
        row["release_published_at_utc"],
    )
    _validate_attestation_identity(
        workflow=row["workflow"],
        workflow_sha=row["workflow_sha"],
        git_ref=row["git_ref"],
        workflow_ref=row["workflow_ref"],
    )
    subjects = statement["subject"]
    if type(subjects) is not list or len(subjects) != 1:
        raise DesignSeedCommitmentError("attestation must bind exactly one subject")
    subject = _closed(
        subjects[0], frozenset({"digest", "name"}), label="design-seed attestation subject"
    )
    _exact("attestation subject name", subject["name"], commitment.attestation_subject_name)
    digest = _closed(subject["digest"], frozenset({"sha256"}), label="design-seed subject digest")
    _exact("attestation subject digest", digest["sha256"], commitment.commitment_sha256)
    return row


def _default_attestation_verifier(
    *,
    subject_path: Path,
    bundle_bytes: bytes,
    repository: str,
    workflow_ref: str,
    workflow_sha: str,
    git_ref: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="fractal-design-seed-attestation-") as directory:
        bundle_path = Path(directory) / "bundle.json"
        gh_config = Path(directory) / "gh-config"
        gh_config.mkdir(mode=0o700)
        bundle_path.write_bytes(bundle_bytes)
        os.chmod(bundle_path, 0o600)
        command = [
            "gh",
            "attestation",
            "verify",
            str(subject_path),
            "--bundle",
            str(bundle_path),
            "--hostname",
            "github.com",
            "--repo",
            repository,
            "--cert-identity",
            f"https://github.com/{workflow_ref}",
            "--cert-oidc-issuer",
            OIDC_ISSUER,
            "--signer-digest",
            workflow_sha,
            "--source-digest",
            workflow_sha,
            "--source-ref",
            git_ref,
            "--deny-self-hosted-runners",
            "--predicate-type",
            ATTESTATION_PREDICATE_TYPE,
            "--format",
            "json",
        ]
        try:
            environment = {
                name: value
                for name, value in os.environ.items()
                if name not in {"GH_ENTERPRISE_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"}
            }
            environment.update(
                {
                    "GH_CONFIG_DIR": str(gh_config),
                    "GH_PROMPT_DISABLED": "1",
                    "NO_COLOR": "1",
                }
            )
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                env=environment,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DesignSeedCommitmentError("GitHub attestation verifier failed to run") from exc
    if completed.returncode != 0:
        raise DesignSeedCommitmentError("GitHub rejected the commitment attestation")
    if len(completed.stdout) > _MAX_GH_OUTPUT_BYTES:
        raise DesignSeedCommitmentError("GitHub attestation output exceeds its byte bound")
    result = _strict_json(completed.stdout, label="GitHub attestation verification output")
    if type(result) is not list or len(result) != 1 or not isinstance(result[0], Mapping):
        raise DesignSeedCommitmentError("GitHub must verify exactly one attestation")
    if not isinstance(result[0].get("verificationResult"), Mapping):
        raise DesignSeedCommitmentError("GitHub output lacks a verified result")


def _verify_bundle(
    *,
    subject_path: Path,
    commitment: DesignSeedCommitment,
    bundle_bytes: bytes,
    verifier: AttestationVerifier | None,
) -> tuple[object, Mapping[str, Any]]:
    try:
        observation = parse_sigstore_bundle(bundle_bytes)
    except Exception as exc:
        raise DesignSeedCommitmentError("Sigstore bundle is not admissible") from exc
    predicate = _validate_predicate(observation.statement, commitment)
    active = _default_attestation_verifier if verifier is None else verifier
    try:
        active(
            subject_path=subject_path,
            bundle_bytes=bundle_bytes,
            repository=REPOSITORY,
            workflow_ref=predicate["workflow_ref"],
            workflow_sha=predicate["workflow_sha"],
            git_ref=predicate["git_ref"],
        )
    except DesignSeedCommitmentError:
        raise
    except Exception as exc:
        raise DesignSeedCommitmentError("commitment attestation verification failed") from exc
    return observation, predicate


def build_design_seed_request(
    *,
    staged_inventory_sha256: str,
    partition_audit_file_sha256: str,
    phase1_view_receipt_sha256: str,
    selection_receipt_sha256: str,
    attestation_workflow: str,
    attestation_workflow_sha: str,
    attestation_git_ref: str,
    output_directory: str | Path,
) -> tuple[Path, DesignSeedCommitmentRequest]:
    _verify_exact_p_source()
    pins = {
        name: _sha256(name, value)
        for name, value in (
            ("staged_inventory_sha256", staged_inventory_sha256),
            ("partition_audit_file_sha256", partition_audit_file_sha256),
            ("phase1_view_receipt_sha256", phase1_view_receipt_sha256),
            ("selection_receipt_sha256", selection_receipt_sha256),
        )
    }
    scope = _derive_scope(**pins)
    attestation_workflow_ref = f"{REPOSITORY}/{attestation_workflow}@{attestation_git_ref}"
    request = DesignSeedCommitmentRequest(
        **pins,
        scope_sha256=scope,
        source_p=SOURCE_P,
        source_tree=SOURCE_TREE,
        attestation_workflow=attestation_workflow,
        attestation_workflow_sha=attestation_workflow_sha,
        attestation_git_ref=attestation_git_ref,
        attestation_workflow_ref=attestation_workflow_ref,
    )
    target = _safe_output_directory(output_directory) / f"design-seed-request-{scope}.json"
    _write_exclusive(target, request.canonical_file_bytes())
    return target, request


def verify_design_seed_request(
    path: str | Path, *, expected_sha256: str | None = None
) -> DesignSeedCommitmentRequest:
    _verify_exact_p_source()
    encoded = _read_control(path, label="design-seed request")
    if expected_sha256 is not None and _digest(encoded) != _sha256(
        "expected request SHA-256", expected_sha256
    ):
        raise DesignSeedCommitmentError("design-seed request digest differs")
    request = DesignSeedCommitmentRequest.from_dict(
        _parse_canonical_file(encoded, label="design-seed request")
    )
    if encoded != request.canonical_file_bytes():
        raise DesignSeedCommitmentError("design-seed request bytes are not canonical")
    expected_name = f"design-seed-request-{request.scope_sha256}.json"
    if Path(path).name != expected_name:
        raise DesignSeedCommitmentError("design-seed request filename differs from its scope")
    return request


def build_design_seed_commitment(
    request_path: str | Path, *, output_directory: str | Path
) -> tuple[Path, DesignSeedCommitment]:
    request = verify_design_seed_request(request_path)
    commitment = DesignSeedCommitment(
        staged_inventory_sha256=request.staged_inventory_sha256,
        partition_audit_file_sha256=request.partition_audit_file_sha256,
        phase1_view_receipt_sha256=request.phase1_view_receipt_sha256,
        selection_receipt_sha256=request.selection_receipt_sha256,
        scope_sha256=request.scope_sha256,
        source_p=request.source_p,
        source_tree=request.source_tree,
        request_sha256=request.request_sha256,
        attestation_subject_name=f"design-seed-commitment-{request.scope_sha256}.json",
        attestation_workflow=request.attestation_workflow,
        attestation_workflow_sha=request.attestation_workflow_sha,
        attestation_git_ref=request.attestation_git_ref,
        attestation_workflow_ref=request.attestation_workflow_ref,
    )
    directory = _safe_output_directory(output_directory)
    marker = directory / f".design-seed-scope-{request.scope_sha256}.local-attempt.json"
    marker_bytes = _canonical_file_bytes(
        {
            "authority": "LOCAL_DEFENSE_ONLY",
            "request_sha256": request.request_sha256,
            "schema_version": LOCAL_ATTEMPT_SCHEMA,
            "scope_sha256": request.scope_sha256,
            "state": "ATTEMPTED",
        }
    )
    _write_exclusive(marker, marker_bytes)
    target = directory / commitment.attestation_subject_name
    _write_exclusive(target, commitment.canonical_file_bytes())
    return target, commitment


def verify_design_seed_commitment(
    path: str | Path, *, expected_sha256: str | None = None
) -> DesignSeedCommitment:
    _verify_exact_p_source()
    encoded = _read_control(path, label="design-seed commitment")
    observed_digest = _digest(encoded)
    if expected_sha256 is not None and observed_digest != _sha256(
        "expected commitment SHA-256", expected_sha256
    ):
        raise DesignSeedCommitmentError("design-seed commitment digest differs")
    commitment = DesignSeedCommitment.from_dict(
        _parse_canonical_file(encoded, label="design-seed commitment")
    )
    if encoded != commitment.canonical_file_bytes():
        raise DesignSeedCommitmentError("design-seed commitment bytes are not canonical")
    if Path(path).name != commitment.attestation_subject_name:
        raise DesignSeedCommitmentError("commitment filename differs from its attested subject")
    if observed_digest != commitment.commitment_sha256:
        raise DesignSeedCommitmentError("commitment digest is internally inconsistent")
    return commitment


def admit_design_seed_attestation(
    commitment_path: str | Path,
    bundle_path: str | Path,
    *,
    output_directory: str | Path,
    verifier: AttestationVerifier | None = None,
    remote_verifier: RemoteAdmissionVerifier | None = None,
) -> tuple[Path, DesignSeedAttestationAdmission]:
    subject_path = Path(commitment_path).resolve(strict=True)
    commitment = verify_design_seed_commitment(subject_path)
    bundle = _read_control(
        bundle_path, label="design-seed Sigstore bundle", max_bytes=_MAX_BUNDLE_BYTES
    )
    observation, predicate = _verify_bundle(
        subject_path=subject_path,
        commitment=commitment,
        bundle_bytes=bundle,
        verifier=verifier,
    )
    integrated = datetime.fromisoformat(observation.integrated_at_utc)
    integrated_seconds = int(integrated.timestamp())
    integrated_at_utc = integrated.astimezone(timezone.utc).isoformat()
    remote_evidence = _verify_remote_admission(
        commitment=commitment,
        predicate=predicate,
        rekor_integrated_at_utc=integrated_at_utc,
        verifier=remote_verifier,
    )
    encoded_remote = {
        name: _encoded_projection(name, remote_evidence[name]) for name in _REMOTE_EVIDENCE_NAMES
    }
    target_round, publication, lead = _derive_target_round(integrated_seconds)
    admission = DesignSeedAttestationAdmission(
        staged_inventory_sha256=commitment.staged_inventory_sha256,
        partition_audit_file_sha256=commitment.partition_audit_file_sha256,
        phase1_view_receipt_sha256=commitment.phase1_view_receipt_sha256,
        selection_receipt_sha256=commitment.selection_receipt_sha256,
        scope_sha256=commitment.scope_sha256,
        source_p=commitment.source_p,
        source_tree=commitment.source_tree,
        commitment_sha256=commitment.commitment_sha256,
        attestation_subject_name=commitment.attestation_subject_name,
        attestation_bundle_base64=base64.b64encode(bundle).decode("ascii"),
        attestation_bundle_sha256=_digest(bundle),
        predicate_sha256=_digest(_canonical_json(predicate)),
        repository=predicate["repository"],
        workflow=predicate["workflow"],
        workflow_ref=predicate["workflow_ref"],
        workflow_sha=predicate["workflow_sha"],
        git_ref=predicate["git_ref"],
        run_id=predicate["run_id"],
        run_attempt=predicate["run_attempt"],
        event=predicate["event"],
        actor=predicate["actor"],
        triggering_actor=predicate["triggering_actor"],
        release_id=predicate["release_id"],
        release_tag=predicate["release_tag"],
        release_name=predicate["release_name"],
        release_published_at_utc=predicate["release_published_at_utc"],
        actions_run_api_projection_base64=encoded_remote["actions_run"][0],
        actions_run_api_projection_sha256=encoded_remote["actions_run"][1],
        release_api_projection_base64=encoded_remote["release"][0],
        release_api_projection_sha256=encoded_remote["release"][1],
        release_tag_api_projection_base64=encoded_remote["release_tag"][0],
        release_tag_api_projection_sha256=encoded_remote["release_tag"][1],
        rekor_log_key_sha256=observation.log_key_sha256,
        rekor_log_index=observation.log_index,
        rekor_entry_id=observation.entry_id,
        rekor_integrated_time_unix_seconds=integrated_seconds,
        rekor_integrated_at_utc=integrated_at_utc,
        rekor_timestamp_token_sha256=observation.timestamp_token_sha256,
        target_round=target_round,
        target_publication_unix_seconds=publication,
        pre_round_lead_seconds=lead,
    )
    target = (
        _safe_output_directory(output_directory)
        / f"design-seed-attestation-{commitment.scope_sha256}.json"
    )
    _write_exclusive(target, admission.canonical_file_bytes())
    return target, admission


def verify_design_seed_attestation(
    path: str | Path,
    *,
    commitment: DesignSeedCommitment,
    expected_sha256: str | None = None,
    verifier: AttestationVerifier | None = None,
    remote_verifier: RemoteAdmissionVerifier | None = None,
) -> DesignSeedAttestationAdmission:
    if not isinstance(commitment, DesignSeedCommitment):
        raise DesignSeedCommitmentError("attestation verification requires a typed commitment")
    encoded = _read_control(path, label="design-seed attestation admission")
    observed_digest = _digest(encoded)
    if expected_sha256 is not None and observed_digest != _sha256(
        "expected attestation-admission SHA-256", expected_sha256
    ):
        raise DesignSeedCommitmentError("attestation-admission digest differs")
    admission = DesignSeedAttestationAdmission.from_dict(
        _parse_canonical_file(encoded, label="design-seed attestation admission")
    )
    if encoded != admission.canonical_file_bytes():
        raise DesignSeedCommitmentError("attestation-admission bytes are not canonical")
    expected_name = f"design-seed-attestation-{commitment.scope_sha256}.json"
    if Path(path).name != expected_name:
        raise DesignSeedCommitmentError("attestation-admission filename differs from scope")
    for name in (
        "staged_inventory_sha256",
        "partition_audit_file_sha256",
        "phase1_view_receipt_sha256",
        "selection_receipt_sha256",
        "scope_sha256",
        "source_p",
        "source_tree",
        "commitment_sha256",
        "attestation_subject_name",
    ):
        expected = (
            commitment.commitment_sha256
            if name == "commitment_sha256"
            else getattr(commitment, name)
        )
        if getattr(admission, name) != expected:
            raise DesignSeedCommitmentError(f"attestation admission {name} differs")
    for admission_name, commitment_name in (
        ("workflow", "attestation_workflow"),
        ("workflow_sha", "attestation_workflow_sha"),
        ("git_ref", "attestation_git_ref"),
        ("workflow_ref", "attestation_workflow_ref"),
    ):
        if getattr(admission, admission_name) != getattr(commitment, commitment_name):
            raise DesignSeedCommitmentError(
                f"attestation admission {admission_name} differs from commitment"
            )
    with tempfile.TemporaryDirectory(prefix="fractal-design-seed-subject-") as directory:
        subject_path = Path(directory) / commitment.attestation_subject_name
        subject_path.write_bytes(commitment.canonical_file_bytes())
        os.chmod(subject_path, 0o400)
        observation, predicate = _verify_bundle(
            subject_path=subject_path,
            commitment=commitment,
            bundle_bytes=admission.bundle_bytes,
            verifier=verifier,
        )
    if _digest(_canonical_json(predicate)) != admission.predicate_sha256:
        raise DesignSeedCommitmentError("attestation predicate digest differs")
    for name in (
        "run_id",
        "run_attempt",
        "event",
        "actor",
        "triggering_actor",
        "release_id",
        "release_tag",
        "release_name",
        "release_published_at_utc",
    ):
        if getattr(admission, name) != predicate[name]:
            raise DesignSeedCommitmentError(f"attestation admission {name} differs")
    integrated = datetime.fromisoformat(observation.integrated_at_utc)
    remote_evidence = _verify_remote_admission(
        commitment=commitment,
        predicate=predicate,
        rekor_integrated_at_utc=integrated.astimezone(timezone.utc).isoformat(),
        verifier=remote_verifier,
    )
    expected_observation = {
        "rekor_entry_id": observation.entry_id,
        "rekor_integrated_at_utc": integrated.astimezone(timezone.utc).isoformat(),
        "rekor_integrated_time_unix_seconds": int(integrated.timestamp()),
        "rekor_log_index": observation.log_index,
        "rekor_log_key_sha256": observation.log_key_sha256,
        "rekor_timestamp_token_sha256": observation.timestamp_token_sha256,
    }
    for name, expected in expected_observation.items():
        if getattr(admission, name) != expected:
            raise DesignSeedCommitmentError(f"attestation admission {name} differs")
    retained_remote = {
        "actions_run": _decode_projection(
            "actions_run",
            admission.actions_run_api_projection_base64,
            admission.actions_run_api_projection_sha256,
        ),
        "release": _decode_projection(
            "release",
            admission.release_api_projection_base64,
            admission.release_api_projection_sha256,
        ),
        "release_tag": _decode_projection(
            "release_tag",
            admission.release_tag_api_projection_base64,
            admission.release_tag_api_projection_sha256,
        ),
    }
    if remote_evidence != retained_remote:
        raise DesignSeedCommitmentError("public GitHub admission differs from retained projections")
    if observed_digest != admission.admission_sha256:
        raise DesignSeedCommitmentError("attestation admission digest is inconsistent")
    return admission


def _beacon_contract(
    commitment: DesignSeedCommitment, target_round: int
) -> ExecutionBeaconContract:
    return ExecutionBeaconContract(
        drand_network=QUICKNET_NETWORK,
        chain_hash=QUICKNET_CHAIN_HASH,
        chain_scheme_id=QUICKNET_SCHEME_ID,
        chain_public_key=QUICKNET_PUBLIC_KEY,
        chain_genesis_unix_seconds=QUICKNET_GENESIS_UNIX_SECONDS,
        chain_period_seconds=QUICKNET_PERIOD_SECONDS,
        execution_round=target_round,
        label_release_round=target_round + 1,
        minimum_label_release_safety_rounds=1,
        verification_identity=commitment.commitment_sha256,
    )


def _derive_design_seed(
    *,
    commitment: DesignSeedCommitment,
    admission_sha256: str,
    target_round: int,
    beacon_sha256: str,
    randomness: str,
    signature: str,
) -> str:
    return _lp_sha256(
        _SEED_DOMAIN,
        (
            ("scope_sha256", bytes.fromhex(commitment.scope_sha256)),
            ("commitment_sha256", bytes.fromhex(commitment.commitment_sha256)),
            ("attestation_admission_sha256", bytes.fromhex(admission_sha256)),
            ("target_round", target_round.to_bytes(8, "big")),
            ("quicknet_beacon_sha256", bytes.fromhex(beacon_sha256)),
            ("quicknet_randomness", bytes.fromhex(randomness)),
            ("quicknet_signature", bytes.fromhex(signature)),
        ),
    )


def build_design_seed_reveal(
    commitment_path: str | Path,
    attestation_admission_path: str | Path,
    beacon_path: str | Path,
    *,
    output_directory: str | Path,
    attestation_verifier: AttestationVerifier | None = None,
    remote_verifier: RemoteAdmissionVerifier | None = None,
) -> tuple[Path, DesignSeedReveal]:
    commitment = verify_design_seed_commitment(commitment_path)
    admission_path = Path(attestation_admission_path).resolve(strict=True)
    admission_encoded = _read_control(admission_path, label="design-seed attestation admission")
    admission = verify_design_seed_attestation(
        admission_path,
        commitment=commitment,
        expected_sha256=_digest(admission_encoded),
        verifier=attestation_verifier,
        remote_verifier=remote_verifier,
    )
    beacon = _read_control(beacon_path, label="Quicknet beacon", max_bytes=2 * 1024)
    try:
        claims = QuicknetExecutionBeaconVerifier().verify(
            contract=_beacon_contract(commitment, admission.target_round),
            beacon_bytes=beacon,
        )
    except Exception as exc:
        raise DesignSeedCommitmentError("Quicknet beacon failed exact-P BLS verification") from exc
    seed = _derive_design_seed(
        commitment=commitment,
        admission_sha256=admission.admission_sha256,
        target_round=admission.target_round,
        beacon_sha256=claims.beacon_bytes_sha256,
        randomness=claims.randomness,
        signature=claims.signature,
    )
    reveal = DesignSeedReveal(
        staged_inventory_sha256=commitment.staged_inventory_sha256,
        partition_audit_file_sha256=commitment.partition_audit_file_sha256,
        phase1_view_receipt_sha256=commitment.phase1_view_receipt_sha256,
        selection_receipt_sha256=commitment.selection_receipt_sha256,
        scope_sha256=commitment.scope_sha256,
        source_p=commitment.source_p,
        source_tree=commitment.source_tree,
        commitment_sha256=commitment.commitment_sha256,
        attestation_admission_path=str(admission_path),
        attestation_admission_sha256=admission.admission_sha256,
        target_round=admission.target_round,
        quicknet_beacon_base64=base64.b64encode(beacon).decode("ascii"),
        quicknet_beacon_sha256=claims.beacon_bytes_sha256,
        quicknet_randomness=claims.randomness,
        quicknet_signature=claims.signature,
        design_seed_sha256=seed,
    )
    target = _safe_output_directory(output_directory) / f"design-seed-reveal-{seed}.json"
    _write_exclusive(target, reveal.canonical_file_bytes())
    return target, reveal


def verify_design_seed_reveal(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    commitment: DesignSeedCommitment,
    attestation_verifier: AttestationVerifier | None = None,
    remote_verifier: RemoteAdmissionVerifier | None = None,
) -> DesignSeedReveal:
    if not isinstance(commitment, DesignSeedCommitment):
        raise DesignSeedCommitmentError("reveal verification requires a typed commitment")
    encoded = _read_control(path, label="design-seed reveal")
    observed_digest = _digest(encoded)
    if expected_sha256 is not None and observed_digest != _sha256(
        "expected reveal SHA-256", expected_sha256
    ):
        raise DesignSeedCommitmentError("design-seed reveal digest differs")
    reveal = DesignSeedReveal.from_dict(_parse_canonical_file(encoded, label="design-seed reveal"))
    if encoded != reveal.canonical_file_bytes():
        raise DesignSeedCommitmentError("design-seed reveal bytes are not canonical")
    expected_name = f"design-seed-reveal-{reveal.design_seed_sha256}.json"
    if Path(path).name != expected_name:
        raise DesignSeedCommitmentError("reveal filename differs from the derived seed")
    for name in (
        "staged_inventory_sha256",
        "partition_audit_file_sha256",
        "phase1_view_receipt_sha256",
        "selection_receipt_sha256",
        "scope_sha256",
        "source_p",
        "source_tree",
        "commitment_sha256",
    ):
        expected = (
            commitment.commitment_sha256
            if name == "commitment_sha256"
            else getattr(commitment, name)
        )
        if getattr(reveal, name) != expected:
            raise DesignSeedCommitmentError(f"reveal {name} differs from commitment")
    admission_path = _canonical_absolute_path(
        "attestation_admission_path", reveal.attestation_admission_path
    )
    admission = verify_design_seed_attestation(
        admission_path,
        commitment=commitment,
        expected_sha256=reveal.attestation_admission_sha256,
        verifier=attestation_verifier,
        remote_verifier=remote_verifier,
    )
    if reveal.target_round != admission.target_round:
        raise DesignSeedCommitmentError("reveal target round differs from attestation admission")
    try:
        claims = QuicknetExecutionBeaconVerifier().verify(
            contract=_beacon_contract(commitment, reveal.target_round),
            beacon_bytes=reveal.beacon_bytes,
        )
    except Exception as exc:
        raise DesignSeedCommitmentError("Quicknet reveal failed exact-P BLS verification") from exc
    for name, expected in (
        ("quicknet_beacon_sha256", claims.beacon_bytes_sha256),
        ("quicknet_randomness", claims.randomness),
        ("quicknet_signature", claims.signature),
    ):
        if getattr(reveal, name) != expected:
            raise DesignSeedCommitmentError(f"reveal {name} differs from verified Quicknet")
    seed = _derive_design_seed(
        commitment=commitment,
        admission_sha256=admission.admission_sha256,
        target_round=reveal.target_round,
        beacon_sha256=claims.beacon_bytes_sha256,
        randomness=claims.randomness,
        signature=claims.signature,
    )
    if reveal.design_seed_sha256 != seed:
        raise DesignSeedCommitmentError("design seed differs from length-prefixed derivation")
    return reveal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    request = commands.add_parser("build-request")
    request.add_argument("--staged-inventory-sha256", required=True)
    request.add_argument("--partition-audit-file-sha256", required=True)
    request.add_argument("--phase1-view-receipt-sha256", required=True)
    request.add_argument("--selection-receipt-sha256", required=True)
    request.add_argument("--attestation-workflow", required=True)
    request.add_argument("--attestation-workflow-sha", required=True)
    request.add_argument("--attestation-git-ref", required=True)
    request.add_argument("--output-directory", type=Path, required=True)

    verify_request = commands.add_parser("verify-request")
    verify_request.add_argument("--request", type=Path, required=True)
    verify_request.add_argument("--expected-sha256")

    commit = commands.add_parser("build-commitment")
    commit.add_argument("--request", type=Path, required=True)
    commit.add_argument("--output-directory", type=Path, required=True)

    verify_commitment = commands.add_parser("verify-commitment")
    verify_commitment.add_argument("--commitment", type=Path, required=True)
    verify_commitment.add_argument("--expected-sha256")

    attest = commands.add_parser("admit-attestation")
    attest.add_argument("--commitment", type=Path, required=True)
    attest.add_argument("--bundle", type=Path, required=True)
    attest.add_argument("--output-directory", type=Path, required=True)

    verify_attestation = commands.add_parser("verify-attestation")
    verify_attestation.add_argument("--commitment", type=Path, required=True)
    verify_attestation.add_argument("--admission", type=Path, required=True)
    verify_attestation.add_argument("--expected-sha256")

    reveal = commands.add_parser("build-reveal")
    reveal.add_argument("--commitment", type=Path, required=True)
    reveal.add_argument("--admission", type=Path, required=True)
    reveal.add_argument("--beacon", type=Path, required=True)
    reveal.add_argument("--output-directory", type=Path, required=True)

    verify_reveal = commands.add_parser("verify-reveal")
    verify_reveal.add_argument("--commitment", type=Path, required=True)
    verify_reveal.add_argument("--reveal", type=Path, required=True)
    verify_reveal.add_argument("--expected-sha256")
    return parser


def _result(kind: str, path: Path, sha256: str, scope: str) -> None:
    print(
        _canonical_json(
            {
                "artifact_kind": kind,
                "artifact_path": str(path.resolve(strict=True)),
                "artifact_sha256": sha256,
                "schema_version": "fractal-design-seed-cli-result-v1",
                "scope_sha256": scope,
            }
        ).decode("ascii")
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "build-request":
            path, value = build_design_seed_request(
                staged_inventory_sha256=arguments.staged_inventory_sha256,
                partition_audit_file_sha256=arguments.partition_audit_file_sha256,
                phase1_view_receipt_sha256=arguments.phase1_view_receipt_sha256,
                selection_receipt_sha256=arguments.selection_receipt_sha256,
                attestation_workflow=arguments.attestation_workflow,
                attestation_workflow_sha=arguments.attestation_workflow_sha,
                attestation_git_ref=arguments.attestation_git_ref,
                output_directory=arguments.output_directory,
            )
            _result("request", path, value.request_sha256, value.scope_sha256)
        elif arguments.command == "verify-request":
            value = verify_design_seed_request(
                arguments.request, expected_sha256=arguments.expected_sha256
            )
            _result("request", arguments.request, value.request_sha256, value.scope_sha256)
        elif arguments.command == "build-commitment":
            path, value = build_design_seed_commitment(
                arguments.request, output_directory=arguments.output_directory
            )
            _result("commitment", path, value.commitment_sha256, value.scope_sha256)
        elif arguments.command == "verify-commitment":
            value = verify_design_seed_commitment(
                arguments.commitment, expected_sha256=arguments.expected_sha256
            )
            _result("commitment", arguments.commitment, value.commitment_sha256, value.scope_sha256)
        elif arguments.command == "admit-attestation":
            path, value = admit_design_seed_attestation(
                arguments.commitment,
                arguments.bundle,
                output_directory=arguments.output_directory,
            )
            _result("attestation-admission", path, value.admission_sha256, value.scope_sha256)
        elif arguments.command == "verify-attestation":
            commitment = verify_design_seed_commitment(arguments.commitment)
            value = verify_design_seed_attestation(
                arguments.admission,
                commitment=commitment,
                expected_sha256=arguments.expected_sha256,
            )
            _result(
                "attestation-admission",
                arguments.admission,
                value.admission_sha256,
                value.scope_sha256,
            )
        elif arguments.command == "build-reveal":
            path, value = build_design_seed_reveal(
                arguments.commitment,
                arguments.admission,
                arguments.beacon,
                output_directory=arguments.output_directory,
            )
            _result("reveal", path, value.reveal_sha256, value.scope_sha256)
        else:
            commitment = verify_design_seed_commitment(arguments.commitment)
            value = verify_design_seed_reveal(
                arguments.reveal,
                commitment=commitment,
                expected_sha256=arguments.expected_sha256,
            )
            _result("reveal", arguments.reveal, value.reveal_sha256, value.scope_sha256)
    except DesignSeedCommitmentError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
