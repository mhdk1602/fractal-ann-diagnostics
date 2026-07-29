"""Non-production provider rehearsal with typed, byte-bound evidence.

This module never reads or writes the confirmatory suite ledger.  It admits the
candidate-normalized form of the three production provider plans, derives
rehearsal-only runner labels, verifies the live GitHub run and job, and launches
only the fixed image help command with networking disabled and no study-data
mounts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from .execution_claim import (
    ANALYSIS_PHASE,
    BASE_EXECUTE_RUNNER_LABELS,
    LABEL_RELEASE_PHASE,
    ONLINE_PHASE,
    PHASE_RUNTIME_BINDINGS,
    ExecutionClaimError,
    PhaseHostToolContract,
    PhaseHostToolReceipt,
    ProviderPhase,
    load_materialized_provider_phase_plan,
    load_provider_phase_plans,
    materialize_provider_phase_plan,
    provider_phase_plan_templates_sha256,
    verify_phase_host_tools,
)
from .study import load_study_manifest

REPOSITORY = "mhdk1602/fractal-ann-diagnostics"
REHEARSAL_WORKFLOW_PATH = ".github/workflows/confirmatory-provider-rehearsal.yml"
REHEARSAL_RUNNER_LABEL_DERIVATION = "sha256-fractal-provider-rehearsal-label-v1"
REHEARSAL_RUNNER_LABEL_PREFIX = "fractal-ann-rehearsal-"
REHEARSAL_PLAN_SCHEMA = "fractal-provider-rehearsal-plan-v2"
REHEARSAL_BOOTSTRAP_SCHEMA = "fractal-provider-rehearsal-bootstrap-v1"
REHEARSAL_LIVE_JOB_SCHEMA = "fractal-provider-rehearsal-live-job-v1"
REHEARSAL_PHASE_RECEIPT_SCHEMA = "fractal-provider-rehearsal-phase-v1"
REHEARSAL_AGGREGATE_SCHEMA = "fractal-provider-rehearsal-aggregate-v2"
REHEARSAL_TAG_PROBE_SCHEMA = "fractal-provider-tag-head-branch-probe-v1"
REHEARSAL_INCIDENT_SCHEMA = "fractal-provider-rehearsal-incident-v1"
REPOSITORY_RUNNER_INVENTORY_SCHEMA = "fractal-repository-runner-inventory-v1"
REPOSITORY_RUNNER_SNAPSHOT_SCHEMA = "fractal-repository-runner-snapshot-v1"
CANDIDATE_IMAGE_CLOSURE_SCHEMA = "fractal-c0-candidate-closure-v2"
CANDIDATE_IMAGE_BOOTSTRAP_CLOSURE_SCHEMA = "fractal-c0-candidate-bootstrap-closure-v1"

PHASES: tuple[ProviderPhase, ...] = (
    ONLINE_PHASE,
    LABEL_RELEASE_PHASE,
    ANALYSIS_PHASE,
)
REHEARSAL_JOB_NAMES: Mapping[ProviderPhase, str] = {
    ONLINE_PHASE: "rehearse-online",
    LABEL_RELEASE_PHASE: "rehearse-label-release",
    ANALYSIS_PHASE: "rehearse-analysis",
}
PHASE_OUTPUT_PREFIX: Mapping[ProviderPhase, str] = {
    ONLINE_PHASE: "online",
    LABEL_RELEASE_PHASE: "label_release",
    ANALYSIS_PHASE: "analysis",
}
FORBIDDEN_CONTAINER_ENVIRONMENT = frozenset(
    {
        "ACTIONS_CACHE_URL",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "ACTIONS_RESULTS_URL",
        "ACTIONS_RUNTIME_TOKEN",
        "ACTIONS_RUNTIME_URL",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OCI_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CANDIDATE_BRANCH = re.compile(r"^c0-candidate/[a-z0-9._-]+$")
_CANDIDATE_TAG = re.compile(r"^c0-head-branch-probe/[a-z0-9._-]+$")
_REHEARSAL_LABEL = re.compile(r"^fractal-ann-rehearsal-[a-z-]+-[0-9a-f]{24}$")


class ProviderRehearsalError(ValueError):
    """Raised when non-production provider evidence is not exact."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ProviderRehearsalError("rehearsal evidence is not canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_output(root: Path, *arguments: str, label: str) -> str:
    try:
        completed = subprocess.run(
            (
                "git",
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-C",
                str(root),
                *arguments,
            ),
            check=False,
            capture_output=True,
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            },
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProviderRehearsalError(f"cannot inspect {label}") from exc
    if (
        completed.returncode != 0
        or len(completed.stdout) > 1024 * 1024
        or len(completed.stderr) > 1024 * 1024
    ):
        raise ProviderRehearsalError(f"cannot inspect {label}")
    try:
        return completed.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProviderRehearsalError(f"{label} is not UTF-8") from exc


def _verify_workflow_package_tree(
    source_root: str | Path,
    *,
    workflow_sha: str,
) -> tuple[str, str]:
    """Bind checkout A to its clean package tree before admitting source P."""

    root = Path(source_root)
    if not root.is_absolute():
        raise ProviderRehearsalError("workflow source root must be absolute")
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ProviderRehearsalError("workflow source root is unavailable") from exc
    if resolved != root or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProviderRehearsalError("workflow source root must be one canonical directory")
    top_level = _git_output(root, "rev-parse", "--show-toplevel", label="workflow Git root").strip()
    if top_level != str(root):
        raise ProviderRehearsalError("workflow source root differs from the Git root")
    observed_head = _git_output(root, "rev-parse", "HEAD", label="workflow commit").strip()
    if observed_head != workflow_sha:
        raise ProviderRehearsalError("workflow checkout commit differs from A")
    package_root = root / "src" / "fractal_ann_diagnostics"
    try:
        package_metadata = package_root.lstat()
        package_real = package_root.resolve(strict=True)
    except OSError as exc:
        raise ProviderRehearsalError("workflow package root is unavailable") from exc
    if (
        package_real != package_root
        or stat.S_ISLNK(package_metadata.st_mode)
        or not stat.S_ISDIR(package_metadata.st_mode)
    ):
        raise ProviderRehearsalError("workflow package root is not canonical")
    package_tree = _git_output(
        root,
        "rev-parse",
        "HEAD:src/fractal_ann_diagnostics",
        label="workflow package tree",
    ).strip()
    _git_commit("workflow_python_package_source_tree", package_tree)
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
        "--",
        "src/fractal_ann_diagnostics",
        label="workflow package worktree",
    )
    if status:
        raise ProviderRehearsalError(
            "workflow package worktree contains changed, untracked, or ignored bytes"
        )
    try:
        workflow_source = _git_output(
            root,
            "show",
            f"HEAD:{REHEARSAL_WORKFLOW_PATH}",
            label="workflow-fixed host-Python launcher",
        )
        workflow = yaml.safe_load(workflow_source)
        launcher_source = workflow["env"]["HOST_PYTHON_VERIFIED_LAUNCHER"]
    except (TypeError, KeyError, yaml.YAMLError) as exc:
        raise ProviderRehearsalError(
            "cannot recover the workflow-fixed host-Python launcher"
        ) from exc
    if type(launcher_source) is not str or not launcher_source:
        raise ProviderRehearsalError(
            "workflow-fixed host-Python launcher is not one literal source string"
        )
    launcher_sha256 = _sha256(launcher_source.encode("utf-8"))
    return package_tree, launcher_sha256


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderRehearsalError(f"{name} must be one canonical non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise ProviderRehearsalError(f"{name} must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ProviderRehearsalError(f"{name} contains a control character")
    return value


def _digest(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProviderRehearsalError(f"{name} must be one lowercase SHA-256")
    return value


def _oci_digest(name: str, value: object) -> str:
    if type(value) is not str or _OCI_DIGEST.fullmatch(value) is None:
        raise ProviderRehearsalError(f"{name} must be one OCI SHA-256 digest")
    return value


def _git_commit(name: str, value: object) -> str:
    if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
        raise ProviderRehearsalError(f"{name} must be one full lowercase Git commit")
    return value


def _positive(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ProviderRehearsalError(f"{name} must be a positive integer")
    return value


def _runner_group(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ProviderRehearsalError("runner_group_id must be null or non-negative")
    return value


def _timestamp(name: str, value: object) -> str:
    text = _text(name, value)
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ProviderRehearsalError(f"{name} must use canonical ISO 8601") from exc
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise ProviderRehearsalError(f"{name} must use UTC")
    if instant.isoformat() != text:
        raise ProviderRehearsalError(f"{name} must use canonical ISO 8601")
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProviderRehearsalError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise ProviderRehearsalError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ProviderRehearsalError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ProviderRehearsalError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderRehearsalError(f"cannot decode {label}") from exc
    if not isinstance(value, Mapping):
        raise ProviderRehearsalError(f"{label} must contain one object")
    return value


def _read_canonical_object(
    path: str | Path,
    *,
    label: str,
    maximum_bytes: int = 4 * 1024 * 1024,
) -> tuple[Mapping[str, Any], bytes]:
    candidate = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > maximum_bytes
            ):
                raise OSError("evidence file is not one bounded singly linked regular file")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise OSError("short evidence read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise OSError("evidence grew while read")
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise OSError("evidence changed while read")
            encoded = b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ProviderRehearsalError(f"cannot read {label}") from exc
    if (
        not encoded
        or len(encoded) > maximum_bytes
        or not encoded.endswith(b"\n")
        or encoded.endswith(b"\n\n")
    ):
        raise ProviderRehearsalError(f"{label} bytes are not bounded canonical JSON")
    value = _decode_object(encoded[:-1], label=label)
    if encoded != _canonical_bytes(value) + b"\n":
        raise ProviderRehearsalError(f"{label} bytes are not canonical")
    return value, encoded


def _write_exclusive(path: Path, encoded: bytes) -> None:
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(encoded)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError("short write")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ProviderRehearsalError(f"cannot write exclusive evidence path {path}") from exc


def _write_receipt(path: Path, value: Mapping[str, object]) -> tuple[Path, str]:
    encoded = _canonical_bytes(value) + b"\n"
    _write_exclusive(path, encoded)
    return path, _sha256(encoded)


def derive_rehearsal_runner_label(
    *,
    phase: ProviderPhase,
    plan_sha256: str,
    workflow_sha: str,
    run_id: int,
    run_attempt: int,
) -> str:
    """Derive the sole queue label after the provider run ID exists."""

    if phase not in PHASES:
        raise ProviderRehearsalError("rehearsal runner phase differs")
    payload = b"\0".join(
        (
            REHEARSAL_RUNNER_LABEL_DERIVATION.encode("ascii"),
            phase.encode("ascii"),
            bytes.fromhex(_digest("plan_sha256", plan_sha256)),
            bytes.fromhex(_git_commit("workflow_sha", workflow_sha)),
            str(_positive("run_id", run_id)).encode("ascii"),
            str(_positive("run_attempt", run_attempt)).encode("ascii"),
        )
    )
    return f"{REHEARSAL_RUNNER_LABEL_PREFIX}{phase}-{_sha256(payload)[:24]}"


def provider_plan_template_closure_sha256(
    manifest_path: str | Path,
    *,
    c0_commit: str,
) -> str:
    """Hash the exact three plans after resolving only the C0 sentinel."""

    manifest = load_study_manifest(manifest_path)
    try:
        return provider_phase_plan_templates_sha256(
            manifest,
            validation_mode="candidate-rehearsal",
            c0_commit=c0_commit,
        )
    except ExecutionClaimError as exc:
        raise ProviderRehearsalError(f"cannot hash provider-plan templates: {exc}") from exc


@dataclass(frozen=True)
class CandidateImageClosure:
    build_context_tree_sha256: str
    candidate_branch: str
    candidate_package_checksums_sha256: str
    github_ref: str
    github_run_attempt: int
    github_run_id: int
    github_sha: str
    github_workflow_ref: str
    github_workflow_sha: str
    mode: str
    release_govulncheck_adjudication_sha256: str
    release_image_index_digest: str
    release_image_reference: str
    release_linux_arm64_manifest_digest: str
    release_oci_attestation_bundle_sha256: str
    release_oci_attestation_verification_sha256: str
    release_reproducibility_receipt_sha256: str
    release_security_adjudication_sha256: str
    release_tle_interoperability_receipt_sha256: str
    repository: str
    scientific_image_index_digest: str
    scientific_image_reference: str
    scientific_linux_amd64_manifest_digest: str
    scientific_linux_amd64_runtime_extraction_sha256: str
    scientific_linux_arm64_manifest_digest: str
    scientific_linux_arm64_runtime_extraction_sha256: str
    scientific_oci_attestation_bundle_sha256: str
    scientific_oci_attestation_verification_sha256: str
    schema_version: str = CANDIDATE_IMAGE_CLOSURE_SCHEMA

    def __post_init__(self) -> None:
        _digest("build_context_tree_sha256", self.build_context_tree_sha256)
        branch = _text("candidate_branch", self.candidate_branch)
        if _CANDIDATE_BRANCH.fullmatch(branch) is None:
            raise ProviderRehearsalError("candidate image branch is outside c0-candidate/*")
        if self.github_ref != f"refs/heads/{branch}":
            raise ProviderRehearsalError("candidate image ref differs from its branch")
        for name in ("github_run_attempt", "github_run_id"):
            _positive(name, getattr(self, name))
        for name in ("github_sha", "github_workflow_sha"):
            _git_commit(name, getattr(self, name))
        if self.github_sha != self.github_workflow_sha:
            raise ProviderRehearsalError("candidate image workflow did not execute its head SHA")
        if self.repository != REPOSITORY or self.mode != "candidate":
            raise ProviderRehearsalError("candidate image closure repository or mode differs")
        expected_workflow_ref = (
            f"{REPOSITORY}/.github/workflows/confirmatory-image.yml@{self.github_ref}"
        )
        if self.github_workflow_ref != expected_workflow_ref:
            raise ProviderRehearsalError("candidate image workflow ref differs")
        _digest(
            "candidate_package_checksums_sha256",
            self.candidate_package_checksums_sha256,
        )
        for name in (
            "release_image_index_digest",
            "release_linux_arm64_manifest_digest",
            "scientific_image_index_digest",
            "scientific_linux_amd64_manifest_digest",
            "scientific_linux_arm64_manifest_digest",
        ):
            _oci_digest(name, getattr(self, name))
        for name in (
            "release_govulncheck_adjudication_sha256",
            "release_oci_attestation_bundle_sha256",
            "release_oci_attestation_verification_sha256",
            "release_reproducibility_receipt_sha256",
            "release_security_adjudication_sha256",
            "release_tle_interoperability_receipt_sha256",
            "scientific_linux_amd64_runtime_extraction_sha256",
            "scientific_linux_arm64_runtime_extraction_sha256",
            "scientific_oci_attestation_bundle_sha256",
            "scientific_oci_attestation_verification_sha256",
        ):
            _digest(name, getattr(self, name))
        expected_references = {
            "scientific_image_reference": (
                "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-candidate@"
                f"{self.scientific_image_index_digest}"
            ),
            "release_image_reference": (
                "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-release-candidate@"
                f"{self.release_image_index_digest}"
            ),
        }
        for name, expected in expected_references.items():
            if getattr(self, name) != expected:
                raise ProviderRehearsalError(f"candidate closure {name} differs")
        if self.schema_version != CANDIDATE_IMAGE_CLOSURE_SCHEMA:
            raise ProviderRehearsalError("candidate image closure schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def bootstrap_closure_dict(self) -> dict[str, str]:
        """Return the executable identity that survives the P-to-C0 transition."""

        return {
            "build_context_tree_sha256": self.build_context_tree_sha256,
            "release_image_index_digest": self.release_image_index_digest,
            "release_linux_arm64_manifest_digest": (self.release_linux_arm64_manifest_digest),
            "schema_version": CANDIDATE_IMAGE_BOOTSTRAP_CLOSURE_SCHEMA,
            "scientific_image_index_digest": self.scientific_image_index_digest,
            "scientific_linux_amd64_manifest_digest": (self.scientific_linux_amd64_manifest_digest),
            "scientific_linux_arm64_manifest_digest": (self.scientific_linux_arm64_manifest_digest),
        }

    @property
    def bootstrap_closure_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.bootstrap_closure_dict()))

    @property
    def file_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()) + b"\n")

    @classmethod
    def from_file(cls, path: str | Path) -> CandidateImageClosure:
        value, encoded = _read_canonical_object(path, label="candidate image closure")
        result = cls(**_closed(value, frozenset(cls.__dataclass_fields__), label="closure"))
        if encoded != _canonical_bytes(result.to_dict()) + b"\n":
            raise ProviderRehearsalError("candidate image closure changed after parsing")
        return result


@dataclass(frozen=True)
class RehearsalPhaseAdmission:
    phase: ProviderPhase
    build_context_tree_sha256: str
    candidate_bootstrap_closure_sha256: str
    candidate_image_closure_file_sha256: str
    candidate_image_reference: str
    candidate_image_index_digest: str
    candidate_platform_manifest_digest: str
    candidate_runtime_probe_receipt_sha256: str
    candidate_image_source_commit: str
    c0_commit: str
    manifest_sha256: str
    plan_closure_sha256: str
    provider_plan_path: str
    provider_plan_sha256: str
    provider_plan_file_sha256: str
    host_controlled_root: str
    host_python_path: str
    host_python_file_sha256: str
    host_python_venv_root: str
    host_python_venv_tree_sha256: str
    host_python_venv_symlink_inventory_sha256: str
    host_python_import_root: str
    host_python_import_tree_sha256: str
    host_python_launcher_sha256: str
    host_python_package_content_sha256: str
    host_python_package_tree_sha256: str
    host_python_package_source_commit: str
    host_python_package_source_tree: str
    workflow_python_package_source_tree: str
    workflow_python_launcher_sha256: str
    host_gh_path: str
    host_gh_file_sha256: str
    host_docker_path: str
    host_docker_file_sha256: str
    host_tools_contract_sha256: str
    runtime_platform: str
    runtime_image_role: str
    runtime_index_role: str
    runner_archive_sha256: str
    runner_group_id: None
    runner_label: str
    runner_version: str
    workflow_sha: str
    run_id: int
    run_attempt: int
    schema_version: str = REHEARSAL_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ProviderRehearsalError("rehearsal admission phase differs")
        for name in (
            "build_context_tree_sha256",
            "candidate_bootstrap_closure_sha256",
            "candidate_image_closure_file_sha256",
            "manifest_sha256",
            "plan_closure_sha256",
            "provider_plan_sha256",
            "provider_plan_file_sha256",
            "candidate_runtime_probe_receipt_sha256",
            "host_python_file_sha256",
            "host_python_venv_tree_sha256",
            "host_python_venv_symlink_inventory_sha256",
            "host_python_import_tree_sha256",
            "host_python_launcher_sha256",
            "host_python_package_content_sha256",
            "host_python_package_tree_sha256",
            "host_gh_file_sha256",
            "host_docker_file_sha256",
            "host_tools_contract_sha256",
            "runner_archive_sha256",
        ):
            _digest(name, getattr(self, name))
        _git_commit("candidate_image_source_commit", self.candidate_image_source_commit)
        _git_commit(
            "host_python_package_source_commit",
            self.host_python_package_source_commit,
        )
        _digest(
            "workflow_python_launcher_sha256",
            self.workflow_python_launcher_sha256,
        )
        _git_commit(
            "host_python_package_source_tree",
            self.host_python_package_source_tree,
        )
        _git_commit(
            "workflow_python_package_source_tree",
            self.workflow_python_package_source_tree,
        )
        if self.host_python_package_source_commit != self.candidate_image_source_commit:
            raise ProviderRehearsalError(
                "host Python package provenance differs from candidate source P"
            )
        if self.workflow_python_package_source_tree != self.host_python_package_source_tree:
            raise ProviderRehearsalError(
                "workflow package tree at A differs from candidate package tree at P"
            )
        if self.workflow_python_launcher_sha256 != self.host_python_launcher_sha256:
            raise ProviderRehearsalError(
                "workflow launcher source differs from the source-P launcher pin"
            )
        _git_commit("c0_commit", self.c0_commit)
        _git_commit("workflow_sha", self.workflow_sha)
        for name in ("run_id", "run_attempt"):
            _positive(name, getattr(self, name))
        for name in (
            "provider_plan_path",
            "host_controlled_root",
            "host_python_path",
            "host_python_venv_root",
            "host_python_import_root",
            "host_gh_path",
            "host_docker_path",
        ):
            if not Path(getattr(self, name)).is_absolute():
                raise ProviderRehearsalError(f"{name} must be absolute")
        if os.pathsep in self.host_python_import_root:
            raise ProviderRehearsalError("host_python_import_root cannot contain multiple roots")
        try:
            relative_import = Path(self.host_python_import_root).relative_to(
                self.host_python_venv_root
            )
            Path(self.host_python_venv_root).relative_to(self.host_controlled_root)
        except ValueError as exc:
            raise ProviderRehearsalError(
                "host Python import closure escapes controlled_root"
            ) from exc
        if relative_import.parts != ("lib", "python3.12", "site-packages"):
            raise ProviderRehearsalError("host Python import root differs")
        _oci_digest("candidate_image_index_digest", self.candidate_image_index_digest)
        _oci_digest(
            "candidate_platform_manifest_digest",
            self.candidate_platform_manifest_digest,
        )
        if not self.candidate_image_reference.endswith(f"@{self.candidate_image_index_digest}"):
            raise ProviderRehearsalError("candidate image reference differs from its index")
        expected_runtime = PHASE_RUNTIME_BINDINGS[self.phase]
        if (
            self.runtime_platform,
            self.runtime_image_role,
            self.runtime_index_role,
        ) != expected_runtime:
            raise ProviderRehearsalError("rehearsal admission runtime binding differs")
        if self.runner_group_id is not None:
            raise ProviderRehearsalError(
                "personal-repository rehearsal runner_group_id must be null"
            )
        _text("runner_version", self.runner_version)
        expected_label = derive_rehearsal_runner_label(
            phase=self.phase,
            plan_sha256=self.provider_plan_sha256,
            workflow_sha=self.workflow_sha,
            run_id=self.run_id,
            run_attempt=self.run_attempt,
        )
        if (
            self.runner_label != expected_label
            or _REHEARSAL_LABEL.fullmatch(self.runner_label) is None
        ):
            raise ProviderRehearsalError("rehearsal runner label derivation differs")
        if self.schema_version != REHEARSAL_PLAN_SCHEMA:
            raise ProviderRehearsalError("rehearsal phase-admission schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def admission_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> RehearsalPhaseAdmission:
        return cls(**_closed(value, frozenset(cls.__dataclass_fields__), label="phase admission"))


def _candidate_binding(
    phase: ProviderPhase,
    closure: CandidateImageClosure,
) -> tuple[str, str, str, str]:
    if phase == ONLINE_PHASE:
        return (
            closure.scientific_image_reference,
            closure.scientific_image_index_digest,
            closure.scientific_linux_arm64_manifest_digest,
            closure.scientific_linux_arm64_runtime_extraction_sha256,
        )
    if phase == ANALYSIS_PHASE:
        return (
            closure.scientific_image_reference,
            closure.scientific_image_index_digest,
            closure.scientific_linux_amd64_manifest_digest,
            closure.scientific_linux_amd64_runtime_extraction_sha256,
        )
    return (
        closure.release_image_reference,
        closure.release_image_index_digest,
        closure.release_linux_arm64_manifest_digest,
        closure.release_reproducibility_receipt_sha256,
    )


def build_rehearsal_admissions(
    *,
    manifest_path: str | Path,
    c0_commit: str,
    candidate_closure: CandidateImageClosure,
    workflow_sha: str,
    run_id: int,
    run_attempt: int,
    materialization_root: str | Path,
    workflow_source_root: str | Path,
) -> tuple[Mapping[ProviderPhase, RehearsalPhaseAdmission], tuple[Path, ...]]:
    """Load the production plans and build the three rehearsal queue admissions."""

    commit = _git_commit("c0_commit", c0_commit)
    workflow = _git_commit("workflow_sha", workflow_sha)
    if commit != workflow:
        raise ProviderRehearsalError("candidate manifest and rehearsal workflow SHA differ")
    workflow_package_tree, workflow_launcher_sha256 = _verify_workflow_package_tree(
        workflow_source_root,
        workflow_sha=workflow,
    )
    try:
        plans = load_provider_phase_plans(
            manifest_path,
            c1_commit=commit,
            validation_mode="candidate-rehearsal",
            c0_commit=commit,
        )
    except (ExecutionClaimError, ValueError) as exc:
        raise ProviderRehearsalError(f"provider-plan loader rejected rehearsal: {exc}") from exc
    closure_sha256 = provider_plan_template_closure_sha256(
        manifest_path,
        c0_commit=commit,
    )
    root = Path(materialization_root)
    if not root.is_absolute():
        raise ProviderRehearsalError("hosted materialization root must be absolute")
    admissions: dict[ProviderPhase, RehearsalPhaseAdmission] = {}
    materializations: list[Path] = []
    for phase in PHASES:
        plan = plans[phase]
        candidate_reference, index_digest, manifest_digest, probe_digest = _candidate_binding(
            phase, candidate_closure
        )
        if (
            plan.oci_index_digest != index_digest
            or plan.oci_platform_manifest_digest != manifest_digest
            or plan.runtime_probe_receipt_sha256 != probe_digest
            or not plan.runtime_image.endswith(f"@{index_digest}")
        ):
            raise ProviderRehearsalError(
                f"{phase} candidate image closure differs from the production plan"
            )
        if plan.host_tools.python_package_source_tree != workflow_package_tree:
            raise ProviderRehearsalError(f"{phase} source P package tree differs from workflow A")
        if plan.host_tools.python_launcher_sha256 != workflow_launcher_sha256:
            raise ProviderRehearsalError(f"{phase} launcher pin differs from workflow A")
        materialization = materialize_provider_phase_plan(plan, root / phase)
        materializations.append(materialization)
        admissions[phase] = RehearsalPhaseAdmission(
            phase=phase,
            build_context_tree_sha256=candidate_closure.build_context_tree_sha256,
            candidate_bootstrap_closure_sha256=(candidate_closure.bootstrap_closure_sha256),
            candidate_image_closure_file_sha256=candidate_closure.file_sha256,
            candidate_image_reference=candidate_reference,
            candidate_image_index_digest=index_digest,
            candidate_platform_manifest_digest=manifest_digest,
            candidate_runtime_probe_receipt_sha256=probe_digest,
            candidate_image_source_commit=candidate_closure.github_sha,
            c0_commit=commit,
            manifest_sha256=plan.manifest_sha256,
            plan_closure_sha256=closure_sha256,
            provider_plan_path=plan.provider_plan_path,
            provider_plan_sha256=plan.plan_sha256,
            provider_plan_file_sha256=plan.file_sha256,
            host_controlled_root=plan.host_tools.controlled_root,
            host_python_path=plan.host_tools.python_executable,
            host_python_file_sha256=plan.host_tools.python_executable_sha256,
            host_python_venv_root=plan.host_tools.venv_root,
            host_python_venv_tree_sha256=plan.host_tools.venv_tree_sha256,
            host_python_venv_symlink_inventory_sha256=(
                plan.host_tools.venv_symlink_inventory_sha256
            ),
            host_python_import_root=plan.host_tools.python_import_root,
            host_python_import_tree_sha256=(plan.host_tools.python_import_tree_sha256),
            host_python_launcher_sha256=plan.host_tools.python_launcher_sha256,
            host_python_package_content_sha256=(plan.host_tools.python_package_content_sha256),
            host_python_package_tree_sha256=(plan.host_tools.python_package_tree_sha256),
            host_python_package_source_commit=(plan.host_tools.python_package_source_commit),
            host_python_package_source_tree=(plan.host_tools.python_package_source_tree),
            workflow_python_package_source_tree=workflow_package_tree,
            workflow_python_launcher_sha256=workflow_launcher_sha256,
            host_gh_path=plan.host_tools.gh_executable,
            host_gh_file_sha256=plan.host_tools.gh_executable_sha256,
            host_docker_path=plan.host_tools.docker_executable,
            host_docker_file_sha256=plan.host_tools.docker_executable_sha256,
            host_tools_contract_sha256=plan.host_tools.contract_sha256,
            runtime_platform=plan.runtime_platform,
            runtime_image_role=plan.runtime_image_role,
            runtime_index_role=plan.runtime_index_role,
            runner_archive_sha256=plan.runner_archive_sha256,
            runner_group_id=plan.runner_group_id,
            runner_label=derive_rehearsal_runner_label(
                phase=phase,
                plan_sha256=plan.plan_sha256,
                workflow_sha=workflow,
                run_id=run_id,
                run_attempt=run_attempt,
            ),
            runner_version=plan.runner_version,
            workflow_sha=workflow,
            run_id=run_id,
            run_attempt=run_attempt,
        )
    return admissions, tuple(materializations)


@dataclass(frozen=True)
class RehearsalRunnerBootstrapReceipt:
    phase: ProviderPhase
    repository: str
    workflow_sha: str
    runner_label: str
    runner_id: int
    runner_name: str
    runner_group_id: int | None
    runner_version: str
    runner_archive_sha256: str
    repository_runner_inventory_sha256: str
    ephemeral: bool
    disable_update: bool
    unattended: bool
    registered_at_utc: str
    schema_version: str = REHEARSAL_BOOTSTRAP_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in PHASES or self.repository != REPOSITORY:
            raise ProviderRehearsalError("runner bootstrap phase or repository differs")
        _git_commit("workflow_sha", self.workflow_sha)
        if _REHEARSAL_LABEL.fullmatch(self.runner_label) is None:
            raise ProviderRehearsalError("runner bootstrap label is invalid")
        _positive("runner_id", self.runner_id)
        if self.runner_group_id is not None:
            raise ProviderRehearsalError(
                "personal-repository rehearsal runner_group_id must be null"
            )
        _text("runner_name", self.runner_name)
        _text("runner_version", self.runner_version)
        _digest("runner_archive_sha256", self.runner_archive_sha256)
        _digest(
            "repository_runner_inventory_sha256",
            self.repository_runner_inventory_sha256,
        )
        if (self.ephemeral, self.disable_update, self.unattended) != (True, True, True):
            raise ProviderRehearsalError("rehearsal runner must be ephemeral and update-disabled")
        _timestamp("registered_at_utc", self.registered_at_utc)
        if self.schema_version != REHEARSAL_BOOTSTRAP_SCHEMA:
            raise ProviderRehearsalError("runner bootstrap schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def file_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()) + b"\n")

    @classmethod
    def from_file(cls, path: str | Path) -> RehearsalRunnerBootstrapReceipt:
        value, encoded = _read_canonical_object(path, label="runner bootstrap receipt")
        result = cls(
            **_closed(value, frozenset(cls.__dataclass_fields__), label="bootstrap receipt")
        )
        if encoded != _canonical_bytes(result.to_dict()) + b"\n":
            raise ProviderRehearsalError("runner bootstrap bytes changed after parsing")
        return result


class GitHubBytesApi(Protocol):
    def get_bytes(self, endpoint: str) -> bytes: ...


class GitHubCliBytesApi:
    """Read GitHub REST bytes with one fixed gh executable."""

    def __init__(self, executable: str, environment: Mapping[str, str]) -> None:
        self._executable = executable
        self._environment = dict(environment)

    def get_bytes(self, endpoint: str) -> bytes:
        completed = subprocess.run(
            (
                self._executable,
                "api",
                "-H",
                "Accept: application/vnd.github+json",
                "-H",
                "X-GitHub-Api-Version: 2022-11-28",
                endpoint,
            ),
            check=False,
            capture_output=True,
            env=self._environment,
            timeout=60,
        )
        if completed.returncode != 0:
            raise ProviderRehearsalError("GitHub REST read failed")
        if not completed.stdout or len(completed.stdout) > 16 * 1024 * 1024:
            raise ProviderRehearsalError("GitHub REST response is empty or oversized")
        return completed.stdout


@dataclass(frozen=True)
class RepositoryRunnerSnapshot:
    runner_id: int
    runner_name: str
    operating_system: str
    status: str
    busy: bool
    labels: tuple[str, ...]
    schema_version: str = REPOSITORY_RUNNER_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        _positive("repository runner ID", self.runner_id)
        _text("repository runner name", self.runner_name)
        _text("repository runner operating system", self.operating_system)
        if self.status not in {"offline", "online"}:
            raise ProviderRehearsalError("repository runner status differs")
        if type(self.busy) is not bool:
            raise ProviderRehearsalError("repository runner busy flag must be Boolean")
        labels = tuple(self.labels)
        if (
            not labels
            or not all(type(item) is str and item for item in labels)
            or labels != tuple(sorted(labels, key=lambda item: item.encode("utf-8")))
            or len(labels) != len(set(labels))
        ):
            raise ProviderRehearsalError(
                "repository runner labels must be unique canonical byte order"
            )
        if self.schema_version != REPOSITORY_RUNNER_SNAPSHOT_SCHEMA:
            raise ProviderRehearsalError("repository runner snapshot schema differs")
        object.__setattr__(self, "labels", labels)

    def to_dict(self) -> dict[str, object]:
        return {
            **{name: getattr(self, name) for name in self.__dataclass_fields__ if name != "labels"},
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, value: object) -> RepositoryRunnerSnapshot:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="repository runner snapshot",
        )
        return cls(
            **{key: item for key, item in row.items() if key != "labels"},
            labels=tuple(row["labels"]),
        )


@dataclass(frozen=True)
class RepositoryRunnerInventoryReceipt:
    repository: str
    total_count: int
    runners: tuple[RepositoryRunnerSnapshot, ...]
    response_sha256: str
    captured_at_utc: str
    schema_version: str = REPOSITORY_RUNNER_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.repository != REPOSITORY:
            raise ProviderRehearsalError("repository runner inventory repository differs")
        if type(self.total_count) is not int or self.total_count < 0:
            raise ProviderRehearsalError("repository runner total_count must be non-negative")
        rows = tuple(self.runners)
        if not all(isinstance(row, RepositoryRunnerSnapshot) for row in rows):
            raise ProviderRehearsalError("repository runner inventory must be typed")
        if (
            len(rows) != self.total_count
            or rows != tuple(sorted(rows, key=lambda row: row.runner_id))
            or len({row.runner_id for row in rows}) != len(rows)
            or len({row.runner_name for row in rows}) != len(rows)
        ):
            raise ProviderRehearsalError(
                "repository runner inventory is truncated, unordered, or duplicated"
            )
        _digest("repository runner response_sha256", self.response_sha256)
        _timestamp("repository runner captured_at_utc", self.captured_at_utc)
        if self.schema_version != REPOSITORY_RUNNER_INVENTORY_SCHEMA:
            raise ProviderRehearsalError("repository runner inventory schema differs")
        object.__setattr__(self, "runners", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name) for name in self.__dataclass_fields__ if name != "runners"
            },
            "runners": [row.to_dict() for row in self.runners],
        }

    @property
    def file_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()) + b"\n")

    @classmethod
    def from_file(cls, path: str | Path) -> RepositoryRunnerInventoryReceipt:
        value, encoded = _read_canonical_object(path, label="repository runner inventory")
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="repository runner inventory",
        )
        raw_runners = row["runners"]
        if not isinstance(raw_runners, list):
            raise ProviderRehearsalError("repository runner inventory runners must be an array")
        receipt = cls(
            **{key: item for key, item in row.items() if key != "runners"},
            runners=tuple(RepositoryRunnerSnapshot.from_dict(item) for item in raw_runners),
        )
        if encoded != _canonical_bytes(receipt.to_dict()) + b"\n":
            raise ProviderRehearsalError("repository runner inventory changed after parsing")
        return receipt


def capture_repository_runner_inventory(
    *,
    api: GitHubBytesApi,
    captured_at_utc: str,
) -> tuple[RepositoryRunnerInventoryReceipt, bytes]:
    """Capture one complete, bounded repository-runner page."""

    endpoint = f"repos/{REPOSITORY}/actions/runners?per_page=100"
    encoded = api.get_bytes(endpoint)
    response = _api_object(
        _decode_object(encoded, label="repository runners response"),
        label="repository runners response",
    )
    total_count = response.get("total_count")
    raw_rows = response.get("runners")
    if type(total_count) is not int or total_count < 0 or not isinstance(raw_rows, list):
        raise ProviderRehearsalError("repository runners response is malformed")
    if total_count != len(raw_rows) or total_count > 100:
        raise ProviderRehearsalError("repository runners response requires another page")
    rows: list[RepositoryRunnerSnapshot] = []
    for raw in raw_rows:
        runner = _api_object(raw, label="repository runner")
        raw_labels = runner.get("labels")
        if not isinstance(raw_labels, list):
            raise ProviderRehearsalError("repository runner labels are malformed")
        labels: list[str] = []
        for raw_label in raw_labels:
            label = _api_object(raw_label, label="repository runner label").get("name")
            labels.append(_text("repository runner label", label))
        rows.append(
            RepositoryRunnerSnapshot(
                runner_id=_positive("repository runner ID", runner.get("id")),
                runner_name=_text("repository runner name", runner.get("name")),
                operating_system=_text("repository runner operating system", runner.get("os")),
                status=_text("repository runner status", runner.get("status")),
                busy=runner.get("busy"),
                labels=tuple(sorted(labels, key=lambda item: item.encode("utf-8"))),
            )
        )
    receipt = RepositoryRunnerInventoryReceipt(
        repository=REPOSITORY,
        total_count=total_count,
        runners=tuple(sorted(rows, key=lambda row: row.runner_id)),
        response_sha256=_sha256(encoded),
        captured_at_utc=_timestamp("captured_at_utc", captured_at_utc),
    )
    return receipt, encoded


def prepare_rehearsal_runner_bootstrap(
    *,
    admission: RehearsalPhaseAdmission,
    runner_name: str,
    api: GitHubBytesApi,
    captured_at_utc: str,
) -> tuple[
    RehearsalRunnerBootstrapReceipt,
    RepositoryRunnerInventoryReceipt,
    bytes,
    Path,
]:
    """Bind one configured, stopped ephemeral runner before its listener starts."""

    name = _text("runner_name", runner_name)
    _, _, host_tools = _load_fixed_plan_components(admission)
    inventory, raw = capture_repository_runner_inventory(
        api=api,
        captured_at_utc=captured_at_utc,
    )
    matches = [row for row in inventory.runners if row.runner_name == name]
    if len(matches) != 1:
        raise ProviderRehearsalError(
            "configured rehearsal runner name is not a singleton in repository inventory"
        )
    runner = matches[0]
    required_labels = tuple(
        sorted(
            (*BASE_EXECUTE_RUNNER_LABELS, admission.runner_label),
            key=lambda item: item.encode("utf-8"),
        )
    )
    if (
        runner.operating_system != "macOS"
        or runner.status != "offline"
        or runner.busy is not False
        or runner.labels != required_labels
    ):
        raise ProviderRehearsalError(
            "configured rehearsal runner is not stopped with the exact queue labels"
        )
    receipt = RehearsalRunnerBootstrapReceipt(
        phase=admission.phase,
        repository=REPOSITORY,
        workflow_sha=admission.workflow_sha,
        runner_label=admission.runner_label,
        runner_id=runner.runner_id,
        runner_name=runner.runner_name,
        runner_group_id=None,
        runner_version=admission.runner_version,
        runner_archive_sha256=admission.runner_archive_sha256,
        repository_runner_inventory_sha256=inventory.file_sha256,
        ephemeral=True,
        disable_update=True,
        unattended=True,
        registered_at_utc=_timestamp("captured_at_utc", captured_at_utc),
    )
    output = Path(host_tools.controlled_root) / "rehearsal" / "runners" / admission.runner_label
    try:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise ProviderRehearsalError(
            "cannot create the fixed rehearsal bootstrap directory"
        ) from exc
    _write_exclusive(output / "repository-runners-api.raw.json", raw)
    _write_exclusive(
        output / "repository-runner-inventory.json",
        _canonical_bytes(inventory.to_dict()) + b"\n",
    )
    _write_exclusive(
        output / "bootstrap-receipt.json",
        _canonical_bytes(receipt.to_dict()) + b"\n",
    )
    return receipt, inventory, raw, output


def _api_object(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProviderRehearsalError(f"{label} must be one GitHub API object")
    return value


@dataclass(frozen=True)
class LiveRehearsalJobReceipt:
    phase: ProviderPhase
    repository: str
    workflow_path: str
    workflow_sha: str
    run_head_branch: str
    run_id: int
    run_attempt: int
    run_api_sha256: str
    jobs_api_sha256: str
    execute_job_id: int
    execute_job_name: str
    runner_id: int
    runner_name: str
    runner_group_id: int | None
    runner_labels: tuple[str, ...]
    runner_bootstrap_receipt_sha256: str
    verified_at_utc: str
    schema_version: str = REHEARSAL_LIVE_JOB_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in PHASES or self.repository != REPOSITORY:
            raise ProviderRehearsalError("live rehearsal phase or repository differs")
        if self.workflow_path != REHEARSAL_WORKFLOW_PATH:
            raise ProviderRehearsalError("live rehearsal workflow path differs")
        _git_commit("workflow_sha", self.workflow_sha)
        if _CANDIDATE_BRANCH.fullmatch(self.run_head_branch) is None:
            raise ProviderRehearsalError("live rehearsal head branch differs")
        for name in ("run_id", "run_attempt", "execute_job_id", "runner_id"):
            _positive(name, getattr(self, name))
        for name in (
            "run_api_sha256",
            "jobs_api_sha256",
            "runner_bootstrap_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.execute_job_name != REHEARSAL_JOB_NAMES[self.phase]:
            raise ProviderRehearsalError("live rehearsal job name differs")
        _text("runner_name", self.runner_name)
        if self.runner_group_id is not None:
            raise ProviderRehearsalError(
                "personal-repository rehearsal runner_group_id must be null"
            )
        labels = tuple(self.runner_labels)
        if (
            labels != tuple(sorted(labels, key=lambda item: item.encode("utf-8")))
            or len(labels) != len(set(labels))
            or not labels
        ):
            raise ProviderRehearsalError("live rehearsal labels are not unique byte order")
        _timestamp("verified_at_utc", self.verified_at_utc)
        if self.schema_version != REHEARSAL_LIVE_JOB_SCHEMA:
            raise ProviderRehearsalError("live rehearsal job schema differs")
        object.__setattr__(self, "runner_labels", labels)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "runner_labels"
            },
            "runner_labels": list(self.runner_labels),
        }

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> LiveRehearsalJobReceipt:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="live job receipt")
        return cls(
            **{key: item for key, item in row.items() if key != "runner_labels"},
            runner_labels=tuple(row["runner_labels"]),
        )


def verify_live_rehearsal_job(
    *,
    api: GitHubBytesApi,
    admission: RehearsalPhaseAdmission,
    bootstrap: RehearsalRunnerBootstrapReceipt,
    run_head_branch: str,
    verified_at_utc: str,
) -> tuple[LiveRehearsalJobReceipt, bytes, bytes]:
    """Read and bind the real run and Jobs API bytes before Docker starts."""

    if bootstrap.phase != admission.phase:
        raise ProviderRehearsalError("runner bootstrap phase differs from admission")
    exact_bootstrap = {
        "repository": REPOSITORY,
        "workflow_sha": admission.workflow_sha,
        "runner_label": admission.runner_label,
    }
    for name, expected in exact_bootstrap.items():
        if getattr(bootstrap, name) != expected:
            raise ProviderRehearsalError(f"runner bootstrap {name} differs")
    branch = _text("run_head_branch", run_head_branch)
    if _CANDIDATE_BRANCH.fullmatch(branch) is None:
        raise ProviderRehearsalError("rehearsal must run from c0-candidate/*")
    run_endpoint = (
        f"repos/{REPOSITORY}/actions/runs/{admission.run_id}/attempts/{admission.run_attempt}"
    )
    jobs_endpoint = f"{run_endpoint}/jobs?per_page=100"
    run_bytes = api.get_bytes(run_endpoint)
    jobs_bytes = api.get_bytes(jobs_endpoint)
    run = _api_object(_decode_object(run_bytes, label="GitHub run response"), label="run")
    repository = _api_object(run.get("repository"), label="run repository")
    actor = _api_object(run.get("actor"), label="run actor")
    triggering_actor = _api_object(run.get("triggering_actor"), label="triggering actor")
    exact_run = {
        "id": admission.run_id,
        "run_attempt": admission.run_attempt,
        "event": "workflow_dispatch",
        "status": "in_progress",
        "conclusion": None,
        "head_sha": admission.workflow_sha,
        "head_branch": branch,
        "path": REHEARSAL_WORKFLOW_PATH,
    }
    for name, expected in exact_run.items():
        if run.get(name) != expected:
            raise ProviderRehearsalError(f"live rehearsal run {name} differs")
    if (
        repository.get("full_name") != REPOSITORY
        or actor.get("login") != "mhdk1602"
        or triggering_actor.get("login") != "mhdk1602"
    ):
        raise ProviderRehearsalError("live rehearsal repository or actor differs")
    response = _api_object(
        _decode_object(jobs_bytes, label="GitHub jobs response"), label="jobs response"
    )
    jobs = response.get("jobs")
    if not isinstance(jobs, list):
        raise ProviderRehearsalError("GitHub jobs response lacks jobs")
    job_name = REHEARSAL_JOB_NAMES[admission.phase]
    matches = [
        _api_object(row, label="rehearsal job")
        for row in jobs
        if isinstance(row, Mapping) and row.get("name") == job_name
    ]
    if len(matches) != 1:
        raise ProviderRehearsalError("live rehearsal job is not a singleton")
    job = matches[0]
    labels_raw = job.get("labels")
    if not isinstance(labels_raw, list) or not all(type(item) is str for item in labels_raw):
        raise ProviderRehearsalError("live rehearsal labels are malformed")
    labels = tuple(sorted(labels_raw, key=lambda item: item.encode("utf-8")))
    required = {*BASE_EXECUTE_RUNNER_LABELS, admission.runner_label}
    if not required.issubset(labels):
        raise ProviderRehearsalError("live rehearsal job lacks its required labels")
    nonce_labels = [label for label in labels if label.startswith(REHEARSAL_RUNNER_LABEL_PREFIX)]
    production_labels = [label for label in labels if label.startswith("fractal-ann-confirmatory-")]
    if nonce_labels != [admission.runner_label] or production_labels:
        raise ProviderRehearsalError("live rehearsal job has another phase label")
    exact_job = {
        "name": job_name,
        "status": "in_progress",
        "conclusion": None,
        "run_id": admission.run_id,
        "run_attempt": admission.run_attempt,
        "runner_id": bootstrap.runner_id,
        "runner_name": bootstrap.runner_name,
        "runner_group_id": bootstrap.runner_group_id,
    }
    for name, expected in exact_job.items():
        if job.get(name) != expected:
            raise ProviderRehearsalError(f"live rehearsal job {name} differs")
    receipt = LiveRehearsalJobReceipt(
        phase=admission.phase,
        repository=REPOSITORY,
        workflow_path=REHEARSAL_WORKFLOW_PATH,
        workflow_sha=admission.workflow_sha,
        run_head_branch=branch,
        run_id=admission.run_id,
        run_attempt=admission.run_attempt,
        run_api_sha256=_sha256(run_bytes),
        jobs_api_sha256=_sha256(jobs_bytes),
        execute_job_id=_positive("execute job ID", job.get("id")),
        execute_job_name=job_name,
        runner_id=bootstrap.runner_id,
        runner_name=bootstrap.runner_name,
        runner_group_id=bootstrap.runner_group_id,
        runner_labels=labels,
        runner_bootstrap_receipt_sha256=bootstrap.file_sha256,
        verified_at_utc=_timestamp("verified_at_utc", verified_at_utc),
    )
    return receipt, run_bytes, jobs_bytes


def _load_fixed_plan_components(
    admission: RehearsalPhaseAdmission,
) -> tuple[Mapping[str, Any], bytes, PhaseHostToolContract]:
    try:
        plan = load_materialized_provider_phase_plan(admission.provider_plan_path)
    except ExecutionClaimError as exc:
        raise ProviderRehearsalError(f"fixed provider-plan loader rejected bytes: {exc}") from exc
    for name, expected in {
        "phase": admission.phase,
        "c1_commit": admission.c0_commit,
        "manifest_sha256": admission.manifest_sha256,
        "provider_plan_path": admission.provider_plan_path,
        "runtime_platform": admission.runtime_platform,
        "runtime_image_role": admission.runtime_image_role,
        "runtime_index_role": admission.runtime_index_role,
        "runner_archive_sha256": admission.runner_archive_sha256,
        "runner_group_id": admission.runner_group_id,
        "runner_version": admission.runner_version,
        "oci_index_digest": admission.candidate_image_index_digest,
        "oci_platform_manifest_digest": admission.candidate_platform_manifest_digest,
        "runtime_probe_receipt_sha256": admission.candidate_runtime_probe_receipt_sha256,
    }.items():
        if getattr(plan, name) != expected:
            raise ProviderRehearsalError(f"fixed provider plan {name} differs")
    if plan.plan_sha256 != admission.provider_plan_sha256:
        raise ProviderRehearsalError("fixed provider-plan semantic digest differs")
    if plan.file_sha256 != admission.provider_plan_file_sha256:
        raise ProviderRehearsalError("fixed provider-plan file digest differs")
    if plan.host_tools.contract_sha256 != admission.host_tools_contract_sha256:
        raise ProviderRehearsalError("fixed provider-plan host-tool contract digest differs")
    exact_tools = {
        "controlled_root": admission.host_controlled_root,
        "python_executable": admission.host_python_path,
        "python_executable_sha256": admission.host_python_file_sha256,
        "venv_root": admission.host_python_venv_root,
        "venv_tree_sha256": admission.host_python_venv_tree_sha256,
        "venv_symlink_inventory_sha256": (admission.host_python_venv_symlink_inventory_sha256),
        "python_import_root": admission.host_python_import_root,
        "python_import_tree_sha256": admission.host_python_import_tree_sha256,
        "python_launcher_sha256": admission.host_python_launcher_sha256,
        "python_package_content_sha256": (admission.host_python_package_content_sha256),
        "python_package_tree_sha256": admission.host_python_package_tree_sha256,
        "python_package_source_commit": admission.host_python_package_source_commit,
        "python_package_source_tree": admission.host_python_package_source_tree,
        "gh_executable": admission.host_gh_path,
        "gh_executable_sha256": admission.host_gh_file_sha256,
        "docker_executable": admission.host_docker_path,
        "docker_executable_sha256": admission.host_docker_file_sha256,
    }
    for name, expected in exact_tools.items():
        if getattr(plan.host_tools, name) != expected:
            raise ProviderRehearsalError(f"fixed provider-plan host tool {name} differs")
    return plan.to_dict(), plan.canonical_file_bytes(), plan.host_tools


def _safe_environment() -> tuple[dict[str, str], tuple[str, ...]]:
    removed = tuple(
        sorted(
            (name for name in FORBIDDEN_CONTAINER_ENVIRONMENT if name in os.environ),
            key=lambda item: item.encode("utf-8"),
        )
    )
    for name in FORBIDDEN_CONTAINER_ENVIRONMENT:
        os.environ.pop(name, None)
    environment = dict(os.environ)
    if FORBIDDEN_CONTAINER_ENVIRONMENT.intersection(environment):
        raise ProviderRehearsalError("GitHub token material survived environment scrubbing")
    return environment, removed


def _run_checked_bytes(
    argv: tuple[str, ...],
    *,
    environment: Mapping[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        env=environment,
        timeout=timeout,
    )
    if len(completed.stdout) > 4 * 1024 * 1024 or len(completed.stderr) > 4 * 1024 * 1024:
        raise ProviderRehearsalError("candidate image command output is oversized")
    return completed


@dataclass(frozen=True)
class RehearsalPhaseReceipt:
    admission: RehearsalPhaseAdmission
    live_job: LiveRehearsalJobReceipt
    runner_bootstrap_receipt_sha256: str
    host_tool_receipt: PhaseHostToolReceipt
    host_tool_receipt_sha256: str
    candidate_image_reference: str
    candidate_image_index_digest: str
    candidate_platform_manifest_digest: str
    runtime_platform: str
    runtime_image_role: str
    runtime_index_role: str
    pull_argv: tuple[str, ...]
    inspect_argv: tuple[str, ...]
    run_argv: tuple[str, ...]
    pull_stdout_sha256: str
    pull_stderr_sha256: str
    inspect_stdout_sha256: str
    inspect_stderr_sha256: str
    run_stdout_sha256: str
    run_stderr_sha256: str
    exit_status: int
    network_mode: str
    read_only_root: bool
    capabilities_dropped: bool
    no_new_privileges: bool
    study_mount_count: int
    token_names_scrubbed: tuple[str, ...]
    scientific_inputs_opened: bool
    provider_state_mutated: bool
    suite_attempt_id: None
    completed_at_utc: str
    schema_version: str = REHEARSAL_PHASE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.admission, RehearsalPhaseAdmission):
            raise ProviderRehearsalError("phase receipt admission must be typed")
        if not isinstance(self.live_job, LiveRehearsalJobReceipt):
            raise ProviderRehearsalError("phase receipt live job must be typed")
        if self.live_job.phase != self.admission.phase:
            raise ProviderRehearsalError("phase receipt live job differs")
        if not isinstance(self.host_tool_receipt, PhaseHostToolReceipt):
            raise ProviderRehearsalError("phase receipt host tool evidence must be typed")
        for name in (
            "runner_bootstrap_receipt_sha256",
            "host_tool_receipt_sha256",
            "pull_stdout_sha256",
            "pull_stderr_sha256",
            "inspect_stdout_sha256",
            "inspect_stderr_sha256",
            "run_stdout_sha256",
            "run_stderr_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.runner_bootstrap_receipt_sha256 != self.live_job.runner_bootstrap_receipt_sha256:
            raise ProviderRehearsalError(
                "phase receipt bootstrap digest differs from live-job evidence"
            )
        if self.host_tool_receipt.receipt_sha256 != self.host_tool_receipt_sha256:
            raise ProviderRehearsalError("host-tool receipt digest differs")
        if self.host_tool_receipt.contract_sha256 != self.admission.host_tools_contract_sha256:
            raise ProviderRehearsalError("host-tool receipt contract differs from the admission")
        exact_runtime = (
            self.admission.candidate_image_reference,
            self.admission.candidate_image_index_digest,
            self.admission.candidate_platform_manifest_digest,
            self.admission.runtime_platform,
            self.admission.runtime_image_role,
            self.admission.runtime_index_role,
        )
        observed_runtime = (
            self.candidate_image_reference,
            self.candidate_image_index_digest,
            self.candidate_platform_manifest_digest,
            self.runtime_platform,
            self.runtime_image_role,
            self.runtime_index_role,
        )
        if observed_runtime != exact_runtime:
            raise ProviderRehearsalError("phase receipt runtime binding differs")
        docker = self.host_tool_receipt.docker_resolved_executable
        expected_pull = (
            docker,
            "pull",
            "--platform",
            self.runtime_platform,
            self.candidate_image_reference,
        )
        expected_inspect = (docker, "image", "inspect", self.candidate_image_reference)
        expected_run = (
            docker,
            "run",
            "--rm",
            "--pull",
            "never",
            "--platform",
            self.runtime_platform,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65532:65532",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            self.candidate_image_reference,
            "--help",
        )
        if (
            tuple(self.pull_argv) != expected_pull
            or tuple(self.inspect_argv) != expected_inspect
            or tuple(self.run_argv) != expected_run
        ):
            raise ProviderRehearsalError("candidate self-check command differs")
        for argv in (self.pull_argv, self.inspect_argv, self.run_argv):
            if any(item in {"-e", "--env", "-v", "--volume", "--mount"} for item in argv):
                raise ProviderRehearsalError("candidate self-check introduces env or study mounts")
        if (
            self.exit_status != 0
            or self.network_mode != "none"
            or (self.read_only_root, self.capabilities_dropped, self.no_new_privileges)
            != (True, True, True)
            or self.study_mount_count != 0
            or self.scientific_inputs_opened is not False
            or self.provider_state_mutated is not False
            or self.suite_attempt_id is not None
        ):
            raise ProviderRehearsalError("candidate self-check crossed a production boundary")
        scrubbed = tuple(self.token_names_scrubbed)
        if scrubbed != tuple(sorted(scrubbed, key=lambda item: item.encode("utf-8"))):
            raise ProviderRehearsalError("scrubbed token names are not byte ordered")
        if not set(scrubbed).issubset(FORBIDDEN_CONTAINER_ENVIRONMENT):
            raise ProviderRehearsalError("phase receipt names an unknown scrubbed token")
        if "GH_TOKEN" not in scrubbed:
            raise ProviderRehearsalError(
                "phase receipt does not prove removal of the live GitHub job token"
            )
        _timestamp("completed_at_utc", self.completed_at_utc)
        if self.schema_version != REHEARSAL_PHASE_RECEIPT_SCHEMA:
            raise ProviderRehearsalError("phase rehearsal receipt schema differs")
        object.__setattr__(self, "pull_argv", tuple(self.pull_argv))
        object.__setattr__(self, "inspect_argv", tuple(self.inspect_argv))
        object.__setattr__(self, "run_argv", tuple(self.run_argv))
        object.__setattr__(self, "token_names_scrubbed", scrubbed)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "admission",
                    "host_tool_receipt",
                    "inspect_argv",
                    "live_job",
                    "pull_argv",
                    "run_argv",
                    "token_names_scrubbed",
                }
            },
            "admission": self.admission.to_dict(),
            "host_tool_receipt": self.host_tool_receipt.to_dict(),
            "inspect_argv": list(self.inspect_argv),
            "live_job": self.live_job.to_dict(),
            "pull_argv": list(self.pull_argv),
            "run_argv": list(self.run_argv),
            "token_names_scrubbed": list(self.token_names_scrubbed),
        }

    @property
    def file_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()) + b"\n")

    @classmethod
    def from_file(cls, path: str | Path) -> RehearsalPhaseReceipt:
        value, encoded = _read_canonical_object(path, label="phase rehearsal receipt")
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="phase receipt")
        result = cls(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "admission",
                    "host_tool_receipt",
                    "inspect_argv",
                    "live_job",
                    "pull_argv",
                    "run_argv",
                    "token_names_scrubbed",
                }
            },
            admission=RehearsalPhaseAdmission.from_dict(row["admission"]),
            host_tool_receipt=PhaseHostToolReceipt.from_dict(row["host_tool_receipt"]),
            inspect_argv=tuple(row["inspect_argv"]),
            live_job=LiveRehearsalJobReceipt.from_dict(row["live_job"]),
            pull_argv=tuple(row["pull_argv"]),
            run_argv=tuple(row["run_argv"]),
            token_names_scrubbed=tuple(row["token_names_scrubbed"]),
        )
        if encoded != _canonical_bytes(result.to_dict()) + b"\n":
            raise ProviderRehearsalError("phase rehearsal bytes changed after parsing")
        return result


def execute_rehearsal_phase(
    *,
    admission: RehearsalPhaseAdmission,
    run_head_branch: str,
    output_dir: str | Path,
    verified_at_utc: str,
    completed_at_utc: str | None,
) -> tuple[RehearsalPhaseReceipt, bytes, bytes, Mapping[str, bytes]]:
    """Verify one live runner and execute the sole networkless image help command."""

    output = Path(output_dir)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _, provider_plan_bytes, host_tools = _load_fixed_plan_components(admission)
    _write_exclusive(output / "provider-plan.json", provider_plan_bytes)
    bootstrap_path = (
        Path(host_tools.controlled_root)
        / "rehearsal"
        / "runners"
        / admission.runner_label
        / "bootstrap-receipt.json"
    )
    bootstrap = RehearsalRunnerBootstrapReceipt.from_file(bootstrap_path)
    inventory_path = bootstrap_path.parent / "repository-runner-inventory.json"
    raw_inventory_path = bootstrap_path.parent / "repository-runners-api.raw.json"
    inventory = RepositoryRunnerInventoryReceipt.from_file(inventory_path)
    try:
        raw_inventory = raw_inventory_path.read_bytes()
    except OSError as exc:
        raise ProviderRehearsalError("cannot read runner bootstrap API evidence") from exc
    inventory_rows = [
        row
        for row in inventory.runners
        if row.runner_id == bootstrap.runner_id and row.runner_name == bootstrap.runner_name
    ]
    if (
        inventory.file_sha256 != bootstrap.repository_runner_inventory_sha256
        or inventory.response_sha256 != _sha256(raw_inventory)
        or len(inventory_rows) != 1
    ):
        raise ProviderRehearsalError("runner bootstrap inventory binding differs")
    _write_exclusive(
        output / "bootstrap-receipt.json",
        _canonical_bytes(bootstrap.to_dict()) + b"\n",
    )
    _write_exclusive(
        output / "repository-runner-inventory.json",
        _canonical_bytes(inventory.to_dict()) + b"\n",
    )
    _write_exclusive(output / "repository-runners-api.raw.json", raw_inventory)
    if (
        bootstrap.runner_version != admission.runner_version
        or bootstrap.runner_archive_sha256 != admission.runner_archive_sha256
    ):
        raise ProviderRehearsalError("rehearsal runner bytes differ from the plan")
    api = GitHubCliBytesApi(host_tools.gh_executable, os.environ)
    live_job, run_bytes, jobs_bytes = verify_live_rehearsal_job(
        api=api,
        admission=admission,
        bootstrap=bootstrap,
        run_head_branch=run_head_branch,
        verified_at_utc=verified_at_utc,
    )
    environment, scrubbed = _safe_environment()
    try:
        host_receipt = verify_phase_host_tools(
            host_tools,
            probe_output_dir=output / "fresh-probes",
            verified_at_utc=verified_at_utc,
        )
    except ExecutionClaimError as exc:
        raise ProviderRehearsalError(f"host-tool verification failed: {exc}") from exc
    docker = host_receipt.docker_resolved_executable
    pull_argv = (
        docker,
        "pull",
        "--platform",
        admission.runtime_platform,
        admission.candidate_image_reference,
    )
    inspect_argv = (docker, "image", "inspect", admission.candidate_image_reference)
    run_argv = (
        docker,
        "run",
        "--rm",
        "--pull",
        "never",
        "--platform",
        admission.runtime_platform,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65532:65532",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",
        admission.candidate_image_reference,
        "--help",
    )
    pull = _run_checked_bytes(pull_argv, environment=environment, timeout=20 * 60)
    if pull.returncode != 0:
        raise ProviderRehearsalError("candidate image pull failed")
    inspect = _run_checked_bytes(inspect_argv, environment=environment, timeout=60)
    if inspect.returncode != 0:
        raise ProviderRehearsalError("candidate image inspection failed")
    try:
        inspection = json.loads(inspect.stdout.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderRehearsalError("candidate image inspection is not JSON") from exc
    if not isinstance(inspection, list) or len(inspection) != 1:
        raise ProviderRehearsalError("candidate image inspection is not a singleton")
    image = _api_object(inspection[0], label="candidate image inspection")
    expected_os, expected_arch = admission.runtime_platform.split("/", maxsplit=1)
    labels = _api_object(
        _api_object(image.get("Config"), label="image Config").get("Labels"),
        label="image labels",
    )
    if (
        image.get("Os") != expected_os
        or image.get("Architecture") != expected_arch
        or labels.get("io.fractal-ann.confirmatory.runtime-role") != admission.runtime_image_role
    ):
        raise ProviderRehearsalError("candidate image platform or runtime role differs")
    run = _run_checked_bytes(run_argv, environment=environment, timeout=5 * 60)
    blobs = {
        "jobs-api.raw.json": jobs_bytes,
        "run-api.raw.json": run_bytes,
        "pull.stderr": pull.stderr,
        "pull.stdout": pull.stdout,
        "inspect.stderr": inspect.stderr,
        "inspect.stdout": inspect.stdout,
        "run.stderr": run.stderr,
        "run.stdout": run.stdout,
    }
    for name, encoded in blobs.items():
        _write_exclusive(output / name, encoded)
    receipt = RehearsalPhaseReceipt(
        admission=admission,
        live_job=live_job,
        runner_bootstrap_receipt_sha256=bootstrap.file_sha256,
        host_tool_receipt=host_receipt,
        host_tool_receipt_sha256=host_receipt.receipt_sha256,
        candidate_image_reference=admission.candidate_image_reference,
        candidate_image_index_digest=admission.candidate_image_index_digest,
        candidate_platform_manifest_digest=admission.candidate_platform_manifest_digest,
        runtime_platform=admission.runtime_platform,
        runtime_image_role=admission.runtime_image_role,
        runtime_index_role=admission.runtime_index_role,
        pull_argv=pull_argv,
        inspect_argv=inspect_argv,
        run_argv=run_argv,
        pull_stdout_sha256=_sha256(pull.stdout),
        pull_stderr_sha256=_sha256(pull.stderr),
        inspect_stdout_sha256=_sha256(inspect.stdout),
        inspect_stderr_sha256=_sha256(inspect.stderr),
        run_stdout_sha256=_sha256(run.stdout),
        run_stderr_sha256=_sha256(run.stderr),
        exit_status=run.returncode,
        network_mode="none",
        read_only_root=True,
        capabilities_dropped=True,
        no_new_privileges=True,
        study_mount_count=0,
        token_names_scrubbed=scrubbed,
        scientific_inputs_opened=False,
        provider_state_mutated=False,
        suite_attempt_id=None,
        completed_at_utc=_timestamp(
            "completed_at_utc",
            _utc_now() if completed_at_utc is None else completed_at_utc,
        ),
    )
    return receipt, run_bytes, jobs_bytes, blobs


@dataclass(frozen=True)
class RehearsalAggregateReceipt:
    repository: str
    workflow_path: str
    workflow_sha: str
    run_head_branch: str
    run_id: int
    run_attempt: int
    c0_commit: str
    candidate_image_source_commit: str
    candidate_python_package_source_tree: str
    workflow_python_package_source_tree: str
    host_python_launcher_sha256: str
    workflow_python_launcher_sha256: str
    candidate_image_closure_file_sha256: str
    candidate_bootstrap_closure_sha256: str
    build_context_tree_sha256: str
    manifest_sha256: str
    plan_closure_sha256: str
    phase_receipt_file_sha256: Mapping[ProviderPhase, str]
    phase_execute_job_ids: Mapping[ProviderPhase, int]
    scientific_inputs_opened: bool
    provider_state_mutated: bool
    suite_attempt_id: None
    completed_at_utc: str
    schema_version: str = REHEARSAL_AGGREGATE_SCHEMA

    def __post_init__(self) -> None:
        if self.repository != REPOSITORY or self.workflow_path != REHEARSAL_WORKFLOW_PATH:
            raise ProviderRehearsalError("aggregate repository or workflow differs")
        _git_commit("workflow_sha", self.workflow_sha)
        _git_commit("c0_commit", self.c0_commit)
        _git_commit("candidate_image_source_commit", self.candidate_image_source_commit)
        _git_commit(
            "candidate_python_package_source_tree",
            self.candidate_python_package_source_tree,
        )
        _git_commit(
            "workflow_python_package_source_tree",
            self.workflow_python_package_source_tree,
        )
        if self.candidate_python_package_source_tree != self.workflow_python_package_source_tree:
            raise ProviderRehearsalError(
                "aggregate candidate package tree P differs from workflow package tree A"
            )
        for name in (
            "host_python_launcher_sha256",
            "workflow_python_launcher_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.host_python_launcher_sha256 != self.workflow_python_launcher_sha256:
            raise ProviderRehearsalError(
                "aggregate host launcher pin differs from workflow launcher source"
            )
        if self.workflow_sha != self.c0_commit:
            raise ProviderRehearsalError("aggregate workflow and candidate C0 commit differ")
        if _CANDIDATE_BRANCH.fullmatch(self.run_head_branch) is None:
            raise ProviderRehearsalError("aggregate branch differs")
        for name in ("run_id", "run_attempt"):
            _positive(name, getattr(self, name))
        for name in (
            "build_context_tree_sha256",
            "candidate_bootstrap_closure_sha256",
            "candidate_image_closure_file_sha256",
            "manifest_sha256",
            "plan_closure_sha256",
        ):
            _digest(name, getattr(self, name))
        if set(self.phase_receipt_file_sha256) != set(PHASES):
            raise ProviderRehearsalError("aggregate lacks exactly three receipt digests")
        if set(self.phase_execute_job_ids) != set(PHASES):
            raise ProviderRehearsalError("aggregate lacks exactly three job IDs")
        for phase in PHASES:
            _digest(f"{phase} receipt", self.phase_receipt_file_sha256[phase])
            _positive(f"{phase} job ID", self.phase_execute_job_ids[phase])
        if (
            self.scientific_inputs_opened is not False
            or self.provider_state_mutated is not False
            or self.suite_attempt_id is not None
        ):
            raise ProviderRehearsalError("aggregate crossed a production boundary")
        _timestamp("completed_at_utc", self.completed_at_utc)
        if self.schema_version != REHEARSAL_AGGREGATE_SCHEMA:
            raise ProviderRehearsalError("aggregate rehearsal schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"phase_execute_job_ids", "phase_receipt_file_sha256"}
            },
            "phase_execute_job_ids": dict(self.phase_execute_job_ids),
            "phase_receipt_file_sha256": dict(self.phase_receipt_file_sha256),
        }

    @property
    def file_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()) + b"\n")


def aggregate_rehearsal_receipts(
    *,
    phase_receipt_paths: Mapping[ProviderPhase, str | Path],
    candidate_closure: CandidateImageClosure,
    completed_at_utc: str,
) -> RehearsalAggregateReceipt:
    if set(phase_receipt_paths) != set(PHASES):
        raise ProviderRehearsalError("completion requires exactly three phase receipts")
    receipts = {
        phase: RehearsalPhaseReceipt.from_file(phase_receipt_paths[phase]) for phase in PHASES
    }
    first = receipts[ONLINE_PHASE]
    exact = (
        first.admission.workflow_sha,
        first.live_job.run_head_branch,
        first.live_job.run_id,
        first.live_job.run_attempt,
        first.admission.c0_commit,
        first.admission.candidate_image_source_commit,
        first.admission.host_python_package_source_tree,
        first.admission.workflow_python_package_source_tree,
        first.admission.host_python_launcher_sha256,
        first.admission.workflow_python_launcher_sha256,
        first.admission.candidate_image_closure_file_sha256,
        first.admission.candidate_bootstrap_closure_sha256,
        first.admission.build_context_tree_sha256,
        first.admission.manifest_sha256,
        first.admission.plan_closure_sha256,
    )
    for phase, receipt in receipts.items():
        if receipt.admission.phase != phase:
            raise ProviderRehearsalError("phase receipt appears under another phase")
        observed = (
            receipt.admission.workflow_sha,
            receipt.live_job.run_head_branch,
            receipt.live_job.run_id,
            receipt.live_job.run_attempt,
            receipt.admission.c0_commit,
            receipt.admission.candidate_image_source_commit,
            receipt.admission.host_python_package_source_tree,
            receipt.admission.workflow_python_package_source_tree,
            receipt.admission.host_python_launcher_sha256,
            receipt.admission.workflow_python_launcher_sha256,
            receipt.admission.candidate_image_closure_file_sha256,
            receipt.admission.candidate_bootstrap_closure_sha256,
            receipt.admission.build_context_tree_sha256,
            receipt.admission.manifest_sha256,
            receipt.admission.plan_closure_sha256,
        )
        if observed != exact:
            raise ProviderRehearsalError("phase receipts do not share one candidate run")
        expected_candidate = _candidate_binding(phase, candidate_closure)
        observed_candidate = (
            receipt.admission.candidate_image_reference,
            receipt.admission.candidate_image_index_digest,
            receipt.admission.candidate_platform_manifest_digest,
            receipt.admission.candidate_runtime_probe_receipt_sha256,
        )
        if observed_candidate != expected_candidate:
            raise ProviderRehearsalError(
                f"{phase} receipt differs from the admitted candidate image closure"
            )
    closure_exact = (
        candidate_closure.github_sha,
        candidate_closure.file_sha256,
        candidate_closure.bootstrap_closure_sha256,
        candidate_closure.build_context_tree_sha256,
        candidate_closure.candidate_branch,
    )
    if closure_exact != (exact[5], exact[10], exact[11], exact[12], exact[1]):
        raise ProviderRehearsalError(
            "candidate image closure differs from the aggregate source, bootstrap, or branch"
        )
    return RehearsalAggregateReceipt(
        repository=REPOSITORY,
        workflow_path=REHEARSAL_WORKFLOW_PATH,
        workflow_sha=exact[0],
        run_head_branch=exact[1],
        run_id=exact[2],
        run_attempt=exact[3],
        c0_commit=exact[4],
        candidate_image_source_commit=exact[5],
        candidate_python_package_source_tree=exact[6],
        workflow_python_package_source_tree=exact[7],
        host_python_launcher_sha256=exact[8],
        workflow_python_launcher_sha256=exact[9],
        candidate_image_closure_file_sha256=exact[10],
        candidate_bootstrap_closure_sha256=exact[11],
        build_context_tree_sha256=exact[12],
        manifest_sha256=exact[13],
        plan_closure_sha256=exact[14],
        phase_receipt_file_sha256={
            phase: receipt.file_sha256 for phase, receipt in receipts.items()
        },
        phase_execute_job_ids={
            phase: receipt.live_job.execute_job_id for phase, receipt in receipts.items()
        },
        scientific_inputs_opened=False,
        provider_state_mutated=False,
        suite_attempt_id=None,
        completed_at_utc=_timestamp("completed_at_utc", completed_at_utc),
    )


@dataclass(frozen=True)
class TagHeadBranchProbeReceipt:
    repository: str
    workflow_path: str
    workflow_sha: str
    github_ref: str
    tag_name: str
    run_id: int
    run_attempt: int
    observed_head_branch: str | None
    run_api_sha256: str
    jobs_api_sha256: str
    probe_job_id: int
    observed_at_utc: str
    schema_version: str = REHEARSAL_TAG_PROBE_SCHEMA

    def __post_init__(self) -> None:
        if self.repository != REPOSITORY or self.workflow_path != REHEARSAL_WORKFLOW_PATH:
            raise ProviderRehearsalError("tag probe repository or workflow differs")
        _git_commit("workflow_sha", self.workflow_sha)
        if _CANDIDATE_TAG.fullmatch(self.tag_name) is None:
            raise ProviderRehearsalError("tag probe name differs")
        if self.github_ref != f"refs/tags/{self.tag_name}":
            raise ProviderRehearsalError("tag probe ref differs")
        for name in ("run_id", "run_attempt", "probe_job_id"):
            _positive(name, getattr(self, name))
        if self.observed_head_branch is not None:
            _text("observed_head_branch", self.observed_head_branch)
        for name in ("run_api_sha256", "jobs_api_sha256"):
            _digest(name, getattr(self, name))
        _timestamp("observed_at_utc", self.observed_at_utc)
        if self.schema_version != REHEARSAL_TAG_PROBE_SCHEMA:
            raise ProviderRehearsalError("tag head-branch probe schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def file_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()) + b"\n")


def probe_tag_head_branch(
    *,
    api: GitHubBytesApi,
    workflow_sha: str,
    github_ref: str,
    run_id: int,
    run_attempt: int,
    observed_at_utc: str,
) -> tuple[TagHeadBranchProbeReceipt, bytes, bytes]:
    tag_ref = _text("github_ref", github_ref)
    if not tag_ref.startswith("refs/tags/"):
        raise ProviderRehearsalError("head-branch probe must run from a tag")
    tag_name = tag_ref.removeprefix("refs/tags/")
    run_endpoint = f"repos/{REPOSITORY}/actions/runs/{run_id}/attempts/{run_attempt}"
    jobs_endpoint = f"{run_endpoint}/jobs?per_page=100"
    run_bytes = api.get_bytes(run_endpoint)
    jobs_bytes = api.get_bytes(jobs_endpoint)
    run = _api_object(_decode_object(run_bytes, label="tag-probe run"), label="tag run")
    exact_run = {
        "id": run_id,
        "run_attempt": run_attempt,
        "event": "workflow_dispatch",
        "status": "in_progress",
        "conclusion": None,
        "head_sha": workflow_sha,
        "path": REHEARSAL_WORKFLOW_PATH,
    }
    for name, expected in exact_run.items():
        if run.get(name) != expected:
            raise ProviderRehearsalError(f"tag-probe run {name} differs")
    response = _api_object(_decode_object(jobs_bytes, label="tag-probe jobs"), label="tag jobs")
    jobs = response.get("jobs")
    if not isinstance(jobs, list):
        raise ProviderRehearsalError("tag-probe jobs response lacks jobs")
    matches = [
        _api_object(row, label="tag-probe job")
        for row in jobs
        if isinstance(row, Mapping) and row.get("name") == "probe-tag-head-branch"
    ]
    if len(matches) != 1:
        raise ProviderRehearsalError("tag-probe job is not a singleton")
    job = matches[0]
    if job.get("status") != "in_progress" or job.get("conclusion") is not None:
        raise ProviderRehearsalError("tag-probe job is not live")
    return (
        TagHeadBranchProbeReceipt(
            repository=REPOSITORY,
            workflow_path=REHEARSAL_WORKFLOW_PATH,
            workflow_sha=_git_commit("workflow_sha", workflow_sha),
            github_ref=tag_ref,
            tag_name=tag_name,
            run_id=_positive("run_id", run_id),
            run_attempt=_positive("run_attempt", run_attempt),
            observed_head_branch=run.get("head_branch"),
            run_api_sha256=_sha256(run_bytes),
            jobs_api_sha256=_sha256(jobs_bytes),
            probe_job_id=_positive("probe job ID", job.get("id")),
            observed_at_utc=_timestamp("observed_at_utc", observed_at_utc),
        ),
        run_bytes,
        jobs_bytes,
    )


@dataclass(frozen=True)
class RehearsalIncidentReceipt:
    repository: str
    workflow_path: str
    workflow_sha: str
    run_head_branch: str
    run_id: int
    run_attempt: int
    plan_result: str
    phase_results: Mapping[ProviderPhase, str]
    production_transition_published: bool
    provider_state_mutated: bool
    suite_attempt_id: None
    recorded_at_utc: str
    schema_version: str = REHEARSAL_INCIDENT_SCHEMA

    def __post_init__(self) -> None:
        if self.repository != REPOSITORY or self.workflow_path != REHEARSAL_WORKFLOW_PATH:
            raise ProviderRehearsalError("incident repository or workflow differs")
        _git_commit("workflow_sha", self.workflow_sha)
        if _CANDIDATE_BRANCH.fullmatch(self.run_head_branch) is None:
            raise ProviderRehearsalError("incident head branch differs")
        for name in ("run_id", "run_attempt"):
            _positive(name, getattr(self, name))
        results = {"cancelled", "failure", "skipped", "success"}
        if self.plan_result not in results:
            raise ProviderRehearsalError("incident plan result differs")
        if set(self.phase_results) != set(PHASES) or any(
            value not in results for value in self.phase_results.values()
        ):
            raise ProviderRehearsalError("incident phase results differ")
        if all(value == "success" for value in (self.plan_result, *self.phase_results.values())):
            raise ProviderRehearsalError("a successful rehearsal cannot emit an incident")
        if (
            self.production_transition_published is not False
            or self.provider_state_mutated is not False
            or self.suite_attempt_id is not None
        ):
            raise ProviderRehearsalError("incident crossed a production boundary")
        _timestamp("recorded_at_utc", self.recorded_at_utc)
        if self.schema_version != REHEARSAL_INCIDENT_SCHEMA:
            raise ProviderRehearsalError("rehearsal incident schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "phase_results"
            },
            "phase_results": dict(self.phase_results),
        }

    @property
    def file_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()) + b"\n")


def _append_github_outputs(path: Path, outputs: Mapping[str, str]) -> None:
    try:
        with path.open("a", encoding="utf-8", errors="strict", newline="\n") as stream:
            for key in sorted(outputs, key=lambda item: item.encode("utf-8")):
                value = outputs[key]
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
                    raise ProviderRehearsalError("GitHub output key is invalid")
                if type(value) is not str or "\n" in value or "\r" in value:
                    raise ProviderRehearsalError("GitHub output value is not one line")
                stream.write(f"{key}={value}\n")
    except OSError as exc:
        raise ProviderRehearsalError("cannot append GitHub outputs") from exc


PLAN_COMMON_OUTPUT_KEYS = frozenset(
    {
        "candidate_branch",
        "candidate_closure_path",
        "candidate_closure_sha256",
        "plan_closure_sha256",
        "plan_receipt_path",
        "plan_receipt_sha256",
    }
)
PLAN_PHASE_OUTPUT_FIELDS = frozenset(
    {
        "admission_json",
        "admission_sha256",
        "candidate_image_reference",
        "host_controlled_root",
        "host_docker_file_sha256",
        "host_docker_path",
        "host_gh_file_sha256",
        "host_gh_path",
        "host_python_file_sha256",
        "host_python_import_root",
        "host_python_import_tree_sha256",
        "host_python_launcher_sha256",
        "host_python_package_content_sha256",
        "host_python_package_source_commit",
        "host_python_package_source_tree",
        "host_python_package_tree_sha256",
        "host_python_path",
        "host_python_venv_root",
        "host_python_venv_symlink_inventory_sha256",
        "host_python_venv_tree_sha256",
        "provider_plan_file_sha256",
        "provider_plan_path",
        "provider_plan_sha256",
        "runner_label",
        "runtime_image_role",
        "runtime_index_role",
        "runtime_platform",
    }
)
PLAN_OUTPUT_KEYS = PLAN_COMMON_OUTPUT_KEYS | frozenset(
    f"{PHASE_OUTPUT_PREFIX[phase]}_{field}"
    for phase in PHASES
    for field in PLAN_PHASE_OUTPUT_FIELDS
)
EXECUTE_OUTPUT_KEYS = frozenset(
    {
        "execute_job_id",
        "jobs_api_path",
        "jobs_api_sha256",
        "phase_receipt_path",
        "phase_receipt_sha256",
        "run_api_path",
        "run_api_sha256",
    }
)
COMPLETE_OUTPUT_KEYS = frozenset(
    {
        "aggregate_receipt_path",
        "aggregate_receipt_sha256",
        "attestation_subject_path",
        "attestation_subject_sha256",
        "plan_closure_sha256",
    }
)
PROBE_OUTPUT_KEYS = frozenset(
    {
        "jobs_api_path",
        "jobs_api_sha256",
        "observed_head_branch",
        "probe_receipt_path",
        "probe_receipt_sha256",
        "run_api_path",
        "run_api_sha256",
    }
)
INCIDENT_OUTPUT_KEYS = frozenset({"incident_receipt_path", "incident_receipt_sha256"})
INVENTORY_OUTPUT_KEYS = frozenset(
    {
        "inventory_receipt_path",
        "inventory_receipt_sha256",
        "raw_response_path",
        "raw_response_sha256",
        "runner_total_count",
    }
)
BOOTSTRAP_OUTPUT_KEYS = frozenset(
    {
        "bootstrap_receipt_path",
        "bootstrap_receipt_sha256",
        "inventory_receipt_path",
        "inventory_receipt_sha256",
        "raw_response_path",
        "raw_response_sha256",
        "runner_id",
        "runner_name",
    }
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.provider_rehearsal",
        description="Run the artifact-only provider rehearsal without confirmatory state.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--manifest", required=True, type=Path)
    plan.add_argument("--c0-commit", required=True)
    plan.add_argument("--candidate-closure", required=True, type=Path)
    plan.add_argument("--workflow-sha", required=True)
    plan.add_argument("--workflow-source-root", required=True, type=Path)
    plan.add_argument("--run-id", required=True, type=int)
    plan.add_argument("--run-attempt", required=True, type=int)
    plan.add_argument("--output-dir", required=True, type=Path)
    plan.add_argument("--github-output", required=True, type=Path)

    execute = commands.add_parser("execute")
    execute.add_argument("--admission-json", required=True)
    execute.add_argument("--run-head-branch", required=True)
    execute.add_argument("--output-dir", required=True, type=Path)
    execute.add_argument("--github-output", required=True, type=Path)

    complete = commands.add_parser("complete")
    complete.add_argument("--online-receipt", required=True, type=Path)
    complete.add_argument("--label-release-receipt", required=True, type=Path)
    complete.add_argument("--analysis-receipt", required=True, type=Path)
    complete.add_argument("--candidate-closure", required=True, type=Path)
    complete.add_argument("--output-dir", required=True, type=Path)
    complete.add_argument("--github-output", required=True, type=Path)

    probe = commands.add_parser("probe-tag-head-branch")
    probe.add_argument("--gh-executable", required=True)
    probe.add_argument("--workflow-sha", required=True)
    probe.add_argument("--github-ref", required=True)
    probe.add_argument("--run-id", required=True, type=int)
    probe.add_argument("--run-attempt", required=True, type=int)
    probe.add_argument("--output-dir", required=True, type=Path)
    probe.add_argument("--github-output", required=True, type=Path)

    incident = commands.add_parser("incident")
    incident.add_argument("--workflow-sha", required=True)
    incident.add_argument("--run-head-branch", required=True)
    incident.add_argument("--run-id", required=True, type=int)
    incident.add_argument("--run-attempt", required=True, type=int)
    incident.add_argument("--plan-result", required=True)
    incident.add_argument("--online-result", required=True)
    incident.add_argument("--label-release-result", required=True)
    incident.add_argument("--analysis-result", required=True)
    incident.add_argument("--output-dir", required=True, type=Path)
    incident.add_argument("--github-output", required=True, type=Path)

    inventory = commands.add_parser("inventory")
    inventory.add_argument("--gh-executable", required=True)
    inventory.add_argument("--output-dir", required=True, type=Path)
    inventory.add_argument("--github-output", required=True, type=Path)

    bootstrap = commands.add_parser("prepare-runner-bootstrap")
    bootstrap.add_argument("--admission-json", required=True)
    bootstrap.add_argument("--runner-name", required=True)
    bootstrap.add_argument("--github-output", required=True, type=Path)
    return parser


def expected_cli_output_keys(arguments: argparse.Namespace) -> frozenset[str]:
    return {
        "plan": PLAN_OUTPUT_KEYS,
        "execute": EXECUTE_OUTPUT_KEYS,
        "complete": COMPLETE_OUTPUT_KEYS,
        "probe-tag-head-branch": PROBE_OUTPUT_KEYS,
        "incident": INCIDENT_OUTPUT_KEYS,
        "inventory": INVENTORY_OUTPUT_KEYS,
        "prepare-runner-bootstrap": BOOTSTRAP_OUTPUT_KEYS,
    }[arguments.command]


def _cli_plan(arguments: argparse.Namespace) -> Mapping[str, str]:
    closure = CandidateImageClosure.from_file(arguments.candidate_closure)
    admissions, materializations = build_rehearsal_admissions(
        manifest_path=arguments.manifest,
        c0_commit=arguments.c0_commit,
        candidate_closure=closure,
        workflow_sha=arguments.workflow_sha,
        run_id=arguments.run_id,
        run_attempt=arguments.run_attempt,
        materialization_root=arguments.output_dir / "hosted-plan-materializations",
        workflow_source_root=arguments.workflow_source_root,
    )
    if len(materializations) != 3:
        raise ProviderRehearsalError("plan did not materialize exactly three hosted copies")
    plan_receipt = {
        "admissions": {phase: admissions[phase].to_dict() for phase in PHASES},
        "candidate_closure_sha256": closure.file_sha256,
        "plan_closure_sha256": admissions[ONLINE_PHASE].plan_closure_sha256,
        "schema_version": "fractal-provider-rehearsal-admissions-v1",
    }
    plan_path, plan_sha = _write_receipt(
        arguments.output_dir / "rehearsal-admissions.json", plan_receipt
    )
    closure_copy = arguments.output_dir / "candidate-closure.json"
    _write_exclusive(closure_copy, _canonical_bytes(closure.to_dict()) + b"\n")
    outputs: dict[str, str] = {
        "candidate_branch": closure.candidate_branch,
        "candidate_closure_path": str(closure_copy),
        "candidate_closure_sha256": closure.file_sha256,
        "plan_closure_sha256": admissions[ONLINE_PHASE].plan_closure_sha256,
        "plan_receipt_path": str(plan_path),
        "plan_receipt_sha256": plan_sha,
    }
    for phase, admission in admissions.items():
        prefix = PHASE_OUTPUT_PREFIX[phase]
        outputs.update(
            {
                f"{prefix}_admission_json": _canonical_bytes(admission.to_dict()).decode("ascii"),
                f"{prefix}_admission_sha256": admission.admission_sha256,
                f"{prefix}_candidate_image_reference": admission.candidate_image_reference,
                f"{prefix}_host_controlled_root": admission.host_controlled_root,
                f"{prefix}_host_docker_file_sha256": admission.host_docker_file_sha256,
                f"{prefix}_host_docker_path": admission.host_docker_path,
                f"{prefix}_host_gh_file_sha256": admission.host_gh_file_sha256,
                f"{prefix}_host_gh_path": admission.host_gh_path,
                f"{prefix}_host_python_file_sha256": admission.host_python_file_sha256,
                f"{prefix}_host_python_import_root": admission.host_python_import_root,
                f"{prefix}_host_python_import_tree_sha256": (
                    admission.host_python_import_tree_sha256
                ),
                f"{prefix}_host_python_launcher_sha256": (admission.host_python_launcher_sha256),
                f"{prefix}_host_python_package_content_sha256": (
                    admission.host_python_package_content_sha256
                ),
                f"{prefix}_host_python_package_source_commit": (
                    admission.host_python_package_source_commit
                ),
                f"{prefix}_host_python_package_source_tree": (
                    admission.host_python_package_source_tree
                ),
                f"{prefix}_host_python_package_tree_sha256": (
                    admission.host_python_package_tree_sha256
                ),
                f"{prefix}_host_python_path": admission.host_python_path,
                f"{prefix}_host_python_venv_root": admission.host_python_venv_root,
                f"{prefix}_host_python_venv_symlink_inventory_sha256": (
                    admission.host_python_venv_symlink_inventory_sha256
                ),
                f"{prefix}_host_python_venv_tree_sha256": (admission.host_python_venv_tree_sha256),
                f"{prefix}_provider_plan_file_sha256": admission.provider_plan_file_sha256,
                f"{prefix}_provider_plan_path": admission.provider_plan_path,
                f"{prefix}_provider_plan_sha256": admission.provider_plan_sha256,
                f"{prefix}_runner_label": admission.runner_label,
                f"{prefix}_runtime_image_role": admission.runtime_image_role,
                f"{prefix}_runtime_index_role": admission.runtime_index_role,
                f"{prefix}_runtime_platform": admission.runtime_platform,
            }
        )
    return outputs


def _cli_execute(arguments: argparse.Namespace) -> Mapping[str, str]:
    admission = RehearsalPhaseAdmission.from_dict(
        _decode_object(arguments.admission_json.encode("ascii"), label="admission argument")
    )
    receipt, run_bytes, jobs_bytes, _ = execute_rehearsal_phase(
        admission=admission,
        run_head_branch=arguments.run_head_branch,
        output_dir=arguments.output_dir,
        verified_at_utc=_utc_now(),
        completed_at_utc=None,
    )
    receipt_path, receipt_sha = _write_receipt(
        arguments.output_dir / "phase-rehearsal-receipt.json", receipt.to_dict()
    )
    return {
        "execute_job_id": str(receipt.live_job.execute_job_id),
        "jobs_api_path": str(arguments.output_dir / "jobs-api.raw.json"),
        "jobs_api_sha256": _sha256(jobs_bytes),
        "phase_receipt_path": str(receipt_path),
        "phase_receipt_sha256": receipt_sha,
        "run_api_path": str(arguments.output_dir / "run-api.raw.json"),
        "run_api_sha256": _sha256(run_bytes),
    }


def _cli_complete(arguments: argparse.Namespace) -> Mapping[str, str]:
    closure = CandidateImageClosure.from_file(arguments.candidate_closure)
    receipt = aggregate_rehearsal_receipts(
        phase_receipt_paths={
            ONLINE_PHASE: arguments.online_receipt,
            LABEL_RELEASE_PHASE: arguments.label_release_receipt,
            ANALYSIS_PHASE: arguments.analysis_receipt,
        },
        candidate_closure=closure,
        completed_at_utc=_utc_now(),
    )
    path, digest = _write_receipt(
        arguments.output_dir / "provider-rehearsal-receipt.json", receipt.to_dict()
    )
    return {
        "aggregate_receipt_path": str(path),
        "aggregate_receipt_sha256": digest,
        "attestation_subject_path": str(path),
        "attestation_subject_sha256": digest,
        "plan_closure_sha256": receipt.plan_closure_sha256,
    }


def _cli_probe(arguments: argparse.Namespace) -> Mapping[str, str]:
    api = GitHubCliBytesApi(arguments.gh_executable, os.environ)
    receipt, run_bytes, jobs_bytes = probe_tag_head_branch(
        api=api,
        workflow_sha=arguments.workflow_sha,
        github_ref=arguments.github_ref,
        run_id=arguments.run_id,
        run_attempt=arguments.run_attempt,
        observed_at_utc=_utc_now(),
    )
    output = arguments.output_dir
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    run_path = output / "run-api.raw.json"
    jobs_path = output / "jobs-api.raw.json"
    _write_exclusive(run_path, run_bytes)
    _write_exclusive(jobs_path, jobs_bytes)
    path, digest = _write_receipt(output / "tag-head-branch-probe.json", receipt.to_dict())
    return {
        "jobs_api_path": str(jobs_path),
        "jobs_api_sha256": _sha256(jobs_bytes),
        "observed_head_branch": (
            "__null__" if receipt.observed_head_branch is None else receipt.observed_head_branch
        ),
        "probe_receipt_path": str(path),
        "probe_receipt_sha256": digest,
        "run_api_path": str(run_path),
        "run_api_sha256": _sha256(run_bytes),
    }


def _cli_incident(arguments: argparse.Namespace) -> Mapping[str, str]:
    receipt = RehearsalIncidentReceipt(
        repository=REPOSITORY,
        workflow_path=REHEARSAL_WORKFLOW_PATH,
        workflow_sha=arguments.workflow_sha,
        run_head_branch=arguments.run_head_branch,
        run_id=arguments.run_id,
        run_attempt=arguments.run_attempt,
        plan_result=arguments.plan_result,
        phase_results={
            ONLINE_PHASE: arguments.online_result,
            LABEL_RELEASE_PHASE: arguments.label_release_result,
            ANALYSIS_PHASE: arguments.analysis_result,
        },
        production_transition_published=False,
        provider_state_mutated=False,
        suite_attempt_id=None,
        recorded_at_utc=_utc_now(),
    )
    arguments.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    path, digest = _write_receipt(
        arguments.output_dir / "provider-rehearsal-incident.json", receipt.to_dict()
    )
    return {"incident_receipt_path": str(path), "incident_receipt_sha256": digest}


def _cli_inventory(arguments: argparse.Namespace) -> Mapping[str, str]:
    api = GitHubCliBytesApi(arguments.gh_executable, os.environ)
    receipt, raw = capture_repository_runner_inventory(
        api=api,
        captured_at_utc=_utc_now(),
    )
    output = arguments.output_dir
    try:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise ProviderRehearsalError("cannot create runner inventory output directory") from exc
    raw_path = output / "repository-runners-api.raw.json"
    _write_exclusive(raw_path, raw)
    receipt_path, receipt_sha = _write_receipt(
        output / "repository-runner-inventory.json",
        receipt.to_dict(),
    )
    return {
        "inventory_receipt_path": str(receipt_path),
        "inventory_receipt_sha256": receipt_sha,
        "raw_response_path": str(raw_path),
        "raw_response_sha256": receipt.response_sha256,
        "runner_total_count": str(receipt.total_count),
    }


def _cli_prepare_runner_bootstrap(arguments: argparse.Namespace) -> Mapping[str, str]:
    admission = RehearsalPhaseAdmission.from_dict(
        _decode_object(arguments.admission_json.encode("ascii"), label="admission argument")
    )
    _, _, host_tools = _load_fixed_plan_components(admission)
    api = GitHubCliBytesApi(host_tools.gh_executable, os.environ)
    receipt, inventory, raw, output = prepare_rehearsal_runner_bootstrap(
        admission=admission,
        runner_name=arguments.runner_name,
        api=api,
        captured_at_utc=_utc_now(),
    )
    receipt_path = output / "bootstrap-receipt.json"
    inventory_path = output / "repository-runner-inventory.json"
    raw_path = output / "repository-runners-api.raw.json"
    return {
        "bootstrap_receipt_path": str(receipt_path),
        "bootstrap_receipt_sha256": receipt.file_sha256,
        "inventory_receipt_path": str(inventory_path),
        "inventory_receipt_sha256": inventory.file_sha256,
        "raw_response_path": str(raw_path),
        "raw_response_sha256": _sha256(raw),
        "runner_id": str(receipt.runner_id),
        "runner_name": receipt.runner_name,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    handlers = {
        "plan": _cli_plan,
        "execute": _cli_execute,
        "complete": _cli_complete,
        "probe-tag-head-branch": _cli_probe,
        "incident": _cli_incident,
        "inventory": _cli_inventory,
        "prepare-runner-bootstrap": _cli_prepare_runner_bootstrap,
    }
    try:
        outputs = dict(handlers[arguments.command](arguments))
        expected = expected_cli_output_keys(arguments)
        if set(outputs) != expected:
            raise ProviderRehearsalError(
                "command output interface differs; "
                f"missing={sorted(expected - set(outputs))}, "
                f"unexpected={sorted(set(outputs) - expected)}"
            )
        _append_github_outputs(arguments.github_output, outputs)
    except ProviderRehearsalError as exc:
        print(f"provider-rehearsal error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
