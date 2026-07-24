"""Closed Docker launcher for the five sealed confirmatory corpus attempts.

The host-side launcher has three distinct phases:

* create one fresh named volume and one private output subdirectory without a
  shell;
* run a label-free preflight against a sentinel-bearing runtime-plan template,
  then replace only the predeclared host observations; and
* consume an external attempt marker before creating and attaching to one
  sealed container.

Docker containers, logs, inspection records, and the named volume are retained.
No function in this module removes them or retries a sealed invocation.
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import inspect as python_inspect
import json
import os
import re
import stat
import subprocess
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Protocol

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    read_secure_control_file,
    write_exclusive_receipt_bytes,
)
from .execution_claim import (
    ExecutionClaimError,
    VerifiedRunClaimCapability,
)
from .opa_runtime_binary import OpaRuntimeBinaryError, load_runtime_attestation_plan_template
from .runtime_attestation import (
    CANDIDATE_C0_COMMIT_SENTINEL,
    RuntimeArtifactMount,
    RuntimeAttestationError,
    RuntimeAttestationPlan,
    RuntimePreflightReceipt,
    capture_runtime_preflight,
    environment_sha256,
    loads_runtime_attestation_plan,
    loads_runtime_preflight_receipt,
    runtime_attestation_plan_template_file_bytes,
)

PREFLIGHT_LAUNCH_CONTRACT_SCHEMA = "fractal-docker-preflight-launch-contract-v1"
SEALED_LAUNCH_CONTRACT_SCHEMA = "fractal-docker-sealed-launch-contract-v1"
VOLUME_INITIALIZATION_RECEIPT_SCHEMA = "fractal-docker-output-volume-initialization-v2"
RUNTIME_PLAN_TRANSITION_SCHEMA = "fractal-runtime-plan-observation-transition-v1"
REGISTERED_PLAN_INSTANTIATION_SCHEMA = "fractal-registered-runtime-plan-instantiation-v1"
PRODUCTION_RUN_CLOSURE_BINDING_SCHEMA = "fractal-production-run-closure-binding-v1"
LAUNCHER_ATTEMPT_MARKER_SCHEMA = "fractal-docker-launcher-attempt-marker-v2"
OUTPUT_COPY_RECEIPT_SCHEMA = "fractal-docker-output-copy-receipt-v1"
CONTAINER_OUTPUT_INVENTORY_SCHEMA = "fractal-container-output-inventory-v1"
SEALED_LAUNCH_RECEIPT_SCHEMA = "fractal-docker-sealed-launch-receipt-v3"
DOCKER_ARGUMENT_RECORD_SCHEMA = "fractal-docker-argument-record-v1"
DOCKER_COMMAND_RESULT_SCHEMA = "fractal-docker-command-result-v1"
SEALED_EVIDENCE_INVENTORY_SCHEMA = "fractal-docker-sealed-evidence-inventory-v1"
SEALED_LAUNCH_FAILURE_ERROR_SCHEMA = "fractal-docker-sealed-launch-failure-error-v1"
SEALED_LAUNCH_FAILURE_RECEIPT_SCHEMA = "fractal-docker-sealed-launch-failure-v2"

PREFLIGHT_TEXT_SENTINEL = "__FRACTAL_PREFLIGHT_OBSERVED_V1__"
PREFLIGHT_INTEGER_SENTINEL = 1
PREFLIGHT_DIGEST_SENTINEL = "0" * 64
PREFLIGHT_OBSERVED_FIELDS = (
    "architecture",
    "cpu_model",
    "kernel_release",
    "logical_cpu_count",
    "memory_limit_bytes",
    "mount_namespace_sha256",
    "operating_system_id",
    "operating_system_version_id",
    "python_version",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OCI_IMAGE = re.compile(r"^[^\s/@]+(?:/[^\s/@]+)+@sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VOLUME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 8 * 1024 * 1024
_MAX_DOCKER_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_SECRET_BYTES = 4096
_MIN_SECRET_BYTES = 32
_PLATFORM = "linux/arm64"
_UID = 65532
_GID = 65532
_OUTPUT_ROOT = "/output"
_TMPFS_ROOT = "/tmp"
_PYTHON = "/opt/venv/bin/python"
_MODULE = "fractal_ann_diagnostics.sealed_container_launcher"
_PREFLIGHT_ARGV = (_PYTHON, "-m", _MODULE, "capture-preflight")
_SEALED_COMMAND_PREFIX = (
    _PYTHON,
    "-m",
    "fractal_ann_diagnostics.cli",
    "run-sealed-corpus",
    "--config",
)
_ALLOWED_TMPFS_FLAGS = ("nodev", "noexec", "nosuid")
PRODUCTION_RUN_CLOSURE_ROLE = "production-run-closure"
_VERIFIED_PRODUCTION_CLOSURE_CAPABILITY = object()
_LAUNCH_RECEIPT_FILENAME = "sealed-launch-receipt.json"
_LAUNCH_FAILURE_RECEIPT_FILENAME = "sealed-launch-failure-receipt.json"
_LAUNCH_FAILURE_ERROR_FILENAME = "sealed-launch-failure-error.json"
_CONTAINER_LABEL_PREFIX = "io.fractal-ann"


class SealedContainerLauncherError(ValueError):
    """The launcher contract, Docker evidence, or one-shot state is invalid."""


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
        raise SealedContainerLauncherError("launcher evidence must be canonical JSON") from exc


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise SealedContainerLauncherError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise SealedContainerLauncherError(
            f"{label} keys differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _parse_json_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    if type(encoded) is not bytes or not encoded or len(encoded) > _MAX_CONTROL_BYTES:
        raise SealedContainerLauncherError(f"{label} must be non-empty bounded bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SealedContainerLauncherError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise SealedContainerLauncherError(f"{label} contains non-finite value {value!r}")

    try:
        decoded = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise SealedContainerLauncherError(f"{label} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise SealedContainerLauncherError(f"{label} is not JSON: {exc.msg}") from exc
    if type(decoded) is not dict:
        raise SealedContainerLauncherError(f"{label} must contain one object")
    return decoded


def _text(name: str, value: object, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not value and not allow_empty) or value != value.strip():
        raise SealedContainerLauncherError(f"{name} must be canonical text")
    if unicodedata.normalize("NFC", value) != value:
        raise SealedContainerLauncherError(f"{name} must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SealedContainerLauncherError(f"{name} cannot contain control characters")
    return value


def _identifier(name: str, value: object) -> str:
    text = _text(name, value)
    if _IDENTIFIER.fullmatch(text) is None:
        raise SealedContainerLauncherError(f"{name} is not a canonical identifier")
    return text


def _sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SealedContainerLauncherError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_integer(name: str, value: object, *, maximum: int | None = None) -> int:
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        raise SealedContainerLauncherError(f"{name} must be a bounded positive integer")
    return value


def _canonical_host_path(name: str, value: object) -> Path:
    text = _text(name, value)
    path = Path(text)
    if (
        not path.is_absolute()
        or str(path) != text
        or ".." in path.parts
        or "," in text
        or "\x00" in text
    ):
        raise SealedContainerLauncherError(f"{name} must be a canonical absolute host path")
    return path


def _canonical_container_path(name: str, value: object) -> str:
    text = _text(name, value)
    path = PurePosixPath(text)
    if (
        not path.is_absolute()
        or str(path) != text
        or text == "/"
        or ".." in path.parts
        or "," in text
    ):
        raise SealedContainerLauncherError(f"{name} must be a canonical absolute container path")
    return text


def _relative_path(name: str, value: object) -> str:
    text = _text(name, value)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SealedContainerLauncherError(f"{name} must be a canonical relative POSIX path")
    return text


def _string_tuple(name: str, value: object, *, allow_empty_items: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SealedContainerLauncherError(f"{name} must be an array")
    return tuple(
        _text(f"{name}[{position}]", item, allow_empty=allow_empty_items)
        for position, item in enumerate(value)
    )


@dataclass(frozen=True)
class LauncherEnvironmentVariable:
    name: str
    value: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or _ENVIRONMENT_NAME.fullmatch(self.name) is None:
            raise SealedContainerLauncherError("environment name is invalid")
        _text(f"environment {self.name}", self.value, allow_empty=True)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, value: object) -> LauncherEnvironmentVariable:
        row = _closed(value, frozenset({"name", "value"}), label="environment row")
        return cls(name=row["name"], value=row["value"])


@dataclass(frozen=True)
class LauncherBindMount:
    source: str
    target: str
    role: str
    kind: Literal["file", "directory"]
    content_sha256: str
    attested_artifact: bool
    read_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", str(_canonical_host_path("mount source", self.source)))
        object.__setattr__(self, "target", _canonical_container_path("mount target", self.target))
        if "," in self.source or "," in self.target:
            raise SealedContainerLauncherError("mount paths cannot contain a comma")
        _identifier("mount role", self.role)
        if self.kind not in {"file", "directory"}:
            raise SealedContainerLauncherError("mount kind must be file or directory")
        _sha256("mount content_sha256", self.content_sha256)
        if type(self.attested_artifact) is not bool or type(self.read_only) is not bool:
            raise SealedContainerLauncherError("mount booleans must be exact booleans")
        if not self.read_only:
            raise SealedContainerLauncherError("every launcher bind mount must be read-only")

    def to_dict(self) -> dict[str, object]:
        return {
            "attested_artifact": self.attested_artifact,
            "content_sha256": self.content_sha256,
            "kind": self.kind,
            "read_only": self.read_only,
            "role": self.role,
            "source": self.source,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, value: object) -> LauncherBindMount:
        row = _closed(
            value,
            frozenset(
                {
                    "attested_artifact",
                    "content_sha256",
                    "kind",
                    "read_only",
                    "role",
                    "source",
                    "target",
                }
            ),
            label="bind mount",
        )
        return cls(**row)  # type: ignore[arg-type]

    def runtime_mount(self) -> RuntimeArtifactMount:
        if not self.attested_artifact:
            raise SealedContainerLauncherError("control-tree mount is not a runtime artifact")
        return RuntimeArtifactMount(
            root=self.target,
            role=self.role,
            kind=self.kind,
            artifact_sha256=self.content_sha256,
            read_only=True,
        )


@dataclass(frozen=True)
class LauncherGeometry:
    corpus_id: str
    oci_image_digest: str
    code_commit: str
    platform: str
    uid: int
    gid: int
    hostname: str
    environment: tuple[LauncherEnvironmentVariable, ...]
    memory_limit_bytes: int
    cpuset_cpus: tuple[int, ...]
    bind_mounts: tuple[LauncherBindMount, ...]
    control_mount_target: str
    runtime_plan_template_relative_path: str
    output_volume: str
    output_volume_subpath: str
    output_root: str
    copy_output_root: str
    tmpfs_root: str
    tmpfs_size_bytes: int
    tmpfs_mode: int
    tmpfs_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier("corpus_id", self.corpus_id)
        if (
            type(self.oci_image_digest) is not str
            or _OCI_IMAGE.fullmatch(self.oci_image_digest) is None
        ):
            raise SealedContainerLauncherError("OCI image must be digest-qualified")
        if type(self.code_commit) is not str or (
            _GIT_COMMIT.fullmatch(self.code_commit) is None
            and self.code_commit != CANDIDATE_C0_COMMIT_SENTINEL
        ):
            raise SealedContainerLauncherError(
                "code_commit must be one full commit or the candidate C0 sentinel"
            )
        if self.platform != _PLATFORM:
            raise SealedContainerLauncherError(f"platform must equal {_PLATFORM}")
        if self.uid != _UID or self.gid != _GID:
            raise SealedContainerLauncherError("sealed identity must equal 65532:65532")
        _identifier("hostname", self.hostname)
        environment = tuple(self.environment)
        if not environment or not all(
            isinstance(item, LauncherEnvironmentVariable) for item in environment
        ):
            raise SealedContainerLauncherError("environment must contain typed rows")
        expected_environment = tuple(
            sorted(environment, key=lambda item: item.name.encode("utf-8"))
        )
        if environment != expected_environment or len({item.name for item in environment}) != len(
            environment
        ):
            raise SealedContainerLauncherError("environment must be unique and name-sorted")
        if dict((item.name, item.value) for item in environment).get("HOSTNAME") != self.hostname:
            raise SealedContainerLauncherError("HOSTNAME must equal the fixed hostname")
        object.__setattr__(self, "environment", environment)
        _positive_integer("memory_limit_bytes", self.memory_limit_bytes)
        cpus = tuple(self.cpuset_cpus)
        if (
            not cpus
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
            or cpus != tuple(sorted(set(cpus)))
        ):
            raise SealedContainerLauncherError("cpuset_cpus must be unique sorted CPU indices")
        object.__setattr__(self, "cpuset_cpus", cpus)
        mounts = tuple(self.bind_mounts)
        if not mounts or not all(isinstance(item, LauncherBindMount) for item in mounts):
            raise SealedContainerLauncherError("bind_mounts must contain typed rows")
        expected_mounts = tuple(sorted(mounts, key=lambda item: item.target.encode("utf-8")))
        if mounts != expected_mounts or len({item.target for item in mounts}) != len(mounts):
            raise SealedContainerLauncherError("bind mounts must be unique and target-sorted")
        object.__setattr__(self, "bind_mounts", mounts)
        control_target = _canonical_container_path(
            "control_mount_target", self.control_mount_target
        )
        controls = [item for item in mounts if not item.attested_artifact]
        if (
            len(controls) != 1
            or controls[0].target != control_target
            or controls[0].kind != "directory"
            or controls[0].role != "runtime-control-tree"
        ):
            raise SealedContainerLauncherError(
                "one non-artifact runtime-control-tree mount is required"
            )
        object.__setattr__(self, "control_mount_target", control_target)
        closures = [item for item in mounts if item.role == PRODUCTION_RUN_CLOSURE_ROLE]
        if (
            len(closures) != 1
            or not closures[0].attested_artifact
            or closures[0].kind != "directory"
            or closures[0].source != closures[0].target
        ):
            raise SealedContainerLauncherError(
                "one source-equals-target attested production-run-closure is required"
            )
        object.__setattr__(
            self,
            "runtime_plan_template_relative_path",
            _relative_path(
                "runtime_plan_template_relative_path",
                self.runtime_plan_template_relative_path,
            ),
        )
        if (
            type(self.output_volume) is not str
            or _VOLUME_NAME.fullmatch(self.output_volume) is None
        ):
            raise SealedContainerLauncherError("output_volume is not a canonical Docker name")
        object.__setattr__(
            self,
            "output_volume_subpath",
            _relative_path("output_volume_subpath", self.output_volume_subpath),
        )
        if self.output_root != _OUTPUT_ROOT or self.tmpfs_root != _TMPFS_ROOT:
            raise SealedContainerLauncherError("writable roots must equal /output and /tmp")
        copy_root = _canonical_host_path("copy_output_root", self.copy_output_root)
        if copy_root.name != self.corpus_id or copy_root.parent.name != "online":
            raise SealedContainerLauncherError(
                "copy_output_root must equal an online/<corpus_id> suite path"
            )
        object.__setattr__(self, "copy_output_root", str(copy_root))
        _positive_integer("tmpfs_size_bytes", self.tmpfs_size_bytes)
        if self.tmpfs_mode != 0o1777:
            raise SealedContainerLauncherError("tmpfs_mode must equal 01777")
        flags = tuple(self.tmpfs_flags)
        if flags != _ALLOWED_TMPFS_FLAGS:
            raise SealedContainerLauncherError("tmpfs_flags differ from nodev,noexec,nosuid")
        object.__setattr__(self, "tmpfs_flags", flags)
        writable = {_OUTPUT_ROOT, _TMPFS_ROOT}
        for mount in mounts:
            if mount.target in writable or any(
                PurePosixPath(mount.target).is_relative_to(PurePosixPath(root))
                or PurePosixPath(root).is_relative_to(PurePosixPath(mount.target))
                for root in writable
            ):
                raise SealedContainerLauncherError("bind mounts overlap a writable root")

    @property
    def control_mount(self) -> LauncherBindMount:
        return next(item for item in self.bind_mounts if not item.attested_artifact)

    @property
    def production_run_closure_mount(self) -> LauncherBindMount:
        return next(item for item in self.bind_mounts if item.role == PRODUCTION_RUN_CLOSURE_ROLE)

    @property
    def environment_dict(self) -> dict[str, str]:
        return {item.name: item.value for item in self.environment}

    @property
    def cpuset_text(self) -> str:
        return ",".join(str(cpu) for cpu in self.cpuset_cpus)

    def to_dict(self) -> dict[str, object]:
        return {
            "bind_mounts": [item.to_dict() for item in self.bind_mounts],
            "code_commit": self.code_commit,
            "control_mount_target": self.control_mount_target,
            "corpus_id": self.corpus_id,
            "cpuset_cpus": list(self.cpuset_cpus),
            "environment": [item.to_dict() for item in self.environment],
            "gid": self.gid,
            "hostname": self.hostname,
            "memory_limit_bytes": self.memory_limit_bytes,
            "oci_image_digest": self.oci_image_digest,
            "output_root": self.output_root,
            "copy_output_root": self.copy_output_root,
            "output_volume": self.output_volume,
            "output_volume_subpath": self.output_volume_subpath,
            "platform": self.platform,
            "runtime_plan_template_relative_path": self.runtime_plan_template_relative_path,
            "tmpfs_flags": list(self.tmpfs_flags),
            "tmpfs_mode": self.tmpfs_mode,
            "tmpfs_root": self.tmpfs_root,
            "tmpfs_size_bytes": self.tmpfs_size_bytes,
            "uid": self.uid,
        }

    @classmethod
    def from_dict(cls, value: object) -> LauncherGeometry:
        fields = frozenset(
            {
                "bind_mounts",
                "code_commit",
                "copy_output_root",
                "control_mount_target",
                "corpus_id",
                "cpuset_cpus",
                "environment",
                "gid",
                "hostname",
                "memory_limit_bytes",
                "oci_image_digest",
                "output_root",
                "output_volume",
                "output_volume_subpath",
                "platform",
                "runtime_plan_template_relative_path",
                "tmpfs_flags",
                "tmpfs_mode",
                "tmpfs_root",
                "tmpfs_size_bytes",
                "uid",
            }
        )
        row = _closed(value, fields, label="launcher geometry")
        environments = row["environment"]
        mounts = row["bind_mounts"]
        cpus = row["cpuset_cpus"]
        flags = row["tmpfs_flags"]
        if not all(type(item) is list for item in (environments, mounts, cpus, flags)):
            raise SealedContainerLauncherError("geometry arrays are malformed")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key not in {"bind_mounts", "cpuset_cpus", "environment", "tmpfs_flags"}
            },
            bind_mounts=tuple(LauncherBindMount.from_dict(item) for item in mounts),
            cpuset_cpus=tuple(cpus),
            environment=tuple(LauncherEnvironmentVariable.from_dict(item) for item in environments),
            tmpfs_flags=tuple(flags),
        )


@dataclass(frozen=True)
class PreflightLaunchContract:
    geometry: LauncherGeometry
    argv: tuple[str, ...]
    provisional_control_tree_sha256: str
    provisional_plan_template_file_sha256: str
    schema_version: str = PREFLIGHT_LAUNCH_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PREFLIGHT_LAUNCH_CONTRACT_SCHEMA:
            raise SealedContainerLauncherError("preflight contract schema differs")
        if not isinstance(self.geometry, LauncherGeometry):
            raise SealedContainerLauncherError("preflight geometry must be typed")
        argv = _string_tuple("preflight argv", self.argv)
        if argv != _PREFLIGHT_ARGV:
            raise SealedContainerLauncherError("preflight argv differs from the fixed helper")
        object.__setattr__(self, "argv", argv)
        _sha256("provisional_control_tree_sha256", self.provisional_control_tree_sha256)
        _sha256(
            "provisional_plan_template_file_sha256",
            self.provisional_plan_template_file_sha256,
        )
        if self.geometry.control_mount.content_sha256 != self.provisional_control_tree_sha256:
            raise SealedContainerLauncherError("control mount digest differs from provisional tree")

    def to_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "geometry": self.geometry.to_dict(),
            "provisional_control_tree_sha256": self.provisional_control_tree_sha256,
            "provisional_plan_template_file_sha256": (self.provisional_plan_template_file_sha256),
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def contract_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @property
    def file_sha256(self) -> str:
        return _digest_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> PreflightLaunchContract:
        row = _closed(
            value,
            frozenset(
                {
                    "argv",
                    "geometry",
                    "provisional_control_tree_sha256",
                    "provisional_plan_template_file_sha256",
                    "schema_version",
                }
            ),
            label="preflight launch contract",
        )
        argv = row["argv"]
        if type(argv) is not list:
            raise SealedContainerLauncherError("preflight argv must be an array")
        return cls(
            geometry=LauncherGeometry.from_dict(row["geometry"]),
            argv=tuple(argv),
            provisional_control_tree_sha256=row["provisional_control_tree_sha256"],
            provisional_plan_template_file_sha256=row["provisional_plan_template_file_sha256"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class RuntimePlanTransitionReceipt:
    corpus_id: str
    allowed_observation_fields: tuple[str, ...]
    preflight_launcher_contract_sha256: str
    preflight_launcher_contract_file_sha256: str
    provisional_control_tree_sha256: str
    provisional_plan_template_file_sha256: str
    provisional_plan_template_semantic_sha256: str
    preflight_receipt_sha256: str
    preflight_receipt_file_sha256: str
    final_control_tree_sha256: str
    final_plan_template_file_sha256: str
    final_plan_template_semantic_sha256: str
    schema_version: str = RUNTIME_PLAN_TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_PLAN_TRANSITION_SCHEMA:
            raise SealedContainerLauncherError("runtime-plan transition schema differs")
        _identifier("corpus_id", self.corpus_id)
        fields = tuple(self.allowed_observation_fields)
        if fields != PREFLIGHT_OBSERVED_FIELDS:
            raise SealedContainerLauncherError("transition allowed fields differ")
        object.__setattr__(self, "allowed_observation_fields", fields)
        for name in (
            "preflight_launcher_contract_sha256",
            "preflight_launcher_contract_file_sha256",
            "provisional_control_tree_sha256",
            "provisional_plan_template_file_sha256",
            "provisional_plan_template_semantic_sha256",
            "preflight_receipt_sha256",
            "preflight_receipt_file_sha256",
            "final_control_tree_sha256",
            "final_plan_template_file_sha256",
            "final_plan_template_semantic_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.provisional_control_tree_sha256 == self.final_control_tree_sha256:
            raise SealedContainerLauncherError("transition did not change the control tree")
        if self.provisional_plan_template_file_sha256 == self.final_plan_template_file_sha256:
            raise SealedContainerLauncherError("transition did not change the plan template")

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_observation_fields": list(self.allowed_observation_fields),
            "corpus_id": self.corpus_id,
            "final_control_tree_sha256": self.final_control_tree_sha256,
            "final_plan_template_file_sha256": self.final_plan_template_file_sha256,
            "final_plan_template_semantic_sha256": self.final_plan_template_semantic_sha256,
            "preflight_launcher_contract_file_sha256": (
                self.preflight_launcher_contract_file_sha256
            ),
            "preflight_launcher_contract_sha256": self.preflight_launcher_contract_sha256,
            "preflight_receipt_file_sha256": self.preflight_receipt_file_sha256,
            "preflight_receipt_sha256": self.preflight_receipt_sha256,
            "provisional_control_tree_sha256": self.provisional_control_tree_sha256,
            "provisional_plan_template_file_sha256": (self.provisional_plan_template_file_sha256),
            "provisional_plan_template_semantic_sha256": (
                self.provisional_plan_template_semantic_sha256
            ),
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @property
    def file_sha256(self) -> str:
        return _digest_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> RuntimePlanTransitionReceipt:
        fields = frozenset(
            {
                "allowed_observation_fields",
                "corpus_id",
                "final_control_tree_sha256",
                "final_plan_template_file_sha256",
                "final_plan_template_semantic_sha256",
                "preflight_launcher_contract_file_sha256",
                "preflight_launcher_contract_sha256",
                "preflight_receipt_file_sha256",
                "preflight_receipt_sha256",
                "provisional_control_tree_sha256",
                "provisional_plan_template_file_sha256",
                "provisional_plan_template_semantic_sha256",
                "schema_version",
            }
        )
        row = _closed(value, fields, label="runtime-plan transition receipt")
        allowed = row["allowed_observation_fields"]
        if type(allowed) is not list:
            raise SealedContainerLauncherError("allowed observation fields must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "allowed_observation_fields"},
            allowed_observation_fields=tuple(allowed),
        )


@dataclass(frozen=True)
class ClosureFileBinding:
    """Exact file member admitted into the post-C1 production closure."""

    relative_path: str
    file_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _relative_path("closure file relative_path", self.relative_path),
        )
        _sha256("closure file SHA-256", self.file_sha256)
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise SealedContainerLauncherError("closure file byte_count must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "file_sha256": self.file_sha256,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> ClosureFileBinding:
        row = _closed(
            value,
            frozenset({"byte_count", "file_sha256", "relative_path"}),
            label="closure file binding",
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProductionRunClosureBindingReceipt:
    """Typed post-C1 substitution for one preflighted closure mount digest."""

    corpus_id: str
    manifest_sha256: str
    preflight_launcher_contract_sha256: str
    runtime_plan_transition_receipt_sha256: str
    closure_source: str
    closure_target: str
    provisional_closure_tree_sha256: str
    instantiated_closure_tree_sha256: str
    config_relative_path: str
    config_file_sha256: str
    workload_spec_relative_path: str
    workload_spec_file_sha256: str
    sealed_run_receipt_relative_path: str
    sealed_run_receipt_file_sha256: str
    entries: tuple[str, ...]
    files: tuple[ClosureFileBinding, ...]
    schema_version: str = PRODUCTION_RUN_CLOSURE_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCTION_RUN_CLOSURE_BINDING_SCHEMA:
            raise SealedContainerLauncherError("production closure binding schema differs")
        _identifier("corpus_id", self.corpus_id)
        for name in (
            "manifest_sha256",
            "preflight_launcher_contract_sha256",
            "runtime_plan_transition_receipt_sha256",
            "provisional_closure_tree_sha256",
            "instantiated_closure_tree_sha256",
            "config_file_sha256",
            "workload_spec_file_sha256",
            "sealed_run_receipt_file_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.provisional_closure_tree_sha256 == self.instantiated_closure_tree_sha256:
            raise SealedContainerLauncherError("production closure tree did not transition")
        source = str(_canonical_host_path("closure_source", self.closure_source))
        target = _canonical_container_path("closure_target", self.closure_target)
        if source != target or "," in source:
            raise SealedContainerLauncherError(
                "production closure source and target must be the same comma-free path"
            )
        object.__setattr__(self, "closure_source", source)
        object.__setattr__(self, "closure_target", target)
        for name in (
            "config_relative_path",
            "workload_spec_relative_path",
            "sealed_run_receipt_relative_path",
        ):
            object.__setattr__(self, name, _relative_path(name, getattr(self, name)))
        if PurePosixPath(self.sealed_run_receipt_relative_path).name != (
            f"{self.manifest_sha256}.json"
        ):
            raise SealedContainerLauncherError(
                "sealed run receipt filename differs from the registered manifest"
            )
        entries = tuple(
            _relative_path(f"closure entries[{position}]", item)
            for position, item in enumerate(self.entries)
        )
        if entries != tuple(sorted(entries, key=lambda item: item.encode("utf-8"))) or len(
            entries
        ) != len(set(entries)):
            raise SealedContainerLauncherError(
                "production closure entries must be unique and bytewise sorted"
            )
        object.__setattr__(self, "entries", entries)
        files = tuple(self.files)
        if not files or not all(isinstance(item, ClosureFileBinding) for item in files):
            raise SealedContainerLauncherError("production closure files must be typed")
        if files != tuple(sorted(files, key=lambda item: item.relative_path.encode("utf-8"))):
            raise SealedContainerLauncherError("production closure files are not sorted")
        if len({item.relative_path for item in files}) != len(files):
            raise SealedContainerLauncherError("production closure repeats a file")
        if any(item.relative_path not in entries for item in files):
            raise SealedContainerLauncherError("production closure file is absent from entries")
        object.__setattr__(self, "files", files)
        by_path = {item.relative_path: item.file_sha256 for item in files}
        expected = {
            self.config_relative_path: self.config_file_sha256,
            self.workload_spec_relative_path: self.workload_spec_file_sha256,
            self.sealed_run_receipt_relative_path: self.sealed_run_receipt_file_sha256,
        }
        if any(by_path.get(path) != digest for path, digest in expected.items()):
            raise SealedContainerLauncherError(
                "production closure named files differ from their member bindings"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "closure_source": self.closure_source,
            "closure_target": self.closure_target,
            "config_file_sha256": self.config_file_sha256,
            "config_relative_path": self.config_relative_path,
            "corpus_id": self.corpus_id,
            "entries": list(self.entries),
            "files": [item.to_dict() for item in self.files],
            "instantiated_closure_tree_sha256": self.instantiated_closure_tree_sha256,
            "manifest_sha256": self.manifest_sha256,
            "preflight_launcher_contract_sha256": self.preflight_launcher_contract_sha256,
            "provisional_closure_tree_sha256": self.provisional_closure_tree_sha256,
            "runtime_plan_transition_receipt_sha256": (self.runtime_plan_transition_receipt_sha256),
            "schema_version": self.schema_version,
            "sealed_run_receipt_file_sha256": self.sealed_run_receipt_file_sha256,
            "sealed_run_receipt_relative_path": self.sealed_run_receipt_relative_path,
            "workload_spec_file_sha256": self.workload_spec_file_sha256,
            "workload_spec_relative_path": self.workload_spec_relative_path,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @property
    def file_sha256(self) -> str:
        return _digest_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionRunClosureBindingReceipt:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="production closure binding",
        )
        entries = row["entries"]
        files = row["files"]
        if type(entries) is not list or type(files) is not list:
            raise SealedContainerLauncherError("production closure arrays are malformed")
        return cls(
            **{key: item for key, item in row.items() if key not in {"entries", "files"}},
            entries=tuple(entries),
            files=tuple(ClosureFileBinding.from_dict(item) for item in files),
        )


@dataclass(frozen=True)
class VerifiedProductionRunClosure:
    """File-backed authority returned only by the fixed finalization verifier."""

    _binding: ProductionRunClosureBindingReceipt
    _fresh_revalidator: Callable[[], ProductionRunClosureBindingReceipt]
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _VERIFIED_PRODUCTION_CLOSURE_CAPABILITY:
            raise SealedContainerLauncherError(
                "verified production closure must come from the fixed finalization verifier"
            )
        if not isinstance(self._binding, ProductionRunClosureBindingReceipt) or not callable(
            self._fresh_revalidator
        ):
            raise SealedContainerLauncherError("verified production closure is malformed")

    @property
    def binding(self) -> ProductionRunClosureBindingReceipt:
        return self._binding

    def assert_current(self) -> None:
        try:
            observed = self._fresh_revalidator()
        except SealedContainerLauncherError:
            raise
        except Exception as exc:
            raise SealedContainerLauncherError(
                "fresh production closure revalidation failed"
            ) from exc
        if observed != self._binding:
            raise SealedContainerLauncherError(
                "production closure changed after fixed finalization"
            )


def _mint_verified_production_run_closure(
    binding: ProductionRunClosureBindingReceipt,
    *,
    fresh_revalidator: Callable[[], ProductionRunClosureBindingReceipt],
) -> VerifiedProductionRunClosure:
    """Private bridge used by production_controls after full fixed rederivation."""

    verified = VerifiedProductionRunClosure(
        _binding=binding,
        _fresh_revalidator=fresh_revalidator,
        _capability=_VERIFIED_PRODUCTION_CLOSURE_CAPABILITY,
    )
    verified.assert_current()
    return verified


@dataclass(frozen=True)
class RegisteredPlanInstantiationReceipt:
    corpus_id: str
    manifest_sha256: str
    production_run_closure_binding_receipt_sha256: str
    runtime_plan_transition_receipt_sha256: str
    template_control_tree_sha256: str
    template_plan_file_sha256: str
    template_plan_semantic_sha256: str
    instantiated_control_tree_sha256: str
    instantiated_plan_file_sha256: str
    instantiated_plan_semantic_sha256: str
    instantiated_plan_relative_path: str
    schema_version: str = REGISTERED_PLAN_INSTANTIATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REGISTERED_PLAN_INSTANTIATION_SCHEMA:
            raise SealedContainerLauncherError("registered-plan instantiation schema differs")
        _identifier("corpus_id", self.corpus_id)
        for name in (
            "manifest_sha256",
            "production_run_closure_binding_receipt_sha256",
            "runtime_plan_transition_receipt_sha256",
            "template_control_tree_sha256",
            "template_plan_file_sha256",
            "template_plan_semantic_sha256",
            "instantiated_control_tree_sha256",
            "instantiated_plan_file_sha256",
            "instantiated_plan_semantic_sha256",
        ):
            _sha256(name, getattr(self, name))
        object.__setattr__(
            self,
            "instantiated_plan_relative_path",
            _relative_path("instantiated_plan_relative_path", self.instantiated_plan_relative_path),
        )
        if self.instantiated_plan_relative_path != "runtime-attestation-plan.json":
            raise SealedContainerLauncherError("instantiated plan filename differs")
        if self.template_control_tree_sha256 == self.instantiated_control_tree_sha256:
            raise SealedContainerLauncherError(
                "registered manifest did not change the control tree"
            )
        if self.template_plan_file_sha256 == self.instantiated_plan_file_sha256:
            raise SealedContainerLauncherError("registered manifest did not change the plan file")

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @property
    def file_sha256(self) -> str:
        return _digest_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> RegisteredPlanInstantiationReceipt:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="plan instantiation")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class SealedLaunchContract:
    geometry: LauncherGeometry
    argv: tuple[str, ...]
    preflight_launcher_contract_sha256: str
    preflight_receipt_sha256: str
    runtime_plan_transition_receipt_sha256: str
    registered_plan_instantiation_receipt_sha256: str
    production_run_closure_binding_receipt_sha256: str
    manifest_sha256: str
    instantiated_control_tree_sha256: str
    instantiated_plan_file_sha256: str
    instantiated_plan_semantic_sha256: str
    schema_version: str = SEALED_LAUNCH_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SEALED_LAUNCH_CONTRACT_SCHEMA:
            raise SealedContainerLauncherError("sealed launch contract schema differs")
        if not isinstance(self.geometry, LauncherGeometry):
            raise SealedContainerLauncherError("sealed geometry must be typed")
        argv = _string_tuple("sealed argv", self.argv, allow_empty_items=True)
        if len(argv) != 6 or argv[:5] != _SEALED_COMMAND_PREFIX or "" in argv:
            raise SealedContainerLauncherError("sealed argv differs from run-sealed-corpus")
        config_path = _canonical_container_path("sealed config path", argv[5])
        closure = PurePosixPath(self.geometry.production_run_closure_mount.target)
        if not PurePosixPath(config_path).is_relative_to(closure):
            raise SealedContainerLauncherError(
                "sealed config must reside below the production closure mount"
            )
        object.__setattr__(self, "argv", argv)
        for name in (
            "preflight_launcher_contract_sha256",
            "preflight_receipt_sha256",
            "runtime_plan_transition_receipt_sha256",
            "registered_plan_instantiation_receipt_sha256",
            "production_run_closure_binding_receipt_sha256",
            "manifest_sha256",
            "instantiated_control_tree_sha256",
            "instantiated_plan_file_sha256",
            "instantiated_plan_semantic_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.geometry.control_mount.content_sha256 != self.instantiated_control_tree_sha256:
            raise SealedContainerLauncherError(
                "sealed control mount differs from instantiated tree"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "geometry": self.geometry.to_dict(),
            "instantiated_control_tree_sha256": self.instantiated_control_tree_sha256,
            "instantiated_plan_file_sha256": self.instantiated_plan_file_sha256,
            "instantiated_plan_semantic_sha256": self.instantiated_plan_semantic_sha256,
            "manifest_sha256": self.manifest_sha256,
            "preflight_launcher_contract_sha256": self.preflight_launcher_contract_sha256,
            "preflight_receipt_sha256": self.preflight_receipt_sha256,
            "production_run_closure_binding_receipt_sha256": (
                self.production_run_closure_binding_receipt_sha256
            ),
            "registered_plan_instantiation_receipt_sha256": (
                self.registered_plan_instantiation_receipt_sha256
            ),
            "runtime_plan_transition_receipt_sha256": (self.runtime_plan_transition_receipt_sha256),
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def contract_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @property
    def file_sha256(self) -> str:
        return _digest_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> SealedLaunchContract:
        fields = frozenset(
            {
                "argv",
                "geometry",
                "instantiated_control_tree_sha256",
                "instantiated_plan_file_sha256",
                "instantiated_plan_semantic_sha256",
                "manifest_sha256",
                "preflight_launcher_contract_sha256",
                "preflight_receipt_sha256",
                "production_run_closure_binding_receipt_sha256",
                "registered_plan_instantiation_receipt_sha256",
                "runtime_plan_transition_receipt_sha256",
                "schema_version",
            }
        )
        row = _closed(value, fields, label="sealed launch contract")
        argv = row["argv"]
        if type(argv) is not list:
            raise SealedContainerLauncherError("sealed argv must be an array")
        return cls(
            geometry=LauncherGeometry.from_dict(row["geometry"]),
            argv=tuple(argv),
            **{key: item for key, item in row.items() if key not in {"argv", "geometry"}},
        )


@dataclass(frozen=True)
class DockerResult:
    returncode: int
    stdout: bytes
    stderr: bytes

    def __post_init__(self) -> None:
        if (
            type(self.returncode) is not int
            or type(self.stdout) is not bytes
            or type(self.stderr) is not bytes
        ):
            raise SealedContainerLauncherError("Docker result has invalid types")
        if len(self.stdout) + len(self.stderr) > _MAX_DOCKER_OUTPUT_BYTES:
            raise SealedContainerLauncherError("Docker output exceeds the retained bound")


@dataclass(frozen=True)
class DockerArgumentRecord:
    """The exact secret-free argument array persisted before one Docker mutation."""

    operation: str
    arguments: tuple[str, ...]
    schema_version: str = DOCKER_ARGUMENT_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DOCKER_ARGUMENT_RECORD_SCHEMA:
            raise SealedContainerLauncherError("Docker argument-record schema differs")
        _identifier("Docker operation", self.operation)
        arguments = _string_tuple("Docker arguments", self.arguments)
        if not arguments:
            raise SealedContainerLauncherError("Docker argument array cannot be empty")
        object.__setattr__(self, "arguments", arguments)

    def to_dict(self) -> dict[str, object]:
        return {
            "arguments": list(self.arguments),
            "operation": self.operation,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def record_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DockerArgumentRecord:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="Docker argument record")
        arguments = row["arguments"]
        if type(arguments) is not list:
            raise SealedContainerLauncherError("Docker arguments must be an array")
        return cls(
            operation=row["operation"],
            arguments=tuple(arguments),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class DockerCommandResultRecord:
    """Bound result of one Docker mutation, including its exact argument record."""

    operation: str
    argument_record_sha256: str
    returncode: int
    stdout_sha256: str
    stdout_byte_count: int
    stderr_sha256: str
    stderr_byte_count: int
    schema_version: str = DOCKER_COMMAND_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DOCKER_COMMAND_RESULT_SCHEMA:
            raise SealedContainerLauncherError("Docker result-record schema differs")
        _identifier("Docker operation", self.operation)
        _sha256("argument_record_sha256", self.argument_record_sha256)
        if type(self.returncode) is not int:
            raise SealedContainerLauncherError("Docker returncode must be an integer")
        for name in ("stdout_sha256", "stderr_sha256"):
            _sha256(name, getattr(self, name))
        for name in ("stdout_byte_count", "stderr_byte_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise SealedContainerLauncherError(f"{name} must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def record_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DockerCommandResultRecord:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="Docker result record")
        return cls(**row)  # type: ignore[arg-type]


class DockerRunner(Protocol):
    def run(
        self,
        arguments: Sequence[str],
        *,
        input_bytes: bytes | None = None,
    ) -> DockerResult: ...


@dataclass(frozen=True)
class SubprocessDockerRunner:
    executable: str = "docker"
    timeout_seconds: int = 7 * 24 * 60 * 60

    def run(self, arguments: Sequence[str], *, input_bytes: bytes | None = None) -> DockerResult:
        command = [self.executable, *arguments]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                input=input_bytes,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SealedContainerLauncherError("Docker command could not complete") from exc
        return DockerResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class VolumeInitializationReceipt:
    corpus_id: str
    preflight_launcher_contract_sha256: str
    output_volume: str
    output_volume_subpath: str
    volume_inspect_sha256: str
    initializer_container_id: str
    inspect_sha256: str
    initializer_start_returncode: int
    initializer_state_status: str
    initializer_exit_code: int
    initializer_oom_killed: bool
    stdout_sha256: str
    stdout_byte_count: int
    stderr_sha256: str
    stderr_byte_count: int
    schema_version: str = VOLUME_INITIALIZATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != VOLUME_INITIALIZATION_RECEIPT_SCHEMA:
            raise SealedContainerLauncherError("volume initialization schema differs")
        _identifier("corpus_id", self.corpus_id)
        _sha256("preflight_launcher_contract_sha256", self.preflight_launcher_contract_sha256)
        if _VOLUME_NAME.fullmatch(self.output_volume) is None:
            raise SealedContainerLauncherError("volume receipt name is invalid")
        _relative_path("output_volume_subpath", self.output_volume_subpath)
        if _CONTAINER_ID.fullmatch(self.initializer_container_id) is None:
            raise SealedContainerLauncherError("initializer container ID is invalid")
        for name in (
            "volume_inspect_sha256",
            "inspect_sha256",
            "stdout_sha256",
            "stderr_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.initializer_start_returncode != 0 or self.initializer_exit_code != 0:
            raise SealedContainerLauncherError("initializer must have a zero exit status")
        if self.initializer_state_status != "exited":
            raise SealedContainerLauncherError("initializer must retain terminal exited state")
        if type(self.initializer_oom_killed) is not bool or self.initializer_oom_killed:
            raise SealedContainerLauncherError("initializer cannot be OOM-killed")
        for name in ("stdout_byte_count", "stderr_byte_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise SealedContainerLauncherError(f"{name} must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "initializer_container_id": self.initializer_container_id,
            "initializer_exit_code": self.initializer_exit_code,
            "initializer_oom_killed": self.initializer_oom_killed,
            "initializer_start_returncode": self.initializer_start_returncode,
            "initializer_state_status": self.initializer_state_status,
            "inspect_sha256": self.inspect_sha256,
            "output_volume": self.output_volume,
            "output_volume_subpath": self.output_volume_subpath,
            "preflight_launcher_contract_sha256": self.preflight_launcher_contract_sha256,
            "schema_version": self.schema_version,
            "stderr_byte_count": self.stderr_byte_count,
            "stderr_sha256": self.stderr_sha256,
            "stdout_byte_count": self.stdout_byte_count,
            "stdout_sha256": self.stdout_sha256,
            "volume_inspect_sha256": self.volume_inspect_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> VolumeInitializationReceipt:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="volume receipt")
        return cls(**row)  # type: ignore[arg-type]


def _load_canonical(encoded: bytes, cls: Any, *, label: str) -> Any:
    value = cls.from_dict(_parse_json_object(encoded, label=label))
    if encoded != value.canonical_file_bytes():
        raise SealedContainerLauncherError(f"{label} bytes are not canonical")
    return value


def loads_preflight_launch_contract(encoded: bytes) -> PreflightLaunchContract:
    return _load_canonical(encoded, PreflightLaunchContract, label="preflight launch contract")


def load_preflight_launch_contract(path: str | Path) -> PreflightLaunchContract:
    return loads_preflight_launch_contract(_read_control(path, "preflight launch contract"))


def write_preflight_launch_contract(
    contract: PreflightLaunchContract,
    target: str | Path,
) -> None:
    """Publish a canonical preflight contract once through a no-follow path."""

    if not isinstance(contract, PreflightLaunchContract):
        raise SealedContainerLauncherError("preflight contract must be typed")
    _write_receipt(
        contract.canonical_file_bytes(),
        target,
        label="preflight launch contract",
    )


def loads_sealed_launch_contract(encoded: bytes) -> SealedLaunchContract:
    return _load_canonical(encoded, SealedLaunchContract, label="sealed launch contract")


def load_sealed_launch_contract(path: str | Path) -> SealedLaunchContract:
    return loads_sealed_launch_contract(_read_control(path, "sealed launch contract"))


def loads_runtime_plan_transition(encoded: bytes) -> RuntimePlanTransitionReceipt:
    return _load_canonical(encoded, RuntimePlanTransitionReceipt, label="runtime-plan transition")


def load_runtime_plan_transition(path: str | Path) -> RuntimePlanTransitionReceipt:
    return loads_runtime_plan_transition(_read_control(path, "runtime-plan transition"))


def loads_production_run_closure_binding(
    encoded: bytes,
) -> ProductionRunClosureBindingReceipt:
    return _load_canonical(
        encoded,
        ProductionRunClosureBindingReceipt,
        label="production closure binding",
    )


def load_production_run_closure_binding(
    path: str | Path,
) -> ProductionRunClosureBindingReceipt:
    source = _canonical_host_path("production closure binding path", path)
    binding = loads_production_run_closure_binding(
        _read_control(source, "production closure binding")
    )
    if PurePosixPath(str(source)).is_relative_to(PurePosixPath(binding.closure_source)):
        raise SealedContainerLauncherError(
            "production closure binding receipt must reside outside the bound closure"
        )
    return binding


def loads_registered_plan_instantiation(
    encoded: bytes,
) -> RegisteredPlanInstantiationReceipt:
    return _load_canonical(
        encoded,
        RegisteredPlanInstantiationReceipt,
        label="registered-plan instantiation",
    )


def load_registered_plan_instantiation(
    path: str | Path,
) -> RegisteredPlanInstantiationReceipt:
    return loads_registered_plan_instantiation(_read_control(path, "registered-plan instantiation"))


def loads_volume_initialization_receipt(encoded: bytes) -> VolumeInitializationReceipt:
    return _load_canonical(encoded, VolumeInitializationReceipt, label="volume receipt")


def load_volume_initialization_receipt(path: str | Path) -> VolumeInitializationReceipt:
    return loads_volume_initialization_receipt(_read_control(path, "volume receipt"))


def _read_control(path: str | Path, label: str) -> bytes:
    try:
        return read_secure_control_file(path, label=label)
    except ArtifactIntegrityError as exc:
        raise SealedContainerLauncherError(f"cannot read {label}: {exc}") from exc


def _write_receipt(encoded: bytes, target: str | Path, *, label: str) -> None:
    try:
        write_exclusive_receipt_bytes(encoded, target)
    except ArtifactIntegrityError as exc:
        raise SealedContainerLauncherError(f"cannot publish {label}: {exc}") from exc


def _content_digest(mount: LauncherBindMount) -> str:
    source = Path(mount.source)
    try:
        resolved = source.resolve(strict=True)
        metadata = source.lstat()
    except OSError as exc:
        raise SealedContainerLauncherError(f"cannot inspect mount role {mount.role!r}") from exc
    if resolved != source or source.is_symlink():
        raise SealedContainerLauncherError(f"mount role {mount.role!r} is not one real path")
    try:
        if mount.kind == "file":
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise SealedContainerLauncherError(
                    f"mount role {mount.role!r} is not a singly linked file"
                )
            return digest_regular_file(source, label=mount.role)
        if not stat.S_ISDIR(metadata.st_mode):
            raise SealedContainerLauncherError(f"mount role {mount.role!r} is not a directory")
        return digest_directory_tree(source).sha256
    except ArtifactIntegrityError as exc:
        raise SealedContainerLauncherError(f"cannot hash mount role {mount.role!r}: {exc}") from exc


def verify_launcher_mounts(geometry: LauncherGeometry) -> None:
    for mount in geometry.bind_mounts:
        if _content_digest(mount) != mount.content_sha256:
            raise SealedContainerLauncherError(f"mount role {mount.role!r} digest differs")


def _plan_path(contract: PreflightLaunchContract | SealedLaunchContract) -> Path:
    geometry = contract.geometry
    return Path(geometry.control_mount.source) / geometry.runtime_plan_template_relative_path


def _load_template(path: Path) -> RuntimeAttestationPlan:
    try:
        return load_runtime_attestation_plan_template(path)
    except OpaRuntimeBinaryError as exc:
        raise SealedContainerLauncherError(f"cannot load runtime-plan template: {exc}") from exc


def _require_provisional_sentinels(plan: RuntimeAttestationPlan) -> None:
    expected: dict[str, object] = {
        "architecture": PREFLIGHT_TEXT_SENTINEL,
        "cpu_model": PREFLIGHT_TEXT_SENTINEL,
        "kernel_release": PREFLIGHT_TEXT_SENTINEL,
        "logical_cpu_count": PREFLIGHT_INTEGER_SENTINEL,
        "memory_limit_bytes": PREFLIGHT_INTEGER_SENTINEL,
        "mount_namespace_sha256": PREFLIGHT_DIGEST_SENTINEL,
        "operating_system_id": PREFLIGHT_TEXT_SENTINEL,
        "operating_system_version_id": PREFLIGHT_TEXT_SENTINEL,
        "python_version": PREFLIGHT_TEXT_SENTINEL,
    }
    for field, value in expected.items():
        if getattr(plan, field) != value:
            raise SealedContainerLauncherError(
                f"provisional plan field {field!r} does not contain its fixed sentinel"
            )


def verify_preflight_receipt(
    contract: PreflightLaunchContract,
    receipt: RuntimePreflightReceipt,
) -> None:
    geometry = contract.geometry
    if receipt.launcher_contract_sha256 != contract.contract_sha256:
        raise SealedContainerLauncherError("preflight receipt names another launcher contract")
    expected = {
        "oci_image_digest": geometry.oci_image_digest,
        "code_commit": geometry.code_commit,
        "hostname": geometry.hostname,
        "memory_limit_bytes": geometry.memory_limit_bytes,
        "logical_cpu_count": len(geometry.cpuset_cpus),
        "environment_sha256": environment_sha256(geometry.environment_dict),
        "environment_allowlist": tuple(sorted(geometry.environment_dict)),
        "output_root": geometry.output_root,
        "tmpfs_root": geometry.tmpfs_root,
        "python_executable": _PYTHON,
        "effective_uid": geometry.uid,
        "effective_gid": geometry.gid,
    }
    for field, value in expected.items():
        if getattr(receipt, field) != value:
            raise SealedContainerLauncherError(f"preflight receipt {field} differs")
    if receipt.architecture not in {"aarch64", "arm64"}:
        raise SealedContainerLauncherError("preflight architecture does not match linux/arm64")
    expected_mounts = tuple(
        sorted(
            (item.runtime_mount() for item in geometry.bind_mounts if item.attested_artifact),
            key=lambda item: item.root.encode("utf-8"),
        )
    )
    if tuple(item.root for item in expected_mounts) != tuple(
        item.root for item in receipt.artifact_mounts
    ):
        raise SealedContainerLauncherError("preflight artifact roots differ from the contract")


def _final_plan(
    provisional: RuntimeAttestationPlan,
    receipt: RuntimePreflightReceipt,
) -> RuntimeAttestationPlan:
    return replace(
        provisional,
        architecture=receipt.architecture,
        cpu_model=receipt.cpu_model,
        kernel_release=receipt.kernel_release,
        logical_cpu_count=receipt.logical_cpu_count,
        memory_limit_bytes=receipt.memory_limit_bytes,
        mount_namespace_sha256=receipt.mount_namespace_sha256,
        operating_system_id=receipt.operating_system_id,
        operating_system_version_id=receipt.operating_system_version_id,
        python_version=receipt.python_version,
    )


def _plan_changed_fields(
    provisional: RuntimeAttestationPlan,
    final: RuntimeAttestationPlan,
) -> tuple[str, ...]:
    before = provisional.to_dict()
    after = final.to_dict()
    return tuple(sorted(key for key in before if before[key] != after[key]))


def materialize_runtime_plan_transition(
    contract: PreflightLaunchContract,
    receipt: RuntimePreflightReceipt,
) -> RuntimePlanTransitionReceipt:
    """Replace only the sentinel observations in the C1-pinnable template.

    Static verification completes before the first write. The plan-template
    replacement uses ``os.replace`` in the same directory. This function does
    not instantiate the registered manifest and cannot produce a sealed launch
    contract. That post-C1 operation has a separate typed receipt.
    """

    if not isinstance(contract, PreflightLaunchContract) or not isinstance(
        receipt, RuntimePreflightReceipt
    ):
        raise SealedContainerLauncherError("transition inputs must be typed")
    verify_launcher_mounts(contract.geometry)
    verify_preflight_receipt(contract, receipt)
    control_root = Path(contract.geometry.control_mount.source)
    if digest_directory_tree(control_root).sha256 != contract.provisional_control_tree_sha256:
        raise SealedContainerLauncherError("provisional control tree differs")
    plan_path = _plan_path(contract)
    if digest_regular_file(plan_path, label="provisional plan template") != (
        contract.provisional_plan_template_file_sha256
    ):
        raise SealedContainerLauncherError("provisional plan-template file differs")
    provisional = _load_template(plan_path)
    _require_provisional_sentinels(provisional)
    _verify_plan_static_bindings(provisional, contract.geometry, expected_argv=None)
    final = _final_plan(provisional, receipt)
    changed = _plan_changed_fields(provisional, final)
    if changed != PREFLIGHT_OBSERVED_FIELDS:
        raise SealedContainerLauncherError(
            f"runtime-plan transition changed forbidden fields: {changed}"
        )
    final_bytes = runtime_attestation_plan_template_file_bytes(final)
    temporary = plan_path.parent / f".{plan_path.name}.transition-{os.getpid()}"
    if os.path.lexists(temporary):
        raise SealedContainerLauncherError("runtime-plan transition staging path exists")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        offset = 0
        while offset < len(final_bytes):
            written = os.write(descriptor, final_bytes[offset:])
            if written <= 0:
                raise SealedContainerLauncherError("plan-template write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, plan_path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    final_loaded = _load_template(plan_path)
    if final_loaded != final:
        raise SealedContainerLauncherError("persisted final plan differs")
    final_tree = digest_directory_tree(control_root).sha256
    final_file = digest_regular_file(plan_path, label="final plan template")
    transition = RuntimePlanTransitionReceipt(
        corpus_id=contract.geometry.corpus_id,
        allowed_observation_fields=PREFLIGHT_OBSERVED_FIELDS,
        preflight_launcher_contract_sha256=contract.contract_sha256,
        preflight_launcher_contract_file_sha256=contract.file_sha256,
        provisional_control_tree_sha256=contract.provisional_control_tree_sha256,
        provisional_plan_template_file_sha256=contract.provisional_plan_template_file_sha256,
        provisional_plan_template_semantic_sha256=provisional.plan_sha256,
        preflight_receipt_sha256=receipt.receipt_sha256,
        preflight_receipt_file_sha256=receipt.file_sha256,
        final_control_tree_sha256=final_tree,
        final_plan_template_file_sha256=final_file,
        final_plan_template_semantic_sha256=final.plan_sha256,
    )
    verify_runtime_plan_transition(contract, receipt, transition)
    return transition


def verify_runtime_plan_transition(
    preflight: PreflightLaunchContract,
    receipt: RuntimePreflightReceipt,
    transition: RuntimePlanTransitionReceipt,
) -> RuntimeAttestationPlan:
    expected_transition = {
        "corpus_id": preflight.geometry.corpus_id,
        "preflight_launcher_contract_sha256": preflight.contract_sha256,
        "preflight_launcher_contract_file_sha256": preflight.file_sha256,
        "provisional_control_tree_sha256": preflight.provisional_control_tree_sha256,
        "provisional_plan_template_file_sha256": (preflight.provisional_plan_template_file_sha256),
        "preflight_receipt_sha256": receipt.receipt_sha256,
        "preflight_receipt_file_sha256": receipt.file_sha256,
    }
    for field, value in expected_transition.items():
        if getattr(transition, field) != value:
            raise SealedContainerLauncherError(f"transition {field} differs")
    plan_path = _plan_path(preflight)
    if digest_directory_tree(Path(preflight.geometry.control_mount.source)).sha256 != (
        transition.final_control_tree_sha256
    ):
        raise SealedContainerLauncherError("transitioned template control tree differs")
    if digest_regular_file(plan_path, label="final plan template") != (
        transition.final_plan_template_file_sha256
    ):
        raise SealedContainerLauncherError("final plan-template file differs")
    plan = _load_template(plan_path)
    if plan.plan_sha256 != transition.final_plan_template_semantic_sha256:
        raise SealedContainerLauncherError("final plan semantic digest differs")
    provisional_payload = plan.to_dict()
    for field in PREFLIGHT_OBSERVED_FIELDS:
        if field == "mount_namespace_sha256":
            provisional_payload[field] = PREFLIGHT_DIGEST_SENTINEL
        elif field in {"logical_cpu_count", "memory_limit_bytes"}:
            provisional_payload[field] = PREFLIGHT_INTEGER_SENTINEL
        else:
            provisional_payload[field] = PREFLIGHT_TEXT_SENTINEL
    provisional = RuntimeAttestationPlan.from_dict(provisional_payload)
    _require_provisional_sentinels(provisional)
    if provisional.plan_sha256 != transition.provisional_plan_template_semantic_sha256:
        raise SealedContainerLauncherError("provisional plan semantic digest differs")
    if _plan_changed_fields(provisional, plan) != PREFLIGHT_OBSERVED_FIELDS:
        raise SealedContainerLauncherError("final plan differs outside observed fields")
    if _final_plan(provisional, receipt) != plan:
        raise SealedContainerLauncherError("final plan values differ from the preflight")
    _verify_plan_static_bindings(plan, preflight.geometry, expected_argv=None)
    return plan


def verify_production_run_closure_binding(
    preflight: PreflightLaunchContract,
    transition: RuntimePlanTransitionReceipt,
    binding: ProductionRunClosureBindingReceipt,
) -> None:
    """Reproduce the only post-C1 artifact-mount digest substitution."""

    if not isinstance(binding, ProductionRunClosureBindingReceipt):
        raise SealedContainerLauncherError("production closure binding must be typed")
    closure = preflight.geometry.production_run_closure_mount
    expected = {
        "corpus_id": preflight.geometry.corpus_id,
        "preflight_launcher_contract_sha256": preflight.contract_sha256,
        "runtime_plan_transition_receipt_sha256": transition.receipt_sha256,
        "closure_source": closure.source,
        "closure_target": closure.target,
        "provisional_closure_tree_sha256": closure.content_sha256,
    }
    for field, value in expected.items():
        if getattr(binding, field) != value:
            raise SealedContainerLauncherError(f"production closure binding {field} differs")
    try:
        observed = digest_directory_tree(binding.closure_source)
    except ArtifactIntegrityError as exc:
        raise SealedContainerLauncherError("cannot hash the production closure") from exc
    if observed.sha256 != binding.instantiated_closure_tree_sha256:
        raise SealedContainerLauncherError("production closure tree differs from its binding")
    if observed.entries != binding.entries:
        raise SealedContainerLauncherError("production closure membership differs")
    expected_files = {item.relative_path: item for item in binding.files}
    observed_files: set[str] = set()
    closure_root = Path(binding.closure_source)
    for relative_path in observed.entries:
        source = closure_root / relative_path
        try:
            metadata = source.lstat()
        except OSError as exc:
            raise SealedContainerLauncherError(
                "cannot inspect a production closure member"
            ) from exc
        if stat.S_ISREG(metadata.st_mode):
            observed_files.add(relative_path)
            row = expected_files.get(relative_path)
            if row is None or row.byte_count != metadata.st_size:
                raise SealedContainerLauncherError(
                    "production closure file size differs from its binding"
                )
            try:
                file_sha256 = digest_regular_file(source, label="production closure file")
            except ArtifactIntegrityError as exc:
                raise SealedContainerLauncherError("cannot hash a production closure file") from exc
            if file_sha256 != row.file_sha256:
                raise SealedContainerLauncherError(
                    "production closure file differs from its binding"
                )
    if observed_files != set(expected_files):
        raise SealedContainerLauncherError("production closure file inventory differs")


def _instantiate_production_closure_mount(
    plan: RuntimeAttestationPlan,
    binding: ProductionRunClosureBindingReceipt,
) -> RuntimeAttestationPlan:
    closure_rows = [item for item in plan.mounts if item.role == PRODUCTION_RUN_CLOSURE_ROLE]
    if (
        len(closure_rows) != 1
        or closure_rows[0].root != binding.closure_target
        or closure_rows[0].kind != "directory"
        or closure_rows[0].artifact_sha256 != binding.provisional_closure_tree_sha256
    ):
        raise SealedContainerLauncherError(
            "C1 runtime template lacks the exact provisional production closure"
        )
    mounts = tuple(
        replace(item, artifact_sha256=binding.instantiated_closure_tree_sha256)
        if item.role == PRODUCTION_RUN_CLOSURE_ROLE
        else item
        for item in plan.mounts
    )
    return replace(plan, mounts=mounts)


def instantiate_registered_runtime_plan(
    preflight: PreflightLaunchContract,
    receipt: RuntimePreflightReceipt,
    transition: RuntimePlanTransitionReceipt,
    *,
    verified_closure: VerifiedProductionRunClosure,
) -> tuple[RegisteredPlanInstantiationReceipt, SealedLaunchContract]:
    """Materialize the canonical executable plan only after the C1 digest exists."""

    if not isinstance(verified_closure, VerifiedProductionRunClosure):
        raise SealedContainerLauncherError(
            "plan instantiation requires verified production closure authority"
        )
    verified_closure.assert_current()
    closure_binding = verified_closure.binding
    verify_production_run_closure_binding(preflight, transition, closure_binding)
    manifest = closure_binding.manifest_sha256
    template = verify_runtime_plan_transition(preflight, receipt, transition)
    instantiated = replace(
        _instantiate_production_closure_mount(template, closure_binding),
        manifest_sha256=manifest,
    )
    expected_config_path = str(
        PurePosixPath(closure_binding.closure_target) / closure_binding.config_relative_path
    )
    if instantiated.argv[-1] != expected_config_path:
        raise SealedContainerLauncherError(
            "runtime template config path differs from the production closure binding"
        )
    if instantiated.workload_sha256 != closure_binding.workload_spec_file_sha256:
        raise SealedContainerLauncherError(
            "runtime template workload differs from the C1 workload specification"
        )
    control_root = Path(preflight.geometry.control_mount.source)
    template_path = _plan_path(preflight)
    target = control_root / "runtime-attestation-plan.json"
    if os.path.lexists(target):
        raise SealedContainerLauncherError("instantiated runtime plan already exists")
    encoded = instantiated.canonical_file_bytes()
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise SealedContainerLauncherError("instantiated plan write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.unlink(template_path)
    except OSError as exc:
        raise SealedContainerLauncherError("cannot retire the pre-C1 plan template") from exc
    instantiated_file = digest_regular_file(target, label="instantiated runtime plan")
    instantiated_tree = digest_directory_tree(control_root).sha256
    instantiation = RegisteredPlanInstantiationReceipt(
        corpus_id=preflight.geometry.corpus_id,
        manifest_sha256=manifest,
        production_run_closure_binding_receipt_sha256=(closure_binding.receipt_sha256),
        runtime_plan_transition_receipt_sha256=transition.receipt_sha256,
        template_control_tree_sha256=transition.final_control_tree_sha256,
        template_plan_file_sha256=transition.final_plan_template_file_sha256,
        template_plan_semantic_sha256=transition.final_plan_template_semantic_sha256,
        instantiated_control_tree_sha256=instantiated_tree,
        instantiated_plan_file_sha256=instantiated_file,
        instantiated_plan_semantic_sha256=instantiated.plan_sha256,
        instantiated_plan_relative_path="runtime-attestation-plan.json",
    )
    instantiated_mounts = tuple(
        replace(item, content_sha256=instantiated_tree)
        if item.target == preflight.geometry.control_mount_target
        else replace(
            item,
            content_sha256=closure_binding.instantiated_closure_tree_sha256,
        )
        if item.role == PRODUCTION_RUN_CLOSURE_ROLE
        else item
        for item in preflight.geometry.bind_mounts
    )
    geometry = replace(preflight.geometry, bind_mounts=instantiated_mounts)
    sealed = SealedLaunchContract(
        geometry=geometry,
        argv=instantiated.argv,
        preflight_launcher_contract_sha256=preflight.contract_sha256,
        preflight_receipt_sha256=receipt.receipt_sha256,
        runtime_plan_transition_receipt_sha256=transition.receipt_sha256,
        registered_plan_instantiation_receipt_sha256=instantiation.receipt_sha256,
        production_run_closure_binding_receipt_sha256=(closure_binding.receipt_sha256),
        manifest_sha256=manifest,
        instantiated_control_tree_sha256=instantiated_tree,
        instantiated_plan_file_sha256=instantiated_file,
        instantiated_plan_semantic_sha256=instantiated.plan_sha256,
    )
    verify_sealed_transition(
        sealed,
        preflight,
        receipt,
        transition,
        instantiation,
        closure_binding,
    )
    return instantiation, sealed


def verify_sealed_transition(
    sealed: SealedLaunchContract,
    preflight: PreflightLaunchContract,
    receipt: RuntimePreflightReceipt,
    transition: RuntimePlanTransitionReceipt,
    instantiation: RegisteredPlanInstantiationReceipt,
    closure_binding: ProductionRunClosureBindingReceipt,
) -> RuntimeAttestationPlan:
    verify_production_run_closure_binding(preflight, transition, closure_binding)
    if sealed.preflight_launcher_contract_sha256 != preflight.contract_sha256:
        raise SealedContainerLauncherError("sealed launcher names another preflight contract")
    if sealed.preflight_receipt_sha256 != receipt.receipt_sha256:
        raise SealedContainerLauncherError("sealed launcher names another preflight receipt")
    if sealed.runtime_plan_transition_receipt_sha256 != transition.receipt_sha256:
        raise SealedContainerLauncherError("sealed launcher names another transition")
    if sealed.registered_plan_instantiation_receipt_sha256 != instantiation.receipt_sha256:
        raise SealedContainerLauncherError("sealed launcher names another plan instantiation")
    if (
        sealed.production_run_closure_binding_receipt_sha256 != closure_binding.receipt_sha256
        or instantiation.production_run_closure_binding_receipt_sha256
        != closure_binding.receipt_sha256
    ):
        raise SealedContainerLauncherError("sealed launcher names another closure binding")
    expected_instantiation = {
        "corpus_id": preflight.geometry.corpus_id,
        "manifest_sha256": sealed.manifest_sha256,
        "production_run_closure_binding_receipt_sha256": (closure_binding.receipt_sha256),
        "runtime_plan_transition_receipt_sha256": transition.receipt_sha256,
        "template_control_tree_sha256": transition.final_control_tree_sha256,
        "template_plan_file_sha256": transition.final_plan_template_file_sha256,
        "template_plan_semantic_sha256": transition.final_plan_template_semantic_sha256,
        "instantiated_control_tree_sha256": sealed.instantiated_control_tree_sha256,
        "instantiated_plan_file_sha256": sealed.instantiated_plan_file_sha256,
        "instantiated_plan_semantic_sha256": sealed.instantiated_plan_semantic_sha256,
    }
    for field, value in expected_instantiation.items():
        if getattr(instantiation, field) != value:
            raise SealedContainerLauncherError(f"plan instantiation {field} differs")
    _verify_same_geometry(preflight.geometry, sealed.geometry, closure_binding)
    verify_launcher_mounts(sealed.geometry)
    control_root = Path(sealed.geometry.control_mount.source)
    if digest_directory_tree(control_root).sha256 != sealed.instantiated_control_tree_sha256:
        raise SealedContainerLauncherError("instantiated control tree differs")
    plan_path = control_root / instantiation.instantiated_plan_relative_path
    if digest_regular_file(plan_path, label="instantiated runtime plan") != (
        sealed.instantiated_plan_file_sha256
    ):
        raise SealedContainerLauncherError("instantiated plan file differs")
    try:
        plan = loads_runtime_attestation_plan(_read_control(plan_path, "instantiated runtime plan"))
    except RuntimeAttestationError as exc:
        raise SealedContainerLauncherError("instantiated runtime plan is invalid") from exc
    if plan.plan_sha256 != sealed.instantiated_plan_semantic_sha256:
        raise SealedContainerLauncherError("instantiated plan semantic digest differs")
    if plan.manifest_sha256 != sealed.manifest_sha256:
        raise SealedContainerLauncherError("instantiated plan names another manifest")
    reversed_mounts = tuple(
        replace(item, artifact_sha256=closure_binding.provisional_closure_tree_sha256)
        if item.role == PRODUCTION_RUN_CLOSURE_ROLE
        else item
        for item in plan.mounts
    )
    template = replace(plan, manifest_sha256="0" * 64, mounts=reversed_mounts)
    template_bytes = runtime_attestation_plan_template_file_bytes(template)
    if _digest_bytes(template_bytes) != transition.final_plan_template_file_sha256:
        raise SealedContainerLauncherError("instantiated plan differs from the C1 template")
    if template.plan_sha256 != transition.final_plan_template_semantic_sha256:
        raise SealedContainerLauncherError("instantiated plan template semantics differ")
    provisional_payload = template.to_dict()
    for field in PREFLIGHT_OBSERVED_FIELDS:
        if field == "mount_namespace_sha256":
            provisional_payload[field] = PREFLIGHT_DIGEST_SENTINEL
        elif field in {"logical_cpu_count", "memory_limit_bytes"}:
            provisional_payload[field] = PREFLIGHT_INTEGER_SENTINEL
        else:
            provisional_payload[field] = PREFLIGHT_TEXT_SENTINEL
    provisional = RuntimeAttestationPlan.from_dict(provisional_payload)
    if provisional.plan_sha256 != transition.provisional_plan_template_semantic_sha256:
        raise SealedContainerLauncherError("provisional plan semantic digest differs")
    if _plan_changed_fields(provisional, template) != PREFLIGHT_OBSERVED_FIELDS:
        raise SealedContainerLauncherError("instantiated plan changed a forbidden field")
    if _final_plan(provisional, receipt) != template:
        raise SealedContainerLauncherError("instantiated plan observations differ")
    _verify_plan_static_bindings(plan, sealed.geometry, expected_argv=sealed.argv)
    return plan


def _verify_same_geometry(
    before: LauncherGeometry,
    after: LauncherGeometry,
    closure_binding: ProductionRunClosureBindingReceipt,
) -> None:
    left = before.to_dict()
    right = after.to_dict()
    left_mounts = left.pop("bind_mounts")
    right_mounts = right.pop("bind_mounts")
    if left != right:
        raise SealedContainerLauncherError("sealed launcher changed fixed geometry")
    assert isinstance(left_mounts, list) and isinstance(right_mounts, list)
    if len(left_mounts) != len(right_mounts):
        raise SealedContainerLauncherError("sealed launcher changed mount geometry")
    for old, new in zip(left_mounts, right_mounts, strict=True):
        old = dict(old)
        new = dict(new)
        old_digest = old.pop("content_sha256")
        new_digest = new.pop("content_sha256")
        if old != new:
            raise SealedContainerLauncherError("sealed launcher changed mount geometry")
        if old["target"] == before.control_mount_target:
            if old_digest == new_digest:
                raise SealedContainerLauncherError("control-tree digest did not transition")
        elif old["role"] == PRODUCTION_RUN_CLOSURE_ROLE:
            if (
                old_digest != closure_binding.provisional_closure_tree_sha256
                or new_digest != closure_binding.instantiated_closure_tree_sha256
            ):
                raise SealedContainerLauncherError(
                    "production closure digest differs from its typed transition"
                )
        elif old_digest != new_digest:
            raise SealedContainerLauncherError("an artifact mount digest changed")


def _verify_plan_static_bindings(
    plan: RuntimeAttestationPlan,
    geometry: LauncherGeometry,
    *,
    expected_argv: tuple[str, ...] | None,
) -> None:
    if (
        plan.oci_image_digest != geometry.oci_image_digest
        or plan.code_commit != geometry.code_commit
    ):
        raise SealedContainerLauncherError("runtime plan image or commit differs")
    if plan.python_binary.path != _PYTHON:
        raise SealedContainerLauncherError("runtime plan Python path differs")
    if plan.environment_allowlist != tuple(sorted(geometry.environment_dict)):
        raise SealedContainerLauncherError("runtime plan environment allowlist differs")
    if plan.environment_sha256 != environment_sha256(geometry.environment_dict):
        raise SealedContainerLauncherError("runtime plan environment digest differs")
    if expected_argv is not None and plan.argv != expected_argv:
        raise SealedContainerLauncherError("runtime plan argv differs from the launcher")
    expected_mounts = tuple(
        sorted(
            (item.runtime_mount() for item in geometry.bind_mounts if item.attested_artifact),
            key=lambda item: item.root.encode("utf-8"),
        )
    )
    if plan.mounts != expected_mounts:
        raise SealedContainerLauncherError("runtime plan artifact mounts differ")


def _docker_mount(mount: LauncherBindMount) -> str:
    return f"type=bind,src={mount.source},dst={mount.target},readonly"


def _output_mount(geometry: LauncherGeometry, *, read_only: bool = False) -> str:
    value = (
        f"type=volume,src={geometry.output_volume},dst={geometry.output_root},"
        f"volume-subpath={geometry.output_volume_subpath}"
    )
    return value + (",readonly" if read_only else "")


def _tmpfs(geometry: LauncherGeometry, *, root: bool = False) -> str:
    uid = 0 if root else geometry.uid
    gid = 0 if root else geometry.gid
    flags = ",".join(geometry.tmpfs_flags)
    return (
        f"{geometry.tmpfs_root}:rw,{flags},size={geometry.tmpfs_size_bytes},"
        f"uid={uid},gid={gid},mode={geometry.tmpfs_mode:o}"
    )


def _container_labels(
    geometry: LauncherGeometry,
    *,
    role: str,
    authority_sha256: str,
) -> dict[str, str]:
    _identifier("container role", role)
    _sha256("container authority", authority_sha256)
    return {
        f"{_CONTAINER_LABEL_PREFIX}.authority-sha256": authority_sha256,
        f"{_CONTAINER_LABEL_PREFIX}.corpus-id": geometry.corpus_id,
        f"{_CONTAINER_LABEL_PREFIX}.role": role,
    }


def _append_labels(arguments: list[str], labels: Mapping[str, str]) -> None:
    for name in sorted(labels, key=lambda item: item.encode("utf-8")):
        arguments.extend(("--label", f"{name}={labels[name]}"))


def _create_arguments(
    geometry: LauncherGeometry,
    *,
    name: str,
    argv: tuple[str, ...],
    role: str,
    authority_sha256: str,
) -> list[str]:
    arguments = [
        "create",
        "--name",
        name,
        "--platform",
        geometry.platform,
        "--user",
        f"{geometry.uid}:{geometry.gid}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--network",
        "none",
        "--memory",
        str(geometry.memory_limit_bytes),
        "--cpuset-cpus",
        geometry.cpuset_text,
        "--hostname",
        geometry.hostname,
    ]
    _append_labels(
        arguments,
        _container_labels(geometry, role=role, authority_sha256=authority_sha256),
    )
    for row in geometry.environment:
        arguments.extend(("--env", f"{row.name}={row.value}"))
    arguments.extend(("--tmpfs", _tmpfs(geometry)))
    for mount in geometry.bind_mounts:
        arguments.extend(("--mount", _docker_mount(mount)))
    arguments.extend(
        (
            "--mount",
            _output_mount(geometry),
            "--entrypoint",
            argv[0],
            geometry.oci_image_digest,
            *argv[1:],
        )
    )
    return arguments


def _run_checked(
    docker: DockerRunner,
    arguments: Sequence[str],
    *,
    label: str,
    input_bytes: bytes | None = None,
) -> DockerResult:
    result = docker.run(tuple(arguments), input_bytes=input_bytes)
    if result.returncode != 0:
        detail = result.stderr[:2048].decode("utf-8", errors="replace").strip()
        raise SealedContainerLauncherError(f"{label} failed: {detail}")
    return result


def _execute_mutating_docker_command(
    docker: DockerRunner,
    arguments: Sequence[str],
    *,
    audit_root: Path,
    operation: str,
    input_bytes: bytes | None = None,
    forbidden_bytes: bytes | None = None,
) -> tuple[DockerResult, DockerArgumentRecord, DockerCommandResultRecord]:
    """Persist argv first, then bind the command result and retained streams."""

    argument_record = DockerArgumentRecord(operation=operation, arguments=tuple(arguments))
    _write_receipt(
        argument_record.canonical_file_bytes(),
        audit_root / f"{operation}-docker-argv.json",
        label=f"{operation} Docker argument record",
    )
    result = docker.run(argument_record.arguments, input_bytes=input_bytes)
    result_record = DockerCommandResultRecord(
        operation=operation,
        argument_record_sha256=argument_record.record_sha256,
        returncode=result.returncode,
        stdout_sha256=_digest_bytes(result.stdout),
        stdout_byte_count=len(result.stdout),
        stderr_sha256=_digest_bytes(result.stderr),
        stderr_byte_count=len(result.stderr),
    )
    if forbidden_bytes is not None and (
        forbidden_bytes in result.stdout or forbidden_bytes in result.stderr
    ):
        _write_receipt(
            result_record.canonical_file_bytes(),
            audit_root / f"{operation}-docker-result.json",
            label=f"{operation} Docker result record",
        )
        raise SealedContainerLauncherError(
            "secret bytes appeared in a Docker command result; streams were not retained"
        )
    _persist_bytes(
        audit_root / f"{operation}-docker-stdout.log",
        result.stdout,
        label=f"{operation} Docker stdout",
    )
    _persist_bytes(
        audit_root / f"{operation}-docker-stderr.log",
        result.stderr,
        label=f"{operation} Docker stderr",
    )
    _write_receipt(
        result_record.canonical_file_bytes(),
        audit_root / f"{operation}-docker-result.json",
        label=f"{operation} Docker result record",
    )
    return result, argument_record, result_record


def _run_checked_mutation(
    docker: DockerRunner,
    arguments: Sequence[str],
    *,
    audit_root: Path,
    operation: str,
    label: str,
    input_bytes: bytes | None = None,
    forbidden_bytes: bytes | None = None,
) -> DockerResult:
    result, _, _ = _execute_mutating_docker_command(
        docker,
        arguments,
        audit_root=audit_root,
        operation=operation,
        input_bytes=input_bytes,
        forbidden_bytes=forbidden_bytes,
    )
    if result.returncode != 0:
        detail = result.stderr[:2048].decode("utf-8", errors="replace").strip()
        raise SealedContainerLauncherError(f"{label} failed: {detail}")
    return result


def _container_id(result: DockerResult) -> str:
    try:
        value = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise SealedContainerLauncherError("Docker returned a non-ASCII container ID") from exc
    if _CONTAINER_ID.fullmatch(value) is None:
        raise SealedContainerLauncherError("Docker returned a malformed container ID")
    return value


def _private_directory(path: Path) -> None:
    if not path.is_absolute():
        raise SealedContainerLauncherError("audit root must be absolute")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SealedContainerLauncherError("audit root does not exist") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise SealedContainerLauncherError("audit root must be one real mode-0700 directory")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise SealedContainerLauncherError("audit root must be owned by the launcher identity")


def _private_real_directory(path: Path, *, label: str) -> os.stat_result:
    if not path.is_absolute():
        raise SealedContainerLauncherError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as exc:
        raise SealedContainerLauncherError(f"cannot inspect {label}") from exc
    if resolved != path or path.is_symlink():
        raise SealedContainerLauncherError(f"{label} cannot contain a path alias")
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise SealedContainerLauncherError(f"{label} must be one owned mode-0700 directory")
    return metadata


def _prepare_copy_destination(path: Path) -> tuple[int, int]:
    _private_real_directory(path.parent, label="sealed output parent")
    if os.path.lexists(path):
        raise SealedContainerLauncherError("sealed output copy already exists")
    try:
        os.mkdir(path, 0o700)
        metadata = path.lstat()
    except OSError as exc:
        raise SealedContainerLauncherError("cannot create sealed output destination") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        or path.resolve(strict=True) != path
    ):
        raise SealedContainerLauncherError(
            "sealed output destination must be one owned mode-0700 directory"
        )
    return metadata.st_dev, metadata.st_ino


def _verify_private_copy_tree(path: Path, identity: tuple[int, int]) -> None:
    _private_real_directory(path, label="sealed output destination")
    root_metadata = path.lstat()
    if (root_metadata.st_dev, root_metadata.st_ino) != identity:
        raise SealedContainerLauncherError("sealed output destination was replaced during copy")
    for member in path.rglob("*"):
        try:
            metadata = member.lstat()
        except OSError as exc:
            raise SealedContainerLauncherError("cannot inspect copied output member") from exc
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise SealedContainerLauncherError("copied output ownership differs")
        if stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o700:
                raise SealedContainerLauncherError("copied output directory mode differs")
        elif stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o600:
                raise SealedContainerLauncherError("copied output file mode or link count differs")
        else:
            raise SealedContainerLauncherError("copied output contains a non-regular member")


def _persist_bytes(path: Path, value: bytes, *, label: str) -> None:
    if type(value) is not bytes:
        raise SealedContainerLauncherError(f"{label} must be bytes")
    parent = path.parent
    _private_directory(parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SealedContainerLauncherError(f"cannot publish {label}") from exc
    try:
        offset = 0
        while offset < len(value):
            written = os.write(descriptor, value[offset:])
            if written <= 0:
                raise SealedContainerLauncherError(f"{label} write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ContainerTerminalState:
    status: str
    exit_code: int
    oom_killed: bool
    error: str

    def __post_init__(self) -> None:
        if self.status != "exited":
            raise SealedContainerLauncherError("retained container is not terminal exited state")
        if type(self.exit_code) is not int:
            raise SealedContainerLauncherError("container exit code must be an integer")
        if type(self.oom_killed) is not bool:
            raise SealedContainerLauncherError("container OOM state must be boolean")
        _text("container state error", self.error, allow_empty=True)


def _volume_labels(geometry: LauncherGeometry, authority_sha256: str) -> dict[str, str]:
    _sha256("volume authority", authority_sha256)
    return {
        f"{_CONTAINER_LABEL_PREFIX}.authority-sha256": authority_sha256,
        f"{_CONTAINER_LABEL_PREFIX}.corpus-id": geometry.corpus_id,
        f"{_CONTAINER_LABEL_PREFIX}.role": "sealed-output-volume",
        f"{_CONTAINER_LABEL_PREFIX}.subpath": geometry.output_volume_subpath,
    }


def _verify_volume_inspect(
    encoded: bytes,
    geometry: LauncherGeometry,
    *,
    authority_sha256: str,
) -> None:
    try:
        value = json.loads(encoded.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedContainerLauncherError("Docker volume inspect evidence is not JSON") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        raise SealedContainerLauncherError("Docker volume inspect must contain one volume")
    record = value[0]
    expected = {
        "Driver": "local",
        "Labels": _volume_labels(geometry, authority_sha256),
        "Name": geometry.output_volume,
        "Options": {},
        "Scope": "local",
    }
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise SealedContainerLauncherError(f"Docker volume inspect {field} differs")


def _verify_docker_inspect(
    encoded: bytes,
    geometry: LauncherGeometry,
    argv: tuple[str, ...],
    *,
    container_id: str,
    container_name: str,
    role: str,
    authority_sha256: str,
    start_returncode: int,
    include_bind_mounts: bool = True,
    output_target: str | None = None,
    output_read_only: bool = False,
    use_output_subpath: bool = True,
    root_identity: bool = False,
) -> ContainerTerminalState:
    try:
        value = json.loads(encoded.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedContainerLauncherError("Docker inspect evidence is not JSON") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        raise SealedContainerLauncherError("Docker inspect must contain one container")
    record = value[0]
    config = record.get("Config")
    host = record.get("HostConfig")
    mounts = record.get("Mounts")
    state = record.get("State")
    if (
        not isinstance(config, Mapping)
        or not isinstance(host, Mapping)
        or not isinstance(mounts, list)
        or not isinstance(state, Mapping)
    ):
        raise SealedContainerLauncherError(
            "Docker inspect lacks typed config, host, mounts, or state"
        )
    if (
        record.get("Id") != container_id
        or record.get("Name") != f"/{container_name}"
        or record.get("Platform") != geometry.platform
    ):
        raise SealedContainerLauncherError("Docker inspect container identity differs")
    uid = 0 if root_identity else geometry.uid
    gid = 0 if root_identity else geometry.gid
    expected_config = {
        "Hostname": geometry.hostname,
        "Image": geometry.oci_image_digest,
        "User": f"{uid}:{gid}",
        "Entrypoint": [argv[0]],
        "Cmd": list(argv[1:]),
        "Labels": _container_labels(
            geometry,
            role=role,
            authority_sha256=authority_sha256,
        ),
    }
    for field, expected in expected_config.items():
        if config.get(field) != expected:
            raise SealedContainerLauncherError(f"Docker inspect Config.{field} differs")
    environment = config.get("Env")
    if not isinstance(environment, list) or any(type(item) is not str for item in environment):
        raise SealedContainerLauncherError("Docker inspect environment is malformed")
    parsed_environment: dict[str, str] = {}
    for item in environment:
        name, separator, contents = item.partition("=")
        if not separator or name in parsed_environment:
            raise SealedContainerLauncherError("Docker inspect environment repeats a name")
        parsed_environment[name] = contents
    if parsed_environment != geometry.environment_dict:
        raise SealedContainerLauncherError("Docker inspect environment differs")
    expected_host = {
        "AutoRemove": False,
        "CapAdd": ["CHOWN", "FOWNER"] if root_identity else None,
        "CapDrop": ["ALL"],
        "CpusetCpus": geometry.cpuset_text,
        "Memory": geometry.memory_limit_bytes,
        "NetworkMode": "none",
        "Privileged": False,
        "ReadonlyRootfs": True,
        "RestartPolicy": {"MaximumRetryCount": 0, "Name": "no"},
    }
    for field, expected in expected_host.items():
        if host.get(field) != expected:
            raise SealedContainerLauncherError(f"Docker inspect HostConfig.{field} differs")
    security = host.get("SecurityOpt")
    if security not in (["no-new-privileges"], ["no-new-privileges:true"]):
        raise SealedContainerLauncherError("Docker inspect no-new-privileges differs")
    tmpfs = host.get("Tmpfs")
    if not isinstance(tmpfs, Mapping) or set(tmpfs) != {geometry.tmpfs_root}:
        raise SealedContainerLauncherError("Docker inspect tmpfs root differs")
    tmpfs_options = tmpfs[geometry.tmpfs_root]
    if type(tmpfs_options) is not str or set(tmpfs_options.split(",")) != {
        "rw",
        *geometry.tmpfs_flags,
        f"size={geometry.tmpfs_size_bytes}",
        f"uid={uid}",
        f"gid={gid}",
        f"mode={geometry.tmpfs_mode:o}",
    }:
        raise SealedContainerLauncherError("Docker inspect tmpfs options differ")
    configured_mounts = host.get("Mounts")
    if not isinstance(configured_mounts, list):
        raise SealedContainerLauncherError("Docker inspect configured mounts are missing")
    expected_configured: dict[str, tuple[str, str, bool, str | None]] = {}
    if include_bind_mounts:
        expected_configured.update(
            {item.target: ("bind", item.source, True, None) for item in geometry.bind_mounts}
        )
    target = geometry.output_root if output_target is None else output_target
    subpath = geometry.output_volume_subpath if use_output_subpath else None
    expected_configured[target] = (
        "volume",
        geometry.output_volume,
        output_read_only,
        subpath,
    )
    if len(configured_mounts) != len(expected_configured):
        raise SealedContainerLauncherError("Docker inspect configured mount count differs")
    for configured in configured_mounts:
        if not isinstance(configured, Mapping):
            raise SealedContainerLauncherError("Docker inspect configured mount is malformed")
        target = configured.get("Target")
        if type(target) is not str or target not in expected_configured:
            raise SealedContainerLauncherError("Docker inspect configured target differs")
        kind, source, read_only, subpath = expected_configured.pop(target)
        if (
            configured.get("Type") != kind
            or configured.get("Source") != source
            or configured.get("ReadOnly") is not read_only
        ):
            raise SealedContainerLauncherError("Docker inspect configured mount differs")
        options = configured.get("VolumeOptions")
        if subpath is None:
            if options not in (None, {}):
                raise SealedContainerLauncherError("bind mount has volume options")
        elif not isinstance(options, Mapping) or options.get("Subpath") != subpath:
            raise SealedContainerLauncherError("Docker inspect volume subpath differs")
    expected_mounts: dict[str, tuple[str, str, bool]] = {}
    if include_bind_mounts:
        expected_mounts.update(
            {item.target: ("bind", item.source, False) for item in geometry.bind_mounts}
        )
    expected_mounts[target] = (
        "volume",
        geometry.output_volume,
        not output_read_only,
    )
    if len(mounts) != len(expected_mounts):
        raise SealedContainerLauncherError("Docker inspect mount count differs")
    observed_targets: set[str] = set()
    for position, value in enumerate(mounts):
        if not isinstance(value, Mapping):
            raise SealedContainerLauncherError(f"Docker inspect Mounts[{position}] is malformed")
        target = value.get("Destination")
        if type(target) is not str or target in observed_targets or target not in expected_mounts:
            raise SealedContainerLauncherError("Docker inspect mount target differs")
        observed_targets.add(target)
        kind, identity, writable = expected_mounts[target]
        if value.get("Type") != kind or value.get("RW") is not writable:
            raise SealedContainerLauncherError("Docker inspect mount type or mode differs")
        observed_identity = value.get("Source") if kind == "bind" else value.get("Name")
        if observed_identity != identity:
            raise SealedContainerLauncherError("Docker inspect mount identity differs")
    expected_state = {
        "Dead": False,
        "ExitCode": start_returncode,
        "Paused": False,
        "Pid": 0,
        "Restarting": False,
        "Running": False,
        "Status": "exited",
    }
    for field, expected in expected_state.items():
        if state.get(field) != expected:
            raise SealedContainerLauncherError(f"Docker inspect State.{field} differs")
    if type(state.get("OOMKilled")) is not bool or type(state.get("Error")) is not str:
        raise SealedContainerLauncherError("Docker inspect terminal state is malformed")
    return ContainerTerminalState(
        status=state["Status"],
        exit_code=state["ExitCode"],
        oom_killed=state["OOMKilled"],
        error=state["Error"],
    )


def _inspect_and_logs(
    docker: DockerRunner,
    container_id: str,
    audit_root: Path,
    prefix: str,
    *,
    secret: bytes | None = None,
) -> tuple[bytes, bytes, bytes]:
    inspect = _run_checked(
        docker,
        ("inspect", "--type", "container", container_id),
        label=f"{prefix} Docker inspect",
    ).stdout
    logs = _run_checked(docker, ("logs", container_id), label=f"{prefix} Docker logs")
    for value in (inspect, logs.stdout, logs.stderr):
        if secret is not None and secret in value:
            raise SealedContainerLauncherError("secret bytes appeared in retained Docker evidence")
    _persist_bytes(audit_root / f"{prefix}-inspect.json", inspect, label="Docker inspect evidence")
    _persist_bytes(audit_root / f"{prefix}-stdout.log", logs.stdout, label="Docker stdout")
    _persist_bytes(audit_root / f"{prefix}-stderr.log", logs.stderr, label="Docker stderr")
    return inspect, logs.stdout, logs.stderr


def initialize_output_volume(
    contract: PreflightLaunchContract,
    *,
    audit_root: str | Path,
    docker: DockerRunner | None = None,
) -> VolumeInitializationReceipt:
    """Create a fresh volume and its root-owned, mode-0700 output subpath."""

    if not isinstance(contract, PreflightLaunchContract):
        raise SealedContainerLauncherError("volume initializer requires a preflight contract")
    root = Path(audit_root)
    _private_directory(root)
    verify_launcher_mounts(contract.geometry)
    active = docker if docker is not None else SubprocessDockerRunner()
    existing = active.run(("volume", "inspect", contract.geometry.output_volume))
    if existing.returncode == 0:
        raise SealedContainerLauncherError("output volume already exists")
    volume_labels = _volume_labels(contract.geometry, contract.contract_sha256)
    volume_create = ["volume", "create"]
    for label_name in sorted(volume_labels, key=lambda item: item.encode("utf-8")):
        volume_create.extend(("--label", f"{label_name}={volume_labels[label_name]}"))
    volume_create.append(contract.geometry.output_volume)
    _run_checked_mutation(
        active,
        volume_create,
        audit_root=root,
        operation="volume-create",
        label="Docker volume creation",
    )
    volume_inspect = _run_checked(
        active,
        ("volume", "inspect", contract.geometry.output_volume),
        label="Docker volume inspection",
    ).stdout
    _verify_volume_inspect(
        volume_inspect,
        contract.geometry,
        authority_sha256=contract.contract_sha256,
    )
    _persist_bytes(
        root / "volume-inspect.json",
        volume_inspect,
        label="Docker volume inspect evidence",
    )
    name = f"fractal-init-{contract.geometry.corpus_id}"
    init_argv = (
        _PYTHON,
        "-m",
        _MODULE,
        "initialize-output",
        "--path",
        f"/volume/{contract.geometry.output_volume_subpath}",
    )
    create = [
        "create",
        "--name",
        name,
        "--platform",
        contract.geometry.platform,
        "--user",
        "0:0",
        "--cap-drop",
        "ALL",
        "--cap-add",
        "CHOWN",
        "--cap-add",
        "FOWNER",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--network",
        "none",
        "--memory",
        str(contract.geometry.memory_limit_bytes),
        "--cpuset-cpus",
        contract.geometry.cpuset_text,
        "--hostname",
        contract.geometry.hostname,
    ]
    _append_labels(
        create,
        _container_labels(
            contract.geometry,
            role="output-initializer",
            authority_sha256=contract.contract_sha256,
        ),
    )
    for row in contract.geometry.environment:
        create.extend(("--env", f"{row.name}={row.value}"))
    create.extend(
        (
            "--tmpfs",
            _tmpfs(contract.geometry, root=True),
            "--mount",
            f"type=volume,src={contract.geometry.output_volume},dst=/volume",
            "--entrypoint",
            init_argv[0],
            contract.geometry.oci_image_digest,
            *init_argv[1:],
        )
    )
    container_id = _container_id(
        _run_checked_mutation(
            active,
            create,
            audit_root=root,
            operation="initializer-create",
            label="Docker output initializer creation",
        )
    )
    start = _run_checked_mutation(
        active,
        ("start", "--attach", container_id),
        audit_root=root,
        operation="initializer-start",
        label="Docker output initializer",
    )
    inspect, stdout, stderr = _inspect_and_logs(active, container_id, root, "initializer")
    state = _verify_docker_inspect(
        inspect,
        contract.geometry,
        init_argv,
        container_id=container_id,
        container_name=name,
        role="output-initializer",
        authority_sha256=contract.contract_sha256,
        start_returncode=start.returncode,
        include_bind_mounts=False,
        output_target="/volume",
        use_output_subpath=False,
        root_identity=True,
    )
    if stdout != start.stdout or stderr != start.stderr:
        raise SealedContainerLauncherError("retained initializer logs differ from attached output")
    receipt = VolumeInitializationReceipt(
        corpus_id=contract.geometry.corpus_id,
        preflight_launcher_contract_sha256=contract.contract_sha256,
        output_volume=contract.geometry.output_volume,
        output_volume_subpath=contract.geometry.output_volume_subpath,
        volume_inspect_sha256=_digest_bytes(volume_inspect),
        initializer_container_id=container_id,
        inspect_sha256=_digest_bytes(inspect),
        initializer_start_returncode=start.returncode,
        initializer_state_status=state.status,
        initializer_exit_code=state.exit_code,
        initializer_oom_killed=state.oom_killed,
        stdout_sha256=_digest_bytes(stdout),
        stdout_byte_count=len(stdout),
        stderr_sha256=_digest_bytes(stderr),
        stderr_byte_count=len(stderr),
    )
    _write_receipt(
        receipt.canonical_file_bytes(),
        root / "volume-initialization-receipt.json",
        label="volume initialization receipt",
    )
    return receipt


def _verify_volume_binding(
    contract: PreflightLaunchContract | SealedLaunchContract,
    volume: VolumeInitializationReceipt,
) -> None:
    geometry = contract.geometry
    if (
        volume.corpus_id != geometry.corpus_id
        or volume.preflight_launcher_contract_sha256
        != (
            contract.contract_sha256
            if isinstance(contract, PreflightLaunchContract)
            else contract.preflight_launcher_contract_sha256
        )
        or volume.output_volume != geometry.output_volume
        or volume.output_volume_subpath != geometry.output_volume_subpath
    ):
        raise SealedContainerLauncherError("volume receipt differs from the launcher")


def run_preflight_once(
    contract: PreflightLaunchContract,
    volume: VolumeInitializationReceipt,
    *,
    audit_root: str | Path,
    docker: DockerRunner | None = None,
) -> RuntimePreflightReceipt:
    """Attach once to a label-free preflight and retain its external evidence."""

    if contract.geometry.code_commit == CANDIDATE_C0_COMMIT_SENTINEL:
        raise SealedContainerLauncherError(
            "preflight contract still contains the candidate commit sentinel"
        )
    root = Path(audit_root)
    _private_directory(root)
    _verify_volume_binding(contract, volume)
    verify_launcher_mounts(contract.geometry)
    plan_path = _plan_path(contract)
    if digest_regular_file(plan_path, label="provisional plan template") != (
        contract.provisional_plan_template_file_sha256
    ):
        raise SealedContainerLauncherError("provisional plan-template file differs")
    _require_provisional_sentinels(_load_template(plan_path))
    active = docker if docker is not None else SubprocessDockerRunner()
    name = f"fractal-preflight-{contract.geometry.corpus_id}"
    container_id = _container_id(
        _run_checked_mutation(
            active,
            _create_arguments(
                contract.geometry,
                name=name,
                argv=contract.argv,
                role="preflight",
                authority_sha256=contract.contract_sha256,
            ),
            audit_root=root,
            operation="preflight-create",
            label="Docker preflight creation",
        )
    )
    start = _run_checked_mutation(
        active,
        ("start", "--attach", "--interactive", container_id),
        audit_root=root,
        operation="preflight-start",
        label="Docker preflight",
        input_bytes=contract.canonical_file_bytes(),
    )
    try:
        receipt = loads_runtime_preflight_receipt(start.stdout)
    except RuntimeAttestationError as exc:
        raise SealedContainerLauncherError(f"Docker preflight receipt is invalid: {exc}") from exc
    verify_preflight_receipt(contract, receipt)
    inspect, stdout, stderr = _inspect_and_logs(active, container_id, root, "preflight")
    state = _verify_docker_inspect(
        inspect,
        contract.geometry,
        contract.argv,
        container_id=container_id,
        container_name=name,
        role="preflight",
        authority_sha256=contract.contract_sha256,
        start_returncode=start.returncode,
    )
    if state.exit_code != 0 or state.oom_killed:
        raise SealedContainerLauncherError("preflight did not exit cleanly")
    if stdout != start.stdout or stderr != start.stderr:
        raise SealedContainerLauncherError("retained preflight logs differ from attached output")
    _persist_bytes(
        root / "preflight-attached-stdout.log",
        start.stdout,
        label="preflight attached stdout",
    )
    _persist_bytes(
        root / "preflight-attached-stderr.log",
        start.stderr,
        label="preflight attached stderr",
    )
    _persist_bytes(
        root / "preflight-inspect-digest.txt",
        (_digest_bytes(inspect) + "\n").encode(),
        label="preflight inspect digest",
    )
    _write_receipt(
        receipt.canonical_file_bytes(),
        root / "runtime-preflight-receipt.json",
        label="runtime preflight receipt",
    )
    return receipt


@dataclass(frozen=True)
class LauncherAttemptMarker:
    corpus_id: str
    sealed_launcher_contract_sha256: str
    runtime_plan_transition_receipt_sha256: str
    preflight_receipt_sha256: str
    runtime_claim_receipt_sha256: str
    claim_state_sha256: str
    claim_ledger_commit: str
    provider_identity_sha256: str
    beacon_receipt_sha256: str
    beacon_bytes_sha256: str
    derived_seed_sha256: str
    permutation_seed: int
    output_aggregate_identity: str
    output_volume: str
    output_volume_subpath: str
    stdin_secret_sha256: str
    stdin_secret_byte_count: int
    schema_version: str = LAUNCHER_ATTEMPT_MARKER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != LAUNCHER_ATTEMPT_MARKER_SCHEMA:
            raise SealedContainerLauncherError("launcher attempt marker schema differs")
        _identifier("corpus_id", self.corpus_id)
        for name in (
            "sealed_launcher_contract_sha256",
            "runtime_plan_transition_receipt_sha256",
            "preflight_receipt_sha256",
            "runtime_claim_receipt_sha256",
            "claim_state_sha256",
            "provider_identity_sha256",
            "beacon_receipt_sha256",
            "beacon_bytes_sha256",
            "derived_seed_sha256",
            "output_aggregate_identity",
            "stdin_secret_sha256",
        ):
            _sha256(name, getattr(self, name))
        if _VOLUME_NAME.fullmatch(self.output_volume) is None:
            raise SealedContainerLauncherError("marker output volume is invalid")
        _relative_path("output_volume_subpath", self.output_volume_subpath)
        _positive_integer(
            "stdin_secret_byte_count",
            self.stdin_secret_byte_count,
            maximum=_MAX_SECRET_BYTES,
        )
        if self.stdin_secret_byte_count < _MIN_SECRET_BYTES:
            raise SealedContainerLauncherError("stdin secret is shorter than 32 bytes")
        if (
            type(self.claim_ledger_commit) is not str
            or _GIT_COMMIT.fullmatch(self.claim_ledger_commit) is None
        ):
            raise SealedContainerLauncherError("claim ledger commit is invalid")
        if type(self.permutation_seed) is not int or not 0 <= self.permutation_seed < 2**64:
            raise SealedContainerLauncherError("claim permutation seed is invalid")
        if self.permutation_seed != int.from_bytes(
            bytes.fromhex(self.derived_seed_sha256)[:8], "big"
        ):
            raise SealedContainerLauncherError("claim permutation seed differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "output_volume": self.output_volume,
            "output_volume_subpath": self.output_volume_subpath,
            "preflight_receipt_sha256": self.preflight_receipt_sha256,
            "runtime_claim_receipt_sha256": self.runtime_claim_receipt_sha256,
            "claim_state_sha256": self.claim_state_sha256,
            "claim_ledger_commit": self.claim_ledger_commit,
            "provider_identity_sha256": self.provider_identity_sha256,
            "beacon_receipt_sha256": self.beacon_receipt_sha256,
            "beacon_bytes_sha256": self.beacon_bytes_sha256,
            "derived_seed_sha256": self.derived_seed_sha256,
            "permutation_seed": self.permutation_seed,
            "output_aggregate_identity": self.output_aggregate_identity,
            "runtime_plan_transition_receipt_sha256": (self.runtime_plan_transition_receipt_sha256),
            "schema_version": self.schema_version,
            "sealed_launcher_contract_sha256": self.sealed_launcher_contract_sha256,
            "stdin_secret_byte_count": self.stdin_secret_byte_count,
            "stdin_secret_sha256": self.stdin_secret_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def marker_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> LauncherAttemptMarker:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="launcher attempt marker")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class OutputFileDigest:
    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path("output relative_path", self.relative_path)
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise SealedContainerLauncherError("output byte_count must be nonnegative")
        _sha256("output file sha256", self.sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class OutputCopyReceipt:
    copied_root: str
    tree_sha256: str
    file_count: int
    directory_count: int
    byte_count: int
    files: tuple[OutputFileDigest, ...]
    schema_version: str = OUTPUT_COPY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != OUTPUT_COPY_RECEIPT_SCHEMA:
            raise SealedContainerLauncherError("output-copy receipt schema differs")
        object.__setattr__(
            self,
            "copied_root",
            str(_canonical_host_path("copied_root", self.copied_root)),
        )
        _sha256("tree_sha256", self.tree_sha256)
        for name in ("file_count", "directory_count", "byte_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise SealedContainerLauncherError(f"{name} must be nonnegative")
        files = tuple(self.files)
        if not all(isinstance(item, OutputFileDigest) for item in files):
            raise SealedContainerLauncherError("output files must be typed")
        if files != tuple(sorted(files, key=lambda item: item.relative_path.encode("utf-8"))):
            raise SealedContainerLauncherError("output files must be path-sorted")
        if len({item.relative_path for item in files}) != len(files):
            raise SealedContainerLauncherError("output files repeat a path")
        if self.file_count != len(files) or self.byte_count != sum(
            item.byte_count for item in files
        ):
            raise SealedContainerLauncherError("output-copy accounting differs")
        object.__setattr__(self, "files", files)

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "copied_root": self.copied_root,
            "directory_count": self.directory_count,
            "file_count": self.file_count,
            "files": [item.to_dict() for item in self.files],
            "schema_version": self.schema_version,
            "tree_sha256": self.tree_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> OutputCopyReceipt:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="output copy receipt")
        files = row["files"]
        if type(files) is not list:
            raise SealedContainerLauncherError("output copy files must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "files"},
            files=tuple(
                OutputFileDigest(
                    **_closed(
                        item,
                        frozenset({"byte_count", "relative_path", "sha256"}),
                        label="output copy file",
                    )
                )
                for item in files
            ),
        )


@dataclass(frozen=True)
class ContainerOutputInventory:
    tree_sha256: str
    file_count: int
    directory_count: int
    byte_count: int
    files: tuple[OutputFileDigest, ...]
    schema_version: str = CONTAINER_OUTPUT_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONTAINER_OUTPUT_INVENTORY_SCHEMA:
            raise SealedContainerLauncherError("container output inventory schema differs")
        _sha256("tree_sha256", self.tree_sha256)
        for name in ("file_count", "directory_count", "byte_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise SealedContainerLauncherError(f"{name} must be nonnegative")
        files = tuple(self.files)
        if not all(isinstance(item, OutputFileDigest) for item in files):
            raise SealedContainerLauncherError("container output files must be typed")
        if files != tuple(sorted(files, key=lambda item: item.relative_path.encode("utf-8"))):
            raise SealedContainerLauncherError("container output files must be sorted")
        if self.file_count != len(files) or self.byte_count != sum(
            item.byte_count for item in files
        ):
            raise SealedContainerLauncherError("container output accounting differs")
        object.__setattr__(self, "files", files)

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "directory_count": self.directory_count,
            "file_count": self.file_count,
            "files": [item.to_dict() for item in self.files],
            "schema_version": self.schema_version,
            "tree_sha256": self.tree_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def inventory_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ContainerOutputInventory:
        row = _closed(
            value,
            frozenset(
                {
                    "byte_count",
                    "directory_count",
                    "file_count",
                    "files",
                    "schema_version",
                    "tree_sha256",
                }
            ),
            label="container output inventory",
        )
        files = row["files"]
        if type(files) is not list:
            raise SealedContainerLauncherError("container output files must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "files"},
            files=tuple(
                OutputFileDigest(
                    **_closed(
                        item,
                        frozenset({"byte_count", "relative_path", "sha256"}),
                        label="container output file",
                    )
                )
                for item in files
            ),
        )


@dataclass(frozen=True)
class SealedEvidenceFile:
    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        path = _relative_path("sealed evidence relative_path", self.relative_path)
        if len(PurePosixPath(path).parts) != 1:
            raise SealedContainerLauncherError("sealed evidence must be one direct file")
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise SealedContainerLauncherError("sealed evidence byte_count must be nonnegative")
        _sha256("sealed evidence sha256", self.sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> SealedEvidenceFile:
        row = _closed(
            value,
            frozenset({"byte_count", "relative_path", "sha256"}),
            label="sealed evidence file",
        )
        return cls(**row)  # type: ignore[arg-type]


def _sealed_evidence_inventory_sha256(files: tuple[SealedEvidenceFile, ...]) -> str:
    payload = {
        "files": [item.to_dict() for item in files],
        "schema_version": SEALED_EVIDENCE_INVENTORY_SCHEMA,
    }
    return _digest_bytes(_canonical_bytes(payload))


@dataclass(frozen=True)
class SealedLaunchFailureErrorRecord:
    """Private, redacted exception evidence for a consumed launcher attempt."""

    failure_stage: str
    exception_class: str
    redacted_message: str
    schema_version: str = SEALED_LAUNCH_FAILURE_ERROR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SEALED_LAUNCH_FAILURE_ERROR_SCHEMA:
            raise SealedContainerLauncherError("sealed failure error schema differs")
        _identifier("sealed failure error stage", self.failure_stage)
        if (
            type(self.exception_class) is not str
            or not self.exception_class
            or len(self.exception_class.encode("utf-8", errors="strict")) > 1024
        ):
            raise SealedContainerLauncherError("sealed failure exception class is invalid")
        if (
            type(self.redacted_message) is not str
            or len(self.redacted_message.encode("utf-8", errors="strict")) > 8192
        ):
            raise SealedContainerLauncherError("sealed failure message is invalid")

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @classmethod
    def from_dict(cls, value: object) -> SealedLaunchFailureErrorRecord:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="sealed failure error")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class SealedLaunchReceipt:
    corpus_id: str
    outcome: Literal["succeeded", "failed"]
    sealed_launcher_contract_sha256: str
    preflight_launcher_contract_sha256: str
    preflight_receipt_sha256: str
    runtime_claim_receipt_sha256: str
    claim_state_sha256: str
    claim_ledger_commit: str
    provider_identity_sha256: str
    beacon_receipt_sha256: str
    beacon_bytes_sha256: str
    derived_seed_sha256: str
    permutation_seed: int
    output_aggregate_identity: str
    runtime_plan_transition_receipt_sha256: str
    registered_plan_instantiation_receipt_sha256: str
    production_run_closure_binding_receipt_sha256: str
    volume_initialization_receipt_sha256: str
    launcher_attempt_marker_sha256: str
    container_id: str
    output_volume: str
    output_volume_subpath: str
    stdin_secret_sha256: str
    stdin_secret_byte_count: int
    docker_start_returncode: int
    container_state_status: str
    container_exit_code: int
    container_oom_killed: bool
    container_state_error_sha256: str
    container_state_error_byte_count: int
    attached_stdout_sha256: str
    attached_stdout_byte_count: int
    attached_stderr_sha256: str
    attached_stderr_byte_count: int
    inspect_sha256: str
    retained_stdout_sha256: str
    retained_stdout_byte_count: int
    retained_stderr_sha256: str
    retained_stderr_byte_count: int
    output_reader_container_id: str | None
    output_reader_start_returncode: int | None
    output_reader_exit_code: int | None
    output_reader_oom_killed: bool | None
    output_reader_inspect_sha256: str | None
    output_reader_inventory_sha256: str | None
    copy_output_root: str | None
    output_copy_receipt_sha256: str | None
    output_tree_sha256: str | None
    evidence_inventory_sha256: str
    evidence_files: tuple[SealedEvidenceFile, ...]
    schema_version: str = SEALED_LAUNCH_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SEALED_LAUNCH_RECEIPT_SCHEMA:
            raise SealedContainerLauncherError("sealed launch receipt schema differs")
        _identifier("corpus_id", self.corpus_id)
        if self.outcome not in {"succeeded", "failed"}:
            raise SealedContainerLauncherError("sealed launch outcome is invalid")
        if _CONTAINER_ID.fullmatch(self.container_id) is None:
            raise SealedContainerLauncherError("sealed container ID is invalid")
        if _VOLUME_NAME.fullmatch(self.output_volume) is None:
            raise SealedContainerLauncherError("sealed output volume is invalid")
        _relative_path("output_volume_subpath", self.output_volume_subpath)
        for name in (
            "sealed_launcher_contract_sha256",
            "preflight_launcher_contract_sha256",
            "preflight_receipt_sha256",
            "runtime_claim_receipt_sha256",
            "claim_state_sha256",
            "provider_identity_sha256",
            "beacon_receipt_sha256",
            "beacon_bytes_sha256",
            "derived_seed_sha256",
            "output_aggregate_identity",
            "runtime_plan_transition_receipt_sha256",
            "registered_plan_instantiation_receipt_sha256",
            "production_run_closure_binding_receipt_sha256",
            "volume_initialization_receipt_sha256",
            "launcher_attempt_marker_sha256",
            "stdin_secret_sha256",
            "container_state_error_sha256",
            "attached_stdout_sha256",
            "attached_stderr_sha256",
            "inspect_sha256",
            "retained_stdout_sha256",
            "retained_stderr_sha256",
            "evidence_inventory_sha256",
        ):
            _sha256(name, getattr(self, name))
        for name in (
            "stdin_secret_byte_count",
            "container_state_error_byte_count",
            "attached_stdout_byte_count",
            "attached_stderr_byte_count",
            "retained_stdout_byte_count",
            "retained_stderr_byte_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise SealedContainerLauncherError(f"{name} must be nonnegative")
        if (
            type(self.claim_ledger_commit) is not str
            or _GIT_COMMIT.fullmatch(self.claim_ledger_commit) is None
        ):
            raise SealedContainerLauncherError("sealed claim ledger commit is invalid")
        if type(self.permutation_seed) is not int or not 0 <= self.permutation_seed < 2**64:
            raise SealedContainerLauncherError("sealed claim permutation seed is invalid")
        if self.permutation_seed != int.from_bytes(
            bytes.fromhex(self.derived_seed_sha256)[:8], "big"
        ):
            raise SealedContainerLauncherError("sealed claim permutation seed differs")
        if (
            type(self.docker_start_returncode) is not int
            or type(self.container_exit_code) is not int
        ):
            raise SealedContainerLauncherError("sealed exit statuses must be integers")
        if self.container_exit_code != self.docker_start_returncode:
            raise SealedContainerLauncherError("sealed Docker and container exit statuses differ")
        if self.container_state_status != "exited" or type(self.container_oom_killed) is not bool:
            raise SealedContainerLauncherError("sealed terminal state is malformed")
        optional_digests = (
            self.output_reader_inspect_sha256,
            self.output_reader_inventory_sha256,
            self.output_copy_receipt_sha256,
            self.output_tree_sha256,
        )
        optional_status = (
            self.output_reader_container_id,
            self.output_reader_start_returncode,
            self.output_reader_exit_code,
            self.output_reader_oom_killed,
            self.copy_output_root,
            *optional_digests,
        )
        if self.outcome == "succeeded":
            if any(item is None for item in optional_status):
                raise SealedContainerLauncherError("successful sealed launch lacks copy evidence")
            assert self.output_reader_container_id is not None
            assert self.copy_output_root is not None
            if _CONTAINER_ID.fullmatch(self.output_reader_container_id) is None:
                raise SealedContainerLauncherError("output reader container ID is invalid")
            object.__setattr__(
                self,
                "copy_output_root",
                str(_canonical_host_path("copy_output_root", self.copy_output_root)),
            )
            if (
                type(self.output_reader_start_returncode) is not int
                or type(self.output_reader_exit_code) is not int
                or self.output_reader_start_returncode != 0
                or self.output_reader_exit_code != 0
                or type(self.output_reader_oom_killed) is not bool
                or self.output_reader_oom_killed
            ):
                raise SealedContainerLauncherError("output reader did not exit cleanly")
            for position, value in enumerate(optional_digests):
                _sha256(f"successful optional digest {position}", value)
            if self.docker_start_returncode != 0 or self.container_oom_killed:
                raise SealedContainerLauncherError("successful sealed launch did not exit cleanly")
        elif any(item is not None for item in optional_status):
            raise SealedContainerLauncherError("failed sealed launch contains copy evidence")
        elif self.docker_start_returncode == 0 and not self.container_oom_killed:
            raise SealedContainerLauncherError("failed sealed launch has no failed exit state")
        files = tuple(self.evidence_files)
        if (
            not files
            or not all(isinstance(item, SealedEvidenceFile) for item in files)
            or files != tuple(sorted(files, key=lambda item: item.relative_path.encode("utf-8")))
            or len({item.relative_path for item in files}) != len(files)
        ):
            raise SealedContainerLauncherError("sealed evidence inventory is not closed and sorted")
        if any(item.relative_path == _LAUNCH_RECEIPT_FILENAME for item in files):
            raise SealedContainerLauncherError("sealed launch receipt cannot inventory itself")
        if _sealed_evidence_inventory_sha256(files) != self.evidence_inventory_sha256:
            raise SealedContainerLauncherError("sealed evidence inventory digest differs")
        object.__setattr__(self, "evidence_files", files)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "evidence_files"
            },
            "evidence_files": [item.to_dict() for item in self.evidence_files],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @property
    def file_sha256(self) -> str:
        return _digest_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> SealedLaunchReceipt:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="sealed launch receipt")
        evidence = row["evidence_files"]
        if type(evidence) is not list:
            raise SealedContainerLauncherError("sealed evidence files must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "evidence_files"},
            evidence_files=tuple(SealedEvidenceFile.from_dict(item) for item in evidence),
        )


@dataclass(frozen=True)
class SealedLaunchFailureReceipt:
    """Terminal host record for any exception after the one-shot marker exists."""

    corpus_id: str
    failure_stage: str
    failure_error_sha256: str
    failure_error_byte_count: int
    sealed_launcher_contract_sha256: str
    preflight_launcher_contract_sha256: str
    preflight_receipt_sha256: str
    runtime_plan_transition_receipt_sha256: str
    registered_plan_instantiation_receipt_sha256: str
    production_run_closure_binding_receipt_sha256: str
    volume_initialization_receipt_sha256: str
    launcher_attempt_marker_sha256: str
    stdin_secret_sha256: str
    stdin_secret_byte_count: int
    sealed_container_id: str | None
    output_reader_container_id: str | None
    evidence_inventory_sha256: str
    evidence_files: tuple[SealedEvidenceFile, ...]
    schema_version: str = SEALED_LAUNCH_FAILURE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SEALED_LAUNCH_FAILURE_RECEIPT_SCHEMA:
            raise SealedContainerLauncherError("sealed failure receipt schema differs")
        _identifier("failure corpus_id", self.corpus_id)
        _identifier("sealed failure stage", self.failure_stage)
        for name in (
            "failure_error_sha256",
            "sealed_launcher_contract_sha256",
            "preflight_launcher_contract_sha256",
            "preflight_receipt_sha256",
            "runtime_plan_transition_receipt_sha256",
            "registered_plan_instantiation_receipt_sha256",
            "production_run_closure_binding_receipt_sha256",
            "volume_initialization_receipt_sha256",
            "launcher_attempt_marker_sha256",
            "stdin_secret_sha256",
            "evidence_inventory_sha256",
        ):
            _sha256(name, getattr(self, name))
        for name in (
            "failure_error_byte_count",
            "stdin_secret_byte_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise SealedContainerLauncherError(f"{name} must be nonnegative")
        for name in ("sealed_container_id", "output_reader_container_id"):
            value = getattr(self, name)
            if value is not None and _CONTAINER_ID.fullmatch(value) is None:
                raise SealedContainerLauncherError(f"{name} is invalid")
        files = tuple(self.evidence_files)
        if (
            not files
            or not all(isinstance(item, SealedEvidenceFile) for item in files)
            or files != tuple(sorted(files, key=lambda item: item.relative_path.encode("utf-8")))
            or len({item.relative_path for item in files}) != len(files)
        ):
            raise SealedContainerLauncherError("sealed failure evidence is not closed and sorted")
        if any(item.relative_path == _LAUNCH_FAILURE_RECEIPT_FILENAME for item in files):
            raise SealedContainerLauncherError("sealed failure receipt cannot inventory itself")
        if _sealed_evidence_inventory_sha256(files) != self.evidence_inventory_sha256:
            raise SealedContainerLauncherError("sealed failure inventory digest differs")
        object.__setattr__(self, "evidence_files", files)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "evidence_files"
            },
            "evidence_files": [item.to_dict() for item in self.evidence_files],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _digest_bytes(self.canonical_bytes())

    @property
    def file_sha256(self) -> str:
        return _digest_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> SealedLaunchFailureReceipt:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="sealed failure receipt")
        evidence = row["evidence_files"]
        if type(evidence) is not list:
            raise SealedContainerLauncherError("sealed failure evidence files must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "evidence_files"},
            evidence_files=tuple(SealedEvidenceFile.from_dict(item) for item in evidence),
        )


def loads_docker_argument_record(encoded: bytes) -> DockerArgumentRecord:
    return _load_canonical(encoded, DockerArgumentRecord, label="Docker argument record")


def loads_docker_command_result_record(encoded: bytes) -> DockerCommandResultRecord:
    return _load_canonical(encoded, DockerCommandResultRecord, label="Docker result record")


def loads_launcher_attempt_marker(encoded: bytes) -> LauncherAttemptMarker:
    return _load_canonical(encoded, LauncherAttemptMarker, label="launcher attempt marker")


def loads_output_copy_receipt(encoded: bytes) -> OutputCopyReceipt:
    return _load_canonical(encoded, OutputCopyReceipt, label="output copy receipt")


def loads_sealed_launch_receipt(encoded: bytes) -> SealedLaunchReceipt:
    return _load_canonical(encoded, SealedLaunchReceipt, label="sealed launch receipt")


def loads_sealed_launch_failure_error_record(
    encoded: bytes,
) -> SealedLaunchFailureErrorRecord:
    return _load_canonical(
        encoded,
        SealedLaunchFailureErrorRecord,
        label="sealed launch failure error",
    )


def loads_sealed_launch_failure_receipt(encoded: bytes) -> SealedLaunchFailureReceipt:
    return _load_canonical(
        encoded,
        SealedLaunchFailureReceipt,
        label="sealed launch failure receipt",
    )


def load_sealed_launch_receipt(
    path: str | Path,
    *,
    expected_receipt_sha256: str | None = None,
    expected_file_sha256: str | None = None,
) -> SealedLaunchReceipt:
    encoded = _read_control(path, "sealed launch receipt")
    receipt = loads_sealed_launch_receipt(encoded)
    if expected_receipt_sha256 is not None:
        _sha256("expected sealed launch receipt SHA-256", expected_receipt_sha256)
        if receipt.receipt_sha256 != expected_receipt_sha256:
            raise SealedContainerLauncherError(
                "sealed launch receipt differs from its semantic pin"
            )
    if expected_file_sha256 is not None:
        _sha256("expected sealed launch file SHA-256", expected_file_sha256)
        if _digest_bytes(encoded) != expected_file_sha256:
            raise SealedContainerLauncherError("sealed launch receipt differs from its file pin")
    return receipt


def load_sealed_launch_failure_receipt(
    path: str | Path,
    *,
    expected_receipt_sha256: str | None = None,
    expected_file_sha256: str | None = None,
) -> SealedLaunchFailureReceipt:
    encoded = _read_control(path, "sealed launch failure receipt")
    receipt = loads_sealed_launch_failure_receipt(encoded)
    if expected_receipt_sha256 is not None:
        _sha256("expected sealed failure receipt SHA-256", expected_receipt_sha256)
        if receipt.receipt_sha256 != expected_receipt_sha256:
            raise SealedContainerLauncherError(
                "sealed failure receipt differs from its semantic pin"
            )
    if expected_file_sha256 is not None:
        _sha256("expected sealed failure file SHA-256", expected_file_sha256)
        if _digest_bytes(encoded) != expected_file_sha256:
            raise SealedContainerLauncherError("sealed failure receipt differs from its file pin")
    return receipt


def _snapshot_sealed_evidence(
    root: Path,
    *,
    excluded_filenames: frozenset[str] = frozenset({_LAUNCH_RECEIPT_FILENAME}),
) -> tuple[SealedEvidenceFile, ...]:
    _private_directory(root)
    rows: list[SealedEvidenceFile] = []
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name.encode("utf-8"))
    except OSError as exc:
        raise SealedContainerLauncherError("cannot enumerate sealed evidence") from exc
    for path in entries:
        if path.name in excluded_filenames:
            continue
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise SealedContainerLauncherError("cannot inspect sealed evidence member") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise SealedContainerLauncherError(
                "sealed evidence must contain only private singly linked regular files"
            )
        try:
            digest = digest_regular_file(path, label="sealed evidence member")
        except ArtifactIntegrityError as exc:
            raise SealedContainerLauncherError("cannot hash sealed evidence member") from exc
        rows.append(
            SealedEvidenceFile(
                relative_path=path.name,
                byte_count=metadata.st_size,
                sha256=digest,
            )
        )
    return tuple(rows)


def _verify_command_evidence(
    root: Path,
    operation: str,
) -> tuple[DockerArgumentRecord, DockerCommandResultRecord]:
    arguments = loads_docker_argument_record(
        _read_control(root / f"{operation}-docker-argv.json", f"{operation} Docker arguments")
    )
    result = loads_docker_command_result_record(
        _read_control(root / f"{operation}-docker-result.json", f"{operation} Docker result")
    )
    stdout = _read_control(root / f"{operation}-docker-stdout.log", f"{operation} stdout")
    stderr = _read_control(root / f"{operation}-docker-stderr.log", f"{operation} stderr")
    if (
        arguments.operation != operation
        or result.operation != operation
        or result.argument_record_sha256 != arguments.record_sha256
        or result.stdout_sha256 != _digest_bytes(stdout)
        or result.stdout_byte_count != len(stdout)
        or result.stderr_sha256 != _digest_bytes(stderr)
        or result.stderr_byte_count != len(stderr)
    ):
        raise SealedContainerLauncherError(f"{operation} Docker command evidence differs")
    return arguments, result


def verify_sealed_launch_evidence(
    receipt: SealedLaunchReceipt,
    *,
    audit_root: str | Path,
    sealed_contract: SealedLaunchContract | None = None,
) -> None:
    """Rehash one closed launch directory and rederive its cross-file bindings."""

    if not isinstance(receipt, SealedLaunchReceipt):
        raise SealedContainerLauncherError("sealed launch evidence requires a typed receipt")
    root = Path(audit_root)
    _private_directory(root)
    persisted = load_sealed_launch_receipt(root / _LAUNCH_RECEIPT_FILENAME)
    if persisted != receipt:
        raise SealedContainerLauncherError("persisted sealed launch receipt differs")
    observed_files = _snapshot_sealed_evidence(root)
    if observed_files != receipt.evidence_files:
        raise SealedContainerLauncherError("sealed launch evidence membership or bytes differ")
    observed_operations = {
        item.relative_path.removesuffix("-docker-argv.json")
        for item in observed_files
        if item.relative_path.endswith("-docker-argv.json")
    }
    allowed_operations = {
        "volume-create",
        "initializer-create",
        "initializer-start",
        "preflight-create",
        "preflight-start",
        "sealed-create",
        "sealed-start",
        "output-reader-create",
        "output-reader-start",
        "output-copy",
    }
    required_operations = {"sealed-create", "sealed-start"}
    if receipt.outcome == "succeeded":
        required_operations.update({"output-reader-create", "output-reader-start", "output-copy"})
    if not required_operations.issubset(observed_operations) or not observed_operations.issubset(
        allowed_operations
    ):
        raise SealedContainerLauncherError("sealed launch Docker operation inventory differs")
    marker = loads_launcher_attempt_marker(
        _read_control(root / "sealed-launcher-attempt-marker.json", "launcher attempt marker")
    )
    if (
        marker.marker_sha256 != receipt.launcher_attempt_marker_sha256
        or marker.corpus_id != receipt.corpus_id
        or marker.sealed_launcher_contract_sha256 != receipt.sealed_launcher_contract_sha256
        or marker.runtime_claim_receipt_sha256 != receipt.runtime_claim_receipt_sha256
        or marker.claim_state_sha256 != receipt.claim_state_sha256
        or marker.claim_ledger_commit != receipt.claim_ledger_commit
        or marker.provider_identity_sha256 != receipt.provider_identity_sha256
        or marker.beacon_receipt_sha256 != receipt.beacon_receipt_sha256
        or marker.beacon_bytes_sha256 != receipt.beacon_bytes_sha256
        or marker.derived_seed_sha256 != receipt.derived_seed_sha256
        or marker.permutation_seed != receipt.permutation_seed
        or marker.output_aggregate_identity != receipt.output_aggregate_identity
        or marker.stdin_secret_sha256 != receipt.stdin_secret_sha256
        or marker.stdin_secret_byte_count != receipt.stdin_secret_byte_count
    ):
        raise SealedContainerLauncherError("sealed launcher marker differs from its receipt")
    create_arguments, create_result = _verify_command_evidence(root, "sealed-create")
    start_arguments, start_result = _verify_command_evidence(root, "sealed-start")
    if create_result.returncode != 0 or start_result.returncode != receipt.docker_start_returncode:
        raise SealedContainerLauncherError("sealed Docker command status differs from its receipt")
    if start_arguments.arguments != (
        "start",
        "--attach",
        "--interactive",
        receipt.container_id,
    ):
        raise SealedContainerLauncherError("sealed start argument array differs")
    attached_stdout = _read_control(root / "sealed-attached-stdout.log", "attached stdout")
    attached_stderr = _read_control(root / "sealed-attached-stderr.log", "attached stderr")
    retained_stdout = _read_control(root / "sealed-stdout.log", "retained sealed stdout")
    retained_stderr = _read_control(root / "sealed-stderr.log", "retained sealed stderr")
    inspect = _read_control(root / "sealed-inspect.json", "sealed inspect")
    if (
        receipt.attached_stdout_sha256 != _digest_bytes(attached_stdout)
        or receipt.attached_stdout_byte_count != len(attached_stdout)
        or receipt.attached_stderr_sha256 != _digest_bytes(attached_stderr)
        or receipt.attached_stderr_byte_count != len(attached_stderr)
        or receipt.retained_stdout_sha256 != _digest_bytes(retained_stdout)
        or receipt.retained_stdout_byte_count != len(retained_stdout)
        or receipt.retained_stderr_sha256 != _digest_bytes(retained_stderr)
        or receipt.retained_stderr_byte_count != len(retained_stderr)
        or receipt.inspect_sha256 != _digest_bytes(inspect)
        or attached_stdout != retained_stdout
        or attached_stderr != retained_stderr
    ):
        raise SealedContainerLauncherError("sealed attached, log, or inspect evidence differs")
    if sealed_contract is not None:
        if (
            not isinstance(sealed_contract, SealedLaunchContract)
            or sealed_contract.contract_sha256 != receipt.sealed_launcher_contract_sha256
        ):
            raise SealedContainerLauncherError("sealed contract differs from launch evidence")
        expected_create = tuple(
            _create_arguments(
                sealed_contract.geometry,
                name=f"fractal-sealed-{receipt.corpus_id}",
                argv=sealed_contract.argv,
                role="sealed",
                authority_sha256=sealed_contract.contract_sha256,
            )
        )
        if create_arguments.arguments != expected_create:
            raise SealedContainerLauncherError("sealed create argument array differs")
        state = _verify_docker_inspect(
            inspect,
            sealed_contract.geometry,
            sealed_contract.argv,
            container_id=receipt.container_id,
            container_name=f"fractal-sealed-{receipt.corpus_id}",
            role="sealed",
            authority_sha256=sealed_contract.contract_sha256,
            start_returncode=receipt.docker_start_returncode,
        )
        if (
            state.exit_code != receipt.container_exit_code
            or state.status != receipt.container_state_status
            or state.oom_killed != receipt.container_oom_killed
            or _digest_bytes(state.error.encode("utf-8")) != receipt.container_state_error_sha256
            or len(state.error.encode("utf-8")) != receipt.container_state_error_byte_count
        ):
            raise SealedContainerLauncherError("sealed terminal state differs from its receipt")
    if receipt.outcome == "failed":
        return
    assert receipt.output_reader_container_id is not None
    reader_create_arguments, reader_create = _verify_command_evidence(root, "output-reader-create")
    reader_start_arguments, reader_start = _verify_command_evidence(root, "output-reader-start")
    copy_arguments, copy_result = _verify_command_evidence(root, "output-copy")
    if reader_create.returncode != 0 or reader_start.returncode != 0 or copy_result.returncode != 0:
        raise SealedContainerLauncherError("successful output-copy command status differs")
    if reader_start_arguments.arguments != (
        "start",
        "--attach",
        receipt.output_reader_container_id,
    ):
        raise SealedContainerLauncherError("output reader start argument array differs")
    inventory = loads_container_output_inventory(
        _read_control(root / "container-output-inventory.json", "container output inventory")
    )
    copy_receipt = loads_output_copy_receipt(
        _read_control(root / "sealed-output-copy-receipt.json", "output copy receipt")
    )
    if (
        inventory.inventory_sha256 != receipt.output_reader_inventory_sha256
        or copy_receipt.receipt_sha256 != receipt.output_copy_receipt_sha256
        or copy_receipt.copied_root != receipt.copy_output_root
        or copy_receipt.tree_sha256 != receipt.output_tree_sha256
        or inventory.tree_sha256 != copy_receipt.tree_sha256
        or inventory.files != copy_receipt.files
        or inventory.file_count != copy_receipt.file_count
        or inventory.directory_count != copy_receipt.directory_count
        or inventory.byte_count != copy_receipt.byte_count
    ):
        raise SealedContainerLauncherError("sealed source inventory and host copy receipt differ")
    copied = Path(copy_receipt.copied_root)
    if copy_arguments.arguments != (
        "cp",
        f"{receipt.output_reader_container_id}:/output/.",
        str(copied),
    ):
        raise SealedContainerLauncherError("output copy did not use the read-only reader identity")
    observed_copy = _output_copy_receipt(copied)
    if observed_copy != copy_receipt:
        raise SealedContainerLauncherError("sealed host output changed after copy")
    reader_inspect = _read_control(root / "output-reader-inspect.json", "output reader inspect")
    if _digest_bytes(reader_inspect) != receipt.output_reader_inspect_sha256:
        raise SealedContainerLauncherError("output reader inspect differs from its receipt")
    if sealed_contract is not None:
        if reader_create_arguments.arguments != tuple(
            _copyout_create_arguments(
                sealed_contract.geometry,
                authority_sha256=sealed_contract.contract_sha256,
            )
        ):
            raise SealedContainerLauncherError("output reader create argument array differs")
        reader_argv = (
            _PYTHON,
            "-m",
            _MODULE,
            "inventory-output",
            "--root",
            sealed_contract.geometry.output_root,
        )
        reader_state = _verify_docker_inspect(
            reader_inspect,
            sealed_contract.geometry,
            reader_argv,
            container_id=receipt.output_reader_container_id,
            container_name=f"fractal-copyout-{receipt.corpus_id}",
            role="output-reader",
            authority_sha256=sealed_contract.contract_sha256,
            start_returncode=receipt.output_reader_start_returncode,
            include_bind_mounts=False,
            output_read_only=True,
        )
        if (
            reader_state.exit_code != receipt.output_reader_exit_code
            or reader_state.oom_killed != receipt.output_reader_oom_killed
        ):
            raise SealedContainerLauncherError("output reader state differs from its receipt")


def _output_copy_receipt(root: Path) -> OutputCopyReceipt:
    tree = digest_directory_tree(root)
    files: list[OutputFileDigest] = []
    for path in sorted(
        root.rglob("*"),
        key=lambda item: str(item.relative_to(root)).encode("utf-8"),
    ):
        metadata = path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            relative = path.relative_to(root).as_posix()
            files.append(
                OutputFileDigest(
                    relative_path=relative,
                    byte_count=metadata.st_size,
                    sha256=digest_regular_file(path, label=relative),
                )
            )
    return OutputCopyReceipt(
        copied_root=str(root),
        tree_sha256=tree.sha256,
        file_count=tree.file_count,
        directory_count=tree.directory_count,
        byte_count=tree.byte_count,
        files=tuple(files),
    )


def _container_output_inventory(root: Path) -> ContainerOutputInventory:
    copied = _output_copy_receipt(root)
    return ContainerOutputInventory(
        tree_sha256=copied.tree_sha256,
        file_count=copied.file_count,
        directory_count=copied.directory_count,
        byte_count=copied.byte_count,
        files=copied.files,
    )


def loads_container_output_inventory(encoded: bytes) -> ContainerOutputInventory:
    inventory = ContainerOutputInventory.from_dict(
        _parse_json_object(encoded, label="container output inventory")
    )
    if encoded != inventory.canonical_file_bytes():
        raise SealedContainerLauncherError("container output inventory is not canonical")
    return inventory


def _copyout_create_arguments(
    geometry: LauncherGeometry,
    *,
    authority_sha256: str,
) -> list[str]:
    argv = (_PYTHON, "-m", _MODULE, "inventory-output", "--root", geometry.output_root)
    arguments = [
        "create",
        "--name",
        f"fractal-copyout-{geometry.corpus_id}",
        "--platform",
        geometry.platform,
        "--user",
        f"{geometry.uid}:{geometry.gid}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--network",
        "none",
        "--memory",
        str(geometry.memory_limit_bytes),
        "--cpuset-cpus",
        geometry.cpuset_text,
        "--hostname",
        geometry.hostname,
    ]
    _append_labels(
        arguments,
        _container_labels(
            geometry,
            role="output-reader",
            authority_sha256=authority_sha256,
        ),
    )
    for row in geometry.environment:
        arguments.extend(("--env", f"{row.name}={row.value}"))
    arguments.extend(
        (
            "--tmpfs",
            _tmpfs(geometry),
            "--mount",
            _output_mount(geometry, read_only=True),
            "--entrypoint",
            argv[0],
            geometry.oci_image_digest,
            *argv[1:],
        )
    )
    return arguments


def _infer_consumed_failure_stage(root: Path) -> str:
    milestones = (
        (_LAUNCH_RECEIPT_FILENAME, "launch-receipt-verification"),
        ("sealed-output-copy-receipt.json", "output-copy-verification"),
        ("output-copy-docker-result.json", "output-copy-verification"),
        ("output-copy-docker-argv.json", "output-copy"),
        ("output-reader-inspect.json", "output-reader-verification"),
        ("output-reader-start-docker-result.json", "output-reader-evidence"),
        ("output-reader-start-docker-argv.json", "output-reader-start"),
        ("output-reader-create-docker-result.json", "output-reader-start"),
        ("output-reader-create-docker-argv.json", "output-reader-create"),
        ("sealed-inspect.json", "sealed-evidence-verification"),
        ("sealed-start-docker-result.json", "sealed-evidence"),
        ("sealed-start-docker-argv.json", "sealed-start"),
        ("sealed-create-docker-result.json", "sealed-create"),
        ("sealed-create-docker-argv.json", "sealed-create"),
    )
    for filename, stage in milestones:
        if os.path.lexists(root / filename):
            return stage
    return "sealed-create"


def _retained_container_id(path: Path) -> str | None:
    if not os.path.lexists(path):
        return None
    try:
        encoded = _read_control(path, "retained Docker container ID")
        value = encoded.decode("ascii", errors="strict").strip()
    except (SealedContainerLauncherError, UnicodeDecodeError):
        return None
    return value if _CONTAINER_ID.fullmatch(value) is not None else None


_SEALED_FAILURE_DOCKER_OPERATIONS = (
    "sealed-create",
    "sealed-start",
    "output-reader-create",
    "output-reader-start",
    "output-copy",
)


def _failure_operation_arguments(
    operation: str,
    *,
    sealed_contract: SealedLaunchContract,
    sealed_container_id: str | None,
    output_reader_container_id: str | None,
) -> tuple[str, ...]:
    geometry = sealed_contract.geometry
    if operation == "sealed-create":
        return tuple(
            _create_arguments(
                geometry,
                name=f"fractal-sealed-{geometry.corpus_id}",
                argv=sealed_contract.argv,
                role="sealed",
                authority_sha256=sealed_contract.contract_sha256,
            )
        )
    if operation == "sealed-start":
        if sealed_container_id is None:
            raise SealedContainerLauncherError("sealed start lacks its created container identity")
        return ("start", "--attach", "--interactive", sealed_container_id)
    if operation == "output-reader-create":
        return tuple(
            _copyout_create_arguments(
                geometry,
                authority_sha256=sealed_contract.contract_sha256,
            )
        )
    if operation == "output-reader-start":
        if output_reader_container_id is None:
            raise SealedContainerLauncherError("output reader start lacks its container identity")
        return ("start", "--attach", output_reader_container_id)
    if operation == "output-copy":
        if output_reader_container_id is None:
            raise SealedContainerLauncherError("output copy lacks its reader container identity")
        return (
            "cp",
            f"{output_reader_container_id}:{geometry.output_root}/.",
            geometry.copy_output_root,
        )
    raise SealedContainerLauncherError("sealed failure contains an unknown Docker operation")


def _load_failure_command_evidence(
    root: Path,
    operation: str,
    *,
    filenames: frozenset[str],
) -> tuple[DockerArgumentRecord, DockerCommandResultRecord | None]:
    argument = loads_docker_argument_record(
        _read_control(root / f"{operation}-docker-argv.json", f"{operation} Docker arguments")
    )
    if argument.operation != operation:
        raise SealedContainerLauncherError(f"{operation} Docker argument operation differs")
    result_name = f"{operation}-docker-result.json"
    stdout_name = f"{operation}-docker-stdout.log"
    stderr_name = f"{operation}-docker-stderr.log"
    result_present = result_name in filenames
    stdout_present = stdout_name in filenames
    stderr_present = stderr_name in filenames
    if stderr_present and not stdout_present:
        raise SealedContainerLauncherError(f"{operation} Docker stream evidence order differs")
    if not result_present:
        return argument, None
    result = loads_docker_command_result_record(
        _read_control(root / result_name, f"{operation} Docker result")
    )
    if result.operation != operation or result.argument_record_sha256 != argument.record_sha256:
        raise SealedContainerLauncherError(f"{operation} Docker result binding differs")
    if stdout_present != stderr_present:
        raise SealedContainerLauncherError(f"{operation} Docker result has partial streams")
    if stdout_present:
        _verify_command_evidence(root, operation)
    elif operation != "sealed-start":
        raise SealedContainerLauncherError(f"{operation} Docker result lacks retained streams")
    return argument, result


def _failure_error_matches(
    record: SealedLaunchFailureErrorRecord,
    error: SealedContainerLauncherError,
) -> bool:
    error_type = type(error)
    return (
        record.exception_class == f"{error_type.__module__}.{error_type.__qualname__}"
        and record.redacted_message == str(error)
    )


def verify_sealed_launch_failure_evidence(
    receipt: SealedLaunchFailureReceipt,
    *,
    audit_root: str | Path,
    sealed_contract: SealedLaunchContract,
) -> None:
    if not isinstance(receipt, SealedLaunchFailureReceipt):
        raise SealedContainerLauncherError("sealed failure evidence requires a typed receipt")
    if not isinstance(sealed_contract, SealedLaunchContract):
        raise SealedContainerLauncherError("sealed failure verifier requires the exact contract")
    root = Path(audit_root)
    _private_directory(root)
    persisted = load_sealed_launch_failure_receipt(root / _LAUNCH_FAILURE_RECEIPT_FILENAME)
    if persisted != receipt:
        raise SealedContainerLauncherError("persisted sealed failure receipt differs")
    observed = _snapshot_sealed_evidence(
        root,
        excluded_filenames=frozenset({_LAUNCH_FAILURE_RECEIPT_FILENAME}),
    )
    if observed != receipt.evidence_files:
        raise SealedContainerLauncherError("sealed failure evidence membership or bytes differ")
    if receipt.failure_stage != _infer_consumed_failure_stage(root):
        raise SealedContainerLauncherError("sealed failure stage differs from retained evidence")
    error_encoded = _read_control(
        root / _LAUNCH_FAILURE_ERROR_FILENAME,
        "sealed launch failure error",
    )
    error_record = loads_sealed_launch_failure_error_record(error_encoded)
    if (
        _digest_bytes(error_encoded) != receipt.failure_error_sha256
        or len(error_encoded) != receipt.failure_error_byte_count
        or error_record.failure_stage != receipt.failure_stage
    ):
        raise SealedContainerLauncherError("sealed failure error evidence differs")
    contract_fields = (
        receipt.sealed_launcher_contract_sha256 == sealed_contract.contract_sha256,
        receipt.corpus_id == sealed_contract.geometry.corpus_id,
        receipt.preflight_launcher_contract_sha256
        == sealed_contract.preflight_launcher_contract_sha256,
        receipt.preflight_receipt_sha256 == sealed_contract.preflight_receipt_sha256,
        receipt.runtime_plan_transition_receipt_sha256
        == sealed_contract.runtime_plan_transition_receipt_sha256,
        receipt.registered_plan_instantiation_receipt_sha256
        == sealed_contract.registered_plan_instantiation_receipt_sha256,
        receipt.production_run_closure_binding_receipt_sha256
        == sealed_contract.production_run_closure_binding_receipt_sha256,
    )
    if not all(contract_fields):
        raise SealedContainerLauncherError("sealed failure protocol binding differs from contract")
    marker = loads_launcher_attempt_marker(
        _read_control(root / "sealed-launcher-attempt-marker.json", "launcher attempt marker")
    )
    if (
        marker.marker_sha256 != receipt.launcher_attempt_marker_sha256
        or marker.corpus_id != receipt.corpus_id
        or marker.sealed_launcher_contract_sha256 != receipt.sealed_launcher_contract_sha256
        or marker.preflight_receipt_sha256 != receipt.preflight_receipt_sha256
        or marker.runtime_plan_transition_receipt_sha256
        != receipt.runtime_plan_transition_receipt_sha256
        or marker.output_volume != sealed_contract.geometry.output_volume
        or marker.output_volume_subpath != sealed_contract.geometry.output_volume_subpath
        or marker.stdin_secret_sha256 != receipt.stdin_secret_sha256
        or marker.stdin_secret_byte_count != receipt.stdin_secret_byte_count
    ):
        raise SealedContainerLauncherError("sealed failure marker differs from its receipt")
    filenames = frozenset(item.relative_path for item in observed)
    argument_operations = {
        item.relative_path.removesuffix("-docker-argv.json")
        for item in observed
        if item.relative_path.endswith("-docker-argv.json")
    }
    result_operations = {
        item.relative_path.removesuffix("-docker-result.json")
        for item in observed
        if item.relative_path.endswith("-docker-result.json")
    }
    allowed = frozenset(_SEALED_FAILURE_DOCKER_OPERATIONS)
    observed_order = tuple(
        operation
        for operation in _SEALED_FAILURE_DOCKER_OPERATIONS
        if operation in argument_operations
    )
    if (
        not argument_operations.issubset(allowed)
        or observed_order != _SEALED_FAILURE_DOCKER_OPERATIONS[: len(observed_order)]
        or not result_operations.issubset(argument_operations)
    ):
        raise SealedContainerLauncherError("sealed failure Docker operation inventory differs")
    if receipt.sealed_container_id != _retained_container_id(
        root / "sealed-create-docker-stdout.log"
    ):
        raise SealedContainerLauncherError("sealed failure container identity differs")
    if receipt.output_reader_container_id != _retained_container_id(
        root / "output-reader-create-docker-stdout.log"
    ):
        raise SealedContainerLauncherError("sealed failure reader identity differs")
    command_results: dict[str, DockerCommandResultRecord | None] = {}
    for operation in observed_order:
        arguments, result = _load_failure_command_evidence(
            root,
            operation,
            filenames=filenames,
        )
        expected = _failure_operation_arguments(
            operation,
            sealed_contract=sealed_contract,
            sealed_container_id=receipt.sealed_container_id,
            output_reader_container_id=receipt.output_reader_container_id,
        )
        if arguments.arguments != expected:
            raise SealedContainerLauncherError(f"{operation} Docker argument array differs")
        command_results[operation] = result

    sealed_inspect_name = "sealed-inspect.json"
    if sealed_inspect_name in filenames:
        sealed_start = command_results.get("sealed-start")
        if sealed_start is None or receipt.sealed_container_id is None:
            raise SealedContainerLauncherError("sealed inspect lacks its start result")
        sealed_inspect = _read_control(root / sealed_inspect_name, "sealed Docker inspect")
        try:
            sealed_state = _verify_docker_inspect(
                sealed_inspect,
                sealed_contract.geometry,
                sealed_contract.argv,
                container_id=receipt.sealed_container_id,
                container_name=f"fractal-sealed-{receipt.corpus_id}",
                role="sealed",
                authority_sha256=sealed_contract.contract_sha256,
                start_returncode=sealed_start.returncode,
            )
            if receipt.failure_stage not in {
                "sealed-evidence-verification",
                "launch-receipt-verification",
            } and (sealed_state.exit_code != 0 or sealed_state.oom_killed):
                raise SealedContainerLauncherError("sealed failure state precedes later evidence")
        except SealedContainerLauncherError as exc:
            if (
                receipt.failure_stage != "sealed-evidence-verification"
                or not _failure_error_matches(error_record, exc)
            ):
                raise SealedContainerLauncherError("sealed failure inspect/state differs") from exc
    elif receipt.failure_stage not in {"sealed-create", "sealed-start", "sealed-evidence"}:
        raise SealedContainerLauncherError("sealed failure lacks required inspect evidence")

    reader_inspect_name = "output-reader-inspect.json"
    if reader_inspect_name in filenames:
        reader_start = command_results.get("output-reader-start")
        if reader_start is None or receipt.output_reader_container_id is None:
            raise SealedContainerLauncherError("output reader inspect lacks its start result")
        reader_argv = (
            _PYTHON,
            "-m",
            _MODULE,
            "inventory-output",
            "--root",
            sealed_contract.geometry.output_root,
        )
        try:
            reader_state = _verify_docker_inspect(
                _read_control(root / reader_inspect_name, "output reader inspect"),
                sealed_contract.geometry,
                reader_argv,
                container_id=receipt.output_reader_container_id,
                container_name=f"fractal-copyout-{receipt.corpus_id}",
                role="output-reader",
                authority_sha256=sealed_contract.contract_sha256,
                start_returncode=reader_start.returncode,
                include_bind_mounts=False,
                output_read_only=True,
            )
            if reader_state.exit_code != 0 or reader_state.oom_killed:
                raise SealedContainerLauncherError("output reader did not exit cleanly")
        except SealedContainerLauncherError as exc:
            if receipt.failure_stage != "output-reader-verification" or not _failure_error_matches(
                error_record, exc
            ):
                raise SealedContainerLauncherError("output reader inspect/state differs") from exc
    elif receipt.failure_stage in {
        "output-reader-verification",
        "output-copy",
        "output-copy-verification",
        "launch-receipt-verification",
    }:
        raise SealedContainerLauncherError("sealed failure lacks required reader inspect evidence")


def _publish_consumed_launch_failure(
    *,
    sealed: SealedLaunchContract,
    preflight: PreflightLaunchContract,
    preflight_receipt: RuntimePreflightReceipt,
    transition: RuntimePlanTransitionReceipt,
    instantiation: RegisteredPlanInstantiationReceipt,
    verified_closure: VerifiedProductionRunClosure,
    volume: VolumeInitializationReceipt,
    secret: bytes,
    audit_root: Path,
    error: BaseException,
) -> SealedLaunchFailureReceipt | None:
    target = audit_root / _LAUNCH_FAILURE_RECEIPT_FILENAME
    if os.path.lexists(target):
        return load_sealed_launch_failure_receipt(target)
    launch_target = audit_root / _LAUNCH_RECEIPT_FILENAME
    if os.path.lexists(launch_target):
        try:
            terminal = load_sealed_launch_receipt(launch_target)
        except SealedContainerLauncherError:
            terminal = None
        if terminal is not None and terminal.outcome == "failed":
            return None
    marker = loads_launcher_attempt_marker(
        _read_control(
            audit_root / "sealed-launcher-attempt-marker.json",
            "launcher attempt marker",
        )
    )
    failure_stage = _infer_consumed_failure_stage(audit_root)
    message = str(error)
    redactions = {repr(secret), secret.hex(), base64.b64encode(secret).decode("ascii")}
    try:
        redactions.add(secret.decode("utf-8", errors="strict"))
    except UnicodeDecodeError:
        pass
    for value in sorted(redactions, key=len, reverse=True):
        if value:
            message = message.replace(value, "[secret-redacted]")
    while len(message.encode("utf-8", errors="strict")) > 8192:
        message = message[: len(message) // 2]
    error_type = type(error)
    error_record = SealedLaunchFailureErrorRecord(
        failure_stage=failure_stage,
        exception_class=f"{error_type.__module__}.{error_type.__qualname__}",
        redacted_message=message,
    )
    encoded_error = error_record.canonical_file_bytes()
    if secret in encoded_error:
        error_record = replace(error_record, redacted_message="[secret-redacted]")
        encoded_error = error_record.canonical_file_bytes()
    _write_receipt(
        encoded_error,
        audit_root / _LAUNCH_FAILURE_ERROR_FILENAME,
        label="sealed launch failure error",
    )
    evidence = _snapshot_sealed_evidence(
        audit_root,
        excluded_filenames=frozenset({_LAUNCH_FAILURE_RECEIPT_FILENAME}),
    )
    failure = SealedLaunchFailureReceipt(
        corpus_id=sealed.geometry.corpus_id,
        failure_stage=failure_stage,
        failure_error_sha256=_digest_bytes(encoded_error),
        failure_error_byte_count=len(encoded_error),
        sealed_launcher_contract_sha256=sealed.contract_sha256,
        preflight_launcher_contract_sha256=preflight.contract_sha256,
        preflight_receipt_sha256=preflight_receipt.receipt_sha256,
        runtime_plan_transition_receipt_sha256=transition.receipt_sha256,
        registered_plan_instantiation_receipt_sha256=instantiation.receipt_sha256,
        production_run_closure_binding_receipt_sha256=verified_closure.binding.receipt_sha256,
        volume_initialization_receipt_sha256=volume.receipt_sha256,
        launcher_attempt_marker_sha256=marker.marker_sha256,
        stdin_secret_sha256=_digest_bytes(secret),
        stdin_secret_byte_count=len(secret),
        sealed_container_id=_retained_container_id(audit_root / "sealed-create-docker-stdout.log"),
        output_reader_container_id=_retained_container_id(
            audit_root / "output-reader-create-docker-stdout.log"
        ),
        evidence_inventory_sha256=_sealed_evidence_inventory_sha256(evidence),
        evidence_files=evidence,
    )
    _write_receipt(
        failure.canonical_file_bytes(),
        target,
        label="sealed launch failure receipt",
    )
    verify_sealed_launch_failure_evidence(
        failure,
        audit_root=audit_root,
        sealed_contract=sealed,
    )
    return failure


def _retain_consumed_launch_failure(
    function: Callable[..., SealedLaunchReceipt],
) -> Callable[..., SealedLaunchReceipt]:
    signature = python_inspect.signature(function)

    @functools.wraps(function)
    def retained(*args: object, **kwargs: object) -> SealedLaunchReceipt:
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            bound = signature.bind(*args, **kwargs)
            root = Path(bound.arguments["audit_root"])
            marker = root / "sealed-launcher-attempt-marker.json"
            if os.path.lexists(marker) and not os.path.lexists(
                root / _LAUNCH_FAILURE_RECEIPT_FILENAME
            ):
                _publish_consumed_launch_failure(
                    sealed=bound.arguments["sealed"],
                    preflight=bound.arguments["preflight"],
                    preflight_receipt=bound.arguments["preflight_receipt"],
                    transition=bound.arguments["transition"],
                    instantiation=bound.arguments["instantiation"],
                    verified_closure=bound.arguments["verified_closure"],
                    volume=bound.arguments["volume"],
                    secret=bound.arguments["secret"],
                    audit_root=root,
                    error=error,
                )
            raise

    return retained


@_retain_consumed_launch_failure
def launch_sealed_once(
    sealed: SealedLaunchContract,
    preflight: PreflightLaunchContract,
    preflight_receipt: RuntimePreflightReceipt,
    transition: RuntimePlanTransitionReceipt,
    instantiation: RegisteredPlanInstantiationReceipt,
    verified_closure: VerifiedProductionRunClosure,
    volume: VolumeInitializationReceipt,
    run_claim: VerifiedRunClaimCapability,
    *,
    secret: bytes,
    audit_root: str | Path,
    docker: DockerRunner | None = None,
) -> SealedLaunchReceipt:
    """Consume the host marker, attach once, retain evidence, and copy output."""

    if not isinstance(run_claim, VerifiedRunClaimCapability):
        raise SealedContainerLauncherError("sealed launch requires typed RUN_CLAIMED authority")
    root = Path(audit_root)
    _private_directory(root)
    _verify_volume_binding(sealed, volume)
    if not isinstance(verified_closure, VerifiedProductionRunClosure):
        raise SealedContainerLauncherError(
            "sealed launch requires verified production closure authority"
        )
    verified_closure.assert_current()
    plan = verify_sealed_transition(
        sealed,
        preflight,
        preflight_receipt,
        transition,
        instantiation,
        verified_closure.binding,
    )
    try:
        runtime_claim = run_claim.require_launch(
            manifest_sha256=sealed.manifest_sha256,
            corpus_id=sealed.geometry.corpus_id,
            runtime_plan_sha256=plan.plan_sha256,
            output_namespace_uri=Path(sealed.geometry.copy_output_root).as_uri(),
        )
    except ExecutionClaimError as exc:
        raise SealedContainerLauncherError(f"RUN_CLAIMED launch gate failed: {exc}") from exc
    expected_secret = runtime_claim.canonical_file_bytes()
    if secret != expected_secret:
        raise SealedContainerLauncherError(
            "stdin bytes must equal the freshly verified runtime-claim receipt"
        )
    if not _MIN_SECRET_BYTES <= len(secret) <= _MAX_SECRET_BYTES:
        raise SealedContainerLauncherError("runtime-claim receipt exceeds launcher stdin bounds")
    secret_sha256 = _digest_bytes(secret)
    marker = LauncherAttemptMarker(
        corpus_id=sealed.geometry.corpus_id,
        sealed_launcher_contract_sha256=sealed.contract_sha256,
        runtime_plan_transition_receipt_sha256=transition.receipt_sha256,
        preflight_receipt_sha256=preflight_receipt.receipt_sha256,
        runtime_claim_receipt_sha256=runtime_claim.receipt_sha256,
        claim_state_sha256=runtime_claim.claim_state_sha256,
        claim_ledger_commit=runtime_claim.claim_ledger_commit,
        provider_identity_sha256=runtime_claim.provider_identity_sha256,
        beacon_receipt_sha256=runtime_claim.beacon_receipt_sha256,
        beacon_bytes_sha256=runtime_claim.beacon_bytes_sha256,
        derived_seed_sha256=runtime_claim.derived_seed_sha256,
        permutation_seed=runtime_claim.permutation_seed,
        output_aggregate_identity=runtime_claim.output_aggregate_identity,
        output_volume=sealed.geometry.output_volume,
        output_volume_subpath=sealed.geometry.output_volume_subpath,
        stdin_secret_sha256=secret_sha256,
        stdin_secret_byte_count=len(secret),
    )
    marker_path = root / "sealed-launcher-attempt-marker.json"
    _write_receipt(marker.canonical_file_bytes(), marker_path, label="launcher attempt marker")
    active = docker if docker is not None else SubprocessDockerRunner()
    name = f"fractal-sealed-{sealed.geometry.corpus_id}"
    container_id = _container_id(
        _run_checked_mutation(
            active,
            _create_arguments(
                sealed.geometry,
                name=name,
                argv=sealed.argv,
                role="sealed",
                authority_sha256=sealed.contract_sha256,
            ),
            audit_root=root,
            operation="sealed-create",
            label="Docker sealed container creation",
        )
    )
    attached, _, _ = _execute_mutating_docker_command(
        active,
        ("start", "--attach", "--interactive", container_id),
        audit_root=root,
        operation="sealed-start",
        input_bytes=secret,
        forbidden_bytes=secret,
    )
    _persist_bytes(root / "sealed-attached-stdout.log", attached.stdout, label="attached stdout")
    _persist_bytes(root / "sealed-attached-stderr.log", attached.stderr, label="attached stderr")
    inspect, retained_stdout, retained_stderr = _inspect_and_logs(
        active,
        container_id,
        root,
        "sealed",
        secret=secret,
    )
    state = _verify_docker_inspect(
        inspect,
        sealed.geometry,
        sealed.argv,
        container_id=container_id,
        container_name=name,
        role="sealed",
        authority_sha256=sealed.contract_sha256,
        start_returncode=attached.returncode,
    )
    if retained_stdout != attached.stdout or retained_stderr != attached.stderr:
        raise SealedContainerLauncherError("retained sealed logs differ from attached output")
    state_error = state.error.encode("utf-8", errors="strict")
    shared_receipt_fields: dict[str, object] = {
        "corpus_id": sealed.geometry.corpus_id,
        "sealed_launcher_contract_sha256": sealed.contract_sha256,
        "preflight_launcher_contract_sha256": preflight.contract_sha256,
        "preflight_receipt_sha256": preflight_receipt.receipt_sha256,
        "runtime_claim_receipt_sha256": runtime_claim.receipt_sha256,
        "claim_state_sha256": runtime_claim.claim_state_sha256,
        "claim_ledger_commit": runtime_claim.claim_ledger_commit,
        "provider_identity_sha256": runtime_claim.provider_identity_sha256,
        "beacon_receipt_sha256": runtime_claim.beacon_receipt_sha256,
        "beacon_bytes_sha256": runtime_claim.beacon_bytes_sha256,
        "derived_seed_sha256": runtime_claim.derived_seed_sha256,
        "permutation_seed": runtime_claim.permutation_seed,
        "output_aggregate_identity": runtime_claim.output_aggregate_identity,
        "runtime_plan_transition_receipt_sha256": transition.receipt_sha256,
        "registered_plan_instantiation_receipt_sha256": instantiation.receipt_sha256,
        "production_run_closure_binding_receipt_sha256": (verified_closure.binding.receipt_sha256),
        "volume_initialization_receipt_sha256": volume.receipt_sha256,
        "launcher_attempt_marker_sha256": marker.marker_sha256,
        "container_id": container_id,
        "output_volume": sealed.geometry.output_volume,
        "output_volume_subpath": sealed.geometry.output_volume_subpath,
        "stdin_secret_sha256": secret_sha256,
        "stdin_secret_byte_count": len(secret),
        "docker_start_returncode": attached.returncode,
        "container_state_status": state.status,
        "container_exit_code": state.exit_code,
        "container_oom_killed": state.oom_killed,
        "container_state_error_sha256": _digest_bytes(state_error),
        "container_state_error_byte_count": len(state_error),
        "attached_stdout_sha256": _digest_bytes(attached.stdout),
        "attached_stdout_byte_count": len(attached.stdout),
        "attached_stderr_sha256": _digest_bytes(attached.stderr),
        "attached_stderr_byte_count": len(attached.stderr),
        "inspect_sha256": _digest_bytes(inspect),
        "retained_stdout_sha256": _digest_bytes(retained_stdout),
        "retained_stdout_byte_count": len(retained_stdout),
        "retained_stderr_sha256": _digest_bytes(retained_stderr),
        "retained_stderr_byte_count": len(retained_stderr),
    }
    if attached.returncode != 0 or state.oom_killed:
        evidence_files = _snapshot_sealed_evidence(root)
        failed = SealedLaunchReceipt(
            **shared_receipt_fields,
            outcome="failed",
            output_reader_container_id=None,
            output_reader_start_returncode=None,
            output_reader_exit_code=None,
            output_reader_oom_killed=None,
            output_reader_inspect_sha256=None,
            output_reader_inventory_sha256=None,
            copy_output_root=None,
            output_copy_receipt_sha256=None,
            output_tree_sha256=None,
            evidence_inventory_sha256=_sealed_evidence_inventory_sha256(evidence_files),
            evidence_files=evidence_files,
        )
        _write_receipt(
            failed.canonical_file_bytes(),
            root / _LAUNCH_RECEIPT_FILENAME,
            label="failed sealed launch receipt",
        )
        verify_sealed_launch_evidence(failed, audit_root=root, sealed_contract=sealed)
        raise SealedContainerLauncherError(
            "sealed invocation failed and consumed the attempt; "
            f"retained receipt={failed.receipt_sha256}"
        )
    reader_id = _container_id(
        _run_checked_mutation(
            active,
            _copyout_create_arguments(
                sealed.geometry,
                authority_sha256=sealed.contract_sha256,
            ),
            audit_root=root,
            operation="output-reader-create",
            label="Docker output reader creation",
        )
    )
    reader_attached = _run_checked_mutation(
        active,
        ("start", "--attach", reader_id),
        audit_root=root,
        operation="output-reader-start",
        label="Docker output inventory",
    )
    source_inventory = loads_container_output_inventory(reader_attached.stdout)
    reader_inspect, reader_stdout, reader_stderr = _inspect_and_logs(
        active,
        reader_id,
        root,
        "output-reader",
    )
    reader_argv = (
        _PYTHON,
        "-m",
        _MODULE,
        "inventory-output",
        "--root",
        sealed.geometry.output_root,
    )
    reader_state = _verify_docker_inspect(
        reader_inspect,
        sealed.geometry,
        reader_argv,
        container_id=reader_id,
        container_name=f"fractal-copyout-{sealed.geometry.corpus_id}",
        role="output-reader",
        authority_sha256=sealed.contract_sha256,
        start_returncode=reader_attached.returncode,
        include_bind_mounts=False,
        output_read_only=True,
    )
    if reader_state.exit_code != 0 or reader_state.oom_killed:
        raise SealedContainerLauncherError("output reader did not exit cleanly")
    if reader_stdout != reader_attached.stdout or reader_stderr != reader_attached.stderr:
        raise SealedContainerLauncherError("retained output-reader logs differ")
    _write_receipt(
        source_inventory.canonical_file_bytes(),
        root / "container-output-inventory.json",
        label="container output inventory",
    )
    copied = Path(sealed.geometry.copy_output_root)
    copy_identity = _prepare_copy_destination(copied)
    _run_checked_mutation(
        active,
        ("cp", f"{reader_id}:{sealed.geometry.output_root}/.", str(copied)),
        audit_root=root,
        operation="output-copy",
        label="Docker sealed output copy",
    )
    _verify_private_copy_tree(copied, copy_identity)
    output = _output_copy_receipt(copied)
    if (
        output.tree_sha256 != source_inventory.tree_sha256
        or output.file_count != source_inventory.file_count
        or output.directory_count != source_inventory.directory_count
        or output.byte_count != source_inventory.byte_count
        or output.files != source_inventory.files
    ):
        raise SealedContainerLauncherError("host output copy differs from the read-only source")
    _write_receipt(
        output.canonical_file_bytes(),
        root / "sealed-output-copy-receipt.json",
        label="sealed output-copy receipt",
    )
    evidence_files = _snapshot_sealed_evidence(root)
    receipt = SealedLaunchReceipt(
        **shared_receipt_fields,
        outcome="succeeded",
        output_reader_container_id=reader_id,
        output_reader_start_returncode=reader_attached.returncode,
        output_reader_exit_code=reader_state.exit_code,
        output_reader_oom_killed=reader_state.oom_killed,
        output_reader_inspect_sha256=_digest_bytes(reader_inspect),
        output_reader_inventory_sha256=source_inventory.inventory_sha256,
        copy_output_root=str(copied),
        output_copy_receipt_sha256=output.receipt_sha256,
        output_tree_sha256=output.tree_sha256,
        evidence_inventory_sha256=_sealed_evidence_inventory_sha256(evidence_files),
        evidence_files=evidence_files,
    )
    _write_receipt(
        receipt.canonical_file_bytes(),
        root / _LAUNCH_RECEIPT_FILENAME,
        label="sealed launch receipt",
    )
    verify_sealed_launch_evidence(receipt, audit_root=root, sealed_contract=sealed)
    return receipt


def _initialize_output_directory(path: str | Path) -> None:
    target = Path(path)
    expected_root = Path("/volume")
    if not target.is_absolute() or target.parent != expected_root or target.name in {"", ".", ".."}:
        raise SealedContainerLauncherError("initializer path must be one direct /volume child")
    if os.geteuid() != 0 or os.getegid() != 0:
        raise SealedContainerLauncherError("output initializer must run as root")
    try:
        os.mkdir(target, 0o700)
        os.chown(target, _UID, _GID, follow_symlinks=False)
        os.chmod(target, 0o700, follow_symlinks=False)
        metadata = target.stat(follow_symlinks=False)
    except OSError as exc:
        raise SealedContainerLauncherError("cannot create the fresh output subpath") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != _UID
        or metadata.st_gid != _GID
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or any(target.iterdir())
    ):
        raise SealedContainerLauncherError("fresh output subpath has wrong ownership or mode")


def _capture_preflight_from_stdin() -> RuntimePreflightReceipt:
    encoded = sys.stdin.buffer.read(_MAX_CONTROL_BYTES + 1)
    contract = loads_preflight_launch_contract(encoded)
    receipt = capture_runtime_preflight(
        launcher_contract_sha256=contract.contract_sha256,
        oci_image_digest=contract.geometry.oci_image_digest,
        code_commit=contract.geometry.code_commit,
        hostname=contract.geometry.hostname,
        artifact_mounts=tuple(
            sorted(
                (
                    mount.runtime_mount()
                    for mount in contract.geometry.bind_mounts
                    if mount.attested_artifact
                ),
                key=lambda item: item.root.encode("utf-8"),
            )
        ),
        environment=contract.geometry.environment_dict,
        output_root=contract.geometry.output_root,
        tmpfs_root=contract.geometry.tmpfs_root,
    )
    sys.stdout.buffer.write(receipt.canonical_file_bytes())
    sys.stdout.buffer.flush()
    return receipt


def _read_secret_fd(fd: int) -> bytes:
    if type(fd) is not int or fd < 0:
        raise SealedContainerLauncherError("secret-fd must be nonnegative")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(4096, _MAX_SECRET_BYTES + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_SECRET_BYTES:
            raise SealedContainerLauncherError("secret FD exceeds 4096 bytes")
    secret = b"".join(chunks)
    if len(secret) < _MIN_SECRET_BYTES:
        raise SealedContainerLauncherError("secret FD contains fewer than 32 bytes")
    return secret


def _verify_closure_from_finalization(
    request_path: Path,
    receipt_path: Path,
    preflight: PreflightLaunchContract,
    transition: RuntimePlanTransitionReceipt,
) -> VerifiedProductionRunClosure:
    # Lazy import keeps the launcher usable by the production-control verifier.
    from .production_controls import verify_production_run_closure_authority

    return verify_production_run_closure_authority(
        finalization_request_path=request_path,
        finalization_receipt_path=receipt_path,
        preflight=preflight,
        transition=transition,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fractal-sealed-container-launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize-output")
    initialize.add_argument("--path", required=True)
    subparsers.add_parser("capture-preflight")
    inventory = subparsers.add_parser("inventory-output")
    inventory.add_argument("--root", required=True, type=Path)

    volume = subparsers.add_parser("initialize-volume")
    volume.add_argument("--contract", type=Path, required=True)
    volume.add_argument("--audit-root", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--contract", type=Path, required=True)
    preflight.add_argument("--volume-receipt", type=Path, required=True)
    preflight.add_argument("--audit-root", type=Path, required=True)

    transition = subparsers.add_parser("materialize-transition")
    transition.add_argument("--contract", type=Path, required=True)
    transition.add_argument("--preflight-receipt", type=Path, required=True)
    transition.add_argument("--transition-receipt", type=Path, required=True)

    instantiate = subparsers.add_parser("instantiate-plan")
    instantiate.add_argument("--preflight-contract", type=Path, required=True)
    instantiate.add_argument("--preflight-receipt", type=Path, required=True)
    instantiate.add_argument("--transition-receipt", type=Path, required=True)
    instantiate.add_argument("--finalization-request", type=Path, required=True)
    instantiate.add_argument("--finalization-receipt", type=Path, required=True)
    instantiate.add_argument("--instantiation-receipt", type=Path, required=True)
    instantiate.add_argument("--sealed-contract", type=Path, required=True)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--preflight-contract", type=Path, required=True)
    launch.add_argument("--preflight-receipt", type=Path, required=True)
    launch.add_argument("--transition-receipt", type=Path, required=True)
    launch.add_argument("--instantiation-receipt", type=Path, required=True)
    launch.add_argument("--finalization-request", type=Path, required=True)
    launch.add_argument("--finalization-receipt", type=Path, required=True)
    launch.add_argument("--sealed-contract", type=Path, required=True)
    launch.add_argument("--volume-receipt", type=Path, required=True)
    launch.add_argument("--audit-root", type=Path, required=True)
    launch.add_argument("--runtime-claim-receipt", type=Path, required=True)

    verify_launch = subparsers.add_parser("verify-launch-evidence")
    verify_launch.add_argument("--receipt", type=Path, required=True)
    verify_launch.add_argument("--audit-root", type=Path, required=True)
    verify_launch.add_argument("--sealed-contract", type=Path, required=True)
    verify_launch.add_argument("--expected-receipt-sha256")

    verify_failure = subparsers.add_parser("verify-launch-failure-evidence")
    verify_failure.add_argument("--receipt", type=Path, required=True)
    verify_failure.add_argument("--audit-root", type=Path, required=True)
    verify_failure.add_argument("--sealed-contract", type=Path, required=True)
    verify_failure.add_argument("--expected-receipt-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "initialize-output":
        _initialize_output_directory(args.path)
        return 0
    if args.command == "capture-preflight":
        _capture_preflight_from_stdin()
        return 0
    if args.command == "inventory-output":
        if args.root != Path(_OUTPUT_ROOT):
            raise SealedContainerLauncherError("inventory root must equal /output")
        inventory = _container_output_inventory(args.root)
        sys.stdout.buffer.write(inventory.canonical_file_bytes())
        sys.stdout.buffer.flush()
        return 0
    if args.command == "initialize-volume":
        receipt = initialize_output_volume(
            load_preflight_launch_contract(args.contract),
            audit_root=args.audit_root,
        )
        print(receipt.receipt_sha256)
        return 0
    if args.command == "preflight":
        receipt = run_preflight_once(
            load_preflight_launch_contract(args.contract),
            load_volume_initialization_receipt(args.volume_receipt),
            audit_root=args.audit_root,
        )
        print(receipt.receipt_sha256)
        return 0
    if args.command == "materialize-transition":
        contract = load_preflight_launch_contract(args.contract)
        try:
            preflight = loads_runtime_preflight_receipt(
                _read_control(args.preflight_receipt, "runtime preflight receipt")
            )
        except RuntimeAttestationError as exc:
            raise SealedContainerLauncherError("runtime preflight receipt is invalid") from exc
        transition = materialize_runtime_plan_transition(contract, preflight)
        _write_receipt(
            transition.canonical_file_bytes(),
            args.transition_receipt,
            label="runtime-plan transition",
        )
        print(transition.receipt_sha256)
        return 0
    if args.command == "instantiate-plan":
        preflight_contract = load_preflight_launch_contract(args.preflight_contract)
        try:
            preflight_receipt = loads_runtime_preflight_receipt(
                _read_control(args.preflight_receipt, "runtime preflight receipt")
            )
        except RuntimeAttestationError as exc:
            raise SealedContainerLauncherError("runtime preflight receipt is invalid") from exc
        transition = load_runtime_plan_transition(args.transition_receipt)
        verified_closure = _verify_closure_from_finalization(
            args.finalization_request,
            args.finalization_receipt,
            preflight_contract,
            transition,
        )
        instantiation, sealed = instantiate_registered_runtime_plan(
            preflight_contract,
            preflight_receipt,
            transition,
            verified_closure=verified_closure,
        )
        _write_receipt(
            instantiation.canonical_file_bytes(),
            args.instantiation_receipt,
            label="registered-plan instantiation",
        )
        _write_receipt(
            sealed.canonical_file_bytes(),
            args.sealed_contract,
            label="sealed launch contract",
        )
        print(instantiation.receipt_sha256)
        print(sealed.contract_sha256)
        return 0
    if args.command == "launch":
        raise SealedContainerLauncherError(
            "standalone launch cannot deserialize authority; invoke launch_sealed_once from "
            "the execution-claim verifier with a freshly revalidated typed capability"
        )
    if args.command == "verify-launch-evidence":
        receipt = load_sealed_launch_receipt(
            args.receipt,
            expected_receipt_sha256=args.expected_receipt_sha256,
        )
        verify_sealed_launch_evidence(
            receipt,
            audit_root=args.audit_root,
            sealed_contract=load_sealed_launch_contract(args.sealed_contract),
        )
        print(receipt.receipt_sha256)
        print(receipt.outcome)
        return 0
    if args.command == "verify-launch-failure-evidence":
        receipt = load_sealed_launch_failure_receipt(
            args.receipt,
            expected_receipt_sha256=args.expected_receipt_sha256,
        )
        verify_sealed_launch_failure_evidence(
            receipt,
            audit_root=args.audit_root,
            sealed_contract=load_sealed_launch_contract(args.sealed_contract),
        )
        print(receipt.receipt_sha256)
        print(receipt.failure_stage)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
