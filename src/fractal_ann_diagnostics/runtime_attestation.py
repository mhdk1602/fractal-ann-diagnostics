"""Fail-closed runtime evidence for the sealed confirmatory process.

The attestation plan is frozen before execution.  A launcher identity document
supplies the digest-qualified OCI reference and C0 commit observed by the
container launcher.  The online process then consumes an exclusive invocation
marker before inspecting its Linux runtime and hashing every declared input.

This is execution evidence, not hardware-backed remote attestation.  A caller
must preserve the launcher record and registry evidence outside the container.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import stat
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Protocol

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    read_secure_control_file,
    write_exclusive_receipt_bytes,
)

RUNTIME_ATTESTATION_PLAN_SCHEMA = "fractal-runtime-attestation-plan-v2"
RUNTIME_ATTESTATION_RECEIPT_SCHEMA = "fractal-runtime-attestation-receipt-v3"
RUNTIME_ATTESTATION_MANIFEST_TOKEN = "{manifest_sha256}"
LAUNCHER_IDENTITY_SCHEMA = "fractal-launcher-runtime-identity-v1"
INVOCATION_MARKER_SCHEMA = "fractal-runtime-invocation-marker-v1"
PROCESS_ARGV_SCHEMA = "fractal-process-argv-v1"
PROCESS_ENVIRONMENT_SCHEMA = "fractal-process-environment-v1"
MOUNT_NAMESPACE_SCHEMA = "fractal-linux-mount-namespace-v2"
RUNTIME_PREFLIGHT_RECEIPT_SCHEMA = "fractal-runtime-preflight-receipt-v1"
CANDIDATE_C0_COMMIT_SENTINEL = "containing-confirmatory-apparatus-c0-commit"

ArtifactKind = Literal["file", "directory"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OCI_DIGEST = re.compile(r"^[^\s/@]+(?:/[^\s/@]+)+@sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,255}$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_MAX_CONTROL_BYTES = 8 * 1024 * 1024

_FILE_PIN_FIELDS = frozenset({"path", "sha256"})
_MOUNT_FIELDS = frozenset({"artifact_sha256", "kind", "read_only", "role", "root"})
_PLAN_FIELDS = frozenset(
    {
        "architecture",
        "argv",
        "argv_sha256",
        "attestation_id",
        "code_commit",
        "cpu_model",
        "environment_allowlist",
        "environment_sha256",
        "invocation_marker_path",
        "kernel_release",
        "launcher_identity",
        "logical_cpu_count",
        "manifest_sha256",
        "memory_limit_bytes",
        "mount_namespace_sha256",
        "mounts",
        "network_mode",
        "oci_image_digest",
        "opa_binary",
        "operating_system_id",
        "operating_system_version_id",
        "python_binary",
        "python_version",
        "runner_identity",
        "schema_version",
        "uv_lock",
        "workload_id",
        "workload_sha256",
    }
)
_LAUNCHER_FIELDS = frozenset({"code_commit", "oci_image_digest", "schema_version"})
_NETWORK_FIELDS = frozenset(
    {
        "interfaces",
        "mode",
        "namespace_inode",
        "non_loopback_route_count",
        "route_tables_sha256",
    }
)
_PROCESS_FIELDS = frozenset(
    {"argument_count", "argv_sha256", "environment_allowlist", "environment_sha256"}
)
_RECEIPT_FIELDS = frozenset(
    {
        "architecture",
        "attestation_id",
        "code_commit",
        "cpu_model",
        "invocation_marker_path",
        "invocation_marker_sha256",
        "kernel_release",
        "launcher_identity",
        "logical_cpu_count",
        "manifest_sha256",
        "memory_limit_bytes",
        "mount_namespace_sha256",
        "mount_namespace_raw_sha256",
        "mounts",
        "network",
        "oci_image_digest",
        "opa_binary",
        "operating_system_id",
        "operating_system_version_id",
        "plan_sha256",
        "process",
        "python_binary",
        "python_version",
        "runner_identity",
        "schema_version",
        "uv_lock",
        "workload_id",
        "workload_sha256",
    }
)
_PREFLIGHT_RECEIPT_FIELDS = frozenset(
    {
        "architecture",
        "artifact_mounts",
        "code_commit",
        "cpu_model",
        "effective_gid",
        "effective_uid",
        "environment_allowlist",
        "environment_sha256",
        "hostname",
        "kernel_release",
        "launcher_contract_sha256",
        "logical_cpu_count",
        "memory_limit_bytes",
        "mount_namespace_raw_sha256",
        "mount_namespace_sha256",
        "network_interfaces",
        "network_mode",
        "non_loopback_route_count",
        "oci_image_digest",
        "operating_system_id",
        "operating_system_version_id",
        "output_root",
        "python_executable",
        "python_version",
        "route_tables_sha256",
        "schema_version",
        "tmpfs_root",
    }
)


class RuntimeAttestationError(ValueError):
    """Raised when the sealed process cannot prove its frozen runtime contract."""


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
        raise RuntimeAttestationError("runtime evidence must be finite canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise RuntimeAttestationError(f"{label} must be a JSON object with string keys")
    observed = set(value)
    if observed != fields:
        raise RuntimeAttestationError(
            f"{label} keys differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _parse_json_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    if type(encoded) is not bytes or not encoded or len(encoded) > _MAX_CONTROL_BYTES:
        raise RuntimeAttestationError(f"{label} must be non-empty bounded bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeAttestationError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise RuntimeAttestationError(f"{label} contains non-finite value {value!r}")

    try:
        decoded = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise RuntimeAttestationError(f"{label} must be valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeAttestationError(f"{label} must be valid JSON: {exc.msg}") from exc
    if type(decoded) is not dict:
        raise RuntimeAttestationError(f"{label} must contain one JSON object")
    return decoded


def _text(name: str, value: object, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value) or value != value.strip():
        raise RuntimeAttestationError(f"{name} must be a canonical string")
    if unicodedata.normalize("NFC", value) != value:
        raise RuntimeAttestationError(f"{name} must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RuntimeAttestationError(f"{name} contains a forbidden control character")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise RuntimeAttestationError(f"{name} must be valid UTF-8") from exc
    return value


def _identifier(name: str, value: object) -> str:
    result = _text(name, value)
    if _IDENTIFIER.fullmatch(result) is None:
        raise RuntimeAttestationError(f"{name} must be a stable ASCII identifier")
    return result


def _sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise RuntimeAttestationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeAttestationError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeAttestationError(f"{name} must be a non-negative integer")
    return value


def _absolute_path(name: str, value: object) -> str:
    text = _text(name, value)
    if "\\" in text:
        raise RuntimeAttestationError(f"{name} must use POSIX separators")
    path = PurePosixPath(text)
    if path.anchor != "/" or str(path) != text:
        raise RuntimeAttestationError(f"{name} must be a canonical absolute POSIX path")
    if len(path.parts) < 2 or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise RuntimeAttestationError(f"{name} cannot alias another path")
    return text


def _string_tuple(name: str, values: object, *, allow_empty: bool = False) -> tuple[str, ...]:
    if type(values) not in {list, tuple}:
        raise RuntimeAttestationError(f"{name} must be an array")
    result = tuple(
        _text(f"{name}[{position}]", value, allow_empty=allow_empty)
        for position, value in enumerate(values)
    )
    return result


def _environment_names(values: object) -> tuple[str, ...]:
    result = _string_tuple("environment_allowlist", values)
    if any(_ENVIRONMENT_NAME.fullmatch(name) is None for name in result):
        raise RuntimeAttestationError("environment allowlist names must use uppercase shell syntax")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise RuntimeAttestationError("environment_allowlist must be unique and sorted")
    return result


def argv_sha256(argv: Sequence[str]) -> str:
    """Return the only registered digest for the exact process argument vector."""

    normalized = _string_tuple("argv", tuple(argv), allow_empty=True)
    return _sha256_bytes(
        _canonical_bytes({"argv": list(normalized), "schema_version": PROCESS_ARGV_SCHEMA})
    )


def environment_sha256(environment: Mapping[str, str]) -> str:
    """Digest exact environment names and values without placing values in a receipt."""

    if type(environment) is not dict:
        raise RuntimeAttestationError("environment must be a concrete mapping")
    names = _environment_names(sorted(environment))
    if set(names) != set(environment):
        raise RuntimeAttestationError("environment keys must be canonical allowlist names")
    variables = []
    for name in names:
        variables.append(
            {
                "name": name,
                "value": _text(f"environment {name}", environment[name], allow_empty=True),
            }
        )
    return _sha256_bytes(
        _canonical_bytes({"schema_version": PROCESS_ENVIRONMENT_SCHEMA, "variables": variables})
    )


@dataclass(frozen=True)
class RuntimeFilePin:
    """One exact regular file admitted into the runtime contract."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _absolute_path("file pin path", self.path))
        _sha256("file pin sha256", self.sha256)

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object, *, label: str = "file pin") -> RuntimeFilePin:
        row = _closed(value, _FILE_PIN_FIELDS, label=label)
        return cls(path=row["path"], sha256=row["sha256"])


@dataclass(frozen=True)
class RuntimeArtifactMount:
    """A digest-pinned artifact root that must be an exact read-only mount."""

    root: str
    role: str
    kind: ArtifactKind
    artifact_sha256: str
    read_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _absolute_path("mount root", self.root))
        _identifier("mount role", self.role)
        if self.kind not in {"file", "directory"}:
            raise RuntimeAttestationError("mount kind must be file or directory")
        _sha256("mount artifact_sha256", self.artifact_sha256)
        if self.read_only is not True:
            raise RuntimeAttestationError("every admitted artifact mount must be read-only")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "kind": self.kind,
            "read_only": self.read_only,
            "role": self.role,
            "root": self.root,
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimeArtifactMount:
        row = _closed(value, _MOUNT_FIELDS, label="artifact mount")
        return cls(
            root=row["root"],
            role=row["role"],
            kind=row["kind"],
            artifact_sha256=row["artifact_sha256"],
            read_only=row["read_only"],
        )


def _paths_overlap(first: str, second: str) -> bool:
    left = PurePosixPath(first).parts
    right = PurePosixPath(second).parts
    common = min(len(left), len(right))
    return left[:common] == right[:common]


def _pin_is_inside_mount(pin: RuntimeFilePin, mount: RuntimeArtifactMount) -> bool:
    pin_parts = PurePosixPath(pin.path).parts
    root_parts = PurePosixPath(mount.root).parts
    if mount.kind == "file":
        return pin.path == mount.root
    return len(pin_parts) > len(root_parts) and pin_parts[: len(root_parts)] == root_parts


@dataclass(frozen=True)
class RuntimeAttestationPlan:
    """Frozen expected state for one noninteractive container invocation."""

    attestation_id: str
    manifest_sha256: str
    runner_identity: str
    oci_image_digest: str
    code_commit: str
    operating_system_id: str
    operating_system_version_id: str
    kernel_release: str
    architecture: str
    cpu_model: str
    logical_cpu_count: int
    memory_limit_bytes: int
    mount_namespace_sha256: str
    mounts: tuple[RuntimeArtifactMount, ...]
    argv: tuple[str, ...]
    argv_sha256: str
    environment_allowlist: tuple[str, ...]
    environment_sha256: str
    opa_binary: RuntimeFilePin
    python_binary: RuntimeFilePin
    python_version: str
    uv_lock: RuntimeFilePin
    launcher_identity: RuntimeFilePin
    workload_id: str
    workload_sha256: str
    invocation_marker_path: str
    network_mode: str = "none"
    schema_version: str = RUNTIME_ATTESTATION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_ATTESTATION_PLAN_SCHEMA:
            raise RuntimeAttestationError("runtime attestation plan schema differs")
        _identifier("attestation_id", self.attestation_id)
        _sha256("manifest_sha256", self.manifest_sha256)
        _identifier("runner_identity", self.runner_identity)
        if (
            type(self.oci_image_digest) is not str
            or _OCI_DIGEST.fullmatch(self.oci_image_digest) is None
        ):
            raise RuntimeAttestationError(
                "oci_image_digest must be a digest-qualified OCI reference"
            )
        if type(self.code_commit) is not str or (
            _GIT_COMMIT.fullmatch(self.code_commit) is None
            and self.code_commit != CANDIDATE_C0_COMMIT_SENTINEL
        ):
            raise RuntimeAttestationError(
                "code_commit must be one full commit or the candidate C0 sentinel"
            )
        for name in (
            "operating_system_id",
            "operating_system_version_id",
            "kernel_release",
            "architecture",
            "cpu_model",
            "python_version",
        ):
            _text(name, getattr(self, name))
        _positive_integer("logical_cpu_count", self.logical_cpu_count)
        _positive_integer("memory_limit_bytes", self.memory_limit_bytes)
        _sha256("mount_namespace_sha256", self.mount_namespace_sha256)
        if self.network_mode != "none":
            raise RuntimeAttestationError("network_mode must equal none")

        mounts = tuple(self.mounts)
        if not mounts or not all(isinstance(item, RuntimeArtifactMount) for item in mounts):
            raise RuntimeAttestationError("mounts must contain typed artifact mounts")
        expected_order = tuple(sorted(mounts, key=lambda item: item.root.encode("utf-8")))
        if mounts != expected_order:
            raise RuntimeAttestationError("mounts must be sorted by UTF-8 root bytes")
        if len({item.root for item in mounts}) != len(mounts):
            raise RuntimeAttestationError("artifact mount roots must be unique")
        for position, first in enumerate(mounts):
            for second in mounts[position + 1 :]:
                if _paths_overlap(first.root, second.root):
                    raise RuntimeAttestationError("artifact mount roots cannot overlap")
        object.__setattr__(self, "mounts", mounts)

        argv = _string_tuple("argv", self.argv, allow_empty=True)
        if not argv or not argv[0]:
            raise RuntimeAttestationError(
                "argv must contain a non-empty executable at position zero"
            )
        object.__setattr__(self, "argv", argv)
        _sha256("argv_sha256", self.argv_sha256)
        if self.argv_sha256 != argv_sha256(argv):
            raise RuntimeAttestationError("argv_sha256 differs from the exact argument vector")
        names = _environment_names(self.environment_allowlist)
        object.__setattr__(self, "environment_allowlist", names)
        _sha256("environment_sha256", self.environment_sha256)

        for name in ("opa_binary", "python_binary", "uv_lock", "launcher_identity"):
            if not isinstance(getattr(self, name), RuntimeFilePin):
                raise RuntimeAttestationError(f"{name} must be a RuntimeFilePin")
        _identifier("workload_id", self.workload_id)
        _sha256("workload_sha256", self.workload_sha256)
        marker = _absolute_path("invocation_marker_path", self.invocation_marker_path)
        object.__setattr__(self, "invocation_marker_path", marker)
        for mount in mounts:
            if _paths_overlap(marker, mount.root):
                raise RuntimeAttestationError(
                    "invocation marker cannot be inside an immutable artifact mount"
                )
        pinned_paths = {
            self.opa_binary.path,
            self.python_binary.path,
            self.uv_lock.path,
            self.launcher_identity.path,
        }
        if len(pinned_paths) != 4:
            raise RuntimeAttestationError("runtime file pins must use distinct canonical paths")
        if marker in pinned_paths:
            raise RuntimeAttestationError("invocation marker must differ from every pinned file")
        for name, pin in (
            ("OPA binary", self.opa_binary),
            ("uv lockfile", self.uv_lock),
            ("launcher identity", self.launcher_identity),
        ):
            if not any(_pin_is_inside_mount(pin, mount) for mount in mounts):
                raise RuntimeAttestationError(
                    f"{name} must reside inside one declared read-only artifact mount"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "argv": list(self.argv),
            "argv_sha256": self.argv_sha256,
            "attestation_id": self.attestation_id,
            "code_commit": self.code_commit,
            "cpu_model": self.cpu_model,
            "environment_allowlist": list(self.environment_allowlist),
            "environment_sha256": self.environment_sha256,
            "invocation_marker_path": self.invocation_marker_path,
            "kernel_release": self.kernel_release,
            "launcher_identity": self.launcher_identity.to_dict(),
            "logical_cpu_count": self.logical_cpu_count,
            "manifest_sha256": self.manifest_sha256,
            "memory_limit_bytes": self.memory_limit_bytes,
            "mount_namespace_sha256": self.mount_namespace_sha256,
            "mounts": [item.to_dict() for item in self.mounts],
            "network_mode": self.network_mode,
            "oci_image_digest": self.oci_image_digest,
            "opa_binary": self.opa_binary.to_dict(),
            "operating_system_id": self.operating_system_id,
            "operating_system_version_id": self.operating_system_version_id,
            "python_binary": self.python_binary.to_dict(),
            "python_version": self.python_version,
            "runner_identity": self.runner_identity,
            "schema_version": self.schema_version,
            "uv_lock": self.uv_lock.to_dict(),
            "workload_id": self.workload_id,
            "workload_sha256": self.workload_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def plan_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> RuntimeAttestationPlan:
        row = _closed(value, _PLAN_FIELDS, label="runtime attestation plan")
        mounts = row["mounts"]
        if type(mounts) is not list:
            raise RuntimeAttestationError("runtime attestation plan mounts must be an array")
        argv = row["argv"]
        environment = row["environment_allowlist"]
        if type(argv) is not list or type(environment) is not list:
            raise RuntimeAttestationError("plan argv and environment_allowlist must be arrays")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "argv",
                    "environment_allowlist",
                    "launcher_identity",
                    "mounts",
                    "opa_binary",
                    "python_binary",
                    "uv_lock",
                }
            },
            argv=tuple(argv),
            environment_allowlist=tuple(environment),
            launcher_identity=RuntimeFilePin.from_dict(
                row["launcher_identity"], label="launcher_identity"
            ),
            mounts=tuple(RuntimeArtifactMount.from_dict(item) for item in mounts),
            opa_binary=RuntimeFilePin.from_dict(row["opa_binary"], label="opa_binary"),
            python_binary=RuntimeFilePin.from_dict(row["python_binary"], label="python_binary"),
            uv_lock=RuntimeFilePin.from_dict(row["uv_lock"], label="uv_lock"),
        )


def runtime_attestation_plan_template_file_bytes(
    plan: RuntimeAttestationPlan,
) -> bytes:
    """Return the C1-pinnable plan template without a manifest-digest cycle.

    The template is byte-identical to the canonical plan file except that the
    future C1 manifest digest is represented by one literal token.  Every
    runtime, mount, workload, process, and binary field therefore remains
    committed before C1.  The instantiated plan may differ only at that token.
    """

    if not isinstance(plan, RuntimeAttestationPlan):
        raise RuntimeAttestationError("plan template source must be RuntimeAttestationPlan")
    payload = plan.to_dict()
    payload["manifest_sha256"] = RUNTIME_ATTESTATION_MANIFEST_TOKEN
    return _canonical_bytes(payload) + b"\n"


def verify_runtime_attestation_plan_template(
    plan: RuntimeAttestationPlan,
    *,
    expected_file_sha256: str,
) -> None:
    """Require a post-C1 plan to instantiate the exact C1 template bytes."""

    expected = _sha256("runtime attestation plan template SHA-256", expected_file_sha256)
    observed = _sha256_bytes(runtime_attestation_plan_template_file_bytes(plan))
    if observed != expected:
        raise RuntimeAttestationError(
            "runtime attestation plan differs from the C1-pinned template"
        )


@dataclass(frozen=True)
class ObservedMount:
    """Linux mount-table evidence for one planned artifact root."""

    root: str
    read_only: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _absolute_path("observed mount root", self.root))
        if type(self.read_only) is not bool:
            raise RuntimeAttestationError("observed mount read_only must be boolean")


@dataclass(frozen=True)
class RuntimeObservation:
    """Injectable observation returned by a runtime probe."""

    operating_system_id: str
    operating_system_version_id: str
    kernel_release: str
    architecture: str
    cpu_model: str
    logical_cpu_count: int
    memory_limit_bytes: int
    mount_namespace_sha256: str
    mount_namespace_raw_sha256: str
    mounts: tuple[ObservedMount, ...]
    network_mode: str
    network_namespace_inode: int
    network_interfaces: tuple[str, ...]
    non_loopback_route_count: int
    route_tables_sha256: str
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    python_executable: str
    python_version: str

    def __post_init__(self) -> None:
        for name in (
            "operating_system_id",
            "operating_system_version_id",
            "kernel_release",
            "architecture",
            "cpu_model",
            "python_version",
        ):
            _text(name, getattr(self, name))
        _positive_integer("logical_cpu_count", self.logical_cpu_count)
        _positive_integer("memory_limit_bytes", self.memory_limit_bytes)
        _sha256("mount_namespace_sha256", self.mount_namespace_sha256)
        _sha256("mount_namespace_raw_sha256", self.mount_namespace_raw_sha256)
        mounts = tuple(self.mounts)
        if not all(isinstance(item, ObservedMount) for item in mounts):
            raise RuntimeAttestationError("observed mounts must contain ObservedMount rows")
        if len({item.root for item in mounts}) != len(mounts):
            raise RuntimeAttestationError("observed mount roots must be unique")
        object.__setattr__(self, "mounts", mounts)
        if self.network_mode not in {"none", "enabled", "unknown"}:
            raise RuntimeAttestationError("network_mode must be none, enabled, or unknown")
        _positive_integer("network_namespace_inode", self.network_namespace_inode)
        interfaces = _string_tuple("network_interfaces", self.network_interfaces)
        if interfaces != tuple(sorted(interfaces)) or len(interfaces) != len(set(interfaces)):
            raise RuntimeAttestationError("network_interfaces must be unique and sorted")
        object.__setattr__(self, "network_interfaces", interfaces)
        _nonnegative_integer("non_loopback_route_count", self.non_loopback_route_count)
        _sha256("route_tables_sha256", self.route_tables_sha256)
        object.__setattr__(self, "argv", _string_tuple("argv", self.argv, allow_empty=True))
        if type(self.environment) is not dict:
            raise RuntimeAttestationError("observed environment must be a concrete mapping")
        normalized_environment: dict[str, str] = {}
        for key, value in self.environment.items():
            if type(key) is not str or type(value) is not str:
                raise RuntimeAttestationError("observed environment must contain string pairs")
            normalized_environment[key] = value
        object.__setattr__(self, "environment", normalized_environment)
        object.__setattr__(
            self, "python_executable", _absolute_path("python_executable", self.python_executable)
        )


@dataclass(frozen=True)
class RuntimePreflightReceipt:
    """Label-free launcher observation used to materialize the C1 runtime plans.

    The preflight runs in a separate container before C1. It records only facts
    that the launcher can reproduce for the admitted process. The raw mount
    digest is retained for audit but is not copied into a plan because fresh
    containers receive different kernel mount identifiers.
    """

    launcher_contract_sha256: str
    oci_image_digest: str
    code_commit: str
    hostname: str
    operating_system_id: str
    operating_system_version_id: str
    kernel_release: str
    architecture: str
    cpu_model: str
    logical_cpu_count: int
    memory_limit_bytes: int
    mount_namespace_sha256: str
    mount_namespace_raw_sha256: str
    artifact_mounts: tuple[ObservedMount, ...]
    output_root: str
    tmpfs_root: str
    network_mode: str
    network_interfaces: tuple[str, ...]
    non_loopback_route_count: int
    route_tables_sha256: str
    environment_allowlist: tuple[str, ...]
    environment_sha256: str
    python_executable: str
    python_version: str
    effective_uid: int
    effective_gid: int
    schema_version: str = RUNTIME_PREFLIGHT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_PREFLIGHT_RECEIPT_SCHEMA:
            raise RuntimeAttestationError("runtime preflight receipt schema differs")
        _sha256("launcher_contract_sha256", self.launcher_contract_sha256)
        if (
            type(self.oci_image_digest) is not str
            or _OCI_DIGEST.fullmatch(self.oci_image_digest) is None
        ):
            raise RuntimeAttestationError(
                "preflight OCI identity must be one digest-qualified image reference"
            )
        if type(self.code_commit) is not str or _GIT_COMMIT.fullmatch(self.code_commit) is None:
            raise RuntimeAttestationError("preflight code commit must be one full Git commit")
        for name in (
            "hostname",
            "operating_system_id",
            "operating_system_version_id",
            "kernel_release",
            "architecture",
            "cpu_model",
            "python_version",
        ):
            _text(name, getattr(self, name))
        _positive_integer("logical_cpu_count", self.logical_cpu_count)
        _positive_integer("memory_limit_bytes", self.memory_limit_bytes)
        _sha256("mount_namespace_sha256", self.mount_namespace_sha256)
        _sha256("mount_namespace_raw_sha256", self.mount_namespace_raw_sha256)
        mounts = tuple(self.artifact_mounts)
        if not mounts or not all(isinstance(item, ObservedMount) for item in mounts):
            raise RuntimeAttestationError(
                "preflight artifact_mounts must contain observed mount rows"
            )
        expected_mounts = tuple(sorted(mounts, key=lambda item: item.root.encode("utf-8")))
        if mounts != expected_mounts or len({item.root for item in mounts}) != len(mounts):
            raise RuntimeAttestationError(
                "preflight artifact mounts must be unique and bytewise sorted"
            )
        if not all(item.read_only for item in mounts):
            raise RuntimeAttestationError("preflight artifact mounts must all be read-only")
        object.__setattr__(self, "artifact_mounts", mounts)
        output_root = _absolute_path("output_root", self.output_root)
        tmpfs_root = _absolute_path("tmpfs_root", self.tmpfs_root)
        if output_root == tmpfs_root or any(
            _paths_overlap(root, item.root) for root in (output_root, tmpfs_root) for item in mounts
        ):
            raise RuntimeAttestationError(
                "preflight writable roots cannot overlap one another or an artifact mount"
            )
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "tmpfs_root", tmpfs_root)
        if self.network_mode != "none":
            raise RuntimeAttestationError("preflight network_mode must equal none")
        interfaces = _string_tuple("network_interfaces", self.network_interfaces)
        if interfaces != tuple(sorted(interfaces)) or len(interfaces) != len(set(interfaces)):
            raise RuntimeAttestationError("preflight network interfaces must be unique and sorted")
        if interfaces != ("lo",):
            raise RuntimeAttestationError("preflight network must expose loopback only")
        object.__setattr__(self, "network_interfaces", interfaces)
        _nonnegative_integer("non_loopback_route_count", self.non_loopback_route_count)
        if self.non_loopback_route_count != 0:
            raise RuntimeAttestationError("preflight cannot expose a non-loopback route")
        _sha256("route_tables_sha256", self.route_tables_sha256)
        names = _environment_names(self.environment_allowlist)
        object.__setattr__(self, "environment_allowlist", names)
        _sha256("environment_sha256", self.environment_sha256)
        object.__setattr__(
            self,
            "python_executable",
            _absolute_path("python_executable", self.python_executable),
        )
        for name in ("effective_uid", "effective_gid"):
            value = getattr(self, name)
            if type(value) is not int or value != 65532:
                raise RuntimeAttestationError(f"preflight {name} must equal 65532")

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "artifact_mounts": [
                {"read_only": item.read_only, "root": item.root} for item in self.artifact_mounts
            ],
            "code_commit": self.code_commit,
            "cpu_model": self.cpu_model,
            "effective_gid": self.effective_gid,
            "effective_uid": self.effective_uid,
            "environment_allowlist": list(self.environment_allowlist),
            "environment_sha256": self.environment_sha256,
            "hostname": self.hostname,
            "kernel_release": self.kernel_release,
            "launcher_contract_sha256": self.launcher_contract_sha256,
            "logical_cpu_count": self.logical_cpu_count,
            "memory_limit_bytes": self.memory_limit_bytes,
            "mount_namespace_raw_sha256": self.mount_namespace_raw_sha256,
            "mount_namespace_sha256": self.mount_namespace_sha256,
            "network_interfaces": list(self.network_interfaces),
            "network_mode": self.network_mode,
            "non_loopback_route_count": self.non_loopback_route_count,
            "oci_image_digest": self.oci_image_digest,
            "operating_system_id": self.operating_system_id,
            "operating_system_version_id": self.operating_system_version_id,
            "output_root": self.output_root,
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "route_tables_sha256": self.route_tables_sha256,
            "schema_version": self.schema_version,
            "tmpfs_root": self.tmpfs_root,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes())

    @property
    def file_sha256(self) -> str:
        return _sha256_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> RuntimePreflightReceipt:
        row = _closed(value, _PREFLIGHT_RECEIPT_FIELDS, label="runtime preflight receipt")
        mount_rows = row["artifact_mounts"]
        if type(mount_rows) is not list:
            raise RuntimeAttestationError("preflight artifact_mounts must be an array")
        mounts: list[ObservedMount] = []
        for position, item in enumerate(mount_rows):
            mount = _closed(
                item,
                frozenset({"read_only", "root"}),
                label=f"runtime preflight artifact_mounts[{position}]",
            )
            mounts.append(ObservedMount(root=mount["root"], read_only=mount["read_only"]))
        interfaces = row["network_interfaces"]
        environment = row["environment_allowlist"]
        if type(interfaces) is not list or type(environment) is not list:
            raise RuntimeAttestationError(
                "preflight network_interfaces and environment_allowlist must be arrays"
            )
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "artifact_mounts",
                    "environment_allowlist",
                    "network_interfaces",
                }
            },
            artifact_mounts=tuple(mounts),
            environment_allowlist=tuple(environment),
            network_interfaces=tuple(interfaces),
        )


class RuntimeProbe(Protocol):
    """A testable provider of Linux process and confinement observations."""

    def __call__(self, plan: RuntimeAttestationPlan) -> RuntimeObservation: ...


@dataclass(frozen=True)
class _LauncherIdentity:
    oci_image_digest: str
    code_commit: str
    schema_version: str = LAUNCHER_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != LAUNCHER_IDENTITY_SCHEMA:
            raise RuntimeAttestationError("launcher identity schema differs")
        if (
            type(self.oci_image_digest) is not str
            or _OCI_DIGEST.fullmatch(self.oci_image_digest) is None
        ):
            raise RuntimeAttestationError("launcher OCI identity is not digest-qualified")
        if type(self.code_commit) is not str or (
            _GIT_COMMIT.fullmatch(self.code_commit) is None
            and self.code_commit != CANDIDATE_C0_COMMIT_SENTINEL
        ):
            raise RuntimeAttestationError(
                "launcher code commit is neither a full commit nor the candidate sentinel"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "code_commit": self.code_commit,
            "oci_image_digest": self.oci_image_digest,
            "schema_version": self.schema_version,
        }


def launcher_identity_file_bytes(*, oci_image_digest: str, code_commit: str) -> bytes:
    """Build canonical launcher evidence for a read-only, digest-pinned mount."""

    return (
        _canonical_bytes(
            _LauncherIdentity(oci_image_digest=oci_image_digest, code_commit=code_commit).to_dict()
        )
        + b"\n"
    )


def _load_launcher_identity(path: RuntimeFilePin) -> _LauncherIdentity:
    try:
        encoded = read_secure_control_file(path.path, label="launcher identity")
    except ArtifactIntegrityError as exc:
        raise RuntimeAttestationError(f"cannot read launcher identity: {exc}") from exc
    if _sha256_bytes(encoded) != path.sha256:
        raise RuntimeAttestationError("launcher identity file digest differs from its plan pin")
    row = _closed(
        _parse_json_object(encoded, label="launcher identity"),
        _LAUNCHER_FIELDS,
        label="launcher identity",
    )
    identity = _LauncherIdentity(
        schema_version=row["schema_version"],
        oci_image_digest=row["oci_image_digest"],
        code_commit=row["code_commit"],
    )
    expected = _canonical_bytes(identity.to_dict()) + b"\n"
    if encoded != expected:
        raise RuntimeAttestationError("launcher identity file is not canonical JSON")
    return identity


def _mount_digest(mount: RuntimeArtifactMount) -> str:
    try:
        if mount.kind == "file":
            return digest_regular_file(mount.root, label=f"{mount.role} mount")
        return digest_directory_tree(mount.root).sha256
    except ArtifactIntegrityError as exc:
        raise RuntimeAttestationError(f"cannot digest {mount.role} mount: {exc}") from exc


def _file_digest(
    pin: RuntimeFilePin,
    *,
    label: str,
    require_executable: bool = False,
) -> str:
    try:
        observed = digest_regular_file(pin.path, label=label)
    except ArtifactIntegrityError as exc:
        raise RuntimeAttestationError(f"cannot digest {label}: {exc}") from exc
    if observed != pin.sha256:
        raise RuntimeAttestationError(f"{label} digest differs from its plan pin")
    try:
        metadata = os.lstat(pin.path)
    except OSError as exc:
        raise RuntimeAttestationError(f"cannot inspect {label} mode") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeAttestationError(f"{label} must remain one regular file")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise RuntimeAttestationError(f"{label} cannot be group/other writable")
    if require_executable and not metadata.st_mode & stat.S_IXUSR:
        raise RuntimeAttestationError(f"{label} must be executable by its owner")
    return observed


def _marker_bytes(plan: RuntimeAttestationPlan) -> bytes:
    return (
        _canonical_bytes(
            {
                "plan_sha256": plan.plan_sha256,
                "schema_version": INVOCATION_MARKER_SCHEMA,
                "workload_id": plan.workload_id,
                "workload_sha256": plan.workload_sha256,
            }
        )
        + b"\n"
    )


@dataclass(frozen=True)
class RuntimeAttestationReceipt:
    """Canonical evidence emitted after every frozen runtime check succeeds."""

    attestation_id: str
    plan_sha256: str
    manifest_sha256: str
    runner_identity: str
    oci_image_digest: str
    code_commit: str
    operating_system_id: str
    operating_system_version_id: str
    kernel_release: str
    architecture: str
    cpu_model: str
    logical_cpu_count: int
    memory_limit_bytes: int
    mount_namespace_sha256: str
    mount_namespace_raw_sha256: str
    mounts: tuple[RuntimeArtifactMount, ...]
    network: Mapping[str, object]
    process: Mapping[str, object]
    opa_binary: RuntimeFilePin
    python_binary: RuntimeFilePin
    python_version: str
    uv_lock: RuntimeFilePin
    launcher_identity: RuntimeFilePin
    workload_id: str
    workload_sha256: str
    invocation_marker_path: str
    invocation_marker_sha256: str
    schema_version: str = RUNTIME_ATTESTATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_ATTESTATION_RECEIPT_SCHEMA:
            raise RuntimeAttestationError("runtime attestation receipt schema differs")
        _identifier("attestation_id", self.attestation_id)
        for name in (
            "plan_sha256",
            "manifest_sha256",
            "workload_sha256",
            "invocation_marker_sha256",
        ):
            _sha256(name, getattr(self, name))
        _identifier("runner_identity", self.runner_identity)
        if (
            type(self.oci_image_digest) is not str
            or _OCI_DIGEST.fullmatch(self.oci_image_digest) is None
        ):
            raise RuntimeAttestationError("receipt OCI image is not digest-qualified")
        if type(self.code_commit) is not str or _GIT_COMMIT.fullmatch(self.code_commit) is None:
            raise RuntimeAttestationError("receipt code commit is not a full Git commit")
        for name in (
            "operating_system_id",
            "operating_system_version_id",
            "kernel_release",
            "architecture",
            "cpu_model",
            "python_version",
        ):
            _text(name, getattr(self, name))
        _positive_integer("logical_cpu_count", self.logical_cpu_count)
        _positive_integer("memory_limit_bytes", self.memory_limit_bytes)
        _sha256("mount_namespace_sha256", self.mount_namespace_sha256)
        _sha256("mount_namespace_raw_sha256", self.mount_namespace_raw_sha256)
        mounts = tuple(self.mounts)
        if not mounts or not all(isinstance(item, RuntimeArtifactMount) for item in mounts):
            raise RuntimeAttestationError("receipt mounts must contain typed mount rows")
        if mounts != tuple(sorted(mounts, key=lambda item: item.root.encode("utf-8"))):
            raise RuntimeAttestationError("receipt mounts must be canonically sorted")
        if len({item.root for item in mounts}) != len(mounts):
            raise RuntimeAttestationError("receipt artifact mount roots must be unique")
        for position, first in enumerate(mounts):
            for second in mounts[position + 1 :]:
                if _paths_overlap(first.root, second.root):
                    raise RuntimeAttestationError("receipt artifact mount roots cannot overlap")
        object.__setattr__(self, "mounts", mounts)
        network = _closed(self.network, _NETWORK_FIELDS, label="receipt network evidence")
        if network["mode"] != "none":
            raise RuntimeAttestationError("receipt network mode must equal none")
        interfaces = _string_tuple("network interfaces", network["interfaces"])
        if interfaces != ("lo",):
            raise RuntimeAttestationError("receipt may contain only the loopback interface")
        _positive_integer("network namespace inode", network["namespace_inode"])
        if (
            _nonnegative_integer("non_loopback_route_count", network["non_loopback_route_count"])
            != 0
        ):
            raise RuntimeAttestationError("receipt cannot admit non-loopback routes")
        _sha256("route_tables_sha256", network["route_tables_sha256"])
        object.__setattr__(
            self,
            "network",
            MappingProxyType(
                {
                    "interfaces": interfaces,
                    "mode": network["mode"],
                    "namespace_inode": network["namespace_inode"],
                    "non_loopback_route_count": network["non_loopback_route_count"],
                    "route_tables_sha256": network["route_tables_sha256"],
                }
            ),
        )
        process = _closed(self.process, _PROCESS_FIELDS, label="receipt process evidence")
        _positive_integer("argument_count", process["argument_count"])
        _sha256("process argv_sha256", process["argv_sha256"])
        names = _environment_names(process["environment_allowlist"])
        _sha256("process environment_sha256", process["environment_sha256"])
        object.__setattr__(
            self,
            "process",
            MappingProxyType(
                {
                    "argument_count": process["argument_count"],
                    "argv_sha256": process["argv_sha256"],
                    "environment_allowlist": names,
                    "environment_sha256": process["environment_sha256"],
                }
            ),
        )
        for name in ("opa_binary", "python_binary", "uv_lock", "launcher_identity"):
            if not isinstance(getattr(self, name), RuntimeFilePin):
                raise RuntimeAttestationError(f"receipt {name} must be a RuntimeFilePin")
        _identifier("workload_id", self.workload_id)
        object.__setattr__(
            self,
            "invocation_marker_path",
            _absolute_path("invocation_marker_path", self.invocation_marker_path),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "attestation_id": self.attestation_id,
            "code_commit": self.code_commit,
            "cpu_model": self.cpu_model,
            "invocation_marker_path": self.invocation_marker_path,
            "invocation_marker_sha256": self.invocation_marker_sha256,
            "kernel_release": self.kernel_release,
            "launcher_identity": self.launcher_identity.to_dict(),
            "logical_cpu_count": self.logical_cpu_count,
            "manifest_sha256": self.manifest_sha256,
            "memory_limit_bytes": self.memory_limit_bytes,
            "mount_namespace_sha256": self.mount_namespace_sha256,
            "mount_namespace_raw_sha256": self.mount_namespace_raw_sha256,
            "mounts": [item.to_dict() for item in self.mounts],
            "network": dict(self.network),
            "oci_image_digest": self.oci_image_digest,
            "opa_binary": self.opa_binary.to_dict(),
            "operating_system_id": self.operating_system_id,
            "operating_system_version_id": self.operating_system_version_id,
            "plan_sha256": self.plan_sha256,
            "process": dict(self.process),
            "python_binary": self.python_binary.to_dict(),
            "python_version": self.python_version,
            "runner_identity": self.runner_identity,
            "schema_version": self.schema_version,
            "uv_lock": self.uv_lock.to_dict(),
            "workload_id": self.workload_id,
            "workload_sha256": self.workload_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> RuntimeAttestationReceipt:
        row = _closed(value, _RECEIPT_FIELDS, label="runtime attestation receipt")
        mounts = row["mounts"]
        if type(mounts) is not list:
            raise RuntimeAttestationError("receipt mounts must be an array")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "launcher_identity",
                    "mounts",
                    "opa_binary",
                    "python_binary",
                    "uv_lock",
                }
            },
            launcher_identity=RuntimeFilePin.from_dict(
                row["launcher_identity"], label="launcher_identity"
            ),
            mounts=tuple(RuntimeArtifactMount.from_dict(item) for item in mounts),
            opa_binary=RuntimeFilePin.from_dict(row["opa_binary"], label="opa_binary"),
            python_binary=RuntimeFilePin.from_dict(row["python_binary"], label="python_binary"),
            uv_lock=RuntimeFilePin.from_dict(row["uv_lock"], label="uv_lock"),
        )


def loads_runtime_attestation_plan(encoded: bytes) -> RuntimeAttestationPlan:
    """Load one exact canonical plan file and reject alternate JSON encodings."""

    plan = RuntimeAttestationPlan.from_dict(
        _parse_json_object(encoded, label="runtime attestation plan")
    )
    if encoded != plan.canonical_file_bytes():
        raise RuntimeAttestationError("runtime attestation plan bytes are not canonical")
    return plan


def load_runtime_attestation_plan(path: str | Path) -> RuntimeAttestationPlan:
    try:
        encoded = read_secure_control_file(path, label="runtime attestation plan")
    except ArtifactIntegrityError as exc:
        raise RuntimeAttestationError(f"cannot read runtime attestation plan: {exc}") from exc
    return loads_runtime_attestation_plan(encoded)


def loads_runtime_attestation_receipt(encoded: bytes) -> RuntimeAttestationReceipt:
    """Load one exact canonical receipt file and reject alternate JSON encodings."""

    receipt = RuntimeAttestationReceipt.from_dict(
        _parse_json_object(encoded, label="runtime attestation receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise RuntimeAttestationError("runtime attestation receipt bytes are not canonical")
    return receipt


def load_runtime_attestation_receipt(path: str | Path) -> RuntimeAttestationReceipt:
    try:
        encoded = read_secure_control_file(path, label="runtime attestation receipt")
    except ArtifactIntegrityError as exc:
        raise RuntimeAttestationError(f"cannot read runtime attestation receipt: {exc}") from exc
    return loads_runtime_attestation_receipt(encoded)


def loads_runtime_preflight_receipt(encoded: bytes) -> RuntimePreflightReceipt:
    """Load one canonical label-free launcher preflight receipt."""

    receipt = RuntimePreflightReceipt.from_dict(
        _parse_json_object(encoded, label="runtime preflight receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise RuntimeAttestationError("runtime preflight receipt bytes are not canonical")
    return receipt


def load_runtime_preflight_receipt(path: str | Path) -> RuntimePreflightReceipt:
    """Read a preflight receipt through the secure control-file boundary."""

    try:
        encoded = read_secure_control_file(path, label="runtime preflight receipt")
    except ArtifactIntegrityError as exc:
        raise RuntimeAttestationError(f"cannot read runtime preflight receipt: {exc}") from exc
    return loads_runtime_preflight_receipt(encoded)


def verify_runtime_attestation_receipt(
    receipt: RuntimeAttestationReceipt,
    plan: RuntimeAttestationPlan,
) -> None:
    """Verify that a typed receipt binds every frozen field in its exact plan."""

    if not isinstance(receipt, RuntimeAttestationReceipt):
        raise RuntimeAttestationError("receipt must be a RuntimeAttestationReceipt")
    if not isinstance(plan, RuntimeAttestationPlan):
        raise RuntimeAttestationError("plan must be a RuntimeAttestationPlan")
    if plan.code_commit == CANDIDATE_C0_COMMIT_SENTINEL:
        raise RuntimeAttestationError(
            "runtime attestation plan still contains the candidate commit sentinel"
        )
    expected = {
        "attestation_id": plan.attestation_id,
        "plan_sha256": plan.plan_sha256,
        "manifest_sha256": plan.manifest_sha256,
        "runner_identity": plan.runner_identity,
        "oci_image_digest": plan.oci_image_digest,
        "code_commit": plan.code_commit,
        "operating_system_id": plan.operating_system_id,
        "operating_system_version_id": plan.operating_system_version_id,
        "kernel_release": plan.kernel_release,
        "architecture": plan.architecture,
        "cpu_model": plan.cpu_model,
        "logical_cpu_count": plan.logical_cpu_count,
        "memory_limit_bytes": plan.memory_limit_bytes,
        "mount_namespace_sha256": plan.mount_namespace_sha256,
        "mounts": plan.mounts,
        "opa_binary": plan.opa_binary,
        "python_binary": plan.python_binary,
        "python_version": plan.python_version,
        "uv_lock": plan.uv_lock,
        "launcher_identity": plan.launcher_identity,
        "workload_id": plan.workload_id,
        "workload_sha256": plan.workload_sha256,
        "invocation_marker_path": plan.invocation_marker_path,
        "invocation_marker_sha256": _sha256_bytes(_marker_bytes(plan)),
    }
    for name, value in expected.items():
        if getattr(receipt, name) != value:
            raise RuntimeAttestationError(f"receipt {name} differs from the frozen plan")
    if receipt.process["argument_count"] != len(plan.argv):
        raise RuntimeAttestationError("receipt process argument count differs from the plan")
    if receipt.process["argv_sha256"] != plan.argv_sha256:
        raise RuntimeAttestationError("receipt process argv digest differs from the plan")
    if tuple(receipt.process["environment_allowlist"]) != plan.environment_allowlist:
        raise RuntimeAttestationError("receipt process environment allowlist differs from the plan")
    if receipt.process["environment_sha256"] != plan.environment_sha256:
        raise RuntimeAttestationError("receipt process environment digest differs from the plan")


def write_runtime_attestation_plan(plan: RuntimeAttestationPlan, target: str | Path) -> None:
    if not isinstance(plan, RuntimeAttestationPlan):
        raise RuntimeAttestationError("plan must be a RuntimeAttestationPlan")
    try:
        write_exclusive_receipt_bytes(plan.canonical_file_bytes(), target)
    except ArtifactIntegrityError as exc:
        raise RuntimeAttestationError(f"cannot publish runtime attestation plan: {exc}") from exc


def write_runtime_attestation_receipt(
    receipt: RuntimeAttestationReceipt, target: str | Path
) -> None:
    if not isinstance(receipt, RuntimeAttestationReceipt):
        raise RuntimeAttestationError("receipt must be a RuntimeAttestationReceipt")
    try:
        write_exclusive_receipt_bytes(receipt.canonical_file_bytes(), target)
    except ArtifactIntegrityError as exc:
        raise RuntimeAttestationError(f"cannot publish runtime attestation receipt: {exc}") from exc


def write_runtime_preflight_receipt(receipt: RuntimePreflightReceipt, target: str | Path) -> None:
    """Publish one preflight receipt without replacing an existing record."""

    if not isinstance(receipt, RuntimePreflightReceipt):
        raise RuntimeAttestationError("receipt must be a RuntimePreflightReceipt")
    try:
        write_exclusive_receipt_bytes(receipt.canonical_file_bytes(), target)
    except ArtifactIntegrityError as exc:
        raise RuntimeAttestationError(f"cannot publish runtime preflight receipt: {exc}") from exc


def _assert_observation(plan: RuntimeAttestationPlan, observed: RuntimeObservation) -> None:
    expected_scalars = {
        "operating_system_id": plan.operating_system_id,
        "operating_system_version_id": plan.operating_system_version_id,
        "kernel_release": plan.kernel_release,
        "architecture": plan.architecture,
        "cpu_model": plan.cpu_model,
        "logical_cpu_count": plan.logical_cpu_count,
        "memory_limit_bytes": plan.memory_limit_bytes,
        "mount_namespace_sha256": plan.mount_namespace_sha256,
        "python_executable": plan.python_binary.path,
        "python_version": plan.python_version,
    }
    for name, expected in expected_scalars.items():
        if getattr(observed, name) != expected:
            raise RuntimeAttestationError(f"observed {name} differs from the frozen plan")
    expected_mounts = {item.root: item for item in plan.mounts}
    observed_mounts = {item.root: item for item in observed.mounts}
    if set(observed_mounts) != set(expected_mounts):
        raise RuntimeAttestationError("observed artifact mount roots differ from the frozen plan")
    if any(not item.read_only for item in observed_mounts.values()):
        raise RuntimeAttestationError("an admitted artifact root is mounted writable")
    if observed.network_mode != "none":
        raise RuntimeAttestationError("sealed runtime network is enabled or indeterminate")
    if observed.network_interfaces != ("lo",) or observed.non_loopback_route_count != 0:
        raise RuntimeAttestationError("sealed runtime exposes a non-loopback network path")
    if observed.argv != plan.argv or argv_sha256(observed.argv) != plan.argv_sha256:
        raise RuntimeAttestationError("observed process argv differs from the frozen plan")
    if set(observed.environment) != set(plan.environment_allowlist):
        raise RuntimeAttestationError("process environment differs from the exact allowlist")
    if environment_sha256(dict(observed.environment)) != plan.environment_sha256:
        raise RuntimeAttestationError("process environment digest differs from the frozen plan")


def _capture_observation(
    plan: RuntimeAttestationPlan,
    probe: RuntimeProbe,
) -> RuntimeObservation:
    if not callable(probe):
        raise RuntimeAttestationError("probe must be callable")
    try:
        observed = probe(plan)
    except RuntimeAttestationError:
        raise
    except Exception as exc:
        raise RuntimeAttestationError("runtime probe failed") from exc
    if not isinstance(observed, RuntimeObservation):
        raise RuntimeAttestationError("runtime probe must return RuntimeObservation")
    return observed


def verify_live_runtime_attestation(
    receipt: RuntimeAttestationReceipt,
    plan: RuntimeAttestationPlan,
    *,
    probe: RuntimeProbe,
) -> None:
    """Match a persisted receipt to the process that is about to open inputs.

    This check performs no artifact-content reads. It reobserves the process,
    mount namespace, environment, and network namespace after loading the
    receipt and before a caller opens a registered workload source.
    """

    verify_runtime_attestation_receipt(receipt, plan)
    observed = _capture_observation(plan, probe)
    _assert_observation(plan, observed)
    expected_network = {
        "interfaces": observed.network_interfaces,
        "mode": observed.network_mode,
        "namespace_inode": observed.network_namespace_inode,
        "non_loopback_route_count": observed.non_loopback_route_count,
        "route_tables_sha256": observed.route_tables_sha256,
    }
    if dict(receipt.network) != expected_network:
        raise RuntimeAttestationError(
            "live network namespace differs from the runtime attestation receipt"
        )
    if receipt.mount_namespace_raw_sha256 != observed.mount_namespace_raw_sha256:
        raise RuntimeAttestationError(
            "live raw mount namespace differs from the runtime attestation receipt"
        )
    if receipt.process["argv_sha256"] != argv_sha256(observed.argv):
        raise RuntimeAttestationError(
            "live process argv differs from the runtime attestation receipt"
        )
    if receipt.process["environment_sha256"] != environment_sha256(dict(observed.environment)):
        raise RuntimeAttestationError(
            "live process environment differs from the runtime attestation receipt"
        )


def attest_runtime_once(
    plan: RuntimeAttestationPlan,
    *,
    probe: RuntimeProbe,
    receipt_target: str | Path | None = None,
) -> RuntimeAttestationReceipt:
    """Consume one marker, verify the runtime, and optionally publish one receipt.

    The marker is deliberately written before probing.  A failed probe or digest
    check consumes the invocation and cannot be retried under the same plan.
    """

    if not isinstance(plan, RuntimeAttestationPlan):
        raise RuntimeAttestationError("plan must be a RuntimeAttestationPlan")
    marker_bytes = _marker_bytes(plan)
    try:
        write_exclusive_receipt_bytes(marker_bytes, plan.invocation_marker_path)
    except ArtifactIntegrityError as exc:
        raise RuntimeAttestationError(
            f"runtime invocation marker already exists or cannot be created: {exc}"
        ) from exc

    identity = _load_launcher_identity(plan.launcher_identity)
    if identity.oci_image_digest != plan.oci_image_digest:
        raise RuntimeAttestationError("launcher OCI image digest differs from the frozen plan")
    if identity.code_commit != plan.code_commit:
        raise RuntimeAttestationError("launcher code commit differs from the frozen plan")

    observed = _capture_observation(plan, probe)
    _assert_observation(plan, observed)

    for mount in plan.mounts:
        if _mount_digest(mount) != mount.artifact_sha256:
            raise RuntimeAttestationError(f"{mount.role} mount digest differs from its plan pin")
    for pin, label in (
        (plan.opa_binary, "OPA binary"),
        (plan.python_binary, "Python binary"),
        (plan.uv_lock, "uv lockfile"),
        (plan.launcher_identity, "launcher identity"),
    ):
        _file_digest(
            pin,
            label=label,
            require_executable=label in {"OPA binary", "Python binary"},
        )

    receipt = RuntimeAttestationReceipt(
        attestation_id=plan.attestation_id,
        plan_sha256=plan.plan_sha256,
        manifest_sha256=plan.manifest_sha256,
        runner_identity=plan.runner_identity,
        oci_image_digest=identity.oci_image_digest,
        code_commit=identity.code_commit,
        operating_system_id=observed.operating_system_id,
        operating_system_version_id=observed.operating_system_version_id,
        kernel_release=observed.kernel_release,
        architecture=observed.architecture,
        cpu_model=observed.cpu_model,
        logical_cpu_count=observed.logical_cpu_count,
        memory_limit_bytes=observed.memory_limit_bytes,
        mount_namespace_sha256=observed.mount_namespace_sha256,
        mount_namespace_raw_sha256=observed.mount_namespace_raw_sha256,
        mounts=plan.mounts,
        network={
            "interfaces": list(observed.network_interfaces),
            "mode": observed.network_mode,
            "namespace_inode": observed.network_namespace_inode,
            "non_loopback_route_count": observed.non_loopback_route_count,
            "route_tables_sha256": observed.route_tables_sha256,
        },
        process={
            "argument_count": len(observed.argv),
            "argv_sha256": argv_sha256(observed.argv),
            "environment_allowlist": list(plan.environment_allowlist),
            "environment_sha256": environment_sha256(dict(observed.environment)),
        },
        opa_binary=plan.opa_binary,
        python_binary=plan.python_binary,
        python_version=observed.python_version,
        uv_lock=plan.uv_lock,
        launcher_identity=plan.launcher_identity,
        workload_id=plan.workload_id,
        workload_sha256=plan.workload_sha256,
        invocation_marker_path=plan.invocation_marker_path,
        invocation_marker_sha256=_sha256_bytes(marker_bytes),
    )

    # A second pass closes mutations between initial admission and serialization.
    for mount in plan.mounts:
        if _mount_digest(mount) != mount.artifact_sha256:
            raise RuntimeAttestationError(f"{mount.role} mount changed during attestation")
    for pin, label in (
        (plan.opa_binary, "OPA binary"),
        (plan.python_binary, "Python binary"),
        (plan.uv_lock, "uv lockfile"),
        (plan.launcher_identity, "launcher identity"),
    ):
        _file_digest(
            pin,
            label=label,
            require_executable=label in {"OPA binary", "Python binary"},
        )

    if receipt_target is not None:
        target = _absolute_path("receipt_target", str(receipt_target))
        if target == plan.invocation_marker_path:
            raise RuntimeAttestationError("receipt target cannot replace the invocation marker")
        write_runtime_attestation_receipt(receipt, target)
    return receipt


def _read_text(path: str, *, label: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as handle:
            return handle.read()
    except (OSError, UnicodeError) as exc:
        raise RuntimeAttestationError(f"cannot read Linux {label}") from exc


def _linux_os_release() -> tuple[str, str]:
    try:
        values = platform.freedesktop_os_release()
    except OSError as exc:
        raise RuntimeAttestationError("Linux os-release is unavailable") from exc
    system_id = values.get("ID")
    version_id = values.get("VERSION_ID")
    return _text("operating_system_id", system_id), _text("operating_system_version_id", version_id)


def _linux_cpu_model() -> str:
    content = _read_text("/proc/cpuinfo", label="CPU inventory")
    models: set[str] = set()
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().casefold() in {"model name", "processor", "hardware", "cpu model"}:
            candidate = value.strip()
            if candidate and not candidate.isdecimal():
                models.add(candidate)
    if len(models) != 1:
        raise RuntimeAttestationError("CPU model is missing or heterogeneous")
    return _text("cpu_model", next(iter(models)))


def _linux_memory_limit() -> int:
    candidates = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
    for path in candidates:
        try:
            value = _read_text(path, label="memory cgroup limit").strip()
        except RuntimeAttestationError:
            continue
        if value == "max":
            raise RuntimeAttestationError("sealed runtime needs a finite cgroup memory limit")
        if value.isdecimal():
            return _positive_integer("memory_limit_bytes", int(value))
        raise RuntimeAttestationError("memory cgroup limit is not canonical decimal bytes")
    raise RuntimeAttestationError("no supported Linux cgroup memory limit is visible")


def _unescape_mount_path(value: str) -> str:
    replacements = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
    result = value
    for encoded, decoded in replacements.items():
        result = result.replace(encoded, decoded)
    if re.search(r"\\[0-7]{3}", result):
        raise RuntimeAttestationError("mountinfo contains an unsupported escaped path")
    return result


def mount_namespace_sha256(mountinfo: str) -> str:
    """Digest the reproducible security profile of a Linux mount namespace.

    Kernel IDs and container-specific overlay paths cannot be frozen before a
    fresh container exists. Their exact bytes are retained separately in the
    runtime receipt and compared again immediately before workload access. This
    digest still binds every mount point, filesystem, read/write flag, security
    option, stable source, and stable backing root, so an added mount changes it.
    """

    if type(mountinfo) is not str or not mountinfo or "\x00" in mountinfo:
        raise RuntimeAttestationError("mountinfo must be non-empty Linux mount-table text")
    rows: list[dict[str, object]] = []
    seen_mount_points: set[str] = set()
    for line in mountinfo.splitlines():
        if not line:
            raise RuntimeAttestationError("Linux mountinfo contains an empty row")
        fields = line.split(" ")
        try:
            separator = fields.index("-")
        except ValueError as exc:
            raise RuntimeAttestationError("Linux mountinfo row lacks its separator") from exc
        if separator < 6 or len(fields) < separator + 4:
            raise RuntimeAttestationError("Linux mountinfo row is truncated")
        mount_root = _unescape_mount_path(fields[3])
        mount_point = _unescape_mount_path(fields[4])
        source = _unescape_mount_path(fields[separator + 2])
        filesystem_type = _text("mount filesystem type", fields[separator + 1])
        if not mount_point.startswith("/") or not mount_root.startswith("/"):
            raise RuntimeAttestationError("Linux mountinfo paths must be absolute")
        if mount_point in seen_mount_points:
            raise RuntimeAttestationError("Linux mount table repeats a mount point")
        seen_mount_points.add(mount_point)
        mount_options = tuple(sorted(set(fields[5].split(","))))
        super_options = tuple(sorted(set(fields[separator + 3].split(","))))
        optional_kinds = tuple(sorted({field.partition(":")[0] for field in fields[6:separator]}))
        if filesystem_type == "overlay":
            mount_root = "<container-overlay-root>"
            source = "<container-overlay-source>"
            super_options = tuple(
                sorted(
                    f"{option.partition('=')[0]}=<container-overlay-path>"
                    if option.partition("=")[0] in {"lowerdir", "upperdir", "workdir"}
                    else option
                    for option in super_options
                )
            )
        elif mount_point in {"/etc/hostname", "/etc/hosts", "/etc/resolv.conf"}:
            mount_root = f"<container-managed:{mount_point}>"
            source = "<container-managed-source>"
        rows.append(
            {
                "filesystem_type": filesystem_type,
                "mount_options": list(mount_options),
                "mount_point": _text("mount point", mount_point),
                "mount_root": _text("mount root", mount_root),
                "optional_field_kinds": list(optional_kinds),
                "source": _text("mount source", source),
                "super_options": list(super_options),
            }
        )
    if not rows:
        raise RuntimeAttestationError("Linux mount table is empty")
    rows.sort(key=lambda row: str(row["mount_point"]).encode("utf-8"))
    return _sha256_bytes(
        _canonical_bytes({"mounts": rows, "schema_version": MOUNT_NAMESPACE_SCHEMA})
    )


def raw_mount_namespace_sha256(mountinfo: str) -> str:
    """Digest the exact mountinfo bytes observed inside the admitted process."""

    if type(mountinfo) is not str or not mountinfo or "\x00" in mountinfo:
        raise RuntimeAttestationError("mountinfo must be non-empty Linux mount-table text")
    try:
        encoded = mountinfo.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise RuntimeAttestationError("mountinfo must be valid UTF-8") from exc
    return _sha256_bytes(encoded)


def _mount_table_by_point(
    content: str,
) -> dict[str, tuple[bool, str, frozenset[str]]]:
    """Return exact mount-point security facts from Linux mountinfo text."""

    by_point: dict[str, tuple[bool, str, frozenset[str]]] = {}
    for line in content.splitlines():
        fields = line.split(" ")
        try:
            separator = fields.index("-")
        except ValueError as exc:
            raise RuntimeAttestationError("Linux mountinfo row lacks its separator") from exc
        if separator < 6 or len(fields) < separator + 4:
            raise RuntimeAttestationError("Linux mountinfo row is truncated")
        point = _unescape_mount_path(fields[4])
        if point in by_point:
            raise RuntimeAttestationError("Linux mount table repeats a mount point")
        mount_options = frozenset(fields[5].split(","))
        super_options = frozenset(fields[separator + 3].split(","))
        read_only = "ro" in mount_options and "rw" not in mount_options
        by_point[point] = (
            read_only,
            _text("mount filesystem type", fields[separator + 1]),
            mount_options | super_options,
        )
    if not by_point:
        raise RuntimeAttestationError("Linux mount table is empty")
    return by_point


def _linux_mounts(
    plan: RuntimeAttestationPlan,
) -> tuple[tuple[ObservedMount, ...], str, str]:
    content = _read_text("/proc/self/mountinfo", label="mount table")
    namespace_sha256 = mount_namespace_sha256(content)
    raw_namespace_sha256 = raw_mount_namespace_sha256(content)
    by_root = _mount_table_by_point(content)
    observed: list[ObservedMount] = []
    for mount in plan.mounts:
        if mount.root not in by_root:
            raise RuntimeAttestationError(
                f"artifact root is not an exact Linux mount point: {mount.root}"
            )
        observed.append(ObservedMount(root=mount.root, read_only=by_root[mount.root][0]))
    return tuple(observed), namespace_sha256, raw_namespace_sha256


def _route_evidence() -> tuple[int, str]:
    ipv4 = _read_text("/proc/net/route", label="IPv4 route table")
    ipv6 = _read_text("/proc/net/ipv6_route", label="IPv6 route table")
    non_loopback = 0
    for position, line in enumerate(ipv4.splitlines()):
        if position == 0 or not line.strip():
            continue
        fields = line.split()
        if not fields:
            raise RuntimeAttestationError("IPv4 route table contains a malformed row")
        non_loopback += fields[0] != "lo"
    for line in ipv6.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 10:
            raise RuntimeAttestationError("IPv6 route table contains a malformed row")
        non_loopback += fields[-1] != "lo"
    digest = _sha256_bytes(
        _canonical_bytes(
            {
                "ipv4_sha256": _sha256_bytes(ipv4.encode("utf-8")),
                "ipv6_sha256": _sha256_bytes(ipv6.encode("utf-8")),
                "schema_version": "fractal-linux-route-tables-v1",
            }
        )
    )
    return non_loopback, digest


def _process_argv() -> tuple[str, ...]:
    try:
        encoded = Path("/proc/self/cmdline").read_bytes()
        values = encoded.split(b"\x00")
        if not values or values[-1] != b"":
            raise RuntimeAttestationError("Linux process argv lacks its NUL terminator")
        return tuple(value.decode("utf-8", errors="strict") for value in values[:-1])
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeAttestationError("cannot read exact Linux process argv") from exc


def capture_runtime_preflight(
    *,
    launcher_contract_sha256: str,
    oci_image_digest: str,
    code_commit: str,
    hostname: str,
    artifact_mounts: Sequence[RuntimeArtifactMount],
    environment: Mapping[str, str],
    output_root: str = "/output",
    tmpfs_root: str = "/tmp",
) -> RuntimePreflightReceipt:
    """Observe a label-free container created from the final launcher contract.

    This function reads no protocol labels and writes no workload output. It
    verifies the exact read-only artifact mounts, empty UID-owned output volume,
    bounded launcher environment, loopback-only network namespace, read-only
    root filesystem, and hardened tmpfs before returning C1-materialization
    facts. A separate admitted process must reproduce the normalized mount
    digest and every frozen hardware field.
    """

    if code_commit == CANDIDATE_C0_COMMIT_SENTINEL:
        raise RuntimeAttestationError(
            "runtime preflight still contains the candidate commit sentinel"
        )
    if platform.system() != "Linux":
        raise RuntimeAttestationError("runtime preflight requires Linux")
    _sha256("launcher_contract_sha256", launcher_contract_sha256)
    expected_hostname = _text("hostname", hostname)
    if socket.gethostname() != expected_hostname:
        raise RuntimeAttestationError("preflight hostname differs from the launcher contract")
    if type(environment) is not dict:
        raise RuntimeAttestationError("preflight environment must be a concrete mapping")
    normalized_environment = {
        _text("environment name", name): _text(f"environment {name}", value, allow_empty=True)
        for name, value in environment.items()
    }
    _environment_names(sorted(normalized_environment))
    if dict(os.environ) != normalized_environment:
        raise RuntimeAttestationError("preflight process environment differs from the contract")
    if normalized_environment.get("HOSTNAME") != expected_hostname:
        raise RuntimeAttestationError("preflight HOSTNAME differs from the fixed hostname")

    mounts = tuple(artifact_mounts)
    if not mounts or not all(isinstance(item, RuntimeArtifactMount) for item in mounts):
        raise RuntimeAttestationError("preflight artifact mounts must be typed")
    if mounts != tuple(sorted(mounts, key=lambda item: item.root.encode("utf-8"))):
        raise RuntimeAttestationError("preflight artifact mounts must be bytewise sorted")
    content = _read_text("/proc/self/mountinfo", label="mount table")
    table = _mount_table_by_point(content)
    if "/" not in table or not table["/"][0]:
        raise RuntimeAttestationError("preflight container root filesystem must be read-only")
    observed_mounts: list[ObservedMount] = []
    for mount in mounts:
        observed = table.get(mount.root)
        if observed is None:
            raise RuntimeAttestationError(
                f"preflight artifact is not an exact mount point: {mount.root}"
            )
        if not observed[0]:
            raise RuntimeAttestationError(f"preflight artifact mount is writable: {mount.root}")
        if _mount_digest(mount) != mount.artifact_sha256:
            raise RuntimeAttestationError(f"preflight artifact digest differs for {mount.role!r}")
        observed_mounts.append(ObservedMount(root=mount.root, read_only=True))

    output_path = Path(_absolute_path("output_root", output_root))
    output_mount = table.get(str(output_path))
    if output_mount is None or output_mount[0]:
        raise RuntimeAttestationError("preflight output root must be one exact writable mount")
    try:
        output_metadata = os.lstat(output_path)
        output_entries = tuple(os.scandir(output_path))
    except OSError as exc:
        raise RuntimeAttestationError("cannot inspect the preflight output volume") from exc
    if (
        not stat.S_ISDIR(output_metadata.st_mode)
        or output_metadata.st_uid != 65532
        or output_metadata.st_gid != 65532
        or stat.S_IMODE(output_metadata.st_mode) != 0o700
        or output_entries
    ):
        raise RuntimeAttestationError(
            "preflight output volume must be empty, mode 0700, and owned by 65532:65532"
        )

    tmp_path = Path(_absolute_path("tmpfs_root", tmpfs_root))
    tmp_mount = table.get(str(tmp_path))
    if tmp_mount is None or tmp_mount[0] or tmp_mount[1] != "tmpfs":
        raise RuntimeAttestationError("preflight /tmp must be one exact writable tmpfs mount")
    if not {"nodev", "noexec", "nosuid"}.issubset(tmp_mount[2]):
        raise RuntimeAttestationError("preflight /tmp lacks nodev, noexec, or nosuid")

    try:
        affinity = os.sched_getaffinity(0)
        interfaces = tuple(sorted(os.listdir("/sys/class/net")))
    except (AttributeError, OSError) as exc:
        raise RuntimeAttestationError("preflight kernel evidence is unavailable") from exc
    if not affinity:
        raise RuntimeAttestationError("preflight CPU affinity is empty")
    non_loopback_routes, route_digest = _route_evidence()
    mode = "none" if interfaces == ("lo",) and non_loopback_routes == 0 else "enabled"
    system_id, version_id = _linux_os_release()
    return RuntimePreflightReceipt(
        launcher_contract_sha256=launcher_contract_sha256,
        oci_image_digest=oci_image_digest,
        code_commit=code_commit,
        hostname=expected_hostname,
        operating_system_id=system_id,
        operating_system_version_id=version_id,
        kernel_release=platform.release(),
        architecture=platform.machine(),
        cpu_model=_linux_cpu_model(),
        logical_cpu_count=len(affinity),
        memory_limit_bytes=_linux_memory_limit(),
        mount_namespace_sha256=mount_namespace_sha256(content),
        mount_namespace_raw_sha256=raw_mount_namespace_sha256(content),
        artifact_mounts=tuple(observed_mounts),
        output_root=str(output_path),
        tmpfs_root=str(tmp_path),
        network_mode=mode,
        network_interfaces=interfaces,
        non_loopback_route_count=non_loopback_routes,
        route_tables_sha256=route_digest,
        environment_allowlist=tuple(sorted(normalized_environment)),
        environment_sha256=environment_sha256(normalized_environment),
        python_executable=str(Path(sys.executable)),
        python_version=platform.python_version(),
        effective_uid=os.geteuid(),
        effective_gid=os.getegid(),
    )


class LinuxRuntimeProbe:
    """Capture a Linux container state without performing network I/O."""

    def __call__(self, plan: RuntimeAttestationPlan) -> RuntimeObservation:
        if platform.system() != "Linux":
            raise RuntimeAttestationError("production runtime attestation requires Linux")
        system_id, version_id = _linux_os_release()
        try:
            affinity = os.sched_getaffinity(0)
        except (AttributeError, OSError) as exc:
            raise RuntimeAttestationError("Linux CPU affinity is unavailable") from exc
        if not affinity:
            raise RuntimeAttestationError("Linux CPU affinity is empty")
        try:
            interfaces = tuple(sorted(os.listdir("/sys/class/net")))
            namespace = os.stat("/proc/self/ns/net").st_ino
        except OSError as exc:
            raise RuntimeAttestationError(
                "Linux network namespace evidence is unavailable"
            ) from exc
        non_loopback_routes, route_digest = _route_evidence()
        mounts, mount_namespace_digest, raw_mount_namespace_digest = _linux_mounts(plan)
        mode = "none" if interfaces == ("lo",) and non_loopback_routes == 0 else "enabled"
        return RuntimeObservation(
            operating_system_id=system_id,
            operating_system_version_id=version_id,
            kernel_release=platform.release(),
            architecture=platform.machine(),
            cpu_model=_linux_cpu_model(),
            logical_cpu_count=len(affinity),
            memory_limit_bytes=_linux_memory_limit(),
            mount_namespace_sha256=mount_namespace_digest,
            mount_namespace_raw_sha256=raw_mount_namespace_digest,
            mounts=mounts,
            network_mode=mode,
            network_namespace_inode=namespace,
            network_interfaces=interfaces,
            non_loopback_route_count=non_loopback_routes,
            route_tables_sha256=route_digest,
            argv=_process_argv(),
            environment=dict(os.environ),
            python_executable=str(Path(sys.executable)),
            python_version=platform.python_version(),
        )
