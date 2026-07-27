#!/usr/bin/env python3
"""Compose the pre-provider candidate study-manifest source from typed evidence.

This operator is intentionally outside the installed scientific package and outside the
confirmatory image context.  It has one write command: consume a canonical, digest-pinned
wiring request and publish one closed two-file package without replacing an existing path.
No manifest field, producer digest, C0 commit, provider plan, or output digest is accepted
on the command line.
"""

from __future__ import annotations

import argparse
import copy
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
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fractal_ann_diagnostics.candidate_manifest_assembler import (  # noqa: E402
    CANDIDATE_ARTIFACT_PIN_RECEIPT_SCHEMA,
    INVENTORY_FILENAME,
    INVENTORY_RECEIPT_FILENAME,
    CandidateArtifactPinInventory,
    apply_candidate_artifact_inventory,
)
from fractal_ann_diagnostics.development_freeze import (  # noqa: E402
    COMPARATOR_ARTIFACT_SCHEMA,
    DEVELOPMENT_FREEZE_SCHEMA,
    GEOMETRY_PROFILE_SCHEMA,
)
from fractal_ann_diagnostics.joint_power_design import (  # noqa: E402
    JointPowerDesignConfig,
    JointPowerDesignReport,
    load_joint_power_config,
    load_joint_power_report,
)
from fractal_ann_diagnostics.post_embedding_development import (  # noqa: E402
    FREEZE_DIRECTORY,
    POST_EMBEDDING_RECEIPT_SCHEMA,
    PostEmbeddingDevelopmentReceipt,
)
from fractal_ann_diagnostics.post_embedding_development import (  # noqa: E402
    RECEIPT_FILENAME as POST_EMBEDDING_RECEIPT_FILENAME,
)
from fractal_ann_diagnostics.production_controls import (  # noqa: E402
    BLUEPRINT_RECEIPT_FILENAME,
    PRODUCTION_HARDWARE_FRAGMENT_FILENAME,
    PRODUCTION_WORKLOADS_FRAGMENT_FILENAME,
    ProductionControlBlueprintReceipt,
    ProductionControlMaterializationConfig,
    ProductionControlMaterializationConfigWriteReceipt,
)
from fractal_ann_diagnostics.production_corpus_run import (  # noqa: E402
    ProductionCorpusRunError,
    ProductionCorpusWorkloadSpec,
)
from fractal_ann_diagnostics.production_workload_registration import (  # noqa: E402
    production_workload_file_sha256,
)
from fractal_ann_diagnostics.provider_plan_operator import (  # noqa: E402
    _admit_candidate_source_shell,
)
from fractal_ann_diagnostics.provider_rehearsal import CandidateImageClosure  # noqa: E402
from fractal_ann_diagnostics.study import (  # noqa: E402
    C0_COMMIT_SENTINEL,
    FIXED_CORPORA,
    manifest_sha256,
    validate_study_manifest,
)

REQUEST_SCHEMA = "fractal-candidate-study-manifest-source-request-v1"
DEPLOYMENT_FRAGMENT_SCHEMA = "fractal-candidate-deployment-fragment-v1"
COMPOSITION_RECEIPT_SCHEMA = "fractal-candidate-study-manifest-source-composition-v2"
INPUT_CUSTODY_CONTRACT = "fractal-exclusive-posix-advisory-custody-v1"

SOURCE_FILENAME = "candidate-study-manifest.source.json"
COMPOSITION_RECEIPT_FILENAME = "candidate-study-manifest.source-receipt.json"
DEPLOYMENT_FRAGMENT_FILENAME = "candidate-deployment.fragment.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_VALUES = frozenset(
    {"", "latest", "main", "master", "tbd", "todo", "unassigned", "unresolved-before-c1"}
)
_ALLOWED_UNRESOLVED = (
    "sealed_execution.c0_evidence_release",
    "sealed_execution.provider_phase_plans",
)
_FREEZE_BLOCKERS = ("the immutable C0 evidence release remains unresolved",)
_EXPECTED_PRE_PROVIDER_SENTINELS = 7
_VALIDATION_COMMIT = "0" * 40
_MAX_JSON_BYTES = 256 * 1024 * 1024

_INPUT_NAMES = (
    "artifact_inventory",
    "artifact_inventory_receipt",
    "candidate_image_closure",
    "deployment_fragment",
    "development_freeze_receipt",
    "geometry_profiles",
    "joint_power_config",
    "joint_power_report",
    "post_embedding_receipt",
    "production_control_blueprint_receipt",
    "production_control_config",
    "production_control_config_write_receipt",
    "production_hardware_fragment",
    "production_workloads_fragment",
    "static_comparator",
    "template",
)

_EXPECTED_DIRECT_NAMES = {
    "artifact_inventory": INVENTORY_FILENAME,
    "artifact_inventory_receipt": INVENTORY_RECEIPT_FILENAME,
    "deployment_fragment": DEPLOYMENT_FRAGMENT_FILENAME,
    "development_freeze_receipt": "freeze-receipt.json",
    "geometry_profiles": "geometry-profiles.json",
    "joint_power_config": "joint-power-config.json",
    "joint_power_report": "report.json",
    "post_embedding_receipt": POST_EMBEDDING_RECEIPT_FILENAME,
    "production_control_blueprint_receipt": BLUEPRINT_RECEIPT_FILENAME,
    "production_control_config": "production-control-materialization.json",
    "production_control_config_write_receipt": (
        "production-control-materialization.write-receipt.json"
    ),
    "production_hardware_fragment": PRODUCTION_HARDWARE_FRAGMENT_FILENAME,
    "production_workloads_fragment": PRODUCTION_WORKLOADS_FRAGMENT_FILENAME,
    "static_comparator": "static-comparator.json",
    "template": "study-manifest.json",
}

_FREEZE_RECEIPT_FIELDS = frozenset(
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
_FREEZE_ARTIFACT_FIELDS = frozenset({"byte_count", "path", "sha256"})
_FREEZE_ARTIFACT_NAMES = frozenset(
    {
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
)
_INVENTORY_RECEIPT_FIELDS = frozenset(
    {
        "artifact_count",
        "artifact_root",
        "inventory_file_sha256",
        "repository_root",
        "schema_version",
        "template_sha256",
    }
)


class CandidateSourceComposerError(ValueError):
    """Raised when typed evidence cannot compose one candidate source."""


class CandidateSourcePublicationIndeterminateError(RuntimeError):
    """Raised when failed transaction state cannot be proved clean and absent."""


class CandidateSourceInterruptedError(BaseException):
    """A handled process signal interrupted a publication transaction."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"received signal {signal.Signals(signum).name}")


class _PublicationSignalGuard:
    """Convert termination signals into transactional exceptions."""

    _signals = tuple(
        item
        for item in (
            signal.SIGINT,
            signal.SIGTERM,
            getattr(signal, "SIGHUP", None),
        )
        if item is not None
    )

    def __init__(self) -> None:
        self._previous: dict[int, Any] = {}

    def _raise(self, signum: int, frame: object) -> None:
        del frame
        for handled in self._signals:
            signal.signal(handled, signal.SIG_IGN)
        raise CandidateSourceInterruptedError(signum)

    def __enter__(self) -> _PublicationSignalGuard:
        try:
            for signum in self._signals:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._raise)
        except (OSError, RuntimeError, ValueError) as exc:
            for signum, previous in self._previous.items():
                signal.signal(signum, previous)
            raise CandidateSourceComposerError(
                f"cannot install publication signal recovery: {exc}"
            ) from exc
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> bool:
        del exc_type, exc, traceback
        restore_failure: BaseException | None = None
        for signum, previous in self._previous.items():
            try:
                signal.signal(signum, previous)
            except BaseException as failure:
                if restore_failure is None:
                    restore_failure = failure
        if restore_failure is not None:
            raise CandidateSourcePublicationIndeterminateError(
                "cannot restore publication signal handlers"
            ) from restore_failure
        return False


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CandidateSourceComposerError("composition evidence is not canonical JSON") from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise CandidateSourceComposerError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise CandidateSourceComposerError(f"{label} must be one object with string fields")
    observed = frozenset(value)
    if observed != fields:
        raise CandidateSourceComposerError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _canonical_absolute_path(value: object, *, label: str) -> Path:
    if type(value) is not str or not value:
        raise CandidateSourceComposerError(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or str(path) != os.path.normpath(value):
        raise CandidateSourceComposerError(f"{label} must be a normalized absolute path")
    return path


@dataclass(frozen=True)
class ExactInput:
    """One exact producer file selected by an externally recorded byte digest."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _canonical_absolute_path(str(self.path), label="input path"),
        )
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, label="input file SHA-256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object, *, label: str) -> ExactInput:
        row = _closed(value, frozenset({"path", "sha256"}), label=label)
        return cls(
            path=_canonical_absolute_path(row["path"], label=f"{label}.path"),
            sha256=_require_sha256(row["sha256"], label=f"{label}.sha256"),
        )


@dataclass(frozen=True)
class CompositionRequest:
    """Closed wiring request. It contains paths and file hashes, never manifest values."""

    inputs: tuple[tuple[str, ExactInput], ...]
    schema_version: str = REQUEST_SCHEMA

    def __post_init__(self) -> None:
        mapping = dict(self.inputs)
        if tuple(sorted(mapping)) != _INPUT_NAMES or len(mapping) != len(self.inputs):
            raise CandidateSourceComposerError("composition request input roles differ")
        if not all(isinstance(value, ExactInput) for value in mapping.values()):
            raise CandidateSourceComposerError("composition request inputs must be typed")
        if len({value.path for value in mapping.values()}) != len(mapping):
            raise CandidateSourceComposerError("composition request input paths must be distinct")
        if self.schema_version != REQUEST_SCHEMA:
            raise CandidateSourceComposerError("composition request schema differs")
        object.__setattr__(self, "inputs", tuple(sorted(mapping.items())))

    def binding(self, name: str) -> ExactInput:
        try:
            return dict(self.inputs)[name]
        except KeyError as exc:  # pragma: no cover - __post_init__ owns the invariant
            raise CandidateSourceComposerError(f"composition request lacks {name}") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "inputs": {name: binding.to_dict() for name, binding in self.inputs},
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> CompositionRequest:
        row = _closed(
            value,
            frozenset({"inputs", "schema_version"}),
            label="composition request",
        )
        raw_inputs = row["inputs"]
        if not isinstance(raw_inputs, Mapping) or not all(type(key) is str for key in raw_inputs):
            raise CandidateSourceComposerError("composition request inputs must be one object")
        return cls(
            inputs=tuple(
                (name, ExactInput.from_dict(binding, label=f"inputs.{name}"))
                for name, binding in raw_inputs.items()
            ),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class CandidateDeploymentFragment:
    """Provisioned non-scientific identities and output locations."""

    custodian: str
    receipt_uri_template: str
    results_store: str
    schema_version: str = DEPLOYMENT_FRAGMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DEPLOYMENT_FRAGMENT_SCHEMA:
            raise CandidateSourceComposerError("candidate deployment fragment schema differs")
        if (
            type(self.custodian) is not str
            or not self.custodian
            or self.custodian != self.custodian.strip()
            or unicodedata.normalize("NFC", self.custodian) != self.custodian
            or self.custodian.casefold() in _PLACEHOLDER_VALUES
            or any(ord(character) < 32 or ord(character) == 127 for character in self.custodian)
        ):
            raise CandidateSourceComposerError("candidate deployment custodian is unresolved")
        if (
            type(self.results_store) is not str
            or not self.results_store
            or self.results_store != self.results_store.strip()
            or unicodedata.normalize("NFC", self.results_store) != self.results_store
            or any(ord(character) < 32 or ord(character) == 127 for character in self.results_store)
        ):
            raise CandidateSourceComposerError("candidate results_store is not one pinned URI")
        if (
            type(self.receipt_uri_template) is not str
            or not self.receipt_uri_template
            or self.receipt_uri_template != self.receipt_uri_template.strip()
            or unicodedata.normalize("NFC", self.receipt_uri_template) != self.receipt_uri_template
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in self.receipt_uri_template
            )
        ):
            raise CandidateSourceComposerError(
                "candidate receipt_uri_template is not the closed file-URI template"
            )
        try:
            results = urlsplit(self.results_store)
        except ValueError as exc:
            raise CandidateSourceComposerError(
                "candidate results_store is not one pinned URI"
            ) from exc
        results_path = unquote(results.path)
        if (
            results.scheme != "file"
            or results.username is not None
            or results.password is not None
            or results.query
            or results.fragment
            or "{" in self.results_store
            or "}" in self.results_store
            or results.netloc
            or not results_path.startswith("/")
            or Path(results_path).as_uri() != self.results_store
            or any(part in {".", ".."} for part in results_path.split("/"))
        ):
            raise CandidateSourceComposerError(
                "candidate results_store must be one canonical absolute file URI"
            )
        try:
            receipt = urlsplit(self.receipt_uri_template)
        except ValueError as exc:
            raise CandidateSourceComposerError(
                "candidate receipt_uri_template is not the closed file-URI template"
            ) from exc
        receipt_path = unquote(receipt.path)
        if (
            receipt.scheme != "file"
            or receipt.netloc not in {"", "localhost"}
            or receipt.query
            or receipt.fragment
            or not receipt_path.startswith("/")
            or any(part in {".", ".."} for part in receipt_path.split("/"))
            or self.receipt_uri_template.count("{manifest_sha256}") != 1
            or Path(receipt_path).name != "{manifest_sha256}.json"
        ):
            raise CandidateSourceComposerError(
                "candidate receipt_uri_template is not the closed file-URI template"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "custodian": self.custodian,
            "receipt_uri_template": self.receipt_uri_template,
            "results_store": self.results_store,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> CandidateDeploymentFragment:
        row = _closed(
            value,
            frozenset({"custodian", "receipt_uri_template", "results_store", "schema_version"}),
            label="candidate deployment fragment",
        )
        return cls(**row)  # type: ignore[arg-type]


def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_directory_chain(path: Path, *, label: str) -> int:
    if not path.is_absolute():
        raise CandidateSourceComposerError(f"{label} must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open("/", flags)
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise CandidateSourceComposerError(f"{label} is not one real directory")
        return descriptor
    except CandidateSourceComposerError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise CandidateSourceComposerError(f"cannot open {label}: {exc}") from exc


def _acquire_exclusive_lease(descriptor: int, *, label: str) -> None:
    """Acquire the cooperative custody lease used by every mutable namespace."""

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise CandidateSourceComposerError(
            f"cannot acquire exclusive custody lease for {label}: {exc}"
        ) from exc


@dataclass
class _RetainedFileRead:
    """One descriptor-bound read retained until its enclosing scan closes."""

    descriptor: int
    parent_descriptor: int
    name: str
    label: str
    before: os.stat_result
    encoded: bytes | None = None

    def read(self) -> bytes:
        if self.encoded is not None:
            return self.encoded
        chunks: list[bytes] = []
        observed = 0
        while observed <= _MAX_JSON_BYTES:
            chunk = os.read(
                self.descriptor,
                min(64 * 1024, _MAX_JSON_BYTES + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        if observed > _MAX_JSON_BYTES:
            raise CandidateSourceComposerError(f"{self.label} exceeds the byte limit")
        self.encoded = b"".join(chunks)
        return self.encoded

    def assert_current(self) -> None:
        try:
            after = os.fstat(self.descriptor)
            named = os.stat(
                self.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise CandidateSourceComposerError(
                f"cannot close the exact read of {self.label}: {exc}"
            ) from exc
        if _stat_signature(self.before) != _stat_signature(after) or _stat_signature(
            self.before
        ) != _stat_signature(named):
            raise CandidateSourceComposerError(f"{self.label} changed during its exact read")

    def close(self) -> None:
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            os.close(descriptor)


def _open_owned_file_at(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
    private: bool = True,
    required_mode: int | None = None,
    exclusive_lease: bool = False,
) -> _RetainedFileRead:
    if not name or name in {".", ".."} or "/" in name:
        raise CandidateSourceComposerError(f"{label} member name is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        if exclusive_lease:
            _acquire_exclusive_lease(descriptor, label=label)
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        allowed_modes = (
            {required_mode}
            if required_mode is not None
            else ({0o400, 0o600} if private else {0o400, 0o444, 0o600, 0o644})
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_JSON_BYTES
            or mode not in allowed_modes
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise CandidateSourceComposerError(f"{label} is not one owned exact regular file")
        retained = _RetainedFileRead(
            descriptor=descriptor,
            parent_descriptor=parent_descriptor,
            name=name,
            label=label,
            before=before,
        )
        descriptor = None
        return retained
    except CandidateSourceComposerError:
        raise
    except OSError as exc:
        raise CandidateSourceComposerError(f"cannot open {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_owned_file_at(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
    private: bool = True,
    required_mode: int | None = None,
) -> bytes:
    retained = _open_owned_file_at(
        parent_descriptor,
        name,
        label=label,
        private=private,
        required_mode=required_mode,
    )
    try:
        encoded = retained.read()
        retained.assert_current()
        return encoded
    finally:
        retained.close()


def _read_owned_file(path: Path, *, label: str, private: bool = True) -> bytes:
    parent_descriptor = _open_directory_chain(path.parent, label=f"{label} parent")
    try:
        return _read_owned_file_at(
            parent_descriptor,
            path.name,
            label=label,
            private=private,
        )
    finally:
        os.close(parent_descriptor)


def _read_exact_file(
    binding: ExactInput,
    *,
    label: str,
    private: bool = True,
) -> bytes:
    encoded = _read_owned_file(binding.path, label=label, private=private)
    if _sha256(encoded) != binding.sha256:
        raise CandidateSourceComposerError(f"{label} differs from the request file digest")
    return encoded


def _reject_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateSourceComposerError(f"canonical JSON repeats field {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> object:
    raise CandidateSourceComposerError(f"canonical JSON contains non-finite number {token}")


def _decode_json(encoded: bytes, *, label: str, canonical: bool = True) -> object:
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite,
        )
    except CandidateSourceComposerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateSourceComposerError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if canonical and encoded != _canonical_bytes(value):
        raise CandidateSourceComposerError(f"{label} bytes are not canonical JSON plus LF")
    return value


def _request_from_exact_bytes(
    binding: ExactInput,
    encoded: bytes,
) -> CompositionRequest:
    if _sha256(encoded) != binding.sha256:
        raise CandidateSourceComposerError(
            "composition request differs from the request file digest"
        )
    request = CompositionRequest.from_dict(_decode_json(encoded, label="composition request"))
    if encoded != request.canonical_file_bytes() or request.file_sha256 != binding.sha256:
        raise CandidateSourceComposerError("composition request changed after typed parsing")
    for name, expected in _EXPECTED_DIRECT_NAMES.items():
        if request.binding(name).path.name != expected:
            raise CandidateSourceComposerError(
                f"{name} must use the fixed producer filename {expected!r}"
            )
    return request


def _load_request(path: Path, expected_sha256: str) -> CompositionRequest:
    binding = ExactInput(
        path=path,
        sha256=_require_sha256(expected_sha256, label="composition request SHA-256"),
    )
    return _request_from_exact_bytes(
        binding,
        _read_exact_file(binding, label="composition request"),
    )


@dataclass
class _RetainedParent:
    path: Path
    descriptor: int
    control_identity: tuple[int, int, int, int, int]

    def assert_current(self) -> None:
        _assert_path_names_directory(
            self.path,
            self.descriptor,
            label=f"input custody parent {self.path}",
            expected_control_identity=self.control_identity,
        )

    def close(self) -> None:
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            os.close(descriptor)


@dataclass
class _CapturedInputSet:
    """Captured request/input bytes held under cooperative exclusive leases."""

    request: CompositionRequest
    request_path: Path
    request_bytes: bytes
    encoded_inputs: dict[str, bytes]
    retained_files: list[_RetainedFileRead]
    retained_parents: list[_RetainedParent]
    closed: bool = False

    @property
    def capture_set_sha256(self) -> str:
        return _sha256(
            _canonical_bytes(
                {
                    "inputs": {
                        name: {
                            "byte_count": len(self.encoded_inputs[name]),
                            "path": str(binding.path),
                            "sha256": _sha256(self.encoded_inputs[name]),
                        }
                        for name, binding in self.request.inputs
                    },
                    "request": {
                        "byte_count": len(self.request_bytes),
                        "path": str(self.request_path),
                        "sha256": _sha256(self.request_bytes),
                    },
                }
            )
        )

    def input_bytes(self, name: str) -> bytes:
        try:
            return self.encoded_inputs[name]
        except KeyError as exc:  # pragma: no cover - capture owns the closed role set
            raise CandidateSourceComposerError(f"captured input set lacks {name}") from exc

    def assert_current(self) -> None:
        if self.closed:
            raise CandidateSourceComposerError("input custody lease is already closed")
        for retained in self.retained_files:
            retained.assert_current()
        for parent in self.retained_parents:
            parent.assert_current()

    def open_leased_parent(self, path: Path) -> int:
        """Open a parent, sharing an already-held lease when identities coincide."""

        descriptor = _open_private_parent(path)
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        for parent in self.retained_parents:
            opened = os.fstat(parent.descriptor)
            if (opened.st_dev, opened.st_ino) == identity:
                os.close(descriptor)
                duplicate = os.dup(parent.descriptor)
                _assert_path_names_directory(
                    path,
                    duplicate,
                    label="composition output parent",
                    expected_control_identity=parent.control_identity,
                )
                return duplicate
        try:
            _acquire_exclusive_lease(descriptor, label="composition output parent")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def close(self) -> None:
        if self.closed:
            return
        first_failure: BaseException | None = None
        for retained in reversed(self.retained_files):
            try:
                retained.close()
            except BaseException as exc:  # cleanup must attempt every retained descriptor
                if first_failure is None:
                    first_failure = exc
        for parent in reversed(self.retained_parents):
            try:
                parent.close()
            except BaseException as exc:  # cleanup must attempt every retained descriptor
                if first_failure is None:
                    first_failure = exc
        self.closed = True
        if first_failure is not None:
            raise first_failure


def _capture_input_set(
    request_path: Path,
    request_sha256: str,
) -> _CapturedInputSet:
    """Capture all authority bytes once and retain their cooperative custody leases."""

    request_binding = ExactInput(
        path=request_path,
        sha256=_require_sha256(request_sha256, label="composition request SHA-256"),
    )
    retained_files: list[_RetainedFileRead] = []
    retained_parents: list[_RetainedParent] = []
    parent_by_inode: dict[tuple[int, int], _RetainedParent] = {}

    def retained_parent(path: Path) -> _RetainedParent:
        descriptor = _open_directory_chain(path, label=f"input custody parent {path}")
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        existing = parent_by_inode.get(identity)
        if existing is not None:
            os.close(descriptor)
            _assert_path_names_directory(
                path,
                existing.descriptor,
                label=f"input custody parent {path}",
                expected_control_identity=existing.control_identity,
            )
            return existing
        try:
            _acquire_exclusive_lease(descriptor, label=f"input custody parent {path}")
            parent = _RetainedParent(
                path=path,
                descriptor=descriptor,
                control_identity=_directory_control_identity(metadata),
            )
            parent.assert_current()
        except BaseException:
            os.close(descriptor)
            raise
        parent_by_inode[identity] = parent
        retained_parents.append(parent)
        return parent

    def capture(
        binding: ExactInput,
        *,
        label: str,
        private: bool = True,
    ) -> bytes:
        parent = retained_parent(binding.path.parent)
        retained = _open_owned_file_at(
            parent.descriptor,
            binding.path.name,
            label=label,
            private=private,
            exclusive_lease=True,
        )
        retained_files.append(retained)
        encoded = retained.read()
        retained.assert_current()
        if _sha256(encoded) != binding.sha256:
            raise CandidateSourceComposerError(f"{label} differs from the request file digest")
        return encoded

    try:
        request_bytes = capture(request_binding, label="composition request")
        request = _request_from_exact_bytes(request_binding, request_bytes)
        encoded_inputs = {
            name: capture(
                binding,
                label=name,
                private=name != "template",
            )
            for name, binding in request.inputs
        }
        captured = _CapturedInputSet(
            request=request,
            request_path=request_path,
            request_bytes=request_bytes,
            encoded_inputs=encoded_inputs,
            retained_files=retained_files,
            retained_parents=retained_parents,
        )
        captured.assert_current()
        return captured
    except BaseException:
        for retained in reversed(retained_files):
            try:
                retained.close()
            except BaseException:
                pass
        for parent in reversed(retained_parents):
            try:
                parent.close()
            except BaseException:
                pass
        raise


def _load_json_input(
    request: CompositionRequest,
    name: str,
    *,
    canonical: bool = True,
    private: bool = True,
    captured: _CapturedInputSet | None = None,
) -> tuple[object, bytes]:
    encoded = (
        captured.input_bytes(name)
        if captured is not None
        else _read_exact_file(request.binding(name), label=name, private=private)
    )
    if _sha256(encoded) != request.binding(name).sha256:
        raise CandidateSourceComposerError(f"{name} differs from the request file digest")
    return _decode_json(encoded, label=name, canonical=canonical), encoded


def _freeze_artifact_pins(value: object) -> dict[str, str]:
    receipt = _closed(value, _FREEZE_RECEIPT_FIELDS, label="development freeze receipt")
    if receipt["schema_version"] != DEVELOPMENT_FREEZE_SCHEMA:
        raise CandidateSourceComposerError("development freeze receipt schema differs")
    for name in ("development_group_digest", "feature_schema_digest", "model_suite_digest"):
        _require_sha256(receipt[name], label=f"development freeze {name}")
    if receipt["static_comparator"] != "hnsw-high":
        raise CandidateSourceComposerError("development freeze comparator differs")
    if not isinstance(receipt["source_bindings"], list) or not receipt["source_bindings"]:
        raise CandidateSourceComposerError("development freeze source bindings are absent")
    raw_artifacts = receipt["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise CandidateSourceComposerError("development freeze artifacts must be an array")
    pins: dict[str, str] = {}
    for value in raw_artifacts:
        row = _closed(value, _FREEZE_ARTIFACT_FIELDS, label="development freeze artifact")
        name = row["path"]
        if type(name) is not str or Path(name).name != name or name in pins:
            raise CandidateSourceComposerError("development freeze artifact path is invalid")
        if (
            type(row["byte_count"]) is not int
            or isinstance(row["byte_count"], bool)
            or row["byte_count"] <= 0
        ):
            raise CandidateSourceComposerError(
                f"development freeze artifact {name!r} byte_count is invalid"
            )
        pins[name] = _require_sha256(
            row["sha256"],
            label=f"development freeze artifact {name!r} SHA-256",
        )
    if frozenset(pins) != _FREEZE_ARTIFACT_NAMES:
        raise CandidateSourceComposerError("development freeze artifact closure differs")
    return pins


def _typed_geometry_profiles(value: object) -> Mapping[str, Any]:
    row = _closed(
        value,
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
        row["schema_version"] != GEOMETRY_PROFILE_SCHEMA
        or row["fit_partition_only"] is not True
        or row["quantile_method"] != "numpy-linear"
        or row["quantiles"] != [0.25, 0.75]
    ):
        raise CandidateSourceComposerError("geometry profile contract differs")
    return row


def _typed_static_comparator(value: object) -> str:
    row = _closed(
        value,
        frozenset({"action", "chosen_a_priori", "schema_version", "selection_data"}),
        label="static comparator",
    )
    if row != {
        "action": "hnsw-high",
        "chosen_a_priori": True,
        "schema_version": COMPARATOR_ARTIFACT_SCHEMA,
        "selection_data": None,
    }:
        raise CandidateSourceComposerError("static comparator contract differs")
    return "hnsw-high"


def _typed_workload_fragment(
    value: object,
    *,
    selected_families_per_corpus: int,
    config: ProductionControlMaterializationConfig,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(FIXED_CORPORA):
        raise CandidateSourceComposerError("production workload fragment must have five rows")
    result: list[dict[str, object]] = []
    for position, (raw, corpus_id) in enumerate(zip(value, FIXED_CORPORA, strict=True)):
        row = _closed(
            raw,
            frozenset({"canonical_file_sha256", "corpus_id", "spec"}),
            label=f"production workload row {position}",
        )
        if row["corpus_id"] != corpus_id:
            raise CandidateSourceComposerError("production workload corpus order differs")
        try:
            spec = ProductionCorpusWorkloadSpec.from_dict(row["spec"])
        except (ProductionCorpusRunError, TypeError, ValueError) as exc:
            raise CandidateSourceComposerError(
                f"production workload {corpus_id!r} is not typed: {exc}"
            ) from exc
        if (
            spec.corpus_id != corpus_id
            or spec.code_commit != C0_COMMIT_SENTINEL
            or spec.selected_family_count != selected_families_per_corpus
            or spec.runner_identity != config.runner_identity
            or spec.runner_image != config.scientific_production_reference
            or row["canonical_file_sha256"] != production_workload_file_sha256(spec.to_dict())
        ):
            raise CandidateSourceComposerError(
                f"production workload {corpus_id!r} retains an override or unresolved binding"
            )
        result.append(
            {
                "canonical_file_sha256": row["canonical_file_sha256"],
                "corpus_id": corpus_id,
                "spec": spec.to_dict(),
            }
        )
    return result


def _typed_hardware_fragment(
    value: object,
    *,
    config: ProductionControlMaterializationConfig,
) -> dict[str, object]:
    row = _closed(
        value,
        frozenset(
            {
                "accelerator",
                "cpu_model",
                "instance_type",
                "logical_cores",
                "memory_gib",
                "operating_system",
                "provider",
                "region",
            }
        ),
        label="production hardware fragment",
    )
    expected = {
        "accelerator": config.hardware_accelerator,
        "cpu_model": config.hardware_cpu_model,
        "instance_type": config.hardware_instance_type,
        "logical_cores": len(config.cpuset_cpus),
        "memory_gib": config.memory_limit_bytes // (1024**3),
        "operating_system": config.hardware_operating_system,
        "provider": config.hardware_provider,
        "region": config.hardware_region,
    }
    if dict(row) != expected:
        raise CandidateSourceComposerError(
            "production hardware fragment differs from the typed materialization config"
        )
    return expected


def _required_joint_lower_bound(report: JointPowerDesignReport) -> float:
    selected = report.selected_families_per_corpus
    if selected is None or not report.selection_satisfied or not report.freeze_ready:
        raise CandidateSourceComposerError("joint-power report is not freeze-ready")
    estimates = [
        item
        for item in report.estimates
        if item.selection_required and item.families_per_corpus == selected
    ]
    required_scenarios = {item.scenario_id for item in report.estimates if item.selection_required}
    if {item.scenario_id for item in estimates} != required_scenarios or not estimates:
        raise CandidateSourceComposerError("joint-power selected-cell closure differs")
    return min(float(item.joint_probability.lower_probability_bound) for item in estimates)


def _assert_design_alignment(
    analysis: Mapping[str, Any],
    config: JointPowerDesignConfig,
    report: JointPowerDesignReport,
) -> None:
    expected = {
        "alpha": config.alpha,
        "evidence_corpora": list(config.evidence_corpora),
        "evidence_sufficiency_noninferiority_margin": (
            config.evidence_sufficiency_noninferiority_margin
        ),
        "fixed_corpora": list(config.fixed_corpora),
        "maximum_entitlement_violations": config.maximum_denied_emissions,
        "maximum_p95_latency_ratio": config.maximum_p95_latency_ratio,
        "minimum_corpora_with_geometry_gain": config.minimum_corpora_with_geometry_gain,
        "minimum_cost_reduction": config.minimum_latency_reduction,
        "power_target": config.target_power,
        "retrieval_target_noninferiority_margin": (config.retrieval_target_noninferiority_margin),
    }
    observed = {name: analysis.get(name) for name in expected}
    if observed != expected:
        raise CandidateSourceComposerError(
            "tracked analysis constants differ from the typed joint-power design"
        )
    power = analysis.get("power")
    if not isinstance(power, Mapping):
        raise CandidateSourceComposerError("tracked analysis power object is malformed")
    expected_power = {
        "candidate_families_per_corpus": list(config.candidate_families_per_corpus),
        "effect_scenarios": sorted(
            item.scenario_id for item in config.effect_scenarios if item.selection_required
        ),
        "selection_cell_alpha": report.selection_cell_alpha,
        "selection_family_size": report.selection_family_size,
        "selection_familywise_confidence": report.selection_familywise_confidence,
        "selection_multiplicity_method": report.selection_multiplicity_method,
        "simulation_count": config.n_simulations,
    }
    for name, value in expected_power.items():
        if power.get(name) != value:
            raise CandidateSourceComposerError(
                f"tracked analysis.power.{name} differs from typed design evidence"
            )


def _assert_freeze_design_provenance(
    *,
    freeze_pins: Mapping[str, str],
    geometry_profiles: Mapping[str, Any],
    config: JointPowerDesignConfig,
    report: JointPowerDesignReport,
) -> None:
    calibration_digest = freeze_pins["development-calibration-outcomes.json"]
    if (
        config.dependence_source.partition != "development-calibration"
        or config.dependence_source.artifact_sha256 != calibration_digest
        or config.dependence_source.artifact_uri != f"urn:sha256:{calibration_digest}"
    ):
        raise CandidateSourceComposerError(
            "joint-power dependence source differs from the development-freeze receipt"
        )
    expected_panels = {
        "conservative-registered-attenuation": freeze_pins["joint-power-conservative-panel.json"],
        "expected-development-effect": freeze_pins["joint-power-expected-panel.json"],
    }
    config_panels = {item.scenario_id: item.panel_sha256 for item in config.effect_scenarios}
    if config_panels != expected_panels or dict(report.panel_sha256s) != expected_panels:
        raise CandidateSourceComposerError(
            "joint-power scenario pins differ from the development-freeze receipt"
        )
    if geometry_profiles["geometry_gain_thresholds"] != config.geometry_gain_thresholds.to_dict():
        raise CandidateSourceComposerError(
            "geometry profile thresholds differ from the joint-power config"
        )


def _collect_exact_values(value: object, targets: frozenset[str], *, path: str = "") -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            result.update(_collect_exact_values(item, targets, path=child))
    elif isinstance(value, list):
        for position, item in enumerate(value):
            result.update(_collect_exact_values(item, targets, path=f"{path}[{position}]"))
    elif type(value) is str and value.strip().casefold() in targets:
        result.add(path)
    return result


def _replace_exact_scalar(value: object, target: str, replacement: str) -> object:
    if isinstance(value, Mapping):
        return {
            key: _replace_exact_scalar(item, target, replacement) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_exact_scalar(item, target, replacement) for item in value]
    return replacement if value == target else value


@dataclass(frozen=True)
class _Authorities:
    template: Mapping[str, Any]
    artifact_inventory: Any
    post_embedding: Any
    power_config: JointPowerDesignConfig
    power_report: JointPowerDesignReport
    geometry_profiles: Mapping[str, Any]
    static_comparator: str
    production_config: ProductionControlMaterializationConfig
    production_config_write_receipt: ProductionControlMaterializationConfigWriteReceipt
    production_blueprint: ProductionControlBlueprintReceipt
    workloads: list[dict[str, object]]
    hardware: dict[str, object]
    candidate_image_closure: CandidateImageClosure
    deployment: CandidateDeploymentFragment


def _load_captured_artifact_inventory(
    *,
    request: CompositionRequest,
    inventory_value: object,
    inventory_bytes: bytes,
    receipt_value: object,
    template_value: Mapping[str, Any],
) -> CandidateArtifactPinInventory:
    """Validate the typed inventory solely from the bytes already admitted by digest."""

    try:
        inventory = CandidateArtifactPinInventory.from_dict(inventory_value)
    except (TypeError, ValueError) as exc:
        raise CandidateSourceComposerError(f"artifact inventory is not typed: {exc}") from exc
    receipt = _closed(
        receipt_value,
        _INVENTORY_RECEIPT_FIELDS,
        label="artifact inventory receipt",
    )
    for name in ("artifact_root", "repository_root"):
        _canonical_absolute_path(
            receipt[name],
            label=f"artifact inventory receipt {name}",
        )
    if (
        inventory.canonical_file_bytes != inventory_bytes
        or inventory.file_sha256 != request.binding("artifact_inventory").sha256
        or receipt["schema_version"] != CANDIDATE_ARTIFACT_PIN_RECEIPT_SCHEMA
        or receipt["artifact_count"] != 79
        or receipt["inventory_file_sha256"] != inventory.file_sha256
        or receipt["template_sha256"] != inventory.template_sha256
        or inventory.template_sha256 != _sha256(_canonical_bytes(template_value))
    ):
        raise CandidateSourceComposerError("artifact inventory captured-byte closure differs")
    return inventory


def _load_authorities(
    request: CompositionRequest,
    captured: _CapturedInputSet | None = None,
) -> _Authorities:
    template_value, template_bytes = _load_json_input(
        request,
        "template",
        canonical=False,
        private=False,
        captured=captured,
    )
    if not isinstance(template_value, Mapping):
        raise CandidateSourceComposerError("study-manifest template must be one object")
    validate_study_manifest(template_value)

    inventory_value, inventory_bytes = _load_json_input(
        request,
        "artifact_inventory",
        captured=captured,
    )
    inventory_receipt_value, _ = _load_json_input(
        request,
        "artifact_inventory_receipt",
        captured=captured,
    )
    if not isinstance(inventory_value, Mapping):
        raise CandidateSourceComposerError("artifact inventory must be one object")
    inventory_root = request.binding("artifact_inventory").path.parent
    if request.binding("artifact_inventory_receipt").path.parent != inventory_root:
        raise CandidateSourceComposerError(
            "artifact inventory and its receipt have different producer roots"
        )
    inventory = _load_captured_artifact_inventory(
        request=request,
        inventory_value=inventory_value,
        inventory_bytes=inventory_bytes,
        receipt_value=inventory_receipt_value,
        template_value=template_value,
    )
    if request.binding("template").sha256 != _sha256(template_bytes):
        raise CandidateSourceComposerError("template exact bytes changed after parsing")

    post_value, post_bytes = _load_json_input(
        request,
        "post_embedding_receipt",
        captured=captured,
    )
    post = PostEmbeddingDevelopmentReceipt.from_dict(post_value)
    if (
        post.schema_version != POST_EMBEDDING_RECEIPT_SCHEMA
        or post.artifact_sha256 != _sha256(post_bytes)
        or post.canonical_file_bytes() != post_bytes
    ):
        raise CandidateSourceComposerError("post-embedding receipt typed digest differs")

    post_root = request.binding("post_embedding_receipt").path.parent
    freeze_root = post_root / FREEZE_DIRECTORY
    expected_paths = {
        "development_freeze_receipt": freeze_root / "freeze-receipt.json",
        "geometry_profiles": freeze_root / "geometry-profiles.json",
        "joint_power_config": freeze_root / "joint-power-config.json",
        "static_comparator": freeze_root / "static-comparator.json",
        "joint_power_report": post_root / "analysis" / "joint-power-design" / "report.json",
    }
    for name, expected in expected_paths.items():
        if request.binding(name).path != expected:
            raise CandidateSourceComposerError(f"{name} is outside its typed producer root")

    freeze_value, _ = _load_json_input(
        request,
        "development_freeze_receipt",
        captured=captured,
    )
    if request.binding("development_freeze_receipt").sha256 != post.freeze_receipt_sha256:
        raise CandidateSourceComposerError("development freeze receipt differs from post-embedding")
    freeze_pins = _freeze_artifact_pins(freeze_value)

    geometry_value, _ = _load_json_input(
        request,
        "geometry_profiles",
        captured=captured,
    )
    comparator_value, _ = _load_json_input(
        request,
        "static_comparator",
        captured=captured,
    )
    power_config_value, power_config_bytes = _load_json_input(
        request,
        "joint_power_config",
        captured=captured,
    )
    _, power_report_bytes = _load_json_input(
        request,
        "joint_power_report",
        captured=captured,
    )
    for name, filename in (
        ("geometry_profiles", "geometry-profiles.json"),
        ("static_comparator", "static-comparator.json"),
        ("joint_power_config", "joint-power-config.json"),
    ):
        if request.binding(name).sha256 != freeze_pins[filename]:
            raise CandidateSourceComposerError(
                f"{name} differs from the development-freeze receipt"
            )
    power_config = load_joint_power_config(power_config_bytes)
    power_report = load_joint_power_report(power_report_bytes)
    geometry_profiles = _typed_geometry_profiles(geometry_value)
    if (
        power_config.sha256 != post.joint_power_config_sha256
        or power_report.sha256 != post.joint_power_report_sha256
        or power_report.config_sha256 != power_config.sha256
        or power_report.selected_families_per_corpus != post.selected_families_per_corpus
    ):
        raise CandidateSourceComposerError(
            "joint-power config/report/post-embedding closure differs"
        )
    if power_config_value != power_config.to_dict():
        raise CandidateSourceComposerError("joint-power config changed after typed parsing")
    _assert_freeze_design_provenance(
        freeze_pins=freeze_pins,
        geometry_profiles=geometry_profiles,
        config=power_config,
        report=power_report,
    )

    config_value, config_bytes = _load_json_input(
        request,
        "production_control_config",
        captured=captured,
    )
    production_config = ProductionControlMaterializationConfig.from_dict(config_value)
    if (
        production_config.file_sha256 != _sha256(config_bytes)
        or production_config.canonical_file_bytes() != config_bytes
    ):
        raise CandidateSourceComposerError("production control config typed digest differs")
    config_receipt_value, config_receipt_bytes = _load_json_input(
        request,
        "production_control_config_write_receipt",
        captured=captured,
    )
    config_receipt = ProductionControlMaterializationConfigWriteReceipt.from_dict(
        config_receipt_value
    )
    if (
        config_receipt.file_sha256 != _sha256(config_receipt_bytes)
        or config_receipt.canonical_file_bytes() != config_receipt_bytes
        or config_receipt.config_path != request.binding("production_control_config").path
        or config_receipt.config_file_sha256 != production_config.file_sha256
        or config_receipt.config_readback_sha256 != production_config.file_sha256
        or config_receipt.candidate_image_source_commit
        != production_config.candidate_image_source_commit
        or config_receipt.approval_environment != production_config.approval_environment
    ):
        raise CandidateSourceComposerError("production config differs from its typed write receipt")

    if (
        request.binding("production_control_blueprint_receipt").path
        != production_config.blueprint_receipt_path
        or request.binding("production_workloads_fragment").path
        != production_config.production_workloads_fragment_path
        or request.binding("production_hardware_fragment").path
        != production_config.production_hardware_fragment_path
    ):
        raise CandidateSourceComposerError("production fragments are outside the typed blueprint")

    blueprint_value, blueprint_bytes = _load_json_input(
        request,
        "production_control_blueprint_receipt",
        captured=captured,
    )
    blueprint = ProductionControlBlueprintReceipt.from_dict(blueprint_value)
    if (
        blueprint.file_sha256 != _sha256(blueprint_bytes)
        or blueprint.canonical_file_bytes() != blueprint_bytes
        or blueprint.materialization_config_sha256 != production_config.file_sha256
        or blueprint.approval_environment != production_config.approval_environment
        or blueprint.runner_image != production_config.scientific_production_reference
    ):
        raise CandidateSourceComposerError(
            "production blueprint differs from its typed materialization config"
        )

    workloads_value, _ = _load_json_input(
        request,
        "production_workloads_fragment",
        captured=captured,
    )
    hardware_value, _ = _load_json_input(
        request,
        "production_hardware_fragment",
        captured=captured,
    )
    if (
        request.binding("production_workloads_fragment").sha256
        != blueprint.production_workloads_fragment_file_sha256
        or request.binding("production_hardware_fragment").sha256
        != blueprint.production_hardware_fragment_file_sha256
    ):
        raise CandidateSourceComposerError("production fragment digest differs from blueprint")
    workloads = _typed_workload_fragment(
        workloads_value,
        selected_families_per_corpus=post.selected_families_per_corpus,
        config=production_config,
    )
    hardware = _typed_hardware_fragment(hardware_value, config=production_config)

    closure_value, closure_bytes = _load_json_input(
        request,
        "candidate_image_closure",
        captured=captured,
    )
    closure_row = _closed(
        closure_value,
        frozenset(CandidateImageClosure.__dataclass_fields__),
        label="candidate image closure",
    )
    closure = CandidateImageClosure(**closure_row)  # type: ignore[arg-type]
    if (
        closure.file_sha256 != _sha256(closure_bytes)
        or _canonical_bytes(closure.to_dict()) != closure_bytes
        or closure.github_sha != production_config.candidate_image_source_commit
        or closure.scientific_image_reference != production_config.scientific_candidate_reference
        or closure.scientific_image_index_digest != production_config.scientific_index_digest
        or production_config.scientific_production_reference.rsplit("@", 1)[1]
        != closure.scientific_image_index_digest
    ):
        raise CandidateSourceComposerError(
            "candidate image and production-control P/T/D authority differ"
        )

    deployment_value, _ = _load_json_input(
        request,
        "deployment_fragment",
        captured=captured,
    )
    deployment = CandidateDeploymentFragment.from_dict(deployment_value)

    return _Authorities(
        template=template_value,
        artifact_inventory=inventory,
        post_embedding=post,
        power_config=power_config,
        power_report=power_report,
        geometry_profiles=geometry_profiles,
        static_comparator=_typed_static_comparator(comparator_value),
        production_config=production_config,
        production_config_write_receipt=config_receipt,
        production_blueprint=blueprint,
        workloads=workloads,
        hardware=hardware,
        candidate_image_closure=closure,
        deployment=deployment,
    )


def _compose_candidate_source(authorities: _Authorities) -> dict[str, Any]:
    candidate = apply_candidate_artifact_inventory(
        authorities.template,
        authorities.artifact_inventory,
    )
    analysis = candidate.get("analysis")
    if not isinstance(analysis, dict):
        raise CandidateSourceComposerError("study template analysis object is malformed")
    _assert_design_alignment(analysis, authorities.power_config, authorities.power_report)

    thresholds = authorities.geometry_profiles["geometry_gain_thresholds"]
    low_geometry = authorities.geometry_profiles["low_geometry"]
    high_geometry = authorities.geometry_profiles["high_geometry"]
    if not all(isinstance(value, Mapping) for value in (thresholds, low_geometry, high_geometry)):
        raise CandidateSourceComposerError("typed geometry profiles are malformed")
    analysis["nested_rows_per_family"] = authorities.power_config.nested_rows_per_family
    analysis["geometry_gain_thresholds"] = copy.deepcopy(dict(thresholds))
    analysis["low_geometry"] = copy.deepcopy(dict(low_geometry))
    analysis["high_geometry"] = copy.deepcopy(dict(high_geometry))
    analysis["static_comparator_action"] = authorities.static_comparator

    power = analysis.get("power")
    if not isinstance(power, dict):
        raise CandidateSourceComposerError("study template power object is malformed")
    required_scenarios = sorted(
        item.scenario_id
        for item in authorities.power_config.effect_scenarios
        if item.selection_required
    )
    power["dependence_source"] = authorities.power_config.dependence_source.artifact_uri
    power["effect_scenarios"] = required_scenarios
    power["candidate_families_per_corpus"] = list(
        authorities.power_config.candidate_families_per_corpus
    )
    power["selected_families_per_corpus"] = authorities.power_report.selected_families_per_corpus
    power["simulation_seed"] = authorities.power_config.simulation_seed
    power["simulation_count"] = authorities.power_config.n_simulations
    power["selection_cell_alpha"] = authorities.power_report.selection_cell_alpha
    power["selection_family_size"] = authorities.power_report.selection_family_size
    power["selection_familywise_confidence"] = (
        authorities.power_report.selection_familywise_confidence
    )
    power["selection_multiplicity_method"] = authorities.power_report.selection_multiplicity_method
    power["selected_joint_power_lower_bound"] = _required_joint_lower_bound(
        authorities.power_report
    )

    candidate["freeze_blockers"] = list(_FREEZE_BLOCKERS)
    candidate["production_workloads"] = copy.deepcopy(authorities.workloads)
    sealed = candidate.get("sealed_execution")
    if not isinstance(sealed, dict):
        raise CandidateSourceComposerError("study template sealed_execution is malformed")
    sealed["custodian"] = authorities.deployment.custodian
    sealed["approval_environment"] = authorities.production_config.approval_environment
    sealed["results_store"] = authorities.deployment.results_store
    sealed["runner_identity"] = authorities.production_config.runner_identity
    sealed["code_commit"] = C0_COMMIT_SENTINEL
    sealed["c0_evidence_release"] = "tbd"
    sealed["runner_image"] = authorities.production_config.scientific_production_reference
    sealed["provider_phase_plans"] = "tbd"
    sealed["production_controls"] = {
        "blueprint_receipt_file_sha256": authorities.production_blueprint.file_sha256,
        "blueprint_receipt_sha256": authorities.production_blueprint.semantic_sha256,
        "materialization_config_file_sha256": authorities.production_config.file_sha256,
    }
    sealed["hardware"] = copy.deepcopy(authorities.hardware)
    sealed["receipt_uri_template"] = authorities.deployment.receipt_uri_template

    unresolved = _collect_exact_values(candidate, _PLACEHOLDER_VALUES)
    if unresolved != set(_ALLOWED_UNRESOLVED):
        raise CandidateSourceComposerError(
            "candidate source unresolved path set differs; "
            f"missing={sorted(set(_ALLOWED_UNRESOLVED) - unresolved)}, "
            f"unexpected={sorted(unresolved - set(_ALLOWED_UNRESOLVED))}"
        )
    sentinels = _collect_exact_values(candidate, frozenset({C0_COMMIT_SENTINEL}))
    if len(sentinels) != _EXPECTED_PRE_PROVIDER_SENTINELS:
        raise CandidateSourceComposerError(
            "candidate source must contain exactly seven pre-provider C0 sentinels"
        )
    try:
        validation_probe = _replace_exact_scalar(
            candidate,
            C0_COMMIT_SENTINEL,
            _VALIDATION_COMMIT,
        )
        if not isinstance(validation_probe, dict):
            raise CandidateSourceComposerError("candidate validation probe is malformed")
        validation_workloads = validation_probe["production_workloads"]
        if not isinstance(validation_workloads, list):
            raise CandidateSourceComposerError("candidate validation workloads are malformed")
        for row in validation_workloads:
            row["canonical_file_sha256"] = production_workload_file_sha256(row["spec"])
        validate_study_manifest(validation_probe)
        _admit_candidate_source_shell(
            candidate,
            candidate_image_source_commit=authorities.candidate_image_closure.github_sha,
            config=authorities.production_config,
            control_blueprint=authorities.production_blueprint,
        )
    except ValueError as exc:
        raise CandidateSourceComposerError(
            f"composed source fails the existing provider boundary: {exc}"
        ) from exc
    return candidate


def _derive_candidate_source(
    request: CompositionRequest,
    captured: _CapturedInputSet | None = None,
) -> dict[str, Any]:
    try:
        return _compose_candidate_source(_load_authorities(request, captured))
    except CandidateSourceComposerError:
        raise
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CandidateSourceComposerError(
            f"typed producer evidence failed closed admission: {exc}"
        ) from exc


def _composition_receipt(
    request_path: Path,
    request: CompositionRequest,
    source: Mapping[str, Any],
    captured: _CapturedInputSet,
) -> dict[str, object]:
    source_bytes = _canonical_bytes(source)
    return {
        "allowed_unresolved_paths": list(_ALLOWED_UNRESOLVED),
        "artifact_count": len(source["artifacts"]),
        "c0_sentinel_count": _EXPECTED_PRE_PROVIDER_SENTINELS,
        "fixed_corpora": list(FIXED_CORPORA),
        "input_file_sha256s": {name: binding.sha256 for name, binding in request.inputs},
        "input_custody": {
            "capture_set_sha256": captured.capture_set_sha256,
            "contract": INPUT_CUSTODY_CONTRACT,
            "noncooperating_same_uid_mutation_excluded": True,
            "producer_parent_and_file_leases_held_through_publication": True,
        },
        "input_paths": {name: str(binding.path) for name, binding in request.inputs},
        "outcome_payloads_opened": False,
        "request_file_sha256": request.file_sha256,
        "request_path": str(request_path),
        "schema_version": COMPOSITION_RECEIPT_SCHEMA,
        "source_file_sha256": _sha256(source_bytes),
        "source_filename": SOURCE_FILENAME,
        "source_semantic_sha256": manifest_sha256(source),
    }


def _directory_control_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _assert_path_names_directory(
    path: Path,
    descriptor: int,
    *,
    label: str,
    expected_control_identity: tuple[int, int, int, int, int] | None = None,
) -> None:
    reopened: int | None = None
    try:
        opened = os.fstat(descriptor)
        reopened = _open_directory_chain(path, label=f"{label} identity readback")
        named = os.fstat(reopened)
        if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            named.st_dev,
            named.st_ino,
        ):
            raise CandidateSourceComposerError(f"{label} path changed after secure open")
        if expected_control_identity is not None and (
            _directory_control_identity(opened) != expected_control_identity
            or _directory_control_identity(named) != expected_control_identity
        ):
            raise CandidateSourceComposerError(
                f"{label} identity, ownership, or mode changed after secure open"
            )
    except CandidateSourceComposerError:
        raise
    except OSError as exc:
        raise CandidateSourceComposerError(f"cannot verify {label}: {exc}") from exc
    finally:
        if reopened is not None:
            os.close(reopened)


def _open_private_parent(path: Path) -> int:
    descriptor = _open_directory_chain(path, label="output parent")
    try:
        _assert_path_names_directory(path, descriptor, label="output parent")
        opened = os.fstat(descriptor)
        if stat.S_IMODE(opened.st_mode) & 0o022 or (
            hasattr(os, "geteuid") and opened.st_uid != os.geteuid()
        ):
            raise CandidateSourceComposerError(
                "output parent must be one owner-controlled real directory"
            )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _write_private_at(parent_descriptor: int, name: str, encoded: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
    try:
        os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise CandidateSourceComposerError(f"private output {name} has unsafe metadata")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise CandidateSourceComposerError(f"cannot complete private write {name}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_inode_at(parent_descriptor: int, name: str, *, label: str) -> tuple[int, int]:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise CandidateSourceComposerError(f"cannot inspect {label}: {exc}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise CandidateSourceComposerError(f"{label} is not one private directory")
    return metadata.st_dev, metadata.st_ino


def _assert_named_directory_matches_descriptor(
    parent_descriptor: int,
    name: str,
    retained_descriptor: int,
    *,
    expected_inode: tuple[int, int],
    label: str,
) -> None:
    """Bind a directory name to a retained descriptor under the parent lease."""

    named_descriptor: int | None = None
    try:
        named_descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        named = os.fstat(named_descriptor)
        retained = os.fstat(retained_descriptor)
        if (
            (named.st_dev, named.st_ino) != expected_inode
            or (retained.st_dev, retained.st_ino) != expected_inode
            or _directory_control_identity(named) != _directory_control_identity(retained)
        ):
            raise CandidateSourceComposerError(f"{label} identity differs")
    except CandidateSourceComposerError:
        raise
    except OSError as exc:
        raise CandidateSourceComposerError(f"cannot bind {label}: {exc}") from exc
    finally:
        if named_descriptor is not None:
            os.close(named_descriptor)


def _entry_identity_at(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
) -> tuple[int, int, int] | None:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CandidateSourceComposerError(f"cannot inspect {label}: {exc}") from exc
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


def _raw_rename_noreplace(
    parent_descriptor: int,
    source: str,
    destination: str,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise CandidateSourceComposerError("exclusive rename is unavailable on macOS")
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
            os.fsencode(source),
            parent_descriptor,
            os.fsencode(destination),
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise CandidateSourceComposerError("exclusive rename is unavailable on Linux")
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
            os.fsencode(source),
            parent_descriptor,
            os.fsencode(destination),
            0x00000001,
        )
    else:
        raise CandidateSourceComposerError(f"exclusive rename is unsupported on {sys.platform!r}")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise CandidateSourceComposerError("composition output already exists")
        raise CandidateSourceComposerError(
            f"cannot publish composition output: {os.strerror(error_number)}"
        )


def _restore_substituted_publication(
    parent_descriptor: int,
    *,
    source: str,
    destination: str,
    published_identity: tuple[int, int, int],
) -> bool:
    """Move a foreign source-name substitution back without replacing another entry."""

    try:
        if (
            _entry_identity_at(
                parent_descriptor,
                destination,
                label="substituted composition output",
            )
            != published_identity
            or _entry_identity_at(
                parent_descriptor,
                source,
                label="composition staging restoration name",
            )
            is not None
        ):
            return False
        _raw_rename_noreplace(parent_descriptor, destination, source)
        if (
            _entry_identity_at(
                parent_descriptor,
                destination,
                label="restored composition destination",
            )
            is not None
            or _entry_identity_at(
                parent_descriptor,
                source,
                label="restored substituted staging entry",
            )
            != published_identity
        ):
            return False
        os.fsync(parent_descriptor)
        return (
            _entry_identity_at(
                parent_descriptor,
                destination,
                label="durable restored composition destination",
            )
            is None
            and _entry_identity_at(
                parent_descriptor,
                source,
                label="durable restored substituted staging entry",
            )
            == published_identity
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _rename_noreplace(
    parent_descriptor: int,
    source: str,
    destination: str,
    *,
    expected_inode: tuple[int, int],
) -> None:
    if (
        _directory_inode_at(
            parent_descriptor,
            source,
            label="composition staging directory",
        )
        != expected_inode
    ):
        raise CandidateSourceComposerError("composition staging directory was substituted")
    _raw_rename_noreplace(parent_descriptor, source, destination)
    published_identity = _entry_identity_at(
        parent_descriptor,
        destination,
        label="composition output directory",
    )
    if (
        published_identity is not None
        and published_identity[:2] == expected_inode
        and stat.S_ISDIR(published_identity[2])
    ):
        return
    if published_identity is not None and _restore_substituted_publication(
        parent_descriptor,
        source=source,
        destination=destination,
        published_identity=published_identity,
    ):
        raise CandidateSourceComposerError(
            "composition staging source was substituted during rename; "
            "the foreign entry was restored"
        )
    raise CandidateSourcePublicationIndeterminateError(
        "exclusive rename returned but the published destination cannot be attributed or restored"
    )


def _read_package_at(
    directory_descriptor: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _acquire_exclusive_lease(
        directory_descriptor,
        label="composition package directory",
    )
    directory_metadata = os.fstat(directory_descriptor)
    if (
        not stat.S_ISDIR(directory_metadata.st_mode)
        or stat.S_IMODE(directory_metadata.st_mode) != 0o700
        or (hasattr(os, "geteuid") and directory_metadata.st_uid != os.geteuid())
    ):
        raise CandidateSourceComposerError(
            "composition package directory mode or ownership differs"
        )
    expected = {SOURCE_FILENAME, COMPOSITION_RECEIPT_FILENAME}
    try:
        observed = set(os.listdir(directory_descriptor))
    except OSError as exc:
        raise CandidateSourceComposerError(f"cannot list composition package: {exc}") from exc
    if observed != expected:
        raise CandidateSourceComposerError("composition package membership differs")
    retained: dict[str, _RetainedFileRead] = {}
    first_close_failure: BaseException | None = None
    try:
        for name in sorted(expected):
            retained[name] = _open_owned_file_at(
                directory_descriptor,
                name,
                label=f"composition package {name}",
                required_mode=0o600,
                exclusive_lease=True,
            )
        encoded_by_name = {name: retained[name].read() for name in sorted(expected)}
        # Close the two-file scan as one unit. A mutation after either member's
        # read changes its retained descriptor signature or its named identity.
        for name in sorted(expected):
            retained[name].assert_current()
        if set(os.listdir(directory_descriptor)) != expected:
            raise CandidateSourceComposerError(
                "composition package membership changed during its joint read"
            )
        values: dict[str, dict[str, Any]] = {}
        for name in sorted(expected):
            value = _decode_json(
                encoded_by_name[name],
                label=f"composition package {name}",
            )
            if not isinstance(value, dict):
                raise CandidateSourceComposerError(f"composition package {name} is not an object")
            values[name] = value
        return values[SOURCE_FILENAME], values[COMPOSITION_RECEIPT_FILENAME]
    finally:
        for name in reversed(sorted(retained)):
            try:
                retained[name].close()
            except BaseException as exc:
                if first_close_failure is None:
                    first_close_failure = exc
        if first_close_failure is not None:
            raise first_close_failure


def _read_published_package(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = _open_private_parent(directory)
    try:
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
            raise CandidateSourceComposerError("composition package is not one private directory")
        return _read_package_at(descriptor)
    finally:
        os.close(descriptor)


def _directory_descriptor_is_unlinked(descriptor: int) -> bool:
    """Prove an open directory no longer has a filesystem name."""

    try:
        opened = os.fstat(descriptor)
        if opened.st_nlink == 0:
            return True
        if sys.platform == "darwin":
            raw_path = fcntl.fcntl(descriptor, 50, b"\0" * 1024)
            path_bytes = raw_path.split(b"\0", 1)[0]
            if not path_bytes:
                return False
            path = Path(os.fsdecode(path_bytes))
        elif sys.platform.startswith("linux"):
            path_text = os.readlink(f"/proc/self/fd/{descriptor}")
            if path_text.endswith(" (deleted)"):
                return True
            path = Path(path_text)
        else:
            return False
        try:
            named = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            return True
        return (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
    except (OSError, ValueError):
        return False


def _remove_staging_directory(
    parent_descriptor: int,
    stage_descriptor: int | None,
    expected_inode: tuple[int, int] | None,
    *,
    preferred_name: str,
    _rescan_budget: int = 2,
    _foreign_inodes: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    """Quarantine and remove only the retained staging inode.

    The parent descriptor is held under the operator's exclusive cooperative
    lease. A candidate name is first moved to a fresh quarantine name and then
    identified. If a name substitution moved a foreign inode, that inode is
    restored before scanning continues; it is never unlinked or removed.
    """

    descriptor_matches = False
    if stage_descriptor is not None and expected_inode is not None:
        try:
            metadata = os.fstat(stage_descriptor)
            descriptor_matches = (metadata.st_dev, metadata.st_ino) == expected_inode
        except OSError:
            return False
    if expected_inode is None:
        return False
    try:
        names = os.listdir(parent_descriptor)
    except OSError:
        return False
    ordered_names = [
        *([preferred_name] if preferred_name in names else []),
        *(name for name in names if name != preferred_name),
    ]
    for name in ordered_names:
        quarantine_name = f".{preferred_name.lstrip('.')}.quarantine-{secrets.token_hex(16)}"
        quarantine_descriptor: int | None = None
        try:
            observed = _entry_identity_at(
                parent_descriptor,
                name,
                label="composition cleanup candidate",
            )
            if observed is None or not stat.S_ISDIR(observed[2]):
                continue
            if observed in _foreign_inodes:
                continue
            _raw_rename_noreplace(parent_descriptor, name, quarantine_name)
            quarantined = _entry_identity_at(
                parent_descriptor,
                quarantine_name,
                label="quarantined composition staging directory",
            )
            if (
                quarantined is None
                or quarantined[:2] != expected_inode
                or not stat.S_ISDIR(quarantined[2])
            ):
                if (
                    quarantined is None
                    or _entry_identity_at(
                        parent_descriptor,
                        name,
                        label="foreign staging restoration destination",
                    )
                    is not None
                ):
                    return False
                _raw_rename_noreplace(parent_descriptor, quarantine_name, name)
                restored = _entry_identity_at(
                    parent_descriptor,
                    name,
                    label="restored foreign staging entry",
                )
                if restored != quarantined:
                    return False
                os.fsync(parent_descriptor)
                if _rescan_budget <= 0:
                    return False
                return _remove_staging_directory(
                    parent_descriptor,
                    stage_descriptor,
                    expected_inode,
                    preferred_name=preferred_name,
                    _rescan_budget=_rescan_budget - 1,
                    _foreign_inodes=_foreign_inodes | frozenset({quarantined}),
                )

            quarantine_descriptor = os.open(
                quarantine_name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            bound = os.fstat(quarantine_descriptor)
            if (bound.st_dev, bound.st_ino) != expected_inode:
                return False
            removal_descriptor = (
                stage_descriptor
                if descriptor_matches and stage_descriptor is not None
                else quarantine_descriptor
            )
            for member in (SOURCE_FILENAME, COMPOSITION_RECEIPT_FILENAME):
                try:
                    os.unlink(member, dir_fd=removal_descriptor)
                except FileNotFoundError:
                    continue
                except OSError:
                    return False
            if os.listdir(removal_descriptor):
                return False

            # No POSIX rmdir-by-descriptor primitive exists. Under the held
            # parent lease, cooperating writers cannot replace this name.
            reopened = os.open(
                quarantine_name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            try:
                final = os.fstat(reopened)
                if (final.st_dev, final.st_ino) != expected_inode:
                    return False
                os.rmdir(quarantine_name, dir_fd=parent_descriptor)
                return _directory_descriptor_is_unlinked(reopened)
            finally:
                os.close(reopened)
        except (OSError, RuntimeError, ValueError):
            return False
        finally:
            if quarantine_descriptor is not None:
                try:
                    os.close(quarantine_descriptor)
                except OSError:
                    pass
    if descriptor_matches:
        return _directory_descriptor_is_unlinked(stage_descriptor)
    return False


def _rollback_published_directory(
    parent_descriptor: int,
    *,
    destination_name: str,
    parent_path: Path,
    parent_control_identity: tuple[int, int, int, int, int],
    stage_descriptor: int,
    expected_inode: tuple[int, int],
) -> bool:
    """Remove a published directory only while its captured provenance is exact."""

    rollback_name = f".{destination_name}.rollback-{secrets.token_hex(16)}"
    try:
        published = _entry_identity_at(
            parent_descriptor,
            destination_name,
            label="composition output rollback source",
        )
        if published is None or published[:2] != expected_inode or not stat.S_ISDIR(published[2]):
            return False
        _raw_rename_noreplace(parent_descriptor, destination_name, rollback_name)
        rolled_back = _entry_identity_at(
            parent_descriptor,
            rollback_name,
            label="rolled-back composition directory",
        )
        if (
            _entry_identity_at(
                parent_descriptor,
                destination_name,
                label="rolled-back composition destination",
            )
            is not None
            or rolled_back is None
            or rolled_back[:2] != expected_inode
            or not stat.S_ISDIR(rolled_back[2])
        ):
            return False
        if not _remove_staging_directory(
            parent_descriptor,
            stage_descriptor,
            expected_inode,
            preferred_name=rollback_name,
        ):
            return False
        os.fsync(parent_descriptor)
        _assert_path_names_directory(
            parent_path,
            parent_descriptor,
            label="output parent",
            expected_control_identity=parent_control_identity,
        )
        return (
            _entry_identity_at(
                parent_descriptor,
                destination_name,
                label="composition destination after rollback",
            )
            is None
        )
    except (OSError, RuntimeError, ValueError):
        return False


def compose_from_request(
    *,
    request_path: str | Path,
    request_sha256: str,
    output_directory: str | Path,
) -> Mapping[str, object]:
    """Compose and exclusively publish one source manifest and receipt."""

    request_file = _canonical_absolute_path(str(request_path), label="request_path")
    destination = _canonical_absolute_path(str(output_directory), label="output_directory")
    if os.path.lexists(destination):
        raise CandidateSourceComposerError("composition output already exists")
    captured = _capture_input_set(request_file, request_sha256)
    request = captured.request
    if any(
        destination == binding.path
        or destination in binding.path.parents
        or binding.path in destination.parents
        for _, binding in request.inputs
    ):
        captured.close()
        raise CandidateSourceComposerError("composition output overlaps an admitted input")
    try:
        source = _derive_candidate_source(request, captured)
        source_bytes = _canonical_bytes(source)
        receipt = _composition_receipt(request_file, request, source, captured)
        receipt_bytes = _canonical_bytes(receipt)
    except BaseException:
        captured.close()
        raise

    parent_descriptor: int | None = None
    parent_control_identity: tuple[int, int, int, int, int] | None = None
    work_name = f".{destination.name}.staging-{secrets.token_hex(16)}"
    stage_descriptor: int | None = None
    stage_created = False
    expected_inode: tuple[int, int] | None = None
    published = False
    guard = _PublicationSignalGuard()
    try:
        guard.__enter__()
    except BaseException:
        captured.close()
        raise
    try:
        open_leased_parent = getattr(captured, "open_leased_parent", None)
        if open_leased_parent is None:
            parent_descriptor = _open_private_parent(destination.parent)
            _acquire_exclusive_lease(
                parent_descriptor,
                label="composition output parent",
            )
        else:
            parent_descriptor = open_leased_parent(destination.parent)
        parent_control_identity = _directory_control_identity(os.fstat(parent_descriptor))
        os.mkdir(work_name, 0o700, dir_fd=parent_descriptor)
        stage_created = True
        created_metadata = os.stat(
            work_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        expected_inode = (created_metadata.st_dev, created_metadata.st_ino)
        os.chmod(
            work_name,
            0o700,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        stage_descriptor = os.open(
            work_name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        os.fchmod(stage_descriptor, 0o700)
        stage_metadata = os.fstat(stage_descriptor)
        if (stage_metadata.st_dev, stage_metadata.st_ino) != expected_inode:
            raise CandidateSourceComposerError("composition staging directory was substituted")
        if (
            _directory_inode_at(
                parent_descriptor,
                work_name,
                label="composition staging directory",
            )
            != expected_inode
        ):
            raise CandidateSourceComposerError("composition staging directory was substituted")
        _write_private_at(stage_descriptor, SOURCE_FILENAME, source_bytes)
        _write_private_at(stage_descriptor, COMPOSITION_RECEIPT_FILENAME, receipt_bytes)
        os.fsync(stage_descriptor)
        observed_source, observed_receipt = _read_package_at(stage_descriptor)
        if observed_source != source or observed_receipt != receipt:
            raise CandidateSourceComposerError("composition staging package differs before publish")

        # Derive twice from the same captured bytes. The request, all sixteen files,
        # and their parents remain descriptor-bound under exclusive cooperative leases.
        revalidated_source = _derive_candidate_source(request, captured)
        revalidated_receipt = _composition_receipt(
            request_file,
            request,
            revalidated_source,
            captured,
        )
        if revalidated_source != source or revalidated_receipt != receipt:
            raise CandidateSourceComposerError(
                "captured producer evidence changed during deterministic reproduction"
            )
        captured.assert_current()
        _assert_path_names_directory(
            destination.parent,
            parent_descriptor,
            label="output parent",
            expected_control_identity=parent_control_identity,
        )
        _rename_noreplace(
            parent_descriptor,
            work_name,
            destination.name,
            expected_inode=expected_inode,
        )
        published = True
        os.fsync(parent_descriptor)
        _assert_path_names_directory(
            destination.parent,
            parent_descriptor,
            label="output parent",
            expected_control_identity=parent_control_identity,
        )
        if (
            parent_control_identity is None
        ):  # pragma: no cover - established before any staged write
            raise CandidateSourceComposerError("composition parent identity is absent")
        _assert_named_directory_matches_descriptor(
            parent_descriptor,
            destination.name,
            stage_descriptor,
            expected_inode=expected_inode,
            label="composition output directory",
        )
        observed_source, observed_receipt = _read_package_at(stage_descriptor)
        if observed_source != source or observed_receipt != receipt:
            raise CandidateSourceComposerError("composition package changed during readback")
        captured.assert_current()
        _assert_path_names_directory(
            destination.parent,
            parent_descriptor,
            label="output parent",
            expected_control_identity=parent_control_identity,
        )
        _assert_named_directory_matches_descriptor(
            parent_descriptor,
            destination.name,
            stage_descriptor,
            expected_inode=expected_inode,
            label="composition output directory final readback",
        )

        # Descriptor closure remains inside the transaction. A failure or injected
        # BaseException here enters the same rollback/indeterminate state machine.
        os.close(stage_descriptor)
        stage_descriptor = None
        captured.close()
        os.close(parent_descriptor)
        parent_descriptor = None
    except BaseException as exc:
        if published and parent_descriptor is not None:
            if (
                stage_descriptor is not None
                and expected_inode is not None
                and parent_control_identity is not None
                and _rollback_published_directory(
                    parent_descriptor,
                    destination_name=destination.name,
                    parent_path=destination.parent,
                    parent_control_identity=parent_control_identity,
                    stage_descriptor=stage_descriptor,
                    expected_inode=expected_inode,
                )
            ):
                if isinstance(exc, CandidateSourceInterruptedError):
                    raise
                raise CandidateSourceComposerError(
                    f"publication failed after rename and was rolled back: {exc}"
                ) from exc
            raise CandidateSourcePublicationIndeterminateError(
                f"publication failed after rename and a clean rollback cannot be proved: {exc}"
            ) from exc
        if stage_created and parent_descriptor is not None:
            cleaned = _remove_staging_directory(
                parent_descriptor,
                stage_descriptor,
                expected_inode,
                preferred_name=work_name,
            )
            if cleaned:
                try:
                    os.fsync(parent_descriptor)
                    _assert_path_names_directory(
                        destination.parent,
                        parent_descriptor,
                        label="output parent",
                    )
                except (OSError, RuntimeError, ValueError):
                    cleaned = False
            if not cleaned and not isinstance(
                exc,
                CandidateSourcePublicationIndeterminateError,
            ):
                raise CandidateSourcePublicationIndeterminateError(
                    "pre-publication staging cleanup cannot be proved"
                ) from exc
        raise
    finally:
        if stage_descriptor is not None:
            try:
                os.close(stage_descriptor)
            except BaseException:
                pass
        try:
            captured.close()
        except BaseException:
            pass
        if parent_descriptor is not None:
            try:
                os.close(parent_descriptor)
            except BaseException:
                pass
        guard.__exit__(*sys.exc_info())

    return receipt


def verify_composed_package(directory: str | Path) -> Mapping[str, object]:
    """Reopen every receipt-bound input and reproduce a published source in memory."""

    root = _canonical_absolute_path(str(directory), label="composition package directory")
    root_descriptor = _open_private_parent(root)
    _acquire_exclusive_lease(root_descriptor, label="composition package")
    root_control_identity = _directory_control_identity(os.fstat(root_descriptor))
    captured: _CapturedInputSet | None = None
    try:
        source, receipt_value = _read_package_at(root_descriptor)
        receipt = _closed(
            receipt_value,
            frozenset(
                {
                    "allowed_unresolved_paths",
                    "artifact_count",
                    "c0_sentinel_count",
                    "fixed_corpora",
                    "input_custody",
                    "input_file_sha256s",
                    "input_paths",
                    "outcome_payloads_opened",
                    "request_file_sha256",
                    "request_path",
                    "schema_version",
                    "source_file_sha256",
                    "source_filename",
                    "source_semantic_sha256",
                }
            ),
            label="composition receipt",
        )
        if receipt["schema_version"] != COMPOSITION_RECEIPT_SCHEMA:
            raise CandidateSourceComposerError("composition receipt schema differs")
        request_path = _canonical_absolute_path(
            receipt["request_path"], label="receipt request_path"
        )
        captured = _capture_input_set(
            request_path,
            _require_sha256(receipt["request_file_sha256"], label="receipt request digest"),
        )
        request = captured.request
        reproduced = _derive_candidate_source(request, captured)
        expected_receipt = _composition_receipt(
            request_path,
            request,
            reproduced,
            captured,
        )
        if source != reproduced or receipt != expected_receipt:
            raise CandidateSourceComposerError("composition package does not reproduce")
        final_source, final_receipt = _read_package_at(root_descriptor)
        if final_source != source or final_receipt != receipt:
            raise CandidateSourceComposerError(
                "composition package changed during authority reproduction"
            )
        captured.assert_current()
        _assert_path_names_directory(
            root,
            root_descriptor,
            label="composition package",
            expected_control_identity=root_control_identity,
        )
        return receipt
    finally:
        if captured is not None:
            captured.close()
        os.close(root_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="candidate-study-manifest-composer",
        allow_abbrev=False,
        description=(
            "Compose a pre-provider candidate study-manifest source from one exact wiring request."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compose = commands.add_parser("compose", allow_abbrev=False)
    compose.add_argument("--request", type=Path, required=True)
    compose.add_argument("--request-sha256", required=True)
    compose.add_argument("--output-directory", type=Path, required=True)
    verify = commands.add_parser("verify", allow_abbrev=False)
    verify.add_argument("--directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with _PublicationSignalGuard():
            if args.command == "compose":
                receipt = compose_from_request(
                    request_path=args.request,
                    request_sha256=args.request_sha256,
                    output_directory=args.output_directory,
                )
                print(_canonical_bytes(receipt).decode("ascii"), end="")
                return 0
            receipt = verify_composed_package(args.directory)
            print(
                _canonical_bytes(
                    {
                        "source_file_sha256": receipt["source_file_sha256"],
                        "status": "verified",
                    }
                ).decode("ascii"),
                end="",
            )
            return 0
    except CandidateSourceInterruptedError as exc:
        print(f"candidate source composition interrupted: {exc}", file=sys.stderr)
        return 128 + exc.signum
    except CandidateSourcePublicationIndeterminateError as exc:
        print(f"candidate source publication indeterminate: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"candidate source composition failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
