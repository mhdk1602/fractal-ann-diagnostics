"""Closed provider-phase driver selected by the frozen C1 phase plan.

The public command accepts only the five bindings named by the plan.  It never
accepts a scientific command, container argument, environment override, corpus
subset, image name, tool path, or retry flag.  Lower-level driver requests are
canonical records below the admitted phase-input root and bind the exact plan,
claim receipt, controls, portable runtime receipts, and output namespaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    write_exclusive_receipt_bytes,
)
from .custody import load_custody_seal_receipt, load_timelock_encryption_receipt
from .execution_claim import (
    ANALYSIS_PHASE,
    LABEL_RELEASE_PHASE,
    ONLINE_PHASE,
    PHASE_RUNTIME_CLAIM_RECEIPT_SCHEMA,
    PROVIDER_PHASE_PLAN_SCHEMA,
    RUNTIME_CLAIM_RECEIPT_SCHEMA,
    ExecutionClaimError,
    LiveExecuteJobReceipt,
    PhaseBeaconReceipt,
    PhaseHostToolReceipt,
    PhaseRuntimeClaimReceipt,
    ProviderPhase,
    ProviderPhasePlan,
    VerifiedPhaseClaimCapability,
    VerifiedRunClaimCapability,
    load_materialized_provider_phase_plan,
    load_provider_runner_bootstrap,
    loads_runtime_claim_receipt,
)
from .post_online_completion import (
    PostOnlineCompletionError,
    revalidate_post_online_completion_authority,
)
from .study import FIXED_CORPORA, PROVIDER_PHASE_COMMAND_IDS, load_study_manifest
from .suite_attempt import VerifiedProviderPredecessor
from .timelock_release import (
    TIMELOCK_DECRYPTION_RECEIPT_FILENAME,
    TIMELOCK_RELEASE_INTENT_FILENAME,
    _run_pinned_tle_decrypt,
    label_release_staging_directory_name,
    load_timelock_decryption_receipt,
    release_timelock_label,
)

PROVIDER_DRIVER_REQUEST_SCHEMA = "fractal-provider-driver-request-v1"
PROVIDER_PHASE_RUNTIME_REQUEST_SCHEMA = "fractal-provider-phase-runtime-request-v2"
PROVIDER_DRIVER_OUTPUT_SCHEMA = "fractal-provider-driver-output-v2"
PROVIDER_PHASE_EXECUTION_RECEIPT_SCHEMA = "fractal-provider-phase-execution-v3"
LABEL_RELEASE_DRIVER_CONTROL_SCHEMA = "fractal-provider-label-release-driver-v1"
ONLINE_SEALED_LAUNCH_DRIVER_CONTROL_SCHEMA = "fractal-provider-online-sealed-launch-driver-v1"
ANALYSIS_RUNTIME_CLAIM_BUNDLE_SCHEMA = "fractal-analysis-runtime-claim-bundle-v1"
LABEL_RELEASE_AUTHORITY_JOURNAL_SCHEMA = "fractal-label-release-authority-journal-v1"

PROVIDER_RUNTIME_REQUEST_FILENAME = "provider-runtime-request.json"
PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME = "provider-phase-execution-receipt.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MAX_CONTROL_BYTES = 8 * 1024 * 1024
_MAX_DOCKER_CONTROL_OUTPUT_BYTES = 64 * 1024
_DOCKER_IMAGE_PULL_TIMEOUT_SECONDS = 600
_DOCKER_CONTAINER_CLEANUP_TIMEOUT_SECONDS = 30
_ALL_FIVE = "all-five"
_DRIVER_IDS: Mapping[ProviderPhase, str] = {
    ONLINE_PHASE: "sealed-online-corpus-v1",
    LABEL_RELEASE_PHASE: "timelock-label-release-v1",
    ANALYSIS_PHASE: "confirmatory-analysis-v1",
}


class ProviderPhaseRuntimeError(ValueError):
    """A phase request differs from the C1-selected driver contract."""


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
        raise ProviderPhaseRuntimeError("provider runtime record is not canonical JSON") from exc


def _digest(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProviderPhaseRuntimeError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _git_commit(name: str, value: object) -> str:
    if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
        raise ProviderPhaseRuntimeError(f"{name} must be one lowercase Git commit")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderPhaseRuntimeError(f"{name} must be canonical non-empty text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ProviderPhaseRuntimeError(f"{name} contains a control character")
    return value


def _absolute_path(name: str, value: object) -> Path:
    path = Path(_text(name, value))
    if not path.is_absolute() or path.anchor != "/":
        raise ProviderPhaseRuntimeError(f"{name} must be an absolute POSIX path")
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ProviderPhaseRuntimeError(f"{name} contains an aliasing component")
    return path


def _below(name: str, value: object, root: Path) -> Path:
    path = _absolute_path(name, value)
    if path == root or not path.is_relative_to(root):
        raise ProviderPhaseRuntimeError(f"{name} must be a strict descendant of its root")
    return path


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProviderPhaseRuntimeError(f"{label} must be one JSON object")
    observed = set(value)
    if observed != fields:
        raise ProviderPhaseRuntimeError(
            f"{label} schema differs; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _strict_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ProviderPhaseRuntimeError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ProviderPhaseRuntimeError(f"{label} contains non-finite number {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderPhaseRuntimeError(f"cannot decode {label}") from exc
    if not isinstance(value, Mapping):
        raise ProviderPhaseRuntimeError(f"{label} must contain one object")
    return value


def _secure_file_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProviderPhaseRuntimeError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_CONTROL_BYTES
        ):
            raise ProviderPhaseRuntimeError(
                f"{label} must be one bounded singly linked regular file"
            )
        chunks: list[bytes] = []
        observed = 0
        while observed <= _MAX_CONTROL_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_CONTROL_BYTES + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ProviderPhaseRuntimeError(f"cannot read {label}") from exc
    finally:
        os.close(descriptor)
    if observed > _MAX_CONTROL_BYTES or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ProviderPhaseRuntimeError(f"{label} changed while read")
    return b"".join(chunks)


def _verified_file(path: Path, expected_sha256: str, *, label: str) -> bytes:
    encoded = _secure_file_bytes(path, label=label)
    if hashlib.sha256(encoded).hexdigest() != _digest(f"{label} SHA-256", expected_sha256):
        raise ProviderPhaseRuntimeError(f"{label} differs from the request digest")
    return encoded


def _controlled_directory_entries(path: Path, *, label: str) -> tuple[str, ...]:
    try:
        metadata = path.stat(follow_symlinks=False)
        resolved = path.resolve(strict=True)
        entries = tuple(
            sorted(
                (entry.name for entry in path.iterdir()),
                key=lambda value: value.encode("utf-8"),
            )
        )
    except OSError as exc:
        raise ProviderPhaseRuntimeError(f"cannot admit {label}") from exc
    if (
        resolved != path
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise ProviderPhaseRuntimeError(f"{label} is not a controlled directory")
    return entries


def admit_analysis_results_store(
    path: Path,
    *,
    manifest_sha256: str,
) -> tuple[str, ...]:
    """Admit only registered analysis-store members across one process restart."""

    from .confirmatory_input_operator import confirmatory_store_closure_filenames

    entries = _controlled_directory_entries(
        path,
        label="analysis results store",
    )
    allowed = set(confirmatory_store_closure_filenames(manifest_sha256))
    if not set(entries).issubset(allowed):
        raise ProviderPhaseRuntimeError(
            "analysis results store contains an unregistered restart member"
        )
    for name in entries:
        target = path / name
        try:
            metadata = target.stat(follow_symlinks=False)
        except OSError as exc:
            raise ProviderPhaseRuntimeError("cannot inspect analysis restart member") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or target.is_symlink()
            or metadata.st_nlink != 1
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise ProviderPhaseRuntimeError(
                "analysis restart member is not a controlled regular file"
            )
    return entries


def analysis_offline_package_root(plan: ProviderPhasePlan) -> Path:
    """Derive the restart-stable package beside the C1 phase-evidence root."""

    if not isinstance(plan, ProviderPhasePlan) or plan.phase != ANALYSIS_PHASE:
        raise ProviderPhaseRuntimeError("analysis package root requires the typed C1 analysis plan")
    evidence_root = Path(plan.phase_evidence_root(plan.suite_attempt_id))
    return evidence_root.with_name(f"{evidence_root.name}.offline-analysis-package")


@dataclass(frozen=True)
class LabelReleasePhaseRootAdmission:
    completed_corpora: tuple[str, ...]
    staged_corpus: str | None
    execution_receipt_present: bool


def label_release_authority_journal_name(corpus_id: str) -> str:
    if corpus_id not in FIXED_CORPORA:
        raise ProviderPhaseRuntimeError("label authority journal names another corpus")
    return f".{corpus_id}.label-release-authority.json"


def admit_label_release_phase_root(
    path: Path,
    *,
    create_if_absent: bool,
) -> LabelReleasePhaseRootAdmission:
    """Admit only a canonical prefix of five release transactions."""

    root = Path(path)
    if not os.path.lexists(root):
        if not create_if_absent:
            raise ProviderPhaseRuntimeError("label phase output root is absent")
        try:
            root.mkdir(mode=0o700, parents=False, exist_ok=False)
        except OSError as exc:
            raise ProviderPhaseRuntimeError("cannot create label phase output root") from exc
    entries = _controlled_directory_entries(root, label="label phase output root")
    ordered = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
    final = tuple(corpus_id for corpus_id in ordered if corpus_id in entries)
    stage_names = {
        corpus_id: label_release_staging_directory_name(corpus_id) for corpus_id in ordered
    }
    staged = [corpus_id for corpus_id, stage_name in stage_names.items() if stage_name in entries]
    journal_names = {
        corpus_id: label_release_authority_journal_name(corpus_id) for corpus_id in ordered
    }
    journals = [
        corpus_id for corpus_id, journal_name in journal_names.items() if journal_name in entries
    ]
    receipt_present = PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME in entries
    allowed = (
        set(final)
        | {stage_names[item] for item in staged}
        | {journal_names[item] for item in journals}
    )
    if receipt_present:
        allowed.add(PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME)
    if set(entries) != allowed or len(staged) > 1 or not set(journals).issubset(final):
        raise ProviderPhaseRuntimeError(
            "label phase output root contains an unexpected transaction"
        )
    expected_prefix = ordered[: len(final)]
    if final != expected_prefix:
        raise ProviderPhaseRuntimeError(
            "label phase output directories are not one canonical prefix"
        )
    if staged and (len(final) == len(ordered) or staged[0] != ordered[len(final)]):
        raise ProviderPhaseRuntimeError(
            "label release stage does not name the next canonical corpus"
        )
    if receipt_present and (final != ordered or staged):
        raise ProviderPhaseRuntimeError(
            "label phase receipt precedes the exact five-corpus closure"
        )
    exact_pair = tuple(
        sorted(
            (TIMELOCK_DECRYPTION_RECEIPT_FILENAME, "released-labels.json"),
            key=lambda value: value.encode("utf-8"),
        )
    )
    committed_pair = tuple(
        sorted(
            (*exact_pair, TIMELOCK_RELEASE_INTENT_FILENAME),
            key=lambda value: value.encode("utf-8"),
        )
    )
    for index, corpus_id in enumerate(final):
        observed_pair = _controlled_directory_entries(
            root / corpus_id,
            label=f"{corpus_id} label release directory",
        )
        has_committed_intent = observed_pair == committed_pair
        if observed_pair != exact_pair and not (
            has_committed_intent and index == len(final) - 1 and not staged and not receipt_present
        ):
            raise ProviderPhaseRuntimeError(
                f"{corpus_id} label release is not the exact output pair"
            )
        if not receipt_present and corpus_id not in journals and not has_committed_intent:
            raise ProviderPhaseRuntimeError(
                f"{corpus_id} label release lacks restart authority evidence"
            )
    for corpus_id in journals:
        _secure_file_bytes(
            root / journal_names[corpus_id],
            label=f"{corpus_id} label authority journal",
        )
    if staged:
        stage_entries = set(
            _controlled_directory_entries(
                root / stage_names[staged[0]],
                label=f"{staged[0]} label release stage",
            )
        )
        if not stage_entries.issubset(
            {
                TIMELOCK_DECRYPTION_RECEIPT_FILENAME,
                TIMELOCK_RELEASE_INTENT_FILENAME,
                "released-labels.json",
            }
        ):
            raise ProviderPhaseRuntimeError("label release stage contains unexpected output")
    if receipt_present:
        _secure_file_bytes(
            root / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME,
            label="provider phase execution receipt",
        )
    return LabelReleasePhaseRootAdmission(
        completed_corpora=final,
        staged_corpus=(None if not staged else staged[0]),
        execution_receipt_present=receipt_present,
    )


@dataclass(frozen=True)
class ProviderDriverRequest:
    corpus_id: str
    driver_id: str
    control_path: str
    control_file_sha256: str
    runtime_claim_receipt_path: str
    runtime_claim_receipt_file_sha256: str
    output_root: str
    schema_version: str = PROVIDER_DRIVER_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        _text("corpus_id", self.corpus_id)
        _text("driver_id", self.driver_id)
        _absolute_path("control_path", self.control_path)
        _digest("control_file_sha256", self.control_file_sha256)
        _absolute_path("runtime_claim_receipt_path", self.runtime_claim_receipt_path)
        _digest(
            "runtime_claim_receipt_file_sha256",
            self.runtime_claim_receipt_file_sha256,
        )
        _absolute_path("output_root", self.output_root)
        if self.schema_version != PROVIDER_DRIVER_REQUEST_SCHEMA:
            raise ProviderPhaseRuntimeError("provider driver request schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> ProviderDriverRequest:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="provider driver request",
            )
        )


AnalysisClaimSupplier = Callable[
    [], tuple[VerifiedProviderPredecessor, VerifiedPhaseClaimCapability]
]


@dataclass(frozen=True)
class FreshOnlineClaimAuthority:
    capability: VerifiedRunClaimCapability
    claim_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.capability, VerifiedRunClaimCapability):
            raise ProviderPhaseRuntimeError("fresh online authority is untyped")
        if type(self.claim_bytes) is not bytes:
            raise ProviderPhaseRuntimeError("fresh online claim bytes are untyped")


@dataclass(frozen=True)
class FreshLabelClaimAuthority:
    capability: VerifiedPhaseClaimCapability
    predecessor: VerifiedProviderPredecessor
    claim_bytes: bytes
    admission_marker_path: str
    admission_marker_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.capability, VerifiedPhaseClaimCapability):
            raise ProviderPhaseRuntimeError("fresh label authority is untyped")
        if not isinstance(self.predecessor, VerifiedProviderPredecessor):
            raise ProviderPhaseRuntimeError("fresh label predecessor is untyped")
        if (
            self.predecessor.state.state != "LABEL_RELEASE_CLAIMED"
            or self.predecessor.state.record_sha256 != self.capability.phase_claim_state_sha256
            or self.predecessor.ledger_commit != self.capability.phase_claim_ledger_commit
        ):
            raise ProviderPhaseRuntimeError(
                "fresh label capability and predecessor name another claim"
            )
        if type(self.claim_bytes) is not bytes:
            raise ProviderPhaseRuntimeError("fresh label claim bytes are untyped")
        _absolute_path("admission_marker_path", self.admission_marker_path)
        _digest("admission_marker_sha256", self.admission_marker_sha256)


def _kill_control_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        process.kill()


def _run_bounded_docker_control(
    executable: Path,
    arguments: tuple[str, ...],
    *,
    timeout_seconds: int,
    max_output_bytes: int,
) -> bytes:
    if (
        type(timeout_seconds) is not int
        or timeout_seconds < 1
        or timeout_seconds > _DOCKER_IMAGE_PULL_TIMEOUT_SECONDS
        or type(max_output_bytes) is not int
        or max_output_bytes < 1
        or max_output_bytes > _MAX_DOCKER_CONTROL_OUTPUT_BYTES
    ):
        raise ProviderPhaseRuntimeError("Docker control bounds are invalid")
    try:
        process = subprocess.Popen(
            [str(executable), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            start_new_session=True,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except OSError as exc:
        raise ProviderPhaseRuntimeError("cannot start the pinned Docker client") from exc
    if process.stdout is None or process.stderr is None:
        _kill_control_process(process)
        process.wait()
        raise ProviderPhaseRuntimeError("Docker control command lacks isolated output streams")

    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    deadline = time.monotonic() + timeout_seconds
    streams = (process.stdout, process.stderr)
    try:
        for stream in streams:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, stream is process.stderr)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderPhaseRuntimeError(
                    f"Docker image preparation exceeded {timeout_seconds} seconds"
                )
            events = selector.select(min(remaining, 0.25))
            readable = [key for key, _ in events]
            if not readable and process.poll() is not None:
                # kqueue can omit the final pipe event after a short-lived
                # child exits. Drain every still-registered descriptor once
                # so EOF or bounded tail bytes close the loop immediately.
                readable = list(selector.get_map().values())
            for key in readable:
                stream = key.fileobj
                try:
                    chunk = os.read(stream.fileno(), 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                target = stderr if key.data else stdout
                if len(stdout) + len(stderr) + len(chunk) > max_output_bytes:
                    raise ProviderPhaseRuntimeError(
                        "Docker image preparation exceeded its output bound"
                    )
                target.extend(chunk)
        try:
            return_code = process.wait(timeout=max(deadline - time.monotonic(), 0.01))
        except subprocess.TimeoutExpired as exc:
            raise ProviderPhaseRuntimeError(
                f"Docker image preparation exceeded {timeout_seconds} seconds"
            ) from exc
    except BaseException:
        _kill_control_process(process)
        process.wait()
        raise
    finally:
        for stream in streams:
            if not stream.closed:
                try:
                    selector.unregister(stream)
                except (KeyError, ValueError):
                    pass
                stream.close()
        selector.close()
    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise ProviderPhaseRuntimeError(
            "pinned Docker image preparation failed"
            f" with exit {return_code}: {message or 'no stderr'}"
        )
    return bytes(stdout)


def _run_quiet_docker_status(
    executable: Path,
    arguments: tuple[str, ...],
) -> int:
    """Run one bounded Docker lifecycle command without retaining output."""

    try:
        completed = subprocess.run(
            [str(executable), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            close_fds=True,
            start_new_session=True,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            timeout=_DOCKER_CONTAINER_CLEANUP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProviderPhaseRuntimeError("cannot execute bounded Docker container cleanup") from exc
    return completed.returncode


def _assert_docker_container_absent(
    executable: Path,
    *,
    config: Path,
    container_name: str,
) -> None:
    """Prove the one-shot release container is absent from the same daemon."""

    observed = _run_bounded_docker_control(
        executable,
        (
            "--config",
            str(config),
            "container",
            "ls",
            "--all",
            "--no-trunc",
            "--filter",
            f"name=^/{container_name}$",
            "--format={{.Names}}",
        ),
        timeout_seconds=_DOCKER_CONTAINER_CLEANUP_TIMEOUT_SECONDS,
        max_output_bytes=_MAX_DOCKER_CONTROL_OUTPUT_BYTES,
    )
    if observed:
        raise ProviderPhaseRuntimeError("Docker TLE container survived its one-shot execution")


def _force_remove_docker_container(
    executable: Path,
    *,
    config: Path,
    container_name: str,
) -> None:
    """Force-remove a possibly orphaned release container, then prove absence."""

    _run_quiet_docker_status(
        executable,
        (
            "--config",
            str(config),
            "container",
            "rm",
            "--force",
            "--volumes",
            container_name,
        ),
    )
    _assert_docker_container_absent(
        executable,
        config=config,
        container_name=container_name,
    )


@dataclass(frozen=True)
class DockerTleDecryptRunner:
    """Fixed Linux/ARM64 release-image adapter for the host-side release gate."""

    docker_executable: str
    docker_resolved_executable: str
    docker_executable_sha256: str
    index_image_reference: str
    platform_image_reference: str
    oci_index_digest: str
    oci_platform_manifest_digest: str
    runtime_platform: str
    tle_binary_sha256: str
    maximum_runtime_seconds: int

    def __post_init__(self) -> None:
        _absolute_path("docker_executable", self.docker_executable)
        _absolute_path("docker_resolved_executable", self.docker_resolved_executable)
        if self.docker_executable == self.docker_resolved_executable:
            raise ProviderPhaseRuntimeError(
                "Docker TLE client must bind its resolved symlink target"
            )
        _digest("docker_executable_sha256", self.docker_executable_sha256)
        image = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
        digest = re.compile(r"^sha256:[0-9a-f]{64}$")
        if (
            image.fullmatch(self.index_image_reference) is None
            or image.fullmatch(self.platform_image_reference) is None
            or digest.fullmatch(self.oci_index_digest) is None
            or digest.fullmatch(self.oci_platform_manifest_digest) is None
            or self.index_image_reference.rsplit("@", 1)[1] != self.oci_index_digest
            or self.platform_image_reference.rsplit("@", 1)[1] != self.oci_platform_manifest_digest
            or self.index_image_reference.rsplit("@", 1)[0]
            != self.platform_image_reference.rsplit("@", 1)[0]
        ):
            raise ProviderPhaseRuntimeError("Docker TLE image differs from its C1 digest pins")
        if self.runtime_platform != "linux/arm64":
            raise ProviderPhaseRuntimeError(
                "Docker TLE runner requires the C1 Linux/ARM64 platform"
            )
        _digest("tle_binary_sha256", self.tle_binary_sha256)
        if type(self.maximum_runtime_seconds) is not int or self.maximum_runtime_seconds < 1:
            raise ProviderPhaseRuntimeError("Docker TLE runtime ceiling is invalid")

    @classmethod
    def from_plan(cls, plan: ProviderPhasePlan) -> DockerTleDecryptRunner:
        if plan.phase != LABEL_RELEASE_PHASE or plan.tle_binary_sha256 is None:
            raise ProviderPhaseRuntimeError("Docker TLE runner requires the label-release plan")
        repository = plan.runtime_image.rsplit("@", 1)[0]
        return cls(
            docker_executable=plan.host_tools.docker_executable,
            docker_resolved_executable=plan.host_tools.docker_resolved_executable,
            docker_executable_sha256=plan.host_tools.docker_executable_sha256,
            index_image_reference=plan.runtime_image,
            platform_image_reference=f"{repository}@{plan.oci_platform_manifest_digest}",
            oci_index_digest=plan.oci_index_digest,
            oci_platform_manifest_digest=plan.oci_platform_manifest_digest,
            runtime_platform=plan.runtime_platform,
            tle_binary_sha256=plan.tle_binary_sha256,
            maximum_runtime_seconds=plan.maximum_runtime_seconds,
        )

    def _verify_docker_client(self) -> None:
        try:
            invocation = Path(self.docker_executable)
            if not stat.S_ISLNK(invocation.lstat().st_mode):
                raise ProviderPhaseRuntimeError(
                    "Docker TLE invocation path is no longer the C1 symlink"
                )
            resolved = invocation.resolve(strict=True)
            if resolved != Path(self.docker_resolved_executable):
                raise ProviderPhaseRuntimeError(
                    "Docker TLE invocation path resolves outside its C1 binding"
                )
            observed = digest_regular_file(
                resolved,
                label="Docker TLE client",
            )
        except OSError as exc:
            raise ProviderPhaseRuntimeError("cannot resolve the Docker TLE client") from exc
        except ArtifactIntegrityError as exc:
            raise ProviderPhaseRuntimeError("cannot revalidate the Docker TLE client") from exc
        if observed != self.docker_executable_sha256:
            raise ProviderPhaseRuntimeError("Docker TLE client differs from C1")

    def prepare(self) -> None:
        """Make the exact C1 platform manifest available without host credentials."""

        self._verify_docker_client()
        with tempfile.TemporaryDirectory(prefix="fractal-anonymous-docker-") as raw_config:
            config = Path(raw_config)
            config.chmod(0o700)
            _run_bounded_docker_control(
                Path(self.docker_resolved_executable),
                (
                    "--config",
                    str(config),
                    "pull",
                    "--quiet",
                    f"--platform={self.runtime_platform}",
                    self.platform_image_reference,
                ),
                timeout_seconds=_DOCKER_IMAGE_PULL_TIMEOUT_SECONDS,
                max_output_bytes=_MAX_DOCKER_CONTROL_OUTPUT_BYTES,
            )

    def __call__(
        self,
        binary: Path,
        arguments: tuple[str, ...],
        ciphertext: bytes,
        timeout_seconds: int,
        max_plaintext_bytes: int,
    ) -> bytes:
        self._verify_docker_client()
        try:
            binary_sha256 = digest_regular_file(binary, label="Docker TLE host binary pin")
        except ArtifactIntegrityError as exc:
            raise ProviderPhaseRuntimeError("cannot revalidate the Docker TLE binary pin") from exc
        if binary_sha256 != self.tle_binary_sha256:
            raise ProviderPhaseRuntimeError("Docker TLE binary argument differs from C1")
        if (
            len(arguments) != 3
            or arguments[0] != "--decrypt"
            or not arguments[1].startswith("--network=https://")
            or not re.fullmatch(r"--chain=[0-9a-f]{64}", arguments[2])
        ):
            raise ProviderPhaseRuntimeError("Docker TLE arguments differ from the release API")
        if (
            type(timeout_seconds) is not int
            or timeout_seconds < 1
            or timeout_seconds > min(60, self.maximum_runtime_seconds)
        ):
            raise ProviderPhaseRuntimeError("Docker TLE timeout exceeds its fixed bound")
        if (
            type(max_plaintext_bytes) is not int
            or max_plaintext_bytes < 1
            or max_plaintext_bytes > 1024 * 1024 * 1024
        ):
            raise ProviderPhaseRuntimeError("Docker TLE plaintext bound is invalid")
        with tempfile.TemporaryDirectory(prefix="fractal-anonymous-docker-") as raw_config:
            config = Path(raw_config)
            config.chmod(0o700)
            container_name = f"fractal-tle-{secrets.token_hex(16)}"
            create_arguments = (
                "--config",
                str(config),
                "container",
                "create",
                f"--name={container_name}",
                "--interactive",
                "--rm",
                "--pull=never",
                "--log-driver=none",
                "--network=bridge",
                f"--platform={self.runtime_platform}",
                "--user=65532:65532",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=64",
                "--memory=256m",
                "--cpus=1",
                "--read-only",
                "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m",
                "--entrypoint=/usr/local/bin/tle",
                self.platform_image_reference,
                *arguments,
            )
            executable = Path(self.docker_resolved_executable)
            try:
                created = _run_bounded_docker_control(
                    executable,
                    create_arguments,
                    timeout_seconds=_DOCKER_CONTAINER_CLEANUP_TIMEOUT_SECONDS,
                    max_output_bytes=_MAX_DOCKER_CONTROL_OUTPUT_BYTES,
                )
                if re.fullmatch(rb"[0-9a-f]{64}\n", created) is None:
                    raise ProviderPhaseRuntimeError(
                        "Docker TLE create returned an invalid container identity"
                    )
                container_id = created[:-1].decode("ascii")
                start_arguments = (
                    "--config",
                    str(config),
                    "container",
                    "start",
                    "--attach",
                    "--interactive",
                    container_id,
                )
                plaintext = _run_pinned_tle_decrypt(
                    executable,
                    start_arguments,
                    ciphertext,
                    timeout_seconds,
                    max_plaintext_bytes,
                )
            except BaseException:
                try:
                    _force_remove_docker_container(
                        executable,
                        config=config,
                        container_name=container_name,
                    )
                except Exception as cleanup_exc:
                    raise ProviderPhaseRuntimeError(
                        "Docker TLE failure left an unverifiable container"
                    ) from cleanup_exc
                raise
            _force_remove_docker_container(
                executable,
                config=config,
                container_name=container_name,
            )
            return plaintext


OnlineRunClaimSupplier = Callable[[ProviderDriverRequest], FreshOnlineClaimAuthority]
LabelPhaseClaimSupplier = Callable[[ProviderDriverRequest], FreshLabelClaimAuthority]


@dataclass(frozen=True)
class ProviderPhaseRuntimeRequest:
    phase: ProviderPhase
    activation_command_id: str
    suite_attempt_id: str
    provider_plan_path: str
    provider_plan_sha256: str
    provider_plan_file_sha256: str
    claim_receipt_path: str
    claim_receipt_file_sha256: str
    phase_host_tool_receipt_path: str
    phase_host_tool_receipt_sha256: str
    phase_host_tool_receipt_file_sha256: str
    phase_input_root: str
    phase_output_root: str
    drivers: tuple[ProviderDriverRequest, ...]
    schema_version: str = PROVIDER_PHASE_RUNTIME_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in _DRIVER_IDS:
            raise ProviderPhaseRuntimeError("provider runtime request has another phase")
        if self.activation_command_id != PROVIDER_PHASE_COMMAND_IDS[self.phase]:
            raise ProviderPhaseRuntimeError("provider runtime command differs from C1")
        _digest("suite_attempt_id", self.suite_attempt_id)
        _absolute_path("provider_plan_path", self.provider_plan_path)
        _digest("provider_plan_sha256", self.provider_plan_sha256)
        _digest("provider_plan_file_sha256", self.provider_plan_file_sha256)
        _absolute_path("claim_receipt_path", self.claim_receipt_path)
        _digest("claim_receipt_file_sha256", self.claim_receipt_file_sha256)
        _absolute_path(
            "phase_host_tool_receipt_path",
            self.phase_host_tool_receipt_path,
        )
        _digest(
            "phase_host_tool_receipt_sha256",
            self.phase_host_tool_receipt_sha256,
        )
        _digest(
            "phase_host_tool_receipt_file_sha256",
            self.phase_host_tool_receipt_file_sha256,
        )
        input_root = _absolute_path("phase_input_root", self.phase_input_root)
        output_root = _absolute_path("phase_output_root", self.phase_output_root)
        if (
            input_root == output_root
            or input_root in output_root.parents
            or output_root in input_root.parents
        ):
            raise ProviderPhaseRuntimeError("provider phase input and output roots overlap")
        rows = tuple(self.drivers)
        expected_ids = set(FIXED_CORPORA) if self.phase != ANALYSIS_PHASE else {_ALL_FIVE}
        if (
            len(rows) != len(expected_ids)
            or not all(isinstance(row, ProviderDriverRequest) for row in rows)
            or {row.corpus_id for row in rows} != expected_ids
            or [row.corpus_id for row in rows]
            != sorted(expected_ids, key=lambda value: value.encode("utf-8"))
        ):
            raise ProviderPhaseRuntimeError("provider driver rows do not cover the fixed phase")
        if any(row.driver_id != _DRIVER_IDS[self.phase] for row in rows):
            raise ProviderPhaseRuntimeError("provider driver ID differs from the phase")
        if len({row.control_path for row in rows}) != len(rows):
            raise ProviderPhaseRuntimeError("provider driver rows reuse one control file")
        if len({row.runtime_claim_receipt_path for row in rows}) != len(rows):
            raise ProviderPhaseRuntimeError("provider driver rows reuse one runtime receipt")
        if self.phase != ANALYSIS_PHASE:
            expected_output_roots = {str(output_root / row.corpus_id) for row in rows}
            if {row.output_root for row in rows} != expected_output_roots:
                raise ProviderPhaseRuntimeError("provider driver output namespaces differ")
        for row in rows:
            _below("driver control_path", row.control_path, input_root)
            _below(
                "driver runtime_claim_receipt_path",
                row.runtime_claim_receipt_path,
                input_root,
            )
            if self.phase != ANALYSIS_PHASE:
                _below("driver output_root", row.output_root, output_root)
        if self.schema_version != PROVIDER_PHASE_RUNTIME_REQUEST_SCHEMA:
            raise ProviderPhaseRuntimeError("provider phase runtime request schema differs")
        object.__setattr__(self, "drivers", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name) for name in self.__dataclass_fields__ if name != "drivers"
            },
            "drivers": [row.to_dict() for row in self.drivers],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> ProviderPhaseRuntimeRequest:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="provider phase runtime request",
        )
        drivers = row["drivers"]
        if not isinstance(drivers, list):
            raise ProviderPhaseRuntimeError("provider phase driver requests must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "drivers"},
            drivers=tuple(ProviderDriverRequest.from_dict(item) for item in drivers),
        )


def load_provider_phase_runtime_request(
    path: str | Path,
) -> ProviderPhaseRuntimeRequest:
    candidate = Path(path)
    encoded = _secure_file_bytes(candidate, label="provider phase runtime request")
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ProviderPhaseRuntimeError("provider runtime request needs one terminal newline")
    request = ProviderPhaseRuntimeRequest.from_dict(
        _strict_object(encoded[:-1], label="provider phase runtime request")
    )
    if request.canonical_file_bytes() != encoded:
        raise ProviderPhaseRuntimeError("provider runtime request bytes are not canonical")
    expected_path = Path(request.phase_input_root) / PROVIDER_RUNTIME_REQUEST_FILENAME
    if candidate != expected_path:
        raise ProviderPhaseRuntimeError("provider runtime request is outside its fixed path")
    return request


def write_provider_phase_runtime_request(
    request: ProviderPhaseRuntimeRequest,
) -> Path:
    if not isinstance(request, ProviderPhaseRuntimeRequest):
        raise ProviderPhaseRuntimeError("runtime request must use the closed typed schema")
    path = Path(request.phase_input_root) / PROVIDER_RUNTIME_REQUEST_FILENAME
    try:
        write_exclusive_receipt_bytes(request.canonical_file_bytes(), path)
    except ArtifactIntegrityError as exc:
        raise ProviderPhaseRuntimeError("cannot write provider runtime request once") from exc
    return path


@dataclass(frozen=True)
class OnlineSealedLaunchDriverControl:
    preflight_contract_path: str
    preflight_receipt_path: str
    transition_receipt_path: str
    instantiation_receipt_path: str
    finalization_request_path: str
    finalization_receipt_path: str
    sealed_contract_path: str
    volume_receipt_path: str
    audit_root: str
    schema_version: str = ONLINE_SEALED_LAUNCH_DRIVER_CONTROL_SCHEMA

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "schema_version":
                _absolute_path(name, getattr(self, name))
        if self.schema_version != ONLINE_SEALED_LAUNCH_DRIVER_CONTROL_SCHEMA:
            raise ProviderPhaseRuntimeError("online sealed-launch control schema differs")

    def canonical_file_bytes(self) -> bytes:
        return (
            _canonical_bytes({name: getattr(self, name) for name in self.__dataclass_fields__})
            + b"\n"
        )

    @classmethod
    def from_bytes(cls, encoded: bytes) -> OnlineSealedLaunchDriverControl:
        if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
            raise ProviderPhaseRuntimeError("online control needs one terminal newline")
        control = cls(
            **_closed(
                _strict_object(encoded[:-1], label="online sealed-launch control"),
                frozenset(cls.__dataclass_fields__),
                label="online sealed-launch control",
            )
        )
        if encoded != control.canonical_file_bytes():
            raise ProviderPhaseRuntimeError("online control bytes are not canonical")
        return control


@dataclass(frozen=True)
class LabelReleaseDriverControl:
    manifest_path: str
    custody_seal_path: str
    encryption_receipt_path: str
    completion_receipt_path: str
    completion_anchor_record_path: str
    completion_anchor_receipt_path: str
    suite_namespace: str
    ciphertext_path: str
    tle_binary_path: str
    plaintext_output_path: str
    decryption_receipt_path: str
    schema_version: str = LABEL_RELEASE_DRIVER_CONTROL_SCHEMA

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "schema_version":
                _absolute_path(name, getattr(self, name))
        if self.schema_version != LABEL_RELEASE_DRIVER_CONTROL_SCHEMA:
            raise ProviderPhaseRuntimeError("label-release driver control schema differs")

    @classmethod
    def from_bytes(cls, encoded: bytes) -> LabelReleaseDriverControl:
        if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
            raise ProviderPhaseRuntimeError("label-release control needs one terminal newline")
        value = _strict_object(encoded[:-1], label="label-release driver control")
        control = cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="label-release driver control",
            )
        )
        expected = (
            _canonical_bytes(
                {name: getattr(control, name) for name in control.__dataclass_fields__}
            )
            + b"\n"
        )
        if encoded != expected:
            raise ProviderPhaseRuntimeError("label-release control bytes are not canonical")
        return control


@dataclass(frozen=True)
class AnalysisRuntimeClaimBundle:
    phase: Literal["analysis"]
    receipts: tuple[PhaseRuntimeClaimReceipt, ...]
    schema_version: str = ANALYSIS_RUNTIME_CLAIM_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.phase != ANALYSIS_PHASE:
            raise ProviderPhaseRuntimeError("analysis runtime bundle has another phase")
        rows = tuple(self.receipts)
        ordered = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
        if (
            len(rows) != len(ordered)
            or not all(isinstance(row, PhaseRuntimeClaimReceipt) for row in rows)
            or tuple(row.corpus_id for row in rows) != ordered
            or any(row.phase != ANALYSIS_PHASE for row in rows)
        ):
            raise ProviderPhaseRuntimeError(
                "analysis runtime bundle must contain the ordered five corpus receipts"
            )
        shared = (
            "manifest_sha256",
            "run_receipt_sha256",
            "c1_commit",
            "phase_claim_contract_sha256",
            "phase_claim_state_sha256",
            "phase_claim_ledger_commit",
            "provider_identity_sha256",
            "live_execute_job_receipt_sha256",
            "execute_job_id",
            "phase_input_aggregate_sha256",
            "phase_output_identity",
        )
        first = rows[0]
        if any(getattr(row, name) != getattr(first, name) for row in rows[1:] for name in shared):
            raise ProviderPhaseRuntimeError("analysis runtime receipts cross provider claims")
        if self.schema_version != ANALYSIS_RUNTIME_CLAIM_BUNDLE_SCHEMA:
            raise ProviderPhaseRuntimeError("analysis runtime bundle schema differs")
        object.__setattr__(self, "receipts", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "receipts": [row.to_dict() for row in self.receipts],
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_bytes(cls, encoded: bytes) -> AnalysisRuntimeClaimBundle:
        if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
            raise ProviderPhaseRuntimeError("analysis runtime bundle needs one terminal newline")
        row = _closed(
            _strict_object(encoded[:-1], label="analysis runtime claim bundle"),
            frozenset({"phase", "receipts", "schema_version"}),
            label="analysis runtime claim bundle",
        )
        receipts = row["receipts"]
        if not isinstance(receipts, list):
            raise ProviderPhaseRuntimeError("analysis runtime receipts must be an array")
        bundle = cls(
            phase=row["phase"],
            receipts=tuple(PhaseRuntimeClaimReceipt.from_dict(item) for item in receipts),
            schema_version=row["schema_version"],
        )
        if bundle.canonical_file_bytes() != encoded:
            raise ProviderPhaseRuntimeError("analysis runtime bundle bytes are not canonical")
        return bundle


def _load_phase_runtime_claim(
    encoded: bytes,
    *,
    phase: Literal["label-release", "analysis"],
    corpus_id: str,
) -> PhaseRuntimeClaimReceipt:
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ProviderPhaseRuntimeError("phase runtime receipt needs one terminal newline")
    receipt = PhaseRuntimeClaimReceipt.from_dict(
        _strict_object(encoded[:-1], label="phase runtime claim receipt")
    )
    if (
        receipt.canonical_file_bytes() != encoded
        or receipt.schema_version != PHASE_RUNTIME_CLAIM_RECEIPT_SCHEMA
        or receipt.phase != phase
        or (corpus_id != _ALL_FIVE and receipt.corpus_id != corpus_id)
    ):
        raise ProviderPhaseRuntimeError("phase runtime receipt differs from the driver request")
    return receipt


@dataclass(frozen=True)
class LabelReleaseOutputAuthority:
    """Action-specific authority retained for later label completion."""

    corpus_id: str
    post_online_completion_aggregate_file_sha256: str
    label_release_claim_state_sha256: str
    label_release_claim_ledger_commit: str
    label_release_phase_claim_contract_sha256: str
    label_release_phase_beacon_receipt_sha256: str
    label_release_live_execute_job_receipt_sha256: str
    label_release_provider_identity_sha256: str
    label_release_phase_beacon_receipt: PhaseBeaconReceipt
    label_release_live_execute_job_receipt: LiveExecuteJobReceipt

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ProviderPhaseRuntimeError("label-release output authority has another corpus")
        for name in (
            "post_online_completion_aggregate_file_sha256",
            "label_release_claim_state_sha256",
            "label_release_phase_claim_contract_sha256",
            "label_release_phase_beacon_receipt_sha256",
            "label_release_live_execute_job_receipt_sha256",
            "label_release_provider_identity_sha256",
        ):
            _digest(name, getattr(self, name))
        _git_commit(
            "label_release_claim_ledger_commit",
            self.label_release_claim_ledger_commit,
        )
        if (
            not isinstance(
                self.label_release_phase_beacon_receipt,
                PhaseBeaconReceipt,
            )
            or not isinstance(
                self.label_release_live_execute_job_receipt,
                LiveExecuteJobReceipt,
            )
            or self.label_release_phase_beacon_receipt.receipt_sha256
            != self.label_release_phase_beacon_receipt_sha256
            or self.label_release_live_execute_job_receipt.receipt_sha256
            != self.label_release_live_execute_job_receipt_sha256
            or self.label_release_phase_beacon_receipt.phase_claim_state_sha256
            != self.label_release_claim_state_sha256
            or self.label_release_phase_beacon_receipt.phase_claim_ledger_commit
            != self.label_release_claim_ledger_commit
            or self.label_release_phase_beacon_receipt.phase_claim_contract_sha256
            != self.label_release_phase_claim_contract_sha256
            or self.label_release_phase_beacon_receipt.provider_identity_sha256
            != self.label_release_provider_identity_sha256
            or self.label_release_live_execute_job_receipt.provider_identity_sha256
            != self.label_release_provider_identity_sha256
        ):
            raise ProviderPhaseRuntimeError(
                "label-release output authority evidence differs from its hashes"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "label_release_live_execute_job_receipt",
                    "label_release_phase_beacon_receipt",
                }
            },
            "label_release_live_execute_job_receipt": (
                self.label_release_live_execute_job_receipt.to_dict()
            ),
            "label_release_phase_beacon_receipt": (
                self.label_release_phase_beacon_receipt.to_dict()
            ),
        }

    @property
    def authority_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> LabelReleaseOutputAuthority:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="label-release output authority",
        )
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "label_release_live_execute_job_receipt",
                    "label_release_phase_beacon_receipt",
                }
            },
            label_release_live_execute_job_receipt=LiveExecuteJobReceipt(
                **_closed(
                    row["label_release_live_execute_job_receipt"],
                    frozenset(LiveExecuteJobReceipt.__dataclass_fields__),
                    label="label-release live execute-job receipt",
                )
            ),
            label_release_phase_beacon_receipt=PhaseBeaconReceipt.from_dict(
                row["label_release_phase_beacon_receipt"]
            ),
        )


def _label_authority_journal_bytes(
    authority: LabelReleaseOutputAuthority,
) -> bytes:
    return (
        _canonical_bytes(
            {
                "authority": authority.to_dict(),
                "authority_sha256": authority.authority_sha256,
                "schema_version": LABEL_RELEASE_AUTHORITY_JOURNAL_SCHEMA,
            }
        )
        + b"\n"
    )


def _load_label_authority_journal(path: Path) -> LabelReleaseOutputAuthority:
    encoded = _secure_file_bytes(path, label="label-release authority journal")
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ProviderPhaseRuntimeError(
            "label-release authority journal needs one terminal newline"
        )
    row = _closed(
        _strict_object(encoded[:-1], label="label-release authority journal"),
        frozenset({"authority", "authority_sha256", "schema_version"}),
        label="label-release authority journal",
    )
    if row["schema_version"] != LABEL_RELEASE_AUTHORITY_JOURNAL_SCHEMA:
        raise ProviderPhaseRuntimeError("label-release authority journal schema differs")
    authority = LabelReleaseOutputAuthority.from_dict(row["authority"])
    if row[
        "authority_sha256"
    ] != authority.authority_sha256 or encoded != _label_authority_journal_bytes(authority):
        raise ProviderPhaseRuntimeError(
            "label-release authority journal differs from its canonical evidence"
        )
    return authority


def _release_intent_action_evidence(
    path: Path,
) -> tuple[LiveExecuteJobReceipt, PhaseBeaconReceipt]:
    encoded = _secure_file_bytes(path, label="committed timelock release intent")
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ProviderPhaseRuntimeError(
            "committed timelock release intent needs one terminal newline"
        )
    row = _strict_object(
        encoded[:-1],
        label="committed timelock release intent",
    )
    try:
        live_row = row["label_release_live_execute_job_receipt"]
        beacon_row = row["label_release_phase_beacon_receipt"]
    except KeyError as exc:
        raise ProviderPhaseRuntimeError(
            "committed timelock release intent lacks action evidence"
        ) from exc
    live = LiveExecuteJobReceipt(
        **_closed(
            live_row,
            frozenset(LiveExecuteJobReceipt.__dataclass_fields__),
            label="committed label live execute-job receipt",
        )
    )
    beacon = PhaseBeaconReceipt.from_dict(beacon_row)
    if encoded != _canonical_bytes(row) + b"\n":
        raise ProviderPhaseRuntimeError("committed timelock release intent bytes are not canonical")
    return live, beacon


def _close_label_release_action_authority(
    *,
    row: ProviderDriverRequest,
    receipt: object,
    fresh: FreshLabelClaimAuthority,
    existing_phase_authority: LabelReleaseOutputAuthority | None,
) -> LabelReleaseOutputAuthority:
    phase_root = Path(row.output_root).parent
    intent_path = Path(row.output_root) / TIMELOCK_RELEASE_INTENT_FILENAME
    journal_path = phase_root / label_release_authority_journal_name(row.corpus_id)
    if os.path.lexists(intent_path):
        live, beacon = _release_intent_action_evidence(intent_path)
    elif os.path.lexists(journal_path):
        journal = _load_label_authority_journal(journal_path)
        live = journal.label_release_live_execute_job_receipt
        beacon = journal.label_release_phase_beacon_receipt
    elif existing_phase_authority is not None:
        live = existing_phase_authority.label_release_live_execute_job_receipt
        beacon = existing_phase_authority.label_release_phase_beacon_receipt
    else:
        raise ProviderPhaseRuntimeError(
            "committed label release lacks recoverable action authority"
        )
    current_beacon = fresh.capability.phase_beacon_receipt
    if (
        not isinstance(current_beacon, PhaseBeaconReceipt)
        or live.job_identity_sha256 != fresh.capability.live_execute_job_receipt.job_identity_sha256
        or beacon.beacon_identity_sha256 != current_beacon.beacon_identity_sha256
    ):
        raise ProviderPhaseRuntimeError(
            "persisted label action evidence differs from fresh authority"
        )
    authority = LabelReleaseOutputAuthority(
        corpus_id=row.corpus_id,
        post_online_completion_aggregate_file_sha256=(
            receipt.post_online_completion_aggregate_file_sha256
        ),
        label_release_claim_state_sha256=(receipt.label_release_claim_state_sha256),
        label_release_claim_ledger_commit=(receipt.label_release_claim_ledger_commit),
        label_release_phase_claim_contract_sha256=(
            receipt.label_release_phase_claim_contract_sha256
        ),
        label_release_phase_beacon_receipt_sha256=(
            receipt.label_release_phase_beacon_receipt_sha256
        ),
        label_release_live_execute_job_receipt_sha256=(
            receipt.label_release_live_execute_job_receipt_sha256
        ),
        label_release_provider_identity_sha256=(receipt.label_release_provider_identity_sha256),
        label_release_phase_beacon_receipt=beacon,
        label_release_live_execute_job_receipt=live,
    )
    if existing_phase_authority is not None and authority != existing_phase_authority:
        raise ProviderPhaseRuntimeError("existing phase receipt changes label action authority")
    journal_bytes = _label_authority_journal_bytes(authority)
    if os.path.lexists(journal_path):
        if (
            _secure_file_bytes(
                journal_path,
                label="label-release authority journal",
            )
            != journal_bytes
        ):
            raise ProviderPhaseRuntimeError(
                "label-release authority journal changes its exact evidence"
            )
    elif existing_phase_authority is None:
        try:
            write_exclusive_receipt_bytes(journal_bytes, journal_path)
        except ArtifactIntegrityError as exc:
            raise ProviderPhaseRuntimeError(
                "cannot persist label-release authority journal"
            ) from exc
    if os.path.lexists(intent_path):
        try:
            intent_path.unlink()
            output_descriptor = os.open(
                Path(row.output_root),
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(output_descriptor)
            finally:
                os.close(output_descriptor)
            phase_descriptor = os.open(
                phase_root,
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(phase_descriptor)
            finally:
                os.close(phase_descriptor)
        except OSError as exc:
            raise ProviderPhaseRuntimeError("cannot close committed label release intent") from exc
    return authority


@dataclass(frozen=True)
class ProviderDriverOutput:
    corpus_id: str
    driver_id: str
    output_root: str
    output_tree_sha256: str
    output_entries: tuple[str, ...]
    analysis_execution_receipt_uri: str | None = None
    analysis_execution_receipt_sha256: str | None = None
    analysis_execution_receipt_file_sha256: str | None = None
    label_release_authority_sha256: str | None = None
    label_release_authority: LabelReleaseOutputAuthority | None = None
    schema_version: str = PROVIDER_DRIVER_OUTPUT_SCHEMA

    def __post_init__(self) -> None:
        _text("corpus_id", self.corpus_id)
        _text("driver_id", self.driver_id)
        _absolute_path("output_root", self.output_root)
        _digest("output_tree_sha256", self.output_tree_sha256)
        entries = tuple(self.output_entries)
        if (
            not entries
            or not all(type(item) is str and item for item in entries)
            or list(entries) != sorted(entries, key=lambda value: value.encode("utf-8"))
        ):
            raise ProviderPhaseRuntimeError("provider driver output inventory differs")
        analysis_values = (
            self.analysis_execution_receipt_uri,
            self.analysis_execution_receipt_sha256,
            self.analysis_execution_receipt_file_sha256,
        )
        if self.driver_id == _DRIVER_IDS[ANALYSIS_PHASE]:
            if any(type(value) is not str for value in analysis_values):
                raise ProviderPhaseRuntimeError(
                    "analysis driver output lacks execution-receipt evidence"
                )
            assert self.analysis_execution_receipt_uri is not None
            parsed = urlsplit(self.analysis_execution_receipt_uri)
            execution_path = Path(unquote(parsed.path))
            if (
                parsed.scheme != "file"
                or parsed.netloc
                or parsed.query
                or parsed.fragment
                or not execution_path.is_absolute()
                or execution_path.as_uri() != self.analysis_execution_receipt_uri
            ):
                raise ProviderPhaseRuntimeError(
                    "analysis execution receipt URI is not canonical local evidence"
                )
            _digest(
                "analysis_execution_receipt_sha256",
                self.analysis_execution_receipt_sha256,
            )
            _digest(
                "analysis_execution_receipt_file_sha256",
                self.analysis_execution_receipt_file_sha256,
            )
        elif any(value is not None for value in analysis_values):
            raise ProviderPhaseRuntimeError(
                "non-analysis output introduced analysis execution evidence"
            )
        if self.driver_id == _DRIVER_IDS[LABEL_RELEASE_PHASE]:
            if (
                not isinstance(
                    self.label_release_authority,
                    LabelReleaseOutputAuthority,
                )
                or self.label_release_authority.corpus_id != self.corpus_id
                or self.label_release_authority_sha256
                != self.label_release_authority.authority_sha256
            ):
                raise ProviderPhaseRuntimeError(
                    "label-release output lacks its exact action authority"
                )
        elif (
            self.label_release_authority is not None
            or self.label_release_authority_sha256 is not None
        ):
            raise ProviderPhaseRuntimeError("non-label output introduced label-release authority")
        if self.schema_version != PROVIDER_DRIVER_OUTPUT_SCHEMA:
            raise ProviderPhaseRuntimeError("provider driver output schema differs")
        object.__setattr__(self, "output_entries", entries)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"label_release_authority", "output_entries"}
            },
            "label_release_authority": (
                None
                if self.label_release_authority is None
                else self.label_release_authority.to_dict()
            ),
            "output_entries": list(self.output_entries),
        }


@dataclass(frozen=True)
class ProviderPhaseExecutionReceipt:
    phase: ProviderPhase
    suite_attempt_id: str
    provider_plan_sha256: str
    provider_plan_file_sha256: str
    claim_receipt_file_sha256: str
    phase_host_tool_receipt_path: str
    phase_host_tool_receipt_sha256: str
    phase_host_tool_receipt_file_sha256: str
    runtime_request_sha256: str
    runtime_request_file_sha256: str
    outputs: tuple[ProviderDriverOutput, ...]
    schema_version: str = PROVIDER_PHASE_EXECUTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in _DRIVER_IDS:
            raise ProviderPhaseRuntimeError("provider execution receipt has another phase")
        for name in (
            "suite_attempt_id",
            "provider_plan_sha256",
            "provider_plan_file_sha256",
            "claim_receipt_file_sha256",
            "phase_host_tool_receipt_sha256",
            "phase_host_tool_receipt_file_sha256",
            "runtime_request_sha256",
            "runtime_request_file_sha256",
        ):
            _digest(name, getattr(self, name))
        _absolute_path(
            "phase_host_tool_receipt_path",
            self.phase_host_tool_receipt_path,
        )
        rows = tuple(self.outputs)
        if (
            not rows
            or not all(isinstance(row, ProviderDriverOutput) for row in rows)
            or any(row.driver_id != _DRIVER_IDS[self.phase] for row in rows)
        ):
            raise ProviderPhaseRuntimeError("provider execution receipt lacks typed outputs")
        if self.schema_version != PROVIDER_PHASE_EXECUTION_RECEIPT_SCHEMA:
            raise ProviderPhaseRuntimeError("provider execution receipt schema differs")
        object.__setattr__(self, "outputs", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name) for name in self.__dataclass_fields__ if name != "outputs"
            },
            "outputs": [row.to_dict() for row in self.outputs],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, encoded: bytes) -> ProviderPhaseExecutionReceipt:
        if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
            raise ProviderPhaseRuntimeError(
                "provider phase execution receipt needs one terminal newline"
            )
        row = _closed(
            _strict_object(
                encoded[:-1],
                label="provider phase execution receipt",
            ),
            frozenset(cls.__dataclass_fields__),
            label="provider phase execution receipt",
        )
        raw_outputs = row["outputs"]
        if not isinstance(raw_outputs, list):
            raise ProviderPhaseRuntimeError("provider phase execution outputs must be an array")
        outputs: list[ProviderDriverOutput] = []
        for raw_output in raw_outputs:
            output = _closed(
                raw_output,
                frozenset(ProviderDriverOutput.__dataclass_fields__),
                label="provider driver output",
            )
            entries = output["output_entries"]
            if not isinstance(entries, list):
                raise ProviderPhaseRuntimeError("provider driver output entries must be an array")
            raw_authority = output["label_release_authority"]
            authority = (
                None
                if raw_authority is None
                else LabelReleaseOutputAuthority.from_dict(raw_authority)
            )
            outputs.append(
                ProviderDriverOutput(
                    **{
                        key: value
                        for key, value in output.items()
                        if key
                        not in {
                            "label_release_authority",
                            "output_entries",
                        }
                    },
                    label_release_authority=authority,
                    output_entries=tuple(entries),
                )
            )
        receipt = cls(
            **{key: value for key, value in row.items() if key != "outputs"},
            outputs=tuple(outputs),
        )
        if receipt.canonical_file_bytes() != encoded:
            raise ProviderPhaseRuntimeError(
                "provider phase execution receipt bytes are not canonical"
            )
        return receipt


def _admit_existing_phase_execution_receipt(
    target: Path,
    request: ProviderPhaseRuntimeRequest,
    fresh: ProviderPhaseExecutionReceipt,
) -> ProviderPhaseExecutionReceipt:
    if request.phase not in {LABEL_RELEASE_PHASE, ANALYSIS_PHASE}:
        raise ProviderPhaseRuntimeError("provider phase execution receipt already exists")
    existing = ProviderPhaseExecutionReceipt.from_bytes(
        _secure_file_bytes(
            target,
            label="provider phase execution receipt",
        )
    )
    if (
        existing.phase != fresh.phase
        or existing.suite_attempt_id != fresh.suite_attempt_id
        or existing.provider_plan_sha256 != fresh.provider_plan_sha256
        or existing.provider_plan_file_sha256 != fresh.provider_plan_file_sha256
        or existing.claim_receipt_file_sha256 != fresh.claim_receipt_file_sha256
        or existing.phase_host_tool_receipt_path != fresh.phase_host_tool_receipt_path
        or existing.phase_host_tool_receipt_sha256 != fresh.phase_host_tool_receipt_sha256
        or existing.phase_host_tool_receipt_file_sha256 != fresh.phase_host_tool_receipt_file_sha256
        or existing.outputs != fresh.outputs
    ):
        raise ProviderPhaseRuntimeError(
            "existing provider phase receipt differs from fresh closure"
        )
    return existing


def _output_receipt(
    row: ProviderDriverRequest,
    *,
    analysis_outcome: object | None = None,
    label_release_authority: LabelReleaseOutputAuthority | None = None,
) -> ProviderDriverOutput:
    try:
        inventory = digest_directory_tree(Path(row.output_root))
    except ArtifactIntegrityError as exc:
        raise ProviderPhaseRuntimeError("cannot close provider driver output tree") from exc
    if not inventory.entries:
        raise ProviderPhaseRuntimeError("provider driver produced an empty output tree")
    analysis_values: dict[str, str | None] = {
        "analysis_execution_receipt_uri": None,
        "analysis_execution_receipt_sha256": None,
        "analysis_execution_receipt_file_sha256": None,
    }
    if analysis_outcome is not None:
        from .offline_analysis_provider import OfflineAnalysisOutcome

        if row.driver_id != _DRIVER_IDS[ANALYSIS_PHASE] or not isinstance(
            analysis_outcome, OfflineAnalysisOutcome
        ):
            raise ProviderPhaseRuntimeError(
                "analysis output receipt received untyped execution evidence"
            )
        analysis_values = {
            "analysis_execution_receipt_uri": (analysis_outcome.execution_receipt_path.as_uri()),
            "analysis_execution_receipt_sha256": (analysis_outcome.execution_receipt_sha256),
            "analysis_execution_receipt_file_sha256": (
                analysis_outcome.execution_receipt_file_sha256
            ),
        }
    return ProviderDriverOutput(
        corpus_id=row.corpus_id,
        driver_id=row.driver_id,
        output_root=row.output_root,
        output_tree_sha256=inventory.sha256,
        output_entries=tuple(inventory.entries),
        **analysis_values,
        label_release_authority_sha256=(
            None if label_release_authority is None else label_release_authority.authority_sha256
        ),
        label_release_authority=label_release_authority,
    )


def _run_online(
    row: ProviderDriverRequest,
    claim_bytes: bytes,
    run_claim: VerifiedRunClaimCapability | None,
) -> None:
    try:
        receipt = loads_runtime_claim_receipt(claim_bytes)
    except ExecutionClaimError as exc:
        raise ProviderPhaseRuntimeError("online runtime receipt is invalid") from exc
    if receipt.schema_version != RUNTIME_CLAIM_RECEIPT_SCHEMA:
        raise ProviderPhaseRuntimeError("online runtime receipt schema differs")
    if not isinstance(run_claim, VerifiedRunClaimCapability):
        raise ProviderPhaseRuntimeError("online sealed launch lacks in-memory claim authority")
    control = OnlineSealedLaunchDriverControl.from_bytes(
        _verified_file(Path(row.control_path), row.control_file_sha256, label="online control")
    )
    from .production_controls import verify_production_run_closure_authority
    from .runtime_attestation import loads_runtime_preflight_receipt
    from .sealed_container_launcher import (
        launch_sealed_once,
        load_preflight_launch_contract,
        load_registered_plan_instantiation,
        load_runtime_plan_transition,
        load_sealed_launch_contract,
        load_volume_initialization_receipt,
    )

    preflight = load_preflight_launch_contract(control.preflight_contract_path)
    try:
        preflight_receipt = loads_runtime_preflight_receipt(
            _secure_file_bytes(
                Path(control.preflight_receipt_path), label="runtime preflight receipt"
            )
        )
    except Exception as exc:
        raise ProviderPhaseRuntimeError("runtime preflight receipt is invalid") from exc
    transition = load_runtime_plan_transition(control.transition_receipt_path)
    verified_closure = verify_production_run_closure_authority(
        finalization_request_path=control.finalization_request_path,
        finalization_receipt_path=control.finalization_receipt_path,
        preflight=preflight,
        transition=transition,
    )
    sealed = load_sealed_launch_contract(control.sealed_contract_path)
    if sealed.geometry.corpus_id != row.corpus_id or Path(sealed.geometry.copy_output_root) != Path(
        row.output_root
    ):
        raise ProviderPhaseRuntimeError("online sealed launch changes corpus or output root")
    launch_sealed_once(
        sealed,
        preflight,
        preflight_receipt,
        transition,
        load_registered_plan_instantiation(control.instantiation_receipt_path),
        verified_closure,
        load_volume_initialization_receipt(control.volume_receipt_path),
        run_claim,
        secret=claim_bytes,
        audit_root=control.audit_root,
    )


def _run_label_release(
    row: ProviderDriverRequest,
    claim_bytes: bytes,
    phase_claim: VerifiedPhaseClaimCapability,
    provider_claimed: VerifiedProviderPredecessor,
    tle_runner: DockerTleDecryptRunner,
) -> object:
    if not isinstance(phase_claim, VerifiedPhaseClaimCapability):
        raise ProviderPhaseRuntimeError("label release lacks in-memory claim authority")
    if not isinstance(provider_claimed, VerifiedProviderPredecessor):
        raise ProviderPhaseRuntimeError("label release lacks verified provider state")
    if not isinstance(tle_runner, DockerTleDecryptRunner):
        raise ProviderPhaseRuntimeError("label release lacks the fixed Docker TLE runner")
    if (
        provider_claimed.state.state != "LABEL_RELEASE_CLAIMED"
        or provider_claimed.state.record_sha256 != phase_claim.phase_claim_state_sha256
        or provider_claimed.ledger_commit != phase_claim.phase_claim_ledger_commit
    ):
        raise ProviderPhaseRuntimeError(
            "label release capability and provider state name another claim"
        )
    receipt = _load_phase_runtime_claim(
        claim_bytes,
        phase=LABEL_RELEASE_PHASE,
        corpus_id=row.corpus_id,
    )
    expected = phase_claim.require_input(
        corpus_id=receipt.corpus_id,
        input_uri=receipt.input_uri,
        input_sha256=receipt.input_sha256,
        supporting_input_uri=receipt.supporting_input_uri,
        supporting_input_sha256=receipt.supporting_input_sha256,
    )
    if expected.canonical_file_bytes() != claim_bytes:
        raise ProviderPhaseRuntimeError(
            "label runtime receipt differs from freshly admitted authority"
        )
    control_bytes = _verified_file(
        Path(row.control_path), row.control_file_sha256, label="label-release driver control"
    )
    control = LabelReleaseDriverControl.from_bytes(control_bytes)
    bindings = [
        binding for binding in phase_claim.contract.corpora if binding.corpus_id == row.corpus_id
    ]
    if len(bindings) != 1:
        raise ProviderPhaseRuntimeError("label-release claim lacks one exact corpus binding")
    binding = bindings[0]
    if Path(control.plaintext_output_path).parent != Path(row.output_root):
        raise ProviderPhaseRuntimeError("label plaintext output differs from its driver root")
    if Path(control.decryption_receipt_path).parent != Path(row.output_root):
        raise ProviderPhaseRuntimeError("label receipt output differs from its driver root")
    if (
        Path(control.ciphertext_path).as_uri() != receipt.input_uri
        or Path(control.encryption_receipt_path).as_uri() != receipt.supporting_input_uri
        or Path(control.plaintext_output_path).as_uri() != binding.output_uri
        or Path(control.decryption_receipt_path).name != "timelock-decryption-receipt.json"
    ):
        raise ProviderPhaseRuntimeError(
            "label-release control paths differ from the fresh phase binding"
        )
    if provider_claimed.namespace != Path(control.suite_namespace):
        raise ProviderPhaseRuntimeError(
            "label-release control differs from the verified canonical suite namespace"
        )
    completion_root = provider_claimed.namespace / "completion"
    if (
        Path(control.completion_receipt_path)
        != completion_root / f"{row.corpus_id}-prediction-completion.json"
        or Path(control.completion_anchor_record_path)
        != completion_root / f"{row.corpus_id}-prediction-completion-anchor.json"
        or Path(control.completion_anchor_receipt_path)
        != completion_root / f"{row.corpus_id}-prediction-completion-anchor-receipt.json"
    ):
        raise ProviderPhaseRuntimeError(
            "label-release completion paths differ from the provider closure"
        )
    try:
        verified_completion = revalidate_post_online_completion_authority(
            provider_claimed,
            phase_claim,
        )
    except PostOnlineCompletionError as exc:
        raise ProviderPhaseRuntimeError(
            "post-online completion authority failed anonymous revalidation"
        ) from exc

    verified_release = release_timelock_label(
        load_study_manifest(control.manifest_path),
        corpus_id=row.corpus_id,
        custody_seal=load_custody_seal_receipt(control.custody_seal_path),
        encryption_receipt=load_timelock_encryption_receipt(control.encryption_receipt_path),
        verified_post_online_completion=verified_completion,
        verified_suite_completion=provider_claimed,
        verified_phase_claim=phase_claim,
        ciphertext_path=control.ciphertext_path,
        tle_binary_path=control.tle_binary_path,
        plaintext_output_path=control.plaintext_output_path,
        decryption_receipt_output_path=control.decryption_receipt_path,
        trusted_tle_runner=tle_runner,
    )
    if (
        load_timelock_decryption_receipt(control.decryption_receipt_path)
        != verified_release.receipt
    ):
        raise ProviderPhaseRuntimeError("persisted label-release receipt differs")
    return verified_release


def _verify_pre_decryption_marker(
    row: ProviderDriverRequest,
    authority: FreshLabelClaimAuthority,
    *,
    suite_attempt_id: str,
) -> None:
    encoded = _verified_file(
        Path(authority.admission_marker_path),
        authority.admission_marker_sha256,
        label=f"{row.corpus_id} pre-decryption admission marker",
    )
    marker = _closed(
        _strict_object(encoded, label="pre-decryption admission marker"),
        frozenset(
            {
                "admitted_at_utc",
                "beacon_receipt_sha256",
                "corpus_id",
                "input_sha256",
                "input_uri",
                "live_execute_job_receipt_sha256",
                "output_identity_sha256",
                "output_uri",
                "phase",
                "phase_claim_contract_sha256",
                "phase_claim_ledger_commit",
                "phase_claim_state_sha256",
                "provider_identity_sha256",
                "runtime_claim_receipt_sha256",
                "schema_version",
                "suite_attempt_id",
                "supporting_input_sha256",
                "supporting_input_uri",
            }
        ),
        label="pre-decryption admission marker",
    )
    if encoded != _canonical_bytes(marker) + b"\n":
        raise ProviderPhaseRuntimeError("pre-decryption marker bytes are not canonical")
    receipt = _load_phase_runtime_claim(
        authority.claim_bytes,
        phase=LABEL_RELEASE_PHASE,
        corpus_id=row.corpus_id,
    )
    matches = [
        binding
        for binding in authority.capability.contract.corpora
        if binding.corpus_id == row.corpus_id
    ]
    if len(matches) != 1:
        raise ProviderPhaseRuntimeError("label marker lacks one claimed corpus binding")
    binding = matches[0]
    expected = {
        "beacon_receipt_sha256": receipt.phase_beacon_receipt_sha256,
        "corpus_id": row.corpus_id,
        "input_sha256": receipt.input_sha256,
        "input_uri": receipt.input_uri,
        "live_execute_job_receipt_sha256": receipt.live_execute_job_receipt_sha256,
        "output_identity_sha256": receipt.phase_output_identity,
        "output_uri": binding.output_uri,
        "phase": LABEL_RELEASE_PHASE,
        "phase_claim_contract_sha256": receipt.phase_claim_contract_sha256,
        "phase_claim_ledger_commit": receipt.phase_claim_ledger_commit,
        "phase_claim_state_sha256": receipt.phase_claim_state_sha256,
        "provider_identity_sha256": receipt.provider_identity_sha256,
        "runtime_claim_receipt_sha256": hashlib.sha256(authority.claim_bytes).hexdigest(),
        "schema_version": "fractal-pre-decryption-admission-v1",
        "suite_attempt_id": suite_attempt_id,
        "supporting_input_sha256": receipt.supporting_input_sha256,
        "supporting_input_uri": receipt.supporting_input_uri,
    }
    if any(marker.get(name) != value for name, value in expected.items()):
        raise ProviderPhaseRuntimeError("pre-decryption marker differs from fresh claim authority")
    _text("admitted_at_utc", marker.get("admitted_at_utc"))


def _run_analysis(
    plan: ProviderPhasePlan,
    row: ProviderDriverRequest,
    claim_bytes: bytes,
    provider_claimed: VerifiedProviderPredecessor | None,
    phase_claim: VerifiedPhaseClaimCapability | None,
    fresh_claim_supplier: AnalysisClaimSupplier | None = None,
) -> object:
    bundle = AnalysisRuntimeClaimBundle.from_bytes(claim_bytes)
    if not isinstance(provider_claimed, VerifiedProviderPredecessor) or not isinstance(
        phase_claim, VerifiedPhaseClaimCapability
    ):
        raise ProviderPhaseRuntimeError("analysis driver lacks provider claim authority")
    for receipt in bundle.receipts:
        fresh_receipt = phase_claim.require_input(
            corpus_id=receipt.corpus_id,
            input_uri=receipt.input_uri,
            input_sha256=receipt.input_sha256,
            supporting_input_uri=receipt.supporting_input_uri,
            supporting_input_sha256=receipt.supporting_input_sha256,
        )
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "c1_commit",
            "phase_claim_contract_sha256",
            "phase_claim_state_sha256",
            "phase_claim_ledger_commit",
            "provider_identity_sha256",
            "phase_input_aggregate_sha256",
            "phase_output_identity",
            "corpus_id",
            "input_uri",
            "input_sha256",
            "supporting_input_uri",
            "supporting_input_sha256",
        ):
            if getattr(fresh_receipt, name) != getattr(receipt, name):
                raise ProviderPhaseRuntimeError(
                    "analysis start authority differs from the fixed input intent"
                )
    if fresh_claim_supplier is None:
        raise ProviderPhaseRuntimeError("analysis runtime lacks a fresh claim supplier")
    from .confirmatory_input_operator import load_confirmatory_input_operator_config
    from .offline_analysis_provider import run_provider_claimed_offline_analysis_once

    completion = run_provider_claimed_offline_analysis_once(
        load_confirmatory_input_operator_config(row.control_path),
        plan,
        provider_claimed,
        phase_claim,
        package_root=analysis_offline_package_root(plan),
        results_root=Path(row.output_root),
        fresh_claim_supplier=fresh_claim_supplier,
    )
    if completion.candidate.state != "ANALYSIS_COMPLETE":
        raise ProviderPhaseRuntimeError("analysis driver did not reach candidate closure")
    return completion


_DRIVERS: dict[ProviderPhase, Callable[[ProviderDriverRequest, bytes], None]] = {}


def execute_provider_phase_request(
    *,
    plan: ProviderPhasePlan,
    request: ProviderPhaseRuntimeRequest,
    online_run_claim: VerifiedRunClaimCapability | None = None,
    online_run_claim_supplier: OnlineRunClaimSupplier | None = None,
    label_phase_claim_supplier: LabelPhaseClaimSupplier | None = None,
    provider_claimed: VerifiedProviderPredecessor | None = None,
    analysis_phase_claim: VerifiedPhaseClaimCapability | None = None,
    analysis_claim_supplier: AnalysisClaimSupplier | None = None,
) -> ProviderPhaseExecutionReceipt:
    """Execute the exact canonical request without caller-selected driver data."""

    if not isinstance(plan, ProviderPhasePlan) or plan.schema_version != PROVIDER_PHASE_PLAN_SCHEMA:
        raise ProviderPhaseRuntimeError("provider runtime requires a typed resolved C1 plan")
    if not isinstance(request, ProviderPhaseRuntimeRequest):
        raise ProviderPhaseRuntimeError("provider runtime requires a typed request")
    if (
        request.phase != plan.phase
        or request.activation_command_id != plan.activation_command_id
        or request.suite_attempt_id != plan.suite_attempt_id
        or request.provider_plan_path != plan.provider_plan_path
        or request.provider_plan_sha256 != plan.plan_sha256
        or request.provider_plan_file_sha256 != plan.file_sha256
        or request.claim_receipt_path != plan.claim_receipt_path(request.suite_attempt_id)
        or request.phase_output_root != plan.phase_evidence_root(request.suite_attempt_id)
    ):
        raise ProviderPhaseRuntimeError("runtime request differs from the resolved C1 plan")
    plan_bytes = _verified_file(
        Path(request.provider_plan_path),
        request.provider_plan_file_sha256,
        label="resolved provider plan",
    )
    if plan_bytes != plan.canonical_file_bytes():
        raise ProviderPhaseRuntimeError("runtime plan bytes differ after typed admission")
    _verified_file(
        Path(request.claim_receipt_path),
        request.claim_receipt_file_sha256,
        label="provider claim receipt",
    )
    host_receipt_bytes = _verified_file(
        Path(request.phase_host_tool_receipt_path),
        request.phase_host_tool_receipt_file_sha256,
        label="phase host-tool receipt",
    )
    if not host_receipt_bytes.endswith(b"\n") or host_receipt_bytes.endswith(b"\n\n"):
        raise ProviderPhaseRuntimeError("phase host-tool receipt needs one terminal newline")
    try:
        host_receipt = PhaseHostToolReceipt.from_dict(
            _strict_object(
                host_receipt_bytes[:-1],
                label="phase host-tool receipt",
            )
        )
    except ExecutionClaimError as exc:
        raise ProviderPhaseRuntimeError("phase host-tool receipt is invalid") from exc
    if (
        host_receipt.receipt_sha256 != request.phase_host_tool_receipt_sha256
        or _canonical_bytes(host_receipt.to_dict()) + b"\n" != host_receipt_bytes
    ):
        raise ProviderPhaseRuntimeError("phase host-tool receipt semantic digest differs")
    expected_host_receipt = {
        "contract_sha256": plan.host_tools.contract_sha256,
        "controlled_root_realpath": plan.host_tools.controlled_root,
        "docker_executable_sha256": plan.host_tools.docker_executable_sha256,
        "docker_resolved_executable": plan.host_tools.docker_resolved_executable,
        "docker_server_probe_receipt_file_sha256": (
            plan.host_tools.docker_server_probe_receipt_sha256
        ),
        "gh_executable_sha256": plan.host_tools.gh_executable_sha256,
        "host_probe_receipt_file_sha256": (plan.host_tools.host_probe_receipt_sha256),
        "python_executable_sha256": plan.host_tools.python_executable_sha256,
        "python_import_tree_sha256": plan.host_tools.python_import_tree_sha256,
        "python_package_content_sha256": (plan.host_tools.python_package_content_sha256),
        "python_package_tree_sha256": plan.host_tools.python_package_tree_sha256,
        "runner_config_sha256": plan.host_tools.runner_config_sha256,
        "runner_listener_dll_sha256": plan.host_tools.runner_listener_dll_sha256,
        "runner_listener_sha256": plan.host_tools.runner_listener_sha256,
        "runner_run_sha256": plan.host_tools.runner_run_sha256,
        "venv_symlink_inventory_sha256": (plan.host_tools.venv_symlink_inventory_sha256),
        "venv_tree_sha256": plan.host_tools.venv_tree_sha256,
    }
    if any(
        getattr(host_receipt, name) != expected for name, expected in expected_host_receipt.items()
    ):
        raise ProviderPhaseRuntimeError("phase host-tool receipt differs from the resolved C1 plan")
    request_path = Path(request.phase_input_root) / PROVIDER_RUNTIME_REQUEST_FILENAME
    request_bytes = _verified_file(
        request_path,
        request.file_sha256,
        label="provider runtime request",
    )
    if request_bytes != request.canonical_file_bytes():
        raise ProviderPhaseRuntimeError("runtime request changed after typed admission")
    if online_run_claim is not None and online_run_claim_supplier is not None:
        raise ProviderPhaseRuntimeError("online runtime received two claim authorities")
    if request.phase == LABEL_RELEASE_PHASE and label_phase_claim_supplier is None:
        raise ProviderPhaseRuntimeError("label runtime lacks a fresh claim supplier")
    phase_output_root = Path(request.phase_output_root)
    existing_label_authorities: dict[str, LabelReleaseOutputAuthority] = {}
    if online_run_claim_supplier is not None:
        if _controlled_directory_entries(phase_output_root, label="online phase output root"):
            raise ProviderPhaseRuntimeError("online phase output root is not empty")
    if label_phase_claim_supplier is not None:
        expected = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
        label_admission = admit_label_release_phase_root(
            phase_output_root,
            create_if_absent=False,
        )
        if label_admission.execution_receipt_present:
            existing_receipt = ProviderPhaseExecutionReceipt.from_bytes(
                _secure_file_bytes(
                    phase_output_root / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME,
                    label="provider phase execution receipt",
                )
            )
            if (
                existing_receipt.phase != LABEL_RELEASE_PHASE
                or existing_receipt.suite_attempt_id != request.suite_attempt_id
                or existing_receipt.provider_plan_sha256 != request.provider_plan_sha256
                or existing_receipt.provider_plan_file_sha256 != request.provider_plan_file_sha256
                or existing_receipt.claim_receipt_file_sha256 != request.claim_receipt_file_sha256
                or existing_receipt.phase_host_tool_receipt_path
                != request.phase_host_tool_receipt_path
                or existing_receipt.phase_host_tool_receipt_sha256
                != request.phase_host_tool_receipt_sha256
                or existing_receipt.phase_host_tool_receipt_file_sha256
                != request.phase_host_tool_receipt_file_sha256
            ):
                raise ProviderPhaseRuntimeError(
                    "existing label phase receipt differs from the current claim"
                )
            existing_label_authorities = {
                output.corpus_id: output.label_release_authority
                for output in existing_receipt.outputs
                if output.label_release_authority is not None
            }
            if set(existing_label_authorities) != set(FIXED_CORPORA):
                raise ProviderPhaseRuntimeError(
                    "existing label phase receipt lacks five action authorities"
                )
        rows = {row.corpus_id: row for row in request.drivers}
        if set(rows) != set(expected) or any(
            Path(rows[corpus_id].output_root) != phase_output_root / corpus_id
            for corpus_id in expected
        ):
            raise ProviderPhaseRuntimeError(
                "label phase output roots differ from the fixed corpus targets"
            )
    if request.phase == ANALYSIS_PHASE:
        if not isinstance(analysis_phase_claim, VerifiedPhaseClaimCapability):
            raise ProviderPhaseRuntimeError("analysis runtime lacks phase claim authority")
        output_uris = {row.output_uri for row in analysis_phase_claim.contract.corpora}
        if len(output_uris) != 1:
            raise ProviderPhaseRuntimeError("analysis claim has no sole output namespace")
        parsed = urlsplit(next(iter(output_uris)))
        authorized_output = Path(unquote(parsed.path))
        if (
            parsed.scheme != "file"
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or authorized_output.as_uri() != next(iter(output_uris))
            or request.drivers[0].output_root != str(authorized_output)
        ):
            raise ProviderPhaseRuntimeError(
                "analysis driver output differs from the claimed results store"
            )
        phase_entries = _controlled_directory_entries(
            phase_output_root,
            label="analysis phase evidence root",
        )
        if not set(phase_entries).issubset({PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME}):
            raise ProviderPhaseRuntimeError(
                "analysis phase evidence root contains an unregistered restart member"
            )
        admit_analysis_results_store(
            authorized_output,
            manifest_sha256=analysis_phase_claim.contract.manifest_sha256,
        )

    outputs: list[ProviderDriverOutput] = []
    label_tle_runner = (
        DockerTleDecryptRunner.from_plan(plan) if request.phase == LABEL_RELEASE_PHASE else None
    )
    if label_tle_runner is not None:
        label_tle_runner.prepare()
    for row in request.drivers:
        analysis_completion: Any | None = None
        label_output_authority: LabelReleaseOutputAuthority | None = None
        _verified_file(
            Path(row.control_path),
            row.control_file_sha256,
            label=f"{row.corpus_id} driver control",
        )
        claim_bytes = _verified_file(
            Path(row.runtime_claim_receipt_path),
            row.runtime_claim_receipt_file_sha256,
            label=f"{row.corpus_id} runtime claim receipt",
        )
        if request.phase == ONLINE_PHASE and online_run_claim_supplier is not None:
            fresh_online = online_run_claim_supplier(row)
            if not isinstance(fresh_online, FreshOnlineClaimAuthority):
                raise ProviderPhaseRuntimeError("online supplier returned untyped authority")
            _run_online(row, fresh_online.claim_bytes, fresh_online.capability)
        elif (
            request.phase == ONLINE_PHASE and online_run_claim is None and ONLINE_PHASE in _DRIVERS
        ):
            _DRIVERS[ONLINE_PHASE](row, claim_bytes)
        elif request.phase == ONLINE_PHASE:
            _run_online(row, claim_bytes, online_run_claim)
        elif request.phase == ANALYSIS_PHASE:
            if analysis_claim_supplier is None:
                raise ProviderPhaseRuntimeError("analysis runtime lacks a fresh claim supplier")
            start_claimed, start_phase_claim = analysis_claim_supplier()
            analysis_completion = _run_analysis(
                plan,
                row,
                claim_bytes,
                start_claimed,
                start_phase_claim,
                analysis_claim_supplier,
            )
        elif request.phase == LABEL_RELEASE_PHASE and label_phase_claim_supplier is not None:
            fresh_label = label_phase_claim_supplier(row)
            if not isinstance(fresh_label, FreshLabelClaimAuthority):
                raise ProviderPhaseRuntimeError("label supplier returned untyped authority")
            if label_tle_runner is None:
                raise ProviderPhaseRuntimeError("label runtime lacks its Docker TLE runner")
            _verify_pre_decryption_marker(
                row,
                fresh_label,
                suite_attempt_id=request.suite_attempt_id,
            )
            verified_release = _run_label_release(
                row,
                fresh_label.claim_bytes,
                fresh_label.capability,
                fresh_label.predecessor,
                label_tle_runner,
            )
            label_output_authority = _close_label_release_action_authority(
                row=row,
                receipt=verified_release.receipt,
                fresh=fresh_label,
                existing_phase_authority=existing_label_authorities.get(row.corpus_id),
            )
        elif request.phase in _DRIVERS:
            _DRIVERS[request.phase](row, claim_bytes)
        else:
            raise ProviderPhaseRuntimeError("provider phase lacks an in-memory execution authority")
        output = _output_receipt(
            row,
            analysis_outcome=(None if analysis_completion is None else analysis_completion.outcome),
            label_release_authority=label_output_authority,
        )
        if request.phase == ANALYSIS_PHASE:
            from .confirmatory_input_operator import confirmatory_store_closure_filenames

            expected_entries = confirmatory_store_closure_filenames(
                analysis_phase_claim.contract.manifest_sha256
            )
            if output.output_entries != expected_entries:
                raise ProviderPhaseRuntimeError(
                    "analysis output tree differs from the exact five-file closure"
                )
        outputs.append(output)

    receipt = ProviderPhaseExecutionReceipt(
        phase=request.phase,
        suite_attempt_id=request.suite_attempt_id,
        provider_plan_sha256=request.provider_plan_sha256,
        provider_plan_file_sha256=request.provider_plan_file_sha256,
        claim_receipt_file_sha256=request.claim_receipt_file_sha256,
        phase_host_tool_receipt_path=request.phase_host_tool_receipt_path,
        phase_host_tool_receipt_sha256=request.phase_host_tool_receipt_sha256,
        phase_host_tool_receipt_file_sha256=(request.phase_host_tool_receipt_file_sha256),
        runtime_request_sha256=request.request_sha256,
        runtime_request_file_sha256=request.file_sha256,
        outputs=tuple(outputs),
    )
    target = Path(request.phase_output_root) / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME
    if os.path.lexists(target):
        receipt = _admit_existing_phase_execution_receipt(
            target,
            request,
            receipt,
        )
    else:
        try:
            write_exclusive_receipt_bytes(receipt.canonical_file_bytes(), target)
        except ArtifactIntegrityError as exc:
            raise ProviderPhaseRuntimeError(
                "cannot close provider phase execution receipt"
            ) from exc
    if digest_regular_file(target, label="provider phase execution receipt") != receipt.file_sha256:
        raise ProviderPhaseRuntimeError("provider phase execution receipt failed readback")
    if request.phase == LABEL_RELEASE_PHASE:
        try:
            for corpus_id in FIXED_CORPORA:
                journal = Path(request.phase_output_root) / label_release_authority_journal_name(
                    corpus_id
                )
                if os.path.lexists(journal):
                    journal.unlink()
            phase_descriptor = os.open(
                request.phase_output_root,
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(phase_descriptor)
            finally:
                os.close(phase_descriptor)
        except OSError as exc:
            raise ProviderPhaseRuntimeError(
                "cannot close label-release authority journals"
            ) from exc
        admission = admit_label_release_phase_root(
            Path(request.phase_output_root),
            create_if_absent=False,
        )
        if (
            admission.completed_corpora
            != tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
            or admission.staged_corpus is not None
            or not admission.execution_receipt_present
        ):
            raise ProviderPhaseRuntimeError(
                "label phase did not close its exact five pairs and one receipt"
            )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.provider_phase_runtime",
        description="Execute one C1-bound provider phase request.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in PROVIDER_PHASE_COMMAND_IDS.values():
        item = commands.add_parser(command)
        item.add_argument("--provider-plan", type=Path, required=True)
        item.add_argument("--suite-attempt-id", required=True)
        item.add_argument("--claim-receipt", type=Path, required=True)
        item.add_argument("--phase-input-root", type=Path, required=True)
        item.add_argument("--phase-output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        plan = load_materialized_provider_phase_plan(arguments.provider_plan)
        # This fixed local receipt is admitted before the request root is even opened.
        # The hosted claim job validates the embedded C1 copy; the self-hosted runner
        # must additionally prove byte equality with its own bootstrap record.
        load_provider_runner_bootstrap(plan)
        phase = next(
            phase
            for phase, command in PROVIDER_PHASE_COMMAND_IDS.items()
            if command == arguments.command
        )
        request_path = arguments.phase_input_root / PROVIDER_RUNTIME_REQUEST_FILENAME
        request = load_provider_phase_runtime_request(request_path)
        if (
            plan.phase != phase
            or arguments.suite_attempt_id != plan.suite_attempt_id
            or arguments.claim_receipt != Path(plan.claim_receipt_path(plan.suite_attempt_id))
            or arguments.phase_output_root != Path(plan.phase_evidence_root(plan.suite_attempt_id))
            or request.phase_input_root != str(arguments.phase_input_root)
        ):
            raise ProviderPhaseRuntimeError("provider runtime argv differs from the C1 bindings")
        receipt = execute_provider_phase_request(plan=plan, request=request)
    except (ExecutionClaimError, ProviderPhaseRuntimeError) as exc:
        print(f"provider-phase-runtime error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "execution_receipt_file_sha256": receipt.file_sha256,
                "execution_receipt_path": str(
                    Path(request.phase_output_root) / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME
                ),
                "execution_receipt_sha256": receipt.receipt_sha256,
                "phase": receipt.phase,
                "suite_attempt_id": receipt.suite_attempt_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
