"""Single-attempt persistence for the complete pre-label online run."""

from __future__ import annotations

import errno
import hashlib
import http.client
import json
import os
import re
import socket
import stat
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Literal
from urllib.parse import unquote, urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_regular_file,
    read_secure_control_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .audit import AdmittedProvenanceRegistry, AuditRecord
from .authorized_index_store import (
    AuthorizedIndexStoreReceipt,
    HnswlibBackend,
    VerifiedAuthorizedIndexProvider,
    load_authorized_index_store_receipt,
    open_verified_document_matrices,
)
from .compiled_policy import (
    CompiledPolicyMaskStore,
    OpenPolicyAgentMaskDecisionPoint,
)
from .controller import ControllerConfig, GovernedRetriever, RuleController
from .custody import OnlineCustodyAdmissionReceipt
from .execution_claim import RuntimeClaimReceipt
from .label_separation import (
    OnlinePrediction,
    PredictionArtifact,
    emit_online_predictions,
)
from .online_runner import OnlineRunArtifacts, OnlineTrialRuntime
from .policy_intervention import (
    CATALOG_FILENAME,
    OPA_DATA_FILENAME,
    OPACompiledMaskData,
    PolicyInterventionError,
    derive_policy_transition_evidence,
    load_canonical_trial_schedule,
    load_opa_compiled_mask_data,
    load_policy_intervention_config,
    load_policy_intervention_receipt,
)
from .policy_intervention import (
    CONFIG_FILENAME as POLICY_CONFIG_FILENAME,
)
from .policy_intervention import (
    RECEIPT_FILENAME as POLICY_RECEIPT_FILENAME,
)
from .policy_intervention import (
    SCHEDULE_FILENAME as POLICY_SCHEDULE_FILENAME,
)
from .retrieval import PolicyTransitionEvidence
from .runtime_attestation import (
    LinuxRuntimeProbe,
    RuntimeArtifactMount,
    RuntimeAttestationError,
    RuntimeAttestationPlan,
    RuntimeAttestationReceipt,
    load_runtime_attestation_plan,
    load_runtime_attestation_receipt,
    verify_live_runtime_attestation,
)
from .scalable_execution import (
    EXECUTION_LEAF_RECEIPT_FILENAME,
    DigestOnlyProvenanceRegistry,
    execution_compatibility_view,
    open_digest_provenance_registry,
)
from .sealed_orchestrator import (
    RequiredArtifactIdBindings,
    run_admitted_online_matrix,
)
from .study import SealedRunReceipt
from .trial_runtime import (
    TrialRuntimeAdmission,
    TrialRuntimeAdmissionReceipt,
    load_trial_runtime,
)

ONLINE_ATTEMPT_SCHEMA = "fractal-sealed-online-attempt-v3"
ONLINE_OUTPUT_PIN_SCHEMA = "fractal-sealed-online-output-pin-v1"
ONLINE_RESULT_RECEIPT_SCHEMA = "fractal-sealed-online-result-receipt-v2"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_SUFFIX = ".sealed-online-attempt.json"
_RESULT_SUFFIX = ".sealed-online-result-receipt.json"
_OUTPUT_SUFFIXES = {
    "action-panel": ".action-panel.json",
    "action-panel-admission": ".action-panel-admission.json",
    "audit-chain": ".audit-chain.jsonl",
    "cache-preparation": ".cache-preparation.json",
    "execution-order": ".execution-order.json",
    "predictions": ".predictions.json",
}

PRODUCTION_OPA_ENDPOINT = "http://127.0.0.1:8181/v1/data/fractal_auth/retrieval/mask_decision"
PRODUCTION_K = 10
PRODUCTION_POLICY_ACTION = "retrieve"
PRODUCTION_PARTITION_LABEL: Literal["primary"] = "primary"
_PRODUCTION_OPA_BINARY = Path("/usr/local/bin/opa")
_PRODUCTION_OPA_REGO = Path("/opt/app/policy/opa_compiled_masks.rego")
_PRODUCTION_OPA_REGO_SHA256 = "18f6eb8a7411a7a1415bd2425ad5720f28fcd3b428d9aa2c1e7d73f6e14e356c"
_PRODUCTION_OPA_HOST = "127.0.0.1"
_PRODUCTION_OPA_PORT = 8181
_PRODUCTION_OPA_HEALTH_PATH = "/health?plugins"
_PRODUCTION_OPA_DECISION_PATH = "/v1/data/fractal_auth/retrieval/mask_decision"
_PRODUCTION_OPA_START_TIMEOUT_SECONDS = 10.0
_PRODUCTION_OPA_STOP_TIMEOUT_SECONDS = 5.0
_PRODUCTION_OPA_POLL_SECONDS = 0.05
_PRODUCTION_OPA_HTTP_TIMEOUT_SECONDS = 0.5
_PRODUCTION_OPA_STDERR_BYTES = 64 * 1024
_PRODUCTION_OPA_HTTP_BYTES = 64 * 1024
_PRODUCTION_OPA_SCRATCH_ROOT = Path("/tmp")
_PSEUDONYM_KEY_MAX_BYTES = 4096
_UNATTESTED_RUNTIME_SHA256 = "0" * 64
_RUNTIME_SOURCE_PATH_NAMES = frozenset(
    {
        "artifact_root",
        "authorized_index_store_root",
        "embedding_store_root",
        "partition_audit_path",
        "policy_intervention_root",
        "pseudonym_key_path",
        "query_package_root",
        "schedule_path",
        "staged_root",
    }
)


class SealedOnlineExecutionError(RuntimeError):
    """Raised when a pre-label attempt cannot be admitted or persisted exactly."""


def _canonical_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SealedOnlineExecutionError(
            "sealed online evidence must contain finite canonical JSON"
        ) from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SealedOnlineExecutionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise SealedOnlineExecutionError(f"{name} must be a canonical non-empty string")
    return value


def _closed_mapping(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SealedOnlineExecutionError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise SealedOnlineExecutionError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise SealedOnlineExecutionError(f"{label} contains duplicate key {key!r}")
            parsed[key] = value
        return parsed

    def reject_nonfinite(value: str) -> None:
        raise SealedOnlineExecutionError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedOnlineExecutionError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SealedOnlineExecutionError(f"{label} must contain one object")
    return value


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise SealedOnlineExecutionError(f"{name} must be a positive integer")
    return value


def _unsigned_seed(value: object) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise SealedOnlineExecutionError("permutation_seed must be an unsigned 64-bit integer")
    return value


def _output_root(value: str | Path) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path.anchor != "/"
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise SealedOnlineExecutionError("output_root must be a canonical absolute path")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SealedOnlineExecutionError(f"cannot inspect output_root: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise SealedOnlineExecutionError("output_root must be a real directory")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise SealedOnlineExecutionError("output_root must be owned by the runner")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise SealedOnlineExecutionError(
            "output_root cannot be writable by group or other identities"
        )
    return path


def _absolute_root(name: str, value: str | Path) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path.anchor != "/"
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise SealedOnlineExecutionError(f"{name} must be a canonical absolute path")
    return path


def _path_is_admitted_by_mount(path: Path, mount: RuntimeArtifactMount) -> bool:
    path_parts = PurePosixPath(str(path)).parts
    root_parts = PurePosixPath(mount.root).parts
    if mount.kind == "file":
        return path_parts == root_parts
    return len(path_parts) >= len(root_parts) and path_parts[: len(root_parts)] == root_parts


def _admit_production_runtime_attestation(
    *,
    plan_path: str | Path,
    expected_plan_sha256: str,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    run_receipt: SealedRunReceipt,
    source_paths: Mapping[str, str | Path],
) -> tuple[RuntimeAttestationPlan, RuntimeAttestationReceipt]:
    """Verify the frozen and live process contract before workload source I/O."""

    if not isinstance(run_receipt, SealedRunReceipt):
        raise SealedOnlineExecutionError("run_receipt must be typed")
    if not isinstance(source_paths, Mapping) or set(source_paths) != _RUNTIME_SOURCE_PATH_NAMES:
        raise SealedOnlineExecutionError("runtime attestation source-path closure differs")
    frozen_plan_digest = _require_sha256(
        "expected_runtime_attestation_plan_sha256",
        expected_plan_sha256,
    )
    frozen_receipt_digest = _require_sha256(
        "expected_runtime_attestation_receipt_sha256",
        expected_receipt_sha256,
    )
    plan_source = _absolute_root("runtime_attestation_plan_path", plan_path)
    receipt_source = _absolute_root("runtime_attestation_receipt_path", receipt_path)
    if plan_source == receipt_source:
        raise SealedOnlineExecutionError(
            "runtime attestation plan and receipt must be distinct files"
        )
    try:
        plan = load_runtime_attestation_plan(plan_source)
        receipt = load_runtime_attestation_receipt(receipt_source)
        if plan.plan_sha256 != frozen_plan_digest:
            raise SealedOnlineExecutionError("runtime attestation plan differs from its frozen pin")
        if receipt.receipt_sha256 != frozen_receipt_digest:
            raise SealedOnlineExecutionError(
                "runtime attestation receipt differs from its frozen pin"
            )
        verify_live_runtime_attestation(
            receipt,
            plan,
            probe=LinuxRuntimeProbe(),
        )
        marker_digest = digest_regular_file(
            receipt.invocation_marker_path,
            label="runtime one-shot invocation marker",
        )
    except RuntimeAttestationError as exc:
        raise SealedOnlineExecutionError(f"runtime attestation failed: {exc}") from exc
    except ArtifactIntegrityError as exc:
        raise SealedOnlineExecutionError(
            f"runtime one-shot invocation marker failed verification: {exc}"
        ) from exc
    if marker_digest != receipt.invocation_marker_sha256:
        raise SealedOnlineExecutionError(
            "runtime one-shot invocation marker differs from its receipt"
        )
    if (
        plan.manifest_sha256 != run_receipt.manifest_sha256
        or plan.runner_identity != run_receipt.runner_identity
        or plan.code_commit != run_receipt.code_commit
        or plan.oci_image_digest != run_receipt.runner_image
    ):
        raise SealedOnlineExecutionError("runtime attestation differs from the sealed run identity")
    for name in sorted(_RUNTIME_SOURCE_PATH_NAMES):
        source = _absolute_root(name, source_paths[name])
        admitted = [mount for mount in plan.mounts if _path_is_admitted_by_mount(source, mount)]
        if len(admitted) != 1:
            raise SealedOnlineExecutionError(
                f"{name} is not inside exactly one attested read-only mount"
            )
    return plan, receipt


def _pseudonym_key_id(expected_sha256: str) -> str:
    digest = _require_sha256("expected_pseudonym_key_sha256", expected_sha256)
    return f"sealed-online-ephemeral-sha256-{digest}"


def _load_pseudonym_key(
    path: str | Path,
    *,
    expected_sha256: str,
) -> bytes:
    target = _absolute_root("pseudonym_key_path", path)
    digest = _require_sha256("expected_pseudonym_key_sha256", expected_sha256)
    try:
        encoded = read_secure_regular_file(
            target,
            max_bytes=_PSEUDONYM_KEY_MAX_BYTES,
            label="sealed online pseudonym key",
        )
    except ArtifactIntegrityError as exc:
        raise SealedOnlineExecutionError(f"cannot load sealed online pseudonym key: {exc}") from exc
    if _sha256(encoded) != digest:
        raise SealedOnlineExecutionError("pseudonym key differs from its frozen pin")
    if len(encoded) < 32 or len(set(encoded)) < 8:
        raise SealedOnlineExecutionError("pseudonym key must contain at least 32 diverse bytes")
    return encoded


def _write(payload: bytes, target: Path, *, label: str) -> None:
    try:
        write_exclusive_receipt_bytes(payload, target)
    except ArtifactIntegrityError as exc:
        raise SealedOnlineExecutionError(f"cannot persist {label}: {exc}") from exc


class _BoundedOPAStderr:
    """Drain OPA stderr continuously while retaining only a fixed prefix."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._buffer = bytearray()
        self._total_bytes = 0
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._drain,
            name="fractal-confirmatory-opa-stderr",
            daemon=True,
        )
        self._thread.start()

    def _drain(self) -> None:
        try:
            while True:
                chunk = self._stream.read(8192)
                if not chunk:
                    return
                if not isinstance(chunk, bytes):
                    raise TypeError("OPA stderr stream returned non-bytes data")
                with self._lock:
                    self._total_bytes += len(chunk)
                    remaining = _PRODUCTION_OPA_STDERR_BYTES - len(self._buffer)
                    if remaining > 0:
                        self._buffer.extend(chunk[:remaining])
        except BaseException as exc:  # pragma: no cover - OS pipe faults are platform-specific
            with self._lock:
                self._error = exc

    def diagnostic(self) -> str:
        with self._lock:
            encoded = bytes(self._buffer)
            total = self._total_bytes
        text = encoded.decode("utf-8", errors="replace")
        text = " ".join(text.split())[:2048]
        suffix = " [truncated]" if total > len(encoded) else ""
        return f"{text}{suffix}" if text else "stderr was empty"

    def finish(self) -> None:
        self._thread.join(timeout=_PRODUCTION_OPA_STOP_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            try:
                self._stream.close()
            except OSError:
                pass
            self._thread.join(timeout=_PRODUCTION_OPA_STOP_TIMEOUT_SECONDS)
        try:
            self._stream.close()
        except OSError as exc:
            raise SealedOnlineExecutionError("cannot close the OPA stderr pipe") from exc
        if self._thread.is_alive():
            raise SealedOnlineExecutionError("OPA stderr drain did not terminate")
        with self._lock:
            error = self._error
        if error is not None:
            raise SealedOnlineExecutionError("OPA stderr drain failed") from error


@dataclass
class _ProductionOPAHandle:
    process: subprocess.Popen[bytes]
    stderr: _BoundedOPAStderr


def _assert_production_attempt_marker(
    attempt_path: Path,
    attempt: SealedOnlineAttemptReceipt,
) -> None:
    try:
        encoded = read_secure_control_file(
            attempt_path,
            label="sealed online attempt receipt",
        )
    except ArtifactIntegrityError as exc:
        raise SealedOnlineExecutionError(
            "OPA cannot start before the sealed online attempt is readable"
        ) from exc
    if encoded != attempt.canonical_bytes() + b"\n":
        raise SealedOnlineExecutionError(
            "OPA cannot start because the sealed online attempt changed bytes"
        )


def _verify_production_opa_image_artifacts() -> None:
    for path, label, executable in (
        (_PRODUCTION_OPA_BINARY, "OPA binary", True),
        (_PRODUCTION_OPA_REGO, "OPA compiled policy", False),
    ):
        try:
            metadata = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise SealedOnlineExecutionError(f"cannot inspect image-baked {label}") from exc
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or mode & 0o222
            or (executable and not mode & 0o111)
        ):
            raise SealedOnlineExecutionError(
                f"image-baked {label} must be a root-owned, immutable regular file"
            )
    try:
        rego_sha256 = digest_regular_file(
            _PRODUCTION_OPA_REGO,
            label="image-baked OPA compiled policy",
        )
    except ArtifactIntegrityError as exc:
        raise SealedOnlineExecutionError(
            "cannot verify the image-baked OPA compiled policy"
        ) from exc
    if rego_sha256 != _PRODUCTION_OPA_REGO_SHA256:
        raise SealedOnlineExecutionError(
            "image-baked OPA compiled policy differs from the runner revision"
        )


def _admit_production_opa_data(
    *,
    policy_root: Path,
    runtime_admission: TrialRuntimeAdmission,
    mask_store: CompiledPolicyMaskStore,
    expected_policy_receipt_sha256: str,
) -> OPACompiledMaskData:
    """Reconstruct the exact receipt-bound value served at data.fractal."""

    try:
        opa_data = load_opa_compiled_mask_data(policy_root / OPA_DATA_FILENAME)
        receipt = load_policy_intervention_receipt(policy_root / POLICY_RECEIPT_FILENAME)
    except PolicyInterventionError as exc:
        raise SealedOnlineExecutionError(f"cannot admit production OPA data: {exc}") from exc
    expected_receipt = _require_sha256(
        "expected_policy_receipt_sha256",
        expected_policy_receipt_sha256,
    )
    artifact_by_role = {row.role: row for row in receipt.artifacts}
    bound = artifact_by_role.get("opa-data")
    encoded = opa_data.canonical_file_bytes()
    if (
        receipt.artifact_sha256 != expected_receipt
        or bound is None
        or bound.path != OPA_DATA_FILENAME
        or bound.byte_count != len(encoded)
        or bound.sha256 != _sha256(encoded)
    ):
        raise SealedOnlineExecutionError(
            "OPA data differs from the frozen policy-intervention receipt"
        )
    catalog = mask_store.catalog
    if (
        opa_data.document_count != catalog.document_count
        or opa_data.document_universe_sha256 != catalog.document_universe_sha256
        or opa_data.mask_catalog_sha256 != mask_store.catalog_sha256
        or opa_data.policy_revision != catalog.policy_revision
    ):
        raise SealedOnlineExecutionError("OPA data differs from the compiled mask catalog")
    expected_assignments = tuple(
        sorted(
            (
                group.subject,
                group.policy_state,
                group.mask_id,
                group.mask_sha256,
                group.authorized_count,
            )
            for group in runtime_admission.receipt.groups
        )
    )
    observed_assignments = tuple(
        (
            row.subject,
            row.policy_state,
            row.mask_id,
            row.mask_sha256,
            row.authorized_count,
        )
        for row in opa_data.assignments
    )
    if observed_assignments != expected_assignments:
        raise SealedOnlineExecutionError("OPA assignments differ from the frozen runtime groups")
    return opa_data


def _production_opa_command(data_path: Path) -> tuple[str, ...]:
    source = _absolute_root("production OPA data path", data_path)
    return (
        str(_PRODUCTION_OPA_BINARY),
        "run",
        "--server",
        "--addr=127.0.0.1:8181",
        "--authentication=off",
        "--authorization=off",
        "--log-format=json",
        "--log-level=error",
        "--max-errors=1",
        "--ready-timeout=5",
        "--set=decision_logs.console=true",
        "--shutdown-grace-period=1",
        "--shutdown-wait-period=0",
        "--skip-version-check",
        str(_PRODUCTION_OPA_REGO),
        f"fractal:{source}",
    )


def _assert_production_opa_port_vacant() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(_PRODUCTION_OPA_HTTP_TIMEOUT_SECONDS)
        status = probe.connect_ex((_PRODUCTION_OPA_HOST, _PRODUCTION_OPA_PORT))
    finally:
        probe.close()
    if status == 0:
        raise SealedOnlineExecutionError(
            "the fixed production OPA loopback port is already occupied"
        )
    if status != errno.ECONNREFUSED:
        raise SealedOnlineExecutionError(
            f"cannot prove the fixed production OPA loopback port is vacant: errno {status}"
        )


def _production_opa_http_request(
    method: str,
    path: str,
    *,
    body: bytes | None = None,
) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection(
        _PRODUCTION_OPA_HOST,
        _PRODUCTION_OPA_PORT,
        timeout=_PRODUCTION_OPA_HTTP_TIMEOUT_SECONDS,
    )
    headers = {"Accept": "application/json", "Connection": "close"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        encoded = response.read(_PRODUCTION_OPA_HTTP_BYTES + 1)
        if len(encoded) > _PRODUCTION_OPA_HTTP_BYTES:
            raise SealedOnlineExecutionError("OPA readiness response exceeds the byte limit")
        return response.status, encoded
    finally:
        connection.close()


def _production_opa_health_ready() -> bool:
    try:
        status, encoded = _production_opa_http_request(
            "GET",
            _PRODUCTION_OPA_HEALTH_PATH,
        )
        payload = json.loads(encoded.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealedOnlineExecutionError):
        return False
    return status == 200 and payload == {}


def _production_opa_decision_ready(opa_data: OPACompiledMaskData) -> bool:
    assignment = opa_data.assignments[0]
    readiness_sha256 = _sha256(b"fractal-confirmatory-opa-readiness-v1")
    decision_input: dict[str, object] = {
        "action": PRODUCTION_POLICY_ACTION,
        "catalog_request_sha256": readiness_sha256,
        "document_count": opa_data.document_count,
        "document_universe_sha256": opa_data.document_universe_sha256,
        "environment": {"policy_state": assignment.policy_state},
        "environment_sha256": readiness_sha256,
        "mask_catalog_sha256": opa_data.mask_catalog_sha256,
        "policy_revision": opa_data.policy_revision,
        "request_nonce": "fractal-confirmatory-opa-readiness-v1",
        "request_sha256": readiness_sha256,
        "subject": assignment.subject,
    }
    expected_result = {key: value for key, value in decision_input.items() if key != "environment"}
    expected_result.update(
        {
            "authorized_count": assignment.authorized_count,
            "mask_id": assignment.mask_id,
            "mask_sha256": assignment.mask_sha256,
        }
    )
    try:
        status, encoded = _production_opa_http_request(
            "POST",
            _PRODUCTION_OPA_DECISION_PATH,
            body=_canonical_bytes({"input": decision_input}),
        )
        payload = _closed_mapping(
            _decode_object(encoded, label="OPA readiness decision"),
            fields={"decision_id", "result"},
            label="OPA readiness decision",
        )
        _require_text("OPA readiness decision_id", payload["decision_id"])
    except (OSError, SealedOnlineExecutionError):
        return False
    return status == 200 and payload["result"] == expected_result


def _opa_diagnostic(handle: _ProductionOPAHandle) -> str:
    return handle.stderr.diagnostic()


def _wait_for_production_opa(
    handle: _ProductionOPAHandle,
    opa_data: OPACompiledMaskData,
) -> None:
    deadline = time.monotonic() + _PRODUCTION_OPA_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        returncode = handle.process.poll()
        if returncode is not None:
            raise SealedOnlineExecutionError(
                f"OPA exited during startup with status {returncode}; {_opa_diagnostic(handle)}"
            )
        if (
            _production_opa_health_ready()
            and _production_opa_decision_ready(opa_data)
            and handle.process.poll() is None
        ):
            return
        time.sleep(_PRODUCTION_OPA_POLL_SECONDS)
    raise SealedOnlineExecutionError(
        f"OPA did not pass health and decision readiness; {_opa_diagnostic(handle)}"
    )


def _stop_production_opa(handle: _ProductionOPAHandle) -> None:
    failures: list[str] = []
    if handle.process.poll() is None:
        try:
            handle.process.terminate()
            handle.process.wait(timeout=_PRODUCTION_OPA_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                handle.process.kill()
                handle.process.wait(timeout=_PRODUCTION_OPA_STOP_TIMEOUT_SECONDS)
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append(f"forced OPA termination failed: {exc}")
        except OSError as exc:
            failures.append(f"OPA termination failed: {exc}")
    try:
        handle.stderr.finish()
    except SealedOnlineExecutionError as exc:
        failures.append(str(exc))
    if handle.process.poll() is None:
        failures.append("OPA remained alive after cleanup")
    if failures:
        raise SealedOnlineExecutionError("; ".join(failures))


def _start_production_opa(
    data_path: Path,
    opa_data: OPACompiledMaskData,
) -> _ProductionOPAHandle:
    _verify_production_opa_image_artifacts()
    _assert_production_opa_port_vacant()
    try:
        process = subprocess.Popen(
            _production_opa_command(data_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd="/",
            env={
                "HOME": "/home/runner",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "TZ": "UTC",
            },
            bufsize=0,
            close_fds=True,
            restore_signals=True,
            start_new_session=True,
            shell=False,
            text=False,
            umask=0o077,
        )
    except OSError as exc:
        raise SealedOnlineExecutionError("cannot launch the image-baked OPA binary") from exc
    if process.stderr is None:  # pragma: no cover - subprocess.PIPE guarantees this
        process.kill()
        process.wait(timeout=_PRODUCTION_OPA_STOP_TIMEOUT_SECONDS)
        raise SealedOnlineExecutionError("OPA stderr pipe was not created")
    handle = _ProductionOPAHandle(
        process=process,
        stderr=_BoundedOPAStderr(process.stderr),
    )
    try:
        _wait_for_production_opa(handle, opa_data)
    except BaseException as exc:
        try:
            _stop_production_opa(handle)
        except SealedOnlineExecutionError as cleanup_exc:
            raise cleanup_exc from exc
        raise
    return handle


@contextmanager
def _production_opa_sidecar(
    *,
    attempt_path: Path,
    attempt: SealedOnlineAttemptReceipt,
    policy_root: Path,
    runtime_admission: TrialRuntimeAdmission,
    mask_store: CompiledPolicyMaskStore,
    expected_policy_receipt_sha256: str,
) -> Iterator[OpenPolicyAgentMaskDecisionPoint]:
    """Own one OPA child strictly inside the already-consumed attempt."""

    _assert_production_attempt_marker(attempt_path, attempt)
    opa_data = _admit_production_opa_data(
        policy_root=policy_root,
        runtime_admission=runtime_admission,
        mask_store=mask_store,
        expected_policy_receipt_sha256=expected_policy_receipt_sha256,
    )
    scratch: tempfile.TemporaryDirectory[str] | None = None
    handle: _ProductionOPAHandle | None = None
    body_error: BaseException | None = None
    try:
        scratch = tempfile.TemporaryDirectory(
            prefix="fractal-confirmatory-opa-",
            dir=_PRODUCTION_OPA_SCRATCH_ROOT,
            ignore_cleanup_errors=False,
        )
        scratch_root = Path(scratch.name)
        metadata = scratch_root.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise SealedOnlineExecutionError("OPA scratch directory is not private")
        data_path = scratch_root / OPA_DATA_FILENAME
        _write(
            opa_data.canonical_file_bytes(),
            data_path,
            label="private production OPA data copy",
        )
        handle = _start_production_opa(data_path, opa_data)
        yield OpenPolicyAgentMaskDecisionPoint(
            PRODUCTION_OPA_ENDPOINT,
            mask_store,
        )
        returncode = handle.process.poll()
        if returncode is not None:
            raise SealedOnlineExecutionError(
                f"OPA exited before the sealed workload closed with status {returncode}; "
                f"{_opa_diagnostic(handle)}"
            )
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        cleanup_failures: list[str] = []
        if handle is not None:
            try:
                _stop_production_opa(handle)
            except SealedOnlineExecutionError as exc:
                cleanup_failures.append(str(exc))
        if scratch is not None:
            try:
                scratch.cleanup()
            except OSError as exc:
                cleanup_failures.append(f"OPA scratch cleanup failed: {exc}")
        if cleanup_failures:
            raise SealedOnlineExecutionError("; ".join(cleanup_failures)) from body_error


def _controller_config_sha256(config: ControllerConfig) -> str:
    return _sha256(
        _canonical_bytes(
            {
                "exact_scan_threshold": config.exact_scan_threshold,
                "exact_threshold": config.exact_threshold,
                "high_ef": config.high_ef,
                "high_effort_threshold": config.high_effort_threshold,
                "low_ef": config.low_ef,
                "probe_k": config.probe_k,
                "schema_version": "fractal-controller-config-binding-v1",
            }
        )
    )


def _query_feature_sources_sha256(
    runtime_receipt: TrialRuntimeAdmissionReceipt,
    *,
    policy_intervention_receipt_sha256: str,
) -> str:
    """Bind every frozen source used by the two query-level control features."""

    if not isinstance(runtime_receipt, TrialRuntimeAdmissionReceipt):
        raise SealedOnlineExecutionError("runtime receipt must be typed")
    return _sha256(
        _canonical_bytes(
            {
                "active_query_epoch": runtime_receipt.active_query_epoch.to_dict(),
                "current_truth_query_epoch": (runtime_receipt.current_truth_query_epoch.to_dict()),
                "embedding_store_receipt_sha256": (runtime_receipt.embedding_store_receipt_sha256),
                "policy_intervention_receipt_sha256": _require_sha256(
                    "policy_intervention_receipt_sha256",
                    policy_intervention_receipt_sha256,
                ),
                "query_trial_store_sha256": runtime_receipt.query_trial_store_sha256,
                "schedule_sha256": runtime_receipt.schedule_sha256,
                "schema_version": "fractal-query-control-feature-source-binding-v1",
            }
        )
    )


def _required_artifact_bindings_sha256(bindings: RequiredArtifactIdBindings) -> str:
    return _sha256(
        _canonical_bytes(
            {
                "execution_artifact_id": bindings.execution_artifact_id,
                "execution_revision_sha256": bindings.execution_revision_sha256,
                "provenance_component_artifact_ids": [
                    list(item) for item in bindings.provenance_component_artifact_ids
                ],
                "retriever_artifact_ids": list(bindings.retriever_artifact_ids),
                "runner_artifact_ids": list(bindings.runner_artifact_ids),
                "source_artifact_ids": list(bindings.source_artifact_ids),
                "verification_receipt_sha256": (bindings.verification_receipt.receipt_sha256),
            }
        )
    )


@dataclass(frozen=True)
class SealedOnlineAttemptReceipt:
    """Durable evidence created immediately before the first governed request."""

    manifest_sha256: str
    run_receipt_sha256: str
    online_custody_admission_receipt_sha256: str
    required_artifact_bindings_sha256: str
    runtime_attestation_plan_sha256: str
    runtime_attestation_receipt_sha256: str
    runtime_claim_receipt_sha256: str
    claim_state_sha256: str
    claim_ledger_commit: str
    provider_identity_sha256: str
    beacon_receipt_sha256: str
    beacon_bytes_sha256: str
    derived_seed_sha256: str
    output_aggregate_identity: str
    trial_runtime_admission_receipt_sha256: str
    authorized_index_store_receipt_sha256: str
    execution_artifact_sha256: str
    query_partition_audit_sha256: str
    controller_config_sha256: str
    query_feature_sources_sha256: str
    policy_revision: str
    permutation_seed: int
    k: int
    policy_action: str
    partition_label: Literal["primary", "reserve"]
    pseudonym_key_id: str
    runner_identity: str
    result_directory_uri: str
    schema_version: str = ONLINE_ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "online_custody_admission_receipt_sha256",
            "required_artifact_bindings_sha256",
            "runtime_attestation_plan_sha256",
            "runtime_attestation_receipt_sha256",
            "runtime_claim_receipt_sha256",
            "claim_state_sha256",
            "provider_identity_sha256",
            "beacon_receipt_sha256",
            "beacon_bytes_sha256",
            "derived_seed_sha256",
            "output_aggregate_identity",
            "trial_runtime_admission_receipt_sha256",
            "authorized_index_store_receipt_sha256",
            "execution_artifact_sha256",
            "query_partition_audit_sha256",
            "controller_config_sha256",
            "query_feature_sources_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (self.runtime_attestation_plan_sha256 == _UNATTESTED_RUNTIME_SHA256) != (
            self.runtime_attestation_receipt_sha256 == _UNATTESTED_RUNTIME_SHA256
        ):
            raise SealedOnlineExecutionError(
                "runtime attestation plan and receipt must be jointly bound"
            )
        for name in (
            "policy_revision",
            "policy_action",
            "pseudonym_key_id",
            "runner_identity",
            "result_directory_uri",
        ):
            _require_text(name, getattr(self, name))
        _unsigned_seed(self.permutation_seed)
        if (
            type(self.claim_ledger_commit) is not str
            or re.fullmatch(r"[0-9a-f]{40}", self.claim_ledger_commit) is None
        ):
            raise SealedOnlineExecutionError("claim_ledger_commit must be one Git commit")
        if self.runtime_claim_receipt_sha256 != _UNATTESTED_RUNTIME_SHA256 and (
            self.permutation_seed
            != int.from_bytes(bytes.fromhex(self.derived_seed_sha256)[:8], "big")
        ):
            raise SealedOnlineExecutionError("permutation seed differs from execution beacon")
        _positive_integer("k", self.k)
        if self.partition_label not in {"primary", "reserve"}:
            raise SealedOnlineExecutionError("partition_label is not registered")
        if self.schema_version != ONLINE_ATTEMPT_SCHEMA:
            raise SealedOnlineExecutionError("online attempt schema differs")
        parsed = urlsplit(self.result_directory_uri)
        try:
            result_path = Path(unquote(parsed.path, errors="strict"))
        except UnicodeDecodeError as exc:
            raise SealedOnlineExecutionError(
                "result_directory_uri contains invalid UTF-8 escaping"
            ) from exc
        if (
            parsed.scheme != "file"
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or not result_path.is_absolute()
            or result_path.as_uri() != self.result_directory_uri
        ):
            raise SealedOnlineExecutionError("result_directory_uri must be a canonical file URI")

    def to_dict(self) -> dict[str, object]:
        return {
            "authorized_index_store_receipt_sha256": (self.authorized_index_store_receipt_sha256),
            "controller_config_sha256": self.controller_config_sha256,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "k": self.k,
            "manifest_sha256": self.manifest_sha256,
            "online_custody_admission_receipt_sha256": (
                self.online_custody_admission_receipt_sha256
            ),
            "partition_label": self.partition_label,
            "permutation_seed": self.permutation_seed,
            "policy_action": self.policy_action,
            "policy_revision": self.policy_revision,
            "pseudonym_key_id": self.pseudonym_key_id,
            "query_partition_audit_sha256": self.query_partition_audit_sha256,
            "query_feature_sources_sha256": self.query_feature_sources_sha256,
            "required_artifact_bindings_sha256": self.required_artifact_bindings_sha256,
            "result_directory_uri": self.result_directory_uri,
            "run_receipt_sha256": self.run_receipt_sha256,
            "runtime_attestation_plan_sha256": self.runtime_attestation_plan_sha256,
            "runtime_attestation_receipt_sha256": self.runtime_attestation_receipt_sha256,
            "runtime_claim_receipt_sha256": self.runtime_claim_receipt_sha256,
            "claim_state_sha256": self.claim_state_sha256,
            "claim_ledger_commit": self.claim_ledger_commit,
            "provider_identity_sha256": self.provider_identity_sha256,
            "beacon_receipt_sha256": self.beacon_receipt_sha256,
            "beacon_bytes_sha256": self.beacon_bytes_sha256,
            "derived_seed_sha256": self.derived_seed_sha256,
            "output_aggregate_identity": self.output_aggregate_identity,
            "runner_identity": self.runner_identity,
            "schema_version": self.schema_version,
            "trial_runtime_admission_receipt_sha256": (self.trial_runtime_admission_receipt_sha256),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> SealedOnlineAttemptReceipt:
        fields = set(cls.__dataclass_fields__)
        row = _closed_mapping(value, fields=fields, label="sealed online attempt receipt")
        return cls(**row)


@dataclass(frozen=True)
class OnlineOutputPin:
    """Exact file bytes and semantic digest for one pre-label output."""

    role: str
    filename: str
    byte_count: int
    file_sha256: str
    semantic_sha256: str
    schema_version: str = ONLINE_OUTPUT_PIN_SCHEMA

    def __post_init__(self) -> None:
        if self.role not in _OUTPUT_SUFFIXES:
            raise SealedOnlineExecutionError("online output role is not registered")
        _require_text("filename", self.filename)
        if Path(self.filename).name != self.filename:
            raise SealedOnlineExecutionError("online output filename must be local")
        _positive_integer("byte_count", self.byte_count)
        _require_sha256("file_sha256", self.file_sha256)
        _require_sha256("semantic_sha256", self.semantic_sha256)
        if self.schema_version != ONLINE_OUTPUT_PIN_SCHEMA:
            raise SealedOnlineExecutionError("online output pin schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "file_sha256": self.file_sha256,
            "filename": self.filename,
            "role": self.role,
            "schema_version": self.schema_version,
            "semantic_sha256": self.semantic_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> OnlineOutputPin:
        fields = set(cls.__dataclass_fields__)
        row = _closed_mapping(value, fields=fields, label="online output pin")
        return cls(**row)


@dataclass(frozen=True)
class SealedOnlineResultReceipt:
    """Exact closure over every output that must precede label release."""

    manifest_sha256: str
    run_receipt_sha256: str
    execution_artifact_sha256: str
    attempt_receipt_sha256: str
    audit_head_sha256: str
    audit_record_count: int
    outputs: tuple[OnlineOutputPin, ...]
    schema_version: str = ONLINE_RESULT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "execution_artifact_sha256",
            "attempt_receipt_sha256",
            "audit_head_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _positive_integer("audit_record_count", self.audit_record_count)
        outputs = tuple(self.outputs)
        if not outputs or not all(isinstance(row, OnlineOutputPin) for row in outputs):
            raise SealedOnlineExecutionError("outputs must contain online output pins")
        canonical = tuple(sorted(outputs, key=lambda row: row.role.encode("utf-8")))
        if outputs != canonical or {row.role for row in outputs} != set(_OUTPUT_SUFFIXES):
            raise SealedOnlineExecutionError("outputs must cover each registered role exactly once")
        filenames = [row.filename for row in outputs]
        if len(filenames) != len(set(filenames)):
            raise SealedOnlineExecutionError("online outputs repeat a filename")
        if self.schema_version != ONLINE_RESULT_RECEIPT_SCHEMA:
            raise SealedOnlineExecutionError("online result receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_receipt_sha256": self.attempt_receipt_sha256,
            "audit_head_sha256": self.audit_head_sha256,
            "audit_record_count": self.audit_record_count,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
            "outputs": [row.to_dict() for row in self.outputs],
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> SealedOnlineResultReceipt:
        row = _closed_mapping(
            value,
            fields={
                "attempt_receipt_sha256",
                "audit_head_sha256",
                "audit_record_count",
                "execution_artifact_sha256",
                "manifest_sha256",
                "outputs",
                "run_receipt_sha256",
                "schema_version",
            },
            label="sealed online result receipt",
        )
        outputs = row["outputs"]
        if not isinstance(outputs, list):
            raise SealedOnlineExecutionError("online result outputs must be an array")
        return cls(
            manifest_sha256=row["manifest_sha256"],
            run_receipt_sha256=row["run_receipt_sha256"],
            execution_artifact_sha256=row["execution_artifact_sha256"],
            attempt_receipt_sha256=row["attempt_receipt_sha256"],
            audit_head_sha256=row["audit_head_sha256"],
            audit_record_count=row["audit_record_count"],
            outputs=tuple(OnlineOutputPin.from_dict(item) for item in outputs),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class PersistedSealedOnlineRun:
    """Typed return value for a completed and fully persisted online attempt."""

    artifacts: OnlineRunArtifacts
    predictions: PredictionArtifact
    attempt_receipt: SealedOnlineAttemptReceipt
    result_receipt: SealedOnlineResultReceipt
    output_root: Path


def sealed_online_attempt_path(output_root: str | Path, manifest_sha256: str) -> Path:
    digest = _require_sha256("manifest_sha256", manifest_sha256)
    return _output_root(output_root) / f"{digest}{_ATTEMPT_SUFFIX}"


def sealed_online_result_path(output_root: str | Path, manifest_sha256: str) -> Path:
    digest = _require_sha256("manifest_sha256", manifest_sha256)
    return _output_root(output_root) / f"{digest}{_RESULT_SUFFIX}"


def _output_path(root: Path, manifest_sha256: str, role: str) -> Path:
    try:
        suffix = _OUTPUT_SUFFIXES[role]
    except KeyError as exc:
        raise SealedOnlineExecutionError("online output role is not registered") from exc
    return root / f"{manifest_sha256}{suffix}"


def _attempt_receipt(
    *,
    root: Path,
    admission_receipt: OnlineCustodyAdmissionReceipt,
    required_artifacts: RequiredArtifactIdBindings,
    execution: object,
    run_receipt: SealedRunReceipt,
    retriever: GovernedRetriever,
    runtime_receipt: TrialRuntimeAdmissionReceipt,
    index_receipt: AuthorizedIndexStoreReceipt,
    permutation_seed: int,
    expected_policy_version: str,
    query_partition_audit_sha256: str,
    pseudonym_key_id: str,
    k: int,
    policy_action: str,
    partition_label: Literal["primary", "reserve"],
) -> SealedOnlineAttemptReceipt:
    if not isinstance(admission_receipt, OnlineCustodyAdmissionReceipt):
        raise SealedOnlineExecutionError("admission_receipt must be typed")
    if not isinstance(required_artifacts, RequiredArtifactIdBindings):
        raise SealedOnlineExecutionError("required_artifacts must be typed")
    if not isinstance(run_receipt, SealedRunReceipt):
        raise SealedOnlineExecutionError("run_receipt must be typed")
    if not isinstance(retriever, GovernedRetriever):
        raise SealedOnlineExecutionError("retriever must be typed")
    if not isinstance(retriever.controller, RuleController):
        raise SealedOnlineExecutionError("retriever controller must be the frozen rule controller")
    if not isinstance(retriever.policy, OpenPolicyAgentMaskDecisionPoint):
        raise SealedOnlineExecutionError("confirmatory retrieval requires the admitted OPA PDP")
    if not isinstance(runtime_receipt, TrialRuntimeAdmissionReceipt):
        raise SealedOnlineExecutionError("runtime_receipt must be typed")
    if not isinstance(index_receipt, AuthorizedIndexStoreReceipt):
        raise SealedOnlineExecutionError("index_receipt must be typed")
    compatibility = execution_compatibility_view(execution)
    if (
        runtime_receipt.execution_artifact_sha256 != compatibility.artifact_sha256
        or runtime_receipt.query_count != len(compatibility.trial_keys)
        or index_receipt.policy_execution_artifact_sha256 != compatibility.artifact_sha256
        or index_receipt.document_count != compatibility.document_count
        or index_receipt.document_universe_sha256 != compatibility.document_universe_sha256
        or index_receipt.embedding_receipt_sha256 != runtime_receipt.embedding_store_receipt_sha256
        or index_receipt.policy_catalog_sha256 != runtime_receipt.mask_catalog_sha256
        or index_receipt.policy_revision != runtime_receipt.policy_bundle_revision
        or expected_policy_version != runtime_receipt.policy_bundle_revision
    ):
        raise SealedOnlineExecutionError(
            "execution, runtime, index, embedding, or policy bindings differ"
        )
    if (
        len(retriever.vectors) != compatibility.document_count
        or retriever.document_universe_sha256 != compatibility.document_universe_sha256
        or retriever.policy.document_universe_sha256 != compatibility.document_universe_sha256
        or retriever.policy.mask_store.catalog_sha256 != runtime_receipt.mask_catalog_sha256
    ):
        raise SealedOnlineExecutionError("live retriever differs from the admitted runtime")
    provider = retriever.authorized_hnsw_provider
    if provider is None or getattr(provider, "receipt", None) != index_receipt:
        raise SealedOnlineExecutionError("live retriever lacks the exact verified index store")
    run_receipt_sha256 = run_receipt.binding_sha256
    if (
        run_receipt.manifest_sha256 != admission_receipt.manifest_sha256
        or run_receipt_sha256 != admission_receipt.run_receipt_sha256
        or run_receipt.runner_identity != admission_receipt.runner_identity
    ):
        raise SealedOnlineExecutionError("run and custody admission receipts differ")
    return SealedOnlineAttemptReceipt(
        manifest_sha256=run_receipt.manifest_sha256,
        run_receipt_sha256=run_receipt_sha256,
        online_custody_admission_receipt_sha256=admission_receipt.receipt_sha256,
        required_artifact_bindings_sha256=(_required_artifact_bindings_sha256(required_artifacts)),
        runtime_attestation_plan_sha256=_UNATTESTED_RUNTIME_SHA256,
        runtime_attestation_receipt_sha256=_UNATTESTED_RUNTIME_SHA256,
        runtime_claim_receipt_sha256=_UNATTESTED_RUNTIME_SHA256,
        claim_state_sha256=_UNATTESTED_RUNTIME_SHA256,
        claim_ledger_commit="0" * 40,
        provider_identity_sha256=_UNATTESTED_RUNTIME_SHA256,
        beacon_receipt_sha256=_UNATTESTED_RUNTIME_SHA256,
        beacon_bytes_sha256=_UNATTESTED_RUNTIME_SHA256,
        derived_seed_sha256=_UNATTESTED_RUNTIME_SHA256,
        output_aggregate_identity=_UNATTESTED_RUNTIME_SHA256,
        trial_runtime_admission_receipt_sha256=runtime_receipt.receipt_sha256,
        authorized_index_store_receipt_sha256=index_receipt.artifact_sha256,
        execution_artifact_sha256=compatibility.artifact_sha256,
        query_partition_audit_sha256=_require_sha256(
            "query_partition_audit_sha256", query_partition_audit_sha256
        ),
        controller_config_sha256=_controller_config_sha256(retriever.controller.config),
        query_feature_sources_sha256=_query_feature_sources_sha256(
            runtime_receipt,
            policy_intervention_receipt_sha256=index_receipt.policy_receipt_sha256,
        ),
        policy_revision=_require_text("expected_policy_version", expected_policy_version),
        permutation_seed=_unsigned_seed(permutation_seed),
        k=_positive_integer("k", k),
        policy_action=_require_text("policy_action", policy_action),
        partition_label=partition_label,
        pseudonym_key_id=_require_text("pseudonym_key_id", pseudonym_key_id),
        runner_identity=run_receipt.runner_identity,
        result_directory_uri=root.as_uri(),
    )


def _production_attempt_receipt(
    *,
    root: Path,
    admission_receipt: OnlineCustodyAdmissionReceipt,
    required_artifacts: RequiredArtifactIdBindings,
    run_receipt: SealedRunReceipt,
    runtime_admission: TrialRuntimeAdmission,
    runtime_attestation_plan: RuntimeAttestationPlan,
    runtime_attestation_receipt: RuntimeAttestationReceipt,
    expected_runtime_receipt_sha256: str,
    expected_authorized_index_store_receipt_sha256: str,
    expected_policy_intervention_receipt_sha256: str,
    pseudonym_key_id: str,
    runtime_claim_receipt: RuntimeClaimReceipt,
) -> SealedOnlineAttemptReceipt:
    """Create the attempt token without reading any runtime or vector source."""

    if not isinstance(admission_receipt, OnlineCustodyAdmissionReceipt):
        raise SealedOnlineExecutionError("admission_receipt must be typed")
    if not isinstance(required_artifacts, RequiredArtifactIdBindings):
        raise SealedOnlineExecutionError("required_artifacts must be typed")
    if not isinstance(run_receipt, SealedRunReceipt):
        raise SealedOnlineExecutionError("run_receipt must be typed")
    if not isinstance(runtime_admission, TrialRuntimeAdmission):
        raise SealedOnlineExecutionError("runtime_admission must be typed")
    if not isinstance(runtime_attestation_plan, RuntimeAttestationPlan):
        raise SealedOnlineExecutionError("runtime attestation plan must be typed")
    if not isinstance(runtime_attestation_receipt, RuntimeAttestationReceipt):
        raise SealedOnlineExecutionError("runtime attestation receipt must be typed")
    if not isinstance(runtime_claim_receipt, RuntimeClaimReceipt):
        raise SealedOnlineExecutionError("production runtime claim receipt must be typed")
    if (
        runtime_claim_receipt.manifest_sha256 != run_receipt.manifest_sha256
        or runtime_claim_receipt.run_receipt_sha256 != run_receipt.binding_sha256
    ):
        raise SealedOnlineExecutionError("runtime claim belongs to another sealed run")
    if (
        runtime_attestation_plan.plan_sha256 == _UNATTESTED_RUNTIME_SHA256
        or runtime_attestation_receipt.receipt_sha256 == _UNATTESTED_RUNTIME_SHA256
    ):
        raise SealedOnlineExecutionError(
            "production attempt cannot use the unattested runtime sentinel"
        )
    if (
        runtime_attestation_receipt.plan_sha256 != runtime_attestation_plan.plan_sha256
        or runtime_attestation_receipt.manifest_sha256 != run_receipt.manifest_sha256
        or runtime_attestation_receipt.runner_identity != run_receipt.runner_identity
        or runtime_attestation_receipt.code_commit != run_receipt.code_commit
        or runtime_attestation_receipt.oci_image_digest != run_receipt.runner_image
    ):
        raise SealedOnlineExecutionError(
            "runtime attestation receipt differs from the admitted plan or run"
        )
    plan = runtime_admission.plan
    runtime_receipt = runtime_admission.receipt
    if not isinstance(runtime_receipt, TrialRuntimeAdmissionReceipt):
        raise SealedOnlineExecutionError("runtime admission receipt must be typed")
    frozen_runtime_digest = _require_sha256(
        "expected_runtime_receipt_sha256",
        expected_runtime_receipt_sha256,
    )
    frozen_index_digest = _require_sha256(
        "expected_authorized_index_store_receipt_sha256",
        expected_authorized_index_store_receipt_sha256,
    )
    frozen_policy_digest = _require_sha256(
        "expected_policy_intervention_receipt_sha256",
        expected_policy_intervention_receipt_sha256,
    )
    compatibility = execution_compatibility_view(plan)
    if (
        runtime_receipt.receipt_sha256 != frozen_runtime_digest
        or runtime_receipt.execution_artifact_sha256 != compatibility.artifact_sha256
        or runtime_receipt.query_count != len(compatibility.trial_keys)
        or runtime_receipt.query_partition_audit_sha256 != plan.query_partition_audit_sha256
        or runtime_receipt.permutation_seed != plan.permutation_seed
    ):
        raise SealedOnlineExecutionError(
            "runtime admission differs from its frozen plan or receipt pin"
        )
    run_receipt_sha256 = run_receipt.binding_sha256
    if (
        run_receipt.manifest_sha256 != admission_receipt.manifest_sha256
        or run_receipt_sha256 != admission_receipt.run_receipt_sha256
        or run_receipt.runner_identity != admission_receipt.runner_identity
    ):
        raise SealedOnlineExecutionError("run and custody admission receipts differ")
    controller_config = ControllerConfig()
    return SealedOnlineAttemptReceipt(
        manifest_sha256=run_receipt.manifest_sha256,
        run_receipt_sha256=run_receipt_sha256,
        online_custody_admission_receipt_sha256=admission_receipt.receipt_sha256,
        required_artifact_bindings_sha256=(_required_artifact_bindings_sha256(required_artifacts)),
        runtime_attestation_plan_sha256=runtime_attestation_plan.plan_sha256,
        runtime_attestation_receipt_sha256=runtime_attestation_receipt.receipt_sha256,
        runtime_claim_receipt_sha256=runtime_claim_receipt.receipt_sha256,
        claim_state_sha256=runtime_claim_receipt.claim_state_sha256,
        claim_ledger_commit=runtime_claim_receipt.claim_ledger_commit,
        provider_identity_sha256=runtime_claim_receipt.provider_identity_sha256,
        beacon_receipt_sha256=runtime_claim_receipt.beacon_receipt_sha256,
        beacon_bytes_sha256=runtime_claim_receipt.beacon_bytes_sha256,
        derived_seed_sha256=runtime_claim_receipt.derived_seed_sha256,
        output_aggregate_identity=runtime_claim_receipt.output_aggregate_identity,
        trial_runtime_admission_receipt_sha256=frozen_runtime_digest,
        authorized_index_store_receipt_sha256=frozen_index_digest,
        execution_artifact_sha256=compatibility.artifact_sha256,
        query_partition_audit_sha256=runtime_receipt.query_partition_audit_sha256,
        controller_config_sha256=_controller_config_sha256(controller_config),
        query_feature_sources_sha256=_query_feature_sources_sha256(
            runtime_receipt,
            policy_intervention_receipt_sha256=frozen_policy_digest,
        ),
        policy_revision=runtime_receipt.policy_bundle_revision,
        permutation_seed=runtime_claim_receipt.permutation_seed,
        k=PRODUCTION_K,
        policy_action=PRODUCTION_POLICY_ACTION,
        partition_label=PRODUCTION_PARTITION_LABEL,
        pseudonym_key_id=_require_text("pseudonym_key_id", pseudonym_key_id),
        runner_identity=run_receipt.runner_identity,
        result_directory_uri=root.as_uri(),
    )


def _verify_production_source_bindings(
    *,
    runtime_admission: TrialRuntimeAdmission,
    index_receipt: AuthorizedIndexStoreReceipt,
    expected_authorized_index_store_receipt_sha256: str,
) -> None:
    plan = runtime_admission.plan
    receipt = runtime_admission.receipt
    compatibility = execution_compatibility_view(plan)
    if index_receipt.artifact_sha256 != expected_authorized_index_store_receipt_sha256:
        raise SealedOnlineExecutionError("authorized index receipt differs from its frozen pin")
    if (
        index_receipt.policy_execution_artifact_sha256 != compatibility.artifact_sha256
        or index_receipt.document_count != compatibility.document_count
        or index_receipt.document_universe_sha256 != compatibility.document_universe_sha256
        or index_receipt.embedding_receipt_sha256 != receipt.embedding_store_receipt_sha256
        or index_receipt.policy_catalog_sha256 != receipt.mask_catalog_sha256
        or index_receipt.policy_revision != receipt.policy_bundle_revision
        or index_receipt.old_active_vector.file_sha256 != plan.active_vector_store.artifact.sha256
        or index_receipt.current_truth_vector.file_sha256
        != plan.current_truth_vector_store.artifact.sha256
        or index_receipt.old_active_vector.shape != plan.active_vector_store.shape
        or index_receipt.current_truth_vector.shape != plan.current_truth_vector_store.shape
    ):
        raise SealedOnlineExecutionError(
            "plan, runtime, document epochs, index store, or policy binding differs"
        )


def _production_policy_transitions(
    *,
    runtime_admission: TrialRuntimeAdmission,
    policy_root: Path,
    expected_policy_receipt_sha256: str,
    mask_store: CompiledPolicyMaskStore,
) -> dict[str, PolicyTransitionEvidence]:
    """Derive every policy feature from complete frozen baseline/current masks."""

    try:
        config = load_policy_intervention_config(policy_root / POLICY_CONFIG_FILENAME)
        schedule = load_canonical_trial_schedule(policy_root / POLICY_SCHEDULE_FILENAME)
        receipt = load_policy_intervention_receipt(policy_root / POLICY_RECEIPT_FILENAME)
    except PolicyInterventionError as exc:
        raise SealedOnlineExecutionError(
            f"cannot load query-level policy transition sources: {exc}"
        ) from exc
    expected_receipt = _require_sha256(
        "expected_policy_receipt_sha256",
        expected_policy_receipt_sha256,
    )
    runtime_receipt = runtime_admission.receipt
    if (
        receipt.artifact_sha256 != expected_receipt
        or receipt.config_sha256 != config.config_sha256
        or receipt.seed_sha256 != config.seed_sha256
        or receipt.baseline_seed_sha256 != config.baseline_seed_sha256
        or receipt.policy_bundle_revision != config.policy_bundle_revision
        or receipt.baseline_policy_revision != config.baseline_policy_revision
        or schedule.artifact_sha256 != runtime_receipt.schedule_sha256
        or schedule.config_sha256 != config.config_sha256
        or schedule.assignment_seed_sha256 != config.seed_sha256
        or schedule.baseline_seed_sha256 != config.baseline_seed_sha256
        or schedule.policy_bundle_revision != config.policy_bundle_revision
        or schedule.baseline_policy_revision != config.baseline_policy_revision
        or schedule.execution_artifact_sha256 != runtime_receipt.execution_artifact_sha256
        or schedule.document_universe_sha256 != runtime_admission.plan.document_universe_sha256
        or schedule.document_count != runtime_admission.plan.document_count
    ):
        raise SealedOnlineExecutionError(
            "policy transition config, schedule, receipt, or runtime binding differs"
        )
    if {row.trial_key for row in schedule.rows} != set(runtime_admission.plan.trial_keys):
        raise SealedOnlineExecutionError(
            "policy transition schedule belongs to another query/trial set"
        )
    receipt_by_state = {row.policy_state: row for row in receipt.transitions}
    if len(receipt_by_state) != len(receipt.transitions):
        raise SealedOnlineExecutionError("policy transition receipt repeats a state")
    schedule_by_group: dict[int, list[object]] = {}
    for row in schedule.rows:
        schedule_by_group.setdefault(row.group_order, []).append(row)
    transitions: dict[str, PolicyTransitionEvidence] = {}
    for group in runtime_receipt.groups:
        rows = schedule_by_group.get(group.group_order)
        if not rows:
            raise SealedOnlineExecutionError("runtime group lacks policy transition source rows")
        representative = rows[0]
        if any(
            (
                row.environment_sha256,
                row.policy_state,
                row.baseline_policy_revision,
                row.baseline_mask_id,
                row.baseline_mask_path,
                row.baseline_mask_sha256,
                row.baseline_mask_byte_count,
                row.baseline_authorized_count,
                row.mask_id,
                row.mask_sha256,
                row.authorized_count,
                row.expected_policy_revision,
                row.policy_churn,
            )
            != (
                representative.environment_sha256,
                representative.policy_state,
                representative.baseline_policy_revision,
                representative.baseline_mask_id,
                representative.baseline_mask_path,
                representative.baseline_mask_sha256,
                representative.baseline_mask_byte_count,
                representative.baseline_authorized_count,
                representative.mask_id,
                representative.mask_sha256,
                representative.authorized_count,
                representative.expected_policy_revision,
                representative.policy_churn,
            )
            for row in rows[1:]
        ):
            raise SealedOnlineExecutionError(
                "one runtime group contains duplicate but inconsistent transition rows"
            )
        if (
            representative.environment_sha256 != group.environment_sha256
            or representative.policy_state != group.policy_state
            or representative.mask_id != group.mask_id
            or representative.mask_sha256 != group.mask_sha256
            or representative.authorized_count != group.authorized_count
            or representative.expected_policy_revision != group.expected_policy_revision
        ):
            raise SealedOnlineExecutionError(
                "runtime group differs from its policy transition source row"
            )
        try:
            current_mask = mask_store.mask(
                representative.mask_id,
                expected_sha256=representative.mask_sha256,
                expected_authorized_count=representative.authorized_count,
            )
            evidence = derive_policy_transition_evidence(
                policy_root,
                representative,
                document_count=schedule.document_count,
                current_mask=current_mask,
            )
        except (PolicyInterventionError, ValueError) as exc:
            raise SealedOnlineExecutionError(
                f"cannot derive a complete policy transition: {exc}"
            ) from exc
        binding = receipt_by_state.get(representative.policy_state)
        if binding is None or (
            binding.baseline_policy_revision != evidence.baseline_policy_revision
            or binding.current_policy_revision != evidence.current_policy_revision
            or binding.baseline_mask_sha256 != evidence.baseline_mask_sha256
            or binding.current_mask_sha256 != evidence.current_mask_sha256
            or binding.baseline_authorized_count != evidence.baseline_authorized_count
            or binding.current_authorized_count != evidence.current_authorized_count
            or binding.policy_churn != evidence.policy_churn
        ):
            raise SealedOnlineExecutionError(
                "policy transition receipt differs from the source-derived evidence"
            )
        if evidence.environment_sha256 in transitions:
            raise SealedOnlineExecutionError(
                "policy transition evidence repeats one runtime environment"
            )
        transitions[evidence.environment_sha256] = evidence
    expected_environments = {row.environment_sha256 for row in runtime_receipt.groups}
    if set(transitions) != expected_environments:
        raise SealedOnlineExecutionError(
            "policy transition evidence does not cover every runtime environment"
        )
    return transitions


def _production_subject(runtime_admission: TrialRuntimeAdmission) -> str:
    subjects = {group.subject for group in runtime_admission.receipt.groups}
    if len(subjects) != 1:
        raise SealedOnlineExecutionError("production runtime must bind exactly one policy subject")
    return _require_text("runtime policy subject", subjects.pop())


def _verify_cache_against_runtime_schedule(
    *,
    artifacts: OnlineRunArtifacts,
    runtime_admission: TrialRuntimeAdmission,
    mask_store: CompiledPolicyMaskStore,
) -> None:
    expected: dict[str, tuple[str, int]] = {}
    for group in runtime_admission.receipt.groups:
        mask = mask_store.mask(
            group.mask_id,
            expected_sha256=group.mask_sha256,
            expected_authorized_count=group.authorized_count,
        )
        raw_sha256 = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
        value = (raw_sha256, group.authorized_count)
        prior = expected.setdefault(group.environment_sha256, value)
        if prior != value:
            raise SealedOnlineExecutionError(
                "one runtime environment maps to more than one frozen mask"
            )
    observed = {
        row.environment_sha256: (row.mask_sha256, row.authorized_count)
        for row in artifacts.cache_preparation_receipt.rows
    }
    if len(observed) != len(artifacts.cache_preparation_receipt.rows) or observed != expected:
        raise SealedOnlineExecutionError(
            "live OPA mask selection differs from the frozen runtime schedule"
        )


def _predictions_from_panel(artifacts: OnlineRunArtifacts) -> tuple[OnlinePrediction, ...]:
    by_trial: dict[str, OnlinePrediction] = {}
    for row in artifacts.admitted_panel.panel.rows:
        if not row.controller_selected:
            continue
        if row.trial_key in by_trial:
            raise SealedOnlineExecutionError("panel selects more than one action for a trial")
        by_trial[row.trial_key] = OnlinePrediction(
            trial_key=row.trial_key,
            family_key=row.family_key,
            returned_document_ids=row.returned_document_ids,
            emitted_answer=None,
        )
    expected = {row.trial_key for row in artifacts.execution_order_receipt.rows}
    if set(by_trial) != expected:
        raise SealedOnlineExecutionError("panel selections do not cover the order receipt")
    return tuple(by_trial[key] for key in sorted(by_trial))


def _audit_bytes(records: tuple[AuditRecord, ...]) -> bytes:
    if not records:
        raise SealedOnlineExecutionError("online execution produced no audit records")
    return b"".join(record.canonical_bytes() + b"\n" for record in records)


def _persist_output(
    *,
    root: Path,
    manifest_sha256: str,
    role: str,
    encoded: bytes,
    semantic_sha256: str,
) -> OnlineOutputPin:
    target = _output_path(root, manifest_sha256, role)
    _write(encoded, target, label=role)
    return OnlineOutputPin(
        role=role,
        filename=target.name,
        byte_count=len(encoded),
        file_sha256=_sha256(encoded),
        semantic_sha256=semantic_sha256,
    )


def _execute_online_objects(
    *,
    admission_receipt: OnlineCustodyAdmissionReceipt,
    required_artifacts: RequiredArtifactIdBindings,
    execution: object,
    run_receipt: SealedRunReceipt,
    retriever: GovernedRetriever,
    provenance_registry: AdmittedProvenanceRegistry,
    trial_runtimes: Mapping[str, OnlineTrialRuntime],
    permutation_seed: int,
    expected_policy_version: str,
    query_partition_audit_sha256: str,
    pseudonym_key: bytes,
    pseudonym_key_id: str,
    k: int,
    policy_action: str,
    partition_label: Literal["primary", "reserve"],
) -> tuple[OnlineRunArtifacts, PredictionArtifact]:
    artifacts = run_admitted_online_matrix(
        admission_receipt=admission_receipt,
        required_artifacts=required_artifacts,
        execution=execution,
        run_receipt=run_receipt,
        retriever=retriever,
        provenance_registry=provenance_registry,
        trial_runtimes=trial_runtimes,
        permutation_seed=permutation_seed,
        expected_policy_version=expected_policy_version,
        query_partition_audit_sha256=query_partition_audit_sha256,
        pseudonym_key=pseudonym_key,
        pseudonym_key_id=pseudonym_key_id,
        k=k,
        policy_action=policy_action,
        partition_label=partition_label,
        occurred_at_factory=None,
    )
    if not isinstance(artifacts, OnlineRunArtifacts):
        raise SealedOnlineExecutionError("online runner returned an untyped artifact set")
    predictions = emit_online_predictions(
        execution,
        _predictions_from_panel(artifacts),
        receipt=run_receipt,
        manifest_sha256=run_receipt.manifest_sha256,
    )
    return artifacts, predictions


def _persist_completed_online_run(
    *,
    root: Path,
    attempt: SealedOnlineAttemptReceipt,
    artifacts: OnlineRunArtifacts,
    predictions: PredictionArtifact,
) -> PersistedSealedOnlineRun:
    """Publish results only after every held-open input context has reclosed."""

    panel = artifacts.admitted_panel.panel
    panel_receipt = artifacts.admitted_panel.admission_receipt
    order = artifacts.execution_order_receipt
    cache_preparation = artifacts.cache_preparation_receipt
    audit = _audit_bytes(artifacts.audit_records)
    pins = (
        _persist_output(
            root=root,
            manifest_sha256=attempt.manifest_sha256,
            role="action-panel",
            encoded=panel.canonical_bytes() + b"\n",
            semantic_sha256=panel.artifact_sha256,
        ),
        _persist_output(
            root=root,
            manifest_sha256=attempt.manifest_sha256,
            role="action-panel-admission",
            encoded=panel_receipt.canonical_bytes() + b"\n",
            semantic_sha256=panel_receipt.receipt_sha256,
        ),
        _persist_output(
            root=root,
            manifest_sha256=attempt.manifest_sha256,
            role="audit-chain",
            encoded=audit,
            semantic_sha256=artifacts.audit_records[-1].record_sha256,
        ),
        _persist_output(
            root=root,
            manifest_sha256=attempt.manifest_sha256,
            role="cache-preparation",
            encoded=cache_preparation.canonical_bytes() + b"\n",
            semantic_sha256=cache_preparation.receipt_sha256,
        ),
        _persist_output(
            root=root,
            manifest_sha256=attempt.manifest_sha256,
            role="execution-order",
            encoded=order.canonical_bytes() + b"\n",
            semantic_sha256=order.receipt_sha256,
        ),
        _persist_output(
            root=root,
            manifest_sha256=attempt.manifest_sha256,
            role="predictions",
            encoded=predictions.canonical_bytes() + b"\n",
            semantic_sha256=predictions.artifact_sha256,
        ),
    )
    result = SealedOnlineResultReceipt(
        manifest_sha256=attempt.manifest_sha256,
        run_receipt_sha256=attempt.run_receipt_sha256,
        execution_artifact_sha256=attempt.execution_artifact_sha256,
        attempt_receipt_sha256=attempt.receipt_sha256,
        audit_head_sha256=artifacts.audit_records[-1].record_sha256,
        audit_record_count=len(artifacts.audit_records),
        outputs=tuple(sorted(pins, key=lambda row: row.role.encode("utf-8"))),
    )
    _write(
        result.canonical_bytes() + b"\n",
        sealed_online_result_path(root, attempt.manifest_sha256),
        label="sealed online result receipt",
    )
    return PersistedSealedOnlineRun(
        artifacts=artifacts,
        predictions=predictions,
        attempt_receipt=attempt,
        result_receipt=result,
        output_root=root,
    )


def _run_sealed_online_once_from_objects(
    *,
    output_root: str | Path,
    admission_receipt: OnlineCustodyAdmissionReceipt,
    required_artifacts: RequiredArtifactIdBindings,
    execution: object,
    run_receipt: SealedRunReceipt,
    retriever: GovernedRetriever,
    provenance_registry: AdmittedProvenanceRegistry,
    trial_runtimes: Mapping[str, OnlineTrialRuntime],
    runtime_receipt: TrialRuntimeAdmissionReceipt,
    index_receipt: AuthorizedIndexStoreReceipt,
    permutation_seed: int,
    expected_policy_version: str,
    query_partition_audit_sha256: str,
    pseudonym_key: bytes,
    pseudonym_key_id: str,
    k: int = 10,
    policy_action: str = "retrieve",
    partition_label: Literal["primary", "reserve"] = "primary",
) -> PersistedSealedOnlineRun:
    """Internal test helper retaining direct object injection."""

    root = _output_root(output_root)
    attempt = _attempt_receipt(
        root=root,
        admission_receipt=admission_receipt,
        required_artifacts=required_artifacts,
        execution=execution,
        run_receipt=run_receipt,
        retriever=retriever,
        runtime_receipt=runtime_receipt,
        index_receipt=index_receipt,
        permutation_seed=permutation_seed,
        expected_policy_version=expected_policy_version,
        query_partition_audit_sha256=query_partition_audit_sha256,
        pseudonym_key_id=pseudonym_key_id,
        k=k,
        policy_action=policy_action,
        partition_label=partition_label,
    )
    attempt_path = sealed_online_attempt_path(root, attempt.manifest_sha256)
    _write(attempt.canonical_bytes() + b"\n", attempt_path, label="sealed online attempt")

    artifacts, predictions = _execute_online_objects(
        admission_receipt=admission_receipt,
        required_artifacts=required_artifacts,
        execution=execution,
        run_receipt=run_receipt,
        retriever=retriever,
        provenance_registry=provenance_registry,
        trial_runtimes=trial_runtimes,
        permutation_seed=permutation_seed,
        expected_policy_version=expected_policy_version,
        query_partition_audit_sha256=query_partition_audit_sha256,
        pseudonym_key=pseudonym_key,
        pseudonym_key_id=pseudonym_key_id,
        k=k,
        policy_action=policy_action,
        partition_label=partition_label,
    )
    return _persist_completed_online_run(
        root=root,
        attempt=attempt,
        artifacts=artifacts,
        predictions=predictions,
    )


def run_sealed_online_once(
    *,
    output_root: str | Path,
    admission_receipt: OnlineCustodyAdmissionReceipt,
    required_artifacts: RequiredArtifactIdBindings,
    run_receipt: SealedRunReceipt,
    runtime_admission: TrialRuntimeAdmission,
    runtime_attestation_plan_path: str | Path,
    expected_runtime_attestation_plan_sha256: str,
    runtime_attestation_receipt_path: str | Path,
    expected_runtime_attestation_receipt_sha256: str,
    expected_runtime_receipt_sha256: str,
    artifact_root: str | Path,
    authorized_index_store_root: str | Path,
    expected_authorized_index_store_receipt_sha256: str,
    policy_intervention_root: str | Path,
    expected_policy_intervention_receipt_sha256: str,
    pseudonym_key_path: str | Path,
    expected_pseudonym_key_sha256: str,
    runtime_claim_receipt: RuntimeClaimReceipt,
) -> PersistedSealedOnlineRun:
    """Consume the token, reconstruct the apparatus, and execute once.

    This is the production boundary. Query vectors, document matrices, feature
    contexts, policy environments, policy revision, controller configuration,
    role, action order seed, K, action, and partition are not injectable here.
    They are loaded from the admitted plan and receipt-bound source packages or
    fixed by this runner revision.
    """

    if not isinstance(runtime_admission, TrialRuntimeAdmission):
        raise SealedOnlineExecutionError("runtime_admission must be typed")
    if not isinstance(runtime_claim_receipt, RuntimeClaimReceipt):
        raise SealedOnlineExecutionError("production execution requires typed RUN_CLAIMED")
    root = _output_root(output_root)
    artifact_source = _absolute_root("artifact_root", artifact_root)
    index_root = _absolute_root(
        "authorized_index_store_root",
        authorized_index_store_root,
    )
    policy_root = _absolute_root(
        "policy_intervention_root",
        policy_intervention_root,
    )
    pseudonym_source = _absolute_root("pseudonym_key_path", pseudonym_key_path)
    runtime_digest = _require_sha256(
        "expected_runtime_receipt_sha256",
        expected_runtime_receipt_sha256,
    )
    index_digest = _require_sha256(
        "expected_authorized_index_store_receipt_sha256",
        expected_authorized_index_store_receipt_sha256,
    )
    policy_digest = _require_sha256(
        "expected_policy_intervention_receipt_sha256",
        expected_policy_intervention_receipt_sha256,
    )
    key_digest = _require_sha256(
        "expected_pseudonym_key_sha256",
        expected_pseudonym_key_sha256,
    )
    key_id = _pseudonym_key_id(key_digest)
    runtime_attestation_plan, runtime_attestation_receipt = _admit_production_runtime_attestation(
        plan_path=runtime_attestation_plan_path,
        expected_plan_sha256=expected_runtime_attestation_plan_sha256,
        receipt_path=runtime_attestation_receipt_path,
        expected_receipt_sha256=expected_runtime_attestation_receipt_sha256,
        run_receipt=run_receipt,
        source_paths={
            "artifact_root": artifact_source,
            "authorized_index_store_root": index_root,
            "embedding_store_root": runtime_admission.embedding_store_root,
            "partition_audit_path": runtime_admission.partition_audit_path,
            "policy_intervention_root": policy_root,
            "pseudonym_key_path": pseudonym_source,
            "query_package_root": runtime_admission.query_package_root,
            "schedule_path": runtime_admission.schedule_path,
            "staged_root": runtime_admission.staged_root,
        },
    )
    if Path(runtime_attestation_plan.opa_binary.path) != _PRODUCTION_OPA_BINARY:
        raise SealedOnlineExecutionError("runtime attestation pins another OPA binary path")
    attempt = _production_attempt_receipt(
        root=root,
        admission_receipt=admission_receipt,
        required_artifacts=required_artifacts,
        run_receipt=run_receipt,
        runtime_admission=runtime_admission,
        runtime_attestation_plan=runtime_attestation_plan,
        runtime_attestation_receipt=runtime_attestation_receipt,
        expected_runtime_receipt_sha256=runtime_digest,
        expected_authorized_index_store_receipt_sha256=index_digest,
        expected_policy_intervention_receipt_sha256=policy_digest,
        pseudonym_key_id=key_id,
        runtime_claim_receipt=runtime_claim_receipt,
    )
    attempt_path = sealed_online_attempt_path(root, attempt.manifest_sha256)
    _write(attempt.canonical_bytes() + b"\n", attempt_path, label="sealed online attempt")

    # Above this line only frozen attestation controls, the one-shot marker, and
    # live Linux process evidence are read. No index, policy, embedding, query,
    # provenance, or secret source is opened before both gates exist.
    index_receipt = load_authorized_index_store_receipt(index_root)
    _verify_production_source_bindings(
        runtime_admission=runtime_admission,
        index_receipt=index_receipt,
        expected_authorized_index_store_receipt_sha256=index_digest,
    )
    if index_receipt.policy_receipt_sha256 != policy_digest:
        raise SealedOnlineExecutionError("policy intervention receipt differs from the frozen pin")
    if runtime_admission.receipt.receipt_sha256 != runtime_digest:
        raise SealedOnlineExecutionError("runtime receipt changed after token creation")

    backend = HnswlibBackend()
    provider = VerifiedAuthorizedIndexProvider(
        index_root,
        embedding_store_root=runtime_admission.embedding_store_root,
        policy_intervention_root=policy_root,
        expected_embedding_receipt_sha256=(
            runtime_admission.receipt.embedding_store_receipt_sha256
        ),
        expected_policy_receipt_sha256=policy_digest,
        expected_store_receipt_sha256=index_digest,
        backend=backend,
    )
    loaded = load_trial_runtime(runtime_admission)
    if (
        loaded.execution.plan != runtime_admission.plan
        or loaded.execution.artifact_sha256 != runtime_admission.receipt.execution_artifact_sha256
        or set(loaded.trial_runtimes) != set(runtime_admission.plan.trial_keys)
    ):
        raise SealedOnlineExecutionError("loaded runtime differs from the frozen admission")
    mask_store = CompiledPolicyMaskStore(policy_root / CATALOG_FILENAME)
    mask_store.verify_all()
    if (
        mask_store.catalog_sha256 != runtime_admission.receipt.mask_catalog_sha256
        or mask_store.catalog.policy_revision != runtime_admission.receipt.policy_bundle_revision
        or mask_store.catalog.document_universe_sha256
        != runtime_admission.plan.document_universe_sha256
    ):
        raise SealedOnlineExecutionError("compiled policy catalog differs from the frozen runtime")
    policy_transitions = _production_policy_transitions(
        runtime_admission=runtime_admission,
        policy_root=policy_root,
        expected_policy_receipt_sha256=policy_digest,
        mask_store=mask_store,
    )
    with _production_opa_sidecar(
        attempt_path=attempt_path,
        attempt=attempt,
        policy_root=policy_root,
        runtime_admission=runtime_admission,
        mask_store=mask_store,
        expected_policy_receipt_sha256=policy_digest,
    ) as policy:
        pseudonym_key = _load_pseudonym_key(
            pseudonym_source,
            expected_sha256=key_digest,
        )

        with open_verified_document_matrices(
            runtime_admission.embedding_store_root,
            index_receipt=index_receipt,
            expected_embedding_receipt_sha256=(
                runtime_admission.receipt.embedding_store_receipt_sha256
            ),
        ) as matrices:
            retriever = GovernedRetriever(
                matrices.old_active,
                policy,
                _production_subject(runtime_admission),
                expected_document_universe_sha256=(runtime_admission.plan.document_universe_sha256),
                exact_truth_vectors=matrices.current_truth,
                metric=provider.retrieval_metric,
                controller=RuleController(ControllerConfig()),
                policy_transitions=policy_transitions,
                require_policy_transition=True,
                trusted_readonly_vectors=True,
                authorized_hnsw_provider=provider,
            )
            with open_digest_provenance_registry(
                runtime_admission.plan,
                artifact_root=artifact_source,
                verification_receipt=(artifact_source / EXECUTION_LEAF_RECEIPT_FILENAME),
                component_verification_receipt=(required_artifacts.verification_receipt),
                component_artifact_ids=(required_artifacts.provenance_component_artifact_ids),
            ) as provenance_registry:
                if not isinstance(provenance_registry, DigestOnlyProvenanceRegistry):
                    raise SealedOnlineExecutionError(
                        "production provenance registry has another implementation"
                    )
                artifacts, predictions = _execute_online_objects(
                    admission_receipt=admission_receipt,
                    required_artifacts=required_artifacts,
                    execution=loaded.execution,
                    run_receipt=run_receipt,
                    retriever=retriever,
                    provenance_registry=provenance_registry,
                    trial_runtimes=loaded.trial_runtimes,
                    permutation_seed=runtime_claim_receipt.permutation_seed,
                    expected_policy_version=(runtime_admission.receipt.policy_bundle_revision),
                    query_partition_audit_sha256=(
                        runtime_admission.receipt.query_partition_audit_sha256
                    ),
                    pseudonym_key=pseudonym_key,
                    pseudonym_key_id=key_id,
                    k=PRODUCTION_K,
                    policy_action=PRODUCTION_POLICY_ACTION,
                    partition_label=PRODUCTION_PARTITION_LABEL,
                )
                _verify_cache_against_runtime_schedule(
                    artifacts=artifacts,
                    runtime_admission=runtime_admission,
                    mask_store=mask_store,
                )

    # The matrix contexts rehash descriptors and paths, then OPA terminates and
    # its scratch data is removed, before any result file can exist. Any input
    # rebind or sidecar-cleanup failure consumes the attempt and publishes no
    # valid result receipt.
    return _persist_completed_online_run(
        root=root,
        attempt=attempt,
        artifacts=artifacts,
        predictions=predictions,
    )


def _load_receipt(path: str | Path, *, label: str) -> Mapping[str, Any]:
    try:
        encoded = read_secure_control_file(path, label=label)
    except ArtifactIntegrityError as exc:
        raise SealedOnlineExecutionError(f"cannot load {label}: {exc}") from exc
    return _decode_object(encoded, label=label)


def load_sealed_online_attempt_receipt(path: str | Path) -> SealedOnlineAttemptReceipt:
    value = SealedOnlineAttemptReceipt.from_dict(
        _load_receipt(path, label="sealed online attempt receipt")
    )
    encoded = read_secure_control_file(path, label="sealed online attempt receipt")
    if encoded != value.canonical_bytes() + b"\n":
        raise SealedOnlineExecutionError("sealed online attempt bytes are not canonical")
    return value


def load_sealed_online_result_receipt(path: str | Path) -> SealedOnlineResultReceipt:
    value = SealedOnlineResultReceipt.from_dict(
        _load_receipt(path, label="sealed online result receipt")
    )
    encoded = read_secure_control_file(path, label="sealed online result receipt")
    if encoded != value.canonical_bytes() + b"\n":
        raise SealedOnlineExecutionError("sealed online result bytes are not canonical")
    return value


def verify_sealed_online_outputs(
    result: SealedOnlineResultReceipt,
    *,
    output_root: str | Path,
) -> None:
    """Rehash every core output named by the result receipt."""

    if not isinstance(result, SealedOnlineResultReceipt):
        raise SealedOnlineExecutionError("result must be a sealed online result receipt")
    root = _output_root(output_root)
    expected = {row.filename: row for row in result.outputs}
    for filename, pin in expected.items():
        try:
            encoded = read_secure_control_file(
                root / filename,
                label=f"online output {pin.role}",
            )
        except ArtifactIntegrityError as exc:
            raise SealedOnlineExecutionError(
                f"cannot verify online output {pin.role}: {exc}"
            ) from exc
        if len(encoded) != pin.byte_count or _sha256(encoded) != pin.file_sha256:
            raise SealedOnlineExecutionError(f"online output {pin.role} changed bytes")
