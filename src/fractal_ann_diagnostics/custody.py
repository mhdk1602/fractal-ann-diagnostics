"""Machine-verifiable label custody and online-run admission.

The custody seal commits to plaintext label bytes without delivering those bytes
to the online runner.  It also commits to separately stored timelock ciphertexts,
an exact drand chain and round, and pinned tool and builder artifacts.  This
module proves byte agreement under those commitments.  It does not prove that a
person lacked another plaintext copy, that a public benchmark was unknown, or
that one administrator is independent from another process it controls.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    ArtifactVerificationReceipt,
    LocalArtifactSpec,
    digest_regular_file,
    load_local_artifact_map,
    load_verification_receipt,
    read_secure_control_file,
    read_secure_regular_file,
    verify_local_artifacts,
    write_exclusive_receipt_bytes,
)
from .label_separation import sealed_run_receipt_sha256
from .study import (
    FIXED_CORPORA,
    SealedRunReceipt,
    StudyManifestError,
    load_sealed_run_receipt,
    manifest_sha256,
    sealed_receipt_uri,
    validate_study_manifest,
)

CUSTODY_CORPUS_COMMITMENT_SCHEMA = "fractal-custody-corpus-commitment-v2"
CUSTODY_SEAL_RECEIPT_SCHEMA = "fractal-custody-seal-receipt-v2"
ONLINE_CUSTODY_ADMISSION_SCHEMA = "fractal-online-custody-admission-v1"
TIMELOCK_ENCRYPTION_RECEIPT_SCHEMA = "fractal-timelock-encryption-v1"

DEFAULT_TLOCK_TIMEOUT_SECONDS = 60
DEFAULT_MAX_PLAINTEXT_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_CIPHERTEXT_BYTES = 128 * 1024 * 1024
_MAX_TLOCK_STDERR_BYTES = 16 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CUSTODY_COMMITMENT_FIELDS = {
    "corpus_id",
    "online_execution_sha256",
    "schema_version",
    "sealed_label_ciphertext_sha256",
    "sealed_label_plaintext_sha256",
    "timelock_encryption_receipt_file_sha256",
}
_CUSTODY_SEAL_FIELDS = {
    "commitments",
    "custody_builder_sha256",
    "drand_chain_hash",
    "drand_round",
    "protocol_version",
    "schema_version",
    "timelock_tool_sha256",
}
_ONLINE_ADMISSION_FIELDS = {
    "artifact_verification_receipt_sha256",
    "custody_seal_receipt_sha256",
    "manifest_sha256",
    "online_artifact_verification_receipt_sha256",
    "run_receipt_sha256",
    "runner_identity",
    "schema_version",
    "verified_artifact_ids",
}
_TIMELOCK_ENCRYPTION_FIELDS = {
    "ciphertext_byte_count",
    "ciphertext_sha256",
    "corpus_id",
    "drand_chain_hash",
    "drand_network",
    "drand_round",
    "plaintext_byte_count",
    "plaintext_sha256",
    "schema_version",
    "tle_arguments",
    "tle_binary_sha256",
}

# These roles can be read by the online admission process.  In particular,
# plaintext labels, decryption receipts, development data, and fitted analysis
# models are absent. The ciphertext remains readable for commitment revalidation.
# The pre-execution query-partition audit is an admitted runtime source because
# every production trial package binds it.
ONLINE_CUSTODY_REVALIDATION_ROLES = frozenset(
    {
        "authorized-index-store",
        "corpus-normalizer",
        "custody-builder",
        "custody-seal-receipt",
        "embedding-store",
        "exact-authorized-oracle",
        "frozen-controller",
        "online-staging-package",
        "online-execution",
        "opa-pdp",
        "opa-runtime-binary",
        "policy-workload",
        "primary-embedding",
        "query-partition-audit",
        "sealed-label-ciphertext",
        "source-code",
        "static-comparator",
        "strict-authorized-hnsw",
        "trial-runtime-package",
    }
)


class CustodyError(ValueError):
    """Raised when a custody commitment or online admission is invalid."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CustodyError("custody evidence must be finite canonical JSON") from exc


def _closed_mapping(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CustodyError(f"{label} must be a JSON object")
    missing = fields - set(value)
    unexpected = set(value) - fields
    if missing or unexpected:
        raise CustodyError(
            f"{label} keys do not match the closed schema; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return value


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CustodyError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise CustodyError(f"{label} contains non-finite number {value!r}")

    try:
        payload = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise CustodyError(f"{label} must be valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise CustodyError(f"{label} must contain valid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise CustodyError(f"{label} must contain one JSON object")
    return payload


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CustodyError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CustodyError(f"{name} must be a non-empty canonical string")
    if unicodedata.normalize("NFC", value) != value:
        raise CustodyError(f"{name} must use NFC Unicode normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise CustodyError(f"{name} cannot contain control characters")
    return value


def _positive_round(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CustodyError("drand_round must be a positive integer")
    return value


def _positive_limit(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise CustodyError(f"{name} must be an integer from 1 through {maximum}")
    return value


def _drand_network(value: object) -> str:
    network = _require_text("drand_network", value)
    parsed = urlsplit(network)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise CustodyError(
            "drand_network must be an HTTPS endpoint without credentials, query, or fragment"
        )
    return network


def _tle_arguments(
    *,
    drand_network: str,
    drand_chain_hash: str,
    drand_round: int,
) -> tuple[str, ...]:
    return (
        "--encrypt",
        f"--network={drand_network}",
        f"--chain={drand_chain_hash}",
        f"--round={drand_round}",
    )


def _artifact_rows(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    values = manifest.get("artifacts")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise CustodyError("study manifest artifacts must be an array")
    if not all(isinstance(value, Mapping) for value in values):
        raise CustodyError("study manifest artifacts must contain objects")
    return tuple(values)  # type: ignore[arg-type]


def _artifacts_for_role(
    manifest: Mapping[str, Any],
    role: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row in _artifact_rows(manifest) if row.get("role") == role)


def _sole_artifact(manifest: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    matches = _artifacts_for_role(manifest, role)
    if len(matches) != 1:
        raise CustodyError(f"study manifest must contain exactly one {role!r} artifact")
    return matches[0]


def _corpus_artifacts(
    manifest: Mapping[str, Any],
    role: str,
) -> dict[str, Mapping[str, Any]]:
    rows = _artifacts_for_role(manifest, role)
    mapped: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        corpus_id = row.get("corpus_id")
        if not isinstance(corpus_id, str) or corpus_id in mapped:
            raise CustodyError(f"{role!r} artifacts have invalid corpus coverage")
        mapped[corpus_id] = row
    if set(mapped) != set(FIXED_CORPORA):
        raise CustodyError(f"{role!r} artifacts must cover the fixed corpus suite")
    return mapped


@dataclass(frozen=True)
class TimelockEncryptionReceipt:
    """Deterministic evidence for one pinned-binary ``tle`` encryption."""

    corpus_id: str
    plaintext_sha256: str
    plaintext_byte_count: int
    ciphertext_sha256: str
    ciphertext_byte_count: int
    tle_binary_sha256: str
    drand_network: str
    drand_chain_hash: str
    drand_round: int
    tle_arguments: tuple[str, ...]
    schema_version: str = TIMELOCK_ENCRYPTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise CustodyError("timelock receipt corpus_id is not registered")
        _require_sha256("plaintext_sha256", self.plaintext_sha256)
        _require_sha256("ciphertext_sha256", self.ciphertext_sha256)
        _require_sha256("tle_binary_sha256", self.tle_binary_sha256)
        if self.plaintext_sha256 == self.ciphertext_sha256:
            raise CustodyError("timelock ciphertext must differ from its plaintext")
        for name in ("plaintext_byte_count", "ciphertext_byte_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CustodyError(f"{name} must be a positive integer")
        network = _drand_network(self.drand_network)
        chain = _require_sha256("drand_chain_hash", self.drand_chain_hash)
        round_number = _positive_round(self.drand_round)
        if not isinstance(self.tle_arguments, Sequence) or isinstance(
            self.tle_arguments, (str, bytes)
        ):
            raise CustodyError("tle_arguments must be an array of exact CLI arguments")
        arguments = tuple(self.tle_arguments)
        expected_arguments = _tle_arguments(
            drand_network=network,
            drand_chain_hash=chain,
            drand_round=round_number,
        )
        if arguments != expected_arguments:
            raise CustodyError(
                "tle_arguments must contain only the exact encrypt, network, chain, and round flags"
            )
        object.__setattr__(self, "tle_arguments", arguments)
        if self.schema_version != TIMELOCK_ENCRYPTION_RECEIPT_SCHEMA:
            raise CustodyError(f"schema_version must equal {TIMELOCK_ENCRYPTION_RECEIPT_SCHEMA!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "ciphertext_byte_count": self.ciphertext_byte_count,
            "ciphertext_sha256": self.ciphertext_sha256,
            "corpus_id": self.corpus_id,
            "drand_chain_hash": self.drand_chain_hash,
            "drand_network": self.drand_network,
            "drand_round": self.drand_round,
            "plaintext_byte_count": self.plaintext_byte_count,
            "plaintext_sha256": self.plaintext_sha256,
            "schema_version": self.schema_version,
            "tle_arguments": list(self.tle_arguments),
            "tle_binary_sha256": self.tle_binary_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def file_sha256(self) -> str:
        """Digest the exact canonical receipt file pinned by the manifest."""

        return hashlib.sha256(self.canonical_bytes() + b"\n").hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> TimelockEncryptionReceipt:
        row = _closed_mapping(
            value,
            fields=_TIMELOCK_ENCRYPTION_FIELDS,
            label="timelock encryption receipt",
        )
        arguments = row["tle_arguments"]
        if not isinstance(arguments, Sequence) or isinstance(arguments, (str, bytes)):
            raise CustodyError("tle_arguments must be an array")
        return cls(
            corpus_id=row["corpus_id"],
            plaintext_sha256=row["plaintext_sha256"],
            plaintext_byte_count=row["plaintext_byte_count"],
            ciphertext_sha256=row["ciphertext_sha256"],
            ciphertext_byte_count=row["ciphertext_byte_count"],
            tle_binary_sha256=row["tle_binary_sha256"],
            drand_network=row["drand_network"],
            drand_chain_hash=row["drand_chain_hash"],
            drand_round=row["drand_round"],
            tle_arguments=tuple(arguments),
            schema_version=row["schema_version"],
        )


def write_timelock_encryption_receipt(
    receipt: TimelockEncryptionReceipt,
    target: str | Path,
) -> None:
    if not isinstance(receipt, TimelockEncryptionReceipt):
        raise CustodyError("receipt must be a TimelockEncryptionReceipt")
    try:
        write_exclusive_receipt_bytes(receipt.canonical_bytes() + b"\n", target)
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"cannot write timelock encryption receipt: {exc}") from exc


def load_timelock_encryption_receipt(
    path: str | Path,
) -> TimelockEncryptionReceipt:
    try:
        encoded = read_secure_control_file(path, label="timelock encryption receipt")
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"cannot load timelock encryption receipt: {exc}") from exc
    receipt = TimelockEncryptionReceipt.from_dict(
        _decode_object(encoded, label="timelock encryption receipt")
    )
    if encoded != receipt.canonical_bytes() + b"\n":
        raise CustodyError(
            "timelock encryption receipt bytes must equal canonical JSON plus one newline"
        )
    return receipt


def verify_timelock_encryption_receipt(
    receipt: TimelockEncryptionReceipt,
    manifest: Mapping[str, Any],
    *,
    custody_seal: CustodySealReceipt | None = None,
    require_frozen: bool = True,
) -> None:
    """Bind one operation receipt to final artifact pins and an optional suite seal."""

    if not isinstance(receipt, TimelockEncryptionReceipt):
        raise CustodyError("receipt must be a TimelockEncryptionReceipt")
    try:
        validate_study_manifest(manifest, require_frozen=require_frozen)
    except StudyManifestError as exc:
        raise CustodyError(f"invalid study manifest: {exc}") from exc
    labels = _corpus_artifacts(manifest, "sealed-labels")
    ciphertexts = _corpus_artifacts(manifest, "sealed-label-ciphertext")
    operation_receipts = _corpus_artifacts(manifest, "timelock-encryption-receipt")
    tool = _sole_artifact(manifest, "timelock-tool")
    expected = (
        _require_sha256(
            f"sealed-labels sha256 for {receipt.corpus_id}",
            labels[receipt.corpus_id].get("sha256"),
        ),
        _require_sha256(
            f"sealed-label-ciphertext sha256 for {receipt.corpus_id}",
            ciphertexts[receipt.corpus_id].get("sha256"),
        ),
        _require_sha256("timelock-tool artifact sha256", tool.get("sha256")),
    )
    observed = (
        receipt.plaintext_sha256,
        receipt.ciphertext_sha256,
        receipt.tle_binary_sha256,
    )
    if observed != expected:
        raise CustodyError("timelock encryption receipt differs from the manifest pins")
    pinned_receipt_file_sha256 = _require_sha256(
        f"timelock-encryption-receipt sha256 for {receipt.corpus_id}",
        operation_receipts[receipt.corpus_id].get("sha256"),
    )
    if receipt.file_sha256 != pinned_receipt_file_sha256:
        raise CustodyError("timelock encryption receipt file digest differs from the manifest pin")
    if custody_seal is None:
        return
    verify_custody_seal_receipt(
        custody_seal,
        manifest,
        require_frozen=require_frozen,
        require_manifest_pin=require_frozen,
    )
    commitment = next(row for row in custody_seal.commitments if row.corpus_id == receipt.corpus_id)
    if (
        receipt.drand_chain_hash != custody_seal.drand_chain_hash
        or receipt.drand_round != custody_seal.drand_round
        or receipt.tle_binary_sha256 != custody_seal.timelock_tool_sha256
        or receipt.plaintext_sha256 != commitment.sealed_label_plaintext_sha256
        or receipt.ciphertext_sha256 != commitment.sealed_label_ciphertext_sha256
        or receipt.file_sha256 != commitment.timelock_encryption_receipt_file_sha256
    ):
        raise CustodyError("timelock encryption receipt differs from the custody seal")


def _admit_tle_binary(path: str | Path, *, expected_sha256: str) -> Path:
    target = Path(path)
    if not target.is_absolute() or target.name in {"", ".", ".."}:
        raise CustodyError("tle_binary_path must be an absolute file path")
    try:
        observed = digest_regular_file(target, label="tle binary")
        metadata = target.lstat()
        parent_metadata = target.parent.stat()
    except (ArtifactIntegrityError, OSError) as exc:
        raise CustodyError(f"cannot admit tle binary: {exc}") from exc
    if observed != expected_sha256:
        raise CustodyError("tle binary digest does not match the manifest pin")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise CustodyError("tle binary must be one regular, singly linked file")
    if metadata.st_mode & 0o022:
        raise CustodyError("tle binary cannot be writable by group or other identities")
    if not metadata.st_mode & stat.S_IXUSR:
        raise CustodyError("tle binary must be executable by its owner")
    if hasattr(os, "geteuid") and (
        metadata.st_uid != os.geteuid() or parent_metadata.st_uid != os.geteuid()
    ):
        raise CustodyError("tle binary and its parent must be owned by the custodian identity")
    if parent_metadata.st_mode & 0o022:
        raise CustodyError("tle binary parent cannot be writable by group or other identities")
    return target


def _close_process_stream(
    selector: selectors.BaseSelector,
    stream: Any,
) -> None:
    try:
        selector.unregister(stream)
    except (KeyError, ValueError):
        pass
    try:
        stream.close()
    except OSError:
        pass


def _kill_tle_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        process.kill()


def _run_tle_encrypt(
    binary: Path,
    *,
    arguments: tuple[str, ...],
    plaintext: bytes,
    timeout_seconds: int,
    max_ciphertext_bytes: int,
) -> bytes:
    command = [str(binary), *arguments]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            start_new_session=True,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except OSError as exc:
        raise CustodyError(f"cannot execute pinned tle binary: {exc}") from exc
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _kill_tle_process(process)
        process.wait()
        raise CustodyError("pinned tle process did not expose isolated standard streams")

    selector = selectors.DefaultSelector()
    ciphertext = bytearray()
    stderr = bytearray()
    stderr_truncated = False
    input_offset = 0
    deadline = time.monotonic() + timeout_seconds
    streams = (process.stdin, process.stdout, process.stderr)
    try:
        for stream in streams:
            os.set_blocking(stream.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")

        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CustodyError(f"pinned tle encryption exceeded {timeout_seconds} seconds")
            events = selector.select(min(remaining, 0.25))
            for key, _ in events:
                stream = key.fileobj
                if key.data == "stdin":
                    if input_offset == len(plaintext):
                        _close_process_stream(selector, stream)
                        continue
                    try:
                        written = os.write(
                            stream.fileno(),
                            plaintext[input_offset : input_offset + 64 * 1024],
                        )
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        _close_process_stream(selector, stream)
                        continue
                    input_offset += written
                    continue

                try:
                    chunk = os.read(stream.fileno(), 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    _close_process_stream(selector, stream)
                    continue
                if key.data == "stdout":
                    if len(ciphertext) + len(chunk) > max_ciphertext_bytes:
                        raise CustodyError("pinned tle ciphertext exceeds max_ciphertext_bytes")
                    ciphertext.extend(chunk)
                elif len(stderr) < _MAX_TLOCK_STDERR_BYTES:
                    remaining_stderr = _MAX_TLOCK_STDERR_BYTES - len(stderr)
                    stderr.extend(chunk[:remaining_stderr])
                    stderr_truncated = stderr_truncated or len(chunk) > remaining_stderr
                else:
                    stderr_truncated = True

        remaining = max(deadline - time.monotonic(), 0.01)
        try:
            return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise CustodyError(f"pinned tle encryption exceeded {timeout_seconds} seconds") from exc
    except BaseException:
        _kill_tle_process(process)
        process.wait()
        raise
    finally:
        for stream in streams:
            if not stream.closed:
                _close_process_stream(selector, stream)
        selector.close()

    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        if stderr_truncated:
            message += " [stderr truncated]"
        detail = message or "no stderr"
        raise CustodyError(f"pinned tle encryption failed with exit {return_code}: {detail}")
    if input_offset != len(plaintext):
        raise CustodyError("pinned tle process closed stdin before reading the plaintext")
    if not ciphertext:
        raise CustodyError("pinned tle encryption produced an empty ciphertext")
    return bytes(ciphertext)


def encrypt_timelock_label(
    manifest: Mapping[str, Any],
    *,
    corpus_id: str,
    plaintext_path: str | Path,
    tle_binary_path: str | Path,
    drand_network: str,
    drand_chain_hash: str,
    drand_round: int,
    ciphertext_path: str | Path,
    timeout_seconds: int = DEFAULT_TLOCK_TIMEOUT_SECONDS,
    max_plaintext_bytes: int = DEFAULT_MAX_PLAINTEXT_BYTES,
    max_ciphertext_bytes: int = DEFAULT_MAX_CIPHERTEXT_BYTES,
) -> TimelockEncryptionReceipt:
    """Encrypt one pinned label file through the exact manifest-pinned ``tle`` binary.

    Plaintext is read through the no-link control-file boundary and supplied on
    stdin. Ciphertext is accepted only from stdout, held under an explicit bound,
    and written with exclusive creation. No decryption operation is exposed.
    """

    try:
        validate_study_manifest(manifest)
    except StudyManifestError as exc:
        raise CustodyError(f"invalid study manifest: {exc}") from exc
    if corpus_id not in FIXED_CORPORA:
        raise CustodyError("corpus_id is not in the fixed corpus suite")
    timeout = _positive_limit("timeout_seconds", timeout_seconds, maximum=60)
    plaintext_limit = _positive_limit(
        "max_plaintext_bytes",
        max_plaintext_bytes,
        maximum=1024 * 1024 * 1024,
    )
    ciphertext_limit = _positive_limit(
        "max_ciphertext_bytes",
        max_ciphertext_bytes,
        maximum=2 * 1024 * 1024 * 1024,
    )
    network = _drand_network(drand_network)
    chain = _require_sha256("drand_chain_hash", drand_chain_hash)
    round_number = _positive_round(drand_round)

    labels = _corpus_artifacts(manifest, "sealed-labels")
    ciphertexts = _corpus_artifacts(manifest, "sealed-label-ciphertext")
    tool = _sole_artifact(manifest, "timelock-tool")
    plaintext_pin = _require_sha256(
        f"sealed-labels sha256 for {corpus_id}",
        labels[corpus_id].get("sha256"),
    )
    tool_pin = _require_sha256("timelock-tool artifact sha256", tool.get("sha256"))
    binary = _admit_tle_binary(tle_binary_path, expected_sha256=tool_pin)

    try:
        plaintext = read_secure_regular_file(
            plaintext_path,
            max_bytes=plaintext_limit,
            label=f"{corpus_id} plaintext labels",
        )
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"cannot admit plaintext labels: {exc}") from exc
    if not plaintext:
        raise CustodyError("plaintext label artifact cannot be empty")
    if hashlib.sha256(plaintext).hexdigest() != plaintext_pin:
        raise CustodyError("plaintext label digest does not match the manifest pin")

    arguments = _tle_arguments(
        drand_network=network,
        drand_chain_hash=chain,
        drand_round=round_number,
    )
    ciphertext = _run_tle_encrypt(
        binary,
        arguments=arguments,
        plaintext=plaintext,
        timeout_seconds=timeout,
        max_ciphertext_bytes=ciphertext_limit,
    )
    try:
        final_binary_digest = digest_regular_file(binary, label="tle binary")
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"cannot revalidate tle binary after encryption: {exc}") from exc
    if final_binary_digest != tool_pin:
        raise CustodyError("tle binary changed during encryption")
    ciphertext_digest = hashlib.sha256(ciphertext).hexdigest()
    if ciphertext_digest == plaintext_pin:
        raise CustodyError("pinned tle process returned the plaintext unchanged")
    declared_ciphertext = ciphertexts[corpus_id].get("sha256")
    if (
        isinstance(declared_ciphertext, str)
        and _SHA256.fullmatch(declared_ciphertext) is not None
        and declared_ciphertext != ciphertext_digest
    ):
        raise CustodyError("generated ciphertext does not match the manifest pin")
    if declared_ciphertext not in {"tbd", ciphertext_digest}:
        raise CustodyError("sealed-label-ciphertext sha256 must be tbd or the generated digest")

    try:
        write_exclusive_receipt_bytes(ciphertext, ciphertext_path)
        written_digest = digest_regular_file(
            ciphertext_path,
            label=f"{corpus_id} timelock ciphertext",
        )
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"cannot write timelock ciphertext: {exc}") from exc
    if written_digest != ciphertext_digest:
        raise CustodyError("timelock ciphertext changed during exclusive write")
    return TimelockEncryptionReceipt(
        corpus_id=corpus_id,
        plaintext_sha256=plaintext_pin,
        plaintext_byte_count=len(plaintext),
        ciphertext_sha256=ciphertext_digest,
        ciphertext_byte_count=len(ciphertext),
        tle_binary_sha256=tool_pin,
        drand_network=network,
        drand_chain_hash=chain,
        drand_round=round_number,
        tle_arguments=arguments,
    )


@dataclass(frozen=True)
class CustodyCorpusCommitment:
    """One corpus's online, plaintext-label, and ciphertext commitments."""

    corpus_id: str
    online_execution_sha256: str
    sealed_label_plaintext_sha256: str
    sealed_label_ciphertext_sha256: str
    timelock_encryption_receipt_file_sha256: str
    schema_version: str = CUSTODY_CORPUS_COMMITMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise CustodyError("custody commitment corpus_id is not registered")
        for name in (
            "online_execution_sha256",
            "sealed_label_plaintext_sha256",
            "sealed_label_ciphertext_sha256",
            "timelock_encryption_receipt_file_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            len(
                {
                    self.online_execution_sha256,
                    self.sealed_label_plaintext_sha256,
                    self.sealed_label_ciphertext_sha256,
                    self.timelock_encryption_receipt_file_sha256,
                }
            )
            != 4
        ):
            raise CustodyError(
                "online execution, plaintext labels, ciphertext, and encryption receipt "
                "must be separately pinned"
            )
        if self.schema_version != CUSTODY_CORPUS_COMMITMENT_SCHEMA:
            raise CustodyError(f"schema_version must equal {CUSTODY_CORPUS_COMMITMENT_SCHEMA!r}")

    def to_dict(self) -> dict[str, str]:
        return {
            "corpus_id": self.corpus_id,
            "online_execution_sha256": self.online_execution_sha256,
            "schema_version": self.schema_version,
            "sealed_label_ciphertext_sha256": (self.sealed_label_ciphertext_sha256),
            "sealed_label_plaintext_sha256": self.sealed_label_plaintext_sha256,
            "timelock_encryption_receipt_file_sha256": (
                self.timelock_encryption_receipt_file_sha256
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> CustodyCorpusCommitment:
        row = _closed_mapping(
            value,
            fields=_CUSTODY_COMMITMENT_FIELDS,
            label="custody corpus commitment",
        )
        return cls(
            corpus_id=row["corpus_id"],
            online_execution_sha256=row["online_execution_sha256"],
            sealed_label_plaintext_sha256=row["sealed_label_plaintext_sha256"],
            sealed_label_ciphertext_sha256=row["sealed_label_ciphertext_sha256"],
            timelock_encryption_receipt_file_sha256=row["timelock_encryption_receipt_file_sha256"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class CustodySealReceipt:
    """Closed commitment to time-locked labels before manifest freeze.

    The receipt deliberately omits the manifest digest.  The manifest pins the
    newline-terminated receipt file, avoiding a digest fixed-point cycle.
    """

    protocol_version: str
    drand_chain_hash: str
    drand_round: int
    timelock_tool_sha256: str
    custody_builder_sha256: str
    commitments: tuple[CustodyCorpusCommitment, ...]
    schema_version: str = CUSTODY_SEAL_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.protocol_version != "0.3.0":
            raise CustodyError("protocol_version must equal '0.3.0'")
        _require_sha256("drand_chain_hash", self.drand_chain_hash)
        _positive_round(self.drand_round)
        _require_sha256("timelock_tool_sha256", self.timelock_tool_sha256)
        _require_sha256("custody_builder_sha256", self.custody_builder_sha256)
        commitments = tuple(self.commitments)
        if not all(isinstance(row, CustodyCorpusCommitment) for row in commitments):
            raise CustodyError("commitments must contain CustodyCorpusCommitment records")
        by_corpus = {row.corpus_id: row for row in commitments}
        if len(by_corpus) != len(commitments) or set(by_corpus) != set(FIXED_CORPORA):
            raise CustodyError("custody commitments must cover every fixed corpus once")
        object.__setattr__(
            self,
            "commitments",
            tuple(by_corpus[corpus_id] for corpus_id in FIXED_CORPORA),
        )
        if self.schema_version != CUSTODY_SEAL_RECEIPT_SCHEMA:
            raise CustodyError(f"schema_version must equal {CUSTODY_SEAL_RECEIPT_SCHEMA!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "commitments": [row.to_dict() for row in self.commitments],
            "custody_builder_sha256": self.custody_builder_sha256,
            "drand_chain_hash": self.drand_chain_hash,
            "drand_round": self.drand_round,
            "protocol_version": self.protocol_version,
            "schema_version": self.schema_version,
            "timelock_tool_sha256": self.timelock_tool_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        """Digest the canonical object, excluding its file newline."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def file_sha256(self) -> str:
        """Digest the exact newline-terminated file pinned by the manifest."""

        return hashlib.sha256(self.canonical_bytes() + b"\n").hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> CustodySealReceipt:
        row = _closed_mapping(
            value,
            fields=_CUSTODY_SEAL_FIELDS,
            label="custody seal receipt",
        )
        commitments = row["commitments"]
        if not isinstance(commitments, Sequence) or isinstance(commitments, (str, bytes)):
            raise CustodyError("custody seal commitments must be an array")
        return cls(
            protocol_version=row["protocol_version"],
            drand_chain_hash=row["drand_chain_hash"],
            drand_round=row["drand_round"],
            timelock_tool_sha256=row["timelock_tool_sha256"],
            custody_builder_sha256=row["custody_builder_sha256"],
            commitments=tuple(CustodyCorpusCommitment.from_dict(item) for item in commitments),
            schema_version=row["schema_version"],
        )


def custody_seal_receipt_from_manifest(
    manifest: Mapping[str, Any],
    *,
    drand_chain_hash: str,
    drand_round: int,
) -> CustodySealReceipt:
    """Build a receipt from already pinned plaintext and ciphertext artifacts."""

    try:
        validate_study_manifest(manifest)
    except StudyManifestError as exc:
        raise CustodyError(f"invalid study manifest: {exc}") from exc
    online = _corpus_artifacts(manifest, "online-execution")
    labels = _corpus_artifacts(manifest, "sealed-labels")
    ciphertexts = _corpus_artifacts(manifest, "sealed-label-ciphertext")
    encryption_receipts = _corpus_artifacts(manifest, "timelock-encryption-receipt")
    timelock = _sole_artifact(manifest, "timelock-tool")
    builder = _sole_artifact(manifest, "custody-builder")
    return CustodySealReceipt(
        protocol_version="0.3.0",
        drand_chain_hash=_require_sha256("drand_chain_hash", drand_chain_hash),
        drand_round=_positive_round(drand_round),
        timelock_tool_sha256=_require_sha256(
            "timelock-tool artifact sha256", timelock.get("sha256")
        ),
        custody_builder_sha256=_require_sha256(
            "custody-builder artifact sha256", builder.get("sha256")
        ),
        commitments=tuple(
            CustodyCorpusCommitment(
                corpus_id=corpus_id,
                online_execution_sha256=_require_sha256(
                    f"online-execution sha256 for {corpus_id}",
                    online[corpus_id].get("sha256"),
                ),
                sealed_label_plaintext_sha256=_require_sha256(
                    f"sealed-labels sha256 for {corpus_id}",
                    labels[corpus_id].get("sha256"),
                ),
                sealed_label_ciphertext_sha256=_require_sha256(
                    f"sealed-label-ciphertext sha256 for {corpus_id}",
                    ciphertexts[corpus_id].get("sha256"),
                ),
                timelock_encryption_receipt_file_sha256=_require_sha256(
                    f"timelock-encryption-receipt sha256 for {corpus_id}",
                    encryption_receipts[corpus_id].get("sha256"),
                ),
            )
            for corpus_id in FIXED_CORPORA
        ),
    )


def verify_custody_seal_receipt(
    receipt: CustodySealReceipt,
    manifest: Mapping[str, Any],
    *,
    require_frozen: bool = True,
    require_manifest_pin: bool = True,
) -> None:
    """Verify receipt content and, by default, its manifest file pin."""

    if not isinstance(receipt, CustodySealReceipt):
        raise CustodyError("receipt must be a CustodySealReceipt")
    try:
        validate_study_manifest(manifest, require_frozen=require_frozen)
    except StudyManifestError as exc:
        raise CustodyError(f"invalid study manifest: {exc}") from exc
    expected = custody_seal_receipt_from_manifest(
        manifest,
        drand_chain_hash=receipt.drand_chain_hash,
        drand_round=receipt.drand_round,
    )
    if receipt != expected:
        raise CustodyError("custody seal receipt commitments differ from the manifest")
    if require_manifest_pin:
        receipt_artifact = _sole_artifact(manifest, "custody-seal-receipt")
        pinned = _require_sha256(
            "custody-seal-receipt artifact sha256",
            receipt_artifact.get("sha256"),
        )
        if receipt.file_sha256 != pinned:
            raise CustodyError("custody seal receipt file digest does not match the manifest pin")


def write_custody_seal_receipt(
    receipt: CustodySealReceipt,
    target: str | Path,
) -> None:
    """Write one canonical receipt exclusively through a no-follow path."""

    if not isinstance(receipt, CustodySealReceipt):
        raise CustodyError("receipt must be a CustodySealReceipt")
    try:
        write_exclusive_receipt_bytes(receipt.canonical_bytes() + b"\n", target)
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"cannot write custody seal receipt: {exc}") from exc


def load_custody_seal_receipt(path: str | Path) -> CustodySealReceipt:
    """Load a canonical receipt without following links or accepting hard links."""

    try:
        encoded = read_secure_control_file(path, label="custody seal receipt")
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"cannot load custody seal receipt: {exc}") from exc
    receipt = CustodySealReceipt.from_dict(_decode_object(encoded, label="custody seal receipt"))
    if encoded != receipt.canonical_bytes() + b"\n":
        raise CustodyError("custody seal receipt bytes must equal canonical JSON plus one newline")
    return receipt


@dataclass(frozen=True)
class OnlineCustodyAdmissionReceipt:
    """Evidence that the online boundary revalidated no plaintext-label role."""

    manifest_sha256: str
    run_receipt_sha256: str
    artifact_verification_receipt_sha256: str
    custody_seal_receipt_sha256: str
    online_artifact_verification_receipt_sha256: str
    runner_identity: str
    verified_artifact_ids: tuple[str, ...]
    schema_version: str = ONLINE_CUSTODY_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "artifact_verification_receipt_sha256",
            "custody_seal_receipt_sha256",
            "online_artifact_verification_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_text("runner_identity", self.runner_identity)
        identifiers = tuple(self.verified_artifact_ids)
        if not identifiers or any(
            not isinstance(identifier, str) or not identifier for identifier in identifiers
        ):
            raise CustodyError("verified_artifact_ids must be non-empty strings")
        canonical = tuple(sorted(identifiers, key=lambda value: value.encode("utf-8")))
        if identifiers != canonical or len(identifiers) != len(set(identifiers)):
            raise CustodyError("verified_artifact_ids must be unique and bytewise sorted")
        if self.schema_version != ONLINE_CUSTODY_ADMISSION_SCHEMA:
            raise CustodyError(f"schema_version must equal {ONLINE_CUSTODY_ADMISSION_SCHEMA!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_verification_receipt_sha256": (self.artifact_verification_receipt_sha256),
            "custody_seal_receipt_sha256": self.custody_seal_receipt_sha256,
            "manifest_sha256": self.manifest_sha256,
            "online_artifact_verification_receipt_sha256": (
                self.online_artifact_verification_receipt_sha256
            ),
            "run_receipt_sha256": self.run_receipt_sha256,
            "runner_identity": self.runner_identity,
            "schema_version": self.schema_version,
            "verified_artifact_ids": list(self.verified_artifact_ids),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> OnlineCustodyAdmissionReceipt:
        row = _closed_mapping(
            value,
            fields=_ONLINE_ADMISSION_FIELDS,
            label="online custody admission receipt",
        )
        identifiers = row["verified_artifact_ids"]
        if not isinstance(identifiers, Sequence) or isinstance(identifiers, (str, bytes)):
            raise CustodyError("verified_artifact_ids must be an array")
        return cls(
            manifest_sha256=row["manifest_sha256"],
            run_receipt_sha256=row["run_receipt_sha256"],
            artifact_verification_receipt_sha256=row["artifact_verification_receipt_sha256"],
            custody_seal_receipt_sha256=row["custody_seal_receipt_sha256"],
            online_artifact_verification_receipt_sha256=row[
                "online_artifact_verification_receipt_sha256"
            ],
            runner_identity=row["runner_identity"],
            verified_artifact_ids=tuple(identifiers),
            schema_version=row["schema_version"],
        )


def write_online_custody_admission_receipt(
    receipt: OnlineCustodyAdmissionReceipt,
    target: str | Path,
) -> None:
    if not isinstance(receipt, OnlineCustodyAdmissionReceipt):
        raise CustodyError("receipt must be an OnlineCustodyAdmissionReceipt")
    try:
        write_exclusive_receipt_bytes(receipt.canonical_bytes() + b"\n", target)
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"cannot write online custody admission receipt: {exc}") from exc


def load_online_custody_admission_receipt(
    path: str | Path,
) -> OnlineCustodyAdmissionReceipt:
    try:
        encoded = read_secure_control_file(
            path,
            label="online custody admission receipt",
        )
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"cannot load online custody admission receipt: {exc}") from exc
    receipt = OnlineCustodyAdmissionReceipt.from_dict(
        _decode_object(encoded, label="online custody admission receipt")
    )
    if encoded != receipt.canonical_bytes() + b"\n":
        raise CustodyError(
            "online custody admission receipt bytes must equal canonical JSON plus one newline"
        )
    return receipt


def _load_secure_manifest(path: str | Path) -> Mapping[str, Any]:
    try:
        encoded = read_secure_control_file(path, label="frozen study manifest")
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"cannot load frozen study manifest: {exc}") from exc
    return _decode_object(encoded, label="frozen study manifest")


def _validate_full_verification_receipt(
    receipt: ArtifactVerificationReceipt,
    *,
    manifest: Mapping[str, Any],
    manifest_digest: str,
) -> None:
    if receipt.manifest_sha256 != manifest_digest:
        raise CustodyError("artifact verification receipt belongs to another manifest")
    pins = {
        str(row["id"]): _require_sha256(
            f"artifact {row['id']!r} sha256",
            row.get("sha256"),
        )
        for row in _artifact_rows(manifest)
    }
    verified = {row.artifact_id: row for row in receipt.artifacts}
    if set(verified) != set(pins):
        raise CustodyError("artifact verification receipt does not cover every manifest artifact")
    for artifact_id, expected in pins.items():
        row = verified[artifact_id]
        if not row.exact or row.expected_sha256 != expected or row.verified_sha256 != expected:
            raise CustodyError(f"artifact verification mismatch for {artifact_id!r}")


def online_custody_artifact_specs(
    manifest: Mapping[str, Any],
    local_artifact_map_path: str | Path,
) -> tuple[LocalArtifactSpec, ...]:
    """Select the exact online-safe local specs without opening excluded paths."""

    try:
        validate_study_manifest(manifest, require_frozen=True)
    except StudyManifestError as exc:
        raise CustodyError(f"invalid frozen study manifest: {exc}") from exc

    pins = {
        str(row["id"]): _require_sha256(
            f"artifact {row['id']!r} sha256",
            row.get("sha256"),
        )
        for row in _artifact_rows(manifest)
    }
    try:
        specs = load_local_artifact_map(
            local_artifact_map_path,
            expected_sha256_by_id=pins,
        )
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"invalid local artifact map: {exc}") from exc
    roles = {str(row["id"]): str(row["role"]) for row in _artifact_rows(manifest)}
    selected = tuple(
        spec for spec in specs if roles[spec.artifact_id] in ONLINE_CUSTODY_REVALIDATION_ROLES
    )
    expected_ids = {
        artifact_id
        for artifact_id, role in roles.items()
        if role in ONLINE_CUSTODY_REVALIDATION_ROLES
    }
    if {spec.artifact_id for spec in selected} != expected_ids:
        raise CustodyError("online custody artifact selection is incomplete")
    if any(roles[spec.artifact_id] == "sealed-labels" for spec in selected):
        raise CustodyError("online custody admission cannot open plaintext labels")
    return selected


def admit_online_custody(
    manifest_path: str | Path,
    *,
    custody_seal_receipt_path: str | Path,
    sealed_run_receipt_path: str | Path,
    artifact_verification_receipt_path: str | Path,
    artifact_root: str | Path,
    local_artifact_map_path: str | Path,
    runner_identity: str,
) -> OnlineCustodyAdmissionReceipt:
    """Revalidate the online-safe artifacts without reading plaintext labels.

    A custodian must already have performed full artifact verification and opened
    the sealed run.  This function validates that evidence, then freshly hashes
    only the roles in :data:`ONLINE_CUSTODY_REVALIDATION_ROLES`.
    """

    manifest = _load_secure_manifest(manifest_path)
    try:
        validate_study_manifest(manifest, require_frozen=True)
    except StudyManifestError as exc:
        raise CustodyError(f"invalid frozen study manifest: {exc}") from exc
    manifest_digest = manifest_sha256(manifest)

    try:
        run_receipt = load_sealed_run_receipt(sealed_run_receipt_path)
    except StudyManifestError as exc:
        raise CustodyError(f"invalid sealed run receipt: {exc}") from exc
    if not isinstance(run_receipt, SealedRunReceipt):  # defensive for integrations
        raise CustodyError("sealed run receipt has the wrong type")
    sealed = manifest["sealed_execution"]
    expected_run_fields = {
        "manifest_sha256": manifest_digest,
        "protocol_version": manifest["protocol_version"],
        "runner_identity": sealed["runner_identity"],
        "code_commit": sealed["code_commit"],
        "runner_image": sealed["runner_image"],
        "receipt_uri": sealed_receipt_uri(manifest),
    }
    for name, expected in expected_run_fields.items():
        if getattr(run_receipt, name) != expected:
            raise CustodyError(f"sealed run receipt {name} differs from the frozen manifest")
    if runner_identity != run_receipt.runner_identity:
        raise CustodyError("runner_identity does not match the sealed run receipt")

    try:
        full_verification = load_verification_receipt(artifact_verification_receipt_path)
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"invalid artifact verification receipt: {exc}") from exc
    _validate_full_verification_receipt(
        full_verification,
        manifest=manifest,
        manifest_digest=manifest_digest,
    )
    verification_path = Path(artifact_verification_receipt_path)
    if (
        not verification_path.is_absolute()
        or verification_path.as_uri() != run_receipt.verification_receipt_uri
        or full_verification.receipt_sha256 != run_receipt.verification_receipt_sha256
    ):
        raise CustodyError("artifact verification receipt does not match the sealed run receipt")

    seal = load_custody_seal_receipt(custody_seal_receipt_path)
    verify_custody_seal_receipt(seal, manifest)

    selected = online_custody_artifact_specs(manifest, local_artifact_map_path)
    try:
        fresh = verify_local_artifacts(
            artifact_root,
            manifest_sha256=manifest_digest,
            artifacts=selected,
        )
    except ArtifactIntegrityError as exc:
        raise CustodyError(f"online-safe artifact revalidation failed: {exc}") from exc
    admitted_by_id = {artifact.artifact_id: artifact for artifact in full_verification.artifacts}
    for artifact in fresh.artifacts:
        if admitted_by_id.get(artifact.artifact_id) != artifact:
            raise CustodyError(
                "online-safe artifact revalidation differs from the custodian receipt "
                f"for {artifact.artifact_id!r}"
            )

    return OnlineCustodyAdmissionReceipt(
        manifest_sha256=manifest_digest,
        run_receipt_sha256=sealed_run_receipt_sha256(run_receipt),
        artifact_verification_receipt_sha256=full_verification.receipt_sha256,
        custody_seal_receipt_sha256=seal.receipt_sha256,
        online_artifact_verification_receipt_sha256=fresh.receipt_sha256,
        runner_identity=runner_identity,
        verified_artifact_ids=tuple(
            sorted(
                (artifact.artifact_id for artifact in fresh.artifacts),
                key=lambda value: value.encode("utf-8"),
            )
        ),
    )
