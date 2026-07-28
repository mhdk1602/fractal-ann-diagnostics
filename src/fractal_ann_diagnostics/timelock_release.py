"""Fail-closed drand timelock release for confirmatory label custody.

The release boundary accepts only the frozen ciphertext as decryption input. It
requires externally verified completion of the whole five-corpus suite as well
as the corpus-specific prediction anchor, re-fetches the exact pinned drand
chain and target-round beacon over HTTPS, runs the pinned ``tle`` binary through
isolated standard streams, and creates the plaintext file exclusively. The
returned capability revalidates that file on every read.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import selectors
import signal
import ssl
import stat
import subprocess
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_regular_file,
    read_secure_control_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .custody import (
    DEFAULT_MAX_CIPHERTEXT_BYTES,
    DEFAULT_MAX_PLAINTEXT_BYTES,
    DEFAULT_TLOCK_TIMEOUT_SECONDS,
    CustodyError,
    CustodySealReceipt,
    TimelockEncryptionReceipt,
    verify_custody_seal_receipt,
    verify_timelock_encryption_receipt,
)
from .external_anchors import VerifiedPredictionCompletionAnchor
from .study import (
    FIXED_CORPORA,
    StudyManifestError,
    manifest_sha256,
    validate_study_manifest,
)

TIMELOCK_DECRYPTION_RECEIPT_SCHEMA = "fractal-timelock-decryption-receipt-v2"
TIMELOCK_DECRYPTION_RECEIPT_FILENAME = "timelock-decryption-receipt.json"
TIMELOCK_RELEASE_INTENT_SCHEMA = "fractal-timelock-release-intent-v1"
TIMELOCK_RELEASE_INTENT_FILENAME = ".timelock-release-intent.json"
MAX_DRAND_CHAIN_METADATA_BYTES = 64 * 1024
MAX_DRAND_BEACON_RESPONSE_BYTES = 64 * 1024
_DRAND_FETCH_TIMEOUT_SECONDS = 10.0
_MAX_TLOCK_STDERR_BYTES = 16 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_LOWER_HEX = re.compile(r"^[0-9a-f]+$")
_RELEASE_CAPABILITY = object()

DrandFetcher = Callable[[str, int], bytes]
TleDecryptRunner = Callable[[Path, tuple[str, ...], bytes, int, int], bytes]
UtcNowFactory = Callable[[], datetime]

_RECEIPT_FIELDS = frozenset(
    {
        "beacon_publication_time_utc",
        "beacon_response_base64",
        "beacon_response_byte_count",
        "beacon_response_sha256",
        "beacon_signature",
        "beacon_uri",
        "chain_genesis_time_utc",
        "chain_metadata_base64",
        "chain_metadata_byte_count",
        "chain_metadata_sha256",
        "chain_metadata_uri",
        "chain_period_seconds",
        "chain_public_key",
        "chain_scheme_id",
        "ciphertext_byte_count",
        "ciphertext_sha256",
        "completed_at_utc",
        "corpus_id",
        "custody_seal_receipt_sha256",
        "drand_chain_hash",
        "drand_network",
        "drand_round",
        "manifest_sha256",
        "label_release_claim_ledger_commit",
        "label_release_claim_state_sha256",
        "label_release_live_execute_job_receipt_sha256",
        "label_release_phase_beacon_receipt_sha256",
        "label_release_phase_claim_contract_sha256",
        "label_release_provider_identity_sha256",
        "online_execution_result_receipt_sha256",
        "plaintext_byte_count",
        "plaintext_sha256",
        "post_online_completion_aggregate_file_sha256",
        "prediction_completion_anchor_receipt_sha256",
        "prediction_completion_anchor_record_sha256",
        "schema_version",
        "started_at_utc",
        "timelock_encryption_receipt_file_sha256",
        "timelock_encryption_receipt_sha256",
        "tle_arguments",
        "tle_binary_sha256",
        "verified_beacon_round",
        "verified_beacon_randomness",
    }
)


class TimelockReleaseError(ValueError):
    """Raised when the label-release ceremony cannot be proved exactly."""


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
        raise TimelockReleaseError("release evidence must be finite canonical JSON") from exc


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise TimelockReleaseError(f"{label} contains duplicate key {key!r}")
            parsed[key] = value
        return parsed

    def reject_nonfinite(value: str) -> None:
        raise TimelockReleaseError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TimelockReleaseError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TimelockReleaseError(f"{label} must contain one JSON object")
    return value


def _closed_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TimelockReleaseError(f"{label} must be a JSON object")
    observed = set(value)
    if observed != fields:
        raise TimelockReleaseError(
            f"{label} schema mismatch; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TimelockReleaseError(f"{name} must be a canonical non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise TimelockReleaseError(f"{name} must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise TimelockReleaseError(f"{name} cannot contain control characters")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TimelockReleaseError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise TimelockReleaseError(f"{name} must be a positive integer")
    return value


def _require_utc_timestamp(name: str, value: object) -> datetime:
    text = _require_text(name, value)
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TimelockReleaseError(f"{name} must use ISO 8601") from exc
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise TimelockReleaseError(f"{name} must use UTC")
    if instant.isoformat() != text:
        raise TimelockReleaseError(f"{name} must use canonical ISO 8601 form")
    return instant


def _utc_now(now_factory: UtcNowFactory) -> datetime:
    try:
        instant = now_factory()
    except Exception as exc:
        raise TimelockReleaseError("UTC clock failed") from exc
    if not isinstance(instant, datetime):
        raise TimelockReleaseError("UTC clock must return datetime")
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise TimelockReleaseError("UTC clock must return a UTC instant")
    return instant.astimezone(timezone.utc)


def _require_https_network(value: object) -> str:
    network = _require_text("drand_network", value).rstrip("/")
    parsed = urlsplit(network)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise TimelockReleaseError(
            "drand_network must be HTTPS without credentials, query, or fragment"
        )
    return network


def _require_lower_hex(name: str, value: object, *, exact_bytes: int | None = None) -> str:
    text = _require_text(name, value)
    if len(text) % 2 or _LOWER_HEX.fullmatch(text) is None:
        raise TimelockReleaseError(f"{name} must be even-length lowercase hexadecimal")
    if exact_bytes is not None and len(text) != exact_bytes * 2:
        raise TimelockReleaseError(f"{name} must encode exactly {exact_bytes} bytes")
    return text


def _decode_base64(name: str, value: object, expected_count: int) -> bytes:
    text = _require_text(name, value)
    try:
        decoded = base64.b64decode(text, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise TimelockReleaseError(f"{name} must be canonical base64") from exc
    if base64.b64encode(decoded).decode("ascii") != text:
        raise TimelockReleaseError(f"{name} must use canonical base64 encoding")
    if len(decoded) != expected_count:
        raise TimelockReleaseError(f"{name} byte count differs from its receipt")
    return decoded


def _decrypt_arguments(network: str, chain_hash: str) -> tuple[str, ...]:
    return (
        "--decrypt",
        f"--network={network}",
        f"--chain={chain_hash}",
    )


@dataclass(frozen=True)
class TimelockDecryptionReceipt:
    """Canonical evidence for one externally gated, pinned-binary release."""

    manifest_sha256: str
    corpus_id: str
    custody_seal_receipt_sha256: str
    timelock_encryption_receipt_sha256: str
    timelock_encryption_receipt_file_sha256: str
    tle_binary_sha256: str
    drand_network: str
    drand_chain_hash: str
    drand_round: int
    chain_metadata_uri: str
    chain_metadata_base64: str
    chain_metadata_sha256: str
    chain_metadata_byte_count: int
    chain_public_key: str
    chain_scheme_id: str
    chain_period_seconds: int
    chain_genesis_time_utc: str
    beacon_uri: str
    beacon_response_base64: str
    beacon_response_sha256: str
    beacon_response_byte_count: int
    beacon_signature: str
    verified_beacon_round: int
    verified_beacon_randomness: str
    beacon_publication_time_utc: str
    ciphertext_sha256: str
    ciphertext_byte_count: int
    plaintext_sha256: str
    plaintext_byte_count: int
    tle_arguments: tuple[str, ...]
    started_at_utc: str
    completed_at_utc: str
    prediction_completion_anchor_record_sha256: str
    prediction_completion_anchor_receipt_sha256: str
    online_execution_result_receipt_sha256: str
    post_online_completion_aggregate_file_sha256: str
    label_release_claim_state_sha256: str
    label_release_claim_ledger_commit: str
    label_release_phase_claim_contract_sha256: str
    label_release_phase_beacon_receipt_sha256: str
    label_release_live_execute_job_receipt_sha256: str
    label_release_provider_identity_sha256: str
    schema_version: str = TIMELOCK_DECRYPTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "custody_seal_receipt_sha256",
            "timelock_encryption_receipt_sha256",
            "timelock_encryption_receipt_file_sha256",
            "tle_binary_sha256",
            "drand_chain_hash",
            "chain_metadata_sha256",
            "beacon_response_sha256",
            "ciphertext_sha256",
            "plaintext_sha256",
            "prediction_completion_anchor_record_sha256",
            "prediction_completion_anchor_receipt_sha256",
            "online_execution_result_receipt_sha256",
            "post_online_completion_aggregate_file_sha256",
            "label_release_claim_state_sha256",
            "label_release_phase_claim_contract_sha256",
            "label_release_phase_beacon_receipt_sha256",
            "label_release_live_execute_job_receipt_sha256",
            "label_release_provider_identity_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if _GIT_COMMIT.fullmatch(self.label_release_claim_ledger_commit) is None:
            raise TimelockReleaseError(
                "label_release_claim_ledger_commit must be one full Git commit"
            )
        if self.corpus_id not in FIXED_CORPORA:
            raise TimelockReleaseError("corpus_id is not registered")
        network = _require_https_network(self.drand_network)
        if network != self.drand_network:
            raise TimelockReleaseError("drand_network cannot have a trailing slash")
        round_number = _require_positive_integer("drand_round", self.drand_round)
        verified_round = _require_positive_integer(
            "verified_beacon_round", self.verified_beacon_round
        )
        if verified_round != round_number:
            raise TimelockReleaseError("verified beacon round differs from the release round")
        for name in (
            "chain_metadata_byte_count",
            "chain_period_seconds",
            "beacon_response_byte_count",
            "ciphertext_byte_count",
            "plaintext_byte_count",
        ):
            _require_positive_integer(name, getattr(self, name))
        metadata = _decode_base64(
            "chain_metadata_base64",
            self.chain_metadata_base64,
            self.chain_metadata_byte_count,
        )
        beacon = _decode_base64(
            "beacon_response_base64",
            self.beacon_response_base64,
            self.beacon_response_byte_count,
        )
        if hashlib.sha256(metadata).hexdigest() != self.chain_metadata_sha256:
            raise TimelockReleaseError("chain metadata digest differs from its exact bytes")
        if hashlib.sha256(beacon).hexdigest() != self.beacon_response_sha256:
            raise TimelockReleaseError("beacon response digest differs from its exact bytes")
        _require_lower_hex("chain_public_key", self.chain_public_key)
        _require_text("chain_scheme_id", self.chain_scheme_id)
        _require_lower_hex("beacon_signature", self.beacon_signature)
        _require_lower_hex(
            "verified_beacon_randomness",
            self.verified_beacon_randomness,
            exact_bytes=32,
        )
        genesis = _require_utc_timestamp("chain_genesis_time_utc", self.chain_genesis_time_utc)
        publication = _require_utc_timestamp(
            "beacon_publication_time_utc", self.beacon_publication_time_utc
        )
        expected_publication = genesis + timedelta(
            seconds=(round_number - 1) * self.chain_period_seconds
        )
        if publication != expected_publication:
            raise TimelockReleaseError(
                "beacon publication time differs from chain genesis, period, and round"
            )
        started = _require_utc_timestamp("started_at_utc", self.started_at_utc)
        completed = _require_utc_timestamp("completed_at_utc", self.completed_at_utc)
        if started < publication:
            raise TimelockReleaseError("decryption started before target-round publication")
        if completed < started:
            raise TimelockReleaseError("decryption completion predates its start")
        metadata_uri = f"{network}/{self.drand_chain_hash}/info"
        beacon_uri = f"{network}/{self.drand_chain_hash}/public/{round_number}"
        if self.chain_metadata_uri != metadata_uri or self.beacon_uri != beacon_uri:
            raise TimelockReleaseError("drand evidence URIs are not exact pinned endpoints")
        evidence_by_uri = {
            metadata_uri: metadata,
            beacon_uri: beacon,
        }

        def embedded_evidence_fetcher(uri: str, max_bytes: int) -> bytes:
            try:
                encoded = evidence_by_uri[uri]
            except KeyError as exc:
                raise TimelockReleaseError("receipt requested an unbound drand URI") from exc
            if len(encoded) > max_bytes:
                raise TimelockReleaseError("embedded drand evidence exceeds its byte limit")
            return encoded

        embedded = _fetch_and_verify_drand_evidence(
            network=network,
            chain_hash=self.drand_chain_hash,
            round_number=round_number,
            fetcher=embedded_evidence_fetcher,
        )
        embedded_bindings = (
            ("chain_public_key", embedded.public_key, self.chain_public_key),
            ("chain_scheme_id", embedded.scheme_id, self.chain_scheme_id),
            ("chain_period_seconds", embedded.period_seconds, self.chain_period_seconds),
            ("chain_genesis_time_utc", embedded.genesis_time, genesis),
            ("beacon_signature", embedded.signature, self.beacon_signature),
            (
                "verified_beacon_randomness",
                embedded.randomness,
                self.verified_beacon_randomness,
            ),
            ("beacon_publication_time_utc", embedded.publication_time, publication),
        )
        for name, observed, expected in embedded_bindings:
            if observed != expected:
                raise TimelockReleaseError(
                    f"{name} differs from the embedded authenticated response bytes"
                )
        arguments = tuple(self.tle_arguments)
        if arguments != _decrypt_arguments(network, self.drand_chain_hash):
            raise TimelockReleaseError(
                "tle_arguments must contain only decrypt, network, and chain flags"
            )
        object.__setattr__(self, "tle_arguments", arguments)
        if self.schema_version != TIMELOCK_DECRYPTION_RECEIPT_SCHEMA:
            raise TimelockReleaseError(
                f"schema_version must equal {TIMELOCK_DECRYPTION_RECEIPT_SCHEMA!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "beacon_publication_time_utc": self.beacon_publication_time_utc,
            "beacon_response_base64": self.beacon_response_base64,
            "beacon_response_byte_count": self.beacon_response_byte_count,
            "beacon_response_sha256": self.beacon_response_sha256,
            "beacon_signature": self.beacon_signature,
            "beacon_uri": self.beacon_uri,
            "chain_genesis_time_utc": self.chain_genesis_time_utc,
            "chain_metadata_base64": self.chain_metadata_base64,
            "chain_metadata_byte_count": self.chain_metadata_byte_count,
            "chain_metadata_sha256": self.chain_metadata_sha256,
            "chain_metadata_uri": self.chain_metadata_uri,
            "chain_period_seconds": self.chain_period_seconds,
            "chain_public_key": self.chain_public_key,
            "chain_scheme_id": self.chain_scheme_id,
            "ciphertext_byte_count": self.ciphertext_byte_count,
            "ciphertext_sha256": self.ciphertext_sha256,
            "completed_at_utc": self.completed_at_utc,
            "corpus_id": self.corpus_id,
            "custody_seal_receipt_sha256": self.custody_seal_receipt_sha256,
            "drand_chain_hash": self.drand_chain_hash,
            "drand_network": self.drand_network,
            "drand_round": self.drand_round,
            "label_release_claim_ledger_commit": (self.label_release_claim_ledger_commit),
            "label_release_claim_state_sha256": self.label_release_claim_state_sha256,
            "label_release_live_execute_job_receipt_sha256": (
                self.label_release_live_execute_job_receipt_sha256
            ),
            "label_release_phase_beacon_receipt_sha256": (
                self.label_release_phase_beacon_receipt_sha256
            ),
            "label_release_phase_claim_contract_sha256": (
                self.label_release_phase_claim_contract_sha256
            ),
            "label_release_provider_identity_sha256": (self.label_release_provider_identity_sha256),
            "manifest_sha256": self.manifest_sha256,
            "online_execution_result_receipt_sha256": (self.online_execution_result_receipt_sha256),
            "plaintext_byte_count": self.plaintext_byte_count,
            "plaintext_sha256": self.plaintext_sha256,
            "post_online_completion_aggregate_file_sha256": (
                self.post_online_completion_aggregate_file_sha256
            ),
            "prediction_completion_anchor_receipt_sha256": (
                self.prediction_completion_anchor_receipt_sha256
            ),
            "prediction_completion_anchor_record_sha256": (
                self.prediction_completion_anchor_record_sha256
            ),
            "schema_version": self.schema_version,
            "started_at_utc": self.started_at_utc,
            "timelock_encryption_receipt_file_sha256": (
                self.timelock_encryption_receipt_file_sha256
            ),
            "timelock_encryption_receipt_sha256": (self.timelock_encryption_receipt_sha256),
            "tle_arguments": list(self.tle_arguments),
            "tle_binary_sha256": self.tle_binary_sha256,
            "verified_beacon_randomness": self.verified_beacon_randomness,
            "verified_beacon_round": self.verified_beacon_round,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> TimelockDecryptionReceipt:
        row = _closed_mapping(
            value,
            fields=_RECEIPT_FIELDS,
            label="timelock decryption receipt",
        )
        arguments = row["tle_arguments"]
        if not isinstance(arguments, Sequence) or isinstance(arguments, (str, bytes)):
            raise TimelockReleaseError("tle_arguments must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "tle_arguments"},
            tle_arguments=tuple(arguments),
        )


@dataclass(frozen=True)
class VerifiedTimelockRelease:
    """Unforgeable-by-construction token for one revalidated plaintext output."""

    receipt: TimelockDecryptionReceipt
    plaintext_path: Path
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _RELEASE_CAPABILITY:
            raise TimelockReleaseError(
                "VerifiedTimelockRelease can only be created by the release verifier"
            )
        if not isinstance(self.receipt, TimelockDecryptionReceipt):
            raise TimelockReleaseError("receipt must be a TimelockDecryptionReceipt")
        path = Path(self.plaintext_path)
        if not path.is_absolute():
            raise TimelockReleaseError("plaintext_path must be absolute")
        object.__setattr__(self, "plaintext_path", path)
        self.read_plaintext()

    def read_plaintext(self) -> bytes:
        """Re-read and rehash the exclusive plaintext before label use."""

        try:
            encoded = read_secure_regular_file(
                self.plaintext_path,
                max_bytes=self.receipt.plaintext_byte_count,
                label="released plaintext labels",
            )
        except ArtifactIntegrityError as exc:
            raise TimelockReleaseError(f"cannot revalidate released plaintext: {exc}") from exc
        if len(encoded) != self.receipt.plaintext_byte_count:
            raise TimelockReleaseError("released plaintext byte count changed")
        if not hmac.compare_digest(
            hashlib.sha256(encoded).hexdigest(),
            self.receipt.plaintext_sha256,
        ):
            raise TimelockReleaseError("released plaintext digest changed")
        return encoded


def write_timelock_decryption_receipt(
    receipt: TimelockDecryptionReceipt,
    target: str | Path,
) -> None:
    if not isinstance(receipt, TimelockDecryptionReceipt):
        raise TimelockReleaseError("receipt must be a TimelockDecryptionReceipt")
    try:
        write_exclusive_receipt_bytes(receipt.canonical_bytes() + b"\n", target)
    except ArtifactIntegrityError as exc:
        raise TimelockReleaseError(f"cannot write decryption receipt: {exc}") from exc


def load_timelock_decryption_receipt(path: str | Path) -> TimelockDecryptionReceipt:
    try:
        encoded = read_secure_control_file(path, label="timelock decryption receipt")
    except ArtifactIntegrityError as exc:
        raise TimelockReleaseError(f"cannot load decryption receipt: {exc}") from exc
    receipt = TimelockDecryptionReceipt.from_dict(
        _decode_object(encoded, label="timelock decryption receipt")
    )
    if encoded != receipt.canonical_bytes() + b"\n":
        raise TimelockReleaseError(
            "decryption receipt bytes must equal canonical JSON plus one newline"
        )
    return receipt


def label_release_staging_directory_name(corpus_id: str) -> str:
    """Return the sole transaction directory allowed beside a corpus output."""

    if corpus_id not in FIXED_CORPORA:
        raise TimelockReleaseError("label release stage names an unregistered corpus")
    return f".{corpus_id}.timelock-release-stage"


def _release_directory_entries(root: Path, *, label: str) -> frozenset[str]:
    try:
        opened = root.lstat()
    except OSError as exc:
        raise TimelockReleaseError(f"cannot inspect {label}") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or stat.S_ISLNK(opened.st_mode)
        or opened.st_uid != os.getuid()
        or stat.S_IMODE(opened.st_mode) & 0o077
    ):
        raise TimelockReleaseError(f"{label} is not a private owned directory")
    try:
        return frozenset(path.name for path in root.iterdir())
    except OSError as exc:
        raise TimelockReleaseError(f"cannot enumerate {label}") from exc


def _release_intent_bytes(
    *,
    corpus_id: str,
    manifest_digest: str,
    claim_state_sha256: str,
    claim_ledger_commit: str,
    claim_contract_sha256: str,
    phase_beacon_identity_sha256: str,
    live_execute_job_identity_sha256: str,
    phase_beacon_receipt: object,
    live_execute_job_receipt: object,
    provider_identity_sha256: str,
    post_online_aggregate_file_sha256: str,
    tle_binary_sha256: str,
    ciphertext_sha256: str,
    drand_round: int,
    plaintext_output_uri: str,
    receipt_output_uri: str,
) -> bytes:
    return (
        _canonical_bytes(
            {
                "ciphertext_sha256": ciphertext_sha256,
                "corpus_id": corpus_id,
                "drand_round": drand_round,
                "label_release_claim_ledger_commit": claim_ledger_commit,
                "label_release_claim_state_sha256": claim_state_sha256,
                "label_release_live_execute_job_identity_sha256": (
                    live_execute_job_identity_sha256
                ),
                "label_release_live_execute_job_receipt": (live_execute_job_receipt.to_dict()),
                "label_release_phase_beacon_identity_sha256": (phase_beacon_identity_sha256),
                "label_release_phase_beacon_receipt": (phase_beacon_receipt.to_dict()),
                "label_release_phase_claim_contract_sha256": (claim_contract_sha256),
                "label_release_provider_identity_sha256": provider_identity_sha256,
                "manifest_sha256": manifest_digest,
                "plaintext_output_uri": plaintext_output_uri,
                "post_online_completion_aggregate_file_sha256": (post_online_aggregate_file_sha256),
                "receipt_output_uri": receipt_output_uri,
                "schema_version": TIMELOCK_RELEASE_INTENT_SCHEMA,
                "tle_binary_sha256": tle_binary_sha256,
            }
        )
        + b"\n"
    )


def _release_intent_stable_identity(encoded: bytes) -> bytes:
    try:
        from .execution_claim import LiveExecuteJobReceipt, PhaseBeaconReceipt
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise TimelockReleaseError("release intent action verifier is unavailable") from exc
    row = _decode_object(encoded, label="timelock release intent")
    expected_fields = frozenset(
        {
            "ciphertext_sha256",
            "corpus_id",
            "drand_round",
            "label_release_claim_ledger_commit",
            "label_release_claim_state_sha256",
            "label_release_live_execute_job_identity_sha256",
            "label_release_live_execute_job_receipt",
            "label_release_phase_beacon_identity_sha256",
            "label_release_phase_beacon_receipt",
            "label_release_phase_claim_contract_sha256",
            "label_release_provider_identity_sha256",
            "manifest_sha256",
            "plaintext_output_uri",
            "post_online_completion_aggregate_file_sha256",
            "receipt_output_uri",
            "schema_version",
            "tle_binary_sha256",
        }
    )
    closed = _closed_mapping(
        row,
        fields=expected_fields,
        label="timelock release intent",
    )
    try:
        live = LiveExecuteJobReceipt(
            **_closed_mapping(
                closed["label_release_live_execute_job_receipt"],
                fields=frozenset(LiveExecuteJobReceipt.__dataclass_fields__),
                label="timelock release live execute-job receipt",
            )
        )
        beacon = PhaseBeaconReceipt.from_dict(closed["label_release_phase_beacon_receipt"])
    except Exception as exc:
        raise TimelockReleaseError(
            "timelock release intent contains invalid action evidence"
        ) from exc
    if (
        live.job_identity_sha256 != closed["label_release_live_execute_job_identity_sha256"]
        or beacon.beacon_identity_sha256 != closed["label_release_phase_beacon_identity_sha256"]
    ):
        raise TimelockReleaseError(
            "timelock release intent action identity differs from its evidence"
        )
    normalized = {
        key: value
        for key, value in closed.items()
        if key
        not in {
            "label_release_live_execute_job_receipt",
            "label_release_phase_beacon_receipt",
        }
    }
    return _canonical_bytes(normalized) + b"\n"


def _prepare_release_transaction(
    release_root: Path,
    *,
    intent_bytes: bytes,
) -> Path:
    """Durably record the exact action before the TLE process may start."""

    stage = release_root.parent / label_release_staging_directory_name(release_root.name)
    if os.path.lexists(release_root) or os.path.lexists(stage):
        raise TimelockReleaseError("label release target changed before decryption intent")
    try:
        stage.mkdir(mode=0o700, parents=False, exist_ok=False)
        write_exclusive_receipt_bytes(
            intent_bytes,
            stage / TIMELOCK_RELEASE_INTENT_FILENAME,
        )
        descriptor = os.open(
            stage,
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_descriptor = os.open(
            release_root.parent,
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except (ArtifactIntegrityError, OSError) as exc:
        raise TimelockReleaseError("cannot durably record pre-decryption intent") from exc
    return stage


def _rename_release_stage(stage: Path, release_root: Path) -> None:
    parent = release_root.parent
    try:
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.rename(
                stage.name,
                release_root.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError as exc:
        raise TimelockReleaseError("cannot atomically commit label release transaction") from exc


def _publish_release_pair(
    *,
    plaintext: bytes,
    receipt: TimelockDecryptionReceipt,
    plaintext_path: Path,
    receipt_path: Path,
    intent_bytes: bytes,
) -> None:
    """Publish plaintext and its receipt through one atomic directory rename."""

    if (
        not plaintext_path.is_absolute()
        or not receipt_path.is_absolute()
        or plaintext_path.parent != receipt_path.parent
        or plaintext_path.name == receipt_path.name
    ):
        raise TimelockReleaseError(
            "plaintext and decryption receipt must be distinct files in one release directory"
        )
    release_root = plaintext_path.parent
    parent = release_root.parent
    stage = parent / label_release_staging_directory_name(release_root.name)
    if os.path.lexists(release_root) or not os.path.lexists(stage):
        raise TimelockReleaseError("label release target changed after transaction admission")
    try:
        if (
            _release_directory_entries(stage, label="label release stage")
            != frozenset({TIMELOCK_RELEASE_INTENT_FILENAME})
            or read_secure_control_file(
                stage / TIMELOCK_RELEASE_INTENT_FILENAME,
                label="timelock release intent",
            )
            != intent_bytes
        ):
            raise TimelockReleaseError("pre-decryption intent changed before publication")
        staged_plaintext = stage / plaintext_path.name
        staged_receipt = stage / receipt_path.name
        write_exclusive_receipt_bytes(plaintext, staged_plaintext)
        write_timelock_decryption_receipt(receipt, staged_receipt)
        if (
            read_secure_regular_file(
                staged_plaintext,
                max_bytes=receipt.plaintext_byte_count,
                label="staged released plaintext",
            )
            != plaintext
            or load_timelock_decryption_receipt(staged_receipt) != receipt
            or {path.name for path in stage.iterdir()}
            != {
                plaintext_path.name,
                receipt_path.name,
                TIMELOCK_RELEASE_INTENT_FILENAME,
            }
        ):
            raise TimelockReleaseError("staged label release pair failed exact readback")
        _rename_release_stage(stage, release_root)
    except TimelockReleaseError:
        raise
    except (ArtifactIntegrityError, OSError) as exc:
        raise TimelockReleaseError(
            "cannot atomically publish plaintext and decryption receipt"
        ) from exc


class _NoRedirects(urllib_request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib_request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def _fetch_drand_bytes(uri: str, max_bytes: int) -> bytes:
    if max_bytes <= 0 or max_bytes > MAX_DRAND_CHAIN_METADATA_BYTES:
        raise TimelockReleaseError("drand fetch exceeds the fixed safety limit")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise TimelockReleaseError("drand evidence URI must be exact HTTPS")
    context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    opener = urllib_request.build_opener(
        _NoRedirects(),
        urllib_request.HTTPSHandler(context=context),
    )
    request = urllib_request.Request(
        uri,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "fractal-ann-diagnostics/0.3 timelock-release",
        },
        method="GET",
    )
    try:
        with opener.open(request, timeout=_DRAND_FETCH_TIMEOUT_SECONDS) as response:
            status = response.getcode()
            if status != 200:
                if isinstance(status, int) and 300 <= status < 400:
                    raise TimelockReleaseError("drand fetch refused an HTTP redirect")
                raise TimelockReleaseError(f"drand fetch returned HTTP status {status}")
            if response.geturl() != uri:
                raise TimelockReleaseError("drand response URL changed")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                if not content_length.isdecimal():
                    raise TimelockReleaseError("drand response has invalid Content-Length")
                if int(content_length) > max_bytes:
                    raise TimelockReleaseError("drand response exceeds its byte limit")
            encoded = response.read(max_bytes + 1)
    except TimelockReleaseError:
        raise
    except urllib_error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise TimelockReleaseError("drand fetch refused an HTTP redirect") from exc
        raise TimelockReleaseError(f"drand fetch returned HTTP status {exc.code}") from exc
    except (OSError, TimeoutError, urllib_error.URLError, ValueError) as exc:
        raise TimelockReleaseError("drand evidence could not be fetched over HTTPS") from exc
    if not isinstance(encoded, bytes) or len(encoded) > max_bytes:
        raise TimelockReleaseError("drand response exceeds its byte limit")
    return encoded


@dataclass(frozen=True)
class _DrandEvidence:
    chain_metadata: bytes
    beacon_response: bytes
    public_key: str
    scheme_id: str
    period_seconds: int
    genesis_time: datetime
    signature: str
    randomness: str
    publication_time: datetime


def _fetch_and_verify_drand_evidence(
    *,
    network: str,
    chain_hash: str,
    round_number: int,
    fetcher: DrandFetcher,
) -> _DrandEvidence:
    metadata_uri = f"{network}/{chain_hash}/info"
    beacon_uri = f"{network}/{chain_hash}/public/{round_number}"
    try:
        metadata_bytes = fetcher(metadata_uri, MAX_DRAND_CHAIN_METADATA_BYTES)
        beacon_bytes = fetcher(beacon_uri, MAX_DRAND_BEACON_RESPONSE_BYTES)
    except TimelockReleaseError:
        raise
    except Exception as exc:
        raise TimelockReleaseError("trusted drand fetcher failed") from exc
    if not isinstance(metadata_bytes, bytes) or not metadata_bytes:
        raise TimelockReleaseError("drand chain metadata must be non-empty bytes")
    if not isinstance(beacon_bytes, bytes) or not beacon_bytes:
        raise TimelockReleaseError("drand beacon response must be non-empty bytes")
    if len(metadata_bytes) > MAX_DRAND_CHAIN_METADATA_BYTES:
        raise TimelockReleaseError("drand chain metadata exceeds its byte limit")
    if len(beacon_bytes) > MAX_DRAND_BEACON_RESPONSE_BYTES:
        raise TimelockReleaseError("drand beacon response exceeds its byte limit")

    metadata = _decode_object(metadata_bytes, label="drand chain metadata")
    required_metadata = {
        "public_key",
        "period",
        "genesis_time",
        "hash",
        "schemeID",
    }
    unknown_metadata = set(metadata) - {
        *required_metadata,
        "groupHash",
        "metadata",
    }
    if required_metadata - set(metadata) or unknown_metadata:
        raise TimelockReleaseError("drand chain metadata does not match the admitted schema")
    observed_chain = _require_sha256("drand chain metadata hash", metadata["hash"])
    if observed_chain != chain_hash:
        raise TimelockReleaseError("drand chain metadata differs from the pinned chain")
    public_key = _require_lower_hex("drand public_key", metadata["public_key"])
    scheme_id = _require_text("drand schemeID", metadata["schemeID"])
    if "unchained" not in scheme_id.lower():
        raise TimelockReleaseError("timelock release requires an unchained drand scheme")
    period = _require_positive_integer("drand period", metadata["period"])
    genesis_epoch = _require_positive_integer("drand genesis_time", metadata["genesis_time"])
    try:
        genesis = datetime.fromtimestamp(genesis_epoch, tz=timezone.utc)
        publication = genesis + timedelta(seconds=(round_number - 1) * period)
    except (OverflowError, OSError) as exc:
        raise TimelockReleaseError("drand publication time is outside datetime range") from exc

    beacon = _decode_object(beacon_bytes, label="drand beacon response")
    required_beacon = {"round", "randomness", "signature"}
    if required_beacon - set(beacon) or set(beacon) - {
        *required_beacon,
        "previous_signature",
    }:
        raise TimelockReleaseError("drand beacon response does not match the admitted schema")
    observed_round = _require_positive_integer("drand beacon round", beacon["round"])
    if observed_round != round_number:
        raise TimelockReleaseError("drand beacon response is for another round")
    signature = _require_lower_hex("drand beacon signature", beacon["signature"])
    randomness = _require_lower_hex("drand beacon randomness", beacon["randomness"], exact_bytes=32)
    derived_randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
    if not hmac.compare_digest(derived_randomness, randomness):
        raise TimelockReleaseError(
            "drand beacon randomness is inconsistent with the signed response"
        )
    return _DrandEvidence(
        chain_metadata=metadata_bytes,
        beacon_response=beacon_bytes,
        public_key=public_key,
        scheme_id=scheme_id,
        period_seconds=period,
        genesis_time=genesis,
        signature=signature,
        randomness=randomness,
        publication_time=publication,
    )


def _admit_tle_binary(path: str | Path, *, expected_sha256: str) -> Path:
    target = Path(path)
    if not target.is_absolute() or target.name in {"", ".", ".."}:
        raise TimelockReleaseError("tle_binary_path must be absolute")
    try:
        observed = digest_regular_file(target, label="tle binary")
        metadata = target.lstat()
        parent_metadata = target.parent.stat()
    except (ArtifactIntegrityError, OSError) as exc:
        raise TimelockReleaseError(f"cannot admit tle binary: {exc}") from exc
    if not hmac.compare_digest(observed, expected_sha256):
        raise TimelockReleaseError("tle binary digest differs from the manifest pin")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise TimelockReleaseError("tle binary must be one regular, singly linked file")
    if metadata.st_mode & 0o022 or parent_metadata.st_mode & 0o022:
        raise TimelockReleaseError("tle binary and parent cannot be group/other writable")
    if not metadata.st_mode & stat.S_IXUSR:
        raise TimelockReleaseError("tle binary must be executable by its owner")
    if hasattr(os, "geteuid") and (
        metadata.st_uid != os.geteuid() or parent_metadata.st_uid != os.geteuid()
    ):
        raise TimelockReleaseError("tle binary and parent must be owned by the custodian identity")
    return target


def _close_stream(selector: selectors.BaseSelector, stream: Any) -> None:
    try:
        selector.unregister(stream)
    except (KeyError, ValueError):
        pass
    try:
        stream.close()
    except OSError:
        pass


def _kill_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        process.kill()


def _run_pinned_tle_decrypt(
    binary: Path,
    arguments: tuple[str, ...],
    ciphertext: bytes,
    timeout_seconds: int,
    max_plaintext_bytes: int,
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
        raise TimelockReleaseError(f"cannot execute pinned tle binary: {exc}") from exc
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _kill_process(process)
        process.wait()
        raise TimelockReleaseError("tle process did not expose isolated standard streams")

    selector = selectors.DefaultSelector()
    plaintext = bytearray()
    stderr_byte_count = 0
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
                raise TimelockReleaseError(
                    f"pinned tle decryption exceeded {timeout_seconds} seconds"
                )
            for key, _ in selector.select(min(remaining, 0.25)):
                stream = key.fileobj
                if key.data == "stdin":
                    if input_offset == len(ciphertext):
                        _close_stream(selector, stream)
                        continue
                    try:
                        written = os.write(
                            stream.fileno(),
                            ciphertext[input_offset : input_offset + 64 * 1024],
                        )
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        _close_stream(selector, stream)
                        continue
                    input_offset += written
                    continue
                try:
                    chunk = os.read(stream.fileno(), 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    _close_stream(selector, stream)
                elif key.data == "stdout":
                    if len(plaintext) + len(chunk) > max_plaintext_bytes:
                        raise TimelockReleaseError(
                            "pinned tle plaintext exceeds max_plaintext_bytes"
                        )
                    plaintext.extend(chunk)
                else:
                    stderr_byte_count = min(
                        stderr_byte_count + len(chunk),
                        _MAX_TLOCK_STDERR_BYTES + 1,
                    )
        try:
            return_code = process.wait(timeout=max(deadline - time.monotonic(), 0.01))
        except subprocess.TimeoutExpired as exc:
            raise TimelockReleaseError(
                f"pinned tle decryption exceeded {timeout_seconds} seconds"
            ) from exc
    except BaseException:
        _kill_process(process)
        process.wait()
        raise
    finally:
        for stream in streams:
            if not stream.closed:
                _close_stream(selector, stream)
        selector.close()
    if return_code != 0:
        stderr_state = "no stderr" if stderr_byte_count == 0 else "stderr suppressed"
        raise TimelockReleaseError(
            f"pinned tle decryption failed with exit {return_code}: {stderr_state}"
        )
    if input_offset != len(ciphertext):
        raise TimelockReleaseError("tle process closed stdin before reading the ciphertext")
    if not plaintext:
        raise TimelockReleaseError("pinned tle decryption produced empty plaintext")
    return bytes(plaintext)


def _artifact_pin(
    manifest: Mapping[str, Any],
    role: str,
    *,
    corpus_id: str | None = None,
) -> str:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise TimelockReleaseError("manifest artifacts must be an array")
    matches = [
        row
        for row in artifacts
        if isinstance(row, Mapping)
        and row.get("role") == role
        and (corpus_id is None or row.get("corpus_id") == corpus_id)
    ]
    if len(matches) != 1:
        raise TimelockReleaseError(f"manifest must contain one {role!r} artifact")
    return _require_sha256(f"{role} artifact sha256", matches[0].get("sha256"))


def _require_suite_online_completion(
    token: object,
    *,
    manifest_digest: str,
    corpus_id: str,
    online_result_receipt_sha256: str,
) -> None:
    """Admit release only from freshly verified all-five suite authority."""

    try:
        from .suite_attempt import SuiteAttemptError, require_verified_online_completion
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise TimelockReleaseError("suite attempt verifier is unavailable") from exc
    try:
        require_verified_online_completion(
            token,
            manifest_digest=manifest_digest,
            corpus_id=corpus_id,
            online_result_receipt_sha256=online_result_receipt_sha256,
        )
    except SuiteAttemptError as exc:
        raise TimelockReleaseError(f"suite completion gate failed: {exc}") from exc


def _require_post_online_completion(
    token: object,
    *,
    corpus_id: str,
    manifest_digest: str,
    verified_phase_claim: object,
    verified_suite_completion: object,
) -> tuple[VerifiedPredictionCompletionAnchor, object]:
    """Admit one anchor only from the unforgeable all-five public token."""

    try:
        from .execution_claim import VerifiedPhaseClaimCapability
        from .post_online_completion import VerifiedPostOnlineCompletionAuthority
        from .suite_attempt import VerifiedProviderPredecessor
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise TimelockReleaseError("post-online completion verifier is unavailable") from exc
    if not isinstance(token, VerifiedPostOnlineCompletionAuthority):
        raise TimelockReleaseError(
            "label release requires provider-bound post-online completion authority"
        )
    if not isinstance(verified_phase_claim, VerifiedPhaseClaimCapability):
        raise TimelockReleaseError(
            "post-online completion lacks a verified provider phase capability"
        )
    if (
        not isinstance(verified_suite_completion, VerifiedProviderPredecessor)
        or token.provider_namespace != verified_suite_completion.namespace
    ):
        raise TimelockReleaseError("post-online completion belongs to another provider namespace")
    aggregate = token.aggregate
    if (
        aggregate.manifest_sha256 != manifest_digest
        or aggregate.run_receipt_sha256 != verified_phase_claim.contract.run_receipt_sha256
        or aggregate.label_release_claim_state_sha256
        != verified_phase_claim.phase_claim_state_sha256
        or aggregate.label_release_claim_ledger_commit
        != verified_phase_claim.phase_claim_ledger_commit
    ):
        raise TimelockReleaseError(
            "post-online completion belongs to another label-release authority"
        )
    try:
        anchor = token.anchor_for(corpus_id)
    except Exception as exc:
        raise TimelockReleaseError(
            "post-online completion lacks the requested corpus anchor"
        ) from exc
    return anchor, aggregate


def _require_phase_release_authority(
    token: object,
    *,
    manifest_digest: str,
    run_receipt_sha256: str,
) -> object:
    """Revalidate the unforgeable provider phase capability in place."""

    try:
        from .execution_claim import (
            LABEL_RELEASE_PHASE,
            ExecutionClaimError,
            PhaseBeaconReceipt,
            VerifiedPhaseClaimCapability,
        )
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise TimelockReleaseError("provider phase authority verifier is unavailable") from exc
    if not isinstance(token, VerifiedPhaseClaimCapability):
        raise TimelockReleaseError("label release requires a verified provider phase capability")
    if (
        token.contract.phase != LABEL_RELEASE_PHASE
        or token.contract.manifest_sha256 != manifest_digest
        or token.contract.run_receipt_sha256 != run_receipt_sha256
    ):
        raise TimelockReleaseError("provider phase capability belongs to another release")
    try:
        token.assert_current()
    except ExecutionClaimError as exc:
        raise TimelockReleaseError("provider phase capability is no longer current") from exc
    beacon = token.phase_beacon_receipt
    contract = token.contract.label_release_beacon
    if not isinstance(beacon, PhaseBeaconReceipt) or contract is None:
        raise TimelockReleaseError(
            "provider phase capability lacks its verified label-release beacon"
        )
    if (
        beacon.phase != LABEL_RELEASE_PHASE
        or beacon.phase_claim_state_sha256 != token.phase_claim_state_sha256
        or beacon.phase_claim_ledger_commit != token.phase_claim_ledger_commit
        or beacon.phase_claim_contract_sha256 != token.contract.contract_sha256
        or beacon.beacon_contract_sha256 != contract.contract_sha256
        or beacon.chain_hash != contract.chain_hash
        or beacon.round != contract.label_release_round
        or beacon.published_at_utc != contract.label_release_publication_time.isoformat()
    ):
        raise TimelockReleaseError(
            "provider phase capability carries a mismatched label-release beacon"
        )
    return beacon


def _require_beacon_evidence_matches_phase_authority(
    evidence: _DrandEvidence,
    phase_beacon: object,
) -> None:
    """Bind the fetched release bytes to the BLS-verified provider receipt."""

    try:
        from .execution_claim import PhaseBeaconReceipt
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise TimelockReleaseError("provider phase beacon verifier is unavailable") from exc
    if not isinstance(phase_beacon, PhaseBeaconReceipt):
        raise TimelockReleaseError("label release lacks a verified provider beacon receipt")
    if (
        hashlib.sha256(evidence.beacon_response).hexdigest() != phase_beacon.beacon_bytes_sha256
        or evidence.signature != phase_beacon.signature
        or evidence.randomness != phase_beacon.randomness
        or evidence.publication_time.isoformat() != phase_beacon.published_at_utc
    ):
        raise TimelockReleaseError(
            "fresh drand evidence differs from the BLS-verified provider beacon"
        )


def _require_existing_release_authority(
    receipt: TimelockDecryptionReceipt,
    *,
    manifest_digest: str,
    corpus_id: str,
    custody_seal: CustodySealReceipt,
    encryption_receipt: TimelockEncryptionReceipt,
    completion_aggregate: object,
    anchor_record: object,
    anchor_receipt: object,
    verified_phase_claim: object,
    phase_beacon: object,
    network: str,
    chain_hash: str,
    round_number: int,
    tool_pin: str,
    ciphertext_digest: str,
    ciphertext_byte_count: int,
    plaintext_pin: str,
) -> None:
    """Admit a prior committed pair against current stable release authority.

    The live-job and phase-beacon receipt hashes include their action timestamp.
    A retry therefore preserves those two original action identities while
    revalidating every stable provider, claim, beacon, and post-online binding.
    """

    try:
        from .execution_claim import PhaseBeaconReceipt, VerifiedPhaseClaimCapability
        from .post_online_completion import PostOnlineCompletionAggregateReceipt
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise TimelockReleaseError("label release authority verifier is unavailable") from exc
    if (
        not isinstance(verified_phase_claim, VerifiedPhaseClaimCapability)
        or not isinstance(phase_beacon, PhaseBeaconReceipt)
        or not isinstance(completion_aggregate, PostOnlineCompletionAggregateReceipt)
    ):
        raise TimelockReleaseError("existing label release lacks typed current authority")
    stable_bindings = (
        ("manifest", receipt.manifest_sha256, manifest_digest),
        ("corpus", receipt.corpus_id, corpus_id),
        (
            "custody seal",
            receipt.custody_seal_receipt_sha256,
            custody_seal.receipt_sha256,
        ),
        (
            "encryption receipt",
            receipt.timelock_encryption_receipt_sha256,
            encryption_receipt.receipt_sha256,
        ),
        (
            "encryption receipt file",
            receipt.timelock_encryption_receipt_file_sha256,
            encryption_receipt.file_sha256,
        ),
        ("tle binary", receipt.tle_binary_sha256, tool_pin),
        ("drand network", receipt.drand_network, network),
        ("drand chain", receipt.drand_chain_hash, chain_hash),
        ("drand round", receipt.drand_round, round_number),
        ("verified drand round", receipt.verified_beacon_round, round_number),
        ("ciphertext digest", receipt.ciphertext_sha256, ciphertext_digest),
        (
            "ciphertext byte count",
            receipt.ciphertext_byte_count,
            ciphertext_byte_count,
        ),
        ("plaintext digest", receipt.plaintext_sha256, plaintext_pin),
        (
            "plaintext byte count",
            receipt.plaintext_byte_count,
            encryption_receipt.plaintext_byte_count,
        ),
        (
            "completion anchor record",
            receipt.prediction_completion_anchor_record_sha256,
            anchor_record.record_sha256,
        ),
        (
            "completion anchor receipt",
            receipt.prediction_completion_anchor_receipt_sha256,
            anchor_receipt.receipt_sha256,
        ),
        (
            "online result",
            receipt.online_execution_result_receipt_sha256,
            anchor_record.online_execution_result_receipt_sha256,
        ),
        (
            "post-online aggregate file",
            receipt.post_online_completion_aggregate_file_sha256,
            completion_aggregate.file_sha256,
        ),
        (
            "label claim state",
            receipt.label_release_claim_state_sha256,
            verified_phase_claim.phase_claim_state_sha256,
        ),
        (
            "label claim ledger commit",
            receipt.label_release_claim_ledger_commit,
            verified_phase_claim.phase_claim_ledger_commit,
        ),
        (
            "label claim contract",
            receipt.label_release_phase_claim_contract_sha256,
            verified_phase_claim.contract.contract_sha256,
        ),
        (
            "label provider identity",
            receipt.label_release_provider_identity_sha256,
            verified_phase_claim.provider_identity.identity_sha256,
        ),
        (
            "beacon response",
            receipt.beacon_response_sha256,
            phase_beacon.beacon_bytes_sha256,
        ),
        ("beacon signature", receipt.beacon_signature, phase_beacon.signature),
        (
            "beacon randomness",
            receipt.verified_beacon_randomness,
            phase_beacon.randomness,
        ),
        (
            "beacon publication",
            receipt.beacon_publication_time_utc,
            phase_beacon.published_at_utc,
        ),
    )
    for label, observed, expected in stable_bindings:
        if observed != expected:
            raise TimelockReleaseError(
                f"existing label release differs from current {label} authority"
            )


def _admit_release_pair(
    *,
    root: Path,
    plaintext_name: str,
    receipt_name: str,
    authority_validator: Callable[[TimelockDecryptionReceipt], None],
    allowed_extra_names: frozenset[str] = frozenset(),
) -> TimelockDecryptionReceipt:
    expected_names = frozenset({plaintext_name, receipt_name}) | allowed_extra_names
    if _release_directory_entries(root, label="label release directory") != expected_names:
        raise TimelockReleaseError(
            "label release directory differs from the exact plaintext-receipt pair"
        )
    receipt = load_timelock_decryption_receipt(root / receipt_name)
    authority_validator(receipt)
    try:
        plaintext = read_secure_regular_file(
            root / plaintext_name,
            max_bytes=receipt.plaintext_byte_count,
            label="existing released plaintext",
        )
    except ArtifactIntegrityError as exc:
        raise TimelockReleaseError("cannot admit existing released plaintext") from exc
    if (
        len(plaintext) != receipt.plaintext_byte_count
        or hashlib.sha256(plaintext).hexdigest() != receipt.plaintext_sha256
    ):
        raise TimelockReleaseError("existing released plaintext differs from its receipt")
    return receipt


def _recover_existing_release(
    *,
    release_root: Path,
    plaintext_name: str,
    receipt_name: str,
    intent_bytes: bytes,
    authority_validator: Callable[[TimelockDecryptionReceipt], None],
) -> TimelockDecryptionReceipt | None:
    stage = release_root.parent / label_release_staging_directory_name(release_root.name)
    final_exists = os.path.lexists(release_root)
    stage_exists = os.path.lexists(stage)
    if final_exists and stage_exists:
        raise TimelockReleaseError("label release has simultaneous staged and final transactions")
    if final_exists:
        entries = _release_directory_entries(
            release_root,
            label="label release directory",
        )
        expected_names = frozenset({plaintext_name, receipt_name})
        committed_names = expected_names | {TIMELOCK_RELEASE_INTENT_FILENAME}
        if entries not in {expected_names, committed_names}:
            raise TimelockReleaseError(
                "label release directory differs from a recoverable transaction"
            )
        receipt = _admit_release_pair(
            root=release_root,
            plaintext_name=plaintext_name,
            receipt_name=receipt_name,
            authority_validator=authority_validator,
            allowed_extra_names=(
                frozenset({TIMELOCK_RELEASE_INTENT_FILENAME})
                if entries == committed_names
                else frozenset()
            ),
        )
        if entries == committed_names:
            try:
                observed_intent = read_secure_control_file(
                    release_root / TIMELOCK_RELEASE_INTENT_FILENAME,
                    label="committed timelock release intent",
                )
            except ArtifactIntegrityError as exc:
                raise TimelockReleaseError(
                    "cannot admit committed timelock release intent"
                ) from exc
            if _release_intent_stable_identity(observed_intent) != _release_intent_stable_identity(
                intent_bytes
            ):
                raise TimelockReleaseError(
                    "committed timelock release intent differs from current authority"
                )
        return receipt
    if not stage_exists:
        return None
    expected_names = frozenset({plaintext_name, receipt_name})
    entries = _release_directory_entries(stage, label="label release stage")
    recoverable_names = expected_names | {TIMELOCK_RELEASE_INTENT_FILENAME}
    if entries == recoverable_names:
        try:
            observed_intent = read_secure_control_file(
                stage / TIMELOCK_RELEASE_INTENT_FILENAME,
                label="timelock release intent",
            )
        except ArtifactIntegrityError as exc:
            raise TimelockReleaseError("cannot admit staged timelock release intent") from exc
        if _release_intent_stable_identity(observed_intent) != _release_intent_stable_identity(
            intent_bytes
        ):
            raise TimelockReleaseError(
                "staged timelock release intent differs from current authority"
            )
        receipt = _admit_release_pair(
            root=stage,
            plaintext_name=plaintext_name,
            receipt_name=receipt_name,
            authority_validator=authority_validator,
            allowed_extra_names=frozenset({TIMELOCK_RELEASE_INTENT_FILENAME}),
        )
        _rename_release_stage(stage, release_root)
        return receipt
    raise TimelockReleaseError("label release has an ambiguous incomplete decryption transaction")


def release_timelock_label(
    manifest: Mapping[str, Any],
    *,
    corpus_id: str,
    custody_seal: CustodySealReceipt,
    encryption_receipt: TimelockEncryptionReceipt,
    verified_post_online_completion: object,
    verified_suite_completion: object | None = None,
    verified_phase_claim: object,
    ciphertext_path: str | Path,
    tle_binary_path: str | Path,
    plaintext_output_path: str | Path,
    decryption_receipt_output_path: str | Path,
    trusted_drand_fetcher: DrandFetcher | None = None,
    trusted_tle_runner: TleDecryptRunner | None = None,
    utc_now_factory: UtcNowFactory | None = None,
    timeout_seconds: int = DEFAULT_TLOCK_TIMEOUT_SECONDS,
    max_ciphertext_bytes: int = DEFAULT_MAX_CIPHERTEXT_BYTES,
    max_plaintext_bytes: int = DEFAULT_MAX_PLAINTEXT_BYTES,
) -> VerifiedTimelockRelease:
    """Release one label artifact after the all-five and corpus gates pass.

    Injected fetchers, runners, and clocks are explicit test/integration seams.
    The command-line ceremony does not expose them.
    """

    try:
        validate_study_manifest(manifest, require_frozen=True)
    except StudyManifestError as exc:
        raise TimelockReleaseError(f"invalid frozen study manifest: {exc}") from exc
    if corpus_id not in FIXED_CORPORA:
        raise TimelockReleaseError("corpus_id is not in the fixed suite")
    if not isinstance(custody_seal, CustodySealReceipt):
        raise TimelockReleaseError("custody_seal must be a CustodySealReceipt")
    if not isinstance(encryption_receipt, TimelockEncryptionReceipt):
        raise TimelockReleaseError("encryption_receipt must be a TimelockEncryptionReceipt")
    if encryption_receipt.corpus_id != corpus_id:
        raise TimelockReleaseError("encryption receipt belongs to another corpus")
    try:
        verify_custody_seal_receipt(custody_seal, manifest)
        verify_timelock_encryption_receipt(
            encryption_receipt,
            manifest,
            custody_seal=custody_seal,
        )
    except CustodyError as exc:
        raise TimelockReleaseError(f"custody evidence failed verification: {exc}") from exc

    manifest_digest = manifest_sha256(manifest)
    verified_completion_anchor, completion_aggregate = _require_post_online_completion(
        verified_post_online_completion,
        corpus_id=corpus_id,
        manifest_digest=manifest_digest,
        verified_phase_claim=verified_phase_claim,
        verified_suite_completion=verified_suite_completion,
    )
    anchor_record = verified_completion_anchor.record
    anchor_receipt = verified_completion_anchor.receipt
    if anchor_record.record_sha256 != anchor_receipt.anchor_record_sha256:
        raise TimelockReleaseError("external completion anchor has mismatched exact bytes")
    shared_anchor_fields = (
        "prediction_completion_receipt_sha256",
        "manifest_sha256",
        "run_receipt_sha256",
        "execution_artifact_sha256",
        "prediction_artifact_sha256",
        "online_execution_result_receipt_sha256",
        "corpus",
        "stage",
        "external_anchor_identity",
        "external_anchor_uri",
        "anchored_at_utc",
    )
    for name in shared_anchor_fields:
        if getattr(anchor_record, name) != getattr(anchor_receipt, name):
            raise TimelockReleaseError(f"external anchor record and receipt have mismatched {name}")
    if anchor_record.action_panel_binding.action_panel_artifact_sha256 != (
        anchor_receipt.action_panel_artifact_sha256
    ):
        raise TimelockReleaseError(
            "external anchor record and receipt bind different action panels"
        )
    if anchor_record.manifest_sha256 != manifest_digest:
        raise TimelockReleaseError("external completion anchor belongs to another manifest")
    if anchor_record.corpus != corpus_id:
        raise TimelockReleaseError("external completion anchor belongs to another corpus")
    if anchor_receipt.prediction_completion_receipt_sha256 != (
        anchor_record.prediction_completion_receipt_sha256
    ):
        raise TimelockReleaseError("external anchor receipt binds another completion record")
    _require_suite_online_completion(
        verified_suite_completion,
        manifest_digest=manifest_digest,
        corpus_id=corpus_id,
        online_result_receipt_sha256=(anchor_record.online_execution_result_receipt_sha256),
    )
    phase_beacon = _require_phase_release_authority(
        verified_phase_claim,
        manifest_digest=manifest_digest,
        run_receipt_sha256=anchor_record.run_receipt_sha256,
    )

    if custody_seal.drand_round != encryption_receipt.drand_round:
        raise TimelockReleaseError("custody seal and encryption receipt name different rounds")
    if custody_seal.drand_chain_hash != encryption_receipt.drand_chain_hash:
        raise TimelockReleaseError("custody seal and encryption receipt name different chains")
    if custody_seal.timelock_tool_sha256 != encryption_receipt.tle_binary_sha256:
        raise TimelockReleaseError("custody seal and encryption receipt name different tools")
    network = _require_https_network(encryption_receipt.drand_network)
    chain_hash = _require_sha256("drand_chain_hash", encryption_receipt.drand_chain_hash)
    round_number = _require_positive_integer("drand_round", encryption_receipt.drand_round)
    commitment = next(row for row in custody_seal.commitments if row.corpus_id == corpus_id)
    if encryption_receipt.file_sha256 != (commitment.timelock_encryption_receipt_file_sha256):
        raise TimelockReleaseError("custody seal binds another encryption receipt file")

    output = Path(plaintext_output_path)
    receipt_output = Path(decryption_receipt_output_path)
    if not output.is_absolute():
        raise TimelockReleaseError("plaintext_output_path must be absolute")
    if not receipt_output.is_absolute():
        raise TimelockReleaseError("decryption_receipt_output_path must be absolute")
    if output.parent != receipt_output.parent or output.name == receipt_output.name:
        raise TimelockReleaseError(
            "plaintext and decryption receipt must share one release directory"
        )
    bindings = [
        binding
        for binding in verified_phase_claim.contract.corpora
        if binding.corpus_id == corpus_id
    ]
    if (
        len(bindings) != 1
        or output.as_uri() != bindings[0].output_uri
        or receipt_output.name != TIMELOCK_DECRYPTION_RECEIPT_FILENAME
    ):
        raise TimelockReleaseError("release outputs differ from the exact claimed corpus binding")
    ciphertext_limit = _require_positive_integer("max_ciphertext_bytes", max_ciphertext_bytes)
    plaintext_limit = _require_positive_integer("max_plaintext_bytes", max_plaintext_bytes)
    timeout = _require_positive_integer("timeout_seconds", timeout_seconds)
    if timeout > 60:
        raise TimelockReleaseError("timeout_seconds cannot exceed 60")
    if ciphertext_limit > 2 * 1024 * 1024 * 1024:
        raise TimelockReleaseError("max_ciphertext_bytes exceeds the hard limit")
    if plaintext_limit > 1024 * 1024 * 1024:
        raise TimelockReleaseError("max_plaintext_bytes exceeds the hard limit")

    ciphertext_pin = _artifact_pin(manifest, "sealed-label-ciphertext", corpus_id=corpus_id)
    plaintext_pin = _artifact_pin(manifest, "sealed-labels", corpus_id=corpus_id)
    receipt_pin = _artifact_pin(manifest, "timelock-encryption-receipt", corpus_id=corpus_id)
    tool_pin = _artifact_pin(manifest, "timelock-tool")
    if receipt_pin != encryption_receipt.file_sha256:
        raise TimelockReleaseError("manifest binds another encryption receipt file")
    binary = _admit_tle_binary(tle_binary_path, expected_sha256=tool_pin)
    try:
        ciphertext = read_secure_regular_file(
            ciphertext_path,
            max_bytes=ciphertext_limit,
            label=f"{corpus_id} timelock ciphertext",
        )
    except ArtifactIntegrityError as exc:
        raise TimelockReleaseError(f"cannot admit ciphertext: {exc}") from exc
    if not ciphertext:
        raise TimelockReleaseError("ciphertext cannot be empty")
    ciphertext_digest = hashlib.sha256(ciphertext).hexdigest()
    if (
        not hmac.compare_digest(ciphertext_digest, ciphertext_pin)
        or not hmac.compare_digest(ciphertext_digest, encryption_receipt.ciphertext_sha256)
        or len(ciphertext) != encryption_receipt.ciphertext_byte_count
    ):
        raise TimelockReleaseError("ciphertext bytes differ from the frozen evidence")

    def validate_existing(receipt: TimelockDecryptionReceipt) -> None:
        _require_existing_release_authority(
            receipt,
            manifest_digest=manifest_digest,
            corpus_id=corpus_id,
            custody_seal=custody_seal,
            encryption_receipt=encryption_receipt,
            completion_aggregate=completion_aggregate,
            anchor_record=anchor_record,
            anchor_receipt=anchor_receipt,
            verified_phase_claim=verified_phase_claim,
            phase_beacon=phase_beacon,
            network=network,
            chain_hash=chain_hash,
            round_number=round_number,
            tool_pin=tool_pin,
            ciphertext_digest=ciphertext_digest,
            ciphertext_byte_count=len(ciphertext),
            plaintext_pin=plaintext_pin,
        )

    intent_bytes = _release_intent_bytes(
        corpus_id=corpus_id,
        manifest_digest=manifest_digest,
        claim_state_sha256=verified_phase_claim.phase_claim_state_sha256,
        claim_ledger_commit=verified_phase_claim.phase_claim_ledger_commit,
        claim_contract_sha256=verified_phase_claim.contract.contract_sha256,
        phase_beacon_identity_sha256=phase_beacon.beacon_identity_sha256,
        live_execute_job_identity_sha256=(
            verified_phase_claim.live_execute_job_receipt.job_identity_sha256
        ),
        phase_beacon_receipt=phase_beacon,
        live_execute_job_receipt=verified_phase_claim.live_execute_job_receipt,
        provider_identity_sha256=(verified_phase_claim.provider_identity.identity_sha256),
        post_online_aggregate_file_sha256=completion_aggregate.file_sha256,
        tle_binary_sha256=tool_pin,
        ciphertext_sha256=ciphertext_digest,
        drand_round=round_number,
        plaintext_output_uri=output.as_uri(),
        receipt_output_uri=receipt_output.as_uri(),
    )
    existing = _recover_existing_release(
        release_root=output.parent,
        plaintext_name=output.name,
        receipt_name=receipt_output.name,
        intent_bytes=intent_bytes,
        authority_validator=validate_existing,
    )
    if existing is not None:
        _require_suite_online_completion(
            verified_suite_completion,
            manifest_digest=manifest_digest,
            corpus_id=corpus_id,
            online_result_receipt_sha256=(anchor_record.online_execution_result_receipt_sha256),
        )
        _require_phase_release_authority(
            verified_phase_claim,
            manifest_digest=manifest_digest,
            run_receipt_sha256=anchor_record.run_receipt_sha256,
        )
        return VerifiedTimelockRelease(
            receipt=existing,
            plaintext_path=output.resolve(strict=True),
            _capability=_RELEASE_CAPABILITY,
        )

    fetcher = _fetch_drand_bytes if trusted_drand_fetcher is None else trusted_drand_fetcher
    if not callable(fetcher):
        raise TimelockReleaseError("trusted_drand_fetcher must be callable")
    evidence = _fetch_and_verify_drand_evidence(
        network=network,
        chain_hash=chain_hash,
        round_number=round_number,
        fetcher=fetcher,
    )
    _require_beacon_evidence_matches_phase_authority(evidence, phase_beacon)
    anchor_time = _require_utc_timestamp(
        "external anchor anchored_at_utc", anchor_record.anchored_at_utc
    )
    if anchor_time >= evidence.publication_time:
        raise TimelockReleaseError(
            "external prediction-completion anchor must strictly predate target-round publication"
        )

    now_factory = (
        (lambda: datetime.now(timezone.utc)) if utc_now_factory is None else utc_now_factory
    )
    if not callable(now_factory):
        raise TimelockReleaseError("utc_now_factory must be callable")
    started = _utc_now(now_factory)
    if started < evidence.publication_time:
        raise TimelockReleaseError("target drand round has not yet been published")
    arguments = _decrypt_arguments(network, chain_hash)
    runner = _run_pinned_tle_decrypt if trusted_tle_runner is None else trusted_tle_runner
    if not callable(runner):
        raise TimelockReleaseError("trusted_tle_runner must be callable")
    _require_suite_online_completion(
        verified_suite_completion,
        manifest_digest=manifest_digest,
        corpus_id=corpus_id,
        online_result_receipt_sha256=(anchor_record.online_execution_result_receipt_sha256),
    )
    current_phase_beacon = _require_phase_release_authority(
        verified_phase_claim,
        manifest_digest=manifest_digest,
        run_receipt_sha256=anchor_record.run_receipt_sha256,
    )
    if current_phase_beacon != phase_beacon:
        raise TimelockReleaseError("provider label-release beacon changed before decryption")
    _prepare_release_transaction(
        output.parent,
        intent_bytes=intent_bytes,
    )
    try:
        plaintext = runner(
            binary,
            arguments,
            ciphertext,
            timeout,
            plaintext_limit,
        )
    except TimelockReleaseError:
        raise
    except Exception as exc:
        raise TimelockReleaseError("trusted tle runner failed") from exc
    completed = _utc_now(now_factory)
    if not isinstance(plaintext, bytes) or not plaintext:
        raise TimelockReleaseError("tle runner must return non-empty plaintext bytes")
    if len(plaintext) > plaintext_limit:
        raise TimelockReleaseError("released plaintext exceeds max_plaintext_bytes")
    plaintext_digest = hashlib.sha256(plaintext).hexdigest()
    if (
        not hmac.compare_digest(plaintext_digest, plaintext_pin)
        or not hmac.compare_digest(plaintext_digest, encryption_receipt.plaintext_sha256)
        or len(plaintext) != encryption_receipt.plaintext_byte_count
    ):
        raise TimelockReleaseError("released plaintext differs from the frozen label bytes")
    try:
        final_binary_digest = digest_regular_file(binary, label="tle binary")
    except ArtifactIntegrityError as exc:
        raise TimelockReleaseError(f"cannot revalidate tle binary: {exc}") from exc
    if not hmac.compare_digest(final_binary_digest, tool_pin):
        raise TimelockReleaseError("tle binary changed during decryption")
    _require_suite_online_completion(
        verified_suite_completion,
        manifest_digest=manifest_digest,
        corpus_id=corpus_id,
        online_result_receipt_sha256=(anchor_record.online_execution_result_receipt_sha256),
    )
    current_phase_beacon = _require_phase_release_authority(
        verified_phase_claim,
        manifest_digest=manifest_digest,
        run_receipt_sha256=anchor_record.run_receipt_sha256,
    )
    if current_phase_beacon != phase_beacon:
        raise TimelockReleaseError("provider label-release beacon changed during decryption")
    receipt = TimelockDecryptionReceipt(
        manifest_sha256=manifest_digest,
        corpus_id=corpus_id,
        custody_seal_receipt_sha256=custody_seal.receipt_sha256,
        timelock_encryption_receipt_sha256=encryption_receipt.receipt_sha256,
        timelock_encryption_receipt_file_sha256=encryption_receipt.file_sha256,
        tle_binary_sha256=tool_pin,
        drand_network=network,
        drand_chain_hash=chain_hash,
        drand_round=round_number,
        chain_metadata_uri=f"{network}/{chain_hash}/info",
        chain_metadata_base64=base64.b64encode(evidence.chain_metadata).decode("ascii"),
        chain_metadata_sha256=hashlib.sha256(evidence.chain_metadata).hexdigest(),
        chain_metadata_byte_count=len(evidence.chain_metadata),
        chain_public_key=evidence.public_key,
        chain_scheme_id=evidence.scheme_id,
        chain_period_seconds=evidence.period_seconds,
        chain_genesis_time_utc=evidence.genesis_time.isoformat(),
        beacon_uri=f"{network}/{chain_hash}/public/{round_number}",
        beacon_response_base64=base64.b64encode(evidence.beacon_response).decode("ascii"),
        beacon_response_sha256=hashlib.sha256(evidence.beacon_response).hexdigest(),
        beacon_response_byte_count=len(evidence.beacon_response),
        beacon_signature=evidence.signature,
        verified_beacon_round=round_number,
        verified_beacon_randomness=evidence.randomness,
        beacon_publication_time_utc=evidence.publication_time.isoformat(),
        ciphertext_sha256=ciphertext_digest,
        ciphertext_byte_count=len(ciphertext),
        plaintext_sha256=plaintext_digest,
        plaintext_byte_count=len(plaintext),
        tle_arguments=arguments,
        started_at_utc=started.isoformat(),
        completed_at_utc=completed.isoformat(),
        prediction_completion_anchor_record_sha256=anchor_record.record_sha256,
        prediction_completion_anchor_receipt_sha256=anchor_receipt.receipt_sha256,
        online_execution_result_receipt_sha256=(
            anchor_record.online_execution_result_receipt_sha256
        ),
        post_online_completion_aggregate_file_sha256=(completion_aggregate.file_sha256),
        label_release_claim_state_sha256=(verified_phase_claim.phase_claim_state_sha256),
        label_release_claim_ledger_commit=(verified_phase_claim.phase_claim_ledger_commit),
        label_release_phase_claim_contract_sha256=(verified_phase_claim.contract.contract_sha256),
        label_release_phase_beacon_receipt_sha256=(phase_beacon.receipt_sha256),
        label_release_live_execute_job_receipt_sha256=(
            verified_phase_claim.live_execute_job_receipt.receipt_sha256
        ),
        label_release_provider_identity_sha256=(
            verified_phase_claim.provider_identity.identity_sha256
        ),
    )
    _publish_release_pair(
        plaintext=plaintext,
        receipt=receipt,
        plaintext_path=output,
        receipt_path=receipt_output,
        intent_bytes=intent_bytes,
    )
    if load_timelock_decryption_receipt(receipt_output) != receipt:
        raise TimelockReleaseError("committed decryption receipt differs after atomic publication")
    return VerifiedTimelockRelease(
        receipt=receipt,
        plaintext_path=output.resolve(strict=True),
        _capability=_RELEASE_CAPABILITY,
    )
