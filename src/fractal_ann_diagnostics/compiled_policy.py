"""Bit-packed authorization masks and a bounded OPA decision adapter.

OPA selects one immutable mask by identifier. It never receives the document
universe as a JSON array and never returns millions of document identifiers.
The local adapter verifies the selected bitset before producing a policy
decision for the governed retriever.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import socket
import ssl
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn
from urllib.error import URLError
from urllib.parse import urlsplit

import numpy as np

from .artifact_integrity import (
    ArtifactIntegrityError,
    read_secure_control_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .policy import (
    EMPTY_POLICY_ENVIRONMENT_SHA256,
    PolicyDecision,
    policy_environment_sha256,
    policy_request_sha256,
)
from .policy_adapters import OPAHTTPResponse, OPATransport, _urllib_transport

COMPILED_POLICY_CATALOG_SCHEMA = "fractal-compiled-policy-catalog-v1"
COMPILED_MASK_ENCODING = "numpy-packbits-little-v1"
MAX_COMPILED_POLICY_CATALOG_BYTES = 1024 * 1024
MAX_COMPILED_POLICY_RESPONSE_BYTES = 64 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")
_CATALOG_FIELDS = {
    "document_count",
    "document_universe_sha256",
    "encoding",
    "masks",
    "policy_revision",
    "schema_version",
}
_MASK_FIELDS = {
    "authorized_count",
    "byte_count",
    "mask_id",
    "path",
    "sha256",
}
_OPA_RESULT_FIELDS = {
    "action",
    "authorized_count",
    "catalog_request_sha256",
    "document_count",
    "document_universe_sha256",
    "environment_sha256",
    "mask_catalog_sha256",
    "mask_id",
    "mask_sha256",
    "policy_revision",
    "request_nonce",
    "request_sha256",
    "subject",
}


class CompiledPolicyError(ValueError):
    """Raised when a compiled mask or OPA response is not admissible."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CompiledPolicyError("compiled policy evidence must be finite JSON") from exc


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CompiledPolicyError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> NoReturn:
        raise CompiledPolicyError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompiledPolicyError(f"cannot decode {label}") from exc
    if not isinstance(value, Mapping):
        raise CompiledPolicyError(f"{label} must contain one JSON object")
    return value


def _closed_mapping(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CompiledPolicyError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise CompiledPolicyError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _require_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise CompiledPolicyError(f"{name} must be a canonical non-empty string")
    return value


def _require_identifier(name: str, value: object) -> str:
    text = _require_text(name, value)
    if _IDENTIFIER.fullmatch(text) is None:
        raise CompiledPolicyError(f"{name} must be a lowercase filesystem-safe identifier")
    return text


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CompiledPolicyError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_integer(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise CompiledPolicyError(f"{name} must be a non-negative integer")
    return value


def _relative_path(name: str, value: object) -> str:
    text = _require_text(name, value)
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or str(pure) != text
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in text
    ):
        raise CompiledPolicyError(f"{name} must be a canonical relative POSIX path")
    return text


@dataclass(frozen=True, order=True)
class CompiledMaskDescriptor:
    """One exact bit-packed authorization mask."""

    mask_id: str
    path: str
    sha256: str
    byte_count: int
    authorized_count: int

    def __post_init__(self) -> None:
        _require_identifier("mask_id", self.mask_id)
        _relative_path("mask path", self.path)
        _require_sha256("mask sha256", self.sha256)
        if self.byte_count <= 0:
            raise CompiledPolicyError("mask byte_count must be positive")
        _require_nonnegative_integer("authorized_count", self.authorized_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "authorized_count": self.authorized_count,
            "byte_count": self.byte_count,
            "mask_id": self.mask_id,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> CompiledMaskDescriptor:
        row = _closed_mapping(value, fields=_MASK_FIELDS, label="compiled mask descriptor")
        return cls(
            mask_id=row["mask_id"],
            path=row["path"],
            sha256=row["sha256"],
            byte_count=row["byte_count"],
            authorized_count=row["authorized_count"],
        )


@dataclass(frozen=True)
class CompiledPolicyCatalog:
    """Closed catalog that binds OPA mask selections to local bytes."""

    document_count: int
    document_universe_sha256: str
    policy_revision: str
    masks: tuple[CompiledMaskDescriptor, ...]
    schema_version: str = COMPILED_POLICY_CATALOG_SCHEMA
    encoding: str = COMPILED_MASK_ENCODING

    def __post_init__(self) -> None:
        if type(self.document_count) is not int or self.document_count <= 0:
            raise CompiledPolicyError("document_count must be a positive integer")
        _require_sha256("document_universe_sha256", self.document_universe_sha256)
        _require_text("policy_revision", self.policy_revision)
        if self.schema_version != COMPILED_POLICY_CATALOG_SCHEMA:
            raise CompiledPolicyError(
                f"schema_version must equal {COMPILED_POLICY_CATALOG_SCHEMA!r}"
            )
        if self.encoding != COMPILED_MASK_ENCODING:
            raise CompiledPolicyError(f"encoding must equal {COMPILED_MASK_ENCODING!r}")
        masks = tuple(self.masks)
        if not masks or not all(isinstance(mask, CompiledMaskDescriptor) for mask in masks):
            raise CompiledPolicyError("masks must contain compiled mask descriptors")
        canonical = tuple(sorted(masks, key=lambda mask: mask.mask_id.encode("ascii")))
        if masks != canonical or len({mask.mask_id for mask in masks}) != len(masks):
            raise CompiledPolicyError("masks must be uniquely bytewise sorted by mask_id")
        if len({mask.path for mask in masks}) != len(masks):
            raise CompiledPolicyError("compiled masks must use unique paths")
        expected_bytes = (self.document_count + 7) // 8
        if any(mask.byte_count != expected_bytes for mask in masks):
            raise CompiledPolicyError("mask byte_count differs from document_count")
        if any(mask.authorized_count > self.document_count for mask in masks):
            raise CompiledPolicyError("authorized_count exceeds document_count")
        object.__setattr__(self, "masks", masks)

    def to_dict(self) -> dict[str, object]:
        return {
            "document_count": self.document_count,
            "document_universe_sha256": self.document_universe_sha256,
            "encoding": self.encoding,
            "masks": [mask.to_dict() for mask in self.masks],
            "policy_revision": self.policy_revision,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes() + b"\n").hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> CompiledPolicyCatalog:
        row = _closed_mapping(value, fields=_CATALOG_FIELDS, label="compiled policy catalog")
        masks = row["masks"]
        if not isinstance(masks, Sequence) or isinstance(masks, (str, bytes)):
            raise CompiledPolicyError("compiled policy masks must be an array")
        return cls(
            document_count=row["document_count"],
            document_universe_sha256=row["document_universe_sha256"],
            policy_revision=row["policy_revision"],
            masks=tuple(CompiledMaskDescriptor.from_dict(mask) for mask in masks),
            schema_version=row["schema_version"],
            encoding=row["encoding"],
        )


def encode_compiled_mask(mask: np.ndarray, *, document_count: int) -> tuple[bytes, int]:
    """Encode one exact boolean universe with deterministic trailing bits."""

    values = np.asarray(mask)
    if values.dtype != np.bool_ or values.shape != (document_count,):
        raise CompiledPolicyError("mask must be a boolean vector matching document_count")
    packed = np.packbits(values, bitorder="little").tobytes()
    return packed, int(values.sum())


def compiled_mask_descriptor(
    mask_id: str,
    relative_path: str,
    mask: np.ndarray,
    *,
    document_count: int,
) -> tuple[CompiledMaskDescriptor, bytes]:
    """Build a descriptor and bytes without writing either artifact."""

    encoded, authorized_count = encode_compiled_mask(mask, document_count=document_count)
    descriptor = CompiledMaskDescriptor(
        mask_id=mask_id,
        path=relative_path,
        sha256=hashlib.sha256(encoded).hexdigest(),
        byte_count=len(encoded),
        authorized_count=authorized_count,
    )
    return descriptor, encoded


def write_compiled_mask(encoded: bytes, target: str | Path) -> None:
    if not isinstance(encoded, bytes) or not encoded:
        raise CompiledPolicyError("compiled mask bytes must be non-empty")
    try:
        write_exclusive_receipt_bytes(encoded, target)
    except ArtifactIntegrityError as exc:
        raise CompiledPolicyError(f"cannot write compiled mask: {exc}") from exc


def write_compiled_policy_catalog(
    catalog: CompiledPolicyCatalog,
    target: str | Path,
) -> None:
    if not isinstance(catalog, CompiledPolicyCatalog):
        raise CompiledPolicyError("catalog must be CompiledPolicyCatalog")
    try:
        write_exclusive_receipt_bytes(catalog.canonical_bytes() + b"\n", target)
    except ArtifactIntegrityError as exc:
        raise CompiledPolicyError(f"cannot write compiled policy catalog: {exc}") from exc


def load_compiled_policy_catalog(path: str | Path) -> CompiledPolicyCatalog:
    try:
        encoded = read_secure_control_file(path, label="compiled policy catalog")
    except ArtifactIntegrityError as exc:
        raise CompiledPolicyError(f"cannot load compiled policy catalog: {exc}") from exc
    catalog = CompiledPolicyCatalog.from_dict(
        _decode_object(encoded, label="compiled policy catalog")
    )
    if encoded != catalog.canonical_bytes() + b"\n":
        raise CompiledPolicyError(
            "compiled policy catalog bytes must equal canonical JSON plus one newline"
        )
    return catalog


class CompiledPolicyMaskStore:
    """Verified local mask store with bounded, no-follow reads."""

    def __init__(self, catalog_path: str | Path) -> None:
        path = Path(catalog_path)
        if not path.is_absolute():
            raise CompiledPolicyError("catalog_path must be absolute")
        self.catalog_path = path
        self.catalog = load_compiled_policy_catalog(path)
        self._descriptors = {mask.mask_id: mask for mask in self.catalog.masks}
        self._cache: dict[str, np.ndarray] = {}

    @property
    def catalog_sha256(self) -> str:
        return self.catalog.artifact_sha256

    def mask(
        self,
        mask_id: str,
        *,
        expected_sha256: str,
        expected_authorized_count: int,
    ) -> np.ndarray:
        try:
            descriptor = self._descriptors[mask_id]
        except KeyError as exc:
            raise CompiledPolicyError(f"OPA selected unknown mask {mask_id!r}") from exc
        if (
            not hmac.compare_digest(descriptor.sha256, expected_sha256)
            or descriptor.authorized_count != expected_authorized_count
        ):
            raise CompiledPolicyError("OPA mask metadata differs from the pinned catalog")
        cached = self._cache.get(mask_id)
        if cached is not None:
            return cached
        target = self.catalog_path.parent / descriptor.path
        try:
            encoded = read_secure_regular_file(
                target,
                max_bytes=descriptor.byte_count,
                label=f"compiled policy mask {mask_id}",
            )
        except ArtifactIntegrityError as exc:
            raise CompiledPolicyError(
                f"cannot read compiled policy mask {mask_id!r}: {exc}"
            ) from exc
        if len(encoded) != descriptor.byte_count:
            raise CompiledPolicyError("compiled mask byte count differs from the catalog")
        observed = hashlib.sha256(encoded).hexdigest()
        if not hmac.compare_digest(observed, descriptor.sha256):
            raise CompiledPolicyError("compiled mask SHA-256 differs from the catalog")
        remainder = self.catalog.document_count % 8
        if remainder and encoded[-1] & (~((1 << remainder) - 1) & 0xFF):
            raise CompiledPolicyError("compiled mask has nonzero trailing bits")
        unpacked = np.unpackbits(
            np.frombuffer(encoded, dtype=np.uint8),
            bitorder="little",
        )[: self.catalog.document_count].astype(bool, copy=True)
        if int(unpacked.sum()) != descriptor.authorized_count:
            raise CompiledPolicyError("compiled mask authorized count differs from the catalog")
        unpacked.setflags(write=False)
        self._cache[mask_id] = unpacked
        return unpacked

    def verify_all(self) -> tuple[str, ...]:
        for descriptor in self.catalog.masks:
            self.mask(
                descriptor.mask_id,
                expected_sha256=descriptor.sha256,
                expected_authorized_count=descriptor.authorized_count,
            )
        return tuple(descriptor.mask_id for descriptor in self.catalog.masks)


def _catalog_request_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _is_literal_loopback(hostname: str | None) -> bool:
    if hostname is None:
        return False
    try:
        import ipaddress

        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class OpenPolicyAgentMaskDecisionPoint:
    """OPA adapter that exchanges one pinned mask identifier per request."""

    def __init__(
        self,
        endpoint: str,
        mask_store: CompiledPolicyMaskStore,
        *,
        timeout_seconds: float = 2.0,
        bearer_token: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        transport: OPATransport | None = None,
    ) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute HTTP or HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("endpoint cannot contain user information")
        if parsed.query or parsed.fragment:
            raise ValueError("endpoint cannot contain a query or fragment")
        loopback = _is_literal_loopback(parsed.hostname)
        if parsed.scheme == "http" and not loopback:
            raise ValueError("plain HTTP OPA endpoints require a literal loopback IP address")
        if not isinstance(mask_store, CompiledPolicyMaskStore):
            raise TypeError("mask_store must be CompiledPolicyMaskStore")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        if ssl_context is not None and (
            ssl_context.verify_mode != ssl.CERT_REQUIRED or ssl_context.check_hostname is not True
        ):
            raise ValueError("ssl_context must verify certificates and hostnames")
        if transport is None and not loopback and bearer_token is None:
            raise ValueError("remote built-in OPA transport requires a bearer token")
        self.endpoint = endpoint
        self.mask_store = mask_store
        self.timeout_seconds = float(timeout_seconds)
        if transport is None:
            self._transport: Callable[[str, bytes, float], OPAHTTPResponse] = (
                lambda endpoint, body, timeout: _urllib_transport(
                    endpoint,
                    body,
                    timeout,
                    bearer_token=bearer_token,
                    ssl_context=ssl_context,
                )
            )
        else:
            self._transport = transport

    @property
    def n_documents(self) -> int:
        return self.mask_store.catalog.document_count

    @property
    def document_universe_sha256(self) -> str:
        return self.mask_store.catalog.document_universe_sha256

    def _deny(
        self,
        subject: str,
        action: str,
        reason: str,
        *,
        environment_sha256: str = EMPTY_POLICY_ENVIRONMENT_SHA256,
        request_nonce: str = "",
    ) -> PolicyDecision:
        return PolicyDecision(
            subject=subject,
            action=action,
            policy_version="unavailable",
            authorized_mask=np.zeros(self.n_documents, dtype=bool),
            available=False,
            reason=reason,
            environment_sha256=environment_sha256,
            document_universe_sha256=self.document_universe_sha256,
            request_nonce=request_nonce,
        )

    def _request(
        self,
        subject: str,
        action: str,
        environment: Mapping[str, object] | None,
        environment_sha256: str,
    ) -> tuple[bytes, str, str, str]:
        _require_text("subject", subject)
        _require_text("action", action)
        request_nonce = secrets.token_hex(32)
        request_sha256 = policy_request_sha256(
            subject=subject,
            action=action,
            environment_sha256=environment_sha256,
            document_universe_sha256=self.document_universe_sha256,
            request_nonce=request_nonce,
        )
        policy_input: dict[str, object] = {
            "action": action,
            "document_count": self.n_documents,
            "document_universe_sha256": self.document_universe_sha256,
            "environment": dict(environment or {}),
            "environment_sha256": environment_sha256,
            "mask_catalog_sha256": self.mask_store.catalog_sha256,
            "policy_revision": self.mask_store.catalog.policy_revision,
            "request_nonce": request_nonce,
            "request_sha256": request_sha256,
            "subject": subject,
        }
        catalog_request_sha256 = _catalog_request_sha256(policy_input)
        policy_input["catalog_request_sha256"] = catalog_request_sha256
        return (
            _canonical_bytes({"input": policy_input}),
            request_nonce,
            request_sha256,
            catalog_request_sha256,
        )

    def _parse_response(
        self,
        response: OPAHTTPResponse,
        *,
        subject: str,
        action: str,
        environment_sha256: str,
        request_nonce: str,
        request_sha256: str,
        catalog_request_sha256: str,
    ) -> PolicyDecision:
        if type(response.status) is not int or not 200 <= response.status < 300:
            raise CompiledPolicyError("OPA returned a non-success HTTP status")
        if (
            not isinstance(response.body, bytes)
            or len(response.body) > MAX_COMPILED_POLICY_RESPONSE_BYTES
        ):
            raise CompiledPolicyError("OPA response body is invalid or exceeds the limit")
        payload = _closed_mapping(
            _decode_object(response.body, label="OPA response"),
            fields={"decision_id", "result"},
            label="OPA response",
        )
        decision_id = _require_text("decision_id", payload["decision_id"])
        result = _closed_mapping(
            payload["result"],
            fields=_OPA_RESULT_FIELDS,
            label="OPA result",
        )
        expected: Mapping[str, object] = {
            "action": action,
            "catalog_request_sha256": catalog_request_sha256,
            "document_count": self.n_documents,
            "document_universe_sha256": self.document_universe_sha256,
            "environment_sha256": environment_sha256,
            "mask_catalog_sha256": self.mask_store.catalog_sha256,
            "policy_revision": self.mask_store.catalog.policy_revision,
            "request_nonce": request_nonce,
            "request_sha256": request_sha256,
            "subject": subject,
        }
        for field, value in expected.items():
            if result[field] != value:
                raise CompiledPolicyError(f"OPA result has mismatched {field}")
        mask_id = _require_identifier("mask_id", result["mask_id"])
        mask_sha256 = _require_sha256("mask_sha256", result["mask_sha256"])
        authorized_count = _require_nonnegative_integer(
            "authorized_count", result["authorized_count"]
        )
        mask = self.mask_store.mask(
            mask_id,
            expected_sha256=mask_sha256,
            expected_authorized_count=authorized_count,
        )
        return PolicyDecision(
            subject=subject,
            action=action,
            policy_version=self.mask_store.catalog.policy_revision,
            authorized_mask=mask,
            decision_id=decision_id,
            reason=f"OPA selected compiled mask {mask_id}",
            environment_sha256=environment_sha256,
            document_universe_sha256=self.document_universe_sha256,
            request_nonce=request_nonce,
            request_sha256=request_sha256,
        )

    def decide(
        self,
        subject: str,
        *,
        action: str = "retrieve",
        environment: Mapping[str, object] | None = None,
    ) -> PolicyDecision:
        try:
            environment_sha256 = policy_environment_sha256(environment)
            body, nonce, request_sha256, catalog_request_sha256 = self._request(
                subject,
                action,
                environment,
                environment_sha256,
            )
        except (CompiledPolicyError, TypeError, ValueError):
            return self._deny(
                subject,
                action,
                "OPA mask request validation failed; deny by default",
            )
        try:
            response = self._transport(self.endpoint, body, self.timeout_seconds)
        except (TimeoutError, socket.timeout):
            return self._deny(
                subject,
                action,
                "OPA mask request timed out; deny by default",
                environment_sha256=environment_sha256,
                request_nonce=nonce,
            )
        except (URLError, ssl.SSLError, ConnectionError, OSError):
            return self._deny(
                subject,
                action,
                "OPA mask transport failed; deny by default",
                environment_sha256=environment_sha256,
                request_nonce=nonce,
            )
        try:
            return self._parse_response(
                response,
                subject=subject,
                action=action,
                environment_sha256=environment_sha256,
                request_nonce=nonce,
                request_sha256=request_sha256,
                catalog_request_sha256=catalog_request_sha256,
            )
        except (CompiledPolicyError, TypeError, ValueError):
            return self._deny(
                subject,
                action,
                "OPA mask response validation failed; deny by default",
                environment_sha256=environment_sha256,
                request_nonce=nonce,
            )
