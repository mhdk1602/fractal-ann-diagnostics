"""Pinned Quicknet retrieval and RFC 9380 BLS verification.

The HTTP relay is transport, not authority.  This module reconstitutes the
Quicknet chain hash from its exact chain information and verifies every beacon
signature against the C1-frozen G2 public key before returning response bytes.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

import truststore
from py_ecc.bls.g2_primitives import subgroup_check
from py_ecc.bls.hash_to_curve import hash_to_G1
from py_ecc.bls.point_compression import (
    compress_G1,
    compress_G2,
    decompress_G1,
    decompress_G2,
)
from py_ecc.optimized_bls12_381 import G2, is_inf, pairing

from .execution_claim import (
    ExecutionBeaconContract,
    ExecutionClaimError,
    VerifiedBeaconClaims,
)

QUICKNET_NETWORK = "https://api.drand.sh"
QUICKNET_CHAIN_HASH = "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
QUICKNET_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)
QUICKNET_GROUP_HASH = "f477d5c89f21a17c863a7f937c6a6d15859414d2be09cd448d4279af331c5d3e"
QUICKNET_SCHEME_ID = "bls-unchained-g1-rfc9380"
QUICKNET_BEACON_ID = "quicknet"
QUICKNET_PERIOD_SECONDS = 3
QUICKNET_GENESIS_UNIX_SECONDS = 1_692_803_367

# The RFC 9380 G1 hash-to-curve domain used by drand's compliant short-signature scheme.
QUICKNET_G1_DST = b"BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_"

MAX_CHAIN_INFO_BYTES = 4 * 1024
MAX_BEACON_BYTES = 2 * 1024
_JSON_MEDIA_TYPE = "application/json"
_HEX = frozenset("0123456789abcdef")


class DrandBeaconError(ExecutionClaimError):
    """Quicknet transport, schema, chain identity, or signature verification failed."""


@dataclass(frozen=True)
class DrandHttpResponse:
    """Bounded response returned by a read-only drand transport."""

    status: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str

    def __post_init__(self) -> None:
        if type(self.status) is not int or not 100 <= self.status <= 599:
            raise DrandBeaconError("drand HTTP status is invalid")
        if not isinstance(self.headers, Mapping) or any(
            type(key) is not str or type(value) is not str for key, value in self.headers.items()
        ):
            raise DrandBeaconError("drand HTTP headers are malformed")
        if not isinstance(self.body, bytes):
            raise DrandBeaconError("drand HTTP body must be bytes")
        _exact_https_url(self.final_url)


class DrandReadApi(Protocol):
    """Read-only API used by the Quicknet verifier."""

    def get(self, url: str, *, max_bytes: int) -> DrandHttpResponse: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:  # type: ignore[override]
        return None


class UrllibDrandReadApi:
    """Production HTTPS reader that never follows redirects."""

    def __init__(self) -> None:
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._opener = build_opener(_NoRedirect(), HTTPSHandler(context=context))

    def get(self, url: str, *, max_bytes: int) -> DrandHttpResponse:
        _exact_https_url(url)
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_CHAIN_INFO_BYTES:
            raise DrandBeaconError("drand response byte bound is invalid")
        request = Request(
            url,
            headers={
                "Accept": _JSON_MEDIA_TYPE,
                "User-Agent": "fractal-ann-diagnostics-quicknet-verifier/1",
            },
            method="GET",
        )
        try:
            response = self._opener.open(request, timeout=20)
        except HTTPError as exc:
            if 300 <= exc.code < 400:
                raise DrandBeaconError("drand fetch refused an HTTP redirect") from exc
            raise DrandBeaconError(f"drand fetch returned HTTP {exc.code}") from exc
        except OSError as exc:
            raise DrandBeaconError("drand HTTPS fetch failed") from exc
        with response:
            headers = dict(response.headers.items())
            _validate_response_headers(headers, max_bytes=max_bytes)
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise DrandBeaconError("drand response exceeds its byte bound")
            return DrandHttpResponse(
                status=response.status,
                headers=headers,
                body=body,
                final_url=response.geturl(),
            )


def _exact_https_url(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise DrandBeaconError("drand URL must be one canonical HTTPS URL")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise DrandBeaconError("drand URL port is malformed") from exc
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.drand.sh"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith(f"/{QUICKNET_CHAIN_HASH}/")
    ):
        raise DrandBeaconError("drand URL differs from the exact Quicknet endpoint")
    return value


def _header(headers: Mapping[str, str], name: str) -> str | None:
    matches = [value for key, value in headers.items() if key.lower() == name.lower()]
    if len(matches) > 1:
        raise DrandBeaconError(f"drand response repeats {name}")
    return matches[0] if matches else None


def _validate_response_headers(headers: Mapping[str, str], *, max_bytes: int) -> None:
    content_type = _header(headers, "Content-Type")
    if content_type is None or content_type.split(";", 1)[0].strip().lower() != _JSON_MEDIA_TYPE:
        raise DrandBeaconError("drand response is not JSON media")
    encoding = _header(headers, "Content-Encoding")
    if encoding is not None and encoding.lower() != "identity":
        raise DrandBeaconError("drand response uses an unsupported content encoding")
    length = _header(headers, "Content-Length")
    if length is not None:
        if not length.isascii() or not length.isdigit():
            raise DrandBeaconError("drand Content-Length is malformed")
        if int(length) > max_bytes:
            raise DrandBeaconError("drand response exceeds its byte bound")


def _response_body(response: DrandHttpResponse, *, url: str, max_bytes: int) -> bytes:
    if not isinstance(response, DrandHttpResponse):
        raise DrandBeaconError("drand transport returned an untyped response")
    if 300 <= response.status < 400:
        raise DrandBeaconError("drand fetch refused an HTTP redirect")
    if response.status != 200:
        raise DrandBeaconError(f"drand fetch returned HTTP {response.status}")
    if response.final_url != url:
        raise DrandBeaconError("drand response URL changed")
    _validate_response_headers(response.headers, max_bytes=max_bytes)
    if not response.body or len(response.body) > max_bytes:
        raise DrandBeaconError("drand response is empty or exceeds its byte bound")
    length = _header(response.headers, "Content-Length")
    if length is not None and int(length) != len(response.body):
        raise DrandBeaconError("drand response length differs from Content-Length")
    return response.body


def _strict_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise DrandBeaconError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = encoded.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DrandBeaconError(f"{label} contains non-finite number {token}")
            ),
        )
    except DrandBeaconError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DrandBeaconError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise DrandBeaconError(f"{label} must be one JSON object")
    return value


def _closed(value: Mapping[str, Any], keys: frozenset[str], *, label: str) -> None:
    if set(value) != keys:
        raise DrandBeaconError(f"{label} does not match the exact schema")


def _lower_hex(value: object, *, label: str, byte_count: int) -> str:
    if (
        type(value) is not str
        or len(value) != byte_count * 2
        or any(character not in _HEX for character in value)
    ):
        raise DrandBeaconError(f"{label} must be exactly {byte_count} lowercase hex bytes")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise DrandBeaconError(f"{label} must be a positive integer")
    return value


def _validate_contract(contract: ExecutionBeaconContract) -> None:
    if not isinstance(contract, ExecutionBeaconContract):
        raise DrandBeaconError("beacon contract must be typed")
    expected = {
        "drand_network": QUICKNET_NETWORK,
        "chain_hash": QUICKNET_CHAIN_HASH,
        "chain_scheme_id": QUICKNET_SCHEME_ID,
        "chain_public_key": QUICKNET_PUBLIC_KEY,
        "chain_genesis_unix_seconds": QUICKNET_GENESIS_UNIX_SECONDS,
        "chain_period_seconds": QUICKNET_PERIOD_SECONDS,
    }
    for name, value in expected.items():
        if getattr(contract, name) != value:
            raise DrandBeaconError(f"beacon contract {name} differs from frozen Quicknet")


def _validate_chain_info(encoded: bytes, contract: ExecutionBeaconContract) -> None:
    value = _strict_object(encoded, label="Quicknet chain information")
    _closed(
        value,
        frozenset(
            {"public_key", "period", "genesis_time", "hash", "groupHash", "schemeID", "metadata"}
        ),
        label="Quicknet chain information",
    )
    metadata = value["metadata"]
    if not isinstance(metadata, Mapping):
        raise DrandBeaconError("Quicknet metadata must be one JSON object")
    _closed(metadata, frozenset({"beaconID"}), label="Quicknet metadata")
    public_key = _lower_hex(value["public_key"], label="Quicknet public key", byte_count=96)
    group_hash = _lower_hex(value["groupHash"], label="Quicknet group hash", byte_count=32)
    chain_hash = _lower_hex(value["hash"], label="Quicknet chain hash", byte_count=32)
    period = _positive_integer(value["period"], label="Quicknet period")
    genesis = _positive_integer(value["genesis_time"], label="Quicknet genesis time")
    if type(value["schemeID"]) is not str or type(metadata["beaconID"]) is not str:
        raise DrandBeaconError("Quicknet scheme and beacon ID must be strings")
    computed_hash = hashlib.sha256(
        struct.pack(">Iq", period, genesis)
        + bytes.fromhex(public_key)
        + bytes.fromhex(group_hash)
        + metadata["beaconID"].encode("utf-8")
    ).hexdigest()
    if computed_hash != chain_hash:
        raise DrandBeaconError("Quicknet chain information hash is not canonical")
    expected = (
        (public_key, contract.chain_public_key, "public key"),
        (group_hash, QUICKNET_GROUP_HASH, "group hash"),
        (chain_hash, contract.chain_hash, "chain hash"),
        (period, contract.chain_period_seconds, "period"),
        (genesis, contract.chain_genesis_unix_seconds, "genesis time"),
        (value["schemeID"], contract.chain_scheme_id, "scheme"),
        (metadata["beaconID"], QUICKNET_BEACON_ID, "beacon ID"),
    )
    for observed, admitted, label in expected:
        if observed != admitted:
            raise DrandBeaconError(f"Quicknet {label} differs from the frozen chain")


def _parse_beacon(encoded: bytes, *, expected_round: int) -> tuple[str, str]:
    value = _strict_object(encoded, label="Quicknet beacon")
    _closed(value, frozenset({"round", "randomness", "signature"}), label="Quicknet beacon")
    round_number = _positive_integer(value["round"], label="Quicknet round")
    if round_number != expected_round:
        raise DrandBeaconError("Quicknet beacon is for another round")
    signature = _lower_hex(value["signature"], label="Quicknet signature", byte_count=48)
    randomness = _lower_hex(value["randomness"], label="Quicknet randomness", byte_count=32)
    if hashlib.sha256(bytes.fromhex(signature)).hexdigest() != randomness:
        raise DrandBeaconError("Quicknet randomness differs from SHA-256(signature)")
    return signature, randomness


def _decode_signature(encoded: bytes) -> Any:
    try:
        compressed = int.from_bytes(encoded, "big")
        point = decompress_G1(compressed)
        if int(compress_G1(point)) != compressed:
            raise DrandBeaconError("Quicknet signature encoding is not canonical")
    except DrandBeaconError:
        raise
    except (TypeError, ValueError, AssertionError) as exc:
        raise DrandBeaconError("Quicknet signature is not a compressed G1 point") from exc
    if is_inf(point):
        raise DrandBeaconError("Quicknet signature is the point at infinity")
    if not subgroup_check(point):
        raise DrandBeaconError("Quicknet signature is outside the prime-order subgroup")
    return point


def _decode_public_key(encoded: bytes) -> Any:
    try:
        compressed = (
            int.from_bytes(encoded[:48], "big"),
            int.from_bytes(encoded[48:], "big"),
        )
        point = decompress_G2(compressed)
        if tuple(int(value) for value in compress_G2(point)) != compressed:
            raise DrandBeaconError("Quicknet public-key encoding is not canonical")
    except DrandBeaconError:
        raise
    except (TypeError, ValueError, AssertionError) as exc:
        raise DrandBeaconError("Quicknet public key is not a compressed G2 point") from exc
    if is_inf(point):
        raise DrandBeaconError("Quicknet public key is the point at infinity")
    if not subgroup_check(point):
        raise DrandBeaconError("Quicknet public key is outside the prime-order subgroup")
    return point


def _verify_bls_signature(*, round_number: int, public_key: bytes, signature: bytes) -> None:
    signature_point = _decode_signature(signature)
    public_key_point = _decode_public_key(public_key)
    round_digest = hashlib.sha256(round_number.to_bytes(8, "big")).digest()
    message_point = hash_to_G1(round_digest, QUICKNET_G1_DST, hashlib.sha256)
    try:
        verified = pairing(public_key_point, message_point) == pairing(G2, signature_point)
    except (TypeError, ValueError, AssertionError) as exc:
        raise DrandBeaconError("Quicknet BLS pairing failed") from exc
    if not verified:
        raise DrandBeaconError("Quicknet BLS signature is invalid")


class QuicknetExecutionBeaconVerifier:
    """Fetch and verify the frozen RFC 9380 Quicknet beacon contract."""

    def __init__(self, api: DrandReadApi | None = None) -> None:
        self._api = UrllibDrandReadApi() if api is None else api
        if not hasattr(self._api, "get"):
            raise DrandBeaconError("drand read API is invalid")

    @staticmethod
    def chain_info_url(contract: ExecutionBeaconContract) -> str:
        _validate_contract(contract)
        return f"{QUICKNET_NETWORK}/{QUICKNET_CHAIN_HASH}/info"

    @staticmethod
    def beacon_url(contract: ExecutionBeaconContract) -> str:
        _validate_contract(contract)
        return f"{QUICKNET_NETWORK}/{QUICKNET_CHAIN_HASH}/public/{contract.execution_round}"

    def fetch_and_verify(
        self, contract: ExecutionBeaconContract
    ) -> tuple[bytes, VerifiedBeaconClaims]:
        """Return exact relay bytes only after metadata and BLS verification."""

        info_url = self.chain_info_url(contract)
        info = _response_body(
            self._api.get(info_url, max_bytes=MAX_CHAIN_INFO_BYTES),
            url=info_url,
            max_bytes=MAX_CHAIN_INFO_BYTES,
        )
        _validate_chain_info(info, contract)
        beacon_url = self.beacon_url(contract)
        beacon = _response_body(
            self._api.get(beacon_url, max_bytes=MAX_BEACON_BYTES),
            url=beacon_url,
            max_bytes=MAX_BEACON_BYTES,
        )
        return beacon, self.verify(contract=contract, beacon_bytes=beacon)

    def fetch(self, contract: ExecutionBeaconContract) -> bytes:
        """Return the exact verified beacon response bytes."""

        return self.fetch_and_verify(contract)[0]

    def verify(
        self,
        *,
        contract: ExecutionBeaconContract,
        beacon_bytes: bytes,
    ) -> VerifiedBeaconClaims:
        """Implement ``ExecutionBeaconVerifier`` with actual short BLS verification."""

        _validate_contract(contract)
        if not isinstance(beacon_bytes, bytes) or not beacon_bytes:
            raise DrandBeaconError("Quicknet beacon bytes must be non-empty bytes")
        if len(beacon_bytes) > MAX_BEACON_BYTES:
            raise DrandBeaconError("Quicknet beacon exceeds its byte bound")
        signature, randomness = _parse_beacon(
            beacon_bytes,
            expected_round=contract.execution_round,
        )
        _verify_bls_signature(
            round_number=contract.execution_round,
            public_key=bytes.fromhex(contract.chain_public_key),
            signature=bytes.fromhex(signature),
        )
        return VerifiedBeaconClaims(
            chain_hash=contract.chain_hash,
            round=contract.execution_round,
            beacon_bytes_sha256=hashlib.sha256(beacon_bytes).hexdigest(),
            randomness=randomness,
            signature=signature,
            scheme_id=contract.chain_scheme_id,
            public_key=contract.chain_public_key,
            signature_verified=True,
        )


def fetch_quicknet_beacon(
    contract: ExecutionBeaconContract,
    *,
    api: DrandReadApi | None = None,
) -> bytes:
    """Fetch and cryptographically verify one exact frozen Quicknet round."""

    return QuicknetExecutionBeaconVerifier(api).fetch(contract)
