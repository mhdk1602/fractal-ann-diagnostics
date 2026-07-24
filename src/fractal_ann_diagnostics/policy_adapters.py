"""Adapters that obtain authorization from external policy decision points."""

from __future__ import annotations

import hmac
import ipaddress
import json
import math
import re
import secrets
import socket
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NoReturn, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

import numpy as np

from .policy import (
    EMPTY_POLICY_ENVIRONMENT_SHA256,
    PolicyDecision,
    policy_environment_sha256,
    policy_request_sha256,
)

MAX_RESPONSE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class OPAHTTPResponse:
    """Minimal HTTP result returned by an OPA transport."""

    status: int
    body: bytes


class OPATransport(Protocol):
    """Injectable transport used by :class:`OpenPolicyAgentDecisionPoint`."""

    def __call__(
        self,
        endpoint: str,
        body: bytes,
        timeout_seconds: float,
    ) -> OPAHTTPResponse: ...


class _RejectRedirects(HTTPRedirectHandler):
    """Keep every decision on the configured PDP origin and scheme."""

    def redirect_request(  # type: ignore[override]
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def _urllib_transport(
    endpoint: str,
    body: bytes,
    timeout_seconds: float,
    *,
    bearer_token: str | None,
    ssl_context: ssl.SSLContext | None,
) -> OPAHTTPResponse:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if bearer_token is not None:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = Request(
        endpoint,
        data=body,
        headers=headers,
        method="POST",
    )
    opener = build_opener(_RejectRedirects(), HTTPSHandler(context=ssl_context))
    try:
        with opener.open(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            return OPAHTTPResponse(status=int(response.status), body=payload)
    except HTTPError as exc:
        return OPAHTTPResponse(
            status=int(exc.code),
            body=exc.read(MAX_RESPONSE_BYTES + 1),
        )


class _ResponseSchemaError(ValueError):
    pass


def _closed_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _ResponseSchemaError(f"OPA response contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> NoReturn:
    raise _ResponseSchemaError(f"OPA response contains non-finite JSON constant {value!r}")


def _is_literal_loopback(hostname: str | None) -> bool:
    if hostname is None:
        return False
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _required_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _ResponseSchemaError(f"{field} must be a non-empty string")
    return value.strip()


class OpenPolicyAgentDecisionPoint:
    """Bulk authorization adapter for the OPA Data API.

    The configured rule receives one request containing ``subject``, ``action``,
    ``environment``, and every integer ``document_id``. OPA must return this shape::

        {
          "decision_id": "opa-generated-id",
          "result": {
            "allowed_document_ids": [0, 4, 9],
            "policy_revision": "bundle-sha-or-version",
            "subject": "the-request-subject",
            "action": "retrieve",
            "environment_sha256": "...",
            "document_universe_sha256": "...",
            "request_nonce": "...",
            "request_sha256": "..."
          }
        }

    OPA emits the top-level ``decision_id`` when decision logging is configured.
    Every malformed or unavailable response becomes an unavailable deny-all
    :class:`PolicyDecision`.
    """

    def __init__(
        self,
        endpoint: str,
        n_documents: int,
        *,
        document_universe_sha256: str,
        expected_policy_revision: str,
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
        is_loopback = _is_literal_loopback(parsed.hostname)
        if parsed.scheme == "http" and not is_loopback:
            raise ValueError("plain HTTP OPA endpoints require a literal loopback IP address")
        if type(n_documents) is not int or n_documents < 0:
            raise ValueError("n_documents must be a non-negative integer")
        if not isinstance(expected_policy_revision, str) or not expected_policy_revision.strip():
            raise ValueError("expected_policy_revision must be a non-empty pinned revision")
        if expected_policy_revision != expected_policy_revision.strip():
            raise ValueError("expected_policy_revision must be canonical")
        if any(
            ord(character) < 32 or ord(character) == 127 for character in expected_policy_revision
        ):
            raise ValueError("expected_policy_revision cannot contain control characters")
        if bearer_token is not None and (
            not isinstance(bearer_token, str)
            or not bearer_token
            or bearer_token != bearer_token.strip()
            or any(ord(character) < 33 or ord(character) == 127 for character in bearer_token)
        ):
            raise ValueError("bearer_token must be a non-empty canonical credential")
        if ssl_context is not None and not isinstance(ssl_context, ssl.SSLContext):
            raise TypeError("ssl_context must be an ssl.SSLContext")
        if ssl_context is not None and (
            ssl_context.verify_mode != ssl.CERT_REQUIRED or ssl_context.check_hostname is not True
        ):
            raise ValueError(
                "ssl_context must require certificate verification and hostname checking"
            )
        if transport is None and not is_loopback and bearer_token is None:
            raise ValueError(
                "remote built-in OPA transport requires a bearer token; an SSLContext alone "
                "does not attest a client identity"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        self.endpoint = endpoint
        self._n_documents = n_documents
        if re.fullmatch(r"[0-9a-f]{64}", document_universe_sha256) is None:
            raise ValueError(
                "document_universe_sha256 must be an explicit lowercase SHA-256 "
                "digest of the ordered stable document IDs"
            )
        self._document_universe_sha256 = document_universe_sha256
        self.expected_policy_revision = expected_policy_revision
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
        return self._n_documents

    @property
    def document_universe_sha256(self) -> str:
        return self._document_universe_sha256

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

    def _request_body(
        self,
        subject: str,
        action: str,
        environment: Mapping[str, object] | None,
        environment_sha256: str,
    ) -> tuple[bytes, str, str]:
        if not isinstance(subject, str) or not subject:
            raise ValueError("subject must be a non-empty string")
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string")
        if environment is not None and not isinstance(environment, Mapping):
            raise ValueError("environment must be a mapping")
        request_nonce = secrets.token_hex(32)
        request_digest = policy_request_sha256(
            subject=subject,
            action=action,
            environment_sha256=environment_sha256,
            document_universe_sha256=self.document_universe_sha256,
            request_nonce=request_nonce,
        )
        policy_input = {
            "subject": subject,
            "action": action,
            "environment": dict(environment or {}),
            "environment_sha256": environment_sha256,
            "document_ids": list(range(self.n_documents)),
            "document_universe_sha256": self.document_universe_sha256,
            "request_nonce": request_nonce,
            "request_sha256": request_digest,
        }
        body = json.dumps(
            {"input": policy_input},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return body, request_nonce, request_digest

    def _parse_response(
        self,
        subject: str,
        action: str,
        environment_sha256: str,
        request_nonce: str,
        request_sha256: str,
        response: OPAHTTPResponse,
    ) -> PolicyDecision:
        if type(response.status) is not int or not 200 <= response.status < 300:
            raise _ResponseSchemaError("OPA returned a non-success HTTP status")
        if not isinstance(response.body, bytes):
            raise _ResponseSchemaError("OPA response body must be bytes")
        if len(response.body) > MAX_RESPONSE_BYTES:
            raise _ResponseSchemaError("OPA response exceeds the size limit")
        try:
            payload = json.loads(
                response.body,
                object_pairs_hook=_closed_json_object,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, _ResponseSchemaError) as exc:
            raise _ResponseSchemaError("OPA response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise _ResponseSchemaError("OPA response must be a JSON object")
        decision_id = _required_nonempty_string(payload.get("decision_id"), "decision_id")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise _ResponseSchemaError("result must be a JSON object")
        revision = _required_nonempty_string(
            result.get("policy_revision"),
            "policy_revision",
        )
        if not hmac.compare_digest(revision, self.expected_policy_revision):
            raise _ResponseSchemaError("OPA policy revision does not match the pinned bundle")
        echoed = {
            "subject": _required_nonempty_string(result.get("subject"), "subject"),
            "action": _required_nonempty_string(result.get("action"), "action"),
            "environment_sha256": _required_nonempty_string(
                result.get("environment_sha256"), "environment_sha256"
            ),
            "document_universe_sha256": _required_nonempty_string(
                result.get("document_universe_sha256"),
                "document_universe_sha256",
            ),
            "request_nonce": _required_nonempty_string(
                result.get("request_nonce"), "request_nonce"
            ),
            "request_sha256": _required_nonempty_string(
                result.get("request_sha256"), "request_sha256"
            ),
        }
        expected = {
            "subject": subject,
            "action": action,
            "environment_sha256": environment_sha256,
            "document_universe_sha256": self.document_universe_sha256,
            "request_nonce": request_nonce,
            "request_sha256": request_sha256,
        }
        if any(not hmac.compare_digest(echoed[field], value) for field, value in expected.items()):
            raise _ResponseSchemaError("OPA response is not bound to this request")
        allowed_ids = result.get("allowed_document_ids")
        if not isinstance(allowed_ids, list):
            raise _ResponseSchemaError("allowed_document_ids must be a JSON array")
        if any(type(document_id) is not int for document_id in allowed_ids):
            raise _ResponseSchemaError("allowed_document_ids must contain only integers")
        if len(set(allowed_ids)) != len(allowed_ids):
            raise _ResponseSchemaError("allowed_document_ids contains duplicates")
        if any(document_id < 0 or document_id >= self.n_documents for document_id in allowed_ids):
            raise _ResponseSchemaError("allowed_document_ids contains an out-of-range ID")

        mask = np.zeros(self.n_documents, dtype=bool)
        mask[allowed_ids] = True
        return PolicyDecision(
            subject=subject,
            action=action,
            policy_version=revision,
            authorized_mask=mask,
            decision_id=decision_id,
            reason="OPA bulk authorization evaluated",
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
            environment_digest = policy_environment_sha256(environment)
            body, request_nonce, request_digest = self._request_body(
                subject,
                action,
                environment,
                environment_digest,
            )
        except (TypeError, ValueError):
            return self._deny(subject, action, "OPA request validation failed; deny by default")

        try:
            response = self._transport(
                self.endpoint,
                body,
                self.timeout_seconds,
            )
        except (TimeoutError, socket.timeout):
            return self._deny(
                subject,
                action,
                "OPA request timed out; deny by default",
                environment_sha256=environment_digest,
                request_nonce=request_nonce,
            )
        except HTTPError as exc:
            return self._deny(
                subject,
                action,
                f"OPA returned HTTP {exc.code}; deny by default",
                environment_sha256=environment_digest,
                request_nonce=request_nonce,
            )
        except (URLError, OSError):
            return self._deny(
                subject,
                action,
                "OPA transport failed; deny by default",
                environment_sha256=environment_digest,
                request_nonce=request_nonce,
            )
        except Exception:
            return self._deny(
                subject,
                action,
                "OPA transport failed; deny by default",
                environment_sha256=environment_digest,
                request_nonce=request_nonce,
            )

        if not isinstance(response, OPAHTTPResponse):
            return self._deny(
                subject,
                action,
                "OPA transport response is invalid; deny by default",
                environment_sha256=environment_digest,
                request_nonce=request_nonce,
            )
        if type(response.status) is not int or not 200 <= response.status < 300:
            return self._deny(
                subject,
                action,
                f"OPA returned HTTP {response.status}; deny by default",
                environment_sha256=environment_digest,
                request_nonce=request_nonce,
            )
        try:
            return self._parse_response(
                subject,
                action,
                environment_digest,
                request_nonce,
                request_digest,
                response,
            )
        except _ResponseSchemaError:
            return self._deny(
                subject,
                action,
                "OPA response validation failed; deny by default",
                environment_sha256=environment_digest,
                request_nonce=request_nonce,
            )
