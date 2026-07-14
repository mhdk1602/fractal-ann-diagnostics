from __future__ import annotations

import json
import ssl
from collections.abc import Callable, Mapping
from typing import NoReturn
from urllib.error import URLError

import pytest

from fractal_ann_diagnostics.policy import policy_document_universe_sha256
from fractal_ann_diagnostics.policy_adapters import (
    MAX_RESPONSE_BYTES,
    OPAHTTPResponse,
    _RejectRedirects,
)
from fractal_ann_diagnostics.policy_adapters import (
    OpenPolicyAgentDecisionPoint as _OpenPolicyAgentDecisionPoint,
)


def OpenPolicyAgentDecisionPoint(*args: object, **kwargs: object) -> object:
    """Test factory with one pinned mock-bundle revision."""

    kwargs.setdefault("expected_policy_revision", "bundle-7f21")
    return _OpenPolicyAgentDecisionPoint(*args, **kwargs)  # type: ignore[arg-type]


def _universe(n_documents: int) -> str:
    return policy_document_universe_sha256(
        f"stable-document-{index}" for index in range(n_documents)
    )


class _RecordingTransport:
    def __init__(
        self,
        response: OPAHTTPResponse | Callable[[Mapping[str, object]], OPAHTTPResponse],
    ) -> None:
        self.response = response
        self.calls: list[tuple[str, bytes, float]] = []

    def __call__(
        self,
        endpoint: str,
        body: bytes,
        timeout_seconds: float,
    ) -> OPAHTTPResponse:
        self.calls.append((endpoint, body, timeout_seconds))
        if callable(self.response):
            payload = json.loads(body)
            return self.response(payload["input"])
        return self.response


def _success_response(
    allowed_ids: object,
    *,
    revision: object = "bundle-7f21",
    decision_id: object = "decision-123",
    binding_overrides: Mapping[str, object] | None = None,
) -> Callable[[Mapping[str, object]], OPAHTTPResponse]:
    def respond(policy_input: Mapping[str, object]) -> OPAHTTPResponse:
        binding = {
            field: policy_input[field]
            for field in (
                "subject",
                "action",
                "environment_sha256",
                "document_universe_sha256",
                "request_nonce",
                "request_sha256",
            )
        }
        binding.update(binding_overrides or {})
        return OPAHTTPResponse(
            status=200,
            body=json.dumps(
                {
                    "decision_id": decision_id,
                    "result": {
                        "allowed_document_ids": allowed_ids,
                        "policy_revision": revision,
                        **binding,
                    },
                }
            ).encode(),
        )

    return respond


def test_bulk_decision_preserves_inputs_and_returns_immutable_mask() -> None:
    transport = _RecordingTransport(_success_response([0, 3, 5]))
    pdp = OpenPolicyAgentDecisionPoint(
        "https://opa.internal:8181/v1/data/fractal/retrieval/decision",
        n_documents=6,
        document_universe_sha256=_universe(6),
        timeout_seconds=1.25,
        transport=transport,
    )
    environment = {
        "tenant": "research",
        "region": "us-east",
        "claims": {"clearance": 4},
    }

    decision = pdp.decide(
        "user-42",
        action="retrieve",
        environment=environment,
    )

    assert decision.available
    assert decision.subject == "user-42"
    assert decision.action == "retrieve"
    assert decision.policy_version == "bundle-7f21"
    assert decision.decision_id == "decision-123"
    assert decision.authorized_mask.tolist() == [True, False, False, True, False, True]
    assert not decision.authorized_mask.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        decision.authorized_mask[0] = False

    assert len(transport.calls) == 1
    endpoint, body, timeout = transport.calls[0]
    assert endpoint == "https://opa.internal:8181/v1/data/fractal/retrieval/decision"
    assert timeout == 1.25
    policy_input = json.loads(body)["input"]
    assert policy_input["subject"] == "user-42"
    assert policy_input["action"] == "retrieve"
    assert policy_input["environment"] == environment
    assert policy_input["document_ids"] == [0, 1, 2, 3, 4, 5]
    assert decision.document_universe_sha256 == _universe(6)
    assert policy_input["environment_sha256"] == decision.environment_sha256
    assert policy_input["document_universe_sha256"] == decision.document_universe_sha256
    assert policy_input["request_nonce"] == decision.request_nonce
    assert policy_input["request_sha256"] == decision.request_sha256


@pytest.mark.parametrize(
    "allowed_ids",
    [
        [1, 1],
        [-1],
        [4],
        [True],
        [1.0],
        ["1"],
    ],
)
def test_invalid_allowed_document_ids_fail_closed(allowed_ids: object) -> None:
    pdp = OpenPolicyAgentDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/decision",
        n_documents=4,
        document_universe_sha256=_universe(4),
        transport=_RecordingTransport(_success_response(allowed_ids)),
    )

    decision = pdp.decide("reader")

    assert not decision.available
    assert decision.authorized_count == 0
    assert decision.policy_version == "unavailable"
    assert "response validation failed" in decision.reason


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b"{}",
        b'{"decision_id":"d","result":false}',
        b'{"decision_id":"d","result":{"allowed_document_ids":[]}}',
        b'{"decision_id":"d","result":{"allowed_document_ids":{},'
        b'"policy_revision":"v1"}}',
        b'{"decision_id":"","result":{"allowed_document_ids":[],'
        b'"policy_revision":"v1"}}',
        b'{"decision_id":"d","result":{"allowed_document_ids":[],'
        b'"policy_revision":""}}',
    ],
)
def test_invalid_json_or_schema_fails_closed(body: bytes) -> None:
    pdp = OpenPolicyAgentDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/decision",
        n_documents=4,
        document_universe_sha256=_universe(4),
        transport=_RecordingTransport(OPAHTTPResponse(status=200, body=body)),
    )

    decision = pdp.decide("reader")

    assert not decision.available
    assert decision.authorized_count == 0
    assert "response validation failed" in decision.reason


@pytest.mark.parametrize("status", [400, 403, 500, 503])
def test_http_errors_fail_closed(status: int) -> None:
    pdp = OpenPolicyAgentDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/decision",
        n_documents=4,
        document_universe_sha256=_universe(4),
        transport=_RecordingTransport(OPAHTTPResponse(status=status, body=b"{}")),
    )

    decision = pdp.decide("reader")

    assert not decision.available
    assert decision.authorized_count == 0
    assert f"HTTP {status}" in decision.reason


def test_timeout_fails_closed() -> None:
    def timeout_transport(
        endpoint: str,
        body: bytes,
        timeout_seconds: float,
    ) -> NoReturn:
        del endpoint, body, timeout_seconds
        raise TimeoutError

    pdp = OpenPolicyAgentDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/decision",
        n_documents=4,
        document_universe_sha256=_universe(4),
        transport=timeout_transport,
    )

    decision = pdp.decide("reader")

    assert not decision.available
    assert decision.authorized_count == 0
    assert "timed out" in decision.reason


def test_transport_error_fails_closed() -> None:
    def failed_transport(
        endpoint: str,
        body: bytes,
        timeout_seconds: float,
    ) -> NoReturn:
        del endpoint, body, timeout_seconds
        raise URLError("connection refused")

    pdp = OpenPolicyAgentDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/decision",
        n_documents=4,
        document_universe_sha256=_universe(4),
        transport=failed_transport,
    )

    decision = pdp.decide("reader")

    assert not decision.available
    assert decision.authorized_count == 0
    assert "transport failed" in decision.reason


def test_unserializable_environment_fails_before_transport() -> None:
    transport = _RecordingTransport(_success_response([0]))
    pdp = OpenPolicyAgentDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/decision",
        n_documents=4,
        document_universe_sha256=_universe(4),
        transport=transport,
    )

    decision = pdp.decide("reader", environment={"value": object()})

    assert not decision.available
    assert decision.authorized_count == 0
    assert not transport.calls


def test_replayed_response_for_an_earlier_nonce_fails_closed() -> None:
    captured: OPAHTTPResponse | None = None

    def replay(policy_input: Mapping[str, object]) -> OPAHTTPResponse:
        nonlocal captured
        if captured is None:
            captured = _success_response([0])(policy_input)
        return captured

    pdp = OpenPolicyAgentDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/decision",
        n_documents=2,
        document_universe_sha256=_universe(2),
        transport=_RecordingTransport(replay),
    )

    assert pdp.decide("reader").available
    replayed = pdp.decide("reader")
    assert not replayed.available
    assert "response validation failed" in replayed.reason


@pytest.mark.parametrize("mutation", ["duplicate-key", "nonfinite"])
def test_noncanonical_json_response_fails_closed(mutation: str) -> None:
    def malformed(policy_input: Mapping[str, object]) -> OPAHTTPResponse:
        response = _success_response([0])(policy_input)
        if mutation == "duplicate-key":
            body = response.body.replace(
                b'"decision_id": "decision-123"',
                b'"decision_id": "first", "decision_id": "second"',
                1,
            )
        else:
            body = response.body.replace(b"{", b'{"diagnostic": NaN,', 1)
        return OPAHTTPResponse(status=200, body=body)

    pdp = OpenPolicyAgentDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/decision",
        n_documents=2,
        document_universe_sha256=_universe(2),
        transport=_RecordingTransport(malformed),
    )
    decision = pdp.decide("reader")
    assert not decision.available
    assert decision.authorized_count == 0
    assert "response validation failed" in decision.reason


def test_unpinned_policy_bundle_revision_fails_closed() -> None:
    pdp = OpenPolicyAgentDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/decision",
        n_documents=2,
        document_universe_sha256=_universe(2),
        transport=_RecordingTransport(_success_response([0], revision="bundle-stale")),
    )
    decision = pdp.decide("reader")
    assert not decision.available
    assert decision.authorized_count == 0
    assert "response validation failed" in decision.reason


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("subject", "another-reader"),
        ("action", "delete"),
        ("environment_sha256", "0" * 64),
        ("document_universe_sha256", "1" * 64),
        ("request_nonce", "stale-nonce"),
        ("request_sha256", "2" * 64),
    ],
)
def test_mismatched_request_binding_fails_closed(field: str, wrong_value: str) -> None:
    pdp = OpenPolicyAgentDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/decision",
        n_documents=2,
        document_universe_sha256=_universe(2),
        transport=_RecordingTransport(
            _success_response([0], binding_overrides={field: wrong_value})
        ),
    )

    decision = pdp.decide("reader", environment={"tenant": "research"})
    assert not decision.available
    assert decision.authorized_count == 0


def test_oversized_and_invalid_transport_responses_fail_closed() -> None:
    oversized = _RecordingTransport(
        OPAHTTPResponse(status=200, body=b"x" * (MAX_RESPONSE_BYTES + 1))
    )

    def invalid(endpoint: str, body: bytes, timeout: float) -> tuple[int, bytes]:
        del endpoint, body, timeout
        return 200, b"{}"

    for transport in (oversized, invalid):
        pdp = OpenPolicyAgentDecisionPoint(
            "https://opa.example.test/v1/data/fractal/retrieval/decision",
            n_documents=4,
            document_universe_sha256=_universe(4),
            transport=transport,  # type: ignore[arg-type]
        )
        decision = pdp.decide("reader")
        assert not decision.available
        assert decision.authorized_count == 0


@pytest.mark.parametrize(
    ("endpoint", "n_documents", "timeout"),
    [
        ("file:///tmp/opa.sock", 4, 1.0),
        ("opa.internal/v1/data/authz", 4, 1.0),
        ("http://opa.internal/v1/data/authz", 4, 1.0),
        ("https://user:secret@opa.internal/v1/data/authz", 4, 1.0),
        ("https://opa.internal/v1/data/authz?debug=true", 4, 1.0),
        ("https://opa.example.test", -1, 1.0),
        ("https://opa.example.test", True, 1.0),
        ("https://opa.example.test", 4, 0.0),
        ("https://opa.example.test", 4, float("nan")),
    ],
)
def test_constructor_rejects_invalid_configuration(
    endpoint: str,
    n_documents: object,
    timeout: float,
) -> None:
    with pytest.raises(ValueError):
        OpenPolicyAgentDecisionPoint(
            endpoint,
            n_documents,  # type: ignore[arg-type]
            document_universe_sha256=_universe(4),
            timeout_seconds=timeout,
        )


def test_plaintext_http_is_only_allowed_for_loopback_development() -> None:
    pdp = OpenPolicyAgentDecisionPoint(
        "http://127.0.0.1:8181/v1/data/fractal/retrieval/decision",
        1,
        document_universe_sha256=_universe(1),
        transport=_RecordingTransport(_success_response([])),
    )
    assert pdp.decide("reader").available


def test_opa_requires_an_explicit_stable_document_universe_digest() -> None:
    with pytest.raises(TypeError, match="document_universe_sha256"):
        OpenPolicyAgentDecisionPoint(  # type: ignore[call-arg]
            "https://opa.example.test/v1/data/fractal/retrieval/decision",
            4,
        )


def test_remote_builtin_transport_requires_application_authentication() -> None:
    with pytest.raises(ValueError, match="requires a bearer token"):
        _OpenPolicyAgentDecisionPoint(
            "https://opa.example.test/v1/data/fractal/retrieval/decision",
            4,
            document_universe_sha256=_universe(4),
            expected_policy_revision="bundle-7f21",
        )

    context = ssl.create_default_context()
    with pytest.raises(ValueError, match="SSLContext alone"):
        _OpenPolicyAgentDecisionPoint(
            "https://opa.example.test/v1/data/fractal/retrieval/decision",
            4,
            document_universe_sha256=_universe(4),
            expected_policy_revision="bundle-7f21",
            ssl_context=context,
        )


def test_remote_builtin_transport_rejects_insecure_tls_context() -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with pytest.raises(ValueError, match="certificate verification and hostname checking"):
        _OpenPolicyAgentDecisionPoint(
            "https://opa.example.test/v1/data/fractal/retrieval/decision",
            4,
            document_universe_sha256=_universe(4),
            expected_policy_revision="bundle-7f21",
            bearer_token="test-token",
            ssl_context=context,
        )


def test_plain_http_rejects_hostname_alias_even_when_it_might_resolve_loopback() -> None:
    with pytest.raises(ValueError, match="literal loopback IP"):
        OpenPolicyAgentDecisionPoint(
            "http://localhost:8181/v1/data/fractal/retrieval/decision",
            1,
            document_universe_sha256=_universe(1),
            transport=_RecordingTransport(_success_response([])),
        )


def test_redirect_handler_rejects_origin_or_scheme_changes() -> None:
    handler = _RejectRedirects()
    assert (
        handler.redirect_request(  # type: ignore[arg-type]
            None,
            None,
            307,
            "redirect",
            None,
            "http://attacker.example/opa",
        )
        is None
    )
