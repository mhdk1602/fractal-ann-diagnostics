from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from fractal_ann_diagnostics.compiled_policy import (
    COMPILED_MASK_ENCODING,
    COMPILED_POLICY_CATALOG_SCHEMA,
    CompiledMaskDescriptor,
    CompiledPolicyCatalog,
    CompiledPolicyError,
    CompiledPolicyMaskStore,
    OpenPolicyAgentMaskDecisionPoint,
    compiled_mask_descriptor,
    load_compiled_policy_catalog,
    write_compiled_mask,
    write_compiled_policy_catalog,
)
from fractal_ann_diagnostics.controller import ControllerConfig, GovernedRetriever, RuleController
from fractal_ann_diagnostics.policy import policy_document_universe_sha256
from fractal_ann_diagnostics.policy_adapters import OPAHTTPResponse


class _Transport:
    def __init__(self, responder: object) -> None:
        self.responder = responder
        self.calls: list[tuple[str, bytes, float]] = []

    def __call__(self, endpoint: str, body: bytes, timeout: float) -> OPAHTTPResponse:
        self.calls.append((endpoint, body, timeout))
        if callable(self.responder):
            return self.responder(json.loads(body)["input"])
        assert isinstance(self.responder, OPAHTTPResponse)
        return self.responder


def _mask_response(
    descriptor: CompiledMaskDescriptor,
    *,
    extra: Mapping[str, object] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> object:
    def respond(policy_input: Mapping[str, object]) -> OPAHTTPResponse:
        result = {
            key: policy_input[key]
            for key in (
                "action",
                "catalog_request_sha256",
                "document_count",
                "document_universe_sha256",
                "environment_sha256",
                "mask_catalog_sha256",
                "policy_revision",
                "request_nonce",
                "request_sha256",
                "subject",
            )
        }
        result.update(
            {
                "authorized_count": descriptor.authorized_count,
                "mask_id": descriptor.mask_id,
                "mask_sha256": descriptor.sha256,
            }
        )
        result.update(overrides or {})
        result.update(extra or {})
        return OPAHTTPResponse(
            status=200,
            body=json.dumps({"decision_id": "opa-decision-1", "result": result}).encode(),
        )

    return respond


def _store(
    tmp_path: Path,
    *,
    document_count: int = 17,
) -> tuple[CompiledPolicyMaskStore, CompiledMaskDescriptor, np.ndarray]:
    mask_directory = tmp_path / "masks"
    mask_directory.mkdir()
    mask = np.zeros(document_count, dtype=bool)
    mask[::3] = True
    descriptor, encoded = compiled_mask_descriptor(
        "reader-us",
        "masks/reader-us.bin",
        mask,
        document_count=document_count,
    )
    write_compiled_mask(encoded, mask_directory / "reader-us.bin")
    universe = policy_document_universe_sha256(
        f"stable-document-{index}" for index in range(document_count)
    )
    catalog = CompiledPolicyCatalog(
        document_count=document_count,
        document_universe_sha256=universe,
        policy_revision="opa-bundle-sha256:0123456789abcdef",
        masks=(descriptor,),
    )
    catalog_path = tmp_path / "catalog.json"
    write_compiled_policy_catalog(catalog, catalog_path)
    return CompiledPolicyMaskStore(catalog_path), descriptor, mask


def test_compiled_mask_round_trip_and_catalog_pin(tmp_path: Path) -> None:
    store, descriptor, expected = _store(tmp_path)

    observed = store.mask(
        descriptor.mask_id,
        expected_sha256=descriptor.sha256,
        expected_authorized_count=descriptor.authorized_count,
    )

    np.testing.assert_array_equal(observed, expected)
    assert not observed.flags.writeable
    assert store.verify_all() == (descriptor.mask_id,)
    assert (
        store.catalog.artifact_sha256
        == hashlib.sha256(store.catalog.canonical_bytes() + b"\n").hexdigest()
    )


def test_opa_request_is_constant_schema_and_contains_no_document_id_array(
    tmp_path: Path,
) -> None:
    store, descriptor, expected = _store(tmp_path, document_count=1_000_003)
    transport = _Transport(_mask_response(descriptor))
    pdp = OpenPolicyAgentMaskDecisionPoint(
        "http://127.0.0.1:8181/v1/data/fractal/retrieval/mask",
        store,
        transport=transport,
    )

    decision = pdp.decide("reader", environment={"region": "us"})

    assert decision.available
    assert decision.authorized_count == int(expected.sum())
    np.testing.assert_array_equal(decision.authorized_mask, expected)
    request = json.loads(transport.calls[0][1])["input"]
    assert request["document_count"] == 1_000_003
    assert "document_ids" not in request
    assert len(transport.calls[0][1]) < 2048
    assert request["mask_catalog_sha256"] == store.catalog_sha256
    assert request["catalog_request_sha256"]


@pytest.mark.parametrize(
    ("overrides", "extra"),
    [
        ({"mask_sha256": "f" * 64}, None),
        ({"authorized_count": 0}, None),
        ({"catalog_request_sha256": "0" * 64}, None),
        ({"document_count": 18}, None),
        (None, {"allowed_document_ids": [0]}),
    ],
)
def test_misbound_or_expanded_opa_result_fails_closed(
    tmp_path: Path,
    overrides: Mapping[str, object] | None,
    extra: Mapping[str, object] | None,
) -> None:
    store, descriptor, _ = _store(tmp_path)
    pdp = OpenPolicyAgentMaskDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/mask",
        store,
        transport=_Transport(_mask_response(descriptor, overrides=overrides, extra=extra)),
    )

    decision = pdp.decide("reader")

    assert not decision.available
    assert decision.authorized_count == 0
    assert "response validation failed" in decision.reason


def test_mask_substitution_fails_closed(tmp_path: Path) -> None:
    store, descriptor, _ = _store(tmp_path)
    (tmp_path / descriptor.path).write_bytes(b"\xff" * descriptor.byte_count)
    pdp = OpenPolicyAgentMaskDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/mask",
        store,
        transport=_Transport(_mask_response(descriptor)),
    )

    decision = pdp.decide("reader")

    assert not decision.available
    assert decision.authorized_count == 0


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_mask_links_are_rejected(tmp_path: Path, link_kind: str) -> None:
    store, descriptor, _ = _store(tmp_path)
    mask_path = tmp_path / descriptor.path
    if link_kind == "symlink":
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(mask_path.read_bytes())
        mask_path.unlink()
        mask_path.symlink_to(replacement)
    else:
        os.link(mask_path, tmp_path / "second-link.bin")

    with pytest.raises(CompiledPolicyError, match="compiled policy mask"):
        store.mask(
            descriptor.mask_id,
            expected_sha256=descriptor.sha256,
            expected_authorized_count=descriptor.authorized_count,
        )


def test_nonzero_unused_mask_bits_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "masks").mkdir()
    malformed = bytes([0x01, 0x80])
    path = tmp_path / "masks" / "reader.bin"
    write_compiled_mask(malformed, path)
    descriptor = CompiledMaskDescriptor(
        mask_id="reader",
        path="masks/reader.bin",
        sha256=hashlib.sha256(malformed).hexdigest(),
        byte_count=2,
        authorized_count=1,
    )
    catalog = CompiledPolicyCatalog(
        document_count=9,
        document_universe_sha256=policy_document_universe_sha256(range(9)),
        policy_revision="bundle-1",
        masks=(descriptor,),
    )
    catalog_path = tmp_path / "catalog.json"
    write_compiled_policy_catalog(catalog, catalog_path)
    store = CompiledPolicyMaskStore(catalog_path)

    with pytest.raises(CompiledPolicyError, match="trailing bits"):
        store.verify_all()


def test_catalog_loader_rejects_unknown_fields_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    store, _, _ = _store(tmp_path)
    payload = store.catalog.to_dict()
    payload["unknown"] = True
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(CompiledPolicyError, match="unknown"):
        load_compiled_policy_catalog(bad)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(store.catalog.to_dict(), indent=2) + "\n")
    with pytest.raises(CompiledPolicyError, match="canonical JSON"):
        load_compiled_policy_catalog(noncanonical)


def test_catalog_literals_and_writer_exclusivity(tmp_path: Path) -> None:
    store, _, _ = _store(tmp_path)
    assert store.catalog.schema_version == COMPILED_POLICY_CATALOG_SCHEMA
    assert store.catalog.encoding == COMPILED_MASK_ENCODING

    with pytest.raises(CompiledPolicyError, match="cannot write"):
        write_compiled_policy_catalog(store.catalog, store.catalog_path)


def test_opa_timeout_fails_closed(tmp_path: Path) -> None:
    store, _, _ = _store(tmp_path)

    def timeout(endpoint: str, body: bytes, seconds: float) -> OPAHTTPResponse:
        del endpoint, body, seconds
        raise TimeoutError

    pdp = OpenPolicyAgentMaskDecisionPoint(
        "https://opa.example.test/v1/data/fractal/retrieval/mask",
        store,
        transport=timeout,
    )

    decision = pdp.decide("reader")

    assert not decision.available
    assert "timed out" in decision.reason


def test_remote_builtin_transport_requires_identity(tmp_path: Path) -> None:
    store, _, _ = _store(tmp_path)
    with pytest.raises(ValueError, match="bearer token"):
        OpenPolicyAgentMaskDecisionPoint(
            "https://opa.example.test/v1/data/fractal/retrieval/mask",
            store,
        )


def test_compiled_mask_adapter_runs_both_governed_authorization_boundaries(
    tmp_path: Path,
) -> None:
    store, descriptor, expected = _store(tmp_path)
    decisions = 0

    def respond(policy_input: Mapping[str, object]) -> OPAHTTPResponse:
        nonlocal decisions
        decisions += 1
        base = _mask_response(descriptor)
        response = base(policy_input)  # type: ignore[operator]
        payload = json.loads(response.body)
        payload["decision_id"] = f"opa-decision-{decisions}"
        return OPAHTTPResponse(status=200, body=json.dumps(payload).encode())

    pdp = OpenPolicyAgentMaskDecisionPoint(
        "http://127.0.0.1:8181/v1/data/fractal/retrieval/mask",
        store,
        transport=_Transport(respond),
    )
    vectors = np.arange(17 * 4, dtype=np.float32).reshape(17, 4)
    retriever = GovernedRetriever(
        vectors,
        pdp,
        "reader",
        expected_document_universe_sha256=store.catalog.document_universe_sha256,
        controller=RuleController(
            ControllerConfig(
                low_ef=8,
                high_ef=16,
                probe_k=5,
                exact_scan_threshold=17,
            )
        ),
    )

    result = retriever.query(vectors[0], k=3)

    assert result.search is not None
    assert decisions == 2
    assert expected[result.search.ids].all()
    assert result.initial_authorization is not None
    assert result.final_authorization is not None
    assert result.initial_authorization.request_nonce != result.final_authorization.request_nonce
