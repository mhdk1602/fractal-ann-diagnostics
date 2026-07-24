from __future__ import annotations

import hashlib
import json
import ssl
from dataclasses import replace
from pathlib import Path
from urllib.request import Request

import pytest

import fractal_ann_diagnostics.external_anchors as anchors
from fractal_ann_diagnostics.external_anchors import (
    ExternalAnchorError,
    PredictionCompletionAnchorReceipt,
    PredictionCompletionAnchorRecord,
    create_protocol_registration_receipt,
    create_protocol_registry_record,
    load_prediction_completion_anchor_receipt,
    load_prediction_completion_anchor_record,
    verify_prediction_completion_anchor,
    write_prediction_completion_anchor_receipt,
    write_prediction_completion_anchor_record,
    write_protocol_registration_receipt,
    write_protocol_registry_record,
)
from fractal_ann_diagnostics.label_separation import (
    ActionPanelBinding,
    PredictionCompletionReceipt,
    write_prediction_completion_receipt,
)
from fractal_ann_diagnostics.study import (
    load_protocol_registration_receipt,
    load_protocol_registry_record,
)

MANIFEST_SHA256 = "a" * 64
RUN_SHA256 = "b" * 64
EXECUTION_SHA256 = "c" * 64
PREDICTION_SHA256 = "d" * 64
PANEL_SHA256 = "e" * 64
ONLINE_RESULT_SHA256 = "f" * 64
ANCHOR_URI = "https://zenodo.org/api/records/123456/files/prediction-completion-anchor.json/content"


def _completion() -> PredictionCompletionReceipt:
    binding = ActionPanelBinding(
        manifest_sha256=MANIFEST_SHA256,
        run_receipt_sha256=RUN_SHA256,
        execution_artifact_sha256=EXECUTION_SHA256,
        corpus="scifact",
        stage="sealed",
        action_panel_artifact_sha256=PANEL_SHA256,
    )
    return PredictionCompletionReceipt(
        manifest_sha256=MANIFEST_SHA256,
        run_receipt_sha256=RUN_SHA256,
        execution_artifact_sha256=EXECUTION_SHA256,
        prediction_artifact_sha256=PREDICTION_SHA256,
        online_execution_result_receipt_sha256=ONLINE_RESULT_SHA256,
        action_panel_binding=binding,
        prediction_count=25,
        corpus="scifact",
        stage="sealed",
        external_anchor_identity="zenodo-record:123456",
        external_anchor_uri=ANCHOR_URI,
        anchored_at_utc="2026-07-14T12:00:00+00:00",
    )


def _anchor_files(
    tmp_path: Path,
    completion: PredictionCompletionReceipt | None = None,
) -> tuple[
    PredictionCompletionReceipt,
    PredictionCompletionAnchorRecord,
    PredictionCompletionAnchorReceipt,
    Path,
    Path,
]:
    completion = completion or _completion()
    record = PredictionCompletionAnchorRecord.from_completion_receipt(completion)
    receipt = PredictionCompletionAnchorReceipt.from_record(record)
    record_path = (tmp_path / "prediction-anchor-record.json").resolve()
    receipt_path = (tmp_path / "prediction-anchor-receipt.json").resolve()
    write_prediction_completion_anchor_record(record, record_path)
    write_prediction_completion_anchor_receipt(receipt, receipt_path)
    return completion, record, receipt, record_path, receipt_path


def test_protocol_writers_bind_one_frozen_manifest_and_write_exclusively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = (tmp_path / "study-manifest.json").resolve()
    manifest_payload = {"protocol_version": "0.3.0"}
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    observed: list[tuple[object, bool]] = []

    def validate(payload: object, *, require_frozen: bool = False) -> None:
        observed.append((payload, require_frozen))

    monkeypatch.setattr(anchors, "validate_study_manifest", validate)
    record = create_protocol_registry_record(
        manifest,
        registered_at_utc="2026-07-14T10:30:00+00:00",
        registry_identity="osf-registration:abc12;zenodo-record:123456",
        registry_uri=(
            "https://zenodo.org/api/records/123456/files/protocol-registry-record.json/content"
        ),
    )
    record_path = (tmp_path / "protocol-registry-record.json").resolve()
    write_protocol_registry_record(record, record_path)
    assert load_protocol_registry_record(record_path) == record
    assert record_path.read_bytes() == record.canonical_bytes() + b"\n"

    receipt = create_protocol_registration_receipt(manifest, record_path)
    receipt_path = (tmp_path / "protocol-registration-receipt.json").resolve()
    write_protocol_registration_receipt(receipt, receipt_path)
    assert load_protocol_registration_receipt(receipt_path) == receipt
    assert receipt.registry_record_sha256 == record.record_sha256
    assert observed == [(manifest_payload, True), (manifest_payload, True)]

    with pytest.raises(ExternalAnchorError, match="already exists"):
        write_protocol_registration_receipt(receipt, receipt_path)


def test_protocol_receipt_rejects_a_record_for_another_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = (tmp_path / "study-manifest.json").resolve()
    manifest.write_text('{"protocol_version":"0.3.0"}', encoding="utf-8")
    monkeypatch.setattr(
        anchors,
        "validate_study_manifest",
        lambda payload, *, require_frozen=False: None,
    )
    record = create_protocol_registry_record(
        manifest,
        registered_at_utc="2026-07-14T10:30:00+00:00",
        registry_identity="zenodo-record:123456",
        registry_uri=(
            "https://zenodo.org/api/records/123456/files/protocol-registry-record.json/content"
        ),
    )
    record = replace(record, manifest_sha256="f" * 64)
    record_path = (tmp_path / "substituted-record.json").resolve()
    write_protocol_registry_record(record, record_path)

    with pytest.raises(ExternalAnchorError, match="another frozen manifest"):
        create_protocol_registration_receipt(manifest, record_path)


def test_prediction_anchor_record_and_receipt_are_closed_canonical_custody_files(
    tmp_path: Path,
) -> None:
    completion, record, receipt, record_path, receipt_path = _anchor_files(tmp_path)
    assert record.prediction_completion_receipt_sha256 == completion.receipt_sha256
    assert record.online_execution_result_receipt_sha256 == ONLINE_RESULT_SHA256
    assert receipt.online_execution_result_receipt_sha256 == ONLINE_RESULT_SHA256
    assert record.record_sha256 == hashlib.sha256(record_path.read_bytes()).hexdigest()
    assert receipt.anchor_record_sha256 == record.record_sha256
    assert load_prediction_completion_anchor_record(record_path) == record
    assert load_prediction_completion_anchor_receipt(receipt_path) == receipt

    payload = record.to_dict()
    payload["unregistered"] = True
    substituted = (tmp_path / "substituted.json").resolve()
    substituted.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ExternalAnchorError, match="unknown"):
        load_prediction_completion_anchor_record(substituted)

    with pytest.raises(ExternalAnchorError, match="already exists"):
        write_prediction_completion_anchor_record(record, record_path)


def test_prediction_anchor_verification_refetches_and_compares_exact_bytes(
    tmp_path: Path,
) -> None:
    completion, record, receipt, record_path, receipt_path = _anchor_files(tmp_path)
    calls: list[tuple[str, int]] = []

    def fetch(uri: str, max_bytes: int) -> bytes:
        calls.append((uri, max_bytes))
        return record_path.read_bytes()

    verified = verify_prediction_completion_anchor(
        completion,
        anchor_record_path=record_path,
        anchor_receipt_path=receipt_path,
        trusted_anchor_record_fetcher=fetch,
    )

    assert verified.record == record
    assert verified.receipt == receipt
    assert calls == [(ANCHOR_URI, anchors.MAX_EXTERNAL_ANCHOR_RECORD_BYTES)]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("different-bytes", "digest does not match"),
        ("missing-newline", "digest does not match"),
        ("oversize", "maximum byte limit"),
        ("non-bytes", "must return bytes"),
        ("unavailable", "fetcher failed"),
    ),
)
def test_prediction_anchor_verification_fails_closed_on_remote_substitution(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    completion, _, _, record_path, receipt_path = _anchor_files(tmp_path)
    local = record_path.read_bytes()

    def fetch(uri: str, max_bytes: int) -> bytes:
        assert uri == ANCHOR_URI
        if mutation == "different-bytes":
            return local.replace(b"scifact", b"SCIFACT", 1)
        if mutation == "missing-newline":
            return local[:-1]
        if mutation == "oversize":
            return b"x" * (max_bytes + 1)
        if mutation == "non-bytes":
            return "not bytes"  # type: ignore[return-value]
        raise TimeoutError("anchor unavailable")

    with pytest.raises(ExternalAnchorError, match=message):
        verify_prediction_completion_anchor(
            completion,
            anchor_record_path=record_path,
            anchor_receipt_path=receipt_path,
            trusted_anchor_record_fetcher=fetch,
        )


def test_prediction_anchor_verification_rejects_local_receipt_rebinding(
    tmp_path: Path,
) -> None:
    completion, _, receipt, record_path, _ = _anchor_files(tmp_path)
    changed = replace(receipt, prediction_artifact_sha256="9" * 64)
    changed_path = (tmp_path / "changed-receipt.json").resolve()
    write_prediction_completion_anchor_receipt(changed, changed_path)

    with pytest.raises(
        ExternalAnchorError,
        match="receipt.prediction_artifact_sha256",
    ):
        verify_prediction_completion_anchor(
            completion,
            anchor_record_path=record_path,
            anchor_receipt_path=changed_path,
            trusted_anchor_record_fetcher=lambda uri, limit: record_path.read_bytes(),
        )


def test_builtin_anchor_fetch_uses_verified_https_and_one_bounded_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b'{"exact":true}\n'
    observed: dict[str, object] = {}

    class Response:
        headers = {"Content-Length": str(len(expected))}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def getcode(self) -> int:
            return 200

        def geturl(self) -> str:
            return ANCHOR_URI

        def read(self, limit: int) -> bytes:
            observed["read_limit"] = limit
            return expected

    class Opener:
        def open(self, request: Request, *, timeout: float) -> Response:
            observed["request"] = request
            observed["timeout"] = timeout
            return Response()

    def build_opener(*handlers: object) -> Opener:
        observed["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(anchors.urllib_request, "build_opener", build_opener)
    fetched = anchors._fetch_external_anchor_record(
        ANCHOR_URI,
        anchors.MAX_EXTERNAL_ANCHOR_RECORD_BYTES,
    )
    assert fetched == expected
    assert observed["read_limit"] == anchors.MAX_EXTERNAL_ANCHOR_RECORD_BYTES + 1
    request = observed["request"]
    assert isinstance(request, Request)
    assert request.full_url == ANCHOR_URI
    assert request.get_method() == "GET"
    assert request.get_header("Accept-encoding") == "identity"
    handlers = observed["handlers"]
    assert isinstance(handlers, tuple)
    assert any(isinstance(handler, anchors._NoExternalAnchorRedirects) for handler in handlers)
    https_handler = next(
        handler for handler in handlers if isinstance(handler, anchors.urllib_request.HTTPSHandler)
    )
    assert https_handler._context.check_hostname is True
    assert https_handler._context.verify_mode == ssl.CERT_REQUIRED


def test_builtin_anchor_fetch_refuses_redirects() -> None:
    handler = anchors._NoExternalAnchorRedirects()
    with pytest.raises(ExternalAnchorError, match="redirect status 302"):
        handler.redirect_request(
            Request(ANCHOR_URI),
            None,
            302,
            "Found",
            {},
            "https://substitute.example.test/record.json",
        )


def test_standalone_cli_writes_prediction_anchor_pair(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    completion = _completion()
    completion_path = (tmp_path / "completion.json").resolve()
    record_path = (tmp_path / "anchor-record.json").resolve()
    receipt_path = (tmp_path / "anchor-receipt.json").resolve()
    write_prediction_completion_receipt(completion, completion_path)

    assert (
        anchors.main(
            [
                "write-prediction-completion-anchor-record",
                "--completion-receipt",
                str(completion_path),
                "--output",
                str(record_path),
            ]
        )
        == 0
    )
    assert (
        anchors.main(
            [
                "write-prediction-completion-anchor-receipt",
                "--anchor-record",
                str(record_path),
                "--output",
                str(receipt_path),
            ]
        )
        == 0
    )
    assert load_prediction_completion_anchor_record(record_path)
    assert load_prediction_completion_anchor_receipt(receipt_path)
    assert "prediction anchor record sha256" in capsys.readouterr().out
