from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.timelock_release as release_module
from fractal_ann_diagnostics.custody import (
    CustodyCorpusCommitment,
    CustodySealReceipt,
    TimelockEncryptionReceipt,
)
from fractal_ann_diagnostics.execution_claim import (
    LiveExecuteJobReceipt,
    PhaseBeaconReceipt,
)
from fractal_ann_diagnostics.external_anchors import (
    PredictionCompletionAnchorReceipt,
    PredictionCompletionAnchorRecord,
    VerifiedPredictionCompletionAnchor,
)
from fractal_ann_diagnostics.label_separation import (
    ActionPanelBinding,
    PredictionCompletionReceipt,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA, manifest_sha256
from fractal_ann_diagnostics.timelock_release import (
    TIMELOCK_DECRYPTION_RECEIPT_FILENAME,
    TIMELOCK_RELEASE_INTENT_FILENAME,
    TimelockReleaseError,
    VerifiedTimelockRelease,
    label_release_staging_directory_name,
    load_timelock_decryption_receipt,
    release_timelock_label,
    write_timelock_decryption_receipt,
)

_CHAIN = "a" * 64
_NETWORK = "https://api.drand.sh"
_ROUND = 1
_GENESIS = datetime(2026, 7, 14, 13, 0, tzinfo=timezone.utc)
_PUBLIC_KEY = "12" * 48
_SIGNATURE = "34" * 48
_RANDOMNESS = hashlib.sha256(bytes.fromhex(_SIGNATURE)).hexdigest()


def _digest(value: bytes | str) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _artifact(
    role: str,
    sha256: str,
    *,
    corpus_id: str | None = None,
) -> dict[str, str]:
    row = {"role": role, "sha256": sha256}
    if corpus_id is not None:
        row["corpus_id"] = corpus_id
    return row


def _anchor(manifest_digest: str, *, anchored_at: str) -> VerifiedPredictionCompletionAnchor:
    binding = ActionPanelBinding(
        manifest_sha256=manifest_digest,
        run_receipt_sha256="1" * 64,
        execution_artifact_sha256="2" * 64,
        corpus=FIXED_CORPORA[0],
        stage="sealed",
        action_panel_artifact_sha256="3" * 64,
    )
    completion = PredictionCompletionReceipt(
        manifest_sha256=manifest_digest,
        run_receipt_sha256="1" * 64,
        execution_artifact_sha256="2" * 64,
        prediction_artifact_sha256="4" * 64,
        online_execution_result_receipt_sha256="5" * 64,
        action_panel_binding=binding,
        prediction_count=25,
        corpus=FIXED_CORPORA[0],
        stage="sealed",
        external_anchor_identity="zenodo-record:123456",
        external_anchor_uri=("https://zenodo.org/records/123456/files/completion.json/content"),
        anchored_at_utc=anchored_at,
    )
    record = PredictionCompletionAnchorRecord.from_completion_receipt(completion)
    receipt = PredictionCompletionAnchorReceipt.from_record(record)
    return VerifiedPredictionCompletionAnchor(record=record, receipt=receipt)


def _drand_responses() -> tuple[bytes, bytes]:
    metadata = json.dumps(
        {
            "genesis_time": int(_GENESIS.timestamp()),
            "hash": _CHAIN,
            "period": 3,
            "public_key": _PUBLIC_KEY,
            "schemeID": "bls-unchained-g1-rfc9380",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    beacon = json.dumps(
        {
            "randomness": _RANDOMNESS,
            "round": _ROUND,
            "signature": _SIGNATURE,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return metadata, beacon


def _release_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    plaintext = b'{"labels":["sealed"]}\n'
    ciphertext = b"tlock-v1:" + plaintext[::-1]
    binary = (tmp_path / "tle").resolve()
    binary.write_bytes(b"#!/bin/sh\nexit 99\n")
    binary.chmod(0o700)
    encryption = TimelockEncryptionReceipt(
        corpus_id=FIXED_CORPORA[0],
        plaintext_sha256=_digest(plaintext),
        plaintext_byte_count=len(plaintext),
        ciphertext_sha256=_digest(ciphertext),
        ciphertext_byte_count=len(ciphertext),
        tle_binary_sha256=_digest(binary.read_bytes()),
        drand_network=_NETWORK,
        drand_chain_hash=_CHAIN,
        drand_round=_ROUND,
        tle_arguments=(
            "--encrypt",
            f"--network={_NETWORK}",
            f"--chain={_CHAIN}",
            f"--round={_ROUND}",
        ),
    )
    commitments = []
    for position, corpus_id in enumerate(FIXED_CORPORA):
        receipt_digest = (
            encryption.file_sha256 if position == 0 else _digest(f"receipt-{corpus_id}")
        )
        commitments.append(
            CustodyCorpusCommitment(
                corpus_id=corpus_id,
                online_execution_sha256=_digest(f"online-{corpus_id}"),
                sealed_label_plaintext_sha256=(
                    encryption.plaintext_sha256
                    if position == 0
                    else _digest(f"plaintext-{corpus_id}")
                ),
                sealed_label_ciphertext_sha256=(
                    encryption.ciphertext_sha256
                    if position == 0
                    else _digest(f"ciphertext-{corpus_id}")
                ),
                timelock_encryption_receipt_file_sha256=receipt_digest,
            )
        )
    seal = CustodySealReceipt(
        protocol_version="0.3.0",
        drand_chain_hash=_CHAIN,
        drand_round=_ROUND,
        timelock_tool_sha256=encryption.tle_binary_sha256,
        custody_builder_sha256=_digest("builder"),
        commitments=tuple(commitments),
    )
    manifest: dict[str, object] = {
        "protocol_version": "0.3.0",
        "status": "frozen",
        "artifacts": [
            _artifact("sealed-labels", encryption.plaintext_sha256, corpus_id=FIXED_CORPORA[0]),
            _artifact(
                "sealed-label-ciphertext",
                encryption.ciphertext_sha256,
                corpus_id=FIXED_CORPORA[0],
            ),
            _artifact(
                "timelock-encryption-receipt",
                encryption.file_sha256,
                corpus_id=FIXED_CORPORA[0],
            ),
            _artifact("timelock-tool", encryption.tle_binary_sha256),
        ],
    }
    ciphertext_path = (tmp_path / "labels.tlock").resolve()
    ciphertext_path.write_bytes(ciphertext)
    release_root = (tmp_path / "release").resolve()
    release_root.mkdir(mode=0o700)
    output = (release_root / FIXED_CORPORA[0] / "labels.json").resolve()
    receipt_output = (
        release_root / FIXED_CORPORA[0] / TIMELOCK_DECRYPTION_RECEIPT_FILENAME
    ).resolve()
    anchor = _anchor(
        manifest_sha256(manifest),
        anchored_at="2026-07-14T12:59:59+00:00",
    )
    metadata, beacon = _drand_responses()

    monkeypatch.setattr(release_module, "validate_study_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(release_module, "verify_custody_seal_receipt", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        release_module,
        "verify_timelock_encryption_receipt",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        release_module,
        "_require_suite_online_completion",
        lambda *args, **kwargs: None,
    )

    def fetcher(uri: str, max_bytes: int) -> bytes:
        assert max_bytes == 64 * 1024
        if uri.endswith("/info"):
            return metadata
        assert uri.endswith(f"/public/{_ROUND}")
        return beacon

    phase_beacon = PhaseBeaconReceipt(
        phase="label-release",
        phase_claim_state_sha256=_digest("label-release-claim"),
        phase_claim_ledger_commit="a" * 40,
        provider_identity_sha256=_digest("provider"),
        phase_claim_contract_sha256=_digest("phase-contract"),
        beacon_contract_sha256=_digest("beacon-contract"),
        beacon_bytes_sha256=_digest(beacon),
        chain_hash=_CHAIN,
        round=_ROUND,
        randomness=_RANDOMNESS,
        signature=_SIGNATURE,
        published_at_utc=_GENESIS.isoformat(),
        verified_at_utc=_GENESIS.isoformat(),
    )
    live_job = LiveExecuteJobReceipt(
        provider_identity_sha256=_digest("provider"),
        repository="owner/repository",
        workflow_path=".github/workflows/provider.yml",
        workflow_sha="b" * 40,
        run_head_branch=None,
        run_id=101,
        run_attempt=1,
        execute_job_id=202,
        execute_job_name="execute",
        runner_id=303,
        runner_name="label-release-runner",
        runner_group_id=None,
        runner_labels=("label-release", "self-hosted"),
        verified_at_utc=_GENESIS.isoformat(),
    )
    phase_token = SimpleNamespace(
        phase_claim_state_sha256=phase_beacon.phase_claim_state_sha256,
        phase_claim_ledger_commit=phase_beacon.phase_claim_ledger_commit,
        contract=SimpleNamespace(
            contract_sha256=phase_beacon.phase_claim_contract_sha256,
            run_receipt_sha256="1" * 64,
            corpora=(
                SimpleNamespace(
                    corpus_id=FIXED_CORPORA[0],
                    output_uri=output.as_uri(),
                ),
            ),
        ),
        live_execute_job_receipt=live_job,
        provider_identity=SimpleNamespace(identity_sha256=_digest("provider")),
    )
    selected_anchor = [anchor]
    completion_aggregate = SimpleNamespace(
        file_sha256=_digest("post-online-completion-aggregate"),
    )
    monkeypatch.setattr(
        release_module,
        "_require_phase_release_authority",
        lambda *args, **kwargs: phase_beacon,
    )
    monkeypatch.setattr(
        release_module,
        "_require_post_online_completion",
        lambda *args, **kwargs: (selected_anchor[0], completion_aggregate),
    )

    return {
        "anchor": anchor,
        "binary": binary,
        "ciphertext": ciphertext,
        "ciphertext_path": ciphertext_path,
        "encryption": encryption,
        "fetcher": fetcher,
        "manifest": manifest,
        "live_job": live_job,
        "output": output,
        "phase_beacon": phase_beacon,
        "phase_token": phase_token,
        "plaintext": plaintext,
        "post_online_token": object(),
        "receipt_output": receipt_output,
        "seal": seal,
        "selected_anchor": selected_anchor,
        "suite_token": object(),
    }


def _release(
    fixture: dict[str, object],
    *,
    anchor: VerifiedPredictionCompletionAnchor | None = None,
    fetcher: object | None = None,
    output: Path | None = None,
    receipt_output: Path | None = None,
    runner: object | None = None,
) -> VerifiedTimelockRelease:
    plaintext = fixture["plaintext"]
    if anchor is not None:
        fixture["selected_anchor"][0] = anchor  # type: ignore[index]

    def default_runner(
        binary: Path,
        arguments: tuple[str, ...],
        ciphertext: bytes,
        timeout_seconds: int,
        max_plaintext_bytes: int,
    ) -> bytes:
        assert binary == fixture["binary"]
        assert arguments == (
            "--decrypt",
            f"--network={_NETWORK}",
            f"--chain={_CHAIN}",
        )
        assert ciphertext == fixture["ciphertext"]
        assert timeout_seconds == 60
        assert max_plaintext_bytes == 64 * 1024 * 1024
        assert isinstance(plaintext, bytes)
        return plaintext

    return release_timelock_label(
        fixture["manifest"],  # type: ignore[arg-type]
        corpus_id=FIXED_CORPORA[0],
        custody_seal=fixture["seal"],  # type: ignore[arg-type]
        encryption_receipt=fixture["encryption"],  # type: ignore[arg-type]
        verified_post_online_completion=fixture["post_online_token"],
        verified_suite_completion=fixture["suite_token"],
        verified_phase_claim=fixture["phase_token"],
        ciphertext_path=fixture["ciphertext_path"],  # type: ignore[arg-type]
        tle_binary_path=fixture["binary"],  # type: ignore[arg-type]
        plaintext_output_path=(
            fixture["output"] if output is None else output  # type: ignore[arg-type]
        ),
        decryption_receipt_output_path=(
            fixture["receipt_output"]  # type: ignore[arg-type]
            if receipt_output is None
            else receipt_output
        ),
        trusted_drand_fetcher=(
            fixture["fetcher"] if fetcher is None else fetcher  # type: ignore[arg-type]
        ),
        trusted_tle_runner=(
            default_runner if runner is None else runner  # type: ignore[arg-type]
        ),
        utc_now_factory=lambda: datetime(2026, 7, 14, 13, 0, 1, tzinfo=timezone.utc),
    )


def _release_stage(fixture: dict[str, object]) -> Path:
    output = fixture["output"]
    assert isinstance(output, Path)
    return output.parent.parent / label_release_staging_directory_name(output.parent.name)


def test_release_fsyncs_intent_file_stage_and_parent_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    stage = _release_stage(fixture)
    output = fixture["output"]
    assert isinstance(output, Path)
    fsynced: set[tuple[int, int]] = set()
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        fsynced.add((metadata.st_dev, metadata.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(release_module.os, "fsync", recording_fsync)

    def runner(*args: object) -> bytes:
        del args
        intent = stage / TIMELOCK_RELEASE_INTENT_FILENAME
        assert stage.is_dir()
        assert intent.is_file()
        assert not os.path.lexists(output.parent)
        for path in (intent, stage, stage.parent):
            metadata = path.stat()
            assert (metadata.st_dev, metadata.st_ino) in fsynced
        assert (stage.stat().st_mode & 0o777) == 0o700
        assert (intent.stat().st_mode & 0o777) == 0o600
        return fixture["plaintext"]  # type: ignore[return-value]

    _release(fixture, runner=runner)


def test_runner_failure_leaves_ambiguous_intent_and_partial_retry_never_runs_tle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    stage = _release_stage(fixture)
    runner_calls = 0

    def failing_runner(*args: object) -> bytes:
        del args
        nonlocal runner_calls
        runner_calls += 1
        raise RuntimeError("synthetic runner crash")

    with pytest.raises(TimelockReleaseError, match="trusted tle runner failed"):
        _release(fixture, runner=failing_runner)
    assert runner_calls == 1
    assert {path.name for path in stage.iterdir()} == {TIMELOCK_RELEASE_INTENT_FILENAME}

    retry_calls: list[str] = []

    def forbidden(*args: object) -> bytes:
        del args
        retry_calls.append("called")
        raise AssertionError("retry must not fetch or decrypt")

    with pytest.raises(TimelockReleaseError, match="ambiguous incomplete"):
        _release(fixture, fetcher=forbidden, runner=forbidden)
    assert retry_calls == []

    output = fixture["output"]
    assert isinstance(output, Path)
    (stage / output.name).write_bytes(b"partial plaintext")
    with pytest.raises(TimelockReleaseError, match="ambiguous incomplete"):
        _release(fixture, fetcher=forbidden, runner=forbidden)
    assert retry_calls == []


@pytest.mark.parametrize("transaction_location", ("complete-stage", "final-with-intent"))
def test_complete_transaction_recovers_without_fetch_or_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transaction_location: str,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    original = _release(fixture)
    output = fixture["output"]
    assert isinstance(output, Path)
    final = output.parent
    stage = _release_stage(fixture)
    if transaction_location == "complete-stage":
        final.rename(stage)

    monkeypatch.setattr(
        release_module,
        "_require_existing_release_authority",
        lambda *args, **kwargs: None,
    )
    calls: list[str] = []

    def forbidden(*args: object) -> bytes:
        del args
        calls.append("called")
        raise AssertionError("committed transaction recovery must not fetch or decrypt")

    recovered = _release(fixture, fetcher=forbidden, runner=forbidden)
    assert calls == []
    assert recovered.receipt == original.receipt
    assert not os.path.lexists(stage)
    assert {path.name for path in final.iterdir()} == {
        output.name,
        TIMELOCK_DECRYPTION_RECEIPT_FILENAME,
        TIMELOCK_RELEASE_INTENT_FILENAME,
    }


def test_tampered_committed_intent_fails_before_fetch_or_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    _release(fixture)
    output = fixture["output"]
    assert isinstance(output, Path)
    intent = output.parent / TIMELOCK_RELEASE_INTENT_FILENAME
    payload = json.loads(intent.read_bytes())
    payload["ciphertext_sha256"] = "f" * 64
    intent.write_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    monkeypatch.setattr(
        release_module,
        "_require_existing_release_authority",
        lambda *args, **kwargs: None,
    )
    calls: list[str] = []

    def forbidden(*args: object) -> bytes:
        del args
        calls.append("called")
        raise AssertionError("tampered transaction must not fetch or decrypt")

    with pytest.raises(TimelockReleaseError, match="intent differs"):
        _release(fixture, fetcher=forbidden, runner=forbidden)
    assert calls == []


def test_alternate_output_uri_stops_before_fetch_or_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    alternate_root = (tmp_path / "alternate-release").resolve()
    calls: list[str] = []

    def forbidden(*args: object) -> bytes:
        del args
        calls.append("called")
        raise AssertionError("unclaimed output must not fetch or decrypt")

    with pytest.raises(TimelockReleaseError, match="claimed corpus binding"):
        _release(
            fixture,
            output=alternate_root / "labels.json",
            receipt_output=alternate_root / TIMELOCK_DECRYPTION_RECEIPT_FILENAME,
            fetcher=forbidden,
            runner=forbidden,
        )
    assert calls == []
    assert not os.path.lexists(alternate_root)


def test_action_receipt_hashes_remain_exact_while_stable_identities_allow_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    original = _release(fixture)
    output = fixture["output"]
    live = fixture["live_job"]
    beacon = fixture["phase_beacon"]
    phase_token = fixture["phase_token"]
    assert isinstance(output, Path)
    assert isinstance(live, LiveExecuteJobReceipt)
    assert isinstance(beacon, PhaseBeaconReceipt)

    intent_path = output.parent / TIMELOCK_RELEASE_INTENT_FILENAME
    intent = json.loads(intent_path.read_bytes())
    assert original.receipt.label_release_live_execute_job_receipt_sha256 == (live.receipt_sha256)
    assert original.receipt.label_release_phase_beacon_receipt_sha256 == (beacon.receipt_sha256)
    assert intent["label_release_live_execute_job_receipt"] == live.to_dict()
    assert intent["label_release_phase_beacon_receipt"] == beacon.to_dict()
    assert intent["label_release_live_execute_job_identity_sha256"] == (live.job_identity_sha256)
    assert intent["label_release_phase_beacon_identity_sha256"] == (beacon.beacon_identity_sha256)

    refreshed_live = replace(
        live,
        verified_at_utc="2026-07-14T13:00:01+00:00",
    )
    refreshed_beacon = replace(
        beacon,
        verified_at_utc="2026-07-14T13:00:01+00:00",
    )
    assert refreshed_live.receipt_sha256 != live.receipt_sha256
    assert refreshed_beacon.receipt_sha256 != beacon.receipt_sha256
    assert refreshed_live.job_identity_sha256 == live.job_identity_sha256
    assert refreshed_beacon.beacon_identity_sha256 == beacon.beacon_identity_sha256
    phase_token.live_execute_job_receipt = refreshed_live
    monkeypatch.setattr(
        release_module,
        "_require_phase_release_authority",
        lambda *args, **kwargs: refreshed_beacon,
    )
    monkeypatch.setattr(
        release_module,
        "_require_existing_release_authority",
        lambda *args, **kwargs: None,
    )

    calls: list[str] = []

    def forbidden(*args: object) -> bytes:
        del args
        calls.append("called")
        raise AssertionError("stable-identity restart must not fetch or decrypt")

    recovered = _release(fixture, fetcher=forbidden, runner=forbidden)
    assert calls == []
    assert recovered.receipt == original.receipt
    assert recovered.receipt.label_release_live_execute_job_receipt_sha256 == (live.receipt_sha256)
    assert recovered.receipt.label_release_phase_beacon_receipt_sha256 == (beacon.receipt_sha256)


def test_suite_completion_gate_rejects_bare_in_memory_object() -> None:
    with pytest.raises(TimelockReleaseError, match="canonical files"):
        release_module._require_suite_online_completion(
            object(),
            manifest_digest="a" * 64,
            corpus_id=FIXED_CORPORA[0],
            online_result_receipt_sha256="b" * 64,
        )


def test_release_is_exact_canonical_exclusive_and_revalidates_plaintext(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    verified = _release(fixture)
    assert verified.read_plaintext() == fixture["plaintext"]
    assert fixture["output"].read_bytes() == fixture["plaintext"]  # type: ignore[union-attr]
    assert (
        load_timelock_decryption_receipt(
            fixture["receipt_output"]  # type: ignore[arg-type]
        )
        == verified.receipt
    )
    assert {path.name for path in fixture["output"].parent.iterdir()} == {  # type: ignore[union-attr]
        "labels.json",
        "timelock-decryption-receipt.json",
        TIMELOCK_RELEASE_INTENT_FILENAME,
    }
    assert verified.receipt.tle_arguments == (
        "--decrypt",
        f"--network={_NETWORK}",
        f"--chain={_CHAIN}",
    )
    assert verified.receipt.verified_beacon_round == _ROUND
    assert verified.receipt.beacon_publication_time_utc == _GENESIS.isoformat()
    assert verified.receipt.prediction_completion_anchor_record_sha256 == (
        fixture["anchor"].record.record_sha256  # type: ignore[union-attr]
    )
    assert verified.receipt.online_execution_result_receipt_sha256 == "5" * 64
    receipt_path = (tmp_path / "decryption.json").resolve()
    write_timelock_decryption_receipt(verified.receipt, receipt_path)
    assert load_timelock_decryption_receipt(receipt_path) == verified.receipt
    with pytest.raises(TimelockReleaseError, match="only be created"):
        VerifiedTimelockRelease(
            receipt=verified.receipt,
            plaintext_path=fixture["output"],  # type: ignore[arg-type]
            _capability=object(),
        )


def test_release_refuses_existing_plaintext_before_fetch_or_decrypt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    fixture["output"].parent.mkdir(mode=0o700)  # type: ignore[union-attr]
    fixture["output"].write_bytes(b"pre-existing")  # type: ignore[union-attr]

    def forbidden(*args: object) -> bytes:
        raise AssertionError("fetcher or runner must not be called")

    with pytest.raises(TimelockReleaseError, match="recoverable transaction"):
        _release(fixture, fetcher=forbidden, runner=forbidden)


def test_release_receipt_failure_cannot_publish_plaintext_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        release_module,
        "write_timelock_decryption_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            TimelockReleaseError("synthetic receipt failure")
        ),
    )

    with pytest.raises(TimelockReleaseError, match="synthetic receipt failure"):
        _release(fixture)
    assert not os.path.lexists(fixture["output"].parent)  # type: ignore[union-attr]


def test_release_requires_anchor_strictly_before_authenticated_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    late_anchor = _anchor(
        manifest_sha256(fixture["manifest"]),  # type: ignore[arg-type]
        anchored_at=_GENESIS.isoformat(),
    )
    with pytest.raises(TimelockReleaseError, match="strictly predate"):
        _release(fixture, anchor=late_anchor)
    assert not os.path.lexists(fixture["output"])  # type: ignore[arg-type]


def test_release_rejects_wrong_beacon_round_and_plaintext(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    metadata, beacon = _drand_responses()
    wrong = json.loads(beacon)
    wrong["round"] = 2
    wrong_beacon = json.dumps(wrong, separators=(",", ":"), sort_keys=True).encode()

    def wrong_fetcher(uri: str, max_bytes: int) -> bytes:
        del max_bytes
        return metadata if uri.endswith("/info") else wrong_beacon

    with pytest.raises(TimelockReleaseError, match="another round"):
        _release(fixture, fetcher=wrong_fetcher)
    assert not os.path.lexists(fixture["output"])  # type: ignore[arg-type]

    def wrong_runner(*args: object) -> bytes:
        return b"wrong plaintext"

    with pytest.raises(TimelockReleaseError, match="frozen label bytes"):
        _release(fixture, runner=wrong_runner)
    assert not os.path.lexists(fixture["output"])  # type: ignore[arg-type]


def test_release_binds_fetched_beacon_to_bls_verified_phase_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    signature = "56" * 48
    mismatched = replace(
        fixture["phase_beacon"],
        signature=signature,
        randomness=hashlib.sha256(bytes.fromhex(signature)).hexdigest(),
    )
    monkeypatch.setattr(
        release_module,
        "_require_phase_release_authority",
        lambda *args, **kwargs: mismatched,
    )

    with pytest.raises(TimelockReleaseError, match="BLS-verified"):
        _release(fixture)
    assert not os.path.lexists(fixture["output"])  # type: ignore[arg-type]


def test_tle_failure_never_surfaces_stderr_label_bytes() -> None:
    secret = "sealed-label-value-must-not-escape"
    with pytest.raises(TimelockReleaseError) as caught:
        release_module._run_pinned_tle_decrypt(
            Path(sys.executable),
            (
                "-c",
                (
                    "import sys;"
                    f"sys.stderr.write({secret!r});"
                    "sys.stderr.flush();"
                    "raise SystemExit(17)"
                ),
            ),
            b"ciphertext",
            10,
            1024,
        )
    assert secret not in str(caught.value)
    assert "stderr suppressed" in str(caught.value)


def test_release_revalidates_suite_after_decryption_before_plaintext_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    events: list[str] = []
    phase_checks = 0

    def phase_gate(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal phase_checks
        phase_checks += 1
        events.append(f"phase-{phase_checks}")
        if phase_checks == 3:
            raise TimelockReleaseError("provider phase capability is no longer current")
        return fixture["phase_beacon"]

    def runner(*args: object) -> bytes:
        del args
        events.append("runner")
        return fixture["plaintext"]  # type: ignore[return-value]

    monkeypatch.setattr(
        release_module,
        "_require_phase_release_authority",
        phase_gate,
    )
    with pytest.raises(TimelockReleaseError, match="no longer current"):
        _release(fixture, runner=runner)
    assert events == ["phase-1", "phase-2", "runner", "phase-3"]
    assert not os.path.lexists(fixture["output"])  # type: ignore[arg-type]


def test_release_refuses_stale_phase_before_decrypting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    decryptions: list[bool] = []
    monkeypatch.setattr(
        release_module,
        "_require_phase_release_authority",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            TimelockReleaseError("provider phase capability is no longer current")
        ),
    )

    with pytest.raises(TimelockReleaseError, match="no longer current"):
        _release(
            fixture,
            runner=lambda *args: decryptions.append(True) or b"plaintext",
        )
    assert decryptions == []
    assert not os.path.lexists(fixture["output"])  # type: ignore[arg-type]


def test_release_rejects_round_disagreement_before_network_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    seal = fixture["seal"]
    fixture["seal"] = CustodySealReceipt(
        protocol_version="0.3.0",
        drand_chain_hash=seal.drand_chain_hash,  # type: ignore[union-attr]
        drand_round=2,
        timelock_tool_sha256=seal.timelock_tool_sha256,  # type: ignore[union-attr]
        custody_builder_sha256=seal.custody_builder_sha256,  # type: ignore[union-attr]
        commitments=seal.commitments,  # type: ignore[union-attr]
    )

    def forbidden(*args: object) -> bytes:
        raise AssertionError("network must not be reached")

    with pytest.raises(TimelockReleaseError, match="different rounds"):
        _release(fixture, fetcher=forbidden)


def test_release_rejects_online_result_digest_disagreement_between_anchor_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, monkeypatch)
    anchor = fixture["anchor"]
    mismatched = VerifiedPredictionCompletionAnchor(
        record=anchor.record,  # type: ignore[union-attr]
        receipt=replace(  # type: ignore[union-attr]
            anchor.receipt,
            online_execution_result_receipt_sha256="f" * 64,
        ),
    )
    with pytest.raises(
        TimelockReleaseError,
        match="mismatched online_execution_result_receipt_sha256",
    ):
        _release(fixture, anchor=mismatched)
