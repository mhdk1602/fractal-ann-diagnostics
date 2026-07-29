from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_execution_claim import (
    _digest as _claim_digest,
)
from test_execution_claim import (
    _live_job,
    _phase_contract,
    _phase_provider,
    _Verifier,
)

import fractal_ann_diagnostics.provider_workflow_validation as validation
from fractal_ann_diagnostics.execution_claim import verify_label_release_beacon
from fractal_ann_diagnostics.provider_phase_runtime import (
    LabelReleaseOutputAuthority,
    ProviderDriverOutput,
    ProviderPhaseExecutionReceipt,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA


@dataclass(frozen=True)
class _Authority:
    authority_sha256: str


@dataclass(frozen=True)
class _Output:
    corpus_id: str
    label_release_authority: _Authority | None
    label_release_authority_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "label_release_authority_sha256": self.label_release_authority_sha256,
        }


def _receipt(
    *,
    phase: str = "label-release",
    suite_attempt_id: str = "a" * 64,
    corpus_ids: tuple[str, ...] | None = None,
    bad_authority: bool = False,
) -> SimpleNamespace:
    rows: list[_Output] = []
    for position, corpus_id in enumerate(
        corpus_ids or tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
    ):
        authority = _Authority(f"{position + 1:064x}")
        rows.append(
            _Output(
                corpus_id=corpus_id,
                label_release_authority=authority,
                label_release_authority_sha256=(
                    "f" * 64 if bad_authority and position == 0 else authority.authority_sha256
                ),
            )
        )
    return SimpleNamespace(
        phase=phase,
        suite_attempt_id=suite_attempt_id,
        outputs=tuple(rows),
    )


def _write_inventory(
    path: Path,
    receipt: SimpleNamespace | ProviderPhaseExecutionReceipt,
) -> None:
    path.write_text(
        json.dumps(
            {
                "outputs": [row.to_dict() for row in receipt.outputs],
                "phase": receipt.phase,
                "schema_version": "fractal-provider-activation-inventory-v1",
                "suite_attempt_id": receipt.suite_attempt_id,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _typed_label_release_receipt(tmp_path: Path) -> ProviderPhaseExecutionReceipt:
    contract = _phase_contract("label-release")
    provider = _phase_provider(contract)
    live_job = _live_job(contract, provider)
    claim_state_sha256 = _claim_digest("label-claim-state")
    claim_ledger_commit = "6" * 40
    beacon = verify_label_release_beacon(
        contract,
        beacon_bytes=b'{"round":101,"randomness":"synthetic"}',
        phase_claim_state_sha256=claim_state_sha256,
        phase_claim_ledger_commit=claim_ledger_commit,
        provider_identity=provider,
        claim_attested_at_utc="2023-11-14T22:18:19+00:00",
        live_execute_job_receipt=live_job,
        verifier=_Verifier(),
        verified_at_utc="2023-11-14T22:18:21+00:00",
    )
    outputs: list[ProviderDriverOutput] = []
    for corpus_id in sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")):
        authority = LabelReleaseOutputAuthority(
            corpus_id=corpus_id,
            post_online_completion_aggregate_file_sha256=_claim_digest("post-online-completion"),
            label_release_claim_state_sha256=claim_state_sha256,
            label_release_claim_ledger_commit=claim_ledger_commit,
            label_release_phase_claim_contract_sha256=contract.contract_sha256,
            label_release_phase_beacon_receipt_sha256=beacon.receipt_sha256,
            label_release_live_execute_job_receipt_sha256=live_job.receipt_sha256,
            label_release_provider_identity_sha256=provider.identity_sha256,
            label_release_phase_beacon_receipt=beacon,
            label_release_live_execute_job_receipt=live_job,
        )
        outputs.append(
            ProviderDriverOutput(
                corpus_id=corpus_id,
                driver_id="timelock-label-release-v1",
                output_root=str((tmp_path / "outputs" / corpus_id).resolve()),
                output_tree_sha256=_claim_digest(f"output-tree:{corpus_id}"),
                output_entries=("released-labels.json",),
                label_release_authority_sha256=authority.authority_sha256,
                label_release_authority=authority,
            )
        )
    return ProviderPhaseExecutionReceipt(
        phase="label-release",
        suite_attempt_id=_claim_digest("suite-attempt"),
        provider_plan_sha256=_claim_digest("provider-plan"),
        provider_plan_file_sha256=_claim_digest("provider-plan-file"),
        claim_receipt_file_sha256=_claim_digest("claim-receipt-file"),
        phase_host_tool_receipt_path=str(
            (tmp_path / "activation" / "phase-host-tool-receipt.json").resolve()
        ),
        phase_host_tool_receipt_sha256=_claim_digest("phase-host-tool-receipt"),
        phase_host_tool_receipt_file_sha256=_claim_digest("phase-host-tool-receipt-file"),
        runtime_request_sha256=_claim_digest("runtime-request"),
        runtime_request_file_sha256=_claim_digest("runtime-request-file"),
        outputs=tuple(outputs),
    )


def test_label_release_inventory_accepts_real_canonical_receipt_through_cli(
    tmp_path: Path,
) -> None:
    receipt = _typed_label_release_receipt(tmp_path)
    inventory = tmp_path / "inventory.json"
    phase_receipt = tmp_path / "phase-receipt.json"
    _write_inventory(inventory, receipt)
    phase_receipt.write_bytes(receipt.canonical_file_bytes())

    assert ProviderPhaseExecutionReceipt.from_bytes(phase_receipt.read_bytes()) == receipt
    assert (
        validation.main(
            [
                "label-release-inventory",
                "--inventory",
                str(inventory),
                "--phase-execution-receipt",
                str(phase_receipt),
                "--suite-attempt-id",
                receipt.suite_attempt_id,
            ]
        )
        == 0
    )


def test_label_release_inventory_accepts_exact_receipt_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt()
    inventory = tmp_path / "inventory.json"
    phase_receipt = tmp_path / "phase-receipt.json"
    _write_inventory(inventory, receipt)
    phase_receipt.write_bytes(b"typed receipt bytes\n")
    monkeypatch.setattr(
        validation.ProviderPhaseExecutionReceipt,
        "from_bytes",
        staticmethod(lambda encoded: receipt),
    )

    validation.validate_label_release_inventory(
        inventory,
        phase_receipt,
        receipt.suite_attempt_id,
    )


@pytest.mark.parametrize(
    ("receipt", "suite_attempt_id"),
    (
        (_receipt(phase="online"), "a" * 64),
        (_receipt(suite_attempt_id="b" * 64), "a" * 64),
        (_receipt(corpus_ids=(*tuple(FIXED_CORPORA)[:-1], "intruder")), "a" * 64),
        (_receipt(bad_authority=True), "a" * 64),
    ),
)
def test_label_release_inventory_rejects_wrong_phase_suite_corpus_or_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: SimpleNamespace,
    suite_attempt_id: str,
) -> None:
    inventory = tmp_path / "inventory.json"
    phase_receipt = tmp_path / "phase-receipt.json"
    _write_inventory(inventory, receipt)
    phase_receipt.write_bytes(b"typed receipt bytes\n")
    monkeypatch.setattr(
        validation.ProviderPhaseExecutionReceipt,
        "from_bytes",
        staticmethod(lambda encoded: receipt),
    )

    with pytest.raises(
        validation.ProviderWorkflowValidationError,
        match="authority admission failed",
    ):
        validation.validate_label_release_inventory(
            inventory,
            phase_receipt,
            suite_attempt_id,
        )


def test_label_release_inventory_rejects_malformed_bytes(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.json"
    phase_receipt = tmp_path / "phase-receipt.json"
    inventory.write_bytes(b"{not-json")
    phase_receipt.write_bytes(b"not-a-receipt")

    with pytest.raises(
        validation.ProviderWorkflowValidationError,
        match="cannot load",
    ):
        validation.validate_label_release_inventory(
            inventory,
            phase_receipt,
            "a" * 64,
        )


def test_analysis_command_delegates_both_registered_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "analysis-execution-receipt.json"
    receipt.write_bytes(b"receipt\n")
    observed: dict[str, object] = {}

    def fake_loader(
        path: Path,
        *,
        expected_receipt_sha256: str,
        expected_file_sha256: str,
    ) -> object:
        observed.update(
            {
                "path": path,
                "receipt": expected_receipt_sha256,
                "file": expected_file_sha256,
            }
        )
        return object()

    monkeypatch.setattr(
        validation,
        "load_offline_analysis_execution_receipt",
        fake_loader,
    )
    assert (
        validation.main(
            [
                "analysis-execution-receipt",
                "--receipt",
                str(receipt),
                "--receipt-sha256",
                "a" * 64,
                "--file-sha256",
                "b" * 64,
            ]
        )
        == 0
    )
    assert observed == {
        "path": receipt,
        "receipt": "a" * 64,
        "file": "b" * 64,
    }


@pytest.mark.parametrize(
    ("receipt_sha256", "file_sha256", "message"),
    (
        ("0" * 64, "b" * 64, "semantic digest differs"),
        ("a" * 64, "0" * 64, "file digest differs"),
    ),
)
def test_analysis_command_rejects_semantic_or_file_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_sha256: str,
    file_sha256: str,
    message: str,
) -> None:
    receipt = tmp_path / "analysis-execution-receipt.json"
    receipt.write_bytes(b"receipt\n")

    def fake_loader(
        path: Path,
        *,
        expected_receipt_sha256: str,
        expected_file_sha256: str,
    ) -> object:
        del path, expected_receipt_sha256, expected_file_sha256
        raise ValueError(message)

    monkeypatch.setattr(
        validation,
        "load_offline_analysis_execution_receipt",
        fake_loader,
    )
    with pytest.raises(SystemExit, match=message):
        validation.main(
            [
                "analysis-execution-receipt",
                "--receipt",
                str(receipt),
                "--receipt-sha256",
                receipt_sha256,
                "--file-sha256",
                file_sha256,
            ]
        )
