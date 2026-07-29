"""Fixed post-execution checks invoked through the verified host-Python launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .offline_analysis_contract import load_offline_analysis_execution_receipt
from .provider_phase_runtime import ProviderPhaseExecutionReceipt
from .study import FIXED_CORPORA


class ProviderWorkflowValidationError(ValueError):
    """A provider workflow output differs from its registered closure."""


def validate_label_release_inventory(
    inventory_path: str | Path,
    phase_execution_receipt_path: str | Path,
    suite_attempt_id: str,
) -> None:
    try:
        inventory = json.loads(Path(inventory_path).read_bytes())
        receipt = ProviderPhaseExecutionReceipt.from_bytes(
            Path(phase_execution_receipt_path).read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise ProviderWorkflowValidationError(
            "cannot load label-release inventory authority"
        ) from exc
    expected = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
    rows = receipt.outputs
    if (
        inventory
        != {
            "outputs": [row.to_dict() for row in rows],
            "phase": receipt.phase,
            "schema_version": "fractal-provider-activation-inventory-v1",
            "suite_attempt_id": receipt.suite_attempt_id,
        }
        or receipt.phase != "label-release"
        or receipt.suite_attempt_id != suite_attempt_id
        or tuple(row.corpus_id for row in rows) != expected
        or any(
            row.label_release_authority is None
            or row.label_release_authority_sha256 != row.label_release_authority.authority_sha256
            for row in rows
        )
    ):
        raise ProviderWorkflowValidationError("label release inventory authority admission failed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.provider_workflow_validation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    label = commands.add_parser("label-release-inventory")
    label.add_argument("--inventory", required=True, type=Path)
    label.add_argument("--phase-execution-receipt", required=True, type=Path)
    label.add_argument("--suite-attempt-id", required=True)
    analysis = commands.add_parser("analysis-execution-receipt")
    analysis.add_argument("--receipt", required=True, type=Path)
    analysis.add_argument("--receipt-sha256", required=True)
    analysis.add_argument("--file-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "label-release-inventory":
            validate_label_release_inventory(
                arguments.inventory,
                arguments.phase_execution_receipt,
                arguments.suite_attempt_id,
            )
        else:
            load_offline_analysis_execution_receipt(
                arguments.receipt,
                expected_receipt_sha256=arguments.receipt_sha256,
                expected_file_sha256=arguments.file_sha256,
            )
    except (ProviderWorkflowValidationError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
