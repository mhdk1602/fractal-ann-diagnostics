from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.cli as cli


def test_verify_study_artifacts_wires_manifest_pins_to_local_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "study.json"
    artifact_root = tmp_path / "artifacts"
    artifact_map = tmp_path / "map.json"
    receipt_path = tmp_path / "receipt.json"
    payload = {"artifacts": [{"id": "model", "sha256": "a" * 64}]}
    specs = (object(),)
    receipt = SimpleNamespace(
        artifacts=(object(),),
        manifest_sha256="b" * 64,
        receipt_sha256="c" * 64,
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_study_manifest", lambda path: payload)

    def validate(value: object, *, require_frozen: bool = False) -> None:
        observed["validated"] = (value, require_frozen)

    monkeypatch.setattr(cli, "validate_study_manifest", validate)
    monkeypatch.setattr(cli, "manifest_sha256", lambda value: "b" * 64)

    def load_map(
        path: Path,
        *,
        expected_sha256_by_id: dict[str, str],
    ) -> tuple[object, ...]:
        observed["map"] = (path, expected_sha256_by_id)
        return specs

    monkeypatch.setattr(cli, "load_local_artifact_map", load_map)

    def verify(
        root: Path,
        *,
        manifest_sha256: str,
        artifacts: tuple[object, ...],
    ) -> object:
        observed["verification"] = (root, manifest_sha256, artifacts)
        return receipt

    monkeypatch.setattr(cli, "verify_local_artifacts", verify)
    monkeypatch.setattr(
        cli,
        "write_verification_receipt",
        lambda value, target: observed.update(written=(value, target)),
    )

    result = cli.main(
        [
            "verify-study-artifacts",
            "--manifest",
            str(manifest),
            "--artifact-root",
            str(artifact_root),
            "--artifact-map",
            str(artifact_map),
            "--receipt",
            str(receipt_path),
        ]
    )

    assert result == 0
    assert observed["validated"] == (payload, True)
    assert observed["map"] == (artifact_map, {"model": "a" * 64})
    assert observed["verification"] == (artifact_root, "b" * 64, specs)
    assert observed["written"] == (receipt, receipt_path)
    assert "verified 1 frozen study artifacts" in capsys.readouterr().out


def test_begin_sealed_run_cli_requires_and_passes_registration_and_artifact_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    verification = tmp_path / "artifact-verification.json"
    artifact_root = tmp_path / "artifacts"
    artifact_map = tmp_path / "artifact-map.json"
    registration = tmp_path / "protocol-registration.json"
    registration_record = tmp_path / "protocol-registration-record.json"
    observed: dict[str, object] = {}
    receipt = SimpleNamespace(
        protocol_version="0.3.0",
        manifest_sha256="a" * 64,
        receipt_uri="file:///tmp/run.json",
    )

    def begin(
        manifest_path: Path,
        lock_path: Path,
        *,
        runner_identity: str,
        artifact_verification_receipt_path: Path,
        artifact_root: Path,
        local_artifact_map_path: Path,
        protocol_registration_receipt_path: Path,
        protocol_registration_record_path: Path,
    ) -> object:
        observed["args"] = (
            manifest_path,
            lock_path,
            runner_identity,
            artifact_verification_receipt_path,
            artifact_root,
            local_artifact_map_path,
            protocol_registration_receipt_path,
            protocol_registration_record_path,
        )
        return receipt

    monkeypatch.setattr(cli, "begin_sealed_run", begin)
    assert (
        cli.main(
            [
                "begin-sealed-run",
                "--manifest",
                str(manifest),
                "--lock",
                str(lock),
                "--artifact-verification-receipt",
                str(verification),
                "--artifact-root",
                str(artifact_root),
                "--artifact-map",
                str(artifact_map),
                "--protocol-registration-receipt",
                str(registration),
                "--protocol-registration-record",
                str(registration_record),
                "--runner-identity",
                "runner",
            ]
        )
        == 0
    )
    assert observed["args"] == (
        manifest,
        lock,
        "runner",
        verification,
        artifact_root,
        artifact_map,
        registration,
        registration_record,
    )

    with pytest.raises(SystemExit):
        cli.main(
            [
                "begin-sealed-run",
                "--manifest",
                str(manifest),
                "--lock",
                str(lock),
                "--runner-identity",
                "runner",
            ]
        )
