from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.cli as cli
import fractal_ann_diagnostics.zenodo_publication as publication


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
    registration_package = tmp_path / "confirmatory-c1-registration"
    observed: dict[str, object] = {}
    verified_registration = object()
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
        verified_protocol_registration: object,
    ) -> object:
        observed["args"] = (
            manifest_path,
            lock_path,
            runner_identity,
            artifact_verification_receipt_path,
            artifact_root,
            local_artifact_map_path,
            verified_protocol_registration,
        )
        return receipt

    def verify_registration(
        package_dir: Path,
        *,
        registration_receipt_path: Path,
        registration_record_path: Path,
    ) -> object:
        observed["registration"] = (
            package_dir,
            registration_receipt_path,
            registration_record_path,
        )
        return verified_registration

    monkeypatch.setattr(cli, "begin_sealed_run", begin)
    monkeypatch.setattr(
        publication,
        "verify_production_protocol_registration",
        verify_registration,
    )
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
                "--registration-package",
                str(registration_package),
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
        verified_registration,
    )
    assert observed["registration"] == (
        registration_package,
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


def test_custody_seal_cli_creates_and_verifies_exact_commitments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "study.json"
    receipt_path = tmp_path / "custody-seal.json"
    payload = {"status": "draft"}
    receipt = SimpleNamespace(
        receipt_sha256="a" * 64,
        file_sha256="b" * 64,
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_study_manifest", lambda path: payload)

    def create(
        manifest: object,
        *,
        drand_chain_hash: str,
        drand_round: int,
    ) -> object:
        observed["create"] = (manifest, drand_chain_hash, drand_round)
        return receipt

    monkeypatch.setattr(cli, "custody_seal_receipt_from_manifest", create)
    monkeypatch.setattr(
        cli,
        "write_custody_seal_receipt",
        lambda value, target: observed.update(written=(value, target)),
    )
    assert (
        cli.main(
            [
                "create-custody-seal-receipt",
                "--manifest",
                str(manifest_path),
                "--drand-chain-hash",
                "c" * 64,
                "--drand-round",
                "24000000",
                "--receipt",
                str(receipt_path),
            ]
        )
        == 0
    )
    assert observed["create"] == (payload, "c" * 64, 24_000_000)
    assert observed["written"] == (receipt, receipt_path)
    assert "manifest artifact sha256: " + "b" * 64 in capsys.readouterr().out

    monkeypatch.setattr(cli, "load_custody_seal_receipt", lambda path: receipt)

    def verify(
        value: object,
        manifest: object,
        *,
        require_frozen: bool,
        require_manifest_pin: bool,
    ) -> None:
        observed["verify"] = (
            value,
            manifest,
            require_frozen,
            require_manifest_pin,
        )

    monkeypatch.setattr(cli, "verify_custody_seal_receipt", verify)
    assert (
        cli.main(
            [
                "verify-custody-seal-receipt",
                "--manifest",
                str(manifest_path),
                "--receipt",
                str(receipt_path),
                "--allow-draft",
            ]
        )
        == 0
    )
    assert observed["verify"] == (receipt, payload, False, False)


def test_online_custody_cli_writes_label_blind_admission_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "manifest",
            "custody",
            "run",
            "verification",
            "root",
            "map",
            "admission",
        )
    }
    receipt = SimpleNamespace(
        verified_artifact_ids=("ciphertext", "runner"),
        manifest_sha256="a" * 64,
        receipt_sha256="b" * 64,
    )
    observed: dict[str, object] = {}

    def admit(
        manifest_path: Path,
        *,
        custody_seal_receipt_path: Path,
        sealed_run_receipt_path: Path,
        artifact_verification_receipt_path: Path,
        artifact_root: Path,
        local_artifact_map_path: Path,
        runner_identity: str,
    ) -> object:
        observed["admit"] = (
            manifest_path,
            custody_seal_receipt_path,
            sealed_run_receipt_path,
            artifact_verification_receipt_path,
            artifact_root,
            local_artifact_map_path,
            runner_identity,
        )
        return receipt

    monkeypatch.setattr(cli, "admit_online_custody", admit)
    monkeypatch.setattr(
        cli,
        "write_online_custody_admission_receipt",
        lambda value, target: observed.update(written=(value, target)),
    )
    assert (
        cli.main(
            [
                "verify-online-custody",
                "--manifest",
                str(paths["manifest"]),
                "--custody-seal-receipt",
                str(paths["custody"]),
                "--sealed-run-receipt",
                str(paths["run"]),
                "--artifact-verification-receipt",
                str(paths["verification"]),
                "--artifact-root",
                str(paths["root"]),
                "--artifact-map",
                str(paths["map"]),
                "--runner-identity",
                "runner",
                "--receipt",
                str(paths["admission"]),
            ]
        )
        == 0
    )
    assert observed["admit"] == (
        paths["manifest"],
        paths["custody"],
        paths["run"],
        paths["verification"],
        paths["root"],
        paths["map"],
        "runner",
    )
    assert observed["written"] == (receipt, paths["admission"])


def test_timelock_encryption_cli_passes_only_exact_round_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    plaintext = tmp_path / "labels.json"
    binary = tmp_path / "tle"
    ciphertext = tmp_path / "labels.tlock"
    receipt_path = tmp_path / "encryption.json"
    payload = {"status": "draft"}
    receipt = SimpleNamespace(
        ciphertext_sha256="a" * 64,
        receipt_sha256="b" * 64,
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_study_manifest", lambda path: payload)

    def encrypt(value: object, **kwargs: object) -> object:
        observed["encrypt"] = (value, kwargs)
        return receipt

    monkeypatch.setattr(cli, "encrypt_timelock_label", encrypt)
    monkeypatch.setattr(
        cli,
        "write_timelock_encryption_receipt",
        lambda value, target: observed.update(written=(value, target)),
    )
    assert (
        cli.main(
            [
                "encrypt-timelock-label",
                "--manifest",
                str(manifest),
                "--corpus-id",
                "scifact",
                "--plaintext",
                str(plaintext),
                "--tle-binary",
                str(binary),
                "--drand-network",
                "https://api2.drand.sh/",
                "--drand-chain-hash",
                "c" * 64,
                "--drand-round",
                "24000000",
                "--ciphertext",
                str(ciphertext),
                "--receipt",
                str(receipt_path),
                "--timeout-seconds",
                "30",
                "--max-plaintext-bytes",
                "1024",
                "--max-ciphertext-bytes",
                "2048",
            ]
        )
        == 0
    )
    assert observed["encrypt"] == (
        payload,
        {
            "corpus_id": "scifact",
            "plaintext_path": plaintext,
            "tle_binary_path": binary,
            "drand_network": "https://api2.drand.sh/",
            "drand_chain_hash": "c" * 64,
            "drand_round": 24_000_000,
            "ciphertext_path": ciphertext,
            "timeout_seconds": 30,
            "max_plaintext_bytes": 1024,
            "max_ciphertext_bytes": 2048,
        },
    )
    assert observed["written"] == (receipt, receipt_path)


def test_timelock_receipt_verification_cli_binds_optional_suite_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "encryption.json"
    seal_path = tmp_path / "custody-seal.json"
    manifest = {"status": "frozen"}
    receipt = SimpleNamespace(ciphertext_sha256="a" * 64)
    seal = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_study_manifest", lambda path: manifest)
    monkeypatch.setattr(
        cli,
        "load_timelock_encryption_receipt",
        lambda path: receipt,
    )
    monkeypatch.setattr(cli, "load_custody_seal_receipt", lambda path: seal)

    def verify(
        value: object,
        payload: object,
        *,
        custody_seal: object,
        require_frozen: bool,
    ) -> None:
        observed["verify"] = (
            value,
            payload,
            custody_seal,
            require_frozen,
        )

    monkeypatch.setattr(cli, "verify_timelock_encryption_receipt", verify)
    assert (
        cli.main(
            [
                "verify-timelock-encryption-receipt",
                "--manifest",
                str(manifest_path),
                "--receipt",
                str(receipt_path),
                "--custody-seal",
                str(seal_path),
            ]
        )
        == 0
    )
    assert observed["verify"] == (receipt, manifest, seal, True)


def test_timelock_release_cli_verifies_external_anchor_before_decryption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "manifest",
            "custody-seal",
            "encryption-receipt",
            "completion-receipt",
            "completion-anchor-record",
            "completion-anchor-receipt",
            "suite-namespace",
            "ciphertext",
            "tle",
            "plaintext",
            "decryption-receipt",
        )
    }
    manifest = {"status": "frozen"}
    completion = object()
    anchor = object()
    seal = object()
    encryption = object()
    suite_verifier = object()
    verified_suite = object()
    decryption_receipt = SimpleNamespace(
        plaintext_sha256="a" * 64,
        receipt_sha256="b" * 64,
    )
    verified_release = SimpleNamespace(receipt=decryption_receipt)
    observed: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "load_study_manifest", lambda path: manifest)
    monkeypatch.setattr(
        cli,
        "GitHubSuiteEvidenceVerifier",
        lambda namespace: observed.append(("suite-verifier", namespace)) or suite_verifier,
    )

    def verify_suite(
        namespace: Path,
        *,
        verifier: object,
        expected_state: str,
    ) -> object:
        observed.append(("suite", (namespace, verifier, expected_state)))
        return verified_suite

    monkeypatch.setattr(cli, "verify_suite_state", verify_suite)
    monkeypatch.setattr(
        cli,
        "load_prediction_completion_receipt",
        lambda path: completion,
    )

    def verify_anchor(
        value: object,
        *,
        anchor_record_path: Path,
        anchor_receipt_path: Path,
    ) -> object:
        observed.append(
            (
                "anchor",
                (value, anchor_record_path, anchor_receipt_path),
            )
        )
        return anchor

    monkeypatch.setattr(cli, "verify_prediction_completion_anchor", verify_anchor)
    monkeypatch.setattr(cli, "load_custody_seal_receipt", lambda path: seal)
    monkeypatch.setattr(
        cli,
        "load_timelock_encryption_receipt",
        lambda path: encryption,
    )

    def release(value: object, **kwargs: object) -> object:
        observed.append(("release", (value, kwargs)))
        return verified_release

    monkeypatch.setattr(cli, "release_timelock_label", release)
    monkeypatch.setattr(
        cli,
        "write_timelock_decryption_receipt",
        lambda value, target: observed.append(("write", (value, target))),
    )
    assert (
        cli.main(
            [
                "release-timelock-label",
                "--manifest",
                str(paths["manifest"]),
                "--corpus-id",
                "scifact",
                "--custody-seal",
                str(paths["custody-seal"]),
                "--encryption-receipt",
                str(paths["encryption-receipt"]),
                "--completion-receipt",
                str(paths["completion-receipt"]),
                "--completion-anchor-record",
                str(paths["completion-anchor-record"]),
                "--completion-anchor-receipt",
                str(paths["completion-anchor-receipt"]),
                "--suite-namespace",
                str(paths["suite-namespace"]),
                "--ciphertext",
                str(paths["ciphertext"]),
                "--tle-binary",
                str(paths["tle"]),
                "--plaintext-output",
                str(paths["plaintext"]),
                "--receipt",
                str(paths["decryption-receipt"]),
                "--timeout-seconds",
                "30",
                "--max-ciphertext-bytes",
                "2048",
                "--max-plaintext-bytes",
                "1024",
            ]
        )
        == 0
    )
    assert observed[0] == ("suite-verifier", paths["suite-namespace"])
    assert observed[1] == (
        "suite",
        (paths["suite-namespace"], suite_verifier, "ONLINE_COMPLETE"),
    )
    assert observed[2] == (
        "anchor",
        (
            completion,
            paths["completion-anchor-record"],
            paths["completion-anchor-receipt"],
        ),
    )
    assert observed[3] == (
        "release",
        (
            manifest,
            {
                "corpus_id": "scifact",
                "custody_seal": seal,
                "encryption_receipt": encryption,
                "verified_completion_anchor": anchor,
                "verified_suite_completion": verified_suite,
                "ciphertext_path": paths["ciphertext"],
                "tle_binary_path": paths["tle"],
                "plaintext_output_path": paths["plaintext"],
                "timeout_seconds": 30,
                "max_ciphertext_bytes": 2048,
                "max_plaintext_bytes": 1024,
            },
        ),
    )
    assert observed[4] == (
        "write",
        (decryption_receipt, paths["decryption-receipt"]),
    )


def test_timelock_release_cli_replay_stops_before_suite_or_anchor_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "decryption-receipt.json"
    receipt.write_bytes(b"already consumed\n")
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "GitHubSuiteEvidenceVerifier",
        lambda namespace: calls.append("suite") or object(),
    )
    monkeypatch.setattr(
        cli,
        "verify_prediction_completion_anchor",
        lambda *args, **kwargs: calls.append("anchor") or object(),
    )

    with pytest.raises(ValueError, match="already exists"):
        cli.main(
            [
                "release-timelock-label",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--corpus-id",
                "scifact",
                "--custody-seal",
                str(tmp_path / "custody.json"),
                "--encryption-receipt",
                str(tmp_path / "encryption.json"),
                "--completion-receipt",
                str(tmp_path / "completion.json"),
                "--completion-anchor-record",
                str(tmp_path / "anchor.json"),
                "--completion-anchor-receipt",
                str(tmp_path / "anchor-receipt.json"),
                "--suite-namespace",
                str(tmp_path / "suite"),
                "--ciphertext",
                str(tmp_path / "labels.tlock"),
                "--tle-binary",
                str(tmp_path / "tle"),
                "--plaintext-output",
                str(tmp_path / "labels.json"),
                "--receipt",
                str(receipt),
            ]
        )
    assert calls == []


def test_timelock_release_cli_rejects_substituted_suite_before_anchor_or_decryption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "load_study_manifest", lambda path: {"status": "frozen"})
    monkeypatch.setattr(
        cli,
        "GitHubSuiteEvidenceVerifier",
        lambda namespace: object(),
    )

    def reject_suite(*args: object, **kwargs: object) -> object:
        calls.append("suite")
        raise ValueError("suite state is ONLINE_COMPLETE, but bytes were substituted")

    monkeypatch.setattr(cli, "verify_suite_state", reject_suite)
    monkeypatch.setattr(
        cli,
        "verify_prediction_completion_anchor",
        lambda *args, **kwargs: calls.append("anchor") or object(),
    )
    monkeypatch.setattr(
        cli,
        "release_timelock_label",
        lambda *args, **kwargs: calls.append("release") or object(),
    )

    with pytest.raises(ValueError, match="substituted"):
        cli.main(
            [
                "release-timelock-label",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--corpus-id",
                "scifact",
                "--custody-seal",
                str(tmp_path / "custody.json"),
                "--encryption-receipt",
                str(tmp_path / "encryption.json"),
                "--completion-receipt",
                str(tmp_path / "completion.json"),
                "--completion-anchor-record",
                str(tmp_path / "anchor.json"),
                "--completion-anchor-receipt",
                str(tmp_path / "anchor-receipt.json"),
                "--suite-namespace",
                str(tmp_path / "substituted-suite"),
                "--ciphertext",
                str(tmp_path / "labels.tlock"),
                "--tle-binary",
                str(tmp_path / "tle"),
                "--plaintext-output",
                str(tmp_path / "labels.json"),
                "--receipt",
                str(tmp_path / "decryption.json"),
            ]
        )
    assert calls == ["suite"]
