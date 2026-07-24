from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from production_workload_fixtures import registered_c0_evidence_release
from test_study import _provider_phase_plans

import fractal_ann_diagnostics.custody as custody_module
from fractal_ann_diagnostics.artifact_integrity import (
    LOCAL_ARTIFACT_MAP_SCHEMA,
    ArtifactVerificationReceipt,
    VerifiedArtifact,
    write_exclusive_receipt_bytes,
    write_verification_receipt,
)
from fractal_ann_diagnostics.custody import (
    ONLINE_CUSTODY_REVALIDATION_ROLES,
    CustodyError,
    CustodySealReceipt,
    OnlineCustodyAdmissionReceipt,
    TimelockEncryptionReceipt,
    admit_online_custody,
    custody_seal_receipt_from_manifest,
    encrypt_timelock_label,
    load_custody_seal_receipt,
    load_online_custody_admission_receipt,
    load_timelock_encryption_receipt,
    online_custody_artifact_specs,
    verify_custody_seal_receipt,
    verify_timelock_encryption_receipt,
    write_custody_seal_receipt,
    write_online_custody_admission_receipt,
    write_timelock_encryption_receipt,
)
from fractal_ann_diagnostics.production_workload_registration import (
    PRODUCTION_WORKLOAD_SPEC_SCHEMA,
    production_workload_file_sha256,
)
from fractal_ann_diagnostics.provider_contract import SOURCE_BUILT_LINUX_ARM64_TLE_SHA256
from fractal_ann_diagnostics.study import (
    FIXED_CORPORA,
    SealedRunReceipt,
    load_study_manifest,
    manifest_sha256,
    sealed_receipt_uri,
    validate_study_manifest,
)

_CHAIN_HASH = "a" * 64
_DRAND_ROUND = 24_000_000
_RUNNER_IDENTITY = "github-actions:environment:confirmatory"
_COMMIT = "1" * 40
_IMAGE = f"ghcr.io/example/confirmatory@sha256:{'2' * 64}"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _digest(value: str | bytes) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _production_workloads() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position, corpus_id in enumerate(FIXED_CORPORA):
        root = f"/opt/study/corpora/{corpus_id}"

        def pin(name: str) -> str:
            return _digest(f"{corpus_id}:{name}")

        spec: dict[str, object] = {
            "artifact_root": f"{root}/factory",
            "artifact_tree_sha256": pin("artifact-tree"),
            "authorized_index_store_root": f"{root}/authorized-index",
            "authorized_index_store_tree_sha256": pin("authorized-index-tree"),
            "available_family_count": 75,
            "code_commit": _COMMIT,
            "corpus_id": corpus_id,
            "embedding_store_root": f"{root}/embedding-store",
            "embedding_store_tree_sha256": pin("embedding-store-tree"),
            "expected_authorized_index_store_receipt_sha256": pin("index-receipt"),
            "expected_policy_intervention_receipt_sha256": pin("policy-receipt"),
            "expected_pseudonym_key_sha256": pin("pseudonym-key"),
            "factory_artifact_tree_sha256": pin("factory-tree"),
            "factory_config_sha256": pin("factory-config"),
            "factory_suite_receipt_sha256": pin("factory-suite"),
            "feature_bindings": [
                {
                    "backend": "hnsw",
                    "drift_family": "current-truth",
                    "group_order": position,
                    "policy_complexity": 1.0,
                    "policy_state": "current",
                    "repetition": 0,
                    "subject": corpus_id,
                    "version_lag": 0.0,
                }
            ],
            "index_bundle_receipt_path": f"{root}/bundles/index-receipt.json",
            "index_bundle_receipt_sha256": pin("index-bundle"),
            "online_execution_plan_sha256": pin("execution-plan"),
            "online_execution_tree_sha256": pin("execution-tree"),
            "partition_audit_file_sha256": pin("partition-file"),
            "partition_audit_path": f"{root}/query-partition-audit.json",
            "partition_audit_sha256": pin("partition-logical"),
            "policy_bundle_receipt_path": f"{root}/bundles/policy-receipt.json",
            "policy_bundle_receipt_sha256": pin("policy-bundle"),
            "policy_intervention_root": f"{root}/policy-intervention",
            "policy_intervention_tree_sha256": pin("policy-tree"),
            "pseudonym_key_path": f"{root}/custody/pseudonym.key",
            "query_package_root": f"{root}/query-package",
            "query_package_tree_sha256": pin("query-tree"),
            "query_receipt_sha256": pin("query-receipt"),
            "runner_identity": _RUNNER_IDENTITY,
            "runner_image": _IMAGE,
            "runner_platform": "linux/arm64",
            "schema_version": PRODUCTION_WORKLOAD_SPEC_SCHEMA,
            "selected_family_count": 75,
            "sharded_execution_plan_file_sha256": pin("plan-file"),
            "staged_root": f"{root}/online-staging",
            "staged_tree_sha256": pin("staged-tree"),
            "trial_runtime_admission_receipt_file_sha256": pin("runtime-receipt-file"),
        }
        rows.append(
            {
                "canonical_file_sha256": production_workload_file_sha256(spec),
                "corpus_id": corpus_id,
                "spec": spec,
            }
        )
    return rows


def _artifact_for(
    manifest: dict[str, object],
    role: str,
    *,
    corpus_id: str | None = None,
) -> dict[str, object]:
    return next(
        artifact
        for artifact in manifest["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] == role
        and (corpus_id is None or artifact.get("corpus_id") == corpus_id)
    )


def _draft_encryption_manifest(
    plaintext: Path,
    tle_binary: Path,
    *,
    corpus_id: str = FIXED_CORPORA[0],
) -> dict[str, object]:
    manifest = load_study_manifest(_REPOSITORY_ROOT / "research/study-manifest.json")
    labels = _artifact_for(manifest, "sealed-labels", corpus_id=corpus_id)
    labels["uri"] = plaintext.as_uri()
    labels["revision"] = "v1.0.0"
    labels["sha256"] = _digest(plaintext.read_bytes())
    tool = _artifact_for(manifest, "timelock-tool")
    tool["uri"] = tle_binary.as_uri()
    tool["revision"] = "v1.0.0"
    tool["sha256"] = _digest(tle_binary.read_bytes())
    validate_study_manifest(manifest)
    return manifest


def _write_fake_tle(path: Path, *, body: str | None = None) -> None:
    program = (
        body
        or """
import sys

data = sys.stdin.buffer.read()
sys.stdout.buffer.write(b"tlock-v1:" + data[::-1])
"""
    )
    path.write_text(f"#!{sys.executable}\n{program.lstrip()}", encoding="utf-8")
    path.chmod(0o700)


def _frozen_manifest(
    tmp_path: Path,
) -> tuple[dict[str, object], CustodySealReceipt]:
    manifest = load_study_manifest(_REPOSITORY_ROOT / "research/study-manifest.json")
    manifest["protocol_version"] = "0.3.0"
    manifest["status"] = "frozen"
    manifest["freeze_blockers"] = []

    analysis = manifest["analysis"]
    analysis["nested_rows_per_family"] = 3
    analysis["geometry_gain_thresholds"] = {
        "log_loss_reduction": 0.0,
        "brier_score_reduction": 0.0,
        "auprc_gain": 0.0,
    }
    analysis["low_geometry"] = {"instability": 0.1, "lid": 1.0}
    analysis["high_geometry"] = {"instability": 0.9, "lid": 9.0}
    analysis["static_comparator_action"] = "hnsw-high"
    power = analysis["power"]
    power["dependence_source"] = "development query-family endpoint vectors"
    power["effect_scenarios"] = [
        "registered-minimum-effects",
        "development-observed-effects",
    ]
    power["selected_families_per_corpus"] = 75
    power["simulation_seed"] = 71
    power["selected_joint_power_lower_bound"] = 0.91

    for artifact in manifest["artifacts"]:
        artifact_id = str(artifact["id"])
        artifact["uri"] = f"https://example.test/artifacts/{artifact_id}"
        artifact_sha256 = (
            SOURCE_BUILT_LINUX_ARM64_TLE_SHA256
            if artifact["role"] == "timelock-tool"
            else _digest(artifact_id)
        )
        if artifact["role"] == "source-code":
            artifact["revision"] = _COMMIT
        elif artifact["role"] == "online-execution":
            artifact["revision"] = f"sha256:{_digest(artifact_id + '-logical')}"
        elif artifact["role"] == "opa-runtime-binary":
            artifact["revision"] = f"sha256:{artifact_sha256}"
        else:
            artifact["revision"] = "v1.0.0"
        artifact["sha256"] = artifact_sha256
    tool_sha256 = str(_artifact_for(manifest, "timelock-tool")["sha256"])
    for corpus_id in FIXED_CORPORA:
        operation = TimelockEncryptionReceipt(
            corpus_id=corpus_id,
            plaintext_sha256=str(
                _artifact_for(manifest, "sealed-labels", corpus_id=corpus_id)["sha256"]
            ),
            plaintext_byte_count=1,
            ciphertext_sha256=str(
                _artifact_for(
                    manifest,
                    "sealed-label-ciphertext",
                    corpus_id=corpus_id,
                )["sha256"]
            ),
            ciphertext_byte_count=2,
            tle_binary_sha256=tool_sha256,
            drand_network="https://api2.drand.sh/",
            drand_chain_hash=_CHAIN_HASH,
            drand_round=_DRAND_ROUND,
            tle_arguments=(
                "--encrypt",
                "--network=https://api2.drand.sh/",
                f"--chain={_CHAIN_HASH}",
                f"--round={_DRAND_ROUND}",
            ),
        )
        _artifact_for(
            manifest,
            "timelock-encryption-receipt",
            corpus_id=corpus_id,
        )["sha256"] = operation.file_sha256

    receipt_root = tmp_path / "run-receipts"
    receipt_root.mkdir()
    sealed = manifest["sealed_execution"]
    sealed["custodian"] = "custodian@example.test"
    sealed["approval_environment"] = "confirmatory"
    sealed["results_store"] = "s3://immutable-confirmatory-results"
    sealed["runner_identity"] = _RUNNER_IDENTITY
    sealed["code_commit"] = _COMMIT
    sealed["c0_evidence_release"] = registered_c0_evidence_release(code_commit=_COMMIT)
    sealed["runner_image"] = _IMAGE
    sealed["provider_phase_plans"] = _provider_phase_plans(
        runner_image=_IMAGE,
        timelock_sha256=tool_sha256,
        code_commit=_COMMIT,
    )
    sealed["production_controls"] = {
        "materialization_config_file_sha256": "1" * 64,
        "blueprint_receipt_sha256": "2" * 64,
        "blueprint_receipt_file_sha256": "3" * 64,
    }
    sealed["hardware"] = {
        "provider": "aws",
        "instance_type": "c7i.4xlarge",
        "cpu_model": "Intel Xeon Platinum 8488C",
        "logical_cores": 16,
        "memory_gib": 32,
        "accelerator": "none",
        "region": "us-east-1",
        "operating_system": "ubuntu-24.04",
    }
    sealed["receipt_uri_template"] = receipt_root.resolve().as_uri() + "/{manifest_sha256}.json"
    manifest["production_workloads"] = _production_workloads()

    validate_study_manifest(manifest, require_frozen=True)
    receipt = custody_seal_receipt_from_manifest(
        manifest,
        drand_chain_hash=_CHAIN_HASH,
        drand_round=_DRAND_ROUND,
    )
    _artifact_for(manifest, "custody-seal-receipt")["sha256"] = receipt.file_sha256
    validate_study_manifest(manifest, require_frozen=True)
    verify_custody_seal_receipt(receipt, manifest)
    return manifest, receipt


def _canonical_manifest_bytes(manifest: dict[str, object]) -> bytes:
    return (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _write_local_artifact_map(
    path: Path,
    manifest: dict[str, object],
) -> dict[str, str]:
    relative_by_id: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for position, artifact in enumerate(manifest["artifacts"]):
        artifact_id = str(artifact["id"])
        relative_path = f"objects/{position:02d}.bin"
        relative_by_id[artifact_id] = relative_path
        rows.append(
            {
                "artifact_id": artifact_id,
                "kind": "file",
                "relative_path": relative_path,
            }
        )
    path.write_text(
        json.dumps(
            {"artifacts": rows, "schema_version": LOCAL_ARTIFACT_MAP_SCHEMA},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return relative_by_id


def _materialize_admission_package(
    tmp_path: Path,
) -> dict[str, object]:
    manifest, seal = _frozen_manifest(tmp_path)
    manifest_path = tmp_path / "frozen-study.json"
    manifest_path.write_bytes(_canonical_manifest_bytes(manifest))
    manifest_digest = manifest_sha256(manifest)

    artifact_root = tmp_path / "artifacts"
    objects = artifact_root / "objects"
    objects.mkdir(parents=True)
    artifact_map_path = tmp_path / "artifact-map.json"
    relative_by_id = _write_local_artifact_map(artifact_map_path, manifest)

    roles_by_id = {str(artifact["id"]): str(artifact["role"]) for artifact in manifest["artifacts"]}
    bytes_by_id = {
        str(artifact["id"]): str(artifact["id"]).encode("utf-8")
        for artifact in manifest["artifacts"]
    }
    seal_artifact = _artifact_for(manifest, "custody-seal-receipt")
    seal_id = str(seal_artifact["id"])
    bytes_by_id[seal_id] = seal.canonical_bytes() + b"\n"

    for artifact_id, role in roles_by_id.items():
        if role not in ONLINE_CUSTODY_REVALIDATION_ROLES:
            continue
        target = artifact_root / relative_by_id[artifact_id]
        if artifact_id == seal_id:
            write_custody_seal_receipt(seal, target)
        else:
            target.write_bytes(bytes_by_id[artifact_id])

    full_rows = tuple(
        VerifiedArtifact(
            artifact_id=str(artifact["id"]),
            relative_path=relative_by_id[str(artifact["id"])],
            kind="file",
            exact=True,
            expected_sha256=str(artifact["sha256"]),
            verified_sha256=str(artifact["sha256"]),
            file_count=1,
            directory_count=0,
            byte_count=len(bytes_by_id[str(artifact["id"])]),
            observed_file_count=1,
            observed_directory_count=0,
            observed_byte_count=len(bytes_by_id[str(artifact["id"])]),
        )
        for artifact in manifest["artifacts"]
    )
    full_verification = ArtifactVerificationReceipt(
        manifest_sha256=manifest_digest,
        artifacts=full_rows,
    )
    full_verification_path = tmp_path / "full-artifact-verification.json"
    write_verification_receipt(full_verification, full_verification_path)

    run_receipt_path = Path(sealed_receipt_uri(manifest).removeprefix("file://"))
    run_receipt = SealedRunReceipt(
        manifest_sha256=manifest_digest,
        protocol_version="0.3.0",
        started_at_utc="2026-07-14T12:00:00Z",
        runner_identity=_RUNNER_IDENTITY,
        code_commit=_COMMIT,
        runner_image=_IMAGE,
        protocol_registration_receipt_uri=(tmp_path / "registration.json").as_uri(),
        protocol_registration_receipt_sha256="3" * 64,
        protocol_registration_record_uri=(tmp_path / "registry-record.json").as_uri(),
        verification_receipt_uri=full_verification_path.as_uri(),
        verification_receipt_sha256=full_verification.receipt_sha256,
        receipt_uri=run_receipt_path.as_uri(),
    )
    write_exclusive_receipt_bytes(
        run_receipt.canonical_bytes() + b"\n",
        run_receipt_path,
    )
    return {
        "artifact_map_path": artifact_map_path,
        "artifact_root": artifact_root,
        "full_verification_path": full_verification_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "relative_by_id": relative_by_id,
        "run_receipt_path": run_receipt_path,
        "seal": seal,
        "seal_path": artifact_root / relative_by_id[seal_id],
    }


def test_pinned_tle_adapter_encrypts_exact_plaintext_without_a_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plaintext = tmp_path / "scifact-labels.json"
    plaintext.write_bytes(b'{"labels":["sealed"]}\n')
    tle_binary = tmp_path / "tle"
    _write_fake_tle(tle_binary)
    manifest = _draft_encryption_manifest(plaintext, tle_binary)
    ciphertext = tmp_path / "scifact-labels.tlock"
    operation_receipt = tmp_path / "scifact-encryption.json"
    network = "https://api2.drand.sh/"

    real_popen = custody_module.subprocess.Popen
    observed: dict[str, object] = {}

    def capture_popen(command: list[str], **kwargs: object):
        observed["command"] = command
        observed["shell"] = kwargs.get("shell")
        observed["env"] = kwargs.get("env")
        return real_popen(command, **kwargs)

    monkeypatch.setattr(custody_module.subprocess, "Popen", capture_popen)
    receipt = encrypt_timelock_label(
        manifest,
        corpus_id=FIXED_CORPORA[0],
        plaintext_path=plaintext,
        tle_binary_path=tle_binary,
        drand_network=network,
        drand_chain_hash=_CHAIN_HASH,
        drand_round=_DRAND_ROUND,
        ciphertext_path=ciphertext,
        timeout_seconds=5,
        max_plaintext_bytes=1024,
        max_ciphertext_bytes=2048,
    )
    expected = b"tlock-v1:" + plaintext.read_bytes()[::-1]
    assert ciphertext.read_bytes() == expected
    assert receipt == TimelockEncryptionReceipt(
        corpus_id=FIXED_CORPORA[0],
        plaintext_sha256=_digest(plaintext.read_bytes()),
        plaintext_byte_count=len(plaintext.read_bytes()),
        ciphertext_sha256=_digest(expected),
        ciphertext_byte_count=len(expected),
        tle_binary_sha256=_digest(tle_binary.read_bytes()),
        drand_network=network,
        drand_chain_hash=_CHAIN_HASH,
        drand_round=_DRAND_ROUND,
        tle_arguments=(
            "--encrypt",
            f"--network={network}",
            f"--chain={_CHAIN_HASH}",
            f"--round={_DRAND_ROUND}",
        ),
    )
    assert observed["command"] == [str(tle_binary), *receipt.tle_arguments]
    assert observed["shell"] is False
    assert observed["env"] == {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    assert not any(
        argument.startswith(("--duration", "--output", "--decrypt")) or "latest" in argument.lower()
        for argument in receipt.tle_arguments
    )

    write_timelock_encryption_receipt(receipt, operation_receipt)
    assert load_timelock_encryption_receipt(operation_receipt) == receipt
    _artifact_for(
        manifest,
        "sealed-label-ciphertext",
        corpus_id=FIXED_CORPORA[0],
    )["sha256"] = receipt.ciphertext_sha256
    _artifact_for(
        manifest,
        "timelock-encryption-receipt",
        corpus_id=FIXED_CORPORA[0],
    )["sha256"] = receipt.file_sha256
    verify_timelock_encryption_receipt(
        receipt,
        manifest,
        require_frozen=False,
    )


def test_tle_adapter_refuses_unpinned_inputs_and_binary(
    tmp_path: Path,
) -> None:
    plaintext = tmp_path / "labels.json"
    plaintext.write_bytes(b"labels\n")
    tle_binary = tmp_path / "tle"
    _write_fake_tle(tle_binary)
    manifest = _draft_encryption_manifest(plaintext, tle_binary)
    output = tmp_path / "labels.tlock"

    wrong_tool = deepcopy(manifest)
    _artifact_for(wrong_tool, "timelock-tool")["sha256"] = "f" * 64
    with pytest.raises(CustodyError, match="binary digest"):
        encrypt_timelock_label(
            wrong_tool,
            corpus_id=FIXED_CORPORA[0],
            plaintext_path=plaintext,
            tle_binary_path=tle_binary,
            drand_network="https://api2.drand.sh/",
            drand_chain_hash=_CHAIN_HASH,
            drand_round=_DRAND_ROUND,
            ciphertext_path=output,
        )
    assert not output.exists()

    wrong_plaintext = deepcopy(manifest)
    _artifact_for(
        wrong_plaintext,
        "sealed-labels",
        corpus_id=FIXED_CORPORA[0],
    )["sha256"] = "e" * 64
    with pytest.raises(CustodyError, match="plaintext label digest"):
        encrypt_timelock_label(
            wrong_plaintext,
            corpus_id=FIXED_CORPORA[0],
            plaintext_path=plaintext,
            tle_binary_path=tle_binary,
            drand_network="https://api2.drand.sh/",
            drand_chain_hash=_CHAIN_HASH,
            drand_round=_DRAND_ROUND,
            ciphertext_path=output,
        )
    assert not output.exists()


def test_tle_adapter_enforces_timeout_output_bound_and_exclusive_target(
    tmp_path: Path,
) -> None:
    plaintext = tmp_path / "labels.json"
    plaintext.write_bytes(b"labels\n")

    oversized_binary = tmp_path / "tle-oversized"
    _write_fake_tle(
        oversized_binary,
        body="""
import sys

sys.stdin.buffer.read()
sys.stdout.buffer.write(b"x" * 4096)
""",
    )
    oversized_manifest = _draft_encryption_manifest(plaintext, oversized_binary)
    oversized_output = tmp_path / "oversized.tlock"
    with pytest.raises(CustodyError, match="exceeds max_ciphertext_bytes"):
        encrypt_timelock_label(
            oversized_manifest,
            corpus_id=FIXED_CORPORA[0],
            plaintext_path=plaintext,
            tle_binary_path=oversized_binary,
            drand_network="https://api2.drand.sh/",
            drand_chain_hash=_CHAIN_HASH,
            drand_round=_DRAND_ROUND,
            ciphertext_path=oversized_output,
            max_ciphertext_bytes=32,
        )
    assert not oversized_output.exists()

    slow_binary = tmp_path / "tle-slow"
    _write_fake_tle(
        slow_binary,
        body="""
import sys
import time

sys.stdin.buffer.read()
time.sleep(2)
sys.stdout.buffer.write(b"late")
""",
    )
    slow_manifest = _draft_encryption_manifest(plaintext, slow_binary)
    slow_output = tmp_path / "slow.tlock"
    with pytest.raises(CustodyError, match="exceeded 1 seconds"):
        encrypt_timelock_label(
            slow_manifest,
            corpus_id=FIXED_CORPORA[0],
            plaintext_path=plaintext,
            tle_binary_path=slow_binary,
            drand_network="https://api2.drand.sh/",
            drand_chain_hash=_CHAIN_HASH,
            drand_round=_DRAND_ROUND,
            ciphertext_path=slow_output,
            timeout_seconds=1,
        )
    assert not slow_output.exists()

    normal_binary = tmp_path / "tle-normal"
    _write_fake_tle(normal_binary)
    normal_manifest = _draft_encryption_manifest(plaintext, normal_binary)
    existing = tmp_path / "existing.tlock"
    existing.write_bytes(b"custodian-owned")
    with pytest.raises(CustodyError, match="already exists"):
        encrypt_timelock_label(
            normal_manifest,
            corpus_id=FIXED_CORPORA[0],
            plaintext_path=plaintext,
            tle_binary_path=normal_binary,
            drand_network="https://api2.drand.sh/",
            drand_chain_hash=_CHAIN_HASH,
            drand_round=_DRAND_ROUND,
            ciphertext_path=existing,
        )
    assert existing.read_bytes() == b"custodian-owned"


def test_timelock_receipt_rejects_duration_or_output_argument(
    tmp_path: Path,
) -> None:
    plaintext = tmp_path / "labels.json"
    plaintext.write_bytes(b"labels\n")
    tle_binary = tmp_path / "tle"
    _write_fake_tle(tle_binary)
    manifest = _draft_encryption_manifest(plaintext, tle_binary)
    receipt = encrypt_timelock_label(
        manifest,
        corpus_id=FIXED_CORPORA[0],
        plaintext_path=plaintext,
        tle_binary_path=tle_binary,
        drand_network="https://api2.drand.sh/",
        drand_chain_hash=_CHAIN_HASH,
        drand_round=_DRAND_ROUND,
        ciphertext_path=tmp_path / "labels.tlock",
    )
    payload = receipt.to_dict()
    payload["tle_arguments"] = ["--encrypt", "--duration=24h"]
    with pytest.raises(CustodyError, match="exact encrypt"):
        TimelockEncryptionReceipt.from_dict(payload)


def test_custody_seal_is_canonical_closed_and_manifest_pinned(
    tmp_path: Path,
) -> None:
    manifest, receipt = _frozen_manifest(tmp_path)
    target = tmp_path / "custody-seal.json"
    write_custody_seal_receipt(receipt, target)

    assert load_custody_seal_receipt(target) == receipt
    assert receipt.receipt_sha256 != receipt.file_sha256
    assert tuple(row.corpus_id for row in receipt.commitments) == FIXED_CORPORA
    verify_custody_seal_receipt(receipt, manifest)

    commitment = receipt.commitments[0]
    operation = TimelockEncryptionReceipt(
        corpus_id=commitment.corpus_id,
        plaintext_sha256=commitment.sealed_label_plaintext_sha256,
        plaintext_byte_count=1,
        ciphertext_sha256=commitment.sealed_label_ciphertext_sha256,
        ciphertext_byte_count=2,
        tle_binary_sha256=receipt.timelock_tool_sha256,
        drand_network="https://api2.drand.sh/",
        drand_chain_hash=receipt.drand_chain_hash,
        drand_round=receipt.drand_round,
        tle_arguments=(
            "--encrypt",
            "--network=https://api2.drand.sh/",
            f"--chain={receipt.drand_chain_hash}",
            f"--round={receipt.drand_round}",
        ),
    )
    verify_timelock_encryption_receipt(
        operation,
        manifest,
        custody_seal=receipt,
    )
    with pytest.raises(CustodyError, match="receipt file digest"):
        verify_timelock_encryption_receipt(
            TimelockEncryptionReceipt.from_dict(
                {
                    **operation.to_dict(),
                    "drand_round": operation.drand_round + 1,
                    "tle_arguments": [
                        "--encrypt",
                        "--network=https://api2.drand.sh/",
                        f"--chain={operation.drand_chain_hash}",
                        f"--round={operation.drand_round + 1}",
                    ],
                }
            ),
            manifest,
            custody_seal=receipt,
        )

    unexpected = receipt.to_dict()
    unexpected["unregistered_assertion"] = True
    with pytest.raises(CustodyError, match="closed schema"):
        CustodySealReceipt.from_dict(unexpected)

    wrong_pin = deepcopy(manifest)
    _artifact_for(wrong_pin, "custody-seal-receipt")["sha256"] = "f" * 64
    with pytest.raises(CustodyError, match="manifest pin"):
        verify_custody_seal_receipt(receipt, wrong_pin)

    wrong_ciphertext = receipt.to_dict()
    wrong_ciphertext["commitments"][0]["sealed_label_ciphertext_sha256"] = "e" * 64
    with pytest.raises(CustodyError, match="differ from the manifest"):
        verify_custody_seal_receipt(
            CustodySealReceipt.from_dict(wrong_ciphertext),
            manifest,
        )


@pytest.mark.parametrize("round_value", (0, -1, "24h", "@latest", True))
def test_custody_seal_requires_one_exact_positive_drand_round(
    tmp_path: Path,
    round_value: object,
) -> None:
    manifest, receipt = _frozen_manifest(tmp_path)
    payload = receipt.to_dict()
    payload["drand_round"] = round_value
    with pytest.raises(CustodyError, match="positive integer"):
        CustodySealReceipt.from_dict(payload)
    assert manifest["status"] == "frozen"


def test_custody_loader_rejects_duplicate_keys_and_noncanonical_order(
    tmp_path: Path,
) -> None:
    _, receipt = _frozen_manifest(tmp_path)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(
        receipt.canonical_bytes().replace(
            b'"commitments":',
            b'"commitments":[],"commitments":',
            1,
        )
        + b"\n"
    )
    with pytest.raises(CustodyError, match="duplicate key"):
        load_custody_seal_receipt(duplicate)

    reordered_payload = receipt.to_dict()
    reordered_payload["commitments"] = list(reversed(reordered_payload["commitments"]))
    reordered = tmp_path / "reordered.json"
    reordered.write_text(
        json.dumps(reordered_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CustodyError, match="canonical JSON"):
        load_custody_seal_receipt(reordered)


def test_online_admission_never_opens_plaintext_labels_or_sealed_inputs(
    tmp_path: Path,
) -> None:
    package = _materialize_admission_package(tmp_path)
    manifest = package["manifest"]
    artifact_root = package["artifact_root"]
    relative_by_id = package["relative_by_id"]

    excluded = [
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] in {"sealed-labels", "sealed-inputs"}
    ]
    assert excluded
    assert all(
        not (artifact_root / relative_by_id[str(artifact["id"])]).exists() for artifact in excluded
    )

    receipt = admit_online_custody(
        package["manifest_path"],
        custody_seal_receipt_path=package["seal_path"],
        sealed_run_receipt_path=package["run_receipt_path"],
        artifact_verification_receipt_path=package["full_verification_path"],
        artifact_root=artifact_root,
        local_artifact_map_path=package["artifact_map_path"],
        runner_identity=_RUNNER_IDENTITY,
    )
    roles_by_id = {str(artifact["id"]): str(artifact["role"]) for artifact in manifest["artifacts"]}
    assert receipt.verified_artifact_ids
    assert {
        roles_by_id[artifact_id] for artifact_id in receipt.verified_artifact_ids
    } <= ONLINE_CUSTODY_REVALIDATION_ROLES
    assert "opa-runtime-binary" in {
        roles_by_id[artifact_id] for artifact_id in receipt.verified_artifact_ids
    }
    assert all(
        roles_by_id[artifact_id] not in {"sealed-labels", "sealed-inputs"}
        for artifact_id in receipt.verified_artifact_ids
    )

    admission_path = tmp_path / "online-custody-admission.json"
    write_online_custody_admission_receipt(receipt, admission_path)
    assert load_online_custody_admission_receipt(admission_path) == receipt


def test_online_admission_rejects_mutated_ciphertext(
    tmp_path: Path,
) -> None:
    package = _materialize_admission_package(tmp_path)
    manifest = package["manifest"]
    artifact_root = package["artifact_root"]
    relative_by_id = package["relative_by_id"]
    ciphertext = _artifact_for(
        manifest,
        "sealed-label-ciphertext",
        corpus_id=FIXED_CORPORA[0],
    )
    (artifact_root / relative_by_id[str(ciphertext["id"])]).write_bytes(b"mutated")

    with pytest.raises(CustodyError, match="revalidation failed"):
        admit_online_custody(
            package["manifest_path"],
            custody_seal_receipt_path=package["seal_path"],
            sealed_run_receipt_path=package["run_receipt_path"],
            artifact_verification_receipt_path=package["full_verification_path"],
            artifact_root=artifact_root,
            local_artifact_map_path=package["artifact_map_path"],
            runner_identity=_RUNNER_IDENTITY,
        )


def test_online_selection_excludes_plaintext_roles_before_artifact_open(
    tmp_path: Path,
) -> None:
    package = _materialize_admission_package(tmp_path)
    specs = online_custody_artifact_specs(
        package["manifest"],
        package["artifact_map_path"],
    )
    roles_by_id = {
        str(artifact["id"]): str(artifact["role"]) for artifact in package["manifest"]["artifacts"]
    }
    assert specs
    assert all(roles_by_id[spec.artifact_id] in ONLINE_CUSTODY_REVALIDATION_ROLES for spec in specs)
    assert "sealed-labels" not in {roles_by_id[spec.artifact_id] for spec in specs}


@pytest.mark.parametrize("case", ("omission", "extra"))
def test_online_selection_rejects_artifact_map_closure_attacks(
    tmp_path: Path,
    case: str,
) -> None:
    package = _materialize_admission_package(tmp_path)
    map_path = package["artifact_map_path"]
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    rows = payload["artifacts"]
    if case == "omission":
        selected_ids = {
            str(row["id"])
            for row in package["manifest"]["artifacts"]
            if row["role"] in ONLINE_CUSTODY_REVALIDATION_ROLES
        }
        rows[:] = [row for row in rows if row["artifact_id"] not in selected_ids]
    else:
        rows.append(
            {
                "artifact_id": "unregistered-online-artifact",
                "kind": "file",
                "relative_path": "objects/unregistered.bin",
            }
        )
    map_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CustodyError, match="invalid local artifact map"):
        online_custody_artifact_specs(package["manifest"], map_path)


def test_online_selection_rejects_wrong_corpus_role_coverage(tmp_path: Path) -> None:
    package = _materialize_admission_package(tmp_path)
    manifest = deepcopy(package["manifest"])
    row = _artifact_for(
        manifest,
        "authorized-index-store",
        corpus_id="scifact",
    )
    row["corpus_id"] = "hotpotqa-fullwiki"

    with pytest.raises(CustodyError, match="invalid frozen study manifest"):
        online_custody_artifact_specs(
            manifest,
            package["artifact_map_path"],
        )


def test_online_admission_receipt_has_a_closed_sorted_schema() -> None:
    payload = {
        "artifact_verification_receipt_sha256": "1" * 64,
        "custody_seal_receipt_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "online_artifact_verification_receipt_sha256": "4" * 64,
        "run_receipt_sha256": "5" * 64,
        "runner_identity": _RUNNER_IDENTITY,
        "schema_version": "fractal-online-custody-admission-v1",
        "verified_artifact_ids": ["a", "b"],
    }
    assert OnlineCustodyAdmissionReceipt.from_dict(payload).verified_artifact_ids == (
        "a",
        "b",
    )
    payload["verified_artifact_ids"] = ["b", "a"]
    with pytest.raises(CustodyError, match="bytewise sorted"):
        OnlineCustodyAdmissionReceipt.from_dict(payload)
