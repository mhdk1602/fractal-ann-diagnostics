from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fractal_ann_diagnostics.artifact_integrity import (
    ArtifactVerificationReceipt,
    VerifiedArtifact,
    digest_directory_tree,
    digest_regular_file,
)
from fractal_ann_diagnostics.opa_runtime_binary import (
    load_runtime_attestation_plan_template,
)
from fractal_ann_diagnostics.production_controls import (
    _PLACEHOLDER_RETENTION_PREFIX,
    C0_INSTANTIATION_RECEIPT_FILENAME,
    PLAN_TEMPLATE_FILENAME,
    PREFLIGHT_CONTRACT_FILENAME,
    PRODUCTION_HARDWARE_FRAGMENT_FILENAME,
    PRODUCTION_WORKLOADS_FRAGMENT_FILENAME,
    ProductionControlError,
    ProductionControlFinalizationRequest,
    ProductionControlMaterializationConfig,
    ProductionControlMaterializationConfigWriteReceipt,
    _AdmittedFactory,
    _atomic_exchange_directories,
    _atomic_publish_file_noreplace,
    _CorpusSources,
    _derive_required_artifact_binding_suite_context,
    _finalization_lock_path,
    _FinalizationContext,
    _FinalizationCorpus,
    _load_closed_required_artifact_binding_suite,
    _load_finalization_context,
    _manifest_workload_spec,
    _open_exchange_directory,
    _parser,
    _production_finalization_lock,
    _rederive_blueprint_launch_contract,
    _verified_hardware_observation,
    _verified_manifest_fragment_document,
    _verify_blueprint_authority_header,
    _verify_blueprint_manifest_fragments,
    _verify_c0_manifest_runtime,
    _verify_c1_production_control_bindings,
    finalize_production_controls,
    instantiate_c0_production_controls,
    load_production_control_c0_instantiation_receipt,
    load_production_control_config,
    load_production_control_config_write_receipt,
    load_production_control_finalization_receipt,
    load_production_control_finalization_request,
    materialize_production_control_blueprint,
    write_production_control_finalization_request,
    write_production_control_materialization_config,
    write_required_artifact_binding_suite,
)
from fractal_ann_diagnostics.production_controls import (
    main as production_controls_main,
)
from fractal_ann_diagnostics.production_corpus_run import (
    PRODUCTION_CORPUS_CONFIG_FILENAME,
    PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME,
    SHARDED_EXECUTION_PLAN_FILENAME,
    TRIAL_RUNTIME_RECEIPT_FILENAME,
    ProductionCorpusWorkloadSpec,
)
from fractal_ann_diagnostics.production_workload_registration import (
    production_workload_file_sha256,
)
from fractal_ann_diagnostics.runtime_attestation import RuntimeFilePin
from fractal_ann_diagnostics.sealed_container_launcher import (
    load_preflight_launch_contract,
)
from fractal_ann_diagnostics.sealed_orchestrator import RequiredArtifactIdBindings
from fractal_ann_diagnostics.study import C0_COMMIT_SENTINEL, FIXED_CORPORA
from fractal_ann_diagnostics.trial_runtime import RuntimeFeatureBinding

SHA = "a" * 64
IMAGE = f"ghcr.io/mhdk1602/fractal-ann-diagnostics@sha256:{'b' * 64}"
PRODUCTION_IMAGE = f"ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@sha256:{'b' * 64}"
C0 = "c" * 40
P = "d" * 40
APPROVAL_ENVIRONMENT = "confirmatory"
RUNNER_IDENTITY = "github-actions:environment:confirmatory"


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _directory(path: Path, name: str) -> tuple[Path, str]:
    path.mkdir(mode=0o700, parents=True)
    _write(path / name, name.encode("ascii"))
    return path, digest_directory_tree(path).sha256


def _writer_arguments(tmp_path: Path) -> dict[str, Path]:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    config_path = control / "materialization.json"
    _write(config_path, b"{}\n")
    closure = tmp_path / "production-run-closure"
    closure.mkdir(mode=0o700)
    blueprint = tmp_path / "blueprint"
    blueprint.mkdir(mode=0o700)
    return {
        "materialization_config_path": config_path,
        "c0_control_instantiation_receipt_path": (
            tmp_path / "c0-instantiated-controls" / C0_INSTANTIATION_RECEIPT_FILENAME
        ),
        "frozen_manifest_path": control / "study-manifest.json",
        "manifest_lock_path": control / "study-manifest.sha256",
        "c1_package_root": tmp_path / "c1-package",
        "protocol_registry_record_path": control / "registry.json",
        "protocol_registration_receipt_path": control / "registration.json",
        "online_custody_admission_path": control / "online-admission.json",
        "custody_seal_receipt_path": control / "custody-seal.json",
        "artifact_verification_receipt_path": control / "verification.json",
        "artifact_root": tmp_path / "artifacts",
        "local_artifact_map_path": control / "artifact-map.json",
        "required_artifact_bindings_root": tmp_path / "required-bindings",
        "runtime_evidence_root": tmp_path / "runtime-evidence",
        "output": control / "finalization-request.json",
        "blueprint_root": blueprint,
        "blueprint_receipt_path": blueprint / "production-control-blueprint-receipt.json",
        "finalized_controls_root": closure,
    }


def _patch_finalization_request_authorities(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, Path],
    *,
    manifest_digest: str = "d" * 64,
    context_error: str | None = None,
) -> tuple[SimpleNamespace, list[ProductionControlFinalizationRequest]]:
    config_digest = hashlib.sha256(b"{}\n").hexdigest()
    materialization = SimpleNamespace(
        blueprint_receipt_path=arguments["blueprint_receipt_path"],
        blueprint_root=arguments["blueprint_root"],
        file_sha256=config_digest,
        finalized_controls_root=arguments["finalized_controls_root"],
    )

    def load_config(_path: Path, *, expected_sha256: str) -> SimpleNamespace:
        assert expected_sha256 == config_digest
        return materialization

    observed: list[ProductionControlFinalizationRequest] = []

    def load_context(request: ProductionControlFinalizationRequest) -> SimpleNamespace:
        observed.append(request)
        if context_error is not None:
            raise ProductionControlError(context_error)
        return SimpleNamespace(
            materialization=materialization,
            instantiation=SimpleNamespace(
                instantiated_root=str(arguments["c0_control_instantiation_receipt_path"].parent)
            ),
        )

    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_production_control_config",
        load_config,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_production_control_blueprint_receipt",
        lambda _path: SimpleNamespace(
            file_sha256="b" * 64,
            semantic_sha256="e" * 64,
            receipt_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_study_manifest",
        lambda _path: {
            "sealed_execution": {
                "production_controls": {
                    "blueprint_receipt_file_sha256": "b" * 64,
                    "blueprint_receipt_sha256": "e" * 64,
                    "materialization_config_file_sha256": config_digest,
                }
            },
            "status": "frozen",
        },
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.validate_study_manifest",
        lambda _manifest, *, require_frozen: require_frozen,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.manifest_sha256",
        lambda _manifest: manifest_digest,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._load_finalization_context",
        load_context,
    )
    return materialization, observed


def _public_writer_arguments(arguments: dict[str, Path]) -> dict[str, Path]:
    internal = {"blueprint_root", "blueprint_receipt_path", "finalized_controls_root"}
    return {key: value for key, value in arguments.items() if key not in internal}


def test_finalization_request_writer_derives_pins_and_is_accepted_by_finalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _writer_arguments(tmp_path)
    materialization, observed = _patch_finalization_request_authorities(
        monkeypatch,
        arguments,
    )
    request = write_production_control_finalization_request(**_public_writer_arguments(arguments))

    expected_config_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    assert request.materialization_config_sha256 == expected_config_sha256
    assert request.blueprint_receipt_path == materialization.blueprint_receipt_path
    assert request.blueprint_receipt_sha256 == "b" * 64
    assert request.sealed_run_receipt_path == (
        materialization.finalized_controls_root / f"{'d' * 64}.json"
    )
    assert (
        load_production_control_finalization_request(
            arguments["output"],
            expected_sha256=request.file_sha256,
        )
        == request
    )
    assert arguments["output"].stat().st_mode & 0o777 == 0o600
    assert observed == [request, request]

    finalized = SimpleNamespace(receipt_sha256="f" * 64)

    def finish(
        request_path: Path,
        *,
        expected_request_sha256: str,
        finalization_receipt_path: Path,
        resume: bool,
    ) -> SimpleNamespace:
        assert request_path == arguments["output"]
        assert expected_request_sha256 == request.file_sha256
        assert finalization_receipt_path == tmp_path / "finalization-receipt.json"
        assert resume is False
        return finalized

    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._finalize_production_controls_unlocked",
        finish,
    )
    assert (
        finalize_production_controls(
            arguments["output"],
            expected_request_sha256=request.file_sha256,
            finalization_receipt_path=tmp_path / "finalization-receipt.json",
        )
        is finalized
    )


def test_finalization_request_writer_fails_before_publication_on_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _writer_arguments(tmp_path)
    _patch_finalization_request_authorities(
        monkeypatch,
        arguments,
        context_error="authority drift",
    )

    with pytest.raises(ProductionControlError, match="authority drift"):
        write_production_control_finalization_request(**_public_writer_arguments(arguments))
    assert not arguments["output"].exists()


@pytest.mark.parametrize(
    "field",
    (
        "materialization_config_file_sha256",
        "blueprint_receipt_sha256",
        "blueprint_receipt_file_sha256",
    ),
)
def test_finalization_request_writer_requires_each_public_c1_control_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    arguments = _writer_arguments(tmp_path)
    _patch_finalization_request_authorities(monkeypatch, arguments)
    config_digest = hashlib.sha256(b"{}\n").hexdigest()
    bindings = {
        "blueprint_receipt_file_sha256": "b" * 64,
        "blueprint_receipt_sha256": "e" * 64,
        "materialization_config_file_sha256": config_digest,
    }
    bindings[field] = "f" * 64
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_study_manifest",
        lambda _path: {
            "sealed_execution": {"production_controls": bindings},
            "status": "frozen",
        },
    )

    with pytest.raises(ProductionControlError, match="public C1 pins"):
        write_production_control_finalization_request(**_public_writer_arguments(arguments))

    assert not arguments["output"].exists()


def test_finalization_context_rejects_self_consistent_local_substitution_before_factory_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _writer_arguments(tmp_path)
    config_digest = hashlib.sha256(b"{}\n").hexdigest()
    materialization = SimpleNamespace(
        blueprint_receipt_path=arguments["blueprint_receipt_path"],
        blueprint_root=arguments["blueprint_root"],
        file_sha256=config_digest,
        finalized_controls_root=arguments["finalized_controls_root"],
    )
    blueprint = SimpleNamespace(
        file_sha256="b" * 64,
        semantic_sha256="e" * 64,
        materialization_config_sha256=config_digest,
        payload_tree_sha256="a" * 64,
        provisional_closure_root=str(arguments["finalized_controls_root"]),
    )
    request = ProductionControlFinalizationRequest(
        materialization_config_path=arguments["materialization_config_path"],
        materialization_config_sha256=config_digest,
        blueprint_receipt_path=arguments["blueprint_receipt_path"],
        blueprint_receipt_sha256=blueprint.file_sha256,
        c0_control_instantiation_receipt_path=arguments["c0_control_instantiation_receipt_path"],
        frozen_manifest_path=arguments["frozen_manifest_path"],
        manifest_lock_path=arguments["manifest_lock_path"],
        c1_package_root=arguments["c1_package_root"],
        protocol_registry_record_path=arguments["protocol_registry_record_path"],
        protocol_registration_receipt_path=arguments["protocol_registration_receipt_path"],
        sealed_run_receipt_path=arguments["finalized_controls_root"] / f"{'d' * 64}.json",
        online_custody_admission_path=arguments["online_custody_admission_path"],
        custody_seal_receipt_path=arguments["custody_seal_receipt_path"],
        artifact_verification_receipt_path=arguments["artifact_verification_receipt_path"],
        artifact_root=arguments["artifact_root"],
        local_artifact_map_path=arguments["local_artifact_map_path"],
        required_artifact_bindings_root=arguments["required_artifact_bindings_root"],
        runtime_evidence_root=arguments["runtime_evidence_root"],
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_production_control_config",
        lambda _path, *, expected_sha256: materialization,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_production_control_blueprint_receipt",
        lambda _path, *, expected_sha256: blueprint,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._scan_exact_tree",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._payload_tree_sha256",
        lambda *_args, **_kwargs: blueprint.payload_tree_sha256,
    )
    registration = SimpleNamespace(
        package_root=request.c1_package_root,
        assert_current=lambda: None,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.verify_production_protocol_registration",
        lambda *_args, **_kwargs: registration,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_study_manifest",
        lambda _path: {
            "sealed_execution": {
                "production_controls": {
                    "blueprint_receipt_file_sha256": blueprint.file_sha256,
                    "blueprint_receipt_sha256": blueprint.semantic_sha256,
                    "materialization_config_file_sha256": "f" * 64,
                }
            }
        },
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.validate_study_manifest",
        lambda *_args, **_kwargs: None,
    )
    factory_accessed = False

    def admit_factory(_materialization: object) -> object:
        nonlocal factory_accessed
        factory_accessed = True
        raise AssertionError("factory must not be opened before C1 pin admission")

    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._admit_factory",
        admit_factory,
    )

    with pytest.raises(ProductionControlError, match="public C1 pins"):
        _load_finalization_context(request)

    assert factory_accessed is False


def test_finalization_request_writer_never_replaces_an_existing_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _writer_arguments(tmp_path)
    _patch_finalization_request_authorities(monkeypatch, arguments)
    first = write_production_control_finalization_request(**_public_writer_arguments(arguments))
    before = arguments["output"].read_bytes()

    with pytest.raises(ProductionControlError, match="already exists"):
        write_production_control_finalization_request(**_public_writer_arguments(arguments))
    assert arguments["output"].read_bytes() == before == first.canonical_file_bytes()
    assert not tuple(arguments["output"].parent.glob(".finalization-request.json.tmp-*"))


def test_finalization_request_writer_rejects_publication_inside_an_admitted_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _writer_arguments(tmp_path)
    _patch_finalization_request_authorities(monkeypatch, arguments)
    arguments["output"] = arguments["artifact_root"] / "finalization-request.json"

    with pytest.raises(ProductionControlError, match="immutable tree"):
        write_production_control_finalization_request(**_public_writer_arguments(arguments))


def test_finalization_request_cli_accepts_no_caller_supplied_digest() -> None:
    parser = _parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    request_parser = subparsers.choices["write-finalization-request"]
    options = set(request_parser._option_string_actions)
    assert "--materialization-config-sha256" not in options
    assert "--blueprint-receipt-sha256" not in options
    assert "--manifest-sha256" not in options
    assert "--request-sha256" not in options


def _config_writer_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    root = tmp_path.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    control = root / "config-control"
    control.mkdir(mode=0o700)
    artifact_root = root / "factory-artifacts"
    artifact_root.mkdir(mode=0o700)
    embedding_source_root = root / "embedding-source"
    embedding_source_root.mkdir(mode=0o700)
    _write(artifact_root / "artifact.bin", b"factory-artifact")
    _write(embedding_source_root / "embedding.bin", b"embedding")
    factory_config_path = control / "factory-config.json"
    factory_suite_path = artifact_root / "production-artifact-factory-suite-receipt.json"
    extraction_path = control / "c0-runtime-extraction-receipt.json"
    opa_path = control / "opa"
    uv_lock_path = control / "uv.lock"
    pseudonym_key_path = control / "pseudonym.key"
    for path, encoded in (
        (factory_config_path, b'{"factory":"typed"}\n'),
        (factory_suite_path, b'{"suite":"typed"}\n'),
        (extraction_path, b'{"c0":"typed"}\n'),
        (opa_path, b"opa-runtime"),
        (uv_lock_path, b"uv-lock"),
        (pseudonym_key_path, b"k" * 32),
    ):
        _write(path, encoded)
    key_sha256 = digest_regular_file(pseudonym_key_path)
    factory = SimpleNamespace(
        artifact_root=artifact_root,
        embedding_source_root=embedding_source_root,
        suite_receipt_path=factory_suite_path,
        runner_image=IMAGE,
        runner_platform="linux/arm64",
        hmac_secret_sha256=key_sha256,
        hmac_key_id=f"sealed-online-ephemeral-sha256-{key_sha256}",
    )
    suite = SimpleNamespace(
        runner_image=IMAGE,
        runner_platform="linux/arm64",
        hmac_secret_sha256=key_sha256,
        hmac_key_id=f"sealed-online-ephemeral-sha256-{key_sha256}",
    )
    extraction = SimpleNamespace(
        c0_sha=P,
        image_reference=IMAGE,
        platform="linux/arm64",
        opa_sha256=digest_regular_file(opa_path),
        opa_byte_count=opa_path.stat().st_size,
        uv_lock_sha256=digest_regular_file(uv_lock_path),
        uv_lock_byte_count=uv_lock_path.stat().st_size,
    )
    admitted = _AdmittedFactory(
        config=factory,  # type: ignore[arg-type]
        suite=suite,  # type: ignore[arg-type]
        extraction=extraction,  # type: ignore[arg-type]
        staged_root=embedding_source_root,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_production_artifact_factory_config",
        lambda _path, *, expected_sha256: factory,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.verify_production_artifact_factory",
        lambda _factory: suite,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_production_artifact_factory_suite",
        lambda _path, *, expected_sha256: suite,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_c0_runtime_extraction_receipt",
        lambda _path: extraction,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._admit_factory",
        lambda _config: admitted,
    )
    arguments: dict[str, object] = {
        "factory_config_path": factory_config_path,
        "c0_runtime_extraction_receipt_path": extraction_path,
        "opa_binary_path": opa_path,
        "uv_lock_path": uv_lock_path,
        "pseudonym_key_path": pseudonym_key_path,
        "scientific_candidate_reference": IMAGE,
        "scientific_production_reference": PRODUCTION_IMAGE,
        "approval_environment": APPROVAL_ENVIRONMENT,
        "runner_platform": "linux/arm64",
        "runner_identity": RUNNER_IDENTITY,
        "hostname": "sealed-runner",
        "hardware_provider": "aws",
        "hardware_instance_type": "c7g.2xlarge",
        "hardware_cpu_model": "AWS Graviton3",
        "hardware_accelerator": "none",
        "hardware_region": "us-east-1",
        "hardware_operating_system": "ubuntu-24.04",
        "memory_limit_bytes": 8 * 1024**3,
        "cpuset_cpus": (0, 1),
        "tmpfs_size_bytes": 1024 * 1024,
        "blueprint_root": root / "blueprint",
        "finalized_controls_root": root / "finalized-controls",
        "suite_base_root": root / "suite-base",
        "output": control / "production-control-materialization.json",
        "receipt_output": control / "production-control-materialization.write-receipt.json",
    }
    return arguments, factory, suite, extraction


def _config_writer_cli_arguments(arguments: dict[str, object]) -> list[str]:
    return [
        "--factory-config",
        str(arguments["factory_config_path"]),
        "--c0-runtime-extraction-receipt",
        str(arguments["c0_runtime_extraction_receipt_path"]),
        "--opa-binary",
        str(arguments["opa_binary_path"]),
        "--uv-lock",
        str(arguments["uv_lock_path"]),
        "--pseudonym-key",
        str(arguments["pseudonym_key_path"]),
        "--scientific-candidate-reference",
        str(arguments["scientific_candidate_reference"]),
        "--scientific-production-reference",
        str(arguments["scientific_production_reference"]),
        "--approval-environment",
        str(arguments["approval_environment"]),
        "--runner-platform",
        str(arguments["runner_platform"]),
        "--runner-identity",
        str(arguments["runner_identity"]),
        "--hostname",
        str(arguments["hostname"]),
        "--hardware-provider",
        str(arguments["hardware_provider"]),
        "--hardware-instance-type",
        str(arguments["hardware_instance_type"]),
        "--hardware-cpu-model",
        str(arguments["hardware_cpu_model"]),
        "--hardware-accelerator",
        str(arguments["hardware_accelerator"]),
        "--hardware-region",
        str(arguments["hardware_region"]),
        "--hardware-operating-system",
        str(arguments["hardware_operating_system"]),
        "--memory-limit-bytes",
        str(arguments["memory_limit_bytes"]),
        "--cpuset-cpus",
        ",".join(str(cpu) for cpu in arguments["cpuset_cpus"]),  # type: ignore[union-attr]
        "--tmpfs-size-bytes",
        str(arguments["tmpfs_size_bytes"]),
        "--blueprint-root",
        str(arguments["blueprint_root"]),
        "--finalized-controls-root",
        str(arguments["finalized_controls_root"]),
        "--suite-base-root",
        str(arguments["suite_base_root"]),
        "--output",
        str(arguments["output"]),
        "--receipt",
        str(arguments["receipt_output"]),
    ]


def test_write_config_derives_pins_and_publishes_private_canonical_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, factory, _, _ = _config_writer_arguments(tmp_path, monkeypatch)
    receipt = write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]

    assert isinstance(receipt, ProductionControlMaterializationConfigWriteReceipt)
    output = arguments["output"]
    receipt_output = arguments["receipt_output"]
    assert isinstance(output, Path) and isinstance(receipt_output, Path)
    assert output.stat().st_mode & 0o777 == 0o600
    assert receipt_output.stat().st_mode & 0o777 == 0o600
    config = load_production_control_config(
        output,
        expected_sha256=receipt.config_file_sha256,
    )
    assert config.factory_config_sha256 == digest_regular_file(
        arguments["factory_config_path"]  # type: ignore[arg-type]
    )
    assert config.factory_suite_receipt_sha256 == digest_regular_file(factory.suite_receipt_path)
    assert (
        config.factory_artifact_tree_sha256 == digest_directory_tree(factory.artifact_root).sha256
    )
    assert config.opa_binary_sha256 == digest_regular_file(
        arguments["opa_binary_path"]  # type: ignore[arg-type]
    )
    assert config.uv_lock_sha256 == digest_regular_file(
        arguments["uv_lock_path"]  # type: ignore[arg-type]
    )
    assert config.pseudonym_key_sha256 == digest_regular_file(
        arguments["pseudonym_key_path"]  # type: ignore[arg-type]
    )
    assert config.scientific_candidate_reference == IMAGE
    assert config.scientific_production_reference == PRODUCTION_IMAGE
    assert config.scientific_index_digest == f"sha256:{'b' * 64}"
    assert config.oci_promotion_required is True
    assert config.approval_environment == APPROVAL_ENVIRONMENT
    assert config.candidate_image_source_commit == P
    assert receipt.scientific_candidate_reference == IMAGE
    assert receipt.scientific_production_reference == PRODUCTION_IMAGE
    assert receipt.scientific_index_digest == f"sha256:{'b' * 64}"
    assert receipt.oci_promotion_required is True
    assert receipt.approval_environment == APPROVAL_ENVIRONMENT
    assert receipt.candidate_image_source_commit == P
    assert load_production_control_config_write_receipt(receipt_output) == receipt


def test_write_config_cli_wires_closed_arguments_and_reports_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments, _, _, _ = _config_writer_arguments(tmp_path, monkeypatch)
    assert production_controls_main(["write-config", *_config_writer_cli_arguments(arguments)]) == 0
    result = json.loads(capsys.readouterr().out)
    output = arguments["output"]
    receipt_output = arguments["receipt_output"]
    assert isinstance(output, Path) and isinstance(receipt_output, Path)
    assert result == {
        "approval_environment": APPROVAL_ENVIRONMENT,
        "config_file_sha256": digest_regular_file(output),
        "config_path": str(output),
        "oci_promotion_required": True,
        "receipt_file_sha256": digest_regular_file(receipt_output),
        "receipt_path": str(receipt_output),
        "scientific_index_digest": f"sha256:{'b' * 64}",
        "status": "written",
    }


def test_write_config_parser_is_closed_and_accepts_no_digest_pin() -> None:
    parser = _parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    write_config = subparsers.choices["write-config"]
    options = set(write_config._option_string_actions)
    assert (
        not {
            "--factory-config-sha256",
            "--factory-suite-receipt-sha256",
            "--factory-artifact-tree-sha256",
            "--c0-runtime-extraction-receipt-sha256",
            "--opa-binary-sha256",
            "--uv-lock-sha256",
            "--pseudonym-key-sha256",
        }
        & options
    )
    assert {
        "--factory-config",
        "--c0-runtime-extraction-receipt",
        "--scientific-candidate-reference",
        "--scientific-production-reference",
        "--approval-environment",
        "--runner-platform",
        "--cpuset-cpus",
        "--blueprint-root",
        "--finalized-controls-root",
        "--suite-base-root",
        "--output",
        "--receipt",
    } <= options
    assert "--runner-image" not in options


def test_write_config_subprocess_exposes_closed_operator_interface() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fractal_ann_diagnostics.production_controls",
            "write-config",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--factory-config FACTORY_CONFIG" in result.stdout
    assert "--scientific-candidate-reference" in result.stdout
    assert "--scientific-production-reference" in result.stdout
    assert "--approval-environment" in result.stdout
    assert "--cpuset-cpus CPUSET_CPUS" in result.stdout
    assert "--factory-config-sha256" not in result.stdout
    assert "--factory-artifact-tree-sha256" not in result.stdout


@pytest.mark.parametrize("cpus", ("1,0", "0,01", "0,0", "0-1", " 0,1", ""))
def test_write_config_rejects_noncanonical_cpu_lists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpus: str,
) -> None:
    arguments, _, _, _ = _config_writer_arguments(tmp_path, monkeypatch)
    arguments["cpuset_cpus"] = cpus
    with pytest.raises(ProductionControlError, match="cpuset_cpus"):
        write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("approval_environment", "staging", "approval_environment"),
        ("runner_identity", "confirmatory-runner", "runner_identity"),
    ),
)
def test_write_config_rejects_unbound_approval_environment_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    arguments, _, _, _ = _config_writer_arguments(tmp_path, monkeypatch)
    arguments[field] = value
    with pytest.raises(ProductionControlError, match=message):
        write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("existing", ("output", "receipt_output"))
def test_write_config_rejects_preexisting_publication_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: str,
) -> None:
    arguments, _, _, _ = _config_writer_arguments(tmp_path, monkeypatch)
    target = arguments[existing]
    assert isinstance(target, Path)
    _write(target, b"preexisting")
    with pytest.raises(
        ProductionControlError,
        match="exact recoverable publication|exists without",
    ):
        write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]
    assert target.read_bytes() == b"preexisting"


def test_write_config_recovers_exact_config_after_receipt_publication_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _, _, _ = _config_writer_arguments(tmp_path, monkeypatch)
    failed = False

    def fail_receipt_once(path: Path, encoded: bytes, *, label: str) -> None:
        nonlocal failed
        if label == "materialization config write receipt" and not failed:
            failed = True
            raise ProductionControlError("simulated receipt publication crash")
        _atomic_publish_file_noreplace(path, encoded, label=label)

    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._atomic_publish_file_noreplace",
        fail_receipt_once,
    )
    with pytest.raises(ProductionControlError, match="simulated receipt publication crash"):
        write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]
    assert Path(arguments["output"]).is_file()  # type: ignore[arg-type]
    assert not Path(arguments["receipt_output"]).exists()  # type: ignore[arg-type]

    receipt = write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]
    assert load_production_control_config_write_receipt(arguments["receipt_output"]) == receipt


def test_write_config_idempotently_recovers_exact_published_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _, _, _ = _config_writer_arguments(tmp_path, monkeypatch)
    first = write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]
    second = write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]
    assert second == first


def test_write_config_rejects_overlapping_or_symlinked_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _, _, _ = _config_writer_arguments(tmp_path, monkeypatch)
    arguments["finalized_controls_root"] = (
        Path(arguments["blueprint_root"]) / "nested"  # type: ignore[arg-type]
    )
    with pytest.raises(ProductionControlError, match="roots overlap"):
        write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]

    arguments, _, _, _ = _config_writer_arguments(tmp_path / "symlink-case", monkeypatch)
    real_root = (tmp_path / "symlink-case" / "real-blueprint").resolve()
    real_root.mkdir(mode=0o700)
    linked_root = (tmp_path / "symlink-case" / "linked-blueprint").resolve()
    linked_root.symlink_to(real_root, target_is_directory=True)
    arguments["blueprint_root"] = linked_root
    with pytest.raises(ProductionControlError, match="symlink"):
        write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]


def test_write_config_rejects_symlinked_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _, _, _ = _config_writer_arguments(tmp_path, monkeypatch)
    source = arguments["pseudonym_key_path"]
    assert isinstance(source, Path)
    linked = source.with_name("linked-pseudonym.key")
    linked.symlink_to(source)
    arguments["pseudonym_key_path"] = linked
    with pytest.raises(ProductionControlError, match="pseudonym key"):
        write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "scientific_candidate_reference",
            f"ghcr.io/mhdk1602/other@sha256:{'b' * 64}",
        ),
        ("runner_platform", "linux/amd64"),
    ),
)
def test_write_config_rejects_runner_image_or_platform_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    arguments, _, _, _ = _config_writer_arguments(tmp_path, monkeypatch)
    arguments[field] = value
    with pytest.raises(ProductionControlError, match="differs from the verified factory"):
        write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("production_reference", "error"),
    (
        (
            f"ghcr.io/mhdk1602/fractal-ann-diagnostics@sha256:{'d' * 64}",
            "share scientific_index_digest",
        ),
        (IMAGE, "must be distinct"),
    ),
)
def test_write_config_rejects_digest_mismatch_or_candidate_locator_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    production_reference: str,
    error: str,
) -> None:
    arguments, _, _, _ = _config_writer_arguments(tmp_path, monkeypatch)
    arguments["scientific_production_reference"] = production_reference
    with pytest.raises(ProductionControlError, match=error):
        write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]


def test_write_config_rejects_mismatched_factory_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, factory, _, extraction = _config_writer_arguments(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._admit_factory",
        lambda _config: _AdmittedFactory(
            config=factory,  # type: ignore[arg-type]
            suite=SimpleNamespace(receipt="different"),  # type: ignore[arg-type]
            extraction=extraction,  # type: ignore[arg-type]
            staged_root=factory.embedding_source_root,
        ),
    )
    with pytest.raises(ProductionControlError, match="factory suite receipt differs"):
        write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]


def test_write_config_rehash_detects_source_mutation_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, factory, suite, extraction = _config_writer_arguments(tmp_path, monkeypatch)
    opa_path = arguments["opa_binary_path"]
    assert isinstance(opa_path, Path)

    def mutate_after_admission(config: ProductionControlMaterializationConfig) -> _AdmittedFactory:
        opa_path.write_bytes(b"X" * extraction.opa_byte_count)
        return _AdmittedFactory(
            config=factory,  # type: ignore[arg-type]
            suite=suite,  # type: ignore[arg-type]
            extraction=extraction,  # type: ignore[arg-type]
            staged_root=factory.embedding_source_root,
        )

    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._admit_factory",
        mutate_after_admission,
    )
    with pytest.raises(ProductionControlError, match="changed before config publication"):
        write_production_control_materialization_config(**arguments)  # type: ignore[arg-type]
    assert not Path(arguments["output"]).exists()  # type: ignore[arg-type]
    assert not Path(arguments["receipt_output"]).exists()  # type: ignore[arg-type]


def _required_binding_rows() -> tuple[RequiredArtifactIdBindings, ...]:
    artifact_sha256 = "9" * 64
    verification = ArtifactVerificationReceipt(
        manifest_sha256="8" * 64,
        artifacts=(
            VerifiedArtifact(
                artifact_id="verified-artifact",
                relative_path="objects/verified.bin",
                kind="file",
                exact=True,
                expected_sha256=artifact_sha256,
                verified_sha256=artifact_sha256,
                file_count=1,
                directory_count=0,
                byte_count=1,
                observed_file_count=1,
                observed_directory_count=0,
                observed_byte_count=1,
            ),
        ),
    )
    return tuple(
        RequiredArtifactIdBindings(
            verification_receipt=verification,
            execution_artifact_id=f"{corpus_id}-execution",
            execution_revision_sha256=f"{position + 1:x}" * 64,
            runner_artifact_ids=("runner",),
            source_artifact_ids=(f"{corpus_id}-source",),
            retriever_artifact_ids=(f"{corpus_id}-application",),
            provenance_component_artifact_ids=(("application", f"{corpus_id}-application"),),
        )
        for position, corpus_id in enumerate(FIXED_CORPORA)
    )


def test_required_binding_authority_rechecks_factory_workloads_and_local_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _required_binding_rows()
    config_bytes = b"materialization-config\n"
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    materialization = SimpleNamespace(
        c0_runtime_extraction_receipt_sha256="1" * 64,
        candidate_image_source_commit=P,
        factory_artifact_tree_sha256="2" * 64,
        file_sha256=config_sha256,
        blueprint_receipt_path=tmp_path / "blueprint" / "receipt.json",
        blueprint_root=tmp_path / "blueprint",
        finalized_controls_root=tmp_path / "production-run-closure",
        approval_environment=APPROVAL_ENVIRONMENT,
        runner_identity=RUNNER_IDENTITY,
        scientific_candidate_reference=IMAGE,
        scientific_production_reference=PRODUCTION_IMAGE,
        scientific_index_digest=f"sha256:{'b' * 64}",
        oci_promotion_required=True,
        runner_platform="linux/arm64",
    )
    specs = {
        corpus_id: SimpleNamespace(
            file_sha256="6" * 64,
            canonical_file_bytes=lambda: b"workload-spec\n",
        )
        for corpus_id in FIXED_CORPORA
    }
    blueprint = SimpleNamespace(
        approval_environment=APPROVAL_ENVIRONMENT,
        materialization_config_sha256=config_sha256,
        c0_runtime_extraction_receipt_sha256="1" * 64,
        candidate_image_source_commit=P,
        factory_artifact_tree_sha256="2" * 64,
        factory_config_sha256="3" * 64,
        factory_suite_receipt_sha256="8" * 64,
        provisional_closure_root=str(materialization.finalized_controls_root),
        payload_tree_sha256="5" * 64,
        file_sha256="4" * 64,
        semantic_sha256="7" * 64,
        runner_image=PRODUCTION_IMAGE,
        runner_platform="linux/arm64",
        receipt_sha256="4" * 64,
        workloads=tuple(
            SimpleNamespace(
                corpus_id=corpus_id,
                relative_path=f"{corpus_id}/production-corpus-workload-spec.json",
                file_sha256="6" * 64,
            )
            for corpus_id in FIXED_CORPORA
        ),
    )
    admitted = SimpleNamespace(
        extraction=SimpleNamespace(c0_sha=P),
        config=SimpleNamespace(
            file_sha256="3" * 64,
            selected_family_count=25,
            artifact_root=tmp_path / "factory-artifacts",
        ),
        suite=SimpleNamespace(receipt_sha256="8" * 64),
    )
    manifest = {
        "analysis": {"power": {"selected_families_per_corpus": 25}},
        "artifacts": [{"id": "verified-artifact", "sha256": "9" * 64}],
        "sealed_execution": {
            "code_commit": C0,
            "production_controls": {
                "blueprint_receipt_file_sha256": "4" * 64,
                "blueprint_receipt_sha256": "7" * 64,
                "materialization_config_file_sha256": config_sha256,
            },
            "runner_identity": RUNNER_IDENTITY,
            "runner_image": PRODUCTION_IMAGE,
        },
        "status": "frozen",
    }
    verification = bindings[0].verification_receipt
    local_specs = (object(),)
    calls: list[str] = []
    instantiation = SimpleNamespace(
        instantiated_root=str(tmp_path / "c0-instantiated-controls"),
        workloads=blueprint.workloads,
    )

    def read(path: Path, *, label: str) -> bytes:
        if path == tmp_path / "materialization.json":
            return config_bytes
        assert label.endswith(("blueprint workload", "A-bound workload"))
        return b"workload-spec\n"

    def verify_local(
        root: Path,
        *,
        manifest_sha256: str,
        artifacts: tuple[object, ...],
    ) -> ArtifactVerificationReceipt:
        calls.append("fresh-artifact-verification")
        assert root == tmp_path / "artifacts"
        assert manifest_sha256 == verification.manifest_sha256
        assert artifacts is local_specs
        return verification

    monkeypatch.setattr("fractal_ann_diagnostics.production_controls._read", read)
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_production_control_config",
        lambda _path, *, expected_sha256: (
            materialization if expected_sha256 == config_sha256 else None
        ),
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_production_control_blueprint_receipt",
        lambda _path: blueprint,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._scan_exact_tree",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._payload_tree_sha256",
        lambda *_args, **_kwargs: "5" * 64,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._admit_factory",
        lambda _materialization: admitted,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_study_manifest",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.validate_study_manifest",
        lambda _manifest, *, require_frozen: require_frozen,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._verify_blueprint_manifest_fragments",
        lambda *_args: tuple(
            {"canonical_file_sha256": "6" * 64, "corpus_id": corpus_id}
            for corpus_id in FIXED_CORPORA
        ),
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._verify_c0_manifest_runtime",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._admit_c0_control_instantiation",
        lambda **_kwargs: instantiation,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._derive_workload_spec",
        lambda _materialization, _admitted, corpus_id, **_kwargs: specs[corpus_id],
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._manifest_workload_spec",
        lambda _manifest, corpus_id: specs[corpus_id],
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._candidate_blueprint_workload_spec",
        lambda _encoded, *, binding: specs[binding.corpus_id],
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._resolve_candidate_workload_spec",
        lambda candidate, *, apparatus_commit: candidate,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_verification_receipt",
        lambda _path: verification,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_local_artifact_map",
        lambda _path, *, expected_sha256_by_id: (
            local_specs if expected_sha256_by_id == {"verified-artifact": "9" * 64} else ()
        ),
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.verify_local_artifacts",
        verify_local,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.manifest_sha256",
        lambda _manifest: verification.manifest_sha256,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.derive_required_artifact_id_bindings",
        lambda _manifest, _verification, *, corpus_id: bindings[FIXED_CORPORA.index(corpus_id)],
    )

    context = _derive_required_artifact_binding_suite_context(
        materialization_config_path=tmp_path / "materialization.json",
        c0_control_instantiation_receipt_path=tmp_path / "instantiation-receipt.json",
        frozen_manifest_path=tmp_path / "manifest.json",
        artifact_verification_receipt_path=tmp_path / "verification.json",
        artifact_root=tmp_path / "artifacts",
        local_artifact_map_path=tmp_path / "artifact-map.json",
    )
    assert context.bindings == bindings
    assert context.factory_artifact_root == admitted.config.artifact_root
    assert calls == ["fresh-artifact-verification"]

    stale_artifact = replace(
        verification.artifacts[0],
        expected_sha256="3" * 64,
        verified_sha256="3" * 64,
    )
    stale = ArtifactVerificationReceipt(
        manifest_sha256=verification.manifest_sha256,
        artifacts=(stale_artifact,),
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.verify_local_artifacts",
        lambda *_args, **_kwargs: stale,
    )
    with pytest.raises(ProductionControlError, match="fresh local artifact verification"):
        _derive_required_artifact_binding_suite_context(
            materialization_config_path=tmp_path / "materialization.json",
            c0_control_instantiation_receipt_path=(tmp_path / "instantiation-receipt.json"),
            frozen_manifest_path=tmp_path / "manifest.json",
            artifact_verification_receipt_path=tmp_path / "verification.json",
            artifact_root=tmp_path / "artifacts",
            local_artifact_map_path=tmp_path / "artifact-map.json",
        )


def _binding_suite_arguments(tmp_path: Path) -> dict[str, Path]:
    control = tmp_path / "binding-control"
    control.mkdir(mode=0o700)
    return {
        "materialization_config_path": control / "materialization.json",
        "c0_control_instantiation_receipt_path": (
            control / "c0-control-instantiation-receipt.json"
        ),
        "frozen_manifest_path": control / "study-manifest.json",
        "artifact_verification_receipt_path": control / "verification.json",
        "artifact_root": tmp_path / "artifact-root",
        "local_artifact_map_path": control / "artifact-map.json",
        "output_root": control / "required-artifact-bindings",
    }


def _patch_required_binding_suite_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bindings: tuple[RequiredArtifactIdBindings, ...],
) -> SimpleNamespace:
    context = SimpleNamespace(
        materialization=SimpleNamespace(
            blueprint_root=tmp_path / "blueprint",
            finalized_controls_root=tmp_path / "production-run-closure",
        ),
        blueprint_receipt_sha256="7" * 64,
        instantiation=SimpleNamespace(instantiated_root=str(tmp_path / "c0-instantiated-controls")),
        factory_artifact_root=tmp_path / "factory-artifacts",
        manifest={"status": "frozen"},
        verification_receipt=bindings[0].verification_receipt,
        bindings=bindings,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls."
        "_derive_required_artifact_binding_suite_context",
        lambda **_kwargs: context,
    )
    return context


def test_required_artifact_binding_suite_is_one_atomic_fixed_corpus_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _binding_suite_arguments(tmp_path)
    bindings = _required_binding_rows()
    _patch_required_binding_suite_context(monkeypatch, tmp_path, bindings)
    fsynced: list[Path] = []
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._fsync_private_directory",
        lambda path, *, label: fsynced.append(path),
    )

    assert write_required_artifact_binding_suite(**arguments) == bindings
    output = arguments["output_root"]
    assert output.stat().st_mode & 0o777 == 0o700
    assert _load_closed_required_artifact_binding_suite(output, expected=bindings) == bindings
    assert frozenset(digest_directory_tree(output).entries) == frozenset(
        entry
        for corpus_id in FIXED_CORPORA
        for entry in (
            corpus_id,
            f"{corpus_id}/required-artifact-bindings.json",
        )
    )
    staging_roots = {path for path in fsynced if path.name.startswith(".required-")}
    assert len(staging_roots) == 1
    staging_root = next(iter(staging_roots))
    assert set(fsynced) == {
        staging_root,
        *(staging_root / corpus_id for corpus_id in FIXED_CORPORA),
    }

    before = digest_directory_tree(output)
    with pytest.raises(ProductionControlError, match="already exists"):
        write_required_artifact_binding_suite(**arguments)
    assert digest_directory_tree(output) == before
    assert not tuple(output.parent.glob(".required-artifact-bindings.tmp-*"))


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "duplicate", "substitution", "hardlink"],
)
def test_required_artifact_binding_suite_rejects_hostile_membership_and_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    arguments = _binding_suite_arguments(tmp_path)
    bindings = _required_binding_rows()
    _patch_required_binding_suite_context(monkeypatch, tmp_path, bindings)
    write_required_artifact_binding_suite(**arguments)
    output = arguments["output_root"]
    first = output / FIXED_CORPORA[0] / "required-artifact-bindings.json"
    second = output / FIXED_CORPORA[1] / "required-artifact-bindings.json"
    if mutation == "missing":
        first.unlink()
    elif mutation == "extra":
        (output / "unregistered-corpus").mkdir(mode=0o700)
    elif mutation == "duplicate":
        (first.parent / "required-artifact-bindings.copy.json").write_bytes(first.read_bytes())
    elif mutation == "substitution":
        first.unlink()
        first.write_bytes(bindings[1].canonical_file_bytes())
    else:
        first.unlink()
        first.hardlink_to(second)

    with pytest.raises(ProductionControlError, match="required-artifact binding suite"):
        _load_closed_required_artifact_binding_suite(output, expected=bindings)


def test_required_artifact_binding_suite_leaves_no_partial_root_before_atomic_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _binding_suite_arguments(tmp_path)
    bindings = _required_binding_rows()
    _patch_required_binding_suite_context(monkeypatch, tmp_path, bindings)
    from fractal_ann_diagnostics import production_controls as controls

    real_rename = controls._rename_noreplace_at

    def fail_suite_rename(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
        *,
        label: str,
    ) -> None:
        if label == "required-artifact binding suite":
            raise ProductionControlError("injected suite rename failure")
        real_rename(
            parent_descriptor,
            source_name,
            destination_name,
            label=label,
        )

    monkeypatch.setattr(controls, "_rename_noreplace_at", fail_suite_rename)
    with pytest.raises(ProductionControlError, match="injected"):
        write_required_artifact_binding_suite(**arguments)
    assert not arguments["output_root"].exists()
    assert not tuple(arguments["output_root"].parent.glob(".required-artifact-bindings.tmp-*"))


def test_required_artifact_binding_cli_has_no_partial_suite_or_id_override() -> None:
    parser = _parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    binding_parser = subparsers.choices["write-required-artifact-bindings"]
    options = set(binding_parser._option_string_actions)
    assert "--corpus" not in options
    assert "--corpus-id" not in options
    assert "--artifact-id" not in options
    assert "--manifest-sha256" not in options
    assert "--verification-receipt-sha256" not in options


def test_production_finalization_lock_rejects_duplicate_and_releases(
    tmp_path: Path,
) -> None:
    closure = tmp_path / "production-run-closure"
    closure.mkdir(mode=0o700)
    lock_path = _finalization_lock_path(closure)

    with _production_finalization_lock(closure):
        assert lock_path.is_file()
        assert lock_path.stat().st_mode & 0o777 == 0o600
        with pytest.raises(ProductionControlError, match="live worker"):
            with _production_finalization_lock(closure):
                pass

    with _production_finalization_lock(closure):
        pass


@pytest.mark.parametrize("mutation", ["mode", "hardlink", "symlink"])
def test_production_finalization_lock_rejects_unsafe_path(
    tmp_path: Path,
    mutation: str,
) -> None:
    closure = tmp_path / "production-run-closure"
    closure.mkdir(mode=0o700)
    lock_path = _finalization_lock_path(closure)
    source = tmp_path / "hostile-lock-source"
    source.write_bytes(b"")
    source.chmod(0o600)
    if mutation == "mode":
        lock_path.write_bytes(b"")
        lock_path.chmod(0o640)
    elif mutation == "hardlink":
        lock_path.hardlink_to(source)
    else:
        lock_path.symlink_to(source)

    with pytest.raises(ProductionControlError, match="lock"):
        with _production_finalization_lock(closure):
            pass


def _workload_spec(
    corpus_id: str,
    *,
    online: tuple[Path, str],
    index: tuple[Path, str],
    embedding: tuple[Path, str],
    policy: tuple[Path, str],
    query: tuple[Path, str],
    staged: tuple[Path, str],
    audit_path: Path,
    key_path: Path,
    policy_bundle: Path,
    index_bundle: Path,
    execution_plan: Path,
    runtime_receipt: Path,
) -> ProductionCorpusWorkloadSpec:
    return ProductionCorpusWorkloadSpec(
        corpus_id=corpus_id,
        available_family_count=75,
        selected_family_count=25,
        factory_config_sha256="1" * 64,
        factory_suite_receipt_sha256="2" * 64,
        factory_artifact_tree_sha256="3" * 64,
        runner_image=PRODUCTION_IMAGE,
        runner_platform="linux/arm64",
        runner_identity=RUNNER_IDENTITY,
        code_commit=C0_COMMIT_SENTINEL,
        artifact_root=Path("/input/online"),
        artifact_tree_sha256=online[1],
        authorized_index_store_root=Path("/input/index"),
        authorized_index_store_tree_sha256=index[1],
        embedding_store_root=Path("/input/embedding"),
        embedding_store_tree_sha256=embedding[1],
        partition_audit_path=Path("/input/partition-audit.json"),
        partition_audit_file_sha256=digest_regular_file(audit_path),
        partition_audit_sha256="4" * 64,
        policy_intervention_root=Path("/input/policy"),
        policy_intervention_tree_sha256=policy[1],
        pseudonym_key_path=Path("/run/secrets/audit-pseudonym.key"),
        expected_pseudonym_key_sha256=digest_regular_file(key_path),
        query_package_root=Path("/input/query-package"),
        query_package_tree_sha256=query[1],
        staged_root=Path("/input/staged"),
        staged_tree_sha256=staged[1],
        expected_authorized_index_store_receipt_sha256="5" * 64,
        expected_policy_intervention_receipt_sha256="6" * 64,
        policy_bundle_receipt_sha256=digest_regular_file(policy_bundle),
        index_bundle_receipt_sha256=digest_regular_file(index_bundle),
        policy_bundle_receipt_path=Path("/input/bundles/policy-stage-bundle.json"),
        index_bundle_receipt_path=Path("/input/bundles/index-stage-bundle.json"),
        query_receipt_sha256="7" * 64,
        online_execution_plan_sha256="8" * 64,
        online_execution_tree_sha256=online[1],
        sharded_execution_plan_file_sha256=digest_regular_file(execution_plan),
        trial_runtime_admission_receipt_file_sha256=(digest_regular_file(runtime_receipt)),
        feature_bindings=(
            RuntimeFeatureBinding(
                group_order=0,
                subject="confirmatory-reader",
                repetition=0,
                policy_state="baseline",
                version_lag=1.0,
                backend="hnsw",
                drift_family="qwen-revision-lag",
                policy_complexity=0.5,
            ),
        ),
    )


def test_blueprint_materializes_five_contracts_on_one_empty_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    online = _directory(source / "online", "online.bin")
    index = _directory(source / "index", "index.bin")
    embedding = _directory(source / "embedding", "embedding.bin")
    policy = _directory(source / "policy", "policy.json")
    query = _directory(source / "query", "query.jsonl")
    staged = _directory(source / "staged", "staged.jsonl")
    audit_path = source / "partition-audit.json"
    key_path = source / "pseudonym.key"
    opa_path = source / "opa"
    uv_path = source / "uv.lock"
    policy_bundle = source / "policy-stage-bundle.json"
    index_bundle = source / "index-stage-bundle.json"
    execution_plan = online[0] / "sharded-online-execution-plan.json"
    runtime_receipt = source / "runtime-receipt.json"
    for path, value in (
        (audit_path, b"audit"),
        (key_path, b"k" * 32),
        (opa_path, b"opa"),
        (uv_path, b"uv"),
        (policy_bundle, b"policy-bundle"),
        (index_bundle, b"index-bundle"),
        (execution_plan, b"execution"),
        (runtime_receipt, b"runtime"),
    ):
        _write(path, value)
    online = (online[0], digest_directory_tree(online[0]).sha256)

    blueprint_root = tmp_path / "blueprint"
    closure_root = tmp_path / "production-run-closure"
    suite_base = tmp_path / "suite-base"
    config = ProductionControlMaterializationConfig(
        factory_config_path=source / "factory-config.json",
        factory_config_sha256="1" * 64,
        factory_suite_receipt_path=source / "factory-suite.json",
        factory_suite_receipt_sha256="2" * 64,
        factory_artifact_tree_sha256="3" * 64,
        c0_runtime_extraction_receipt_path=source / "c0-extraction.json",
        c0_runtime_extraction_receipt_sha256="4" * 64,
        candidate_image_source_commit=P,
        opa_binary_path=opa_path,
        opa_binary_sha256=digest_regular_file(opa_path),
        uv_lock_path=uv_path,
        uv_lock_sha256=digest_regular_file(uv_path),
        pseudonym_key_path=key_path,
        pseudonym_key_sha256=digest_regular_file(key_path),
        scientific_candidate_reference=IMAGE,
        scientific_production_reference=PRODUCTION_IMAGE,
        scientific_index_digest=f"sha256:{'b' * 64}",
        oci_promotion_required=True,
        approval_environment=APPROVAL_ENVIRONMENT,
        runner_platform="linux/arm64",
        runner_identity=RUNNER_IDENTITY,
        hostname="sealed-runner",
        hardware_provider="aws",
        hardware_instance_type="c7g.2xlarge",
        hardware_cpu_model="AWS Graviton3",
        hardware_accelerator="none",
        hardware_region="us-east-1",
        hardware_operating_system="ubuntu-24.04",
        memory_limit_bytes=8 * 1024 * 1024 * 1024,
        cpuset_cpus=(0, 1),
        tmpfs_size_bytes=1024 * 1024,
        blueprint_root=blueprint_root,
        finalized_controls_root=closure_root,
        suite_base_root=suite_base,
    )
    config_path = tmp_path / "materialization.json"
    _write(config_path, config.canonical_file_bytes())

    extraction = SimpleNamespace(
        c0_sha=P,
        opa_image_path="/usr/local/bin/opa",
        opa_sha256=digest_regular_file(opa_path),
        python_binary_image_path="/opt/venv/bin/python",
        python_binary_sha256="9" * 64,
        uv_lock_image_path="/opt/app/uv.lock",
        uv_lock_sha256=digest_regular_file(uv_path),
    )
    factory = SimpleNamespace(
        artifact_root=source,
        partition_audit_path=audit_path,
        file_sha256="1" * 64,
        selected_family_count=25,
    )
    admitted = _AdmittedFactory(
        config=factory,  # type: ignore[arg-type]
        suite=SimpleNamespace(receipt_sha256="2" * 64),  # type: ignore[arg-type]
        extraction=extraction,  # type: ignore[arg-type]
        staged_root=staged[0],
    )
    specs = {
        corpus_id: _workload_spec(
            corpus_id,
            online=online,
            index=index,
            embedding=embedding,
            policy=policy,
            query=query,
            staged=staged,
            audit_path=audit_path,
            key_path=key_path,
            policy_bundle=policy_bundle,
            index_bundle=index_bundle,
            execution_plan=execution_plan,
            runtime_receipt=runtime_receipt,
        )
        for corpus_id in FIXED_CORPORA
    }
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._admit_factory",
        lambda _: admitted,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._derive_workload_spec",
        lambda _materialization, _admitted, corpus_id, *, code_commit=C0_COMMIT_SENTINEL: replace(
            specs[corpus_id], code_commit=code_commit
        ),
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._corpus_sources",
        lambda _admitted, _corpus_id: _CorpusSources(
            online_root=online[0],
            index_root=index[0],
            embedding_root=embedding[0],
            policy_root=policy[0],
            query_root=query[0],
            execution_plan_path=execution_plan,
            trial_runtime_receipt_path=runtime_receipt,
            policy_bundle_receipt_path=policy_bundle,
            index_bundle_receipt_path=index_bundle,
        ),
    )

    receipt = materialize_production_control_blueprint(
        config_path,
        expected_config_sha256=config.file_sha256,
    )
    assert receipt.semantic_sha256 == hashlib.sha256(receipt.canonical_bytes()).hexdigest()
    assert receipt.file_sha256 == digest_regular_file(config.blueprint_receipt_path)
    assert receipt.semantic_sha256 != receipt.file_sha256
    assert receipt.approval_environment == APPROVAL_ENVIRONMENT
    _verify_blueprint_authority_header(config, receipt, admitted)
    header_substitutions = (
        {"c0_runtime_extraction_receipt_sha256": "d" * 64},
        {"candidate_image_source_commit": "e" * 40},
        {"factory_artifact_tree_sha256": "d" * 64},
        {"factory_config_sha256": "d" * 64},
        {"factory_suite_receipt_sha256": "d" * 64},
        {"materialization_config_sha256": "d" * 64},
        {"provisional_closure_root": str(tmp_path / "alternate-closure")},
        {"runner_image": f"ghcr.io/mhdk1602/fractal-ann-diagnostics@sha256:{'d' * 64}"},
    )
    for substitution in header_substitutions:
        with pytest.raises(ProductionControlError, match="authority header"):
            _verify_blueprint_authority_header(
                config,
                replace(receipt, **substitution),
                admitted,
            )
    with pytest.raises(ProductionControlError, match="runner platform"):
        replace(receipt, runner_platform="linux/amd64")

    placeholder = digest_directory_tree(closure_root)
    assert placeholder.entries == ()
    assert placeholder.sha256 == receipt.provisional_closure_tree_sha256
    assert tuple(row.corpus_id for row in receipt.workloads) == FIXED_CORPORA
    workload_fragment = json.loads(
        (blueprint_root / PRODUCTION_WORKLOADS_FRAGMENT_FILENAME).read_bytes()
    )
    assert [row["corpus_id"] for row in workload_fragment] == list(FIXED_CORPORA)
    assert [row["canonical_file_sha256"] for row in workload_fragment] == [
        specs[corpus_id].file_sha256 for corpus_id in FIXED_CORPORA
    ]
    assert receipt.production_workloads_fragment_file_sha256 == digest_regular_file(
        blueprint_root / PRODUCTION_WORKLOADS_FRAGMENT_FILENAME
    )
    frozen_workload_fragment = copy.deepcopy(workload_fragment)
    for row in frozen_workload_fragment:
        row["spec"]["code_commit"] = C0
        row["canonical_file_sha256"] = production_workload_file_sha256(row["spec"])
    hardware_fragment = json.loads(
        (blueprint_root / PRODUCTION_HARDWARE_FRAGMENT_FILENAME).read_bytes()
    )
    assert hardware_fragment == {
        "accelerator": "none",
        "cpu_model": "AWS Graviton3",
        "instance_type": "c7g.2xlarge",
        "logical_cores": 2,
        "memory_gib": 8,
        "operating_system": "ubuntu-24.04",
        "provider": "aws",
        "region": "us-east-1",
    }
    manifest = {
        "analysis": {"power": {"selected_families_per_corpus": 25}},
        "production_workloads": frozen_workload_fragment,
        "sealed_execution": {
            "code_commit": C0,
            "hardware": hardware_fragment,
            "production_controls": {
                "blueprint_receipt_file_sha256": receipt.file_sha256,
                "blueprint_receipt_sha256": receipt.semantic_sha256,
                "materialization_config_file_sha256": config.file_sha256,
            },
            "runner_identity": RUNNER_IDENTITY,
            "runner_image": PRODUCTION_IMAGE,
        },
    }
    assert (
        tuple(
            row["corpus_id"]
            for row in _verify_blueprint_manifest_fragments(config, receipt, manifest)
        )
        == FIXED_CORPORA
    )
    printed_fragment, printed_config = _verified_manifest_fragment_document(
        config_path,
        expected_config_sha256=config.file_sha256,
        expected_blueprint_receipt_file_sha256=receipt.file_sha256,
    )
    assert printed_config == config
    assert json.loads(printed_fragment) == {
        "production_workloads": workload_fragment,
        "sealed_execution": {
            "hardware": hardware_fragment,
            "production_controls": {
                "blueprint_receipt_file_sha256": receipt.file_sha256,
                "blueprint_receipt_sha256": receipt.semantic_sha256,
                "materialization_config_file_sha256": config.file_sha256,
            },
        },
    }

    raw_blueprint = digest_directory_tree(blueprint_root)
    candidate_manifest = {
        "analysis": {"power": {"selected_families_per_corpus": 25}},
        "production_workloads": workload_fragment,
        "sealed_execution": {
            "code_commit": C0_COMMIT_SENTINEL,
            "hardware": hardware_fragment,
            "production_controls": {
                "blueprint_receipt_file_sha256": receipt.file_sha256,
                "blueprint_receipt_sha256": receipt.semantic_sha256,
                "materialization_config_file_sha256": config.file_sha256,
            },
            "runner_identity": RUNNER_IDENTITY,
            "runner_image": PRODUCTION_IMAGE,
        },
    }
    candidate_path = tmp_path / "candidate-manifest.json"
    candidate_bytes = (
        json.dumps(
            candidate_manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    _write(candidate_path, candidate_bytes)
    candidate_image_closure_path = tmp_path / "candidate-image-closure.json"
    _write(candidate_image_closure_path, b"candidate-image-closure\n")
    candidate_image_closure = SimpleNamespace(
        bootstrap_closure_sha256="a" * 64,
        build_context_tree_sha256="b" * 64,
        file_sha256="c" * 64,
        github_sha=P,
        release_image_index_digest=f"sha256:{'e' * 64}",
        scientific_image_index_digest=config.scientific_index_digest,
        scientific_image_reference=config.scientific_candidate_reference,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.validate_candidate_rehearsal_manifest",
        lambda _manifest, *, c0_commit: None,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.CandidateImageClosure",
        SimpleNamespace(from_file=lambda _path: candidate_image_closure),
    )
    candidate_package = SimpleNamespace(
        manifest=candidate_manifest,
        manifest_bytes=candidate_bytes,
        receipt=SimpleNamespace(file_sha256="d" * 64),
        receipt_bytes=b"candidate assembly receipt\n",
    )

    def load_candidate_package(path: Path) -> SimpleNamespace:
        if Path(path) == candidate_path:
            return candidate_package
        snapshot = Path(path)
        return SimpleNamespace(
            manifest_bytes=(snapshot / "candidate-study-manifest.json").read_bytes(),
            receipt_bytes=(snapshot / "candidate-manifest-assembly-receipt.json").read_bytes(),
        )

    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.load_closed_candidate_manifest_package",
        load_candidate_package,
    )
    instantiated_root = tmp_path / "c0-instantiated-controls"
    instantiated = instantiate_c0_production_controls(
        materialization_config_path=config_path,
        candidate_manifest_package_path=candidate_path,
        candidate_image_closure_path=candidate_image_closure_path,
        apparatus_commit=C0,
        output_root=instantiated_root,
    )
    assert instantiated.apparatus_commit == C0
    assert instantiated.approval_environment == APPROVAL_ENVIRONMENT
    assert instantiated.candidate_image_source_commit == P
    assert instantiated.apparatus_commit != instantiated.candidate_image_source_commit
    assert (
        instantiated.candidate_manifest_file_sha256 == hashlib.sha256(candidate_bytes).hexdigest()
    )
    assert (
        instantiated.candidate_manifest_assembly_receipt_file_sha256
        == hashlib.sha256(candidate_package.receipt_bytes).hexdigest()
    )
    assert (
        instantiated_root / instantiated.candidate_manifest_relative_path
    ).read_bytes() == candidate_bytes
    assert (
        instantiated_root / instantiated.candidate_manifest_assembly_receipt_relative_path
    ).read_bytes() == candidate_package.receipt_bytes
    assert (
        load_production_control_c0_instantiation_receipt(
            instantiated_root / C0_INSTANTIATION_RECEIPT_FILENAME,
            expected_sha256=instantiated.file_sha256,
        )
        == instantiated
    )
    assert digest_directory_tree(blueprint_root) == raw_blueprint
    assert C0_COMMIT_SENTINEL.encode("ascii") in candidate_bytes
    raw_plan = load_runtime_attestation_plan_template(
        blueprint_root / FIXED_CORPORA[0] / "launcher-control" / PLAN_TEMPLATE_FILENAME
    )
    instantiated_plan = load_runtime_attestation_plan_template(
        instantiated_root / FIXED_CORPORA[0] / "launcher-control" / PLAN_TEMPLATE_FILENAME
    )
    assert raw_plan.code_commit == C0_COMMIT_SENTINEL
    assert instantiated_plan.code_commit == C0
    instantiated_spec = ProductionCorpusWorkloadSpec.from_dict(
        json.loads(
            (
                instantiated_root / FIXED_CORPORA[0] / PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME
            ).read_bytes()
        )
    )
    assert instantiated_spec.code_commit == C0
    assert instantiated.workloads[0].file_sha256 == instantiated_spec.file_sha256

    failed_root = tmp_path / "failed-c0-instantiated-controls"

    def reject_instantiation_publish(
        _directory_fd: int,
        _source_name: str,
        _destination_name: str,
        *,
        label: str,
    ) -> None:
        raise ProductionControlError(f"cannot publish {label}")

    with monkeypatch.context() as publish_patch:
        publish_patch.setattr(
            "fractal_ann_diagnostics.production_controls._rename_noreplace_at",
            reject_instantiation_publish,
        )
        with pytest.raises(ProductionControlError, match="C0 instantiated controls"):
            instantiate_c0_production_controls(
                materialization_config_path=config_path,
                candidate_manifest_package_path=candidate_path,
                candidate_image_closure_path=candidate_image_closure_path,
                apparatus_commit=C0,
                output_root=failed_root,
            )
    assert not failed_root.exists()
    assert not tuple(tmp_path.glob(f".{failed_root.name}.tmp-*"))

    hostile_manifests = []
    omitted = copy.deepcopy(manifest)
    del omitted["production_workloads"]
    hostile_manifests.append(omitted)
    reordered = copy.deepcopy(manifest)
    reordered["production_workloads"] = list(  # type: ignore[index]
        reversed(reordered["production_workloads"])  # type: ignore[arg-type,index]
    )
    hostile_manifests.append(reordered)
    substituted_spec = copy.deepcopy(manifest)
    substituted_spec["production_workloads"][0]["spec"][  # type: ignore[index]
        "query_receipt_sha256"
    ] = "f" * 64
    hostile_manifests.append(substituted_spec)
    substituted_hash = copy.deepcopy(manifest)
    substituted_hash["production_workloads"][0][  # type: ignore[index]
        "canonical_file_sha256"
    ] = "e" * 64
    hostile_manifests.append(substituted_hash)
    for hostile in hostile_manifests:
        with pytest.raises(ProductionControlError, match="production workloads differ"):
            _verify_blueprint_manifest_fragments(config, receipt, hostile)

    numeric_alias = copy.deepcopy(manifest)
    numeric_alias["sealed_execution"]["hardware"]["memory_gib"] = 8.0  # type: ignore[index]
    with pytest.raises(ProductionControlError, match="public C1 hardware differs"):
        _verify_blueprint_manifest_fragments(config, receipt, numeric_alias)

    malformed = copy.deepcopy(manifest)
    malformed["production_workloads"][0]["spec"][  # type: ignore[index]
        "feature_bindings"
    ] = "not-an-array"
    with pytest.raises(ProductionControlError, match="public C1 production workloads"):
        _manifest_workload_spec(malformed, FIXED_CORPORA[0])
    with pytest.raises(ProductionControlError, match="public C1 pins"):
        _verify_blueprint_manifest_fragments(
            replace(config, hardware_provider="gcp"),
            receipt,
            manifest,
        )

    observation = {
        "architecture": "arm64",
        "cpu_model": "AWS Graviton3",
        "logical_cpu_count": 2,
        "memory_limit_bytes": 8 * 1024 * 1024 * 1024,
        "operating_system_id": "ubuntu",
        "operating_system_version_id": "24.04",
    }
    assert _verified_hardware_observation(
        config,
        SimpleNamespace(**observation),  # type: ignore[arg-type]
        SimpleNamespace(**observation),  # type: ignore[arg-type]
    ) == (
        "AWS Graviton3",
        2,
        8 * 1024 * 1024 * 1024,
        "ubuntu-24.04",
        "arm64",
    )
    drift = {**observation, "cpu_model": "substituted CPU"}
    with pytest.raises(ProductionControlError, match="pre-C1 claims"):
        _verified_hardware_observation(
            config,
            SimpleNamespace(**drift),  # type: ignore[arg-type]
            SimpleNamespace(**drift),  # type: ignore[arg-type]
        )
    for row in receipt.workloads:
        corpus_root = blueprint_root / row.corpus_id
        contract = load_preflight_launch_contract(corpus_root / PREFLIGHT_CONTRACT_FILENAME)
        closure_mount = contract.geometry.production_run_closure_mount
        assert closure_mount.source == closure_mount.target == str(closure_root)
        assert closure_mount.content_sha256 == placeholder.sha256
        assert digest_directory_tree(corpus_root / "launcher-control").entries == (
            PLAN_TEMPLATE_FILENAME,
        )
        plan = load_runtime_attestation_plan_template(
            corpus_root / "launcher-control" / PLAN_TEMPLATE_FILENAME
        )
        assert plan.workload_sha256 == specs[row.corpus_id].file_sha256
        assert plan.python_binary == RuntimeFilePin(
            path="/opt/venv/bin/python",
            sha256="9" * 64,
        )
        assert plan.uv_lock == RuntimeFilePin(
            path="/opt/app/uv.lock",
            sha256=digest_regular_file(uv_path),
        )
        assert f".pre-c1-output-{config.file_sha256[:20]}" in (contract.geometry.copy_output_root)
        assert (corpus_root / PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME).read_bytes() == (
            specs[row.corpus_id].canonical_file_bytes()
        )
        rederived_contract, rederived_plan = _rederive_blueprint_launch_contract(
            config,
            admitted,
            row.corpus_id,
            specs[row.corpus_id],
            row,
            receipt,
        )
        assert rederived_contract == contract
        assert rederived_plan == plan

    first_binding = receipt.workloads[0]
    first_corpus = first_binding.corpus_id
    first_contract_path = blueprint_root / first_corpus / PREFLIGHT_CONTRACT_FILENAME
    original_contract_bytes = first_contract_path.read_bytes()
    original_contract_row = json.loads(original_contract_bytes)
    geometry_fields = set(original_contract_row["geometry"])
    geometry_mutations = {
        "bind_mounts": lambda row: next(
            mount for mount in row["bind_mounts"] if mount["role"] == "runtime-control-tree"
        ).__setitem__("source", str(tmp_path / "alternate-control-root")),
        "code_commit": lambda row: row.__setitem__("code_commit", "d" * 40),
        "control_mount_target": lambda row: row.__setitem__(
            "control_mount_target", "/input/alternate-control"
        ),
        "copy_output_root": lambda row: row.__setitem__(
            "copy_output_root", str(tmp_path / "alternate" / "online" / first_corpus)
        ),
        "corpus_id": lambda row: row.__setitem__("corpus_id", "alternate-corpus"),
        "cpuset_cpus": lambda row: row.__setitem__("cpuset_cpus", [0, 2]),
        "environment": lambda row: row["environment"][0].__setitem__("value", "substituted"),
        "gid": lambda row: row.__setitem__("gid", 65531),
        "hostname": lambda row: row.__setitem__("hostname", "alternate-runner"),
        "memory_limit_bytes": lambda row: row.__setitem__(
            "memory_limit_bytes", row["memory_limit_bytes"] + 1
        ),
        "oci_image_digest": lambda row: row.__setitem__(
            "oci_image_digest",
            f"ghcr.io/mhdk1602/fractal-ann-diagnostics@sha256:{'d' * 64}",
        ),
        "output_root": lambda row: row.__setitem__("output_root", "/alternate-output"),
        "output_volume": lambda row: row.__setitem__("output_volume", "alternate-volume"),
        "output_volume_subpath": lambda row: row.__setitem__(
            "output_volume_subpath", f"alternate/{first_corpus}"
        ),
        "platform": lambda row: row.__setitem__("platform", "linux/amd64"),
        "runtime_plan_template_relative_path": lambda row: row.__setitem__(
            "runtime_plan_template_relative_path", "alternate-plan.json"
        ),
        "tmpfs_flags": lambda row: row.__setitem__("tmpfs_flags", ["nodev", "nosuid", "noexec"]),
        "tmpfs_mode": lambda row: row.__setitem__("tmpfs_mode", 0o700),
        "tmpfs_root": lambda row: row.__setitem__("tmpfs_root", "/alternate-tmp"),
        "tmpfs_size_bytes": lambda row: row.__setitem__(
            "tmpfs_size_bytes", row["tmpfs_size_bytes"] + 1
        ),
        "uid": lambda row: row.__setitem__("uid", 65531),
    }
    assert set(geometry_mutations) == geometry_fields
    for field, mutate in geometry_mutations.items():
        hostile = copy.deepcopy(original_contract_row)
        mutate(hostile["geometry"])
        first_contract_path.write_bytes(
            json.dumps(
                hostile,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        with pytest.raises(ProductionControlError, match=first_corpus):
            _rederive_blueprint_launch_contract(
                config,
                admitted,
                first_corpus,
                specs[first_corpus],
                first_binding,
                receipt,
            )
        first_contract_path.write_bytes(original_contract_bytes)

    contract_fields = set(original_contract_row) - {"geometry"}
    contract_mutations = {
        "argv": lambda row: row.__setitem__("argv", [*row["argv"][:-1], "substituted-preflight"]),
        "provisional_control_tree_sha256": lambda row: row.__setitem__(
            "provisional_control_tree_sha256", "d" * 64
        ),
        "provisional_plan_template_file_sha256": lambda row: row.__setitem__(
            "provisional_plan_template_file_sha256", "e" * 64
        ),
        "schema_version": lambda row: row.__setitem__(
            "schema_version", "fractal-preflight-launch-contract-substituted"
        ),
    }
    assert set(contract_mutations) == contract_fields
    for field, mutate in contract_mutations.items():
        hostile = copy.deepcopy(original_contract_row)
        mutate(hostile)
        first_contract_path.write_bytes(
            json.dumps(
                hostile,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        with pytest.raises(ProductionControlError, match=first_corpus):
            _rederive_blueprint_launch_contract(
                config,
                admitted,
                first_corpus,
                specs[first_corpus],
                first_binding,
                receipt,
            )
        first_contract_path.write_bytes(original_contract_bytes)

    alternate_candidate_image = f"ghcr.io/mhdk1602/fractal-ann-diagnostics@sha256:{'d' * 64}"
    alternate_production_image = (
        f"ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@sha256:{'d' * 64}"
    )
    config_only_substitution = replace(config, hostname="alternate-runner")
    blueprint_only_substitution = replace(receipt, runner_image=alternate_production_image)
    for hostile_config, hostile_blueprint in (
        (config_only_substitution, receipt),
        (config, blueprint_only_substitution),
    ):
        with pytest.raises(ProductionControlError, match="public C1 pins"):
            _verify_c1_production_control_bindings(
                hostile_config,
                hostile_blueprint,
                manifest,
            )

    substitute_config = replace(
        config,
        scientific_candidate_reference=alternate_candidate_image,
        scientific_production_reference=alternate_production_image,
        scientific_index_digest=f"sha256:{'d' * 64}",
        blueprint_root=tmp_path / "substitute-blueprint",
        finalized_controls_root=tmp_path / "substitute-production-run-closure",
        suite_base_root=tmp_path / "substitute-suite-base",
    )
    substitute_config_path = tmp_path / "substitute-materialization.json"
    _write(substitute_config_path, substitute_config.canonical_file_bytes())
    substitute_specs = {
        corpus_id: replace(spec, runner_image=alternate_production_image)
        for corpus_id, spec in specs.items()
    }
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._derive_workload_spec",
        lambda _materialization, _admitted, corpus_id, **_kwargs: substitute_specs[corpus_id],
    )
    substitute_blueprint = materialize_production_control_blueprint(
        substitute_config_path,
        expected_config_sha256=substitute_config.file_sha256,
    )
    assert substitute_blueprint.materialization_config_sha256 == substitute_config.file_sha256
    assert substitute_blueprint.runner_image == substitute_config.scientific_production_reference
    with pytest.raises(ProductionControlError, match="public C1 pins"):
        _verify_c1_production_control_bindings(
            substitute_config,
            substitute_blueprint,
            manifest,
        )


def test_atomic_exchange_retains_the_receipt_only_tree(tmp_path: Path) -> None:
    closure = tmp_path / "closure"
    staged = tmp_path / "retained"
    closure.mkdir(mode=0o700)
    staged.mkdir(mode=0o700)
    _write(closure / f"{'d' * 64}.json", b"sealed")
    _write(staged / "scifact" / "control" / "config.json", b"final")
    before = digest_directory_tree(closure)
    final = digest_directory_tree(staged)
    before_closure_identity = (closure.stat().st_dev, closure.stat().st_ino)
    before_staged_identity = (staged.stat().st_dev, staged.stat().st_ino)

    _atomic_exchange_directories(closure, staged)

    assert digest_directory_tree(closure) == final
    assert digest_directory_tree(staged) == before
    assert (closure.stat().st_dev, closure.stat().st_ino) == before_staged_identity
    assert (staged.stat().st_dev, staged.stat().st_ino) == before_closure_identity


@pytest.mark.parametrize("mutation", ["symlink", "mode"])
def test_atomic_exchange_rejects_unsafe_input(
    tmp_path: Path,
    mutation: str,
) -> None:
    closure = tmp_path / "closure"
    staged = tmp_path / "staged"
    closure.mkdir(mode=0o700)
    _write(closure / "receipt.json", b"receipt")
    if mutation == "symlink":
        staged_source = tmp_path / "staged-source"
        staged_source.mkdir(mode=0o700)
        _write(staged_source / "control.json", b"control")
        staged.symlink_to(staged_source, target_is_directory=True)
    else:
        staged.mkdir(mode=0o700)
        _write(staged / "control.json", b"control")
        staged.chmod(0o750)
    before = digest_directory_tree(closure)

    with pytest.raises(ProductionControlError, match="full-final production closure"):
        _atomic_exchange_directories(closure, staged)

    assert digest_directory_tree(closure) == before


def test_atomic_exchange_rejects_name_substitution_before_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = tmp_path / "closure"
    staged = tmp_path / "staged"
    displaced = tmp_path / "displaced"
    closure.mkdir(mode=0o700)
    staged.mkdir(mode=0o700)
    _write(closure / "receipt.json", b"receipt")
    _write(staged / "control.json", b"control")
    staged_before = digest_directory_tree(staged)

    def substitute_after_open(parent_descriptor: int, name: str, *, label: str) -> int:
        descriptor = _open_exchange_directory(parent_descriptor, name, label=label)
        if name == closure.name:
            closure.rename(displaced)
            closure.mkdir(mode=0o700)
            _write(closure / "hostile.json", b"hostile")
        return descriptor

    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._open_exchange_directory",
        substitute_after_open,
    )
    with pytest.raises(ProductionControlError, match="name changed before the swap"):
        _atomic_exchange_directories(closure, staged)

    assert digest_directory_tree(staged) == staged_before
    assert (displaced / "receipt.json").read_bytes() == b"receipt"


def test_atomic_exchange_unsupported_platform_leaves_both_names_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = tmp_path / "closure"
    staged = tmp_path / "staged"
    closure.mkdir(mode=0o700)
    staged.mkdir(mode=0o700)
    _write(closure / "receipt.json", b"receipt")
    _write(staged / "control.json", b"control")
    before_closure = digest_directory_tree(closure)
    before_staged = digest_directory_tree(staged)
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.sys",
        SimpleNamespace(platform="unsupported"),
    )

    with pytest.raises(ProductionControlError, match="lacks an admitted atomic"):
        _atomic_exchange_directories(closure, staged)

    assert digest_directory_tree(closure) == before_closure
    assert digest_directory_tree(staged) == before_staged


def test_hnsw_wheel_and_runtime_receipt_are_independently_bound() -> None:
    extraction = SimpleNamespace(
        opa_sha256="1" * 64,
        hnswlib_wheel_sha256="2" * 64,
        hnswlib_receipt_sha256="3" * 64,
    )
    admitted = SimpleNamespace(extraction=extraction)
    manifest = {
        "artifacts": [
            {"role": "opa-runtime-binary", "sha256": "1" * 64},
            {
                "role": "strict-authorized-hnsw",
                "sha256": "2" * 64,
                "revision": f"sha256:{'3' * 64}",
            },
        ]
    }
    _verify_c0_manifest_runtime(manifest, admitted)  # type: ignore[arg-type]

    manifest["artifacts"][1]["sha256"] = "4" * 64
    with pytest.raises(ProductionControlError, match="HNSW"):
        _verify_c0_manifest_runtime(manifest, admitted)  # type: ignore[arg-type]
    manifest["artifacts"][1]["sha256"] = "2" * 64
    manifest["artifacts"][1]["revision"] = f"sha256:{'5' * 64}"
    with pytest.raises(ProductionControlError, match="HNSW"):
        _verify_c0_manifest_runtime(manifest, admitted)  # type: ignore[arg-type]


def test_synthetic_five_corpus_finalization_exchanges_one_exact_shared_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = tmp_path / "production-run-closure"
    closure.mkdir(mode=0o700)
    provisional = digest_directory_tree(closure)
    manifest_digest = "d" * 64
    sealed_bytes = b'{"sealed":"run"}\n'
    _write(closure / f"{manifest_digest}.json", sealed_bytes)
    suite_base = tmp_path / "suite-base"
    materialization_sha = "1" * 64
    request = ProductionControlFinalizationRequest(
        materialization_config_path=tmp_path / "materialization.json",
        materialization_config_sha256=materialization_sha,
        blueprint_receipt_path=tmp_path / "blueprint-receipt.json",
        blueprint_receipt_sha256="2" * 64,
        c0_control_instantiation_receipt_path=(
            tmp_path / "c0-instantiated-controls" / C0_INSTANTIATION_RECEIPT_FILENAME
        ),
        frozen_manifest_path=tmp_path / "study-manifest.json",
        manifest_lock_path=tmp_path / "study-manifest.sha256",
        c1_package_root=tmp_path / "c1-package",
        protocol_registry_record_path=tmp_path / "registry.json",
        protocol_registration_receipt_path=tmp_path / "registration.json",
        sealed_run_receipt_path=closure / f"{manifest_digest}.json",
        online_custody_admission_path=tmp_path / "online-admission.json",
        custody_seal_receipt_path=tmp_path / "custody-seal.json",
        artifact_verification_receipt_path=tmp_path / "verification.json",
        artifact_root=tmp_path / "artifacts",
        local_artifact_map_path=tmp_path / "artifact-map.json",
        required_artifact_bindings_root=tmp_path / "required",
        runtime_evidence_root=tmp_path / "runtime-evidence",
    )
    request_path = tmp_path / "finalization-request.json"
    _write(request_path, request.canonical_file_bytes())
    admission_bytes = b'{"admission":"closed"}\n'
    source_by_corpus: dict[str, _CorpusSources] = {}
    corpora: list[_FinalizationCorpus] = []
    for position, corpus_id in enumerate(FIXED_CORPORA):
        source_root = tmp_path / "sources" / corpus_id
        execution = source_root / "execution.json"
        runtime = source_root / "runtime.json"
        execution_bytes = f"execution-{corpus_id}".encode("ascii")
        runtime_bytes = f"runtime-{corpus_id}".encode("ascii")
        _write(execution, execution_bytes)
        _write(runtime, runtime_bytes)
        source_by_corpus[corpus_id] = _CorpusSources(
            online_root=source_root,
            index_root=source_root,
            embedding_root=source_root,
            policy_root=source_root,
            query_root=source_root,
            execution_plan_path=execution,
            trial_runtime_receipt_path=runtime,
            policy_bundle_receipt_path=execution,
            index_bundle_receipt_path=runtime,
        )
        spec_bytes = f'{{"corpus":"{corpus_id}"}}\n'.encode("ascii")
        spec = SimpleNamespace(
            file_sha256=hashlib.sha256(spec_bytes).hexdigest(),
            canonical_file_bytes=lambda value=spec_bytes: value,
        )
        required_bytes = f'{{"required":"{corpus_id}"}}\n'.encode("ascii")
        required = SimpleNamespace(
            file_sha256=hashlib.sha256(required_bytes).hexdigest(),
            canonical_file_bytes=lambda value=required_bytes: value,
        )
        preflight = SimpleNamespace(
            contract_sha256=f"{position + 2:x}" * 64,
            file_sha256=f"{position + 7:x}" * 64,
        )
        preflight_receipt = SimpleNamespace(
            receipt_sha256=f"{position + 3:x}" * 64,
            file_sha256=f"{position + 8:x}" * 64,
        )
        transition = SimpleNamespace(
            receipt_sha256=f"{position + 4:x}" * 64,
            file_sha256=f"{position + 9:x}" * 64,
            final_plan_template_file_sha256=f"{position + 5:x}" * 64,
            final_plan_template_semantic_sha256=f"{position + 6:x}" * 64,
            final_control_tree_sha256=f"{position + 10:x}" * 64,
        )
        corpora.append(
            _FinalizationCorpus(
                corpus_id=corpus_id,
                spec=spec,  # type: ignore[arg-type]
                sharded_execution_plan_bytes=execution_bytes,
                trial_runtime_receipt_bytes=runtime_bytes,
                instantiated_binding=SimpleNamespace(),  # type: ignore[arg-type]
                preflight=preflight,  # type: ignore[arg-type]
                preflight_receipt=preflight_receipt,  # type: ignore[arg-type]
                transition=transition,  # type: ignore[arg-type]
                final_plan=SimpleNamespace(),  # type: ignore[arg-type]
                required_artifacts=required,  # type: ignore[arg-type]
            )
        )

    context_load_count = 0

    def context_for(loaded_request: ProductionControlFinalizationRequest) -> _FinalizationContext:
        nonlocal context_load_count
        context_load_count += 1
        context = _FinalizationContext(
            request=loaded_request,
            materialization=SimpleNamespace(
                file_sha256=materialization_sha,
                finalized_controls_root=closure,
                suite_base_root=suite_base,
            ),  # type: ignore[arg-type]
            blueprint=SimpleNamespace(
                receipt_sha256="2" * 64,
                launcher_identity_file_sha256="3" * 64,
                provisional_closure_tree_sha256=provisional.sha256,
            ),  # type: ignore[arg-type]
            instantiation=SimpleNamespace(
                file_sha256="4" * 64,
                launcher_identity_file_sha256="3" * 64,
            ),  # type: ignore[arg-type]
            admitted=SimpleNamespace(),  # type: ignore[arg-type]
            manifest={},
            manifest_sha256=manifest_digest,
            c0_commit="a" * 40,
            c1_commit="b" * 40,
            sealed_run=SimpleNamespace(),  # type: ignore[arg-type]
            sealed_run_bytes=sealed_bytes,
            online_admission=SimpleNamespace(),  # type: ignore[arg-type]
            online_admission_bytes=admission_bytes,
            verification_receipt=SimpleNamespace(),  # type: ignore[arg-type]
            corpora=tuple(corpora),
        )
        if context_load_count == 2:
            for sources in source_by_corpus.values():
                sources.execution_plan_path.write_bytes(b"hostile-execution")
                sources.trial_runtime_receipt_path.write_bytes(b"hostile-runtime")
        return context

    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._load_finalization_context",
        context_for,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._corpus_sources",
        lambda _admitted, corpus_id: source_by_corpus[corpus_id],
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls.verify_production_run_closure_binding",
        lambda *_args: None,
    )
    receipt_path = tmp_path / "finalization-receipt.json"
    fsynced_directories: list[Path] = []
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._fsync_private_directory",
        lambda path, *, label: fsynced_directories.append(path),
    )

    def crash_before_exchange(_first: Path, _second: Path) -> None:
        raise RuntimeError("injected crash before exchange")

    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._atomic_exchange_directories",
        crash_before_exchange,
    )
    with pytest.raises(RuntimeError, match="before exchange"):
        finalize_production_controls(
            request_path,
            expected_request_sha256=request.file_sha256,
            finalization_receipt_path=receipt_path,
        )
    staged_roots = {
        path for path in fsynced_directories if path.name.startswith(_PLACEHOLDER_RETENTION_PREFIX)
    }
    assert len(staged_roots) == 1
    staged_root = next(iter(staged_roots))
    assert set(fsynced_directories) == {
        staged_root,
        *(staged_root / corpus_id for corpus_id in FIXED_CORPORA),
        *(staged_root / corpus_id / "control" for corpus_id in FIXED_CORPORA),
    }
    retained = closure.parent / (f".production-run-closure.retained-{request.file_sha256[:20]}")
    assert digest_directory_tree(closure).entries == (f"{manifest_digest}.json",)
    assert frozenset(digest_directory_tree(retained).entries) != {f"{manifest_digest}.json"}
    for corpus_id in FIXED_CORPORA:
        control_root = retained / corpus_id / "control"
        assert (control_root / SHARDED_EXECUTION_PLAN_FILENAME).read_bytes() == (
            next(row for row in corpora if row.corpus_id == corpus_id).sharded_execution_plan_bytes
        )
        assert (control_root / TRIAL_RUNTIME_RECEIPT_FILENAME).read_bytes() == (
            next(row for row in corpora if row.corpus_id == corpus_id).trial_runtime_receipt_bytes
        )
    assert all(
        b"hostile" not in path.read_bytes()
        for root in (closure, retained)
        for path in root.rglob("*")
        if path.is_file()
    )

    partial = retained / "unexpected.partial"
    _write(partial, b"partial")
    with pytest.raises(ProductionControlError, match="neither exact"):
        finalize_production_controls(
            request_path,
            expected_request_sha256=request.file_sha256,
            finalization_receipt_path=receipt_path,
            resume=True,
        )
    partial.unlink()

    def crash_after_exchange(first: Path, second: Path) -> None:
        _atomic_exchange_directories(first, second)
        raise RuntimeError("injected crash after exchange")

    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._atomic_exchange_directories",
        crash_after_exchange,
    )
    with pytest.raises(RuntimeError, match="after exchange"):
        finalize_production_controls(
            request_path,
            expected_request_sha256=request.file_sha256,
            finalization_receipt_path=receipt_path,
            resume=True,
        )
    assert digest_directory_tree(retained).entries == (f"{manifest_digest}.json",)

    mutated = closure / FIXED_CORPORA[0] / "control" / PRODUCTION_CORPUS_CONFIG_FILENAME
    original = mutated.read_bytes()
    mutated.write_bytes(b"hostile-substitution")
    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_controls._atomic_exchange_directories",
        _atomic_exchange_directories,
    )
    with pytest.raises(ProductionControlError, match="finalized control"):
        finalize_production_controls(
            request_path,
            expected_request_sha256=request.file_sha256,
            finalization_receipt_path=receipt_path,
            resume=True,
        )
    mutated.write_bytes(original)
    receipt = finalize_production_controls(
        request_path,
        expected_request_sha256=request.file_sha256,
        finalization_receipt_path=receipt_path,
        resume=True,
    )

    assert load_production_control_finalization_receipt(receipt_path) == receipt
    assert digest_directory_tree(closure).sha256 == receipt.instantiated_closure_tree_sha256
    retained = Path(receipt.retained_intermediate_closure_path)
    assert digest_directory_tree(retained).entries == (f"{manifest_digest}.json",)
    assert (retained / f"{manifest_digest}.json").read_bytes() == sealed_bytes
    shared = receipt.corpora[0].closure_binding
    assert all(
        row.closure_binding.entries == shared.entries
        and row.closure_binding.files == shared.files
        and row.closure_binding.instantiated_closure_tree_sha256
        == shared.instantiated_closure_tree_sha256
        for row in receipt.corpora
    )
    assert PLAN_TEMPLATE_FILENAME not in shared.entries
    for corpus_id in FIXED_CORPORA:
        assert (closure / corpus_id / "control" / PRODUCTION_CORPUS_CONFIG_FILENAME).is_file()
    receipt_path.unlink()
    recovered = finalize_production_controls(
        request_path,
        expected_request_sha256=request.file_sha256,
        finalization_receipt_path=receipt_path,
        resume=True,
    )
    assert recovered == receipt
    assert (
        finalize_production_controls(
            request_path,
            expected_request_sha256=request.file_sha256,
            finalization_receipt_path=receipt_path,
            resume=True,
        )
        == receipt
    )
