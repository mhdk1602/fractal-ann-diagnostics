from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_execution_claim import (
    _beacon_receipt as _execution_beacon_receipt,
)
from test_execution_claim import (
    _claim as _execution_claim,
)
from test_execution_claim import (
    _live_job as _execution_live_job,
)
from test_execution_claim import (
    _provider as _execution_provider,
)
from test_execution_claim import (
    _zenodo as _execution_zenodo,
)

from fractal_ann_diagnostics.artifact_integrity import (
    digest_directory_tree,
    digest_regular_file,
)
from fractal_ann_diagnostics.execution_claim import (
    ClaimCorpusBinding,
    VerifiedRunClaimCapability,
    _mint_verified_run_claim,
    loads_runtime_claim_receipt,
)
from fractal_ann_diagnostics.opa_runtime_binary import load_runtime_attestation_plan_template
from fractal_ann_diagnostics.runtime_attestation import (
    ObservedMount,
    RuntimeArtifactMount,
    RuntimeAttestationPlan,
    RuntimeFilePin,
    RuntimePreflightReceipt,
    argv_sha256,
    environment_sha256,
    load_runtime_attestation_plan,
    runtime_attestation_plan_template_file_bytes,
)
from fractal_ann_diagnostics.sealed_container_launcher import (
    PREFLIGHT_DIGEST_SENTINEL,
    PREFLIGHT_INTEGER_SENTINEL,
    PREFLIGHT_OBSERVED_FIELDS,
    PREFLIGHT_TEXT_SENTINEL,
    PRODUCTION_RUN_CLOSURE_ROLE,
    ClosureFileBinding,
    ContainerOutputInventory,
    DockerResult,
    LauncherBindMount,
    LauncherEnvironmentVariable,
    LauncherGeometry,
    OutputFileDigest,
    PreflightLaunchContract,
    ProductionRunClosureBindingReceipt,
    RuntimePlanTransitionReceipt,
    SealedContainerLauncherError,
    SealedLaunchContract,
    VolumeInitializationReceipt,
    _mint_verified_production_run_closure,
    _sealed_evidence_inventory_sha256,
    _snapshot_sealed_evidence,
    initialize_output_volume,
    instantiate_registered_runtime_plan,
    launch_sealed_once,
    load_sealed_launch_failure_receipt,
    load_sealed_launch_receipt,
    loads_preflight_launch_contract,
    materialize_runtime_plan_transition,
    run_preflight_once,
    verify_sealed_launch_evidence,
    verify_sealed_launch_failure_evidence,
    verify_sealed_transition,
)
from fractal_ann_diagnostics.sealed_container_launcher import (
    _parser as _launcher_parser,
)

_IMAGE = "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@sha256:" + "a" * 64
_COMMIT = "b" * 40
_HOSTNAME = "fractal-confirmatory"
_MEMORY = 8 * 1024**3


def test_operator_document_uses_the_closed_launcher_parser_contract() -> None:
    parser = _launcher_parser()
    initialize = parser.parse_args(
        [
            "initialize-volume",
            "--contract",
            "/controlled/launcher/scifact/preflight-launch-contract.json",
            "--audit-root",
            "/controlled/launcher/scifact/volume-initialization-evidence",
        ]
    )
    assert initialize.command == "initialize-volume"
    assert initialize.audit_root.is_absolute()

    instantiate = parser.parse_args(
        [
            "instantiate-plan",
            "--preflight-contract",
            "/controlled/launcher/scifact/preflight-launch-contract.json",
            "--preflight-receipt",
            "/controlled/runtime-evidence/scifact/runtime-preflight-receipt.json",
            "--transition-receipt",
            "/controlled/runtime-evidence/scifact/runtime-plan-transition-receipt.json",
            "--finalization-request",
            "/controlled/suite/finalization-request.json",
            "--finalization-receipt",
            "/controlled/suite/production-control-finalization-receipt.json",
            "--instantiation-receipt",
            "/controlled/launcher/scifact/plan-instantiation-receipt.json",
            "--sealed-contract",
            "/controlled/launcher/scifact/sealed-launch-contract.json",
        ]
    )
    assert instantiate.command == "instantiate-plan"

    repository = Path(__file__).parents[1]
    launcher_document = (repository / "research" / "sealed-container-launcher.md").read_text(
        encoding="utf-8"
    )
    operator_section = launcher_document.split("## Operator commands", maxsplit=1)[1].split(
        "## Retained evidence", maxsplit=1
    )[0]
    assert "--audit-root" in operator_section
    assert "--finalization-request" in operator_section
    assert "fiqa" not in operator_section
    assert "/controlled/runtime-evidence/scifact/runtime-preflight-receipt.json" in (
        operator_section
    )
    for stale_option in ("--evidence-root", "--attempt-marker", "--manifest-sha256"):
        assert stale_option not in operator_section
    runner_image = (repository / "research" / "runner-image.md").read_text(encoding="utf-8")
    assert "\n  --config-sha256" not in runner_image


def _write(path: Path, value: bytes) -> str:
    path.write_bytes(value)
    path.chmod(0o600)
    return hashlib.sha256(value).hexdigest()


def _fixture(
    tmp_path: Path,
) -> tuple[PreflightLaunchContract, RuntimePreflightReceipt, Path]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(mode=0o700)
    uv_sha256 = _write(artifacts / "uv.lock", b"version = 1\n")
    launcher_sha256 = _write(artifacts / "launcher-identity.json", b"{}\n")
    artifact_sha256 = digest_directory_tree(artifacts).sha256

    opa = tmp_path / "opa"
    opa_sha256 = _write(opa, b"\x7fELFopa-1.18.2\n")
    controls = tmp_path / "controls"
    controls.mkdir(mode=0o700)
    closure = tmp_path / "production-run-closure"
    closure.mkdir(mode=0o700)
    corpus_control = closure / "scifact" / "control"
    corpus_control.mkdir(parents=True, mode=0o700)
    workload_spec_sha256 = _write(
        corpus_control / "production-corpus-workload-spec.json",
        b'{"schema_version":"test-workload-spec-v1"}\n',
    )
    _write(closure / "pre-c1-placeholder.json", b'{"state":"pre-c1"}\n')
    closure_sha256 = digest_directory_tree(closure).sha256

    environment = {
        "HOSTNAME": _HOSTNAME,
        "LANG": "C.UTF-8",
    }
    final_argv = (
        "/opt/venv/bin/python",
        "-m",
        "fractal_ann_diagnostics.cli",
        "run-sealed-corpus",
        "--config",
        str(closure / "scifact" / "control" / "corpus-run-config.json"),
    )
    runtime_mounts = tuple(
        sorted(
            (
                RuntimeArtifactMount(
                    root=str(closure),
                    role=PRODUCTION_RUN_CLOSURE_ROLE,
                    kind="directory",
                    artifact_sha256=closure_sha256,
                ),
                RuntimeArtifactMount(
                    root="/input/artifacts",
                    role="sealed-inputs",
                    kind="directory",
                    artifact_sha256=artifact_sha256,
                ),
                RuntimeArtifactMount(
                    root="/usr/local/bin/opa",
                    role="opa-runtime-binary",
                    kind="file",
                    artifact_sha256=opa_sha256,
                ),
            ),
            key=lambda item: item.root.encode("utf-8"),
        )
    )
    provisional = RuntimeAttestationPlan(
        attestation_id="confirmatory-scifact",
        manifest_sha256="d" * 64,
        runner_identity="mhdk1602-confirmatory",
        oci_image_digest=_IMAGE,
        code_commit=_COMMIT,
        operating_system_id=PREFLIGHT_TEXT_SENTINEL,
        operating_system_version_id=PREFLIGHT_TEXT_SENTINEL,
        kernel_release=PREFLIGHT_TEXT_SENTINEL,
        architecture=PREFLIGHT_TEXT_SENTINEL,
        cpu_model=PREFLIGHT_TEXT_SENTINEL,
        logical_cpu_count=PREFLIGHT_INTEGER_SENTINEL,
        memory_limit_bytes=PREFLIGHT_INTEGER_SENTINEL,
        mount_namespace_sha256=PREFLIGHT_DIGEST_SENTINEL,
        mounts=runtime_mounts,
        argv=final_argv,
        argv_sha256=argv_sha256(final_argv),
        environment_allowlist=tuple(sorted(environment)),
        environment_sha256=environment_sha256(environment),
        opa_binary=RuntimeFilePin(path="/usr/local/bin/opa", sha256=opa_sha256),
        python_binary=RuntimeFilePin(path="/opt/venv/bin/python", sha256="e" * 64),
        python_version=PREFLIGHT_TEXT_SENTINEL,
        uv_lock=RuntimeFilePin(path="/input/artifacts/uv.lock", sha256=uv_sha256),
        launcher_identity=RuntimeFilePin(
            path="/input/artifacts/launcher-identity.json",
            sha256=launcher_sha256,
        ),
        workload_id="sealed-scifact",
        workload_sha256=workload_spec_sha256,
        invocation_marker_path="/output/runtime-attempt.json",
    )
    plan_path = controls / "runtime-attestation-plan.template.json"
    plan_path.write_bytes(runtime_attestation_plan_template_file_bytes(provisional))
    plan_path.chmod(0o600)
    control_sha256 = digest_directory_tree(controls).sha256
    plan_file_sha256 = digest_regular_file(plan_path, label="plan template")

    bind_mounts = tuple(
        sorted(
            (
                LauncherBindMount(
                    source=str(closure),
                    target=str(closure),
                    role=PRODUCTION_RUN_CLOSURE_ROLE,
                    kind="directory",
                    content_sha256=closure_sha256,
                    attested_artifact=True,
                ),
                LauncherBindMount(
                    source=str(controls),
                    target="/input/control",
                    role="runtime-control-tree",
                    kind="directory",
                    content_sha256=control_sha256,
                    attested_artifact=False,
                ),
                LauncherBindMount(
                    source=str(artifacts),
                    target="/input/artifacts",
                    role="sealed-inputs",
                    kind="directory",
                    content_sha256=artifact_sha256,
                    attested_artifact=True,
                ),
                LauncherBindMount(
                    source=str(opa),
                    target="/usr/local/bin/opa",
                    role="opa-runtime-binary",
                    kind="file",
                    content_sha256=opa_sha256,
                    attested_artifact=True,
                ),
            ),
            key=lambda item: item.target.encode("utf-8"),
        )
    )
    (tmp_path / "suite" / "online").mkdir(parents=True, mode=0o700)
    geometry = LauncherGeometry(
        corpus_id="scifact",
        oci_image_digest=_IMAGE,
        code_commit=_COMMIT,
        platform="linux/arm64",
        uid=65532,
        gid=65532,
        hostname=_HOSTNAME,
        environment=tuple(
            LauncherEnvironmentVariable(name=name, value=environment[name])
            for name in sorted(environment)
        ),
        memory_limit_bytes=_MEMORY,
        cpuset_cpus=(0, 1),
        bind_mounts=bind_mounts,
        control_mount_target="/input/control",
        runtime_plan_template_relative_path="runtime-attestation-plan.template.json",
        output_volume="fractal-confirmatory-scifact",
        output_volume_subpath="sealed-output",
        output_root="/output",
        copy_output_root=str(tmp_path / "suite" / "online" / "scifact"),
        tmpfs_root="/tmp",
        tmpfs_size_bytes=1024**3,
        tmpfs_mode=0o1777,
        tmpfs_flags=("nodev", "noexec", "nosuid"),
    )
    contract = PreflightLaunchContract(
        geometry=geometry,
        argv=(
            "/opt/venv/bin/python",
            "-m",
            "fractal_ann_diagnostics.sealed_container_launcher",
            "capture-preflight",
        ),
        provisional_control_tree_sha256=control_sha256,
        provisional_plan_template_file_sha256=plan_file_sha256,
    )
    receipt = RuntimePreflightReceipt(
        launcher_contract_sha256=contract.contract_sha256,
        oci_image_digest=_IMAGE,
        code_commit=_COMMIT,
        hostname=_HOSTNAME,
        operating_system_id="debian",
        operating_system_version_id="12",
        kernel_release="6.12.0-linuxkit",
        architecture="aarch64",
        cpu_model="Apple M4 Max",
        logical_cpu_count=2,
        memory_limit_bytes=_MEMORY,
        mount_namespace_sha256="f" * 64,
        mount_namespace_raw_sha256="1" * 64,
        artifact_mounts=tuple(
            ObservedMount(root=root, read_only=True)
            for root in sorted(
                (str(closure), "/input/artifacts", "/usr/local/bin/opa"),
                key=lambda value: value.encode("utf-8"),
            )
        ),
        output_root="/output",
        tmpfs_root="/tmp",
        network_mode="none",
        network_interfaces=("lo",),
        non_loopback_route_count=0,
        route_tables_sha256="2" * 64,
        environment_allowlist=tuple(sorted(environment)),
        environment_sha256=environment_sha256(environment),
        python_executable="/opt/venv/bin/python",
        python_version="3.12.11",
        effective_uid=65532,
        effective_gid=65532,
    )
    return contract, receipt, plan_path


def _audit_root(tmp_path: Path) -> Path:
    root = tmp_path / "audit"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def _volume(contract: PreflightLaunchContract) -> VolumeInitializationReceipt:
    return VolumeInitializationReceipt(
        corpus_id=contract.geometry.corpus_id,
        preflight_launcher_contract_sha256=contract.contract_sha256,
        output_volume=contract.geometry.output_volume,
        output_volume_subpath=contract.geometry.output_volume_subpath,
        volume_inspect_sha256="5" * 64,
        initializer_container_id="3" * 64,
        inspect_sha256="4" * 64,
        initializer_start_returncode=0,
        initializer_state_status="exited",
        initializer_exit_code=0,
        initializer_oom_killed=False,
        stdout_sha256=hashlib.sha256(b"").hexdigest(),
        stdout_byte_count=0,
        stderr_sha256=hashlib.sha256(b"").hexdigest(),
        stderr_byte_count=0,
    )


def _closure_binding(
    contract: PreflightLaunchContract,
    transition: RuntimePlanTransitionReceipt,
    *,
    manifest_sha256: str = "9" * 64,
) -> ProductionRunClosureBindingReceipt:
    closure = Path(contract.geometry.production_run_closure_mount.source)
    (closure / "pre-c1-placeholder.json").unlink()
    config_path = closure / "scifact" / "control" / "corpus-run-config.json"
    _write(config_path, b'{"schema_version":"test-config-v1"}\n')
    sealed_path = closure / f"{manifest_sha256}.json"
    _write(sealed_path, b'{"schema_version":"test-sealed-run-v1"}\n')
    tree = digest_directory_tree(closure)
    files = tuple(
        ClosureFileBinding(
            relative_path=relative_path,
            file_sha256=digest_regular_file(
                closure / relative_path,
                label="test closure file",
            ),
            byte_count=(closure / relative_path).stat().st_size,
        )
        for relative_path in tree.entries
        if (closure / relative_path).is_file()
    )
    workload_relative_path = "scifact/control/production-corpus-workload-spec.json"
    return ProductionRunClosureBindingReceipt(
        corpus_id="scifact",
        manifest_sha256=manifest_sha256,
        preflight_launcher_contract_sha256=contract.contract_sha256,
        runtime_plan_transition_receipt_sha256=transition.receipt_sha256,
        closure_source=str(closure),
        closure_target=str(closure),
        provisional_closure_tree_sha256=(
            contract.geometry.production_run_closure_mount.content_sha256
        ),
        instantiated_closure_tree_sha256=tree.sha256,
        config_relative_path="scifact/control/corpus-run-config.json",
        config_file_sha256=digest_regular_file(config_path, label="test config"),
        workload_spec_relative_path=workload_relative_path,
        workload_spec_file_sha256=digest_regular_file(
            closure / workload_relative_path,
            label="test workload spec",
        ),
        sealed_run_receipt_relative_path=f"{manifest_sha256}.json",
        sealed_run_receipt_file_sha256=digest_regular_file(
            sealed_path,
            label="test sealed receipt",
        ),
        entries=tree.entries,
        files=files,
    )


def _materialize_registered(
    contract: PreflightLaunchContract,
    receipt: RuntimePreflightReceipt,
):  # type: ignore[no-untyped-def]
    transition = materialize_runtime_plan_transition(contract, receipt)
    closure_binding = _closure_binding(contract, transition)
    verified_closure = _mint_verified_production_run_closure(
        closure_binding,
        fresh_revalidator=lambda: closure_binding,
    )
    instantiation, sealed = instantiate_registered_runtime_plan(
        contract,
        receipt,
        transition,
        verified_closure=verified_closure,
    )
    return transition, verified_closure, instantiation, sealed


def _run_claim_authority(
    sealed: SealedLaunchContract,
    preflight: PreflightLaunchContract,
    preflight_receipt: RuntimePreflightReceipt,
    transition: RuntimePlanTransitionReceipt,
    instantiation,  # type: ignore[no-untyped-def]
    verified_closure,  # type: ignore[no-untyped-def]
) -> tuple[VerifiedRunClaimCapability, bytes]:
    """Mint the same typed provider authority required in production."""

    runtime_plan = verify_sealed_transition(
        sealed,
        preflight,
        preflight_receipt,
        transition,
        instantiation,
        verified_closure.binding,
    )
    base = _execution_claim()
    staging_uri = Path(sealed.geometry.copy_output_root).as_uri()
    corpora = tuple(
        ClaimCorpusBinding(
            corpus_id=row.corpus_id,
            staging_namespace_uri=(
                staging_uri
                if row.corpus_id == sealed.geometry.corpus_id
                else row.staging_namespace_uri
            ),
            canonical_namespace_uri=row.canonical_namespace_uri,
            runtime_plan_sha256=(
                runtime_plan.plan_sha256
                if row.corpus_id == sealed.geometry.corpus_id
                else row.runtime_plan_sha256
            ),
            runtime_plan_file_sha256=row.runtime_plan_file_sha256,
        )
        for row in base.corpora
    )
    aggregate = hashlib.sha256(
        json.dumps(
            {
                "corpora": [
                    {
                        "canonical_namespace_uri": row.canonical_namespace_uri,
                        "corpus_id": row.corpus_id,
                        "staging_namespace_uri": row.staging_namespace_uri,
                    }
                    for row in corpora
                ],
                "derivation": "sha256-five-canonical-output-trees-v1",
                "manifest_sha256": sealed.manifest_sha256,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    contract = replace(
        base,
        corpora=corpora,
        manifest_sha256=sealed.manifest_sha256,
        output_aggregate_identity=aggregate,
    )
    provider = _execution_provider(contract)
    capability = _mint_verified_run_claim(
        contract=contract,
        provider_identity=provider,
        claim_state_sha256=hashlib.sha256(b"claimed-state").hexdigest(),
        claim_ledger_commit="4" * 40,
        claim_attested_at_utc="2023-11-14T22:13:19+00:00",
        beacon_receipt=_execution_beacon_receipt(contract, provider),
        live_execute_job_receipt=_execution_live_job(contract, provider),
        zenodo_admission=_execution_zenodo(),
        fresh_revalidator=lambda: None,
    )
    receipt = capability.require_launch(
        manifest_sha256=sealed.manifest_sha256,
        corpus_id=sealed.geometry.corpus_id,
        runtime_plan_sha256=runtime_plan.plan_sha256,
        output_namespace_uri=staging_uri,
    )
    return capability, receipt.canonical_file_bytes()


def test_sealed_launch_rejects_omitted_run_claim_before_docker(tmp_path: Path) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    transition, verified_closure, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    docker = _FailCreateDocker()
    with pytest.raises(TypeError, match="run_claim"):
        launch_sealed_once(
            sealed,
            contract,
            receipt,
            transition,
            instantiation,
            verified_closure,
            _volume(contract),
            secret=b"missing-authority",
            audit_root=_audit_root(tmp_path),
            docker=docker,
        )
    assert docker.calls == []


def test_preflight_contract_round_trips_and_is_closed(tmp_path: Path) -> None:
    contract, _, _ = _fixture(tmp_path)
    assert loads_preflight_launch_contract(contract.canonical_file_bytes()) == contract
    changed = json.loads(contract.canonical_file_bytes())
    changed["unknown"] = True
    with pytest.raises(SealedContainerLauncherError, match="unexpected"):
        loads_preflight_launch_contract(
            json.dumps(changed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
    with pytest.raises(SealedContainerLauncherError, match="not canonical"):
        loads_preflight_launch_contract(
            json.dumps(contract.to_dict(), sort_keys=True, indent=2).encode() + b"\n"
        )


def test_launcher_geometry_rejects_production_closure_beneath_tmpfs(
    tmp_path: Path,
) -> None:
    contract, _, _ = _fixture(tmp_path)
    hostile_closure = replace(
        contract.geometry.production_run_closure_mount,
        source="/tmp/production-run-closure",
        target="/tmp/production-run-closure",
    )
    hostile_mounts = tuple(
        sorted(
            (
                hostile_closure if mount.role == PRODUCTION_RUN_CLOSURE_ROLE else mount
                for mount in contract.geometry.bind_mounts
            ),
            key=lambda mount: mount.target.encode("utf-8"),
        )
    )
    with pytest.raises(SealedContainerLauncherError, match="overlap a writable root"):
        replace(contract.geometry, bind_mounts=hostile_mounts)


def test_transition_changes_only_predeclared_observations(tmp_path: Path) -> None:
    contract, receipt, plan_path = _fixture(tmp_path)
    transition = materialize_runtime_plan_transition(contract, receipt)
    closure_binding = _closure_binding(contract, transition)
    verified_closure = _mint_verified_production_run_closure(
        closure_binding,
        fresh_revalidator=lambda: closure_binding,
    )
    assert b'"{manifest_sha256}"' in plan_path.read_bytes()
    instantiation, sealed = instantiate_registered_runtime_plan(
        contract,
        receipt,
        transition,
        verified_closure=verified_closure,
    )
    final = verify_sealed_transition(
        sealed,
        contract,
        receipt,
        transition,
        instantiation,
        verified_closure.binding,
    )

    assert transition.allowed_observation_fields == PREFLIGHT_OBSERVED_FIELDS
    assert final.operating_system_id == "debian"
    assert final.architecture == "aarch64"
    assert final.logical_cpu_count == 2
    assert final.memory_limit_bytes == _MEMORY
    assert final.mount_namespace_sha256 == receipt.mount_namespace_sha256
    assert final.python_version == "3.12.11"
    assert not plan_path.exists()
    assert (plan_path.parent / "runtime-attestation-plan.json").read_bytes() == (
        final.canonical_file_bytes()
    )
    assert sealed.argv == final.argv
    assert instantiation.manifest_sha256 == "9" * 64
    assert final.manifest_sha256 == "9" * 64


def test_transition_rejects_a_non_sentinel_observation(tmp_path: Path) -> None:
    contract, receipt, plan_path = _fixture(tmp_path)
    plan = load_runtime_attestation_plan_template(plan_path)
    changed = replace(plan, cpu_model="operator-selected")
    plan_path.write_bytes(runtime_attestation_plan_template_file_bytes(changed))
    changed_tree = digest_directory_tree(Path(contract.geometry.control_mount.source)).sha256
    changed_mounts = tuple(
        replace(item, content_sha256=changed_tree) if not item.attested_artifact else item
        for item in contract.geometry.bind_mounts
    )
    changed_contract = replace(
        contract,
        geometry=replace(contract.geometry, bind_mounts=changed_mounts),
        provisional_control_tree_sha256=changed_tree,
        provisional_plan_template_file_sha256=digest_regular_file(plan_path, label="changed plan"),
    )
    receipt = replace(receipt, launcher_contract_sha256=changed_contract.contract_sha256)
    with pytest.raises(SealedContainerLauncherError, match="cpu_model.*sentinel"):
        materialize_runtime_plan_transition(changed_contract, receipt)


def test_transition_verifier_rejects_a_rehashed_forbidden_delta(tmp_path: Path) -> None:
    contract, receipt, plan_path = _fixture(tmp_path)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    plan_path = plan_path.parent / "runtime-attestation-plan.json"
    final = load_runtime_attestation_plan(plan_path)
    hostile = replace(final, workload_id="sealed-scifact-hostile")
    plan_path.write_bytes(hostile.canonical_file_bytes())
    final_tree = digest_directory_tree(Path(sealed.geometry.control_mount.source)).sha256
    final_file = digest_regular_file(plan_path, label="hostile final plan")
    hostile_instantiation = replace(
        instantiation,
        instantiated_control_tree_sha256=final_tree,
        instantiated_plan_file_sha256=final_file,
        instantiated_plan_semantic_sha256=hostile.plan_sha256,
    )
    hostile_mounts = tuple(
        replace(item, content_sha256=final_tree) if not item.attested_artifact else item
        for item in sealed.geometry.bind_mounts
    )
    hostile_sealed = replace(
        sealed,
        geometry=replace(sealed.geometry, bind_mounts=hostile_mounts),
        registered_plan_instantiation_receipt_sha256=hostile_instantiation.receipt_sha256,
        instantiated_control_tree_sha256=final_tree,
        instantiated_plan_file_sha256=final_file,
        instantiated_plan_semantic_sha256=hostile.plan_sha256,
    )
    with pytest.raises(SealedContainerLauncherError, match="C1 template"):
        verify_sealed_transition(
            hostile_sealed,
            contract,
            receipt,
            transition,
            hostile_instantiation,
            closure_binding.binding,
        )


def test_sealed_contract_rejects_shell_or_alternate_command(tmp_path: Path) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    transition, _, _, sealed = _materialize_registered(contract, receipt)
    assert isinstance(transition, RuntimePlanTransitionReceipt)
    with pytest.raises(SealedContainerLauncherError, match="run-sealed-corpus"):
        replace(sealed, argv=("/opt/venv/bin/python", "-c", "import os"))


def _inspect_payload(
    contract: PreflightLaunchContract | SealedLaunchContract,
    *,
    container_id: str = "1" * 64,
    container_name: str = "fractal-preflight-scifact",
    role: str = "preflight",
    authority_sha256: str | None = None,
    start_returncode: int = 0,
    oom_killed: bool = False,
    state_error: str = "",
    include_bind_mounts: bool = True,
    output_target: str | None = None,
    output_read_only: bool = False,
    use_output_subpath: bool = True,
    root_identity: bool = False,
    argv: tuple[str, ...] | None = None,
) -> bytes:
    geometry = contract.geometry
    command = contract.argv if argv is None else argv
    authority = contract.contract_sha256 if authority_sha256 is None else authority_sha256
    target = geometry.output_root if output_target is None else output_target
    configured_mounts: list[dict[str, object]] = []
    observed_mounts: list[dict[str, object]] = []
    if include_bind_mounts:
        for mount in geometry.bind_mounts:
            configured_mounts.append(
                {
                    "ReadOnly": True,
                    "Source": mount.source,
                    "Target": mount.target,
                    "Type": "bind",
                }
            )
            observed_mounts.append(
                {
                    "Destination": mount.target,
                    "RW": False,
                    "Source": mount.source,
                    "Type": "bind",
                }
            )
    volume_mount: dict[str, object] = {
        "ReadOnly": output_read_only,
        "Source": geometry.output_volume,
        "Target": target,
        "Type": "volume",
    }
    if use_output_subpath:
        volume_mount["VolumeOptions"] = {"Subpath": geometry.output_volume_subpath}
    configured_mounts.append(volume_mount)
    observed_mounts.append(
        {
            "Destination": target,
            "Name": geometry.output_volume,
            "RW": not output_read_only,
            "Type": "volume",
        }
    )
    uid = 0 if root_identity else geometry.uid
    gid = 0 if root_identity else geometry.gid
    payload = [
        {
            "Id": container_id,
            "Name": f"/{container_name}",
            "Platform": geometry.platform,
            "Config": {
                "Cmd": list(command[1:]),
                "Entrypoint": [command[0]],
                "Env": [f"{row.name}={row.value}" for row in geometry.environment],
                "Hostname": geometry.hostname,
                "Image": geometry.oci_image_digest,
                "Labels": {
                    "io.fractal-ann.authority-sha256": authority,
                    "io.fractal-ann.corpus-id": geometry.corpus_id,
                    "io.fractal-ann.role": role,
                },
                "User": f"{uid}:{gid}",
            },
            "HostConfig": {
                "AutoRemove": False,
                "CapAdd": ["CHOWN", "FOWNER"] if root_identity else None,
                "CapDrop": ["ALL"],
                "CpusetCpus": geometry.cpuset_text,
                "Memory": geometry.memory_limit_bytes,
                "Mounts": configured_mounts,
                "NetworkMode": "none",
                "Privileged": False,
                "ReadonlyRootfs": True,
                "RestartPolicy": {"MaximumRetryCount": 0, "Name": "no"},
                "SecurityOpt": ["no-new-privileges"],
                "Tmpfs": {
                    geometry.tmpfs_root: (
                        "rw,nodev,noexec,nosuid,"
                        f"size={geometry.tmpfs_size_bytes},uid={uid},"
                        f"gid={gid},mode={geometry.tmpfs_mode:o}"
                    )
                },
            },
            "Mounts": observed_mounts,
            "State": {
                "Dead": False,
                "Error": state_error,
                "ExitCode": start_returncode,
                "OOMKilled": oom_killed,
                "Paused": False,
                "Pid": 0,
                "Restarting": False,
                "Running": False,
                "Status": "exited",
            },
        }
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


class _FakeDocker:
    def __init__(
        self,
        preflight: PreflightLaunchContract,
        receipt: RuntimePreflightReceipt,
        *,
        sealed_result: DockerResult | None = None,
        sealed_oom_killed: bool = False,
    ) -> None:
        self.preflight = preflight
        self.receipt = receipt
        self.sealed: SealedLaunchContract | None = None
        self.calls: list[tuple[tuple[str, ...], bytes | None]] = []
        self.kinds: dict[str, str] = {}
        self.logs: dict[str, DockerResult] = {}
        self.names: dict[str, str] = {}
        self.counter = 0
        self.volume_exists = False
        self.sealed_result = sealed_result or DockerResult(0, b"complete\n", b"")
        self.sealed_oom_killed = sealed_oom_killed
        self.output_files = {
            "runtime-attestation-receipt.json": b"{}\n",
            "scientific-result.json": b'{"complete":true}\n',
        }
        source = (
            Path(preflight.geometry.control_mount.source).parent / f"fake-volume-output-{id(self)}"
        )
        source.mkdir(mode=0o700)
        for name, encoded in self.output_files.items():
            (source / name).write_bytes(encoded)
            (source / name).chmod(0o600)
        tree = digest_directory_tree(source)
        self.inventory = ContainerOutputInventory(
            tree_sha256=tree.sha256,
            file_count=tree.file_count,
            directory_count=tree.directory_count,
            byte_count=tree.byte_count,
            files=tuple(
                OutputFileDigest(
                    relative_path=name,
                    byte_count=len(encoded),
                    sha256=hashlib.sha256(encoded).hexdigest(),
                )
                for name, encoded in sorted(self.output_files.items())
            ),
        )

    def run(
        self,
        arguments: tuple[str, ...] | list[str],
        *,
        input_bytes: bytes | None = None,
    ) -> DockerResult:
        args = tuple(arguments)
        self.calls.append((args, input_bytes))
        if args[:2] == ("volume", "inspect"):
            if not self.volume_exists:
                return DockerResult(1, b"", b"not found")
            payload = [
                {
                    "Driver": "local",
                    "Labels": {
                        "io.fractal-ann.authority-sha256": self.preflight.contract_sha256,
                        "io.fractal-ann.corpus-id": self.preflight.geometry.corpus_id,
                        "io.fractal-ann.role": "sealed-output-volume",
                        "io.fractal-ann.subpath": (self.preflight.geometry.output_volume_subpath),
                    },
                    "Name": self.preflight.geometry.output_volume,
                    "Options": {},
                    "Scope": "local",
                }
            ]
            return DockerResult(
                0,
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n",
                b"",
            )
        if args[:2] == ("volume", "create"):
            self.volume_exists = True
            return DockerResult(0, (self.preflight.geometry.output_volume + "\n").encode(), b"")
        if args[0] == "create":
            self.counter += 1
            container_id = f"{self.counter:064x}"
            name = args[args.index("--name") + 1]
            if name.startswith("fractal-init-"):
                kind = "initializer"
            elif name.startswith("fractal-preflight-"):
                kind = "preflight"
            elif name.startswith("fractal-copyout-"):
                kind = "output-reader"
            else:
                kind = "sealed"
            self.kinds[container_id] = kind
            self.names[container_id] = name
            return DockerResult(0, (container_id + "\n").encode(), b"")
        if args[0] == "start":
            container_id = args[-1]
            kind = self.kinds[container_id]
            if kind == "preflight":
                assert input_bytes == self.preflight.canonical_file_bytes()
                result = DockerResult(0, self.receipt.canonical_file_bytes(), b"")
            elif kind == "sealed":
                assert input_bytes is not None
                runtime_claim = loads_runtime_claim_receipt(input_bytes)
                assert runtime_claim.canonical_file_bytes() == input_bytes
                result = self.sealed_result
            elif kind == "output-reader":
                assert input_bytes is None
                result = DockerResult(0, self.inventory.canonical_file_bytes(), b"")
            else:
                assert input_bytes is None
                result = DockerResult(0, b"", b"")
            self.logs[container_id] = result
            return result
        if args[0] == "inspect":
            container_id = args[-1]
            kind = self.kinds[container_id]
            start_result = self.logs[container_id]
            if kind == "initializer":
                argv = (
                    "/opt/venv/bin/python",
                    "-m",
                    "fractal_ann_diagnostics.sealed_container_launcher",
                    "initialize-output",
                    "--path",
                    "/volume/sealed-output",
                )
                payload = _inspect_payload(
                    self.preflight,
                    container_id=container_id,
                    container_name=self.names[container_id],
                    role="output-initializer",
                    authority_sha256=self.preflight.contract_sha256,
                    start_returncode=start_result.returncode,
                    include_bind_mounts=False,
                    output_target="/volume",
                    use_output_subpath=False,
                    root_identity=True,
                    argv=argv,
                )
            elif kind == "output-reader":
                assert self.sealed is not None
                argv = (
                    "/opt/venv/bin/python",
                    "-m",
                    "fractal_ann_diagnostics.sealed_container_launcher",
                    "inventory-output",
                    "--root",
                    "/output",
                )
                payload = _inspect_payload(
                    self.sealed,
                    container_id=container_id,
                    container_name=self.names[container_id],
                    role="output-reader",
                    authority_sha256=self.sealed.contract_sha256,
                    start_returncode=start_result.returncode,
                    include_bind_mounts=False,
                    output_read_only=True,
                    argv=argv,
                )
            else:
                contract = self.preflight if kind == "preflight" else self.sealed
                assert contract is not None
                payload = _inspect_payload(
                    contract,
                    container_id=container_id,
                    container_name=self.names[container_id],
                    role=kind,
                    authority_sha256=contract.contract_sha256,
                    start_returncode=start_result.returncode,
                    oom_killed=self.sealed_oom_killed if kind == "sealed" else False,
                )
            return DockerResult(0, payload, b"")
        if args[0] == "logs":
            retained = self.logs[args[-1]]
            return DockerResult(0, retained.stdout, retained.stderr)
        if args[0] == "cp":
            target = Path(args[-1])
            assert target.is_dir()
            assert self.kinds[args[1].split(":", 1)[0]] == "output-reader"
            for name, encoded in self.output_files.items():
                (target / name).write_bytes(encoded)
                (target / name).chmod(0o600)
            return DockerResult(0, b"", b"")
        raise AssertionError(args)


def test_fake_docker_e2e_is_shell_free_single_attach_and_retains_output(
    tmp_path: Path,
) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    audit = _audit_root(tmp_path)
    docker = _FakeDocker(contract, receipt)
    volume = initialize_output_volume(contract, audit_root=audit, docker=docker)
    observed = run_preflight_once(contract, volume, audit_root=audit, docker=docker)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        observed,
    )
    docker.sealed = sealed
    run_claim, claim_secret = _run_claim_authority(
        sealed, contract, observed, transition, instantiation, closure_binding
    )
    completed = launch_sealed_once(
        sealed,
        contract,
        observed,
        transition,
        instantiation,
        closure_binding,
        volume,
        run_claim,
        secret=claim_secret,
        audit_root=audit,
        docker=docker,
    )

    assert completed.stdin_secret_sha256 == hashlib.sha256(claim_secret).hexdigest()
    assert completed.stdin_secret_byte_count == len(claim_secret)
    assert (audit / "sealed-launcher-attempt-marker.json").is_file()
    assert (audit / "sealed-inspect.json").is_file()
    assert Path(sealed.geometry.copy_output_root, "scientific-result.json").is_file()
    assert claim_secret not in b"".join(
        path.read_bytes() for path in audit.rglob("*") if path.is_file()
    )
    commands = [arguments for arguments, _ in docker.calls]
    assert not any(
        "/bin/sh" in argument or argument == "-c" for row in commands for argument in row
    )
    sealed_starts = [
        row
        for row in commands
        if row[:3] == ("start", "--attach", "--interactive")
        and docker.kinds.get(row[-1]) == "sealed"
    ]
    assert len(sealed_starts) == 1
    assert not any(row[0] in {"rm", "container", "volume"} and "rm" in row for row in commands)
    sealed_creates = [
        row for row in commands if row[0] == "create" and "fractal-sealed-scifact" in row
    ]
    assert len(sealed_creates) == 1
    assert "volume-subpath=sealed-output" in " ".join(sealed_creates[0])
    readers = [row for row in commands if row[0] == "create" and "fractal-copyout-scifact" in row]
    assert len(readers) == 1
    assert readers[0][readers[0].index("--user") + 1] == "65532:65532"
    assert "volume-subpath=sealed-output,readonly" in " ".join(readers[0])


class _FailCreateDocker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        arguments: tuple[str, ...] | list[str],
        *,
        input_bytes: bytes | None = None,
    ) -> DockerResult:
        del input_bytes
        args = tuple(arguments)
        self.calls.append(args)
        if args[0] == "create":
            return DockerResult(1, b"", b"synthetic create failure")
        raise AssertionError(args)


def test_attempt_marker_precedes_create_failure_and_blocks_retry(tmp_path: Path) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    volume = _volume(contract)
    audit = _audit_root(tmp_path)
    docker = _FailCreateDocker()
    run_claim, claim_secret = _run_claim_authority(
        sealed, contract, receipt, transition, instantiation, closure_binding
    )

    with pytest.raises(SealedContainerLauncherError, match="creation failed"):
        launch_sealed_once(
            sealed,
            contract,
            receipt,
            transition,
            instantiation,
            closure_binding,
            volume,
            run_claim,
            secret=claim_secret,
            audit_root=audit,
            docker=docker,
        )
    marker = audit / "sealed-launcher-attempt-marker.json"
    assert marker.is_file()
    failure = load_sealed_launch_failure_receipt(audit / "sealed-launch-failure-receipt.json")
    assert failure.failure_stage == "sealed-create"
    assert failure.sealed_container_id is None
    verify_sealed_launch_failure_evidence(
        failure,
        audit_root=audit,
        sealed_contract=sealed,
    )
    assert len(docker.calls) == 1

    with pytest.raises(SealedContainerLauncherError, match="attempt marker"):
        launch_sealed_once(
            sealed,
            contract,
            receipt,
            transition,
            instantiation,
            closure_binding,
            volume,
            run_claim,
            secret=claim_secret,
            audit_root=audit,
            docker=docker,
        )
    assert len(docker.calls) == 1


def _reclose_failure_evidence(audit: Path, failure, **changes):  # type: ignore[no-untyped-def]
    evidence = _snapshot_sealed_evidence(
        audit,
        excluded_filenames=frozenset({"sealed-launch-failure-receipt.json"}),
    )
    hostile = replace(
        failure,
        evidence_files=evidence,
        evidence_inventory_sha256=_sealed_evidence_inventory_sha256(evidence),
        **changes,
    )
    (audit / "sealed-launch-failure-receipt.json").write_bytes(hostile.canonical_file_bytes())
    return hostile


def _create_failure(tmp_path: Path):  # type: ignore[no-untyped-def]
    contract, receipt, _ = _fixture(tmp_path)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    audit = _audit_root(tmp_path)
    docker = _FailCreateDocker()
    run_claim, claim_secret = _run_claim_authority(
        sealed, contract, receipt, transition, instantiation, closure_binding
    )
    with pytest.raises(SealedContainerLauncherError, match="creation failed"):
        launch_sealed_once(
            sealed,
            contract,
            receipt,
            transition,
            instantiation,
            closure_binding,
            _volume(contract),
            run_claim,
            secret=claim_secret,
            audit_root=audit,
            docker=docker,
        )
    failure = load_sealed_launch_failure_receipt(audit / "sealed-launch-failure-receipt.json")
    return audit, sealed, failure


def test_failure_verifier_rederives_stage(tmp_path: Path) -> None:
    audit, sealed, failure = _create_failure(tmp_path)
    hostile = replace(failure, failure_stage="sealed-start")
    (audit / "sealed-launch-failure-receipt.json").write_bytes(hostile.canonical_file_bytes())

    with pytest.raises(SealedContainerLauncherError, match="stage differs"):
        verify_sealed_launch_failure_evidence(
            hostile,
            audit_root=audit,
            sealed_contract=sealed,
        )


def test_failure_verifier_rejects_argv_only_substitution(tmp_path: Path) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    audit = _audit_root(tmp_path)

    class RaiseBeforeResult:
        def run(self, arguments, *, input_bytes=None):  # type: ignore[no-untyped-def]
            del arguments, input_bytes
            raise RuntimeError("synthetic Docker transport failure")

    run_claim, claim_secret = _run_claim_authority(
        sealed, contract, receipt, transition, instantiation, closure_binding
    )

    with pytest.raises(RuntimeError, match="transport failure"):
        launch_sealed_once(
            sealed,
            contract,
            receipt,
            transition,
            instantiation,
            closure_binding,
            _volume(contract),
            run_claim,
            secret=claim_secret,
            audit_root=audit,
            docker=RaiseBeforeResult(),
        )
    failure = load_sealed_launch_failure_receipt(audit / "sealed-launch-failure-receipt.json")
    assert not (audit / "sealed-create-docker-result.json").exists()
    argument_path = audit / "sealed-create-docker-argv.json"
    argument = json.loads(argument_path.read_bytes())
    argument["arguments"][argument["arguments"].index("none")] = "bridge"
    argument_path.write_bytes(
        json.dumps(argument, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    hostile = _reclose_failure_evidence(audit, failure)

    with pytest.raises(SealedContainerLauncherError, match="argument array differs"):
        verify_sealed_launch_failure_evidence(
            hostile,
            audit_root=audit,
            sealed_contract=sealed,
        )


def test_failure_verifier_rejects_contract_substitution(tmp_path: Path) -> None:
    audit, sealed, failure = _create_failure(tmp_path)
    substituted = replace(sealed, manifest_sha256="c" * 64)

    with pytest.raises(SealedContainerLauncherError, match="binding differs from contract"):
        verify_sealed_launch_failure_evidence(
            failure,
            audit_root=audit,
            sealed_contract=substituted,
        )


def test_failure_verifier_rederives_private_error_record(tmp_path: Path) -> None:
    audit, sealed, failure = _create_failure(tmp_path)
    error_path = audit / "sealed-launch-failure-error.json"
    error_record = json.loads(error_path.read_bytes())
    error_record["redacted_message"] = "substituted error"
    error_path.write_bytes(
        json.dumps(error_record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    hostile = _reclose_failure_evidence(audit, failure)

    with pytest.raises(SealedContainerLauncherError, match="error evidence differs"):
        verify_sealed_launch_failure_evidence(
            hostile,
            audit_root=audit,
            sealed_contract=sealed,
        )


def test_inspect_volume_subpath_tampering_is_rejected(tmp_path: Path) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    audit = _audit_root(tmp_path)
    docker = _FakeDocker(contract, receipt)
    volume = initialize_output_volume(contract, audit_root=audit, docker=docker)

    class Hostile(_FakeDocker):
        def run(self, arguments, *, input_bytes=None):  # type: ignore[no-untyped-def]
            result = super().run(arguments, input_bytes=input_bytes)
            args = tuple(arguments)
            if args[0] == "inspect" and self.kinds[args[-1]] == "preflight":
                payload = json.loads(result.stdout)
                output = next(
                    item
                    for item in payload[0]["HostConfig"]["Mounts"]
                    if item["Target"] == "/output"
                )
                output["VolumeOptions"]["Subpath"] = "alternate"
                hostile = (
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                )
                return DockerResult(0, hostile, b"")
            return result

    hostile_docker = Hostile(contract, receipt)
    with pytest.raises(SealedContainerLauncherError, match="volume subpath"):
        run_preflight_once(contract, volume, audit_root=audit, docker=hostile_docker)


def test_host_copy_must_match_read_only_container_inventory(tmp_path: Path) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    audit = _audit_root(tmp_path)

    class HostileCopy(_FakeDocker):
        def run(self, arguments, *, input_bytes=None):  # type: ignore[no-untyped-def]
            result = super().run(arguments, input_bytes=input_bytes)
            args = tuple(arguments)
            if args[0] == "cp" and result.returncode == 0:
                target = Path(args[-1]) / "scientific-result.json"
                target.write_bytes(b'{"complete":false}\n')
            return result

    docker = HostileCopy(contract, receipt)
    docker.sealed = sealed
    run_claim, claim_secret = _run_claim_authority(
        sealed, contract, receipt, transition, instantiation, closure_binding
    )
    with pytest.raises(SealedContainerLauncherError, match="read-only source"):
        launch_sealed_once(
            sealed,
            contract,
            receipt,
            transition,
            instantiation,
            closure_binding,
            _volume(contract),
            run_claim,
            secret=claim_secret,
            audit_root=audit,
            docker=docker,
        )
    failure = load_sealed_launch_failure_receipt(audit / "sealed-launch-failure-receipt.json")
    assert failure.failure_stage == "output-copy-verification"
    assert failure.sealed_container_id is not None
    assert failure.output_reader_container_id is not None
    verify_sealed_launch_failure_evidence(
        failure,
        audit_root=audit,
        sealed_contract=sealed,
    )


def test_nonzero_oom_start_publishes_closed_failure_and_blocks_retry(tmp_path: Path) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    audit = _audit_root(tmp_path)
    docker = _FakeDocker(
        contract,
        receipt,
        sealed_result=DockerResult(137, b"", b"synthetic OOM\n"),
        sealed_oom_killed=True,
    )
    docker.sealed = sealed
    run_claim, claim_secret = _run_claim_authority(
        sealed, contract, receipt, transition, instantiation, closure_binding
    )

    with pytest.raises(SealedContainerLauncherError, match="consumed the attempt"):
        launch_sealed_once(
            sealed,
            contract,
            receipt,
            transition,
            instantiation,
            closure_binding,
            _volume(contract),
            run_claim,
            secret=claim_secret,
            audit_root=audit,
            docker=docker,
        )

    failed = load_sealed_launch_receipt(audit / "sealed-launch-receipt.json")
    assert failed.outcome == "failed"
    assert failed.docker_start_returncode == 137
    assert failed.container_exit_code == 137
    assert failed.container_oom_killed is True
    assert failed.output_reader_container_id is None
    verify_sealed_launch_evidence(failed, audit_root=audit, sealed_contract=sealed)
    calls_after_failure = len(docker.calls)

    with pytest.raises(SealedContainerLauncherError, match="attempt marker"):
        launch_sealed_once(
            sealed,
            contract,
            receipt,
            transition,
            instantiation,
            closure_binding,
            _volume(contract),
            run_claim,
            secret=claim_secret,
            audit_root=audit,
            docker=docker,
        )
    assert len(docker.calls) == calls_after_failure


@pytest.mark.parametrize(
    ("tampering", "message"),
    (
        ("environment", "environment differs"),
        ("capability", "CapAdd differs"),
        ("mount", "mount count differs"),
    ),
)
def test_sealed_inspect_rejects_extra_environment_capability_or_mount(
    tmp_path: Path,
    tampering: str,
    message: str,
) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    audit = _audit_root(tmp_path)

    class HostileInspect(_FakeDocker):
        def run(self, arguments, *, input_bytes=None):  # type: ignore[no-untyped-def]
            result = super().run(arguments, input_bytes=input_bytes)
            args = tuple(arguments)
            if args[0] == "inspect" and self.kinds[args[-1]] == "sealed":
                payload = json.loads(result.stdout)
                if tampering == "environment":
                    payload[0]["Config"]["Env"].append("EVIL=1")
                elif tampering == "capability":
                    payload[0]["HostConfig"]["CapAdd"] = ["NET_ADMIN"]
                else:
                    payload[0]["HostConfig"]["Mounts"].append(
                        {
                            "ReadOnly": True,
                            "Source": "/host/extra",
                            "Target": "/input/extra",
                            "Type": "bind",
                        }
                    )
                    payload[0]["Mounts"].append(
                        {
                            "Destination": "/input/extra",
                            "RW": False,
                            "Source": "/host/extra",
                            "Type": "bind",
                        }
                    )
                hostile = (
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                )
                return DockerResult(0, hostile, b"")
            return result

    docker = HostileInspect(contract, receipt)
    docker.sealed = sealed
    run_claim, claim_secret = _run_claim_authority(
        sealed, contract, receipt, transition, instantiation, closure_binding
    )
    with pytest.raises(SealedContainerLauncherError, match=message):
        launch_sealed_once(
            sealed,
            contract,
            receipt,
            transition,
            instantiation,
            closure_binding,
            _volume(contract),
            run_claim,
            secret=claim_secret,
            audit_root=audit,
            docker=docker,
        )
    failure = load_sealed_launch_failure_receipt(audit / "sealed-launch-failure-receipt.json")
    assert failure.failure_stage == "sealed-evidence-verification"
    assert failure.sealed_container_id is not None
    verify_sealed_launch_failure_evidence(
        failure,
        audit_root=audit,
        sealed_contract=sealed,
    )


def test_output_reader_inspect_must_retain_read_only_volume(tmp_path: Path) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    audit = _audit_root(tmp_path)

    class WritableReader(_FakeDocker):
        def run(self, arguments, *, input_bytes=None):  # type: ignore[no-untyped-def]
            result = super().run(arguments, input_bytes=input_bytes)
            args = tuple(arguments)
            if args[0] == "inspect" and self.kinds[args[-1]] == "output-reader":
                payload = json.loads(result.stdout)
                payload[0]["HostConfig"]["Mounts"][0]["ReadOnly"] = False
                payload[0]["Mounts"][0]["RW"] = True
                return DockerResult(
                    0,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n",
                    b"",
                )
            return result

    docker = WritableReader(contract, receipt)
    docker.sealed = sealed
    run_claim, claim_secret = _run_claim_authority(
        sealed, contract, receipt, transition, instantiation, closure_binding
    )
    with pytest.raises(SealedContainerLauncherError, match="configured mount differs"):
        launch_sealed_once(
            sealed,
            contract,
            receipt,
            transition,
            instantiation,
            closure_binding,
            _volume(contract),
            run_claim,
            secret=claim_secret,
            audit_root=audit,
            docker=docker,
        )


def test_closed_evidence_revalidation_rejects_file_substitution(tmp_path: Path) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    audit = _audit_root(tmp_path)
    docker = _FakeDocker(contract, receipt)
    docker.sealed = sealed
    run_claim, claim_secret = _run_claim_authority(
        sealed, contract, receipt, transition, instantiation, closure_binding
    )
    completed = launch_sealed_once(
        sealed,
        contract,
        receipt,
        transition,
        instantiation,
        closure_binding,
        _volume(contract),
        run_claim,
        secret=claim_secret,
        audit_root=audit,
        docker=docker,
    )
    verify_sealed_launch_evidence(completed, audit_root=audit, sealed_contract=sealed)

    evidence = audit / "sealed-start-docker-argv.json"
    evidence.write_bytes(evidence.read_bytes().replace(b'"start"', b'"stats"', 1))
    with pytest.raises(SealedContainerLauncherError, match="membership or bytes differ"):
        verify_sealed_launch_evidence(completed, audit_root=audit, sealed_contract=sealed)


def test_command_substitution_text_remains_one_literal_docker_argument(tmp_path: Path) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    original = Path(
        next(item.source for item in contract.geometry.bind_mounts if item.role == "sealed-inputs")
    )
    hostile = original.with_name("artifacts-$(echo-pwned)")
    original.rename(hostile)
    mounts = tuple(
        replace(item, source=str(hostile)) if item.role == "sealed-inputs" else item
        for item in contract.geometry.bind_mounts
    )
    contract = replace(contract, geometry=replace(contract.geometry, bind_mounts=mounts))
    receipt = replace(receipt, launcher_contract_sha256=contract.contract_sha256)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    audit = _audit_root(tmp_path)
    docker = _FakeDocker(contract, receipt)
    docker.sealed = sealed
    run_claim, claim_secret = _run_claim_authority(
        sealed, contract, receipt, transition, instantiation, closure_binding
    )
    launch_sealed_once(
        sealed,
        contract,
        receipt,
        transition,
        instantiation,
        closure_binding,
        _volume(contract),
        run_claim,
        secret=claim_secret,
        audit_root=audit,
        docker=docker,
    )

    argument_record = json.loads((audit / "sealed-create-docker-argv.json").read_bytes())
    mount_argument = next(
        item for item in argument_record["arguments"] if "artifacts-$(echo-pwned)" in item
    )
    assert "$(echo-pwned)" in mount_argument
    assert not any(argument in {"/bin/sh", "-c"} for argument in argument_record["arguments"])


def test_copy_destination_parent_alias_is_rejected_after_reader_inventory(tmp_path: Path) -> None:
    contract, receipt, _ = _fixture(tmp_path)
    real_parent = Path(contract.geometry.copy_output_root).parent
    alias_root = tmp_path / "aliased"
    alias_root.mkdir(mode=0o700)
    alias_parent = alias_root / "online"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    aliased_geometry = replace(
        contract.geometry,
        copy_output_root=str(alias_parent / contract.geometry.corpus_id),
    )
    contract = replace(contract, geometry=aliased_geometry)
    receipt = replace(receipt, launcher_contract_sha256=contract.contract_sha256)
    transition, closure_binding, instantiation, sealed = _materialize_registered(
        contract,
        receipt,
    )
    audit = _audit_root(tmp_path)
    docker = _FakeDocker(contract, receipt)
    docker.sealed = sealed
    run_claim, claim_secret = _run_claim_authority(
        sealed, contract, receipt, transition, instantiation, closure_binding
    )

    with pytest.raises(SealedContainerLauncherError, match="path alias"):
        launch_sealed_once(
            sealed,
            contract,
            receipt,
            transition,
            instantiation,
            closure_binding,
            _volume(contract),
            run_claim,
            secret=claim_secret,
            audit_root=audit,
            docker=docker,
        )
