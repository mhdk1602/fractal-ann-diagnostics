from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.runtime_attestation as runtime_module
import fractal_ann_diagnostics.sealed_online_execution as sealed_online
from fractal_ann_diagnostics.artifact_integrity import (
    digest_directory_tree,
    digest_regular_file,
)
from fractal_ann_diagnostics.runtime_attestation import (
    LinuxRuntimeProbe,
    ObservedMount,
    RuntimeArtifactMount,
    RuntimeAttestationError,
    RuntimeAttestationPlan,
    RuntimeAttestationReceipt,
    RuntimeFilePin,
    RuntimeObservation,
    RuntimePreflightReceipt,
    argv_sha256,
    attest_runtime_once,
    capture_runtime_preflight,
    environment_sha256,
    launcher_identity_file_bytes,
    load_runtime_attestation_plan,
    load_runtime_attestation_receipt,
    loads_runtime_attestation_plan,
    loads_runtime_attestation_receipt,
    loads_runtime_preflight_receipt,
    mount_namespace_sha256,
    raw_mount_namespace_sha256,
    verify_live_runtime_attestation,
    verify_runtime_attestation_receipt,
    write_runtime_attestation_plan,
    write_runtime_attestation_receipt,
    write_runtime_preflight_receipt,
)
from fractal_ann_diagnostics.sealed_online_execution import SealedOnlineExecutionError
from fractal_ann_diagnostics.study import SealedRunReceipt

_COMMIT = "a" * 40
_IMAGE = "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@sha256:" + "b" * 64
_MANIFEST = "c" * 64
_WORKLOAD = "d" * 64
_ROUTES = "e" * 64
_MOUNT_NAMESPACE = "1" * 64


def _write(path: Path, payload: bytes, *, executable: bool = False) -> Path:
    path.write_bytes(payload)
    path.chmod(0o500 if executable else 0o400)
    return path


def _fixture(tmp_path: Path) -> tuple[RuntimeAttestationPlan, RuntimeObservation, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    controlled = tmp_path / "controlled"
    controlled.mkdir(mode=0o700)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(mode=0o700)
    _write(artifacts / "query-package.json", b'{"query":"opaque"}\n')

    opa = _write(controlled / "opa", b"opa-v1\n", executable=True)
    python = _write(controlled / "python", b"python-v1\n", executable=True)
    uv_lock = _write(controlled / "uv.lock", b"version = 1\n")
    launcher = controlled / "launcher-identity.json"
    _write(
        launcher,
        launcher_identity_file_bytes(oci_image_digest=_IMAGE, code_commit=_COMMIT),
    )

    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    environment = {"LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}
    argv = (str(python), "-m", "fractal_ann_diagnostics.cli", "sealed-run")
    plan = RuntimeAttestationPlan(
        attestation_id="confirmatory-v0.3.0",
        manifest_sha256=_MANIFEST,
        runner_identity="runner-65532",
        oci_image_digest=_IMAGE,
        code_commit=_COMMIT,
        operating_system_id="debian",
        operating_system_version_id="12",
        kernel_release="6.12.0-linuxkit",
        architecture="x86_64",
        cpu_model="AMD EPYC 7763",
        logical_cpu_count=8,
        memory_limit_bytes=16 * 1024**3,
        mount_namespace_sha256=_MOUNT_NAMESPACE,
        mounts=tuple(
            sorted(
                (
                    RuntimeArtifactMount(
                        root=str(artifacts),
                        role="sealed-inputs",
                        kind="directory",
                        artifact_sha256=digest_directory_tree(artifacts).sha256,
                    ),
                    RuntimeArtifactMount(
                        root=str(controlled),
                        role="runtime-controls",
                        kind="directory",
                        artifact_sha256=digest_directory_tree(controlled).sha256,
                    ),
                ),
                key=lambda row: row.root.encode(),
            )
        ),
        argv=argv,
        argv_sha256=argv_sha256(argv),
        environment_allowlist=tuple(sorted(environment)),
        environment_sha256=environment_sha256(environment),
        opa_binary=RuntimeFilePin(path=str(opa), sha256=digest_regular_file(opa)),
        python_binary=RuntimeFilePin(path=str(python), sha256=digest_regular_file(python)),
        python_version="3.12.11",
        uv_lock=RuntimeFilePin(path=str(uv_lock), sha256=digest_regular_file(uv_lock)),
        launcher_identity=RuntimeFilePin(path=str(launcher), sha256=digest_regular_file(launcher)),
        workload_id="sealed-action-matrix-v1",
        workload_sha256=_WORKLOAD,
        invocation_marker_path=str(output / "runtime-attempt.json"),
    )
    observed = RuntimeObservation(
        operating_system_id=plan.operating_system_id,
        operating_system_version_id=plan.operating_system_version_id,
        kernel_release=plan.kernel_release,
        architecture=plan.architecture,
        cpu_model=plan.cpu_model,
        logical_cpu_count=plan.logical_cpu_count,
        memory_limit_bytes=plan.memory_limit_bytes,
        mount_namespace_sha256=plan.mount_namespace_sha256,
        mount_namespace_raw_sha256="2" * 64,
        mounts=tuple(ObservedMount(root=row.root, read_only=True) for row in plan.mounts),
        network_mode="none",
        network_namespace_inode=4815162342,
        network_interfaces=("lo",),
        non_loopback_route_count=0,
        route_tables_sha256=_ROUTES,
        argv=plan.argv,
        environment=environment,
        python_executable=plan.python_binary.path,
        python_version=plan.python_version,
    )
    return plan, observed, output


def _sealed_run_receipt(plan: RuntimeAttestationPlan) -> SealedRunReceipt:
    return SealedRunReceipt(
        manifest_sha256=plan.manifest_sha256,
        protocol_version="0.3.0",
        started_at_utc="2026-07-14T12:00:00+00:00",
        runner_identity=plan.runner_identity,
        code_commit=plan.code_commit,
        runner_image=plan.oci_image_digest,
        protocol_registration_receipt_uri="file:///controlled/protocol-receipt.json",
        protocol_registration_receipt_sha256="4" * 64,
        protocol_registration_record_uri="file:///controlled/protocol-record.json",
        verification_receipt_uri="file:///controlled/verification.json",
        verification_receipt_sha256="5" * 64,
        receipt_uri="file:///controlled/run.json",
    )


def _sealed_source_paths(plan: RuntimeAttestationPlan) -> dict[str, Path]:
    mounts = {row.role: Path(row.root) for row in plan.mounts}
    artifacts = mounts["sealed-inputs"]
    query = artifacts / "query-package.json"
    return {
        "artifact_root": artifacts,
        "authorized_index_store_root": artifacts,
        "embedding_store_root": artifacts,
        "partition_audit_path": query,
        "policy_intervention_root": artifacts,
        "pseudonym_key_path": Path(plan.uv_lock.path),
        "query_package_root": artifacts,
        "schedule_path": query,
        "staged_root": artifacts,
    }


def test_attest_once_round_trips_canonical_plan_and_receipt(tmp_path: Path) -> None:
    plan, observed, output = _fixture(tmp_path)
    plan_path = output / "runtime-plan.json"
    receipt_path = output / "runtime-receipt.json"
    write_runtime_attestation_plan(plan, plan_path)

    loaded_plan = load_runtime_attestation_plan(plan_path)
    assert loaded_plan == plan
    receipt = attest_runtime_once(
        loaded_plan,
        probe=lambda _: observed,
        receipt_target=receipt_path,
    )
    loaded_receipt = load_runtime_attestation_receipt(receipt_path)

    assert loaded_receipt == receipt
    assert loaded_receipt.oci_image_digest == _IMAGE
    assert loaded_receipt.code_commit == _COMMIT
    assert loaded_receipt.network["mode"] == "none"
    assert loaded_receipt.process["environment_sha256"] == plan.environment_sha256
    assert (output / "runtime-attempt.json").exists()
    verify_runtime_attestation_receipt(loaded_receipt, loaded_plan)


def test_canonical_loaders_reject_unknown_duplicate_and_alternate_json(tmp_path: Path) -> None:
    plan, observed, _ = _fixture(tmp_path)
    payload = plan.to_dict()
    payload["unknown"] = True
    with pytest.raises(RuntimeAttestationError, match="unexpected"):
        loads_runtime_attestation_plan(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
    with pytest.raises(RuntimeAttestationError, match="not canonical"):
        loads_runtime_attestation_plan(
            json.dumps(plan.to_dict(), sort_keys=True, indent=2).encode() + b"\n"
        )
    duplicate = plan.canonical_file_bytes().replace(
        b'{"architecture":', b'{"architecture":"x86_64","architecture":', 1
    )
    with pytest.raises(RuntimeAttestationError, match="repeats key"):
        loads_runtime_attestation_plan(duplicate)

    receipt = attest_runtime_once(plan, probe=lambda _: observed)
    receipt_payload = receipt.to_dict()
    receipt_payload["unknown"] = 1
    with pytest.raises(RuntimeAttestationError, match="unexpected"):
        loads_runtime_attestation_receipt(
            json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )


def test_second_invocation_is_rejected(tmp_path: Path) -> None:
    plan, observed, _ = _fixture(tmp_path)
    attest_runtime_once(plan, probe=lambda _: observed)
    with pytest.raises(RuntimeAttestationError, match="invocation marker already exists"):
        attest_runtime_once(plan, probe=lambda _: observed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: replace(
                row,
                mounts=(replace(row.mounts[0], read_only=False), *row.mounts[1:]),
            ),
            "writable",
        ),
        (lambda row: replace(row, network_mode="enabled"), "network is enabled"),
        (
            lambda row: replace(
                row,
                network_interfaces=("eth0", "lo"),
                non_loopback_route_count=1,
            ),
            "non-loopback",
        ),
        (lambda row: replace(row, argv=(*row.argv, "--retry")), "argv differs"),
        (
            lambda row: replace(row, environment={**row.environment, "SECRET": "value"}),
            "environment differs",
        ),
        (lambda row: replace(row, cpu_model="different CPU"), "cpu_model differs"),
        (lambda row: replace(row, memory_limit_bytes=1024), "memory_limit_bytes differs"),
        (
            lambda row: replace(row, mount_namespace_sha256="2" * 64),
            "mount_namespace_sha256 differs",
        ),
    ],
)
def test_runtime_observation_mismatch_consumes_attempt(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    plan, observed, _ = _fixture(tmp_path)
    assert callable(mutation)
    with pytest.raises(RuntimeAttestationError, match=message):
        attest_runtime_once(plan, probe=lambda _: mutation(observed))
    assert Path(plan.invocation_marker_path).is_file()
    with pytest.raises(RuntimeAttestationError, match="invocation marker already exists"):
        attest_runtime_once(plan, probe=lambda _: observed)


def test_artifact_digest_mismatch_consumes_attempt(tmp_path: Path) -> None:
    plan, observed, _ = _fixture(tmp_path)
    query = Path(plan.mounts[0].root) / "query-package.json"
    query.chmod(0o600)
    query.write_bytes(b'{"query":"changed"}\n')
    query.chmod(0o400)

    with pytest.raises(RuntimeAttestationError, match="mount digest differs"):
        attest_runtime_once(plan, probe=lambda _: observed)
    with pytest.raises(RuntimeAttestationError, match="invocation marker already exists"):
        attest_runtime_once(plan, probe=lambda _: observed)


def test_launcher_identity_must_match_image_and_commit(tmp_path: Path) -> None:
    plan, observed, _ = _fixture(tmp_path)
    changed = replace(
        plan,
        oci_image_digest=(
            "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@sha256:" + "f" * 64
        ),
    )
    with pytest.raises(RuntimeAttestationError, match="launcher OCI image digest differs"):
        attest_runtime_once(changed, probe=lambda _: observed)


def test_symlink_binary_and_path_alias_fail_closed(tmp_path: Path) -> None:
    plan, observed, _ = _fixture(tmp_path)
    alias = Path(plan.opa_binary.path).parent / "opa-alias"
    alias.symlink_to(plan.opa_binary.path)
    changed = replace(
        plan,
        opa_binary=RuntimeFilePin(path=str(alias), sha256=plan.opa_binary.sha256),
    )
    with pytest.raises(RuntimeAttestationError, match="symlink"):
        attest_runtime_once(changed, probe=lambda _: observed)

    with pytest.raises(
        RuntimeAttestationError,
        match="alias another path|canonical absolute POSIX",
    ):
        RuntimeFilePin(path="/tmp/controlled/../opa", sha256="0" * 64)


def test_writable_or_nonexecutable_binary_is_rejected(tmp_path: Path) -> None:
    plan, observed, _ = _fixture(tmp_path)
    opa = Path(plan.opa_binary.path)
    opa.chmod(0o720)
    with pytest.raises(RuntimeAttestationError, match="group/other writable"):
        attest_runtime_once(plan, probe=lambda _: observed)

    plan, observed, _ = _fixture(tmp_path / "second")
    Path(plan.opa_binary.path).chmod(0o400)
    with pytest.raises(RuntimeAttestationError, match="executable"):
        attest_runtime_once(plan, probe=lambda _: observed)


def test_receipt_verifier_rejects_a_typed_but_different_plan_binding(tmp_path: Path) -> None:
    plan, observed, _ = _fixture(tmp_path)
    receipt = attest_runtime_once(plan, probe=lambda _: observed)
    tampered = replace(receipt, workload_sha256="f" * 64)
    with pytest.raises(RuntimeAttestationError, match="workload_sha256 differs"):
        verify_runtime_attestation_receipt(tampered, plan)

    process = {**receipt.process, "argument_count": 1}
    tampered = replace(receipt, process=process)
    with pytest.raises(RuntimeAttestationError, match="argument count differs"):
        verify_runtime_attestation_receipt(tampered, plan)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: replace(row, mount_namespace_sha256="2" * 64),
            "mount_namespace_sha256 differs",
        ),
        (
            lambda row: replace(
                row,
                environment={**row.environment, "SECRET": "injected"},
            ),
            "environment differs",
        ),
        (
            lambda row: replace(row, route_tables_sha256="3" * 64),
            "network namespace differs",
        ),
    ],
)
def test_live_verifier_rejects_runtime_drift_after_receipt(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    plan, observed, _ = _fixture(tmp_path)
    receipt = attest_runtime_once(plan, probe=lambda _: observed)
    assert callable(mutation)

    with pytest.raises(RuntimeAttestationError, match=message):
        verify_live_runtime_attestation(
            receipt,
            plan,
            probe=lambda _: mutation(observed),
        )


def test_live_verifier_accepts_the_same_process_contract(tmp_path: Path) -> None:
    plan, observed, _ = _fixture(tmp_path)
    receipt = attest_runtime_once(plan, probe=lambda _: observed)

    verify_live_runtime_attestation(receipt, plan, probe=lambda _: observed)


def test_sealed_gate_loads_pinned_files_and_reobserves_before_source_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, observed, output = _fixture(tmp_path)
    plan_path = output / "runtime-plan.json"
    receipt_path = output / "runtime-receipt.json"
    write_runtime_attestation_plan(plan, plan_path)
    receipt = attest_runtime_once(
        plan,
        probe=lambda _: observed,
        receipt_target=receipt_path,
    )
    probe_calls = 0

    def live_probe() -> object:
        def observe(_: RuntimeAttestationPlan) -> RuntimeObservation:
            nonlocal probe_calls
            probe_calls += 1
            return observed

        return observe

    monkeypatch.setattr(sealed_online, "LinuxRuntimeProbe", live_probe)
    admitted_plan, admitted_receipt = sealed_online._admit_production_runtime_attestation(
        plan_path=plan_path,
        expected_plan_sha256=plan.plan_sha256,
        receipt_path=receipt_path,
        expected_receipt_sha256=receipt.receipt_sha256,
        run_receipt=_sealed_run_receipt(plan),
        source_paths=_sealed_source_paths(plan),
    )

    assert admitted_plan == plan
    assert admitted_receipt == receipt
    assert probe_calls == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: replace(row, mount_namespace_sha256="2" * 64),
            "mount_namespace_sha256 differs",
        ),
        (
            lambda row: replace(
                row,
                process={**row.process, "environment_sha256": "3" * 64},
            ),
            "environment digest differs",
        ),
    ],
)
def test_sealed_gate_rejects_receipt_runtime_fact_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    message: str,
) -> None:
    plan, observed, output = _fixture(tmp_path)
    plan_path = output / "runtime-plan.json"
    receipt_path = output / "runtime-receipt.json"
    write_runtime_attestation_plan(plan, plan_path)
    receipt = attest_runtime_once(plan, probe=lambda _: observed)
    assert callable(mutation)
    changed = mutation(receipt)
    assert isinstance(changed, RuntimeAttestationReceipt)
    write_runtime_attestation_receipt(changed, receipt_path)
    monkeypatch.setattr(
        sealed_online,
        "LinuxRuntimeProbe",
        lambda: lambda _: observed,
    )

    with pytest.raises(SealedOnlineExecutionError, match=message):
        sealed_online._admit_production_runtime_attestation(
            plan_path=plan_path,
            expected_plan_sha256=plan.plan_sha256,
            receipt_path=receipt_path,
            expected_receipt_sha256=changed.receipt_sha256,
            run_receipt=_sealed_run_receipt(plan),
            source_paths=_sealed_source_paths(plan),
        )


def test_sealed_gate_rejects_workload_source_outside_attested_mounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, observed, output = _fixture(tmp_path)
    plan_path = output / "runtime-plan.json"
    receipt_path = output / "runtime-receipt.json"
    write_runtime_attestation_plan(plan, plan_path)
    receipt = attest_runtime_once(
        plan,
        probe=lambda _: observed,
        receipt_target=receipt_path,
    )
    monkeypatch.setattr(
        sealed_online,
        "LinuxRuntimeProbe",
        lambda: lambda _: observed,
    )
    sources = _sealed_source_paths(plan)
    sources["query_package_root"] = (tmp_path / "unattested-query-root").resolve()

    with pytest.raises(SealedOnlineExecutionError, match="query_package_root.*attested"):
        sealed_online._admit_production_runtime_attestation(
            plan_path=plan_path,
            expected_plan_sha256=plan.plan_sha256,
            receipt_path=receipt_path,
            expected_receipt_sha256=receipt.receipt_sha256,
            run_receipt=_sealed_run_receipt(plan),
            source_paths=sources,
        )


def test_plan_rejects_overlapping_mount_roots(tmp_path: Path) -> None:
    plan, _, _ = _fixture(tmp_path)
    nested = Path(plan.mounts[0].root) / "nested"
    nested.mkdir()
    extra = RuntimeArtifactMount(
        root=str(nested),
        role="nested",
        kind="directory",
        artifact_sha256=digest_directory_tree(nested).sha256,
    )
    mounts = tuple(sorted((*plan.mounts, extra), key=lambda row: row.root.encode()))
    with pytest.raises(RuntimeAttestationError, match="cannot overlap"):
        replace(plan, mounts=mounts)


def test_linux_probe_rejects_non_linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(runtime_module.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeAttestationError, match="requires Linux"):
        LinuxRuntimeProbe()(plan)


def test_mount_namespace_digest_covers_unregistered_mounts() -> None:
    base = (
        "36 25 0:32 / / rw,relatime - overlay overlay rw,lowerdir=/base\n"
        "37 36 0:40 / /input ro,nosuid,nodev - ext4 /dev/vda1 ro\n"
    )
    repeated = (
        "52 49 0:32 / / rw,relatime - overlay overlay rw,lowerdir=/base\n"
        "53 52 0:40 / /input ro,nodev,nosuid - ext4 /dev/vda1 ro\n"
    )
    added_labels = repeated + ("54 52 0:41 /labels /labels ro,nosuid,nodev - ext4 /dev/vdb1 ro\n")

    assert mount_namespace_sha256(base) == mount_namespace_sha256(repeated)
    assert mount_namespace_sha256(base) != mount_namespace_sha256(added_labels)
    assert raw_mount_namespace_sha256(base) != raw_mount_namespace_sha256(repeated)


def test_mount_namespace_profile_normalizes_only_container_specific_paths() -> None:
    first = (
        "36 25 0:32 /abc / rw,relatime - overlay overlay "
        "rw,lowerdir=/var/lib/docker/abc/l,upperdir=/var/lib/docker/abc/u,"
        "workdir=/var/lib/docker/abc/w\n"
        "37 36 0:40 /docker/containers/abc/hostname /etc/hostname rw,relatime "
        "- tmpfs /dev/sda1 rw,size=1024k\n"
    )
    second = (
        "52 49 0:99 /def / rw,relatime - overlay overlay "
        "rw,lowerdir=/var/lib/docker/def/l,upperdir=/var/lib/docker/def/u,"
        "workdir=/var/lib/docker/def/w\n"
        "53 52 0:77 /docker/containers/def/hostname /etc/hostname rw,relatime "
        "- tmpfs /dev/sdb2 rw,size=1024k\n"
    )
    writable_hostname = second.replace(
        "/etc/hostname rw,relatime", "/etc/hostname rw,exec,relatime"
    )

    assert mount_namespace_sha256(first) == mount_namespace_sha256(second)
    assert raw_mount_namespace_sha256(first) != raw_mount_namespace_sha256(second)
    assert mount_namespace_sha256(first) != mount_namespace_sha256(writable_hostname)


@pytest.mark.parametrize(
    "mountinfo",
    (
        "",
        "36 25 0:32 / / rw,relatime overlay overlay rw\n",
        "36 25 0:32 / / rw,relatime - overlay\n",
        ("36 25 0:32 / / rw,relatime - overlay overlay rw\n37 25 0:33 / / ro - tmpfs tmpfs ro\n"),
    ),
)
def test_mount_namespace_digest_rejects_malformed_or_duplicate_rows(
    mountinfo: str,
) -> None:
    with pytest.raises(RuntimeAttestationError, match="mountinfo|truncated|repeats"):
        mount_namespace_sha256(mountinfo)


def test_file_loader_rejects_symlink_plan(tmp_path: Path) -> None:
    plan, _, output = _fixture(tmp_path)
    target = output / "plan.json"
    target.write_bytes(plan.canonical_file_bytes())
    alias = output / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(RuntimeAttestationError, match="symlink"):
        load_runtime_attestation_plan(alias)


def test_receipt_constructor_rejects_network_evidence_with_unknown_field(
    tmp_path: Path,
) -> None:
    plan, observed, _ = _fixture(tmp_path)
    receipt = attest_runtime_once(plan, probe=lambda _: observed)
    changed_network = {**receipt.network, "socket_count": 0}
    with pytest.raises(RuntimeAttestationError, match="unexpected"):
        replace(receipt, network=changed_network)


def test_receipt_is_a_typed_closed_record(tmp_path: Path) -> None:
    plan, observed, _ = _fixture(tmp_path)
    receipt = attest_runtime_once(plan, probe=lambda _: observed)
    loaded = RuntimeAttestationReceipt.from_dict(receipt.to_dict())
    assert loaded.canonical_file_bytes() == receipt.canonical_file_bytes()


def test_runtime_preflight_receipt_is_canonical_and_write_once(tmp_path: Path) -> None:
    receipt = RuntimePreflightReceipt(
        launcher_contract_sha256="2" * 64,
        oci_image_digest=_IMAGE,
        code_commit=_COMMIT,
        hostname="fractal-confirmatory",
        operating_system_id="debian",
        operating_system_version_id="12",
        kernel_release="6.12.0-linuxkit",
        architecture="aarch64",
        cpu_model="Apple M4 Max",
        logical_cpu_count=8,
        memory_limit_bytes=16 * 1024**3,
        mount_namespace_sha256="3" * 64,
        mount_namespace_raw_sha256="4" * 64,
        artifact_mounts=(ObservedMount(root="/input/control", read_only=True),),
        output_root="/output",
        tmpfs_root="/tmp",
        network_mode="none",
        network_interfaces=("lo",),
        non_loopback_route_count=0,
        route_tables_sha256=_ROUTES,
        environment_allowlist=("HOSTNAME", "LANG"),
        environment_sha256="5" * 64,
        python_executable="/opt/venv/bin/python",
        python_version="3.12.11",
        effective_uid=65532,
        effective_gid=65532,
    )
    assert loads_runtime_preflight_receipt(receipt.canonical_file_bytes()) == receipt
    target = tmp_path / "preflight.json"
    write_runtime_preflight_receipt(receipt, target)
    assert target.read_bytes() == receipt.canonical_file_bytes()
    with pytest.raises(RuntimeAttestationError, match="publish runtime preflight"):
        write_runtime_preflight_receipt(receipt, target)
    changed = json.loads(receipt.canonical_file_bytes())
    changed["unknown"] = True
    with pytest.raises(RuntimeAttestationError, match="unexpected"):
        RuntimePreflightReceipt.from_dict(changed)


def test_capture_runtime_preflight_checks_exact_confinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {"HOSTNAME": "fractal-confirmatory", "LANG": "C.UTF-8"}
    mountinfo = (
        "36 25 0:32 / / ro,relatime - overlay overlay ro,lowerdir=/base\n"
        "37 36 0:40 / /input/control ro,nosuid,nodev - ext4 /dev/vda1 ro\n"
        "38 36 0:41 / /output rw,nosuid,nodev - ext4 /dev/vda1 rw\n"
        "39 36 0:42 / /tmp rw,noexec,nosuid,nodev - tmpfs tmpfs "
        "rw,noexec,nosuid,nodev,size=1048576k\n"
    )
    artifact = RuntimeArtifactMount(
        root="/input/control",
        role="runtime-controls",
        kind="directory",
        artifact_sha256="6" * 64,
    )
    monkeypatch.setattr(runtime_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_module.platform, "release", lambda: "6.12.0-linuxkit")
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(runtime_module.platform, "python_version", lambda: "3.12.11")
    monkeypatch.setattr(runtime_module.socket, "gethostname", lambda: "fractal-confirmatory")
    monkeypatch.setattr(runtime_module.os, "environ", environment)
    monkeypatch.setattr(runtime_module, "_read_text", lambda *_args, **_kwargs: mountinfo)
    monkeypatch.setattr(runtime_module, "_mount_digest", lambda _mount: "6" * 64)
    monkeypatch.setattr(
        runtime_module.os,
        "lstat",
        lambda _path: SimpleNamespace(
            st_mode=runtime_module.stat.S_IFDIR | 0o700,
            st_uid=65532,
            st_gid=65532,
        ),
    )
    monkeypatch.setattr(runtime_module.os, "scandir", lambda _path: iter(()))
    monkeypatch.setattr(
        runtime_module.os,
        "sched_getaffinity",
        lambda _pid: {0, 1, 2, 3},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_module.os,
        "listdir",
        lambda path: ["lo"] if path == "/sys/class/net" else [],
    )
    monkeypatch.setattr(runtime_module, "_route_evidence", lambda: (0, _ROUTES))
    monkeypatch.setattr(runtime_module, "_linux_os_release", lambda: ("debian", "12"))
    monkeypatch.setattr(runtime_module, "_linux_cpu_model", lambda: "Apple M4 Max")
    monkeypatch.setattr(runtime_module, "_linux_memory_limit", lambda: 8 * 1024**3)
    monkeypatch.setattr(runtime_module.os, "geteuid", lambda: 65532)
    monkeypatch.setattr(runtime_module.os, "getegid", lambda: 65532)

    receipt = capture_runtime_preflight(
        launcher_contract_sha256="7" * 64,
        oci_image_digest=_IMAGE,
        code_commit=_COMMIT,
        hostname="fractal-confirmatory",
        artifact_mounts=(artifact,),
        environment=environment,
    )
    assert receipt.artifact_mounts == (ObservedMount(root="/input/control", read_only=True),)
    assert receipt.mount_namespace_sha256 == mount_namespace_sha256(mountinfo)
    assert receipt.environment_sha256 == environment_sha256(environment)


def test_capture_runtime_preflight_rejects_writable_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {"HOSTNAME": "fractal-confirmatory"}
    mountinfo = (
        "36 25 0:32 / / ro,relatime - overlay overlay ro\n"
        "37 36 0:40 / /input/control rw,nosuid,nodev - ext4 /dev/vda1 rw\n"
        "38 36 0:41 / /output rw,nosuid,nodev - ext4 /dev/vda1 rw\n"
        "39 36 0:42 / /tmp rw,noexec,nosuid,nodev - tmpfs tmpfs "
        "rw,noexec,nosuid,nodev,size=1048576k\n"
    )
    artifact = RuntimeArtifactMount(
        root="/input/control",
        role="runtime-controls",
        kind="directory",
        artifact_sha256="6" * 64,
    )
    monkeypatch.setattr(runtime_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_module.socket, "gethostname", lambda: "fractal-confirmatory")
    monkeypatch.setattr(runtime_module.os, "environ", environment)
    monkeypatch.setattr(runtime_module, "_read_text", lambda *_args, **_kwargs: mountinfo)
    with pytest.raises(RuntimeAttestationError, match="artifact mount is writable"):
        capture_runtime_preflight(
            launcher_contract_sha256="7" * 64,
            oci_image_digest=_IMAGE,
            code_commit=_COMMIT,
            hostname="fractal-confirmatory",
            artifact_mounts=(artifact,),
            environment=environment,
        )
