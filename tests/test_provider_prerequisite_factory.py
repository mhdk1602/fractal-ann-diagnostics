from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import test_provider_state_transport as state_fixtures
import test_zenodo_publication as zenodo_fixtures
from test_provider_phase_runtime import _plan
from test_provider_workflow_orchestration import _environment

import fractal_ann_diagnostics.github_state_attestation as state_attestation_module
import fractal_ann_diagnostics.provider_state_transport as state_transport_module
from fractal_ann_diagnostics.artifact_integrity import digest_directory_tree
from fractal_ann_diagnostics.execution_claim import (
    C1_REGISTRATION_PACKAGE_FILE_COUNT,
    PREREQUISITE_OUTPUT_KEYS,
    ProviderPhasePlan,
    materialize_provider_phase_plan,
    required_execute_runner_labels,
    verify_provider_runner_ready,
)
from fractal_ann_diagnostics.github_artifact_transport import GitHubHttpResponse
from fractal_ann_diagnostics.github_state_attestation import (
    ZENODO_REGISTRY_URI,
    ZENODO_RESERVED_DOI,
    LedgerSnapshot,
)
from fractal_ann_diagnostics.provider_prerequisite_factory import (
    AnonymousZenodoReadbackReceipt,
    HostedPrerequisiteError,
    HostedPrerequisiteServices,
    build_hosted_production_prerequisites,
    materialize_anonymous_c1_package,
)
from fractal_ann_diagnostics.provider_workflow_orchestration import (
    ProviderPrerequisiteReceipt,
    ProviderWorkflowContext,
)
from fractal_ann_diagnostics.study import (
    ProtocolRegistrationReceipt,
    ProtocolRegistryRecord,
)
from fractal_ann_diagnostics.suite_attempt import suite_attempt_id
from fractal_ann_diagnostics.zenodo_publication import (
    ValidatedRegistrationPackage,
    validate_registration_package,
)


def _digest(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_anonymous_readback_receipt_closes_the_27_file_contract() -> None:
    receipt = AnonymousZenodoReadbackReceipt(
        record_id=21_361_837,
        doi=ZENODO_RESERVED_DOI,
        record_uri=ZENODO_REGISTRY_URI,
        published_at_utc="2026-07-14T12:01:00+00:00",
        verified_at_utc="2026-07-14T12:01:01+00:00",
        file_count=C1_REGISTRATION_PACKAGE_FILE_COUNT,
        package_tree_sha256=_digest("package-tree"),
        package_aggregate_sha256=_digest("package-aggregate"),
        public_payload_sha256=_digest("public-payload"),
    )

    assert receipt.schema_version == "fractal-zenodo-anonymous-readback-v2"
    with pytest.raises(HostedPrerequisiteError, match="identity differs"):
        replace(
            receipt,
            file_count=C1_REGISTRATION_PACKAGE_FILE_COUNT - 1,
        )


def _registration_receipt(
    package: ValidatedRegistrationPackage,
) -> ProtocolRegistrationReceipt:
    record = ProtocolRegistryRecord.from_dict(
        json.loads(package.registry_record_bytes.decode("utf-8"))
    )
    return ProtocolRegistrationReceipt(
        manifest_sha256=record.manifest_sha256,
        protocol_version=record.protocol_version,
        registered_at_utc=record.registered_at_utc,
        registry_identity=record.registry_identity,
        registry_uri=record.registry_uri,
        registry_record_sha256=record.record_sha256,
    )


def _align_plan(
    plan: ProviderPhasePlan,
    package: ValidatedRegistrationPackage,
) -> ProviderPhasePlan:
    bootstrap = replace(
        plan.runner_bootstrap_receipt,
        workflow_sha=package.c0_commit,
    )
    return replace(
        plan,
        manifest_sha256=package.manifest_sha256,
        c1_commit=package.c1_commit,
        workflow_sha=package.c0_commit,
        runner_bootstrap_receipt=bootstrap,
        runner_bootstrap_receipt_file_sha256=bootstrap.file_sha256,
    )


def _matching_snapshot(
    root: Path,
    package: ValidatedRegistrationPackage,
) -> tuple[LedgerSnapshot, dict[str, bytes]]:
    descriptor, original, rows = state_fixtures._snapshot(root)
    descriptor = replace(descriptor, expected_signer_digest=package.c0_commit)
    descriptor_bytes = descriptor.canonical_bytes() + b"\n"
    control = replace(
        original.controls[0],
        file_sha256=_digest(descriptor_bytes),
        blob_oid=state_fixtures._blob_oid(descriptor_bytes),
        encoded=descriptor_bytes,
    )
    inventory = (
        json.dumps(
            [control.to_inventory_dict()],
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    receipt = _registration_receipt(package)
    opening = replace(
        original.tip.state.payload,
        protocol_registration_receipt_sha256=receipt.receipt_sha256,
        protocol_registration_receipt_file_sha256=_digest(receipt.canonical_bytes() + b"\n"),
        protocol_registry_record_sha256=package.registry_record_sha256,
        registered_at_utc=receipt.registered_at_utc,
        run_started_at_utc="2026-07-14T12:03:00+00:00",
        code_commit=package.c0_commit,
        attestation_descriptor_sha256=descriptor.descriptor_sha256,
    )
    state = replace(original.tip.state, payload=opening)
    state_bytes = state.canonical_bytes() + b"\n"
    transition = replace(original.tip, state=state, state_bytes=state_bytes)
    snapshot = replace(
        original,
        transitions=(transition,),
        controls=(control,),
        control_inventory_bytes=inventory,
    )
    bundle = state_fixtures._bundle(snapshot, transition)
    evidence = state_fixtures._evidence(descriptor, snapshot, bundle)
    rows.update(
        {
            "000.state.json": state_bytes,
            "000.attestation.json": evidence.canonical_bytes() + b"\n",
            "000.sigstore.bundle.json": bundle,
            "ledger-controls/inventory.json": inventory,
            "ledger-controls/attestation-descriptor.json": descriptor_bytes,
        }
    )
    return snapshot, rows


class _C0ArtifactApi(state_fixtures._ArtifactApi):
    def __init__(self, snapshot: LedgerSnapshot, archive: bytes, c0_commit: str) -> None:
        super().__init__(snapshot, archive)
        self.c0_commit = c0_commit

    def get(self, location: str, *, accept: str) -> GitHubHttpResponse:
        response = super().get(location, accept=accept)
        if response.status != 200 or accept != "application/vnd.github+json":
            return response
        value = json.loads(response.body)

        def replace_commit(item: Any) -> Any:
            if item == "1" * 40:
                return self.c0_commit
            if isinstance(item, list):
                return [replace_commit(row) for row in item]
            if isinstance(item, dict):
                return {key: replace_commit(row) for key, row in item.items()}
            return item

        return GitHubHttpResponse(
            response.status,
            response.headers,
            json.dumps(replace_commit(value)).encode("utf-8"),
        )


class _RunnerApi:
    def __init__(self, plan: ProviderPhasePlan) -> None:
        self.plan = plan
        self.calls = 0
        self.busy_after: int | None = None
        self.duplicate = False

    def get(self, endpoint: str) -> object:
        assert endpoint == f"repos/{self.plan.repository}/actions/runners?per_page=100"
        self.calls += 1
        labels = required_execute_runner_labels(self.plan.runner_bootstrap_receipt.runner_label)
        runner = {
            "id": self.plan.runner_id,
            "name": self.plan.runner_name,
            "os": "macOS",
            "status": "offline",
            "busy": self.busy_after is not None and self.calls >= self.busy_after,
            "labels": [{"name": label} for label in labels],
        }
        rows = [runner, dict(runner)] if self.duplicate else [runner]
        return {"total_count": len(rows), "runners": rows}


def _harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    source = tmp_path / "zenodo-source"
    source.mkdir(mode=0o700)
    original_digest = zenodo_fixtures._digest
    manifest_digest = _digest("manifest")
    monkeypatch.setattr(
        zenodo_fixtures,
        "_digest",
        lambda value: (
            manifest_digest if value == "semantic frozen manifest" else original_digest(value)
        ),
    )
    package_root, _ = zenodo_fixtures._make_package(source, monkeypatch)
    package = validate_registration_package(package_root)
    zenodo = zenodo_fixtures._FakeTransport(package, set(package.inventory))
    zenodo.public["created"] = "2026-07-14T12:02:00+00:00"

    plans = {
        phase: _align_plan(_plan(tmp_path / f"plan-{phase}", phase=phase), package)
        for phase in ("online", "label-release", "analysis")
    }
    plan = plans["online"]
    environment = _environment(phase="online", job="claim")
    environment["GITHUB_SHA"] = package.c0_commit
    environment["GITHUB_WORKFLOW_SHA"] = package.c0_commit
    context = ProviderWorkflowContext.from_environment("online", environment)

    state_root = tmp_path / "state-source"
    state_root.mkdir(mode=0o700)
    snapshot, rows = _matching_snapshot(state_root, package)
    artifact_api = _C0ArtifactApi(
        snapshot,
        state_fixtures._archive(rows),
        package.c0_commit,
    )
    monkeypatch.setattr(
        state_transport_module,
        "load_ledger_snapshot",
        lambda **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        state_attestation_module,
        "load_ledger_snapshot",
        lambda **_kwargs: snapshot,
    )
    attestation_verifier = state_fixtures._AttestationVerifier()

    def materialize_predecessor(
        phase: str,
        attempt: str,
        parent: Path,
        *,
        ledger_api: object,
        artifact_api: object,
    ) -> object:
        return state_transport_module._materialize_provider_predecessor(
            phase,
            attempt,
            parent,
            ledger_api=ledger_api,
            artifact_api=artifact_api,
            attestation_verifier=attestation_verifier,
        )

    services = HostedPrerequisiteServices(
        phase_plan_loader=lambda _path, *, c1_commit: (
            plans if c1_commit == package.c1_commit else {}
        ),
        manifest_loader=lambda _path: {"fixture": "C1 manifest"},
        phase_plan_materializer=materialize_provider_phase_plan,
        runner_readiness_verifier=verify_provider_runner_ready,
        predecessor_materializer=materialize_predecessor,
        snapshot_loader=lambda **_kwargs: snapshot,
        plan_templates_hasher=lambda _manifest: _digest("provider-plan-templates"),
    )
    return SimpleNamespace(
        artifact_api=artifact_api,
        context=context,
        package=package,
        plan=plan,
        plans=plans,
        runner_api=_RunnerApi(plan),
        services=services,
        snapshot=snapshot,
        suite=suite_attempt_id(package.manifest_sha256),
        verifier=zenodo_fixtures._AcceptingVerifier(),
        zenodo=zenodo,
    )


def _build(harness: SimpleNamespace, output_root: Path):
    return build_hosted_production_prerequisites(
        harness.context,
        "online",
        harness.suite,
        output_root,
        verified_at_utc="2026-07-17T12:00:00+00:00",
        runner_api=harness.runner_api,
        ledger_api=object(),
        artifact_api=harness.artifact_api,
        zenodo_transport=harness.zenodo,
        c1_attestation_verifier=harness.verifier,
        services=harness.services,
    )


def test_builds_one_fresh_typed_hosted_prerequisite_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    result = _build(harness, tmp_path / "hosted-prerequisites")

    receipt = ProviderPrerequisiteReceipt(**result.prerequisite_fields())
    assert receipt.phase == "online"
    assert receipt.c1_package_file_count == C1_REGISTRATION_PACKAGE_FILE_COUNT
    assert receipt.predecessor_state == "OPENED"
    assert receipt.predecessor_ledger_tree == harness.snapshot.tip.tree_oid
    assert Path(receipt.c1_package_root).is_dir()
    assert result.plan_materialization_path.read_bytes() == result.plan.canonical_file_bytes()
    assert (
        result.bootstrap_materialization_path.read_bytes()
        == result.bootstrap.canonical_file_bytes()
    )
    assert set(result.execution_output_fields()) == PREREQUISITE_OUTPUT_KEYS - {
        "prerequisite_receipt_path",
        "prerequisite_receipt_sha256",
    }
    assert harness.runner_api.calls == 2
    assert all(call[2] is False for call in harness.zenodo.calls)
    assert result.manifest_rekor_integrated_at_utc == result.registration.record.registered_at_utc
    assert result.registry_record_rekor_integrated_at_utc > (
        result.manifest_rekor_integrated_at_utc
    )
    tree = digest_directory_tree(result.package.root)
    assert (tree.file_count, tree.directory_count) == (
        C1_REGISTRATION_PACKAGE_FILE_COUNT,
        0,
    )
    assert (tree.observed_file_count, tree.observed_directory_count) == (
        C1_REGISTRATION_PACKAGE_FILE_COUNT,
        0,
    )
    result.assert_current()
    assert harness.runner_api.calls == 3


def test_rejects_serialized_context_and_wrong_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    arguments = {
        "verified_at_utc": "2026-07-17T12:00:00+00:00",
        "runner_api": harness.runner_api,
        "ledger_api": object(),
        "artifact_api": harness.artifact_api,
        "zenodo_transport": harness.zenodo,
        "c1_attestation_verifier": harness.verifier,
        "services": harness.services,
    }
    with pytest.raises(HostedPrerequisiteError, match="admitted hosted claim context"):
        build_hosted_production_prerequisites(
            harness.context.identity_dict(),  # type: ignore[arg-type]
            "online",
            harness.suite,
            tmp_path / "serialized-context",
            **arguments,
        )
    with pytest.raises(HostedPrerequisiteError, match="suite-attempt ID"):
        build_hosted_production_prerequisites(
            harness.context,
            "online",
            _digest("another suite"),
            tmp_path / "wrong-suite",
            **arguments,
        )
    assert not (tmp_path / "wrong-suite").exists()


def test_anonymous_materialization_rejects_inventory_race_and_file_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)

    class RacingTransport(zenodo_fixtures._FakeTransport):
        def __init__(self) -> None:
            super().__init__(harness.package, set(harness.package.inventory))
            self.public["created"] = "2026-07-14T12:02:00+00:00"
            self.reads = 0

        def get_json(self, url: str, *, authenticated: bool) -> dict[str, object]:
            value = super().get_json(url, authenticated=authenticated)
            self.reads += 1
            if self.reads == 2:
                return {**value, "created": "2026-07-14T12:02:01+00:00"}
            return value

    with pytest.raises(HostedPrerequisiteError, match="changed during materialization"):
        materialize_anonymous_c1_package(
            tmp_path / "racing-package",
            transport=RacingTransport(),
        )

    class OversizeTransport(zenodo_fixtures._FakeTransport):
        def get_bytes(self, url: str, *, authenticated: bool) -> bytes:
            if url.endswith("/content"):
                return b"x" * (32 * 1024 * 1024 + 1)
            return super().get_bytes(url, authenticated=authenticated)

    oversize = OversizeTransport(harness.package, set(harness.package.inventory))
    oversize.public["created"] = "2026-07-14T12:02:00+00:00"
    with pytest.raises(HostedPrerequisiteError, match="exceeds its bound"):
        materialize_anonymous_c1_package(
            tmp_path / "oversize-package",
            transport=oversize,
        )


def test_rejects_plan_substitution_and_removes_partial_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    hostile_bootstrap = replace(
        harness.plan.runner_bootstrap_receipt,
        workflow_sha="f" * 40,
    )
    hostile = replace(
        harness.plan,
        workflow_sha="f" * 40,
        runner_bootstrap_receipt=hostile_bootstrap,
        runner_bootstrap_receipt_file_sha256=hostile_bootstrap.file_sha256,
    )
    harness.services = replace(
        harness.services,
        phase_plan_loader=lambda _path, *, c1_commit: {
            **harness.plans,
            "online": hostile,
        },
    )
    output = tmp_path / "substituted-plan"
    with pytest.raises(HostedPrerequisiteError, match="differs from package or workflow"):
        _build(harness, output)
    assert not output.exists()


def test_fresh_revalidation_rejects_runner_race_and_removes_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    harness.runner_api.busy_after = 2
    output = tmp_path / "runner-race"
    with pytest.raises(Exception, match="idle singleton"):
        _build(harness, output)
    assert harness.runner_api.calls == 2
    assert not output.exists()


def test_fresh_revalidation_rejects_rekor_receipt_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    output = tmp_path / "rekor-substitution"
    result = _build(harness, output)
    receipt = result.package.root / "registry-attestation-validation.json"
    value = json.loads(receipt.read_bytes())
    value["registry_record_rekor_integrated_at_utc"] = "2026-07-14T12:04:00+00:00"
    receipt.write_bytes(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n"
    )

    with pytest.raises(HostedPrerequisiteError, match="revalidation failed"):
        result.assert_current()


def test_rejects_opened_genesis_from_another_c0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    valid = _build(harness, tmp_path / "valid-prerequisites")
    hostile_snapshot = replace(
        harness.snapshot,
        transitions=(
            replace(
                harness.snapshot.tip,
                state=replace(
                    harness.snapshot.tip.state,
                    payload=replace(
                        harness.snapshot.tip.state.payload,
                        code_commit="f" * 40,
                    ),
                ),
            ),
        ),
    )
    import fractal_ann_diagnostics.provider_prerequisite_factory as factory_module

    with pytest.raises(HostedPrerequisiteError, match="OPENED ledger genesis differs"):
        factory_module._assert_snapshot(
            hostile_snapshot,
            phase="online",
            suite=harness.suite,
            package=harness.package,
            registration=valid.registration,
            predecessor=valid.predecessor,
        )
