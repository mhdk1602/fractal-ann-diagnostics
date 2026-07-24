from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import fractal_ann_diagnostics.candidate_manifest_assembler as assembler_module
from fractal_ann_diagnostics.candidate_manifest_assembler import (
    CANDIDATE_ARTIFACT_PIN_INVENTORY_SCHEMA,
    CANDIDATE_ARTIFACT_PIN_RECEIPT_SCHEMA,
    INVENTORY_FILENAME,
    INVENTORY_RECEIPT_FILENAME,
    CandidateArtifactPinInventory,
    CandidateManifestAssemblyError,
    _cross_check_candidate_image_closure,
    _revision_for,
    apply_candidate_artifact_inventory,
    build_candidate_artifact_pin_inventory,
    load_candidate_artifact_pin_inventory,
    load_closed_candidate_manifest_package,
    publish_closed_candidate_manifest,
)
from fractal_ann_diagnostics.freeze_package import layout_from_manifest
from fractal_ann_diagnostics.production_embedding_build import (
    QWEN_CURRENT_REVISION,
    QWEN_CURRENT_TREE_SHA256,
)
from fractal_ann_diagnostics.production_workload_registration import (
    production_workload_file_sha256,
)
from fractal_ann_diagnostics.provider_rehearsal import CandidateImageClosure
from fractal_ann_diagnostics.study import (
    C0_COMMIT_SENTINEL,
    load_study_manifest,
    validate_candidate_rehearsal_manifest,
)

REPOSITORY = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPOSITORY / "research" / "study-manifest.json"
SHA = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
COMMIT = "a" * 40


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _inventory() -> tuple[dict[str, object], CandidateArtifactPinInventory]:
    template = load_study_manifest(TEMPLATE_PATH)
    layouts = layout_from_manifest(template, REPOSITORY)
    template_rows = {row["id"]: row for row in template["artifacts"]}
    rows: list[dict[str, object]] = []
    for position, layout in enumerate(layouts):
        source = template_rows[layout.artifact_id]
        digest = f"{position + 1:064x}"
        uri = source["uri"]
        if uri == "tbd":
            uri = f"file:///controlled/{layout.relative_path}"
        rows.append(
            {
                "artifact_id": layout.artifact_id,
                "byte_count": position + 1,
                "corpus_id": source.get("corpus_id"),
                "directory_count": 0 if layout.kind == "file" else 1,
                "evidence_class": "test-controlled-content",
                "file_count": 1,
                "kind": layout.kind,
                "license": source["license"],
                "relative_path": layout.relative_path,
                "revision": (
                    C0_COMMIT_SENTINEL if layout.role == "source-code" else f"sha256:{digest}"
                ),
                "role": layout.role,
                "sha256": digest,
                "uri": uri,
            }
        )
    template_sha = __import__("hashlib").sha256(_canonical(template)).hexdigest()
    return template, CandidateArtifactPinInventory(template_sha, tuple(rows))


def _write_inventory_directory(root: Path, inventory: CandidateArtifactPinInventory) -> None:
    root.mkdir(mode=0o700)
    (root / INVENTORY_FILENAME).write_bytes(inventory.canonical_file_bytes)
    receipt = {
        "artifact_count": 79,
        "artifact_root": "/controlled",
        "inventory_file_sha256": inventory.file_sha256,
        "repository_root": str(REPOSITORY),
        "schema_version": CANDIDATE_ARTIFACT_PIN_RECEIPT_SCHEMA,
        "template_sha256": inventory.template_sha256,
    }
    (root / INVENTORY_RECEIPT_FILENAME).write_bytes(_canonical(receipt))


def test_inventory_round_trip_and_artifact_application(tmp_path: Path) -> None:
    template, inventory = _inventory()
    root = tmp_path / "inventory"
    _write_inventory_directory(root, inventory)
    assert load_candidate_artifact_pin_inventory(root) == inventory

    candidate = apply_candidate_artifact_inventory(template, inventory)
    assert len(candidate["artifacts"]) == 79
    assert candidate["artifacts"][78]["revision"] == C0_COMMIT_SENTINEL
    assert candidate["artifacts"][30]["uri"].endswith("/policy-workloads/scifact")


def test_inventory_rejects_corpus_swap() -> None:
    _, inventory = _inventory()
    payload = inventory.to_dict()
    first = payload["artifacts"][0]
    second = payload["artifacts"][1]
    first["corpus_id"], second["corpus_id"] = second["corpus_id"], first["corpus_id"]
    with pytest.raises(CandidateManifestAssemblyError, match="corpus order or coverage"):
        CandidateArtifactPinInventory.from_dict(payload)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_inventory_rejects_missing_or_extra_source(mutation: str) -> None:
    _, inventory = _inventory()
    payload = inventory.to_dict()
    if mutation == "missing":
        payload["artifacts"].pop()
        payload["artifact_count"] = 78
    else:
        payload["artifacts"].append(copy.deepcopy(payload["artifacts"][-1]))
        payload["artifacts"][-1]["artifact_id"] = "unexpected-artifact"
        payload["artifact_count"] = 80
    with pytest.raises(CandidateManifestAssemblyError, match="exactly 79|cardinality"):
        CandidateArtifactPinInventory.from_dict(payload)


def test_inventory_rejects_stale_receipt_digest(tmp_path: Path) -> None:
    _, inventory = _inventory()
    root = tmp_path / "inventory"
    _write_inventory_directory(root, inventory)
    receipt = json.loads((root / INVENTORY_RECEIPT_FILENAME).read_text())
    receipt["inventory_file_sha256"] = "f" * 64
    (root / INVENTORY_RECEIPT_FILENAME).write_bytes(_canonical(receipt))
    with pytest.raises(CandidateManifestAssemblyError, match="receipt binding differs"):
        load_candidate_artifact_pin_inventory(root)


def test_inventory_rejects_candidate_production_locator_confusion() -> None:
    template, inventory = _inventory()
    payload = inventory.to_dict()
    policy = next(row for row in payload["artifacts"] if row["role"] == "policy-workload")
    policy["uri"] = "file:///controlled/production/policy-workloads/scifact-candidate"
    confused = CandidateArtifactPinInventory.from_dict(payload)
    with pytest.raises(CandidateManifestAssemblyError, match="controlled layout"):
        apply_candidate_artifact_inventory(template, confused)


def test_inventory_publication_is_all_or_nothing(tmp_path: Path) -> None:
    template, inventory = _inventory()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    output = tmp_path / "published"
    layouts = tuple(
        SimpleNamespace(
            artifact_id=row["artifact_id"],
            role=row["role"],
            relative_path=row["relative_path"],
            kind=row["kind"],
        )
        for row in inventory.artifacts
    )
    pins = {str(row["artifact_id"]): row for row in inventory.artifacts}

    def inspect(layout: object, *_args: object) -> dict[str, object]:
        row = pins[layout.artifact_id]
        return {
            "artifact_id": layout.artifact_id,
            "byte_count": row["byte_count"],
            "directory_count": row["directory_count"],
            "file_count": row["file_count"],
            "kind": layout.kind,
            "relative_path": layout.relative_path,
            "revision": row["revision"],
            "role": layout.role,
            "sha256": row["sha256"],
            "source_path": None,
            "source_sha256": None,
            "state": "present",
        }

    calls = 0

    def fail_second_write(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected receipt write failure")
        path.write_bytes(payload)

    with (
        patch(
            "fractal_ann_diagnostics.candidate_manifest_assembler.layout_from_manifest",
            return_value=layouts,
        ),
        patch(
            "fractal_ann_diagnostics.candidate_manifest_assembler.verify_staged_data",
            return_value=SimpleNamespace(inventory_sha256="a" * 64),
        ),
        patch(
            "fractal_ann_diagnostics.candidate_manifest_assembler._inspect_target",
            side_effect=inspect,
        ),
        patch(
            "fractal_ann_diagnostics.candidate_manifest_assembler._inventory_row",
            side_effect=[dict(row) for row in inventory.artifacts],
        ),
        patch(
            "fractal_ann_diagnostics.candidate_manifest_assembler._write_private",
            side_effect=fail_second_write,
        ),
        pytest.raises(OSError, match="injected receipt write failure"),
    ):
        build_candidate_artifact_pin_inventory(
            template_path=TEMPLATE_PATH,
            repository_root=REPOSITORY,
            artifact_root=artifact_root,
            output_directory=output,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".published.*")) == []


def _candidate_publication_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, int, dict[str, bytes]]:
    parent = tmp_path / "controlled-parent"
    parent.mkdir(mode=0o700)
    work = parent / ".candidate-staging"
    work.mkdir(mode=0o700)
    members = {
        "candidate.json": _canonical({"candidate": "expected"}),
        "receipt.json": _canonical({"receipt": "expected"}),
    }
    for name, encoded in members.items():
        path = work / name
        path.write_bytes(encoded)
        path.chmod(0o600)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return work, parent / "published", os.open(work, flags), members


def test_candidate_publication_rejects_permissive_output_parent(tmp_path: Path) -> None:
    work, destination, descriptor, members = _candidate_publication_fixture(tmp_path)
    work.parent.chmod(0o777)
    try:
        with pytest.raises(CandidateManifestAssemblyError, match="owner-controlled"):
            assembler_module._publish_directory_exclusive(
                work,
                destination,
                work_descriptor=descriptor,
                expected_members=members,
            )
    finally:
        os.close(descriptor)
    assert not destination.exists()


def test_candidate_publication_rejects_staging_name_substitution(tmp_path: Path) -> None:
    work, destination, descriptor, members = _candidate_publication_fixture(tmp_path)
    displaced = work.parent / ".displaced-original"
    work.rename(displaced)
    work.mkdir(mode=0o700)
    for name in members:
        path = work / name
        path.write_bytes(_canonical({"hostile": name}))
        path.chmod(0o600)
    try:
        with pytest.raises(CandidateManifestAssemblyError, match="name changed"):
            assembler_module._publish_directory_exclusive(
                work,
                destination,
                work_descriptor=descriptor,
                expected_members=members,
            )
    finally:
        os.close(descriptor)
    assert not destination.exists()


def test_candidate_publication_fsyncs_stage_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work, destination, descriptor, members = _candidate_publication_fixture(tmp_path)
    real_fsync = os.fsync
    real_rename = assembler_module._rename_noreplace_at
    events: list[str] = []

    def recording_fsync(target: int) -> None:
        if target == descriptor:
            events.append("stage-fsync")
        real_fsync(target)

    def recording_rename(parent: int, source: str, output: str) -> None:
        events.append("rename")
        real_rename(parent, source, output)

    monkeypatch.setattr(assembler_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(assembler_module, "_rename_noreplace_at", recording_rename)
    try:
        assembler_module._publish_directory_exclusive(
            work,
            destination,
            work_descriptor=descriptor,
            expected_members=members,
        )
    finally:
        os.close(descriptor)
    assert events.index("stage-fsync") < events.index("rename")


def test_candidate_publication_rejects_post_rename_member_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work, destination, descriptor, members = _candidate_publication_fixture(tmp_path)
    real_rename = assembler_module._rename_noreplace_at

    def substitute_after_rename(parent: int, source: str, output: str) -> None:
        real_rename(parent, source, output)
        (destination / "candidate.json").write_bytes(_canonical({"candidate": "changed"}))

    monkeypatch.setattr(assembler_module, "_rename_noreplace_at", substitute_after_rename)
    try:
        with pytest.raises(CandidateManifestAssemblyError, match="bytes differ"):
            assembler_module._publish_directory_exclusive(
                work,
                destination,
                work_descriptor=descriptor,
                expected_members=members,
            )
    finally:
        os.close(descriptor)


def test_inventory_schema_is_explicit() -> None:
    _, inventory = _inventory()
    assert inventory.schema_version == CANDIDATE_ARTIFACT_PIN_INVENTORY_SCHEMA


def test_inventory_rejects_symlinked_member(tmp_path: Path) -> None:
    _, inventory = _inventory()
    root = tmp_path / "inventory"
    _write_inventory_directory(root, inventory)
    external = tmp_path / "external.json"
    external.write_bytes(inventory.canonical_file_bytes)
    (root / INVENTORY_FILENAME).unlink()
    (root / INVENTORY_FILENAME).symlink_to(external)
    with pytest.raises(CandidateManifestAssemblyError, match="cannot read artifact inventory"):
        load_candidate_artifact_pin_inventory(root)


def test_inventory_rejects_symlinked_template(tmp_path: Path) -> None:
    template_link = tmp_path / "study-manifest.json"
    template_link.symlink_to(TEMPLATE_PATH)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    with pytest.raises(CandidateManifestAssemblyError, match="cannot read manifest template"):
        build_candidate_artifact_pin_inventory(
            template_path=template_link,
            repository_root=REPOSITORY,
            artifact_root=artifact_root,
            output_directory=tmp_path / "output",
        )


def test_upstream_revision_cannot_fall_back_to_tree_digest() -> None:
    staged = _revision_for(
        {"artifact_id": "scifact-sealed-inputs", "role": "sealed-inputs"},
        {"sha256": SHA, "revision": None},
        staged_inventory_revision=f"sha256:{SHA_B}",
    )
    assert staged == (f"sha256:{SHA_B}", "staged-study-data-inventory-revision")

    qwen = _revision_for(
        {"artifact_id": "qwen", "role": "primary-embedding"},
        {"sha256": QWEN_CURRENT_TREE_SHA256, "revision": None},
        staged_inventory_revision=f"sha256:{SHA_B}",
    )
    assert qwen == (QWEN_CURRENT_REVISION, "admitted-upstream-model-revision")
    with pytest.raises(CandidateManifestAssemblyError, match="admitted Qwen revision"):
        _revision_for(
            {"artifact_id": "qwen", "role": "primary-embedding"},
            {"sha256": SHA, "revision": None},
            staged_inventory_revision=f"sha256:{SHA_B}",
        )


def _closure() -> CandidateImageClosure:
    return CandidateImageClosure(
        build_context_tree_sha256=SHA_C,
        candidate_branch="c0-candidate/test",
        candidate_package_checksums_sha256=SHA,
        github_ref="refs/heads/c0-candidate/test",
        github_run_attempt=1,
        github_run_id=10,
        github_sha=COMMIT,
        github_workflow_ref=(
            "mhdk1602/fractal-ann-diagnostics/.github/workflows/"
            "confirmatory-image.yml@refs/heads/c0-candidate/test"
        ),
        github_workflow_sha=COMMIT,
        mode="candidate",
        release_govulncheck_adjudication_sha256=SHA,
        release_image_index_digest=f"sha256:{SHA_B}",
        release_image_reference=(
            "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-release-candidate@"
            f"sha256:{SHA_B}"
        ),
        release_linux_arm64_manifest_digest=f"sha256:{SHA}",
        release_oci_attestation_bundle_sha256=SHA,
        release_oci_attestation_verification_sha256=SHA_B,
        release_reproducibility_receipt_sha256=SHA_C,
        release_security_adjudication_sha256=SHA,
        release_tle_interoperability_receipt_sha256=SHA_B,
        repository="mhdk1602/fractal-ann-diagnostics",
        scientific_image_index_digest=f"sha256:{SHA}",
        scientific_image_reference=(
            f"ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-candidate@sha256:{SHA}"
        ),
        scientific_linux_amd64_manifest_digest=f"sha256:{SHA_B}",
        scientific_linux_amd64_runtime_extraction_sha256=SHA,
        scientific_linux_arm64_manifest_digest=f"sha256:{SHA_C}",
        scientific_linux_arm64_runtime_extraction_sha256=SHA_B,
        scientific_oci_attestation_bundle_sha256=SHA,
        scientific_oci_attestation_verification_sha256=SHA_B,
    )


def test_candidate_image_phase_swap_is_rejected() -> None:
    closure = _closure()
    candidate = {
        "production_workloads": [{"spec": {"runner_image": closure.scientific_image_reference}}],
        "sealed_execution": {
            "runner_image": closure.scientific_image_reference,
            "provider_phase_plans": {
                "online": {
                    "runtime_image": closure.scientific_image_reference,
                    "oci_index_digest": closure.scientific_image_index_digest,
                    "oci_platform_manifest_digest": (
                        closure.scientific_linux_arm64_manifest_digest
                    ),
                },
                "label-release": {
                    "runtime_image": closure.release_image_reference,
                    "oci_index_digest": closure.release_image_index_digest,
                    "oci_platform_manifest_digest": closure.release_linux_arm64_manifest_digest,
                },
                "analysis": {
                    "runtime_image": closure.scientific_image_reference,
                    "oci_index_digest": closure.scientific_image_index_digest,
                    "oci_platform_manifest_digest": (
                        closure.scientific_linux_amd64_manifest_digest
                    ),
                },
            },
        },
    }
    _cross_check_candidate_image_closure(candidate, closure)
    plans = candidate["sealed_execution"]["provider_phase_plans"]
    plans["online"], plans["analysis"] = plans["analysis"], plans["online"]
    with pytest.raises(CandidateManifestAssemblyError, match="online image binding"):
        _cross_check_candidate_image_closure(candidate, closure)


def test_pre_a_package_is_identical_across_later_a_certifications(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(REPOSITORY / "tests" / "test_study.py"))
    candidate = namespace["_candidate_rehearsal_manifest"]()
    closure = _closure()
    sealed = candidate["sealed_execution"]
    sealed["runner_image"] = closure.scientific_image_reference
    plans = sealed["provider_phase_plans"]
    image_bindings = {
        "online": (
            closure.scientific_image_reference,
            closure.scientific_image_index_digest,
            closure.scientific_linux_arm64_manifest_digest,
            closure.scientific_linux_arm64_runtime_extraction_sha256,
        ),
        "label-release": (
            closure.release_image_reference,
            closure.release_image_index_digest,
            closure.release_linux_arm64_manifest_digest,
            closure.release_reproducibility_receipt_sha256,
        ),
        "analysis": (
            closure.scientific_image_reference,
            closure.scientific_image_index_digest,
            closure.scientific_linux_amd64_manifest_digest,
            closure.scientific_linux_amd64_runtime_extraction_sha256,
        ),
    }
    for phase, binding in image_bindings.items():
        plans[phase]["runtime_image"] = binding[0]
        plans[phase]["oci_index_digest"] = binding[1]
        plans[phase]["oci_platform_manifest_digest"] = binding[2]
        plans[phase]["runtime_probe_receipt_sha256"] = binding[3]
    for row in candidate["production_workloads"]:
        row["spec"]["runner_image"] = closure.scientific_image_reference
        row["canonical_file_sha256"] = production_workload_file_sha256(row["spec"])

    inventory_rows = []
    for position, artifact in enumerate(candidate["artifacts"]):
        inventory_rows.append(
            {
                "artifact_id": artifact["id"],
                "byte_count": 1,
                "corpus_id": artifact.get("corpus_id"),
                "directory_count": 0,
                "evidence_class": "typed-test-evidence",
                "file_count": 1,
                "kind": "file",
                "license": artifact["license"],
                "relative_path": f"objects/{position:02d}",
                "revision": artifact["revision"],
                "role": artifact["role"],
                "sha256": artifact["sha256"],
                "uri": artifact["uri"],
            }
        )
    inventory = CandidateArtifactPinInventory(SHA, tuple(inventory_rows))

    first = tmp_path / "pre-a-one"
    second = tmp_path / "pre-a-two"
    publish_closed_candidate_manifest(
        candidate=candidate,
        artifact_inventory=inventory,
        candidate_image_closure=closure,
        output_directory=first,
    )
    publish_closed_candidate_manifest(
        candidate=candidate,
        artifact_inventory=inventory,
        candidate_image_closure=closure,
        output_directory=second,
    )
    for later_a in ("a" * 40, "b" * 40):
        validate_candidate_rehearsal_manifest(candidate, c0_commit=later_a)
    assert (first / "candidate-study-manifest.json").read_bytes() == (
        second / "candidate-study-manifest.json"
    ).read_bytes()
    assert (first / "candidate-manifest-assembly-receipt.json").read_bytes() == (
        second / "candidate-manifest-assembly-receipt.json"
    ).read_bytes()
    receipt = json.loads((first / "candidate-manifest-assembly-receipt.json").read_text())
    assert "c0_commit" not in receipt
    assert (
        receipt["manifest_file_sha256"]
        == hashlib.sha256((first / "candidate-study-manifest.json").read_bytes()).hexdigest()
    )
    admitted = load_closed_candidate_manifest_package(first)
    assert admitted.manifest == candidate
    assert admitted.receipt.to_dict() == receipt


def test_closed_candidate_package_rejects_orphan_and_unlisted_members(tmp_path: Path) -> None:
    package = tmp_path / "candidate-package"
    package.mkdir(mode=0o700)
    manifest = package / "candidate-study-manifest.json"
    manifest.write_bytes(b"{}\n")
    manifest.chmod(0o600)

    with pytest.raises(CandidateManifestAssemblyError, match="membership differs"):
        load_closed_candidate_manifest_package(package)

    extra = package / "candidate-manifest-assembly-receipt.json"
    extra.write_bytes(b"{}\n")
    extra.chmod(0o600)
    (package / "unlisted.json").write_bytes(b"{}\n")
    with pytest.raises(CandidateManifestAssemblyError, match="membership differs"):
        load_closed_candidate_manifest_package(package)
