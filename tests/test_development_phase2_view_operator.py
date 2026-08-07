from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fractal_ann_diagnostics import post_embedding_development as exact_p_post
from fractal_ann_diagnostics.post_embedding_development import (
    OPERATOR_CONFIG_FILENAME,
    SELECTION_FILENAME,
    PostEmbeddingDevelopmentConfig,
)
from operators import development_phase2_view as operator


def _digest(value: bytes | str) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"


def _write(path: Path, encoded: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(encoded)
    os.chmod(path, mode)


@pytest.fixture(autouse=True)
def _stub_design_seed_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    def verify_seed_chain(
        *,
        commitment_path: Path,
        commitment_sha256: str,
        reveal_path: Path,
        reveal_sha256: str,
    ) -> tuple[object, object, Path, str]:
        commitment_bytes = commitment_path.read_bytes()
        reveal_bytes = reveal_path.read_bytes()
        if _digest(commitment_bytes) != commitment_sha256:
            raise operator.DevelopmentPhase2ViewError("test commitment digest differs")
        if _digest(reveal_bytes) != reveal_sha256:
            raise operator.DevelopmentPhase2ViewError("test reveal digest differs")
        commitment_values = json.loads(commitment_bytes)
        reveal_values = json.loads(reveal_bytes)
        admission_path = Path(reveal_values["attestation_admission_path"])
        admission_sha256 = reveal_values["attestation_admission_sha256"]
        if _digest(admission_path.read_bytes()) != admission_sha256:
            raise operator.DevelopmentPhase2ViewError("test admission digest differs")
        commitment = SimpleNamespace(
            **commitment_values,
            commitment_sha256=commitment_sha256,
        )
        reveal = SimpleNamespace(
            **reveal_values,
            commitment_sha256=commitment_sha256,
        )
        return commitment, reveal, admission_path, admission_sha256

    monkeypatch.setattr(operator, "_verify_seed_chain", verify_seed_chain)


def _seal_tree(root: Path) -> None:
    for directory, _directory_names, file_names in os.walk(root, topdown=False):
        path = Path(directory)
        for name in file_names:
            member = path / name
            if not member.is_symlink():
                os.chmod(member, 0o400)
        os.chmod(path, 0o500)


def _thaw_tree(root: Path) -> None:
    os.chmod(root, 0o700)
    for directory, directory_names, file_names in os.walk(root):
        path = Path(directory)
        os.chmod(path, 0o700)
        for name in directory_names:
            member = path / name
            if not member.is_symlink():
                os.chmod(member, 0o700)
        for name in file_names:
            member = path / name
            if not member.is_symlink():
                os.chmod(member, 0o600)


@dataclass(frozen=True)
class _PublishedFixture:
    root: Path
    receipt: operator.DevelopmentPhase2ViewReceipt
    payloads: dict[str, bytes]


def _published_fixture(tmp_path: Path) -> _PublishedFixture:
    root = (tmp_path / "development-phase-two").resolve()
    view = root / operator.PHASE2_VIEW_DIRECTORY
    control = (tmp_path / "control").resolve()
    selection = control / "selection-receipt.json"
    selection_bytes = b'{"selection":"independent"}\n'
    _write(selection, selection_bytes, mode=0o400)

    paths = {
        "source_root": (tmp_path / "source").resolve(),
        "partition_audit_path": (control / "partition-audit.json").resolve(),
        "phase1_view_root": (tmp_path / "phase-one").resolve(),
        "seed_commitment_path": (control / "seed-commitment.json").resolve(),
        "seed_attestation_admission_path": (control / "seed-attestation-admission.json").resolve(),
        "seed_reveal_path": (control / "seed-reveal.json").resolve(),
    }
    payloads: dict[str, bytes] = {}
    artifacts: list[operator.Phase2Artifact] = []
    for relative, (role, dataset, stage) in operator._artifact_contract().items():
        encoded = f"{relative}\n".encode("utf-8")
        payloads[relative] = encoded
        _write(view / relative, encoded)
        artifacts.append(
            operator.Phase2Artifact(
                path=relative,
                sha256=_digest(encoded),
                byte_count=len(encoded),
                record_count=1,
                role=role,
                dataset=dataset,
                stage=stage,
            )
        )
    artifact_tuple = tuple(sorted(artifacts, key=lambda row: row.path.encode("utf-8")))
    pins = {
        "staged_inventory_sha256": _digest("inventory"),
        "partition_audit_file_sha256": _digest("audit"),
        "partition_component_membership_sha256": _digest("components"),
        "partition_source_artifact_set_sha256": _digest("source-artifacts"),
        "phase1_view_receipt_sha256": _digest("phase-one"),
        "selection_receipt_sha256": _digest(selection_bytes),
        "design_seed_sha256": _digest("design-seed"),
    }
    admission_bytes = _canonical({"admission": "verified"})
    _write(paths["seed_attestation_admission_path"], admission_bytes, mode=0o400)
    pins["seed_attestation_admission_sha256"] = _digest(admission_bytes)
    commitment_bytes = _canonical(
        {
            "partition_audit_file_sha256": pins["partition_audit_file_sha256"],
            "phase1_view_receipt_sha256": pins["phase1_view_receipt_sha256"],
            "selection_receipt_sha256": pins["selection_receipt_sha256"],
            "staged_inventory_sha256": pins["staged_inventory_sha256"],
        }
    )
    _write(paths["seed_commitment_path"], commitment_bytes, mode=0o400)
    pins["seed_commitment_sha256"] = _digest(commitment_bytes)
    reveal_bytes = _canonical(
        {
            "attestation_admission_path": str(paths["seed_attestation_admission_path"]),
            "attestation_admission_sha256": pins["seed_attestation_admission_sha256"],
            "design_seed_sha256": pins["design_seed_sha256"],
        }
    )
    _write(paths["seed_reveal_path"], reveal_bytes, mode=0o400)
    pins["seed_reveal_sha256"] = _digest(reveal_bytes)
    artifact_set = operator._artifact_set_sha256(artifact_tuple)
    capture_set = operator._capture_set_sha256(
        artifact_tuple,
        partition_audit_file_sha256=pins["partition_audit_file_sha256"],
        phase1_view_receipt_sha256=pins["phase1_view_receipt_sha256"],
        selection_receipt_sha256=pins["selection_receipt_sha256"],
        seed_commitment_sha256=pins["seed_commitment_sha256"],
        seed_attestation_admission_path=paths["seed_attestation_admission_path"],
        seed_attestation_admission_sha256=pins["seed_attestation_admission_sha256"],
        seed_reveal_sha256=pins["seed_reveal_sha256"],
    )
    receipt = operator.DevelopmentPhase2ViewReceipt(
        source_root=paths["source_root"],
        output_root=root,
        partition_audit_path=paths["partition_audit_path"],
        phase1_view_root=paths["phase1_view_root"],
        selection_receipt_path=selection,
        seed_commitment_path=paths["seed_commitment_path"],
        seed_attestation_admission_path=paths["seed_attestation_admission_path"],
        seed_reveal_path=paths["seed_reveal_path"],
        artifacts=artifact_tuple,
        artifact_set_sha256=artifact_set,
        input_custody=operator.Phase2InputCustody(capture_set_sha256=capture_set),
        **pins,
    )
    _write(root / operator.PHASE2_RECEIPT_FILENAME, receipt.canonical_file_bytes())
    _seal_tree(root)
    return _PublishedFixture(root=root, receipt=receipt, payloads=payloads)


def _post_config(
    tmp_path: Path,
    fixture: _PublishedFixture,
    *,
    output_root: Path,
    design_seed_sha256: str | None = None,
) -> tuple[Path, PostEmbeddingDevelopmentConfig]:
    controls = (tmp_path / "post-controls").resolve()
    controls.mkdir(mode=0o700, parents=True, exist_ok=True)
    config = PostEmbeddingDevelopmentConfig(
        production_embedding_config_path=(tmp_path / "embedding-config.json").resolve(),
        production_embedding_config_sha256=_digest("embedding-config"),
        full_staged_root=fixture.receipt.view_root,
        full_staged_inventory_sha256=fixture.receipt.staged_inventory_sha256,
        partition_audit_path=fixture.receipt.partition_audit_path,
        partition_audit_file_sha256=fixture.receipt.partition_audit_file_sha256,
        design_seed_sha256=(
            fixture.receipt.design_seed_sha256 if design_seed_sha256 is None else design_seed_sha256
        ),
        output_root=output_root,
    )
    path = controls / "operator-config.json"
    _write(path, config.canonical_file_bytes(), mode=0o400)
    return path, config


@dataclass(frozen=True)
class _BuildFixture:
    arguments: dict[str, object]
    output: Path
    output_parent: Path


def _build_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _BuildFixture:
    source = (tmp_path / "complete-nfc-source").resolve()
    source.mkdir(mode=0o700)
    rows: list[object] = []
    for relative, (role, dataset, stage) in operator._artifact_contract().items():
        if relative in {"inventory.json", "inventory.sha256"}:
            continue
        encoded = f"{relative}\n".encode()
        _write(source / relative, encoded)
        rows.append(
            operator.SourceArtifact(
                path=relative,
                sha256=_digest(encoded),
                byte_count=len(encoded),
                record_count=1,
                dataset=dataset,
                stage=stage,
                role=role,
                visibility="online",
            )
        )
    source_rows = tuple(sorted(rows, key=lambda row: row.path.encode("utf-8")))
    inventory_bytes = _canonical(
        {
            "artifacts": [row.to_dict() for row in source_rows],
            "assignment_algorithm": {},
            "assignment_seed_sha256": _digest("assignment-seed"),
            "bright_document_identity": {},
            "bright_domains": [],
            "config_sha256": _digest("staging-config"),
            "counts": {},
            "hotpotqa_fullwiki_scope": {},
            "schema_version": operator.INVENTORY_SCHEMA,
            "sources": [],
            "withhold_sealed_labels_from_online_process": True,
        }
    )
    inventory_sha256 = _digest(inventory_bytes)
    _write(source / "inventory.json", inventory_bytes)
    _write(
        source / "inventory.sha256",
        f"{inventory_sha256}  inventory.json\n".encode("ascii"),
    )
    _seal_tree(source)

    controls = (tmp_path / "controls").resolve()
    controls.mkdir(mode=0o700)
    control_bytes = {
        "partition-audit.json": b'{"audit":"fixed"}\n',
        "selection.json": b'{"selection":"fixed"}\n',
        "seed-commitment.json": b'{"commitment":"fixed"}\n',
        "seed-admission.json": b'{"admission":"fixed"}\n',
        "seed-reveal.json": b'{"reveal":"fixed"}\n',
    }
    for name, encoded in control_bytes.items():
        _write(controls / name, encoded, mode=0o400)

    phase1_root = (tmp_path / "phase-one-view").resolve()
    _write(
        phase1_root / operator.PHASE1_RECEIPT_FILENAME,
        b'{"phase_one":"fixed"}\n',
    )
    _seal_tree(phase1_root)
    output_parent = (tmp_path / "phase-two-output").resolve()
    output_parent.mkdir(mode=0o700)
    output = output_parent / "phase-two-v1"

    audit = SimpleNamespace(
        source_artifacts=tuple(
            row for row in source_rows if row.role in {"assignments", "queries", "qrels"}
        ),
        component_membership_sha256=_digest("component-membership"),
        source_artifact_set_sha256=_digest("source-artifact-set"),
    )
    phase1_receipt = SimpleNamespace(artifacts=())
    commitment_sha256 = _digest(control_bytes["seed-commitment.json"])
    commitment = SimpleNamespace(
        staged_inventory_sha256=inventory_sha256,
        partition_audit_file_sha256=_digest(control_bytes["partition-audit.json"]),
        phase1_view_receipt_sha256=_digest("phase-one-receipt"),
        selection_receipt_sha256=_digest(control_bytes["selection.json"]),
    )
    admission_path = controls / "seed-admission.json"
    admission_sha256 = _digest(control_bytes["seed-admission.json"])
    reveal = SimpleNamespace(
        commitment_sha256=commitment_sha256,
        design_seed_sha256=_digest("future-beacon-design-seed"),
    )
    label_free = operator._LabelFreeAdmission(
        audit=audit,
        phase1_receipt=phase1_receipt,
        selection=SimpleNamespace(scope="fixed"),
        seed_commitment=commitment,
        seed_reveal=reveal,
        seed_attestation_admission_path=admission_path,
        seed_attestation_admission_sha256=admission_sha256,
        selection_bytes=control_bytes["selection.json"],
    )
    monkeypatch.setattr(operator, "_admit_label_free_controls", lambda **_kwargs: label_free)
    monkeypatch.setattr(
        operator,
        "_verify_seed_chain",
        lambda **_kwargs: (commitment, reveal, admission_path, admission_sha256),
    )
    arguments: dict[str, object] = {
        "source_root": source,
        "staged_inventory_sha256": inventory_sha256,
        "partition_audit_path": controls / "partition-audit.json",
        "partition_audit_file_sha256": _digest(control_bytes["partition-audit.json"]),
        "phase1_view_root": phase1_root,
        "phase1_view_receipt_sha256": _digest("phase-one-receipt"),
        "selection_receipt_path": controls / "selection.json",
        "selection_receipt_sha256": _digest(control_bytes["selection.json"]),
        "seed_commitment_path": controls / "seed-commitment.json",
        "seed_commitment_sha256": commitment_sha256,
        "seed_reveal_path": controls / "seed-reveal.json",
        "seed_reveal_sha256": _digest(control_bytes["seed-reveal.json"]),
        "output_root": output,
    }
    return _BuildFixture(arguments=arguments, output=output, output_parent=output_parent)


def _add_extended_acl(path: Path) -> object:
    if sys.platform == "darwin":
        subprocess.run(
            ["chmod", "+a", "everyone allow read", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return lambda: subprocess.run(
            ["chmod", "-N", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    if sys.platform.startswith("linux"):
        import struct

        acl = struct.pack("<I", 2) + b"".join(
            struct.pack("<HHI", tag, permissions, identity)
            for tag, permissions, identity in (
                (0x01, 0x04, 0xFFFFFFFF),
                (0x02, 0x00, os.getuid() + 1),
                (0x04, 0x00, 0xFFFFFFFF),
                (0x10, 0x00, 0xFFFFFFFF),
                (0x20, 0x00, 0xFFFFFFFF),
            )
        )
        try:
            os.setxattr(path, "system.posix_acl_access", acl)
        except OSError as exc:
            pytest.skip(f"test filesystem cannot create a POSIX ACL: {exc}")
        return lambda: os.removexattr(path, "system.posix_acl_access")
    pytest.skip("descriptor-bound ACL probes are implemented for macOS and Linux")


def test_closed_contract_is_exact_and_round_trips(tmp_path: Path) -> None:
    fixture = _published_fixture(tmp_path)
    contract = operator._artifact_contract()

    assert len(contract) == 29
    assert sum(path.endswith("/queries.jsonl") for path in contract) == 10
    assert sum(path.endswith("/qrels.jsonl") for path in contract) == 10
    assert sum(path.endswith("/evidence-bundles.jsonl") for path in contract) == 6
    assert not any("sealed" in Path(path).parts for path in contract)
    assert operator.DevelopmentPhase2ViewReceipt.from_dict(fixture.receipt.to_dict()) == (
        fixture.receipt
    )
    assert fixture.receipt.canonical_file_bytes().endswith(b"\n")

    with pytest.raises(operator.DevelopmentPhase2ViewError, match="membership"):
        replace(fixture.receipt, artifacts=fixture.receipt.artifacts[:-1])
    with pytest.raises(operator.DevelopmentPhase2ViewError, match="artifact-set"):
        replace(fixture.receipt, artifact_set_sha256=_digest("another-set"))
    with pytest.raises(operator.DevelopmentPhase2ViewError, match="capture set"):
        replace(
            fixture.receipt,
            seed_attestation_admission_path=(tmp_path / "another-admission.json").resolve(),
        )
    with pytest.raises(operator.DevelopmentPhase2ViewError, match="capture set"):
        replace(
            fixture.receipt,
            seed_attestation_admission_sha256=_digest("another-admission"),
        )
    malformed = fixture.receipt.to_dict()
    malformed["unexpected"] = True
    with pytest.raises(RuntimeError, match="fields differ"):
        operator.DevelopmentPhase2ViewReceipt.from_dict(malformed)


def test_full_build_publishes_and_verifies_the_exact_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch)

    receipt = operator.build_development_phase2_view(**fixture.arguments)
    verified = operator.verify_development_phase2_view(
        fixture.output,
        expected_receipt_sha256=receipt.artifact_sha256,
    )

    assert verified == receipt
    assert len(receipt.artifacts) == 29
    observed_files = {
        path.relative_to(fixture.output).as_posix()
        for path in fixture.output.rglob("*")
        if path.is_file()
    }
    assert observed_files == operator._package_files(receipt.artifacts)
    assert oct(fixture.output.stat().st_mode & 0o777) == "0o500"
    _thaw_tree(fixture.output)


def test_verifier_accepts_exact_sealed_package(tmp_path: Path) -> None:
    fixture = _published_fixture(tmp_path)

    observed = operator.verify_development_phase2_view(
        fixture.root,
        expected_receipt_sha256=fixture.receipt.artifact_sha256,
    )

    assert observed == fixture.receipt
    assert len(observed.artifacts) == 29
    assert oct(fixture.root.stat().st_mode & 0o777) == "0o500"
    assert all(
        oct(path.stat().st_mode & 0o777) == "0o400"
        for path in fixture.root.rglob("*")
        if path.is_file()
    )


def test_verifier_rederives_seed_and_rejects_reveal_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _published_fixture(tmp_path)
    original_verify = operator._verify_seed_chain

    def substituted(**kwargs: object) -> tuple[object, object, Path, str]:
        commitment, reveal, admission_path, admission_sha256 = original_verify(**kwargs)
        changed = SimpleNamespace(**vars(reveal))
        changed.design_seed_sha256 = _digest("substituted-design-seed")
        return commitment, changed, admission_path, admission_sha256

    monkeypatch.setattr(operator, "_verify_seed_chain", substituted)

    with pytest.raises(operator.DevelopmentPhase2ViewError, match="verified reveal"):
        operator.verify_development_phase2_view(
            fixture.root,
            expected_receipt_sha256=fixture.receipt.artifact_sha256,
        )


def test_verifier_rejects_changed_attestation_admission(
    tmp_path: Path,
) -> None:
    fixture = _published_fixture(tmp_path)
    admission = fixture.receipt.seed_attestation_admission_path
    os.chmod(admission, 0o600)
    admission.write_bytes(b'{"admission":"changed"}\n')
    os.chmod(admission, 0o400)

    with pytest.raises(operator.DevelopmentPhase2ViewError, match="admission digest"):
        operator.verify_development_phase2_view(
            fixture.root,
            expected_receipt_sha256=fixture.receipt.artifact_sha256,
        )


def test_verifier_rejects_payload_mutation_and_injection(tmp_path: Path) -> None:
    fixture = _published_fixture(tmp_path)
    target = fixture.root / operator.PHASE2_VIEW_DIRECTORY / "assignments.jsonl"
    _thaw_tree(fixture.root)
    target.write_bytes(b"changed\n")
    injected = fixture.root / operator.PHASE2_VIEW_DIRECTORY / "unexpected.jsonl"
    injected.write_bytes(b"unexpected\n")
    _seal_tree(fixture.root)

    try:
        with pytest.raises(RuntimeError):
            operator.verify_development_phase2_view(
                fixture.root,
                expected_receipt_sha256=fixture.receipt.artifact_sha256,
            )
    finally:
        _thaw_tree(fixture.root)


def test_verifier_rejects_symlinked_member(tmp_path: Path) -> None:
    fixture = _published_fixture(tmp_path)
    target = fixture.root / operator.PHASE2_VIEW_DIRECTORY / "assignments.jsonl"
    external = tmp_path / "external.jsonl"
    external.write_bytes(fixture.payloads["assignments.jsonl"])
    _thaw_tree(fixture.root)
    target.unlink()
    target.symlink_to(external)
    _seal_tree(fixture.root)

    try:
        with pytest.raises(RuntimeError):
            operator.verify_development_phase2_view(
                fixture.root,
                expected_receipt_sha256=fixture.receipt.artifact_sha256,
            )
    finally:
        _thaw_tree(fixture.root)
        target.unlink(missing_ok=True)


def test_verifier_rejects_hard_linked_member(tmp_path: Path) -> None:
    fixture = _published_fixture(tmp_path)
    target = fixture.root / operator.PHASE2_VIEW_DIRECTORY / "assignments.jsonl"
    external = tmp_path / "external.jsonl"
    external.write_bytes(fixture.payloads["assignments.jsonl"])
    _thaw_tree(fixture.root)
    target.unlink()
    os.link(external, target)
    os.chmod(target, 0o400)
    _seal_tree(fixture.root)

    try:
        with pytest.raises(RuntimeError, match="link"):
            operator.verify_development_phase2_view(
                fixture.root,
                expected_receipt_sha256=fixture.receipt.artifact_sha256,
            )
    finally:
        _thaw_tree(fixture.root)


def test_verifier_rejects_an_extended_acl_on_the_published_package(tmp_path: Path) -> None:
    fixture = _published_fixture(tmp_path)
    target = fixture.root / operator.PHASE2_VIEW_DIRECTORY / "assignments.jsonl"
    cleanup = _add_extended_acl(target)

    try:
        with pytest.raises(operator.DevelopmentPhase2ViewError, match="ACL"):
            operator.verify_development_phase2_view(
                fixture.root,
                expected_receipt_sha256=fixture.receipt.artifact_sha256,
            )
    finally:
        cleanup()
        _thaw_tree(fixture.root)


def test_label_free_gate_finishes_before_source_root_is_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_parent = (tmp_path / "output").resolve()
    output_parent.mkdir(mode=0o700)
    sequence: list[str] = []
    original_open = operator._open_absolute_directory
    admission_path = (tmp_path / "seed-attestation-admission.json").resolve()
    seed_reveal = SimpleNamespace(design_seed_sha256="6" * 64)
    phase1_receipt = SimpleNamespace(artifacts=())
    label_free = operator._LabelFreeAdmission(
        audit=None,
        phase1_receipt=phase1_receipt,
        selection=None,
        seed_commitment=None,
        seed_reveal=seed_reveal,
        seed_attestation_admission_path=admission_path,
        seed_attestation_admission_sha256="6" * 64,
        selection_bytes=b"selection\n",
    )

    def fake_label_free(**_kwargs: object) -> operator._LabelFreeAdmission:
        sequence.append("label-free")
        return label_free

    class SourceReached(RuntimeError):
        pass

    def observed_open(path: Path, *, label: str, **kwargs: object) -> tuple[int, os.stat_result]:
        if label == "complete NFC source root":
            assert sequence[-1] == "label-free"
            assert sequence.count("label-free") == 2
            assert "lease:design-seed attestation admission" in sequence
            sequence.append("source")
            raise SourceReached
        if label == "phase-one view root":
            descriptor = os.open(output_parent, os.O_RDONLY | os.O_DIRECTORY)
            return descriptor, os.fstat(descriptor)
        return original_open(path, label=label, **kwargs)

    def lock_control(
        _path: Path,
        *,
        expected_sha256: str,
        label: str,
        leases: object,
        acl_guard: object,
    ) -> None:
        del expected_sha256, leases, acl_guard
        sequence.append(f"lease:{label}")

    monkeypatch.setattr(operator, "_require_nonroot", lambda: None)
    monkeypatch.setattr(operator, "_admit_label_free_controls", fake_label_free)
    monkeypatch.setattr(operator, "_open_absolute_directory", observed_open)
    monkeypatch.setattr(operator, "_lock_control", lock_control)
    monkeypatch.setattr(operator, "_retain_exact_tree_leases_acl", lambda *args, **kwargs: None)

    with pytest.raises(SourceReached):
        operator.build_development_phase2_view(
            source_root=(tmp_path / "source").resolve(),
            staged_inventory_sha256="0" * 64,
            partition_audit_path=(tmp_path / "audit.json").resolve(),
            partition_audit_file_sha256="1" * 64,
            phase1_view_root=(tmp_path / "phase-one").resolve(),
            phase1_view_receipt_sha256="2" * 64,
            selection_receipt_path=(tmp_path / "selection.json").resolve(),
            selection_receipt_sha256="3" * 64,
            seed_commitment_path=(tmp_path / "commitment.json").resolve(),
            seed_commitment_sha256="4" * 64,
            seed_reveal_path=(tmp_path / "reveal.json").resolve(),
            seed_reveal_sha256="5" * 64,
            output_root=output_parent / "phase-two",
        )

    assert sequence[-2:] == ["label-free", "source"]
    assert not (output_parent / "phase-two").exists()


def test_changed_admission_under_lease_blocks_source_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_parent = (tmp_path / "output").resolve()
    output_parent.mkdir(mode=0o700)
    original_open = operator._open_absolute_directory
    source_opened = False
    admission_path = (tmp_path / "seed-attestation-admission.json").resolve()
    phase1_receipt = SimpleNamespace(artifacts=())
    first = operator._LabelFreeAdmission(
        audit=None,
        phase1_receipt=phase1_receipt,
        selection=None,
        seed_commitment=None,
        seed_reveal=SimpleNamespace(design_seed_sha256="6" * 64),
        seed_attestation_admission_path=admission_path,
        seed_attestation_admission_sha256="6" * 64,
        selection_bytes=b"selection\n",
    )
    second = replace(first, seed_attestation_admission_sha256="7" * 64)
    admissions = iter((first, second))

    def observed_open(
        path: Path,
        *,
        label: str,
        **kwargs: object,
    ) -> tuple[int, os.stat_result]:
        nonlocal source_opened
        if label == "complete NFC source root":
            source_opened = True
            raise AssertionError("label-bearing source opened after admission drift")
        if label == "phase-one view root":
            descriptor = os.open(output_parent, os.O_RDONLY | os.O_DIRECTORY)
            return descriptor, os.fstat(descriptor)
        return original_open(path, label=label, **kwargs)

    monkeypatch.setattr(operator, "_require_nonroot", lambda: None)
    monkeypatch.setattr(
        operator,
        "_admit_label_free_controls",
        lambda **_kwargs: next(admissions),
    )
    monkeypatch.setattr(operator, "_open_absolute_directory", observed_open)
    monkeypatch.setattr(operator, "_lock_control", lambda *args, **kwargs: None)
    monkeypatch.setattr(operator, "_retain_exact_tree_leases_acl", lambda *args, **kwargs: None)

    with pytest.raises(operator.DevelopmentPhase2ViewError, match="controls changed"):
        operator.build_development_phase2_view(
            source_root=(tmp_path / "source").resolve(),
            staged_inventory_sha256="0" * 64,
            partition_audit_path=(tmp_path / "audit.json").resolve(),
            partition_audit_file_sha256="1" * 64,
            phase1_view_root=(tmp_path / "phase-one").resolve(),
            phase1_view_receipt_sha256="2" * 64,
            selection_receipt_path=(tmp_path / "selection.json").resolve(),
            selection_receipt_sha256="3" * 64,
            seed_commitment_path=(tmp_path / "commitment.json").resolve(),
            seed_commitment_sha256="4" * 64,
            seed_reveal_path=(tmp_path / "reveal.json").resolve(),
            seed_reveal_sha256="5" * 64,
            output_root=output_parent / "phase-two",
        )

    assert source_opened is False
    assert not (output_parent / "phase-two").exists()


def test_indeterminate_phase_two_rollback_preserves_the_pinned_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path, monkeypatch)
    original_rename = operator._rename_exclusive_at
    original_classify = operator._classify_no_replace_move

    def fail_published_verification(*_args: object, **_kwargs: object) -> object:
        raise operator.DevelopmentPhase2ViewError("injected published verification failure")

    def move_then_interrupt(*args: object, **kwargs: object) -> None:
        original_rename(*args, **kwargs)  # type: ignore[arg-type]
        raise KeyboardInterrupt("injected rollback interruption")

    def interrupt_rollback_classification(**kwargs: object) -> bool:
        if kwargs["label"] == "phase-two rollback":
            raise operator.DevelopmentPhase2PublicationIndeterminate(
                "injected rollback classification interruption"
            )
        return original_classify(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(operator, "verify_development_phase2_view", fail_published_verification)
    monkeypatch.setattr(operator, "_rename_exclusive_at", move_then_interrupt)
    monkeypatch.setattr(operator, "_classify_no_replace_move", interrupt_rollback_classification)

    with pytest.raises(operator.DevelopmentPhase2PublicationIndeterminate):
        operator.build_development_phase2_view(**fixture.arguments)

    assert not fixture.output.exists()
    retained = list(fixture.output_parent.glob(".phase-two-v1.development-view-*"))
    assert len(retained) == 1
    assert (retained[0] / operator.PHASE2_RECEIPT_FILENAME).is_file()
    _thaw_tree(retained[0])


def test_exact_p_resume_bootstrap_publishes_only_config_and_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _published_fixture(tmp_path)
    output_parent = (tmp_path / "post-output").resolve()
    output_parent.mkdir(mode=0o700)
    output = output_parent / "operator-v1"
    receipt_output = (tmp_path / "post-controls" / "bootstrap-receipt.json").resolve()
    config_path, config = _post_config(tmp_path, fixture, output_root=output)
    seed_verifications: list[tuple[Path, Path]] = []
    original_seed_verifier = operator._verify_seed_chain

    def observed_seed_verifier(**kwargs: object) -> tuple[object, object, Path, str]:
        seed_verifications.append(
            (kwargs["commitment_path"], kwargs["reveal_path"])  # type: ignore[arg-type]
        )
        return original_seed_verifier(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(operator, "_verify_seed_chain", observed_seed_verifier)

    receipt = operator.bootstrap_post_embedding_resume(
        phase2_root=fixture.root,
        phase2_receipt_sha256=fixture.receipt.artifact_sha256,
        post_config_path=config_path,
        post_config_sha256=config.file_sha256,
        post_output_root=output,
        bootstrap_receipt_output=receipt_output,
    )

    assert {path.name for path in output.iterdir()} == {
        OPERATOR_CONFIG_FILENAME,
        SELECTION_FILENAME,
    }
    assert (output / OPERATOR_CONFIG_FILENAME).read_bytes() == config.canonical_file_bytes()
    assert (output / SELECTION_FILENAME).read_bytes() == (
        fixture.receipt.selection_receipt_path.read_bytes()
    )
    assert receipt.initial_artifact_set_sha256 == operator._initial_post_artifact_set_sha256(
        config.canonical_file_bytes(),
        fixture.receipt.selection_receipt_path.read_bytes(),
    )
    assert receipt_output.read_bytes() == receipt.canonical_file_bytes()
    assert oct(receipt_output.stat().st_mode & 0o777) == "0o400"
    assert seed_verifications == [
        (fixture.receipt.seed_commitment_path, fixture.receipt.seed_reveal_path)
    ]

    class ExactPResumeReached(RuntimeError):
        pass

    monkeypatch.setattr(
        exact_p_post,
        "_admit_upstream",
        lambda _config: (_ for _ in ()).throw(ExactPResumeReached),
    )
    with pytest.raises(ExactPResumeReached):
        exact_p_post.resume_post_embedding_development(config)


def test_interrupted_bootstrap_receipt_publish_preserves_the_sealed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = (tmp_path / "receipt-parent").resolve()
    parent.mkdir(mode=0o700)
    receipt_path = parent / "bootstrap-receipt.json"
    encoded = b'{"bootstrap":"fixed"}\n'
    original_rename = operator._rename_exclusive_at

    def move_then_interrupt(*args: object, **kwargs: object) -> None:
        original_rename(*args, **kwargs)  # type: ignore[arg-type]
        raise KeyboardInterrupt("injected receipt publication interruption")

    monkeypatch.setattr(operator, "_rename_exclusive_at", move_then_interrupt)

    with pytest.raises(operator.DevelopmentPhase2PublicationIndeterminate):
        operator._write_external_receipt_exclusive(receipt_path, encoded)

    assert receipt_path.read_bytes() == encoded
    assert oct(receipt_path.stat().st_mode & 0o777) == "0o400"
    assert not list(parent.glob(".*.tmp-*"))


def test_indeterminate_bootstrap_rollback_preserves_the_pinned_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _published_fixture(tmp_path)
    output_parent = (tmp_path / "post-output").resolve()
    output_parent.mkdir(mode=0o700)
    output = output_parent / "operator-v1"
    receipt_output = (tmp_path / "post-controls" / "bootstrap-receipt.json").resolve()
    config_path, config = _post_config(tmp_path, fixture, output_root=output)
    original_rename = operator._rename_exclusive_at
    original_classify = operator._classify_no_replace_move
    rename_count = 0

    def fail_receipt_publish(_path: Path, _encoded: bytes) -> None:
        raise operator.DevelopmentPhase2ViewError("injected receipt publication failure")

    def interrupt_second_move(*args: object, **kwargs: object) -> None:
        nonlocal rename_count
        rename_count += 1
        original_rename(*args, **kwargs)  # type: ignore[arg-type]
        if rename_count == 2:
            raise KeyboardInterrupt("injected prefix rollback interruption")

    def interrupt_rollback_classification(**kwargs: object) -> bool:
        if kwargs["label"] == "post-resume prefix rollback":
            raise operator.DevelopmentPhase2PublicationIndeterminate(
                "injected prefix rollback classification interruption"
            )
        return original_classify(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(operator, "_write_external_receipt_exclusive", fail_receipt_publish)
    monkeypatch.setattr(operator, "_rename_exclusive_at", interrupt_second_move)
    monkeypatch.setattr(operator, "_classify_no_replace_move", interrupt_rollback_classification)

    with pytest.raises(operator.DevelopmentPhase2PublicationIndeterminate):
        operator.bootstrap_post_embedding_resume(
            phase2_root=fixture.root,
            phase2_receipt_sha256=fixture.receipt.artifact_sha256,
            post_config_path=config_path,
            post_config_sha256=config.file_sha256,
            post_output_root=output,
            bootstrap_receipt_output=receipt_output,
        )

    assert not output.exists()
    retained = list(output_parent.glob(".operator-v1.development-view-*"))
    assert len(retained) == 1
    assert {path.name for path in retained[0].iterdir()} == {
        OPERATOR_CONFIG_FILENAME,
        SELECTION_FILENAME,
    }
    _thaw_tree(retained[0])


def test_resume_bootstrap_rejects_config_mismatch_without_output(tmp_path: Path) -> None:
    fixture = _published_fixture(tmp_path)
    output_parent = (tmp_path / "post-output").resolve()
    output_parent.mkdir(mode=0o700)
    output = output_parent / "operator-v1"
    receipt_output = (tmp_path / "post-controls" / "bootstrap-receipt.json").resolve()
    config_path, config = _post_config(
        tmp_path,
        fixture,
        output_root=output,
        design_seed_sha256=_digest("wrong-design-seed"),
    )

    with pytest.raises(operator.DevelopmentPhase2ViewError, match="design_seed_sha256"):
        operator.bootstrap_post_embedding_resume(
            phase2_root=fixture.root,
            phase2_receipt_sha256=fixture.receipt.artifact_sha256,
            post_config_path=config_path,
            post_config_sha256=config.file_sha256,
            post_output_root=output,
            bootstrap_receipt_output=receipt_output,
        )

    assert not output.exists()
    assert not receipt_output.exists()


def test_resume_bootstrap_uses_freshly_verified_reveal_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _published_fixture(tmp_path)
    output_parent = (tmp_path / "post-output").resolve()
    output_parent.mkdir(mode=0o700)
    output = output_parent / "operator-v1"
    receipt_output = (tmp_path / "post-controls" / "bootstrap-receipt.json").resolve()
    config_path, config = _post_config(tmp_path, fixture, output_root=output)
    verified_seed = _digest("freshly-verified-reveal")
    monkeypatch.setattr(
        operator,
        "_verify_development_phase2_view_with_seed",
        lambda *args, **kwargs: (
            fixture.receipt,
            SimpleNamespace(design_seed_sha256=verified_seed),
        ),
    )

    with pytest.raises(operator.DevelopmentPhase2ViewError, match="design_seed_sha256"):
        operator.bootstrap_post_embedding_resume(
            phase2_root=fixture.root,
            phase2_receipt_sha256=fixture.receipt.artifact_sha256,
            post_config_path=config_path,
            post_config_sha256=config.file_sha256,
            post_output_root=output,
            bootstrap_receipt_output=receipt_output,
        )

    assert config.design_seed_sha256 == fixture.receipt.design_seed_sha256
    assert config.design_seed_sha256 != verified_seed
    assert not output.exists()
    assert not receipt_output.exists()


def test_resume_bootstrap_never_overwrites_existing_output(tmp_path: Path) -> None:
    fixture = _published_fixture(tmp_path)
    output_parent = (tmp_path / "post-output").resolve()
    output_parent.mkdir(mode=0o700)
    output = output_parent / "operator-v1"
    receipt_output = (tmp_path / "post-controls" / "bootstrap-receipt.json").resolve()
    config_path, config = _post_config(tmp_path, fixture, output_root=output)
    first = operator.bootstrap_post_embedding_resume(
        phase2_root=fixture.root,
        phase2_receipt_sha256=fixture.receipt.artifact_sha256,
        post_config_path=config_path,
        post_config_sha256=config.file_sha256,
        post_output_root=output,
        bootstrap_receipt_output=receipt_output,
    )
    before = {path.name: path.read_bytes() for path in (*output.iterdir(), receipt_output)}

    with pytest.raises(operator.DevelopmentPhase2ViewError, match="already exists"):
        operator.bootstrap_post_embedding_resume(
            phase2_root=fixture.root,
            phase2_receipt_sha256=fixture.receipt.artifact_sha256,
            post_config_path=config_path,
            post_config_sha256=config.file_sha256,
            post_output_root=output,
            bootstrap_receipt_output=(
                tmp_path / "post-controls" / "second-bootstrap-receipt.json"
            ).resolve(),
        )

    after = {path.name: path.read_bytes() for path in (*output.iterdir(), receipt_output)}
    assert after == before
    assert first.post_output_root == output
