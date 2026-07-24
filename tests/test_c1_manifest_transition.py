from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest
from test_study import _COMMIT, _c0_evidence_release_binding, _candidate_rehearsal_manifest

import fractal_ann_diagnostics.c1_manifest_transition as transition_module
from fractal_ann_diagnostics.c0_evidence_release import canonical_apparatus_evidence_bytes
from fractal_ann_diagnostics.c1_manifest_transition import (
    C1_FROZEN_MANIFEST_FILENAME,
    C1_MANIFEST_TRANSITION_RECEIPT_FILENAME,
    C1_MANIFEST_TRANSITION_RECEIPT_SCHEMA,
    C1ManifestTransitionError,
    load_c1_manifest_transition_receipt,
    verify_c1_manifest_transition_receipt_bindings,
    write_c1_manifest_transition,
)
from fractal_ann_diagnostics.candidate_manifest_assembler import (
    ASSEMBLY_RECEIPT_FILENAME,
    CANDIDATE_MANIFEST_FILENAME,
    CandidateManifestAssemblyReceipt,
)
from fractal_ann_diagnostics.execution_claim import provider_phase_plan_templates_sha256
from fractal_ann_diagnostics.study import (
    C0_COMMIT_SENTINEL,
    manifest_sha256,
    validate_candidate_rehearsal_to_frozen_transition,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _binding_for(
    candidate: dict[str, object],
    *,
    manifest_file_sha256: str,
    assembly_receipt_file_sha256: str,
) -> dict[str, object]:
    binding = _c0_evidence_release_binding(_COMMIT)
    apparatus = binding["apparatus_evidence"]
    assert isinstance(apparatus, dict)
    apparatus["rehearsal_manifest_sha256"] = manifest_sha256(candidate)
    apparatus["candidate_manifest_file_sha256"] = manifest_file_sha256
    apparatus["candidate_manifest_assembly_receipt_file_sha256"] = assembly_receipt_file_sha256
    apparatus["provider_phase_plan_closure_sha256"] = provider_phase_plan_templates_sha256(
        candidate,
        validation_mode="candidate-rehearsal",
        c0_commit=_COMMIT,
    )
    binding["apparatus_evidence_sha256"] = hashlib.sha256(
        canonical_apparatus_evidence_bytes(apparatus)
    ).hexdigest()
    return binding


def _inputs(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    candidate = _candidate_rehearsal_manifest()
    candidate_path = tmp_path / "candidate-package"
    binding_path = tmp_path / "c0-evidence-release.json"
    candidate_path.mkdir(mode=0o700)
    manifest_bytes = _canonical(candidate) + b"\n"
    plans = candidate["sealed_execution"]["provider_phase_plans"]
    receipt = CandidateManifestAssemblyReceipt(
        artifact_count=79,
        artifact_inventory_file_sha256="1" * 64,
        build_context_tree_sha256="2" * 64,
        candidate_image_closure_file_sha256="3" * 64,
        candidate_image_source_commit="4" * 40,
        c0_sentinel_count=13,
        manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_semantic_sha256=manifest_sha256(candidate),
        provider_plan_template_closure_sha256=hashlib.sha256(_canonical(plans)).hexdigest(),
        release_image_index_digest=f"sha256:{'5' * 64}",
        scientific_image_index_digest=f"sha256:{'6' * 64}",
    )
    manifest_path = candidate_path / CANDIDATE_MANIFEST_FILENAME
    receipt_path = candidate_path / ASSEMBLY_RECEIPT_FILENAME
    manifest_path.write_bytes(manifest_bytes)
    receipt_path.write_bytes(receipt.canonical_file_bytes())
    manifest_path.chmod(0o600)
    receipt_path.chmod(0o600)
    binding = _binding_for(
        candidate,
        manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        assembly_receipt_file_sha256=receipt.file_sha256,
    )
    binding_path.write_bytes(_canonical(binding) + b"\n")
    return candidate, binding, candidate_path, binding_path


def _write(
    tmp_path: Path,
    *,
    candidate_path: Path,
    binding_path: Path,
    c0_commit: str = _COMMIT,
):
    return write_c1_manifest_transition(
        candidate_manifest_package_path=candidate_path,
        c0_commit=c0_commit,
        c0_evidence_release_path=binding_path,
        output_directory=tmp_path / "transition",
    )


def test_transition_derives_only_registered_fields_and_reads_back_typed_receipt(
    tmp_path: Path,
) -> None:
    candidate, binding, candidate_path, binding_path = _inputs(tmp_path)
    candidate_before = copy.deepcopy(candidate)

    result = _write(
        tmp_path,
        candidate_path=candidate_path,
        binding_path=binding_path,
    )

    assert candidate == candidate_before
    assert result.frozen_manifest["status"] == "frozen"
    assert result.frozen_manifest["protocol_version"] == "0.3.0"
    assert result.frozen_manifest["freeze_blockers"] == []
    assert C0_COMMIT_SENTINEL not in _canonical(result.frozen_manifest).decode("ascii")
    sealed = result.frozen_manifest["sealed_execution"]
    assert isinstance(sealed, dict)
    assert sealed["c0_evidence_release"] == binding
    validate_candidate_rehearsal_to_frozen_transition(
        candidate,
        result.frozen_manifest,
        c0_commit=_COMMIT,
    )
    assert stat.S_IMODE(result.frozen_manifest_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(result.receipt_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(result.frozen_manifest_path.parent.stat().st_mode) == 0o700
    assert result.frozen_manifest_path.name == C1_FROZEN_MANIFEST_FILENAME
    assert result.receipt_path.name == C1_MANIFEST_TRANSITION_RECEIPT_FILENAME
    assert {path.name for path in result.frozen_manifest_path.parent.iterdir()} == {
        C1_FROZEN_MANIFEST_FILENAME,
        C1_MANIFEST_TRANSITION_RECEIPT_FILENAME,
    }
    assert result.receipt.schema_version == C1_MANIFEST_TRANSITION_RECEIPT_SCHEMA
    assert result.receipt.candidate_manifest_sha256 == manifest_sha256(candidate)
    assert result.receipt.frozen_manifest_sha256 == manifest_sha256(result.frozen_manifest)
    assert load_c1_manifest_transition_receipt(result.receipt_path) == result.receipt
    verify_c1_manifest_transition_receipt_bindings(
        result.receipt,
        frozen_manifest=result.frozen_manifest,
        frozen_manifest_bytes=result.frozen_manifest_path.read_bytes(),
        c0_commit=_COMMIT,
    )
    with pytest.raises(C1ManifestTransitionError, match="differs from the frozen manifest"):
        verify_c1_manifest_transition_receipt_bindings(
            replace(result.receipt, candidate_manifest_sha256="0" * 64),
            frozen_manifest=result.frozen_manifest,
            frozen_manifest_bytes=result.frozen_manifest_path.read_bytes(),
            c0_commit=_COMMIT,
        )


def test_receipt_mode_is_bound_to_same_fd_snapshot_during_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _candidate, _binding, candidate_path, binding_path = _inputs(tmp_path)
    result = _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)
    target = tmp_path / "mode-race-receipt.json"
    replacement = tmp_path / "replacement-receipt.json"
    displaced = tmp_path / "displaced-receipt.json"
    target.write_bytes(result.receipt.canonical_file_bytes())
    target.chmod(0o644)
    replacement.write_bytes(result.receipt.canonical_file_bytes())
    replacement.chmod(0o600)
    real_snapshot = transition_module._read_regular_file_snapshot
    swapped = False

    def swap_name_after_snapshot(*args: object, **kwargs: object):
        nonlocal swapped
        snapshot = real_snapshot(*args, **kwargs)
        if not swapped:
            swapped = True
            target.rename(displaced)
            replacement.rename(target)
        return snapshot

    monkeypatch.setattr(
        transition_module,
        "_read_regular_file_snapshot",
        swap_name_after_snapshot,
    )
    with pytest.raises(C1ManifestTransitionError, match="mode must equal 0600"):
        load_c1_manifest_transition_receipt(target)
    assert swapped
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(displaced.stat().st_mode) == 0o644


@pytest.mark.parametrize(
    "field",
    (
        "c0_evidence_release_file_sha256",
        "candidate_manifest_file_sha256",
        "candidate_manifest_assembly_receipt_file_sha256",
    ),
)
def test_transition_receipt_rejects_each_digest_not_grounded_in_c0(
    tmp_path: Path,
    field: str,
) -> None:
    _candidate, _binding, candidate_path, binding_path = _inputs(tmp_path)
    result = _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)
    forged = replace(result.receipt, **{field: "0" * 64})
    with pytest.raises(C1ManifestTransitionError, match="differs from the frozen manifest"):
        verify_c1_manifest_transition_receipt_bindings(
            forged,
            frozen_manifest=result.frozen_manifest,
            frozen_manifest_bytes=result.frozen_manifest_path.read_bytes(),
            c0_commit=_COMMIT,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("candidate_image_source_commit", "3" * 40),
        ("build_context_tree_sha256", "4" * 64),
        ("candidate_bootstrap_closure_sha256", "5" * 64),
    ),
)
def test_transition_rejects_wrong_p_t_or_d_inside_bound_apparatus(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    _candidate, binding, candidate_path, binding_path = _inputs(tmp_path)
    apparatus = binding["apparatus_evidence"]
    assert isinstance(apparatus, dict)
    apparatus[field] = replacement
    binding_path.write_bytes(_canonical(binding) + b"\n")

    with pytest.raises(C1ManifestTransitionError, match="canonical apparatus evidence"):
        _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)


def test_transition_rejects_wrong_a_in_target_and_apparatus(tmp_path: Path) -> None:
    _candidate, _binding, candidate_path, binding_path = _inputs(tmp_path)
    with pytest.raises(C1ManifestTransitionError, match="target_commit differs from C0"):
        _write(
            tmp_path,
            candidate_path=candidate_path,
            binding_path=binding_path,
            c0_commit="0" * 40,
        )


def test_transition_rejects_wrong_normalized_provider_plan_closure(tmp_path: Path) -> None:
    _candidate, binding, candidate_path, binding_path = _inputs(tmp_path)
    apparatus = binding["apparatus_evidence"]
    assert isinstance(apparatus, dict)
    apparatus["provider_phase_plan_closure_sha256"] = "0" * 64
    binding["apparatus_evidence_sha256"] = hashlib.sha256(
        canonical_apparatus_evidence_bytes(apparatus)
    ).hexdigest()
    binding_path.write_bytes(_canonical(binding) + b"\n")

    with pytest.raises(C1ManifestTransitionError, match="normalized candidate plans"):
        _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)


def test_transition_rejects_candidate_changed_after_c0_evidence(tmp_path: Path) -> None:
    candidate, _binding, candidate_path, binding_path = _inputs(tmp_path)
    sealed = candidate["sealed_execution"]
    assert isinstance(sealed, dict)
    sealed["custodian"] = "changed-custodian@example.test"
    (candidate_path / CANDIDATE_MANIFEST_FILENAME).write_bytes(_canonical(candidate) + b"\n")

    with pytest.raises(
        C1ManifestTransitionError,
        match="manifest differs from its assembly receipt",
    ):
        _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)


def test_transition_rejects_v1_binding_and_extra_sentinel(tmp_path: Path) -> None:
    candidate, binding, candidate_path, binding_path = _inputs(tmp_path)
    binding["schema_version"] = "fractal-c0-evidence-release-binding-v1"
    binding_path.write_bytes(_canonical(binding) + b"\n")
    with pytest.raises(C1ManifestTransitionError, match="schema_version differs from C0"):
        _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)

    binding = _binding_for(
        candidate,
        manifest_file_sha256=hashlib.sha256(
            (candidate_path / CANDIDATE_MANIFEST_FILENAME).read_bytes()
        ).hexdigest(),
        assembly_receipt_file_sha256=hashlib.sha256(
            (candidate_path / ASSEMBLY_RECEIPT_FILENAME).read_bytes()
        ).hexdigest(),
    )
    binding_path.write_bytes(_canonical(binding) + b"\n")
    sealed = candidate["sealed_execution"]
    assert isinstance(sealed, dict)
    sealed["custodian"] = C0_COMMIT_SENTINEL
    (candidate_path / CANDIDATE_MANIFEST_FILENAME).write_bytes(_canonical(candidate) + b"\n")
    with pytest.raises(C1ManifestTransitionError, match="sentinel path set differs"):
        _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)


def test_transition_rejects_noncanonical_and_symlink_inputs(tmp_path: Path) -> None:
    candidate, _binding, candidate_path, binding_path = _inputs(tmp_path)
    (candidate_path / CANDIDATE_MANIFEST_FILENAME).write_text(
        json.dumps(candidate, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(C1ManifestTransitionError, match="bytes are not canonical"):
        _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)

    (candidate_path / CANDIDATE_MANIFEST_FILENAME).write_bytes(_canonical(candidate) + b"\n")
    linked = tmp_path / "candidate-link"
    linked.symlink_to(candidate_path)
    with pytest.raises(C1ManifestTransitionError, match="package path cannot contain links"):
        _write(tmp_path, candidate_path=linked, binding_path=binding_path)


@pytest.mark.parametrize("defect", ("naked", "missing-receipt", "extra", "forged-receipt"))
def test_transition_rejects_package_bypass_and_orphan_inputs(
    tmp_path: Path,
    defect: str,
) -> None:
    _candidate, _binding, candidate_path, binding_path = _inputs(tmp_path)
    supplied = candidate_path
    if defect == "naked":
        supplied = candidate_path / CANDIDATE_MANIFEST_FILENAME
    elif defect == "missing-receipt":
        (candidate_path / ASSEMBLY_RECEIPT_FILENAME).unlink()
    elif defect == "extra":
        (candidate_path / "orphan.json").write_bytes(b"{}\n")
    else:
        receipt_path = candidate_path / ASSEMBLY_RECEIPT_FILENAME
        receipt = json.loads(receipt_path.read_bytes())
        receipt["manifest_file_sha256"] = "0" * 64
        receipt_path.write_bytes(_canonical(receipt) + b"\n")

    with pytest.raises(C1ManifestTransitionError, match="package is not closed"):
        _write(tmp_path, candidate_path=supplied, binding_path=binding_path)


@pytest.mark.parametrize("occupied", ("directory", "file", "link"))
def test_transition_never_replaces_preexisting_or_symlink_output_directory(
    tmp_path: Path,
    occupied: str,
) -> None:
    _candidate, _binding, candidate_path, binding_path = _inputs(tmp_path)
    output = tmp_path / "transition"
    preserved = tmp_path / "preserved"
    preserved.write_bytes(b"preserved")
    if occupied == "directory":
        output.mkdir()
        (output / "existing").write_bytes(b"existing")
    elif occupied == "file":
        output.write_bytes(b"existing")
    else:
        output.symlink_to(preserved)

    with pytest.raises(C1ManifestTransitionError, match="already exists"):
        _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)

    if occupied == "directory":
        assert (output / "existing").read_bytes() == b"existing"
    elif occupied == "file":
        assert output.read_bytes() == b"existing"
    else:
        assert output.is_symlink() and output.read_bytes() == b"preserved"
    assert preserved.read_bytes() == b"preserved"
    assert not list(tmp_path.glob(".transition.tmp-*"))


def test_transition_completes_short_writes_without_partial_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _candidate, _binding, candidate_path, binding_path = _inputs(tmp_path)
    real_write = os.write

    def short_write(descriptor: int, payload: bytes) -> int:
        return real_write(descriptor, payload[:7])

    monkeypatch.setattr(transition_module.os, "write", short_write)
    result = _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)
    assert result.frozen_manifest_path.read_bytes().endswith(b"\n")
    assert result.receipt_path.read_bytes() == result.receipt.canonical_file_bytes()


def test_interrupted_temporary_write_leaves_no_output_or_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _candidate, _binding, candidate_path, binding_path = _inputs(tmp_path)
    real_write = os.write
    calls = 0

    def interrupted_write(descriptor: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write interruption")
        return real_write(descriptor, payload[:11])

    monkeypatch.setattr(transition_module.os, "write", interrupted_write)
    with pytest.raises(OSError, match="injected write interruption"):
        _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)
    assert not (tmp_path / "transition").exists()
    assert not list(tmp_path.glob(".transition.tmp-*"))


def test_publication_uses_one_no_replace_directory_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _candidate, _binding, candidate_path, binding_path = _inputs(tmp_path)
    real_rename = transition_module._rename_noreplace_at
    calls: list[tuple[str, str]] = []

    def recording_rename(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
        *,
        label: str,
    ) -> None:
        calls.append((source_name, destination_name))
        real_rename(
            parent_descriptor,
            source_name,
            destination_name,
            label=label,
        )

    monkeypatch.setattr(transition_module, "_rename_noreplace_at", recording_rename)
    _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)
    assert len(calls) == 1
    assert calls[0][1] == "transition"


@pytest.mark.parametrize("source", ("candidate", "evidence"))
def test_final_input_snapshot_rejects_change_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    candidate, binding, candidate_path, binding_path = _inputs(tmp_path)
    derive = transition_module.derive_frozen_c1_manifest

    def mutate_after_derivation(*args: object, **kwargs: object):
        result = derive(*args, **kwargs)
        if source == "candidate":
            changed = copy.deepcopy(candidate)
            sealed = changed["sealed_execution"]
            assert isinstance(sealed, dict)
            sealed["custodian"] = "post-admission-change@example.test"
            (candidate_path / CANDIDATE_MANIFEST_FILENAME).write_bytes(_canonical(changed) + b"\n")
        else:
            changed_binding = copy.deepcopy(binding)
            changed_binding["asset_size"] = 101
            verification = changed_binding["verification_receipt"]
            assert isinstance(verification, dict)
            verification["anonymous_asset_size"] = 101
            binding_path.write_bytes(_canonical(changed_binding) + b"\n")
        return result

    monkeypatch.setattr(
        transition_module,
        "derive_frozen_c1_manifest",
        mutate_after_derivation,
    )
    with pytest.raises(C1ManifestTransitionError, match="changed before C1 publication"):
        _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)
    assert not (tmp_path / "transition").exists()


def test_evidence_substitution_during_readback_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _candidate, _binding, candidate_path, binding_path = _inputs(tmp_path)
    publish = transition_module._atomic_publish_directory_noreplace

    def substitute_after_publish(
        destination: Path,
        members: dict[str, bytes],
    ) -> dict[str, bytes]:
        readback = dict(publish(destination, members))
        value = json.loads(readback[C1_FROZEN_MANIFEST_FILENAME])
        value["sealed_execution"]["c0_evidence_release"]["asset_sha256"] = "0" * 64
        readback[C1_FROZEN_MANIFEST_FILENAME] = _canonical(value) + b"\n"
        return readback

    monkeypatch.setattr(
        transition_module,
        "_atomic_publish_directory_noreplace",
        substitute_after_publish,
    )
    with pytest.raises(C1ManifestTransitionError, match="readback differs"):
        _write(tmp_path, candidate_path=candidate_path, binding_path=binding_path)
