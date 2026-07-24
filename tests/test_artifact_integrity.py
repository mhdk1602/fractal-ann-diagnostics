from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

import fractal_ann_diagnostics.artifact_integrity as integrity
from fractal_ann_diagnostics.artifact_integrity import (
    LOCAL_ARTIFACT_MAP_SCHEMA,
    ArtifactIntegrityError,
    LocalArtifactSpec,
    artifact_specs_from_local_map,
    digest_directory_tree,
    load_local_artifact_map,
    load_verification_receipt,
    verify_local_artifacts,
    write_verification_receipt,
)

MANIFEST_SHA256 = "a" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _file_spec(
    artifact_id: str,
    relative_path: str,
    payload: bytes,
) -> LocalArtifactSpec:
    return LocalArtifactSpec(
        artifact_id=artifact_id,
        relative_path=relative_path,
        kind="file",
        expected_sha256=_sha256(payload),
    )


def test_directory_digest_is_deterministic_and_counts_the_tree(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    (left / "empty").mkdir(parents=True)
    _write(left / "nested" / "b.bin", b"beta")
    _write(left / "a.txt", b"alpha")

    _write(right / "a.txt", b"alpha")
    _write(right / "nested" / "b.bin", b"beta")
    (right / "empty").mkdir()

    first = digest_directory_tree(left)
    second = digest_directory_tree(right)

    assert first.sha256 == second.sha256
    assert first.entries == ("a.txt", "empty", "nested", "nested/b.bin")
    assert first.file_count == 2
    assert first.directory_count == 2
    assert first.byte_count == len(b"alpha") + len(b"beta")
    assert first.file_count == first.observed_file_count
    assert first.directory_count == first.observed_directory_count
    assert first.byte_count == first.observed_byte_count


def test_directory_digest_changes_for_content_or_empty_directory_changes(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    _write(tree / "same-size.bin", b"first")
    original = digest_directory_tree(tree)

    (tree / "same-size.bin").write_bytes(b"other")
    changed_content = digest_directory_tree(tree)
    assert changed_content.sha256 != original.sha256

    (tree / "empty").mkdir()
    changed_structure = digest_directory_tree(tree)
    assert changed_structure.sha256 != changed_content.sha256


def test_verification_receipt_is_canonical_ordered_and_manifest_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    file_payload = b"sealed model"
    _write(root / "model.bin", file_payload)
    _write(root / "dataset" / "a.jsonl", b"one\n")
    (root / "dataset" / "empty").mkdir()
    tree = digest_directory_tree(root / "dataset")

    model = _file_spec("model", "model.bin", file_payload)
    dataset = LocalArtifactSpec(
        artifact_id="dataset",
        relative_path="dataset",
        kind="directory",
        expected_sha256=tree.sha256,
        expected_entries=tree.entries,
    )
    first = verify_local_artifacts(
        root,
        manifest_sha256=MANIFEST_SHA256,
        artifacts=(model, dataset),
    )
    second = verify_local_artifacts(
        root,
        manifest_sha256=MANIFEST_SHA256,
        artifacts=(dataset, model),
    )

    assert [row.artifact_id for row in first.artifacts] == ["dataset", "model"]
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.receipt_sha256 == second.receipt_sha256
    assert json.loads(first.canonical_bytes())["manifest_sha256"] == MANIFEST_SHA256
    assert b"\n" not in first.canonical_bytes()
    dataset_row = first.artifacts[0]
    assert dataset_row.file_count == 1
    assert dataset_row.directory_count == 1
    assert dataset_row.byte_count == len(b"one\n")

    rebound = verify_local_artifacts(
        root,
        manifest_sha256="b" * 64,
        artifacts=(dataset, model),
    )
    assert rebound.receipt_sha256 != first.receipt_sha256


def test_receipt_write_is_exclusive_and_uses_canonical_payload(tmp_path: Path) -> None:
    root = tmp_path / "root"
    payload = b"artifact"
    _write(root / "artifact.bin", payload)
    receipt = verify_local_artifacts(
        root,
        manifest_sha256=MANIFEST_SHA256,
        artifacts=(_file_spec("artifact", "artifact.bin", payload),),
    )
    target = tmp_path / "receipts" / "verification.json"
    target.parent.mkdir()

    write_verification_receipt(receipt, target)

    assert target.read_bytes() == receipt.canonical_bytes() + b"\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(ArtifactIntegrityError, match="already exists"):
        write_verification_receipt(receipt, target)


def test_receipt_write_rejects_shared_writable_parent(tmp_path: Path) -> None:
    root = tmp_path / "root"
    payload = b"artifact"
    _write(root / "artifact.bin", payload)
    receipt = verify_local_artifacts(
        root,
        manifest_sha256=MANIFEST_SHA256,
        artifacts=(_file_spec("artifact", "artifact.bin", payload),),
    )
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    shared.chmod(0o777)
    try:
        with pytest.raises(ArtifactIntegrityError, match="writable by group or other"):
            write_verification_receipt(receipt, shared / "verification.json")
    finally:
        shared.chmod(0o700)


def test_local_artifact_map_is_closed_exact_and_manifest_derived(tmp_path: Path) -> None:
    model_payload = b"model"
    dataset_payload = b"dataset"
    pins = {
        "dataset": _sha256(dataset_payload),
        "model": _sha256(model_payload),
    }
    payload = {
        "schema_version": LOCAL_ARTIFACT_MAP_SCHEMA,
        "artifacts": [
            {
                "artifact_id": "model",
                "relative_path": "model.bin",
                "kind": "file",
            },
            {
                "artifact_id": "dataset",
                "relative_path": "dataset.bin",
                "kind": "file",
            },
        ],
    }

    specs = artifact_specs_from_local_map(payload, expected_sha256_by_id=pins)
    assert [spec.artifact_id for spec in specs] == ["dataset", "model"]
    assert all(spec.exact for spec in specs)
    assert {spec.artifact_id: spec.expected_sha256 for spec in specs} == pins

    map_path = tmp_path / "artifact-map.json"
    map_path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        load_local_artifact_map(
            map_path,
            expected_sha256_by_id=pins,
        )
        == specs
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "missing manifest artifact IDs"),
        ("extra", "unexpected artifact ID"),
        ("duplicate", "duplicate artifact ID"),
        ("unknown-field", "unknown fields"),
    ),
)
def test_local_artifact_map_rejects_inexact_coverage(
    mutation: str,
    message: str,
) -> None:
    pins = {"first": "1" * 64, "second": "2" * 64}
    entries: list[dict[str, object]] = [
        {"artifact_id": "first", "relative_path": "first.bin", "kind": "file"},
        {"artifact_id": "second", "relative_path": "second.bin", "kind": "file"},
    ]
    if mutation == "missing":
        entries.pop()
    elif mutation == "extra":
        entries[1]["artifact_id"] = "third"
    elif mutation == "duplicate":
        entries[1]["artifact_id"] = "first"
    else:
        entries[0]["sha256"] = "1" * 64
    payload = {"schema_version": LOCAL_ARTIFACT_MAP_SCHEMA, "artifacts": entries}

    with pytest.raises(ArtifactIntegrityError, match=message):
        artifact_specs_from_local_map(payload, expected_sha256_by_id=pins)


def test_verification_receipt_loader_requires_canonical_no_follow_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    _write(root / "artifact.bin", b"artifact")
    receipt = verify_local_artifacts(
        root,
        manifest_sha256=MANIFEST_SHA256,
        artifacts=(_file_spec("artifact", "artifact.bin", b"artifact"),),
    )
    receipt_path = tmp_path / "receipt.json"
    write_verification_receipt(receipt, receipt_path)
    loaded = load_verification_receipt(receipt_path)
    assert loaded == receipt

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(receipt.to_dict(), indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="not canonical"):
        load_verification_receipt(noncanonical)

    linked = tmp_path / "linked-receipt.json"
    linked.symlink_to(receipt_path)
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        load_verification_receipt(linked)


def test_control_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    artifact_map = tmp_path / "duplicate-map.json"
    artifact_map.write_bytes(
        b'{"artifacts":[],"schema_version":"fractal-local-artifact-map-v1",'
        b'"schema_version":"fractal-local-artifact-map-v1"}'
    )
    with pytest.raises(ArtifactIntegrityError, match="duplicate key"):
        load_local_artifact_map(
            artifact_map,
            expected_sha256_by_id={"artifact": "a" * 64},
        )


def test_missing_and_changed_files_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    missing = _file_spec("missing", "missing.bin", b"expected")
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(missing,),
        )

    _write(root / "changed.bin", b"actual")
    changed = _file_spec("changed", "changed.bin", b"expected")
    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(changed,),
        )


def test_missing_directory_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    spec = LocalArtifactSpec(
        artifact_id="missing-tree",
        relative_path="absent",
        kind="directory",
        expected_sha256="0" * 64,
    )
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(spec,),
        )


def test_exact_directory_rejects_extra_files_and_empty_directories(tmp_path: Path) -> None:
    root = tmp_path / "root"
    tree = root / "tree"
    _write(tree / "declared.txt", b"declared")
    expected = digest_directory_tree(tree)
    spec = LocalArtifactSpec(
        artifact_id="tree",
        relative_path="tree",
        kind="directory",
        expected_sha256=expected.sha256,
        expected_entries=expected.entries,
    )

    _write(tree / "extra.txt", b"extra")
    with pytest.raises(ArtifactIntegrityError, match="unexpected entries"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(spec,),
        )

    (tree / "extra.txt").unlink()
    (tree / "extra-empty").mkdir()
    with pytest.raises(ArtifactIntegrityError, match="unexpected entries"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(spec,),
        )


def test_exact_directory_rejects_missing_declared_entry(tmp_path: Path) -> None:
    root = tmp_path / "root"
    tree = root / "tree"
    _write(tree / "present.txt", b"present")
    digest = digest_directory_tree(tree)
    spec = LocalArtifactSpec(
        artifact_id="tree",
        relative_path="tree",
        kind="directory",
        expected_sha256=digest.sha256,
        expected_entries=("present.txt", "missing.txt"),
    )
    with pytest.raises(ArtifactIntegrityError, match="missing declared entries"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(spec,),
        )


def test_nonexact_directory_verifies_declared_subset_and_records_observed_counts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    tree = root / "tree"
    _write(tree / "pinned.txt", b"pinned")
    _write(tree / "allowed-extra.txt", b"extra")
    selected = digest_directory_tree(tree, included_entries=("pinned.txt",))
    spec = LocalArtifactSpec(
        artifact_id="tree-subset",
        relative_path="tree",
        kind="directory",
        expected_sha256=selected.sha256,
        exact=False,
        expected_entries=("pinned.txt",),
    )

    receipt = verify_local_artifacts(
        root,
        manifest_sha256=MANIFEST_SHA256,
        artifacts=(spec,),
    )

    row = receipt.artifacts[0]
    assert row.file_count == 1
    assert row.byte_count == len(b"pinned")
    assert row.observed_file_count == 2
    assert row.observed_byte_count == len(b"pinned") + len(b"extra")


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../secret",
        "data/../../secret",
        "/absolute/path",
        "./data",
        "data//file",
        "data/",
        "data\\..\\secret",
        "https://example.com/artifact",
        "s3:sealed-bucket/key",
        "C:/sealed/artifact",
    ],
)
def test_declarations_reject_traversal_remote_and_noncanonical_paths(
    unsafe_path: str,
) -> None:
    with pytest.raises(ArtifactIntegrityError):
        LocalArtifactSpec(
            artifact_id="unsafe",
            relative_path=unsafe_path,
            kind="file",
            expected_sha256="0" * 64,
        )


def test_duplicate_expected_entries_and_unverifiable_subset_are_rejected() -> None:
    with pytest.raises(ArtifactIntegrityError, match="duplicate"):
        LocalArtifactSpec(
            artifact_id="duplicate-members",
            relative_path="tree",
            kind="directory",
            expected_sha256="0" * 64,
            expected_entries=("a", "a"),
        )
    with pytest.raises(ArtifactIntegrityError, match="at least one expected entry"):
        LocalArtifactSpec(
            artifact_id="empty-subset",
            relative_path="tree",
            kind="directory",
            expected_sha256="0" * 64,
            exact=False,
            expected_entries=(),
        )


def test_duplicate_and_overlapping_artifact_declarations_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    _write(root / "a.bin", b"a")
    _write(root / "b.bin", b"b")
    first = _file_spec("same-id", "a.bin", b"a")
    duplicate_id = _file_spec("same-id", "b.bin", b"b")
    with pytest.raises(ArtifactIntegrityError, match="duplicate artifact ID"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(first, duplicate_id),
        )

    duplicate_path = _file_spec("other-id", "a.bin", b"a")
    with pytest.raises(ArtifactIntegrityError, match="duplicate artifact path"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(first, duplicate_path),
        )

    tree_digest = digest_directory_tree(root)
    tree = LocalArtifactSpec(
        artifact_id="tree",
        relative_path="tree",
        kind="directory",
        expected_sha256=tree_digest.sha256,
    )
    nested = _file_spec("nested", "tree/file.bin", b"nested")
    with pytest.raises(ArtifactIntegrityError, match="cannot overlap"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(tree, nested),
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_file_and_ancestor_symlink_escapes_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside_file = tmp_path / "outside.bin"
    outside_file.write_bytes(b"outside")
    (root / "file-link").symlink_to(outside_file)
    direct = _file_spec("direct-link", "file-link", b"outside")
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(direct,),
        )

    outside_directory = tmp_path / "outside-directory"
    outside_directory.mkdir()
    (outside_directory / "secret.bin").write_bytes(b"secret")
    (root / "directory-link").symlink_to(outside_directory, target_is_directory=True)
    ancestor = _file_spec("ancestor-link", "directory-link/secret.bin", b"secret")
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(ancestor,),
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_symlink_anywhere_in_directory_tree_is_rejected(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    _write(tree / "real.bin", b"real")
    (tree / "alias.bin").symlink_to(tree / "real.bin")

    with pytest.raises(ArtifactIntegrityError, match="symlink is forbidden"):
        digest_directory_tree(tree)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_nonexact_directory_still_rejects_an_extra_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    tree = root / "tree"
    _write(tree / "pinned.bin", b"pinned")
    selected = digest_directory_tree(tree, included_entries=("pinned.bin",))
    (tree / "extra-link").symlink_to(tree / "pinned.bin")
    spec = LocalArtifactSpec(
        artifact_id="subset",
        relative_path="tree",
        kind="directory",
        expected_sha256=selected.sha256,
        exact=False,
        expected_entries=("pinned.bin",),
    )

    with pytest.raises(ArtifactIntegrityError, match="symlink is forbidden"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(spec,),
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_symlinked_artifact_root_and_receipt_parent_are_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real-root"
    payload = b"artifact"
    _write(real_root / "artifact.bin", payload)
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    spec = _file_spec("artifact", "artifact.bin", payload)

    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        verify_local_artifacts(
            linked_root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(spec,),
        )

    receipt = verify_local_artifacts(
        real_root,
        manifest_sha256=MANIFEST_SHA256,
        artifacts=(spec,),
    )
    real_receipts = tmp_path / "real-receipts"
    real_receipts.mkdir()
    linked_receipts = tmp_path / "linked-receipts"
    linked_receipts.symlink_to(real_receipts, target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        write_verification_receipt(receipt, linked_receipts / "receipt.json")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs unavailable")
def test_nonregular_tree_entry_is_rejected_without_blocking(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    os.mkfifo(tree / "pipe")

    with pytest.raises(ArtifactIntegrityError, match="non-regular"):
        digest_directory_tree(tree)


@pytest.mark.skipif(not hasattr(os, "link"), reason="hard links unavailable")
def test_hard_link_alias_is_rejected_as_duplicate_physical_input(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    _write(tree / "original.bin", b"same inode")
    os.link(tree / "original.bin", tree / "alias.bin")

    with pytest.raises(ArtifactIntegrityError, match="hard-linked"):
        digest_directory_tree(tree)


def test_wrong_artifact_kind_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root / "regular.bin", b"regular")
    (root / "directory").mkdir()

    declared_directory = LocalArtifactSpec(
        artifact_id="not-a-directory",
        relative_path="regular.bin",
        kind="directory",
        expected_sha256="0" * 64,
    )
    with pytest.raises(ArtifactIntegrityError, match="non-directory"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(declared_directory,),
        )

    declared_file = _file_spec("not-a-file", "directory", b"")
    with pytest.raises(ArtifactIntegrityError, match="non-regular"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(declared_file,),
        )


def test_file_mutation_during_hashing_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    payload = b"x" * (integrity._READ_CHUNK_BYTES + 32)
    target = root / "large.bin"
    _write(target, payload)
    spec = _file_spec("large", "large.bin", payload)
    original_read = integrity.os.read
    mutated = False

    def read_then_mutate(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, size)
        if chunk and not mutated:
            mutated = True
            target.write_bytes(b"y" * len(payload))
        return chunk

    monkeypatch.setattr(integrity.os, "read", read_then_mutate)

    with pytest.raises(ArtifactIntegrityError, match="changed during verification"):
        verify_local_artifacts(
            root,
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(spec,),
        )


def test_invalid_manifest_digest_and_relative_root_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root / "artifact.bin", b"artifact")
    spec = _file_spec("artifact", "artifact.bin", b"artifact")
    with pytest.raises(ArtifactIntegrityError, match="manifest_sha256"):
        verify_local_artifacts(root, manifest_sha256="not-a-digest", artifacts=(spec,))
    with pytest.raises(ArtifactIntegrityError, match="absolute path"):
        verify_local_artifacts(
            Path("relative-root"),
            manifest_sha256=MANIFEST_SHA256,
            artifacts=(spec,),
        )


def test_digest_subset_rejects_missing_entry(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    _write(tree / "present.bin", b"present")
    with pytest.raises(ArtifactIntegrityError, match="missing declared entries"):
        digest_directory_tree(tree, included_entries=("missing.bin",))
