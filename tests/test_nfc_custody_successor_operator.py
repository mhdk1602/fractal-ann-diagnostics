from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from operators import nfc_custody_successor as operator


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _digest(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_bytes(encoded)
    os.chmod(path, 0o600)


def _freeze(root: Path) -> None:
    for directory, _directory_names, file_names in os.walk(root, topdown=False):
        directory_path = Path(directory)
        for name in file_names:
            path = directory_path / name
            if not path.is_symlink():
                os.chmod(path, 0o400)
        os.chmod(directory_path, 0o500)


def _thaw(root: Path) -> None:
    os.chmod(root, 0o700)
    for directory, directory_names, file_names in os.walk(root):
        directory_path = Path(directory)
        os.chmod(directory_path, 0o700)
        for name in directory_names:
            os.chmod(directory_path / name, 0o700)
        for name in file_names:
            path = directory_path / name
            if not path.is_symlink():
                os.chmod(path, 0o600)


def _artifact(
    path: str,
    encoded: bytes,
    *,
    role: str,
    ordinal: int,
) -> dict[str, object]:
    return {
        "byte_count": len(encoded),
        "dataset": f"fixture-{ordinal % 5}",
        "path": path,
        "record_count": encoded.count(b"\n"),
        "role": role,
        "sha256": _digest(encoded),
        "stage": ("fit", "calibration", "sealed")[ordinal % 3],
        "visibility": "custody" if role in {"qrels", "evidence-bundles"} else "online",
    }


def _inventory(artifacts: list[dict[str, object]], *, marker: str) -> dict[str, object]:
    return {
        "artifacts": artifacts,
        "assignment_algorithm": {"fixture": marker},
        "assignment_seed_sha256": _digest(f"seed:{marker}".encode()),
        "bright_document_identity": "fixture",
        "bright_domains": [],
        "config_sha256": _digest(f"config:{marker}".encode()),
        "counts": {},
        "hotpotqa_fullwiki_scope": "fixture",
        "schema_version": operator.INVENTORY_SCHEMA,
        "sources": [],
        "withhold_sealed_labels_from_online_process": True,
    }


@dataclass
class _Fixture:
    projection: Path
    original: Path
    output: Path
    receipt: Path
    successor_inventory_sha256: str
    original_inventory_sha256: str
    projection_receipt_sha256: str
    successor_rows: list[dict[str, object]]
    projected_payloads: dict[str, bytes]
    original_payloads: dict[str, bytes]

    def build(self) -> operator.NfcCustodySuccessorReceipt:
        return operator.build_nfc_custody_successor(
            projection_root=self.projection,
            successor_inventory_sha256=self.successor_inventory_sha256,
            projection_receipt_sha256=self.projection_receipt_sha256,
            original_root=self.original,
            original_inventory_sha256=self.original_inventory_sha256,
            output_root=self.output,
            receipt_output=self.receipt,
            max_total_artifact_bytes=1024 * 1024,
        )

    def verify(self, receipt_sha256: str) -> operator.NfcCustodySuccessorReceipt:
        return operator.verify_nfc_custody_successor(
            projection_root=self.projection,
            successor_inventory_sha256=self.successor_inventory_sha256,
            projection_receipt_sha256=self.projection_receipt_sha256,
            original_root=self.original,
            original_inventory_sha256=self.original_inventory_sha256,
            output_root=self.output,
            receipt_output=self.receipt,
            receipt_sha256=receipt_sha256,
            max_total_artifact_bytes=1024 * 1024,
        )


def _fixture(tmp_path: Path) -> _Fixture:
    projection = (tmp_path / "projection").resolve()
    original = (tmp_path / "original").resolve()
    publication = (tmp_path / "publication").resolve()
    receipts = (tmp_path / "receipts").resolve()
    projection.mkdir(mode=0o700)
    original.mkdir(mode=0o700)
    publication.mkdir(mode=0o700)
    receipts.mkdir(mode=0o700)

    projected_payloads: dict[str, bytes] = {}
    original_payloads: dict[str, bytes] = {}
    successor_rows: list[dict[str, object]] = []
    original_rows: list[dict[str, object]] = []

    for index in range(operator.EXPECTED_PROJECTED_COUNT):
        path = f"projected/member-{index:03d}.jsonl"
        successor = _canonical({"id": index, "text": f"NFC member {index}"})
        prior = _canonical({"id": index, "text": f"prior member {index}"})
        projected_payloads[path] = successor
        original_payloads[path] = prior
        successor_rows.append(_artifact(path, successor, role="corpus", ordinal=index))
        original_rows.append(_artifact(path, prior, role="corpus", ordinal=index))

    outcome_ordinal = operator.EXPECTED_PROJECTED_COUNT
    for role, count in (
        ("qrels", operator.EXPECTED_QREL_COUNT),
        ("evidence-bundles", operator.EXPECTED_EVIDENCE_COUNT),
    ):
        folder = "qrels" if role == "qrels" else "evidence"
        for index in range(count):
            path = f"outcomes/{folder}/member-{index:03d}.jsonl"
            encoded = _canonical({"id": index, "payload": f"opaque {role} {index}"})
            original_payloads[path] = encoded
            row = _artifact(path, encoded, role=role, ordinal=outcome_ordinal)
            successor_rows.append(row)
            original_rows.append(dict(row))
            outcome_ordinal += 1

    successor_rows.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    original_rows.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    for path, encoded in projected_payloads.items():
        _write(projection / path, encoded)
    for path, encoded in original_payloads.items():
        _write(original / path, encoded)

    successor_inventory_bytes = _canonical(_inventory(successor_rows, marker="successor"))
    successor_inventory_sha256 = _digest(successor_inventory_bytes)
    _write(projection / "inventory.json", successor_inventory_bytes)
    _write(
        projection / "inventory.sha256",
        f"{successor_inventory_sha256}  inventory.json\n".encode("ascii"),
    )
    projected_rows = [
        row for row in successor_rows if row["role"] not in {"qrels", "evidence-bundles"}
    ]
    projected_set_sha256 = _digest(_canonical(projected_rows)[:-1])
    projection_receipt_bytes = _canonical(
        {
            "projected_artifact_count": operator.EXPECTED_PROJECTED_COUNT,
            "projected_artifact_set_sha256": projected_set_sha256,
            "projected_artifacts": projected_rows,
            "projection_policy": operator.PROJECTION_POLICY,
            "schema_version": operator.PROJECTION_SCHEMA,
            "source_artifact_count": operator.EXPECTED_ARTIFACT_COUNT,
            "source_inventory_sha256": successor_inventory_sha256,
        }
    )
    projection_receipt_sha256 = _digest(projection_receipt_bytes)
    _write(projection / operator.PROJECTION_RECEIPT_FILENAME, projection_receipt_bytes)

    original_inventory_bytes = _canonical(_inventory(original_rows, marker="original"))
    original_inventory_sha256 = _digest(original_inventory_bytes)
    _write(original / "inventory.json", original_inventory_bytes)
    _write(
        original / "inventory.sha256",
        f"{original_inventory_sha256}  inventory.json\n".encode("ascii"),
    )
    _freeze(projection)
    _freeze(original)
    return _Fixture(
        projection=projection,
        original=original,
        output=publication / "complete-nfc-root",
        receipt=receipts / "nfc-custody-successor-receipt.json",
        successor_inventory_sha256=successor_inventory_sha256,
        original_inventory_sha256=original_inventory_sha256,
        projection_receipt_sha256=projection_receipt_sha256,
        successor_rows=successor_rows,
        projected_payloads=projected_payloads,
        original_payloads=original_payloads,
    )


def test_build_and_verify_preserve_the_86_24_source_split(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    receipt = fixture.build()

    assert receipt.artifact_count == 110
    assert receipt.projected_artifact_count == 86
    assert receipt.custody_artifact_count == 24
    assert fixture.receipt.parent != fixture.output
    assert not (fixture.output / fixture.receipt.name).exists()
    assert len([path for path in fixture.output.rglob("*") if path.is_file()]) == 112
    assert stat.S_IMODE(fixture.output.stat().st_mode) == 0o500
    assert stat.S_IMODE(fixture.receipt.stat().st_mode) == 0o400
    for path, encoded in fixture.projected_payloads.items():
        assert (fixture.output / path).read_bytes() == encoded
        assert (fixture.output / path).read_bytes() != fixture.original_payloads[path]
    for path, encoded in fixture.original_payloads.items():
        if path.startswith("outcomes/"):
            assert (fixture.output / path).read_bytes() == encoded

    assert fixture.verify(receipt.artifact_sha256) == receipt


def test_rejects_changed_custody_payload(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = next(path for path in fixture.original_payloads if path.startswith("outcomes/"))
    _thaw(fixture.original)
    (fixture.original / target).write_bytes(b"changed\n")
    _freeze(fixture.original)

    with pytest.raises(operator.NfcCustodySuccessorError, match="differs from its inventory"):
        fixture.build()


def test_rejects_original_contract_drift_even_for_a_nonoutcome_role(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _thaw(fixture.original)
    inventory_path = fixture.original / "inventory.json"
    value = json.loads(inventory_path.read_text(encoding="utf-8"))
    value["artifacts"][-1]["role"] = "queries"
    encoded = _canonical(value)
    fixture.original_inventory_sha256 = _digest(encoded)
    inventory_path.write_bytes(encoded)
    (fixture.original / "inventory.sha256").write_text(
        f"{fixture.original_inventory_sha256}  inventory.json\n",
        encoding="ascii",
    )
    _freeze(fixture.original)

    with pytest.raises(operator.NfcCustodySuccessorError, match="contract differs"):
        fixture.build()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_rejects_linked_and_special_source_members(tmp_path: Path, kind: str) -> None:
    fixture = _fixture(tmp_path)
    target_relative = sorted(fixture.projected_payloads)[0]
    target = fixture.projection / target_relative
    other = fixture.projection / sorted(fixture.projected_payloads)[1]
    _thaw(fixture.projection)
    target.unlink()
    if kind == "symlink":
        target.symlink_to(other)
    elif kind == "hardlink":
        os.link(other, target)
    else:
        os.mkfifo(target, mode=0o600)
    _freeze(fixture.projection)

    with pytest.raises(operator.NfcCustodySuccessorError):
        fixture.build()


def test_rejects_unlisted_source_member(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _thaw(fixture.projection)
    _write(fixture.projection / "unexpected.jsonl", b"{}\n")
    _freeze(fixture.projection)

    with pytest.raises(operator.NfcCustodySuccessorError, match="unexpected file"):
        fixture.build()


@pytest.mark.parametrize("occupied", ["output", "receipt"])
def test_no_replace_publication_preserves_existing_destination(
    tmp_path: Path,
    occupied: str,
) -> None:
    fixture = _fixture(tmp_path)
    if occupied == "output":
        fixture.output.mkdir(mode=0o700)
        sentinel = fixture.output / "sentinel"
    else:
        sentinel = fixture.receipt
    sentinel.write_bytes(b"do not replace\n")
    os.chmod(sentinel, 0o600)

    with pytest.raises(operator.NfcCustodySuccessorError, match="already exists"):
        fixture.build()
    assert sentinel.read_bytes() == b"do not replace\n"


def test_rejects_receipt_inside_staged_root(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(operator.NfcCustodySuccessorError, match="outside every staged root"):
        operator.build_nfc_custody_successor(
            projection_root=fixture.projection,
            successor_inventory_sha256=fixture.successor_inventory_sha256,
            projection_receipt_sha256=fixture.projection_receipt_sha256,
            original_root=fixture.original,
            original_inventory_sha256=fixture.original_inventory_sha256,
            output_root=fixture.output,
            receipt_output=fixture.output / "receipt.json",
            max_total_artifact_bytes=1024 * 1024,
        )


def test_rejects_resource_limit_before_publication(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(operator.NfcCustodySuccessorError, match="max_total_artifact_bytes"):
        operator.build_nfc_custody_successor(
            projection_root=fixture.projection,
            successor_inventory_sha256=fixture.successor_inventory_sha256,
            projection_receipt_sha256=fixture.projection_receipt_sha256,
            original_root=fixture.original,
            original_inventory_sha256=fixture.original_inventory_sha256,
            output_root=fixture.output,
            receipt_output=fixture.receipt,
            max_total_artifact_bytes=1,
        )
    assert not fixture.output.exists()
    assert not fixture.receipt.exists()


def test_detects_source_mutation_after_copy_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_copy = operator._copy_pinned_file
    mutated = False

    def mutate_after_copy(**arguments: Any) -> None:
        nonlocal mutated
        original_copy(**arguments)
        if not mutated:
            artifact = arguments["artifact"]
            root = (
                fixture.original
                if artifact.role in {"qrels", "evidence-bundles"}
                else fixture.projection
            )
            path = root / artifact.path
            os.chmod(path, 0o600)
            path.write_bytes(path.read_bytes() + b"mutation\n")
            os.chmod(path, 0o400)
            mutated = True

    monkeypatch.setattr(operator, "_copy_pinned_file", mutate_after_copy)

    with pytest.raises(operator.NfcCustodySuccessorError, match="changed"):
        fixture.build()
    assert not fixture.output.exists()
    assert not fixture.receipt.exists()


def test_receipt_publication_failure_rolls_back_the_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_rename = operator._rename_no_replace
    calls = 0

    def fail_second_rename(*arguments: Any, **keywords: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise operator.NfcCustodySuccessorError("injected receipt publication failure")
        original_rename(*arguments, **keywords)

    monkeypatch.setattr(operator, "_rename_no_replace", fail_second_rename)

    with pytest.raises(
        operator.NfcCustodySuccessorError,
        match="injected receipt publication failure",
    ):
        fixture.build()
    assert not fixture.output.exists()
    assert not fixture.receipt.exists()


def test_post_rename_error_is_classified_and_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_publish = operator._rename_sealed_directory
    calls = 0

    def move_then_fail(*arguments: Any, **keywords: Any) -> None:
        nonlocal calls
        calls += 1
        original_publish(*arguments, **keywords)
        if calls == 1:
            raise operator.NfcCustodySuccessorError("injected post-rename failure")

    monkeypatch.setattr(operator, "_rename_sealed_directory", move_then_fail)

    with pytest.raises(
        operator.NfcCustodySuccessorError,
        match="injected post-rename failure",
    ):
        fixture.build()
    assert not fixture.output.exists()
    assert not fixture.receipt.exists()


def test_interrupted_publication_classification_never_empties_the_published_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_classify = operator._classify_no_replace_move

    def interrupt_output_classification(*arguments: Any, **keywords: Any) -> bool:
        label = keywords.get("label")
        if label in {"output publication", "output publication recovery"}:
            raise operator.NfcCustodyInterrupted(signal.SIGTERM)
        return original_classify(*arguments, **keywords)

    monkeypatch.setattr(operator, "_classify_no_replace_move", interrupt_output_classification)

    with pytest.raises(
        operator.NfcCustodyPublicationIndeterminate,
        match="rollback could not be proved",
    ):
        fixture.build()

    assert fixture.output.is_dir()
    assert (fixture.output / "inventory.json").read_bytes()
    assert len([path for path in fixture.output.rglob("*") if path.is_file()]) == 112
    assert not fixture.receipt.exists()


def test_standalone_verify_holds_the_publication_parent_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    receipt = fixture.build()
    original_verify_output = operator._verify_output
    observed = False

    def prove_lease(**arguments: Any) -> os.stat_result:
        nonlocal observed
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import fcntl, os, sys; "
                    "fd=os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY); "
                    "blocked=False; "
                    "\ntry:\n"
                    " try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                    " except BlockingIOError: blocked=True\n"
                    "finally: os.close(fd)\n"
                    "raise SystemExit(0 if blocked else 1)"
                ),
                str(fixture.output.parent),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert probe.returncode == 0, probe.stderr
        observed = True
        return original_verify_output(**arguments)

    monkeypatch.setattr(operator, "_verify_output", prove_lease)

    assert fixture.verify(receipt.artifact_sha256) == receipt
    assert observed


def test_standalone_verify_rejects_output_name_rebinding_after_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    receipt = fixture.build()
    original_verify_output = operator._verify_output
    displaced = fixture.output.with_name(f"{fixture.output.name}.displaced")

    def replace_after_proof(**arguments: Any) -> os.stat_result:
        proved = original_verify_output(**arguments)
        fixture.output.rename(displaced)
        fixture.output.mkdir(mode=0o500)
        return proved

    monkeypatch.setattr(operator, "_verify_output", replace_after_proof)

    with pytest.raises(operator.NfcCustodySuccessorError, match="path after proof changed"):
        fixture.verify(receipt.artifact_sha256)

    os.chmod(fixture.output, 0o700)
    _thaw(displaced)


@pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"},
    reason="descriptor-bound ACL probes are implemented for macOS and Linux",
)
def test_rejects_extended_acl_on_a_sealed_source_file(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = fixture.projection / sorted(fixture.projected_payloads)[0]

    if sys.platform == "darwin":
        subprocess.run(
            ["chmod", "+a", "everyone allow read", str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        cleanup = lambda: subprocess.run(  # noqa: E731
            ["chmod", "-N", str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
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
            os.setxattr(target, "system.posix_acl_access", acl)
        except OSError as exc:
            pytest.skip(f"test filesystem cannot create a POSIX ACL: {exc}")
        cleanup = lambda: os.removexattr(target, "system.posix_acl_access")  # noqa: E731

    try:
        with pytest.raises(operator.NfcCustodySuccessorError, match="ACL"):
            fixture.build()
    finally:
        cleanup()


def test_rejects_nonprivate_source_boundary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    os.chmod(fixture.projection, 0o550)

    with pytest.raises(operator.NfcCustodySuccessorError, match="mode must be 0500"):
        fixture.build()


def test_cli_build_and_verify_emit_only_control_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    common = [
        "--projection-root",
        str(fixture.projection),
        "--successor-inventory-sha256",
        fixture.successor_inventory_sha256,
        "--projection-receipt-sha256",
        fixture.projection_receipt_sha256,
        "--original-root",
        str(fixture.original),
        "--original-inventory-sha256",
        fixture.original_inventory_sha256,
        "--output-root",
        str(fixture.output),
        "--receipt-output",
        str(fixture.receipt),
        "--max-total-artifact-bytes",
        str(1024 * 1024),
    ]

    assert operator.main(["build", *common]) == 0
    built = json.loads(capsys.readouterr().out)
    assert set(built) == {
        "artifact_count",
        "custody_artifact_count",
        "output_root",
        "projected_artifact_count",
        "receipt_output",
        "receipt_sha256",
        "schema_version",
        "successor_inventory_sha256",
    }
    assert operator.main(["verify", *common, "--receipt-sha256", built["receipt_sha256"]]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified == built
