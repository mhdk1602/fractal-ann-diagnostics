from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from fractal_ann_diagnostics.development_cohort import (
    CALIBRATION_FAMILY_COUNT,
    FIT_FAMILY_COUNT,
    select_development_cohort,
)
from fractal_ann_diagnostics.scalable_partition_audit import (
    ScalableQueryPartitionAuditReceipt,
)
from fractal_ann_diagnostics.study_data import ASSIGNMENT_SCHEMA
from operators import development_staging_view as operator


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


def _digest(value: bytes | str) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_bytes(encoded)
    os.chmod(path, 0o600)


def _freeze_projection(root: Path) -> None:
    for directory, _directory_names, file_names in os.walk(root):
        directory_path = Path(directory)
        for name in file_names:
            os.chmod(directory_path / name, 0o400)
        os.chmod(directory_path, 0o500)


def _thaw_projection(root: Path) -> None:
    for directory, _directory_names, file_names in os.walk(root):
        directory_path = Path(directory)
        os.chmod(directory_path, 0o700)
        for name in file_names:
            path = directory_path / name
            if not path.is_symlink():
                os.chmod(path, 0o600)


def _thaw_view(root: Path) -> None:
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


def _seal_view(root: Path) -> None:
    for directory, _directory_names, file_names in os.walk(root, topdown=False):
        directory_path = Path(directory)
        for name in file_names:
            path = directory_path / name
            if not path.is_symlink():
                os.chmod(path, 0o400)
        os.chmod(directory_path, 0o500)


def _source_row(
    path: str,
    encoded: bytes,
    *,
    role: str,
    dataset: str | None,
    stage: str | None,
    visibility: str = "online",
) -> dict[str, object]:
    return {
        "byte_count": len(encoded),
        "dataset": dataset,
        "path": path,
        "record_count": encoded.count(b"\n"),
        "role": role,
        "sha256": _digest(encoded),
        "stage": stage,
        "visibility": visibility,
    }


@dataclass
class _Fixture:
    projection_root: Path
    partition_audit: Path
    output_parent: Path
    inventory_sha256: str
    projection_receipt_sha256: str
    partition_audit_sha256: str
    projected_rows: list[dict[str, object]]
    inventory_rows: list[dict[str, object]]

    @property
    def output_root(self) -> Path:
        return self.output_parent / "selection-view"


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _fixture(
    tmp_path: Path,
    *,
    fit_count: int = 1,
    calibration_count: int = 1,
    sealed_count: int = 1,
) -> _Fixture:
    projection = (tmp_path / "online-projection").resolve()
    output_parent = (tmp_path / "operator-output").resolve()
    audit_path = (tmp_path / "controls" / "query-partition-audit.json").resolve()
    projection.mkdir(mode=0o700, parents=True)
    output_parent.mkdir(mode=0o700, parents=True)
    audit_path.parent.mkdir(mode=0o700)

    payloads: dict[str, tuple[bytes, str, str | None, str | None, str]] = {
        "partition-exclusions.jsonl": (
            b"",
            "query-partition-structural-exclusions",
            None,
            None,
            "protocol",
        ),
    }
    assignments: list[dict[str, object]] = []
    query_counts_by_stage = {
        "fit": fit_count,
        "calibration": calibration_count,
        "sealed": sealed_count,
    }
    qrel_rows: list[dict[str, object]] = []
    inventory_counts: dict[str, dict[str, int]] = {}
    for corpus in operator.FIXED_CORPORA:
        corpus_path = f"datasets/{corpus}/corpus.jsonl"
        corpus_bytes = _jsonl(
            [
                {
                    "id": f"{corpus}-document",
                    "text": f"Document for {corpus}.",
                    "title": corpus,
                }
            ]
        )
        payloads[corpus_path] = (
            corpus_bytes,
            "corpus",
            corpus,
            None,
            "online",
        )
        inventory_counts[corpus] = {
            "calibration_queries": calibration_count,
            "documents": 1,
            "fit_queries": fit_count,
            "qrels": fit_count + calibration_count + sealed_count,
            "sealed_queries": sealed_count,
            "structural_excluded_queries": 0,
        }
        for stage in operator.REGISTERED_STAGES:
            queries: list[dict[str, object]] = []
            qrels: list[dict[str, object]] = []
            for index in range(query_counts_by_stage[stage]):
                query_id = f"{corpus}-{stage}-{index:04d}"
                text = f"{corpus} {stage} registered query {index:04d}"
                component = _digest(f"component:{query_id}")
                queries.append({"id": query_id, "text": text})
                assignments.append(
                    {
                        "assignment_key_sha256": _digest(f"assignment:{query_id}"),
                        "dataset": corpus,
                        "domain": None,
                        "partition_component_sha256": component,
                        "query_id": query_id,
                        "query_text_sha256": _digest(text),
                        "schema_version": ASSIGNMENT_SCHEMA,
                        "source_split": f"fixture-{stage}",
                        "stage": stage,
                    }
                )
                qrels.append(
                    {
                        "document_id": f"{corpus}-document",
                        "query_id": query_id,
                        "relevance": 1,
                    }
                )
            query_path = (
                f"datasets/{corpus}/sealed/online/queries.jsonl"
                if stage == "sealed"
                else f"datasets/{corpus}/{stage}/queries.jsonl"
            )
            payloads[query_path] = (
                _jsonl(queries),
                "queries",
                corpus,
                stage,
                "online",
            )
            qrel_path = (
                f"datasets/{corpus}/sealed/custody/qrels.jsonl"
                if stage == "sealed"
                else f"datasets/{corpus}/{stage}/qrels.jsonl"
            )
            qrel_encoded = _jsonl(qrels)
            qrel_rows.append(
                _source_row(
                    qrel_path,
                    qrel_encoded,
                    role="qrels",
                    dataset=corpus,
                    stage=stage,
                    visibility="custody" if stage == "sealed" else "online",
                )
            )
    assignments.sort(key=lambda row: str(row["query_id"]).encode("utf-8"))
    payloads["assignments.jsonl"] = (
        _jsonl(assignments),
        "assignments",
        None,
        None,
        "online",
    )

    for stage in operator.DEVELOPMENT_SOURCE_STAGES:
        for corpus in operator.FIXED_CORPORA:
            path = f"datasets/{corpus}/{stage}/queries.jsonl"
            assert path in payloads

    projected_rows: list[dict[str, object]] = []
    for path, (encoded, role, dataset, stage, visibility) in payloads.items():
        _write(projection / path, encoded)
        projected_rows.append(
            _source_row(
                path,
                encoded,
                role=role,
                dataset=dataset,
                stage=stage,
                visibility=visibility,
            )
        )
    projected_rows.sort(key=lambda row: str(row["path"]).encode("utf-8"))

    inventory_rows = sorted(
        [*projected_rows, *qrel_rows],
        key=lambda row: str(row["path"]).encode("utf-8"),
    )
    inventory = {
        "artifacts": inventory_rows,
        "assignment_algorithm": {
            "component_edges": [
                "normalized-query-text-equality",
                "registered-near-duplicate-token-rule",
                "shared-positive-document-content",
                "shared-positive-relevance-document",
            ],
            "cross_source_split_policy": "exclude-entire-component-v1",
            "fit_calibration_component_ratio": "4:1",
            "name": operator.ASSIGNMENT_ALGORITHM,
            "three_way_component_ratio": "3:1:1",
        },
        "assignment_seed_sha256": _digest("assignment seed"),
        "bright_document_identity": "fixture",
        "bright_domains": [],
        "config_sha256": _digest("staging config"),
        "counts": inventory_counts,
        "hotpotqa_fullwiki_scope": "fixture",
        "schema_version": operator.INVENTORY_SCHEMA,
        "sources": [
            {
                "byte_count": 1,
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "sha256": _digest("fixture source"),
                "source_id": "fixture",
            }
        ],
        "withhold_sealed_labels_from_online_process": True,
    }
    inventory_bytes = _canonical(inventory)
    inventory_sha256 = _digest(inventory_bytes)
    _write(projection / "inventory.json", inventory_bytes)
    _write(
        projection / "inventory.sha256",
        f"{inventory_sha256}  inventory.json\n".encode("ascii"),
    )

    projected_set_sha256 = _digest(_canonical(projected_rows)[:-1])
    projection_receipt = {
        "projected_artifact_count": len(projected_rows),
        "projected_artifact_set_sha256": projected_set_sha256,
        "projected_artifacts": projected_rows,
        "projection_policy": operator.PROJECTION_POLICY,
        "schema_version": operator.PROJECTION_SCHEMA,
        "source_artifact_count": len(inventory_rows),
        "source_inventory_sha256": inventory_sha256,
    }
    projection_receipt_bytes = _canonical(projection_receipt)
    _write(
        projection / operator.PROJECTION_RECEIPT_FILENAME,
        projection_receipt_bytes,
    )

    audit_sources = tuple(inventory_rows)
    audit_source_set_sha256 = _digest(_canonical(audit_sources)[:-1])
    assignment = next(row for row in inventory_rows if row["role"] == "assignments")
    query_rows = [row for row in inventory_rows if row["role"] == "queries"]
    qrel_rows = [row for row in inventory_rows if row["role"] == "qrels"]
    query_counts = [
        {
            "dataset": row["dataset"],
            "query_count": row["record_count"],
            "stage": row["stage"],
        }
        for row in query_rows
    ]
    query_counts.sort(key=lambda row: (str(row["dataset"]), str(row["stage"])))
    total_queries = sum(int(row["record_count"]) for row in query_rows)
    total_qrels = sum(int(row["record_count"]) for row in qrel_rows)
    audit = {
        "algorithm_sha256": operator.PARTITION_AUDIT_ALGORITHM_SHA256,
        "assignment_artifact_sha256": assignment["sha256"],
        "assignment_component_count": total_queries,
        "assignment_count": total_queries,
        "assignment_seed_sha256": inventory["assignment_seed_sha256"],
        "audit_component_count": total_queries,
        "component_membership_sha256": _digest("component membership"),
        "corpus_artifact_count": len(operator.FIXED_CORPORA),
        "cross_stage_component_count": 0,
        "exact_text_edge_count": 0,
        "near_duplicate_config_sha256": operator.NEAR_DUPLICATE_CONFIG_SHA256,
        "near_duplicate_edge_count": 0,
        "normalized_text_membership_sha256": _digest("normalized membership"),
        "positive_document_content_membership_sha256": _digest("positive document content"),
        "positive_document_membership_sha256": _digest("positive document"),
        "positive_qrel_count": total_qrels,
        "qrel_artifact_count": len(qrel_rows),
        "qrel_count": total_qrels,
        "query_artifact_count": len(query_rows),
        "query_count": total_queries,
        "query_counts": query_counts,
        "query_coverage_sha256": _digest("query coverage"),
        "schema_version": operator.PARTITION_AUDIT_SCHEMA,
        "shared_positive_document_content_edge_count": 0,
        "shared_positive_document_edge_count": 0,
        "source_artifact_set_sha256": audit_source_set_sha256,
        "source_artifacts": list(audit_sources),
        "staged_inventory_sha256": inventory_sha256,
        "staging_config_sha256": inventory["config_sha256"],
        "structural_exclusion_artifact_sha256": next(
            row["sha256"]
            for row in inventory_rows
            if row["role"] == "query-partition-structural-exclusions"
        ),
        "structural_exclusion_component_count": 0,
        "structural_exclusion_counts": [],
        "structural_exclusion_membership_sha256": _digest("structural exclusion membership"),
        "structural_exclusion_query_count": 0,
    }
    typed_audit = ScalableQueryPartitionAuditReceipt.from_dict(audit)
    audit_bytes = typed_audit.canonical_file_bytes()
    assert audit_bytes == _canonical(audit)
    _write(audit_path, audit_bytes)
    _freeze_projection(projection)
    os.chmod(audit_path, 0o400)
    return _Fixture(
        projection_root=projection,
        partition_audit=audit_path,
        output_parent=output_parent,
        inventory_sha256=inventory_sha256,
        projection_receipt_sha256=_digest(projection_receipt_bytes),
        partition_audit_sha256=_digest(audit_bytes),
        projected_rows=projected_rows,
        inventory_rows=inventory_rows,
    )


def _build(fixture: _Fixture) -> operator.DevelopmentStagingViewReceipt:
    return operator.build_development_staging_view(
        projection_root=fixture.projection_root,
        staged_inventory_sha256=fixture.inventory_sha256,
        projection_receipt_sha256=fixture.projection_receipt_sha256,
        partition_audit_path=fixture.partition_audit,
        partition_audit_file_sha256=fixture.partition_audit_sha256,
        output_root=fixture.output_root,
    )


def _hidden_work_paths(fixture: _Fixture) -> list[Path]:
    return list(fixture.output_parent.glob(f".{fixture.output_root.name}.development-view-*"))


def test_builds_exact_fit_calibration_view_and_canonical_receipt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    receipt = _build(fixture)
    loaded = operator.verify_development_staging_view(
        fixture.output_root,
        expected_receipt_sha256=receipt.artifact_sha256,
    )

    assert loaded == receipt
    assert receipt.staged_inventory_sha256 == fixture.inventory_sha256
    assert receipt.projection_receipt_sha256 == fixture.projection_receipt_sha256
    assert receipt.partition_audit_file_sha256 == fixture.partition_audit_sha256
    assert receipt.schema_version == operator.VIEW_RECEIPT_SCHEMA
    assert receipt.input_custody.contract == operator.INPUT_CUSTODY_CONTRACT
    assert receipt.input_custody.noncooperating_same_uid_mutation_excluded is True
    assert receipt.input_custody.producer_parent_and_file_leases_held_through_publication is True
    assert len(receipt.artifacts) == 13
    expected_payloads = set(operator._expected_payload_contract())
    assert {row.path for row in receipt.artifacts} == {
        "inventory.json",
        "inventory.sha256",
        *expected_payloads,
    }
    assert not any("sealed" in row.path for row in receipt.artifacts)
    assert not any(row.role in {"qrels", "evidence-bundles"} for row in receipt.artifacts)
    assert not (fixture.output_root / "datasets/scifact/sealed").exists()
    assert stat_mode(fixture.output_root) == 0o500
    assert all(stat_mode(fixture.output_root / row.path) == 0o400 for row in receipt.artifacts)
    receipt_bytes = (fixture.output_root / operator.VIEW_RECEIPT_FILENAME).read_bytes()
    assert receipt_bytes == receipt.canonical_file_bytes()
    assert not _hidden_work_paths(fixture)


def test_build_normalizes_exact_private_modes_under_restrictive_umask(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    previous_umask = os.umask(0o777)
    try:
        receipt = _build(fixture)
    finally:
        os.umask(previous_umask)

    for directory, _directory_names, file_names in os.walk(fixture.output_root):
        directory_path = Path(directory)
        assert stat_mode(directory_path) == 0o500
        assert all(stat_mode(directory_path / name) == 0o400 for name in file_names)
    operator.verify_development_staging_view(
        fixture.output_root,
        expected_receipt_sha256=receipt.artifact_sha256,
    )


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


@pytest.mark.parametrize(
    "field",
    (
        "staged_inventory_sha256",
        "projection_receipt_sha256",
        "partition_audit_file_sha256",
    ),
)
def test_rejects_each_external_digest_substitution_and_leaves_no_output(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _fixture(tmp_path)
    arguments: dict[str, Any] = {
        "projection_root": fixture.projection_root,
        "staged_inventory_sha256": fixture.inventory_sha256,
        "projection_receipt_sha256": fixture.projection_receipt_sha256,
        "partition_audit_path": fixture.partition_audit,
        "partition_audit_file_sha256": fixture.partition_audit_sha256,
        "output_root": fixture.output_root,
    }
    arguments[field] = _digest(f"substituted {field}")

    with pytest.raises(operator.DevelopmentStagingViewError, match="caller pin"):
        operator.build_development_staging_view(**arguments)

    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)


def test_rejects_injected_qrel_payload_before_copy_and_fails_clean(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _thaw_projection(fixture.projection_root)
    injected = fixture.projection_root / "datasets/scifact/fit/qrels.jsonl"
    _write(
        injected,
        b'{"document_id":"document-1","query_id":"scifact-fit-1","relevance":1}\n',
    )
    _freeze_projection(fixture.projection_root)

    with pytest.raises(operator.DevelopmentStagingViewError, match="unexpected file"):
        _build(fixture)

    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)


def test_rejects_linked_query_source_without_opening_it(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _thaw_projection(fixture.projection_root)
    query = fixture.projection_root / "datasets/scifact/fit/queries.jsonl"
    replacement = fixture.projection_root / "assignments.jsonl"
    query.unlink()
    query.symlink_to(replacement)
    _freeze_projection(fixture.projection_root)

    with pytest.raises(
        operator.DevelopmentStagingViewError,
        match="regular file|linked or special",
    ):
        _build(fixture)

    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)
    _thaw_projection(fixture.projection_root)
    query.unlink()


def test_regular_opens_are_nonblocking_and_source_fifo_fails_promptly(
    tmp_path: Path,
) -> None:
    if hasattr(os, "O_NONBLOCK"):
        assert operator._regular_open_flags() & os.O_NONBLOCK

    fixture = _fixture(tmp_path)
    _thaw_projection(fixture.projection_root)
    query = fixture.projection_root / "datasets/scifact/fit/queries.jsonl"
    query.unlink()
    os.mkfifo(query, mode=0o400)
    _freeze_projection(fixture.projection_root)

    started = time.monotonic()
    with pytest.raises(
        operator.DevelopmentStagingViewError,
        match="regular file|linked or special",
    ):
        _build(fixture)
    assert time.monotonic() - started < 2.0
    assert list(fixture.output_parent.iterdir()) == []


def test_audit_fifo_fails_promptly_and_leaves_exactly_no_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.partition_audit.unlink()
    os.mkfifo(fixture.partition_audit, mode=0o400)

    started = time.monotonic()
    with pytest.raises(operator.DevelopmentStagingViewError, match="regular file"):
        _build(fixture)
    assert time.monotonic() - started < 2.0
    assert list(fixture.output_parent.iterdir()) == []


def test_rejects_writable_input_and_nonprivate_output_parent(tmp_path: Path) -> None:
    writable = _fixture(tmp_path / "writable")
    os.chmod(writable.projection_root, 0o700)
    with pytest.raises(operator.DevelopmentStagingViewError, match="read-only"):
        _build(writable)
    assert not writable.output_root.exists()

    exposed = _fixture(tmp_path / "exposed")
    os.chmod(exposed.output_parent, 0o755)
    with pytest.raises(operator.DevelopmentStagingViewError, match="no permissions"):
        _build(exposed)
    assert not exposed.output_root.exists()


def test_rejects_partition_audit_crossing_even_with_new_caller_digest(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    os.chmod(fixture.partition_audit, 0o600)
    audit = json.loads(fixture.partition_audit.read_text(encoding="utf-8"))
    audit["cross_stage_component_count"] = 1
    encoded = _canonical(audit)
    _write(fixture.partition_audit, encoded)
    os.chmod(fixture.partition_audit, 0o400)
    fixture.partition_audit_sha256 = _digest(encoded)

    with pytest.raises(operator.DevelopmentStagingViewError, match="zero stage crossings"):
        _build(fixture)

    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)


def test_rejects_control_disagreement_for_registered_query(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    os.chmod(fixture.partition_audit, 0o600)
    audit = json.loads(fixture.partition_audit.read_text(encoding="utf-8"))
    target = next(
        row
        for row in audit["source_artifacts"]
        if row["path"] == "datasets/scifact/fit/queries.jsonl"
    )
    target["sha256"] = _digest("substituted query")
    ordered = sorted(
        audit["source_artifacts"],
        key=lambda row: row["path"].encode("utf-8"),
    )
    audit["source_artifacts"] = ordered
    audit["source_artifact_set_sha256"] = _digest(_canonical(ordered)[:-1])
    encoded = _canonical(audit)
    _write(fixture.partition_audit, encoded)
    os.chmod(fixture.partition_audit, 0o400)
    fixture.partition_audit_sha256 = _digest(encoded)

    with pytest.raises(operator.DevelopmentStagingViewError, match="source set differs"):
        _build(fixture)

    assert not fixture.output_root.exists()


def test_rejects_overwrite_and_preserves_first_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt = _build(fixture)
    receipt_path = fixture.output_root / operator.VIEW_RECEIPT_FILENAME
    before = receipt_path.read_bytes()

    with pytest.raises(operator.DevelopmentStagingViewError, match="already exists"):
        _build(fixture)

    assert receipt_path.read_bytes() == before == receipt.canonical_file_bytes()


def test_verify_rejects_forbidden_extra_payload_and_private_mode_drift(
    tmp_path: Path,
) -> None:
    first = _fixture(tmp_path / "first")
    first_receipt = _build(first)
    injected = first.output_root / "datasets/scifact/sealed/qrels.jsonl"
    _thaw_view(first.output_root)
    _write(
        injected,
        b'{"document_id":"document-1","query_id":"sealed-1","relevance":1}\n',
    )
    _seal_view(first.output_root)
    with pytest.raises(operator.DevelopmentStagingViewError, match="forbidden"):
        operator.verify_development_staging_view(
            first.output_root,
            expected_receipt_sha256=first_receipt.artifact_sha256,
        )

    second = _fixture(tmp_path / "second")
    second_receipt = _build(second)
    target = second.output_root / "assignments.jsonl"
    os.chmod(target, 0o644)
    with pytest.raises(
        operator.DevelopmentStagingViewError,
        match="read-only|no permissions",
    ):
        operator.verify_development_staging_view(
            second.output_root,
            expected_receipt_sha256=second_receipt.artifact_sha256,
        )


def test_verify_rejects_noncanonical_duplicate_receipt_keys(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _build(fixture)
    receipt_path = fixture.output_root / operator.VIEW_RECEIPT_FILENAME
    original = receipt_path.read_bytes()
    duplicate = original[:-2] + b',"schema_version":"duplicate"}\n'
    _thaw_view(fixture.output_root)
    _write(receipt_path, duplicate)
    _seal_view(fixture.output_root)

    with pytest.raises(operator.DevelopmentStagingViewError, match="repeats key"):
        operator.verify_development_staging_view(
            fixture.output_root,
            expected_receipt_sha256=_digest(duplicate),
        )


def test_closed_cli_build_verify_and_unknown_argument(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    assert (
        operator.main(
            [
                "build",
                "--projection-root",
                str(fixture.projection_root),
                "--staged-inventory-sha256",
                fixture.inventory_sha256,
                "--projection-receipt-sha256",
                fixture.projection_receipt_sha256,
                "--partition-audit",
                str(fixture.partition_audit),
                "--partition-audit-file-sha256",
                fixture.partition_audit_sha256,
                "--output-root",
                str(fixture.output_root),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == operator.CLI_RESULT_SCHEMA
    assert result["artifact_count"] == 13
    assert (
        operator.main(
            [
                "verify",
                "--root",
                str(fixture.output_root),
                "--receipt-sha256",
                result["receipt_sha256"],
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["receipt_sha256"] == result["receipt_sha256"]
    with pytest.raises(SystemExit) as failure:
        operator.main(
            [
                "verify",
                "--root",
                str(fixture.output_root),
                "--receipt-sha256",
                result["receipt_sha256"],
                "--unregistered-input",
                "value",
            ]
        )
    assert failure.value.code == 2


def test_cli_disables_abbreviations_on_root_and_each_subparser(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    parser = operator._parser()
    assert parser.allow_abbrev is False
    choices = next(
        action.choices
        for action in parser._actions
        if isinstance(getattr(action, "choices", None), dict)
        and {"build", "verify"}.issubset(action.choices)
    )
    assert choices["build"].allow_abbrev is False
    assert choices["verify"].allow_abbrev is False

    build_arguments = [
        "build",
        "--projection-roo",
        str(fixture.projection_root),
        "--staged-inventory-sha256",
        fixture.inventory_sha256,
        "--projection-receipt-sha256",
        fixture.projection_receipt_sha256,
        "--partition-audit",
        str(fixture.partition_audit),
        "--partition-audit-file-sha256",
        fixture.partition_audit_sha256,
        "--output-root",
        str(fixture.output_root),
    ]
    with pytest.raises(SystemExit) as build_failure:
        parser.parse_args(build_arguments)
    assert build_failure.value.code == 2

    with pytest.raises(SystemExit) as verify_failure:
        parser.parse_args(
            [
                "verify",
                "--root",
                str(fixture.output_root),
                "--receipt-sha",
                _digest("receipt"),
            ]
        )
    assert verify_failure.value.code == 2


def test_rejects_empty_unexpected_and_unreadable_expected_directories(
    tmp_path: Path,
) -> None:
    unexpected_fixture = _fixture(tmp_path / "unexpected")
    _thaw_projection(unexpected_fixture.projection_root)
    unexpected = unexpected_fixture.projection_root / "empty-unregistered-directory"
    unexpected.mkdir(mode=0o700)
    _freeze_projection(unexpected_fixture.projection_root)
    os.chmod(unexpected, 0)
    try:
        with pytest.raises(
            operator.DevelopmentStagingViewError,
            match="unexpected directory",
        ):
            _build(unexpected_fixture)
    finally:
        os.chmod(unexpected, 0o500)
    assert not unexpected_fixture.output_root.exists()
    assert not _hidden_work_paths(unexpected_fixture)

    unreadable_fixture = _fixture(tmp_path / "unreadable")
    unreadable = unreadable_fixture.projection_root / "datasets" / "scifact" / "fit"
    os.chmod(unreadable, 0)
    try:
        with pytest.raises(
            operator.DevelopmentStagingViewError,
            match="cannot traverse",
        ):
            _build(unreadable_fixture)
    finally:
        os.chmod(unreadable, 0o500)
    assert not unreadable_fixture.output_root.exists()
    assert not _hidden_work_paths(unreadable_fixture)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("assignment_seed_sha256", "seed or staging config"),
        ("staging_config_sha256", "seed or staging config"),
    ),
)
def test_binds_partition_seed_and_config_to_inventory(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    os.chmod(fixture.partition_audit, 0o600)
    audit = json.loads(fixture.partition_audit.read_text(encoding="utf-8"))
    audit[field] = _digest(f"substituted {field}")
    encoded = _canonical(audit)
    _write(fixture.partition_audit, encoded)
    os.chmod(fixture.partition_audit, 0o400)
    fixture.partition_audit_sha256 = _digest(encoded)

    with pytest.raises(operator.DevelopmentStagingViewError, match=message):
        _build(fixture)

    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)


def test_binds_inventory_counts_to_partition_strata(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _thaw_projection(fixture.projection_root)
    inventory_path = fixture.projection_root / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["counts"]["scifact"]["fit_queries"] = 2
    inventory_bytes = _canonical(inventory)
    inventory_sha256 = _digest(inventory_bytes)
    _write(inventory_path, inventory_bytes)
    _write(
        fixture.projection_root / "inventory.sha256",
        f"{inventory_sha256}  inventory.json\n".encode("ascii"),
    )
    projection_path = fixture.projection_root / operator.PROJECTION_RECEIPT_FILENAME
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection["source_inventory_sha256"] = inventory_sha256
    projection_bytes = _canonical(projection)
    _write(projection_path, projection_bytes)
    _freeze_projection(fixture.projection_root)

    os.chmod(fixture.partition_audit, 0o600)
    audit = json.loads(fixture.partition_audit.read_text(encoding="utf-8"))
    audit["staged_inventory_sha256"] = inventory_sha256
    audit_bytes = _canonical(audit)
    _write(fixture.partition_audit, audit_bytes)
    os.chmod(fixture.partition_audit, 0o400)

    fixture.inventory_sha256 = inventory_sha256
    fixture.projection_receipt_sha256 = _digest(projection_bytes)
    fixture.partition_audit_sha256 = _digest(audit_bytes)
    with pytest.raises(
        operator.DevelopmentStagingViewError,
        match="inventory and audit query counts differ",
    ):
        _build(fixture)

    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)


def test_production_shaped_view_feeds_development_cohort_select(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        fit_count=FIT_FAMILY_COUNT,
        calibration_count=CALIBRATION_FAMILY_COUNT,
    )
    view = _build(fixture)
    selection_path = (tmp_path / "development-cohort-selection.json").resolve()

    selection = select_development_cohort(
        fixture.output_root,
        selection_path,
        staged_inventory_sha256=fixture.inventory_sha256,
        partition_audit_path=fixture.partition_audit,
        partition_audit_sha256=fixture.partition_audit_sha256,
    )

    assert len(selection.selections) == 10
    assert selection.staged_inventory_sha256 == view.staged_inventory_sha256
    assert {
        (row.development_stage, row.requested_family_count) for row in selection.selections
    } == {
        ("development-fit", FIT_FAMILY_COUNT),
        ("development-calibration", CALIBRATION_FAMILY_COUNT),
    }


def _race_one_read(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: Path,
    writable_descriptor: int,
) -> None:
    original_read = os.read
    target_metadata = os.fstat(writable_descriptor)
    changed = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal changed
        chunk = original_read(descriptor, count)
        if chunk and not changed and operator._same_inode(os.fstat(descriptor), target_metadata):
            changed = True
            os.pwrite(writable_descriptor, b"X", 0)
            os.fsync(writable_descriptor)
        return chunk

    monkeypatch.setattr(operator.os, "read", racing_read)
    assert target.exists()


def test_rejects_source_and_audit_hardlinks(tmp_path: Path) -> None:
    source_fixture = _fixture(tmp_path / "source")
    source = source_fixture.projection_root / "datasets" / "scifact" / "fit" / "queries.jsonl"
    os.link(source, tmp_path / "source-query-hardlink")
    with pytest.raises(operator.DevelopmentStagingViewError, match="hard link"):
        _build(source_fixture)
    assert not source_fixture.output_root.exists()

    audit_fixture = _fixture(tmp_path / "audit")
    os.link(audit_fixture.partition_audit, tmp_path / "partition-audit-hardlink")
    with pytest.raises(operator.DevelopmentStagingViewError, match="hard link"):
        _build(audit_fixture)
    assert not audit_fixture.output_root.exists()


def test_detects_source_and_audit_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_fixture = _fixture(tmp_path / "source")
    source = source_fixture.projection_root / "datasets" / "scifact" / "fit" / "queries.jsonl"
    os.chmod(source, 0o600)
    source_writer = os.open(source, os.O_RDWR)
    os.chmod(source, 0o400)
    try:
        _race_one_read(
            monkeypatch,
            target=source,
            writable_descriptor=source_writer,
        )
        with pytest.raises(operator.DevelopmentStagingViewError, match="changed"):
            _build(source_fixture)
    finally:
        os.close(source_writer)
        monkeypatch.undo()
    assert not source_fixture.output_root.exists()
    assert not _hidden_work_paths(source_fixture)

    audit_fixture = _fixture(tmp_path / "audit")
    os.chmod(audit_fixture.partition_audit, 0o600)
    audit_writer = os.open(audit_fixture.partition_audit, os.O_RDWR)
    os.chmod(audit_fixture.partition_audit, 0o400)
    try:
        _race_one_read(
            monkeypatch,
            target=audit_fixture.partition_audit,
            writable_descriptor=audit_writer,
        )
        with pytest.raises(operator.DevelopmentStagingViewError, match="changed"):
            _build(audit_fixture)
    finally:
        os.close(audit_writer)
        monkeypatch.undo()
    assert not audit_fixture.output_root.exists()


def test_verifier_requires_external_receipt_digest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _build(fixture)

    with pytest.raises(TypeError, match="expected_receipt_sha256"):
        operator.verify_development_staging_view(fixture.output_root)  # type: ignore[call-arg]


def test_published_receipt_fifo_fails_promptly(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt = _build(fixture)
    receipt_path = fixture.output_root / operator.VIEW_RECEIPT_FILENAME
    _thaw_view(fixture.output_root)
    receipt_path.unlink()
    os.mkfifo(receipt_path, mode=0o400)
    _seal_view(fixture.output_root)

    started = time.monotonic()
    with pytest.raises(operator.DevelopmentStagingViewError, match="regular file"):
        operator.verify_development_staging_view(
            fixture.output_root,
            expected_receipt_sha256=receipt.artifact_sha256,
        )
    assert time.monotonic() - started < 2.0


def test_verifier_rejects_receipt_and_artifact_hardlinks(tmp_path: Path) -> None:
    receipt_fixture = _fixture(tmp_path / "receipt")
    receipt = _build(receipt_fixture)
    receipt_path = receipt_fixture.output_root / operator.VIEW_RECEIPT_FILENAME
    os.link(receipt_path, tmp_path / "receipt-hardlink")
    with pytest.raises(operator.DevelopmentStagingViewError, match="hard link"):
        operator.verify_development_staging_view(
            receipt_fixture.output_root,
            expected_receipt_sha256=receipt.artifact_sha256,
        )

    artifact_fixture = _fixture(tmp_path / "artifact")
    artifact_receipt = _build(artifact_fixture)
    os.link(
        artifact_fixture.output_root / "assignments.jsonl",
        tmp_path / "artifact-hardlink",
    )
    with pytest.raises(operator.DevelopmentStagingViewError, match="hard link"):
        operator.verify_development_staging_view(
            artifact_fixture.output_root,
            expected_receipt_sha256=artifact_receipt.artifact_sha256,
        )


def test_verifier_detects_receipt_and_artifact_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_fixture = _fixture(tmp_path / "receipt")
    receipt = _build(receipt_fixture)
    receipt_path = receipt_fixture.output_root / operator.VIEW_RECEIPT_FILENAME
    os.chmod(receipt_path, 0o600)
    receipt_writer = os.open(receipt_path, os.O_RDWR)
    os.chmod(receipt_path, 0o400)
    try:
        _race_one_read(
            monkeypatch,
            target=receipt_path,
            writable_descriptor=receipt_writer,
        )
        with pytest.raises(operator.DevelopmentStagingViewError, match="changed"):
            operator.verify_development_staging_view(
                receipt_fixture.output_root,
                expected_receipt_sha256=receipt.artifact_sha256,
            )
    finally:
        os.close(receipt_writer)
        monkeypatch.undo()

    artifact_fixture = _fixture(tmp_path / "artifact")
    artifact_receipt = _build(artifact_fixture)
    artifact_path = artifact_fixture.output_root / "assignments.jsonl"
    os.chmod(artifact_path, 0o600)
    artifact_writer = os.open(artifact_path, os.O_RDWR)
    os.chmod(artifact_path, 0o400)
    try:
        _race_one_read(
            monkeypatch,
            target=artifact_path,
            writable_descriptor=artifact_writer,
        )
        with pytest.raises(operator.DevelopmentStagingViewError, match="changed"):
            operator.verify_development_staging_view(
                artifact_fixture.output_root,
                expected_receipt_sha256=artifact_receipt.artifact_sha256,
            )
    finally:
        os.close(artifact_writer)
        monkeypatch.undo()


def test_prepublication_failure_removes_temporary_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    def fail_verification(
        _descriptor: int,
        _receipt: operator.DevelopmentStagingViewReceipt,
    ) -> None:
        raise operator.DevelopmentStagingViewError("injected temporary verification failure")

    monkeypatch.setattr(operator, "_verify_temporary_tree", fail_verification)
    with pytest.raises(
        operator.DevelopmentStagingViewError,
        match="injected temporary verification failure",
    ):
        _build(fixture)

    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)
    assert list(fixture.output_parent.iterdir()) == []


def test_prerename_parent_mode_drift_is_fail_clean_not_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_fsync_tree = operator._fsync_temporary_directories
    drifted = False

    def drift_parent_mode(
        descriptor: int,
        receipt: operator.DevelopmentStagingViewReceipt,
    ) -> None:
        nonlocal drifted
        original_fsync_tree(descriptor, receipt)
        os.chmod(fixture.output_parent, 0o755)
        drifted = True

    monkeypatch.setattr(
        operator,
        "_fsync_temporary_directories",
        drift_parent_mode,
    )
    try:
        with pytest.raises(
            operator.DevelopmentStagingViewError,
            match="parent identity or mode changed",
        ) as failure:
            _build(fixture)
    finally:
        os.chmod(fixture.output_parent, 0o700)

    assert drifted
    assert not isinstance(
        failure.value,
        operator.DevelopmentStagingPublicationIndeterminate,
    )
    assert list(fixture.output_parent.iterdir()) == []


def test_postrename_fsync_failure_rolls_back_and_proves_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_fsync = operator._fsync_directory
    failed = False

    def fail_published_parent(descriptor: int, *, label: str) -> None:
        nonlocal failed
        if (
            not failed
            and label == "development staging output parent"
            and operator._entry_stat(descriptor, fixture.output_root.name) is not None
        ):
            failed = True
            raise operator.DevelopmentStagingViewError("injected parent fsync failure")
        original_fsync(descriptor, label=label)

    monkeypatch.setattr(operator, "_fsync_directory", fail_published_parent)
    with pytest.raises(
        operator.DevelopmentStagingViewError,
        match="rollback was completed and proved",
    ):
        _build(fixture)

    assert failed
    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)
    assert list(fixture.output_parent.iterdir()) == []


def test_postfsync_mutation_is_detected_then_rolled_back_and_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_fsync = operator._fsync_directory
    mutated = False

    def mutate_after_published_parent_fsync(descriptor: int, *, label: str) -> None:
        nonlocal mutated
        original_fsync(descriptor, label=label)
        if (
            not mutated
            and label == "development staging output parent"
            and operator._entry_stat(descriptor, fixture.output_root.name) is not None
        ):
            target = fixture.output_root / "assignments.jsonl"
            os.chmod(target, 0o600)
            writable = os.open(target, os.O_WRONLY)
            try:
                os.pwrite(writable, b"X", 0)
                os.fsync(writable)
            finally:
                os.close(writable)
                os.chmod(target, 0o400)
            mutated = True

    monkeypatch.setattr(
        operator,
        "_fsync_directory",
        mutate_after_published_parent_fsync,
    )
    with pytest.raises(
        operator.DevelopmentStagingViewError,
        match="rollback was completed and proved",
    ):
        _build(fixture)

    assert mutated
    assert list(fixture.output_parent.iterdir()) == []


def test_postrename_mutation_after_prior_fingerprint_rolls_back_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_read = operator._read_relative_regular_with_metadata
    mutated = False

    def mutate_after_receipt_snapshot(
        root_descriptor: int,
        relative: str,
        *,
        maximum: int,
        label: str,
        private: bool,
        read_only: bool,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal mutated
        result = original_read(
            root_descriptor,
            relative,
            maximum=maximum,
            label=label,
            private=private,
            read_only=read_only,
        )
        if (
            not mutated
            and relative == operator.VIEW_RECEIPT_FILENAME
            and label == "temporary development staging view receipt"
            and fixture.output_root.exists()
        ):
            target = fixture.output_root / "assignments.jsonl"
            os.chmod(target, 0o600)
            descriptor = os.open(target, os.O_WRONLY)
            try:
                os.pwrite(descriptor, b"X", 0)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
                os.chmod(target, 0o400)
            mutated = True
        return result

    monkeypatch.setattr(
        operator,
        "_read_relative_regular_with_metadata",
        mutate_after_receipt_snapshot,
    )
    with pytest.raises(
        operator.DevelopmentStagingViewError,
        match="rollback was completed and proved",
    ) as failure:
        _build(fixture)

    assert mutated
    assert not isinstance(
        failure.value,
        operator.DevelopmentStagingPublicationIndeterminate,
    )
    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)
    assert list(fixture.output_parent.iterdir()) == []


def test_rename_then_raise_is_classified_and_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_rename = operator._rename_exclusive_at
    injected = False

    def rename_then_raise(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        nonlocal injected
        if destination_name == fixture.output_root.name and not injected:
            original_rename(parent_descriptor, source_name, destination_name)
            injected = True
            raise operator.DevelopmentStagingViewError("injected exception after native rename")
        original_rename(parent_descriptor, source_name, destination_name)

    monkeypatch.setattr(operator, "_rename_exclusive_at", rename_then_raise)
    with pytest.raises(
        operator.DevelopmentStagingViewError,
        match="rollback was completed and proved",
    ) as failure:
        _build(fixture)

    assert injected
    assert not isinstance(
        failure.value,
        operator.DevelopmentStagingPublicationIndeterminate,
    )
    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)
    assert list(fixture.output_parent.iterdir()) == []


def test_mutation_after_closing_file_stat_is_caught_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_seal = operator._seal_temporary_tree
    original_scan = operator._scan_exact_tree
    writer: int | None = None
    mutated = False

    def capture_writer(
        descriptor: int,
        receipt: operator.DevelopmentStagingViewReceipt,
    ) -> None:
        nonlocal writer
        writer = os.open(
            "assignments.jsonl",
            os.O_RDWR,
            dir_fd=descriptor,
        )
        original_seal(descriptor, receipt)

    def mutate_after_closing_scan(
        descriptor: int,
        **arguments: Any,
    ) -> None:
        nonlocal mutated
        original_scan(descriptor, **arguments)
        if (
            not mutated
            and fixture.output_root.exists()
            and arguments.get("expected_file_metadata") is not None
        ):
            assert writer is not None
            os.pwrite(writer, b"X", 0)
            os.fsync(writer)
            mutated = True

    monkeypatch.setattr(operator, "_seal_temporary_tree", capture_writer)
    monkeypatch.setattr(operator, "_scan_exact_tree", mutate_after_closing_scan)
    try:
        with pytest.raises(
            operator.DevelopmentStagingViewError,
            match="rollback was completed and proved",
        ):
            _build(fixture)
    finally:
        if writer is not None:
            os.close(writer)

    assert mutated
    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)
    assert list(fixture.output_parent.iterdir()) == []


def test_parent_replacement_after_published_verifier_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    moved_parent = (tmp_path / "moved-published-parent").resolve()
    original_verify = operator._verify_published_tree_by_name
    replaced = False

    def replace_parent_after_verification(**arguments: Any) -> None:
        nonlocal replaced
        original_verify(**arguments)
        if not replaced:
            os.rename(fixture.output_parent, moved_parent)
            fixture.output_parent.mkdir(mode=0o700)
            replaced = True

    monkeypatch.setattr(
        operator,
        "_verify_published_tree_by_name",
        replace_parent_after_verification,
    )
    with pytest.raises(
        operator.DevelopmentStagingPublicationIndeterminate,
        match="indeterminate",
    ):
        _build(fixture)

    assert replaced
    assert not fixture.output_root.exists()
    moved_temporary = list(moved_parent.glob(f".{fixture.output_root.name}.development-view-*"))
    assert len(moved_temporary) == 1
    assert not (moved_parent / fixture.output_root.name).exists()

    temporary_basename = moved_temporary[0].name
    fixture.output_parent.rmdir()
    os.rename(moved_parent, fixture.output_parent)
    _thaw_view(fixture.output_parent / temporary_basename)


@pytest.mark.parametrize("mutation", ("bytes", "name"))
def test_postpublication_source_change_after_scan_is_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    target = fixture.projection_root / "datasets/scifact/fit/queries.jsonl"
    original_source_binding = operator._require_source_binding
    writer: int | None = None
    replacement: Path | None = None
    if mutation == "bytes":
        os.chmod(target, 0o600)
        writer = os.open(target, os.O_RDWR)
        os.chmod(target, 0o400)
    else:
        replacement = tmp_path / "postpublication-replacement-query.jsonl"
        replacement.write_bytes(b"X" + target.read_bytes()[1:])
        os.chmod(replacement, 0o400)
    mutated = False

    def mutate_after_source_scan(**arguments: Any) -> None:
        nonlocal mutated
        original_source_binding(**arguments)
        if mutated:
            return
        if mutation == "bytes":
            assert writer is not None
            os.pwrite(writer, b"X", 0)
            os.fsync(writer)
        else:
            assert replacement is not None
            os.chmod(target.parent, 0o700)
            target.unlink()
            replacement.rename(target)
            os.chmod(target, 0o400)
            os.chmod(target.parent, 0o500)
        mutated = True

    monkeypatch.setattr(
        operator,
        "_require_source_binding",
        mutate_after_source_scan,
    )
    try:
        with pytest.raises(
            operator.DevelopmentStagingViewError,
            match="rollback was completed and proved",
        ):
            _build(fixture)
    finally:
        if writer is not None:
            os.close(writer)

    assert mutated
    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)
    assert list(fixture.output_parent.iterdir()) == []


def test_rollback_rejects_output_restoration_after_stale_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_verify = operator._verify_published_tree_by_name
    original_rename = operator._rename_exclusive_at
    original_entry_stat = operator._entry_stat
    verification_failed = False
    rolled_back = False
    restored = False

    def fail_first_published_verification(**arguments: Any) -> None:
        nonlocal verification_failed
        if not verification_failed:
            verification_failed = True
            raise operator.DevelopmentStagingViewError("injected published verification failure")
        original_verify(**arguments)

    def record_rollback(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        nonlocal rolled_back
        original_rename(parent_descriptor, source_name, destination_name)
        if source_name == fixture.output_root.name:
            rolled_back = True

    def restore_after_stale_absence(
        parent_descriptor: int,
        name: str,
    ) -> os.stat_result | None:
        nonlocal restored
        observed = original_entry_stat(parent_descriptor, name)
        if rolled_back and not restored and name == fixture.output_root.name and observed is None:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            restored = True
            return None
        return observed

    monkeypatch.setattr(
        operator,
        "_verify_published_tree_by_name",
        fail_first_published_verification,
    )
    monkeypatch.setattr(operator, "_rename_exclusive_at", record_rollback)
    monkeypatch.setattr(operator, "_entry_stat", restore_after_stale_absence)
    with pytest.raises(
        operator.DevelopmentStagingPublicationIndeterminate,
        match="indeterminate",
    ):
        _build(fixture)

    assert verification_failed
    assert rolled_back
    assert restored
    assert fixture.output_root.is_dir()
    assert list(fixture.output_root.iterdir()) == []
    hidden = _hidden_work_paths(fixture)
    assert len(hidden) == 1
    _thaw_view(hidden[0])


def test_standalone_verifier_catches_mutation_during_path_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    receipt = _build(fixture)
    target = fixture.output_root / "assignments.jsonl"
    os.chmod(target, 0o600)
    writer = os.open(target, os.O_RDWR)
    os.chmod(target, 0o400)
    original_path_check = operator._directory_path_matches_descriptor
    mutated = False

    def mutate_during_path_check(
        path: Path,
        descriptor: int,
        *,
        private: bool,
    ) -> bool:
        nonlocal mutated
        matches = original_path_check(
            path,
            descriptor,
            private=private,
        )
        if not mutated and path == fixture.output_root:
            os.pwrite(writer, b"X", 0)
            os.fsync(writer)
            mutated = True
        return matches

    monkeypatch.setattr(
        operator,
        "_directory_path_matches_descriptor",
        mutate_during_path_check,
    )
    try:
        with pytest.raises(
            operator.DevelopmentStagingViewError,
            match="changed|differs",
        ):
            operator.verify_development_staging_view(
                fixture.output_root,
                expected_receipt_sha256=receipt.artifact_sha256,
            )
    finally:
        os.close(writer)

    assert mutated


@pytest.mark.parametrize("substitution", ("output", "temporary"))
def test_final_publication_name_substitution_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
) -> None:
    fixture = _fixture(tmp_path)
    original_rename = operator._rename_exclusive_at
    original_source_binding = operator._require_source_binding
    temporary_name: str | None = None
    source_proof_count = 0
    substituted = False
    displaced = fixture.output_parent / "displaced-selection-view"

    def capture_temporary_name(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        nonlocal temporary_name
        if destination_name == fixture.output_root.name:
            temporary_name = source_name
        original_rename(parent_descriptor, source_name, destination_name)

    def substitute_after_final_source_proof(**arguments: Any) -> None:
        nonlocal source_proof_count, substituted
        original_source_binding(**arguments)
        source_proof_count += 1
        if source_proof_count != 2:
            return
        if substitution == "output":
            os.chmod(fixture.output_root, 0o700)
            os.rename(fixture.output_root, displaced)
            os.chmod(displaced, 0o500)
            fixture.output_root.mkdir(mode=0o700)
        else:
            assert temporary_name is not None
            (fixture.output_parent / temporary_name).mkdir(mode=0o700)
        substituted = True

    monkeypatch.setattr(
        operator,
        "_rename_exclusive_at",
        capture_temporary_name,
    )
    monkeypatch.setattr(
        operator,
        "_require_source_binding",
        substitute_after_final_source_proof,
    )
    with pytest.raises(
        operator.DevelopmentStagingPublicationIndeterminate,
        match="publication",
    ):
        _build(fixture)

    assert substituted
    if substitution == "output":
        assert displaced.is_dir()
        assert (displaced / operator.VIEW_RECEIPT_FILENAME).is_file()
        assert fixture.output_root.is_dir()
        assert list(fixture.output_root.iterdir()) == []
        _thaw_view(displaced)
    else:
        assert temporary_name is not None
        assert (fixture.output_parent / temporary_name).is_dir()
        receipt_path = fixture.output_root / operator.VIEW_RECEIPT_FILENAME
        operator.verify_development_staging_view(
            fixture.output_root,
            expected_receipt_sha256=_digest(receipt_path.read_bytes()),
        )
        _thaw_view(fixture.output_root)


@pytest.mark.parametrize("mutation", ("bytes", "name"))
def test_final_source_recheck_rejects_mutation_after_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    target = fixture.projection_root / "datasets/scifact/fit/queries.jsonl"
    original_fsync = operator._fsync_temporary_directories
    writer: int | None = None
    replacement: Path | None = None
    if mutation == "bytes":
        os.chmod(target, 0o600)
        writer = os.open(target, os.O_RDWR)
        os.chmod(target, 0o400)
    else:
        replacement = tmp_path / "replacement-query.jsonl"
        replacement.write_bytes(b"X" + target.read_bytes()[1:])
        os.chmod(replacement, 0o400)
    mutated = False

    def mutate_after_copy(
        descriptor: int,
        receipt: operator.DevelopmentStagingViewReceipt,
    ) -> None:
        nonlocal mutated
        original_fsync(descriptor, receipt)
        if mutation == "bytes":
            assert writer is not None
            os.pwrite(writer, b"X", 0)
            os.fsync(writer)
        else:
            assert replacement is not None
            os.chmod(target.parent, 0o700)
            target.unlink()
            replacement.rename(target)
            os.chmod(target, 0o400)
            os.chmod(target.parent, 0o500)
        mutated = True

    monkeypatch.setattr(
        operator,
        "_fsync_temporary_directories",
        mutate_after_copy,
    )
    try:
        with pytest.raises(
            operator.DevelopmentStagingViewError,
            match="final source artifact .* differs",
        ):
            _build(fixture)
    finally:
        if writer is not None:
            os.close(writer)

    assert mutated
    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)
    assert list(fixture.output_parent.iterdir()) == []


def test_rollback_destination_race_is_indeterminate_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_fsync = operator._fsync_directory
    original_rename = operator._rename_exclusive_at
    failed = False
    temporary_name: str | None = None
    raced_inode: tuple[int, int] | None = None

    def fail_published_parent(descriptor: int, *, label: str) -> None:
        nonlocal failed
        if (
            not failed
            and label == "development staging output parent"
            and operator._entry_stat(descriptor, fixture.output_root.name) is not None
        ):
            failed = True
            raise operator.DevelopmentStagingViewError("injected parent fsync failure")
        original_fsync(descriptor, label=label)

    def race_rollback(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        nonlocal raced_inode, temporary_name
        if destination_name == fixture.output_root.name:
            temporary_name = source_name
        elif source_name == fixture.output_root.name:
            assert destination_name == temporary_name
            os.mkdir(destination_name, mode=0o700, dir_fd=parent_descriptor)
            raced = os.stat(
                destination_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            raced_inode = (raced.st_dev, raced.st_ino)
        original_rename(parent_descriptor, source_name, destination_name)

    monkeypatch.setattr(operator, "_fsync_directory", fail_published_parent)
    monkeypatch.setattr(operator, "_rename_exclusive_at", race_rollback)
    with pytest.raises(
        operator.DevelopmentStagingPublicationIndeterminate,
        match="indeterminate",
    ):
        _build(fixture)

    assert failed
    assert temporary_name is not None
    assert raced_inode is not None
    assert fixture.output_root.is_dir()
    raced_path = fixture.output_parent / temporary_name
    raced = raced_path.stat()
    assert (raced.st_dev, raced.st_ino) == raced_inode
    assert list(raced_path.iterdir()) == []
    receipt_path = fixture.output_root / operator.VIEW_RECEIPT_FILENAME
    operator.verify_development_staging_view(
        fixture.output_root,
        expected_receipt_sha256=_digest(receipt_path.read_bytes()),
    )


def test_parent_path_replacement_cannot_redirect_post_temp_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    moved_parent = (tmp_path / "moved-operator-output").resolve()
    original_fsync_tree = operator._fsync_temporary_directories
    replaced = False

    def replace_parent_path(
        descriptor: int,
        receipt: operator.DevelopmentStagingViewReceipt,
    ) -> None:
        nonlocal replaced
        original_fsync_tree(descriptor, receipt)
        os.rename(fixture.output_parent, moved_parent)
        fixture.output_parent.mkdir(mode=0o700)
        replaced = True

    monkeypatch.setattr(
        operator,
        "_fsync_temporary_directories",
        replace_parent_path,
    )
    with pytest.raises(
        operator.DevelopmentStagingViewError,
        match="parent path changed",
    ):
        _build(fixture)

    assert replaced
    assert not fixture.output_root.exists()
    assert not list(moved_parent.glob(f".{fixture.output_root.name}.development-view-*"))


def _cooperative_exclusive_lease_acquired(descriptor: int) -> bool:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        assert exc.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}
        return False
    return True


def test_final_name_proof_excludes_cooperative_temporary_name_recreation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_rename = operator._rename_exclusive_at
    original_name_proof = operator._require_published_name_state
    temporary_name: str | None = None
    proof_count = 0
    competing_lease_acquired = False

    def capture_temporary_name(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        nonlocal temporary_name
        if destination_name == fixture.output_root.name:
            temporary_name = source_name
        original_rename(parent_descriptor, source_name, destination_name)

    def attempt_recreation_after_final_proof(**arguments: Any) -> None:
        nonlocal competing_lease_acquired, proof_count
        original_name_proof(**arguments)
        proof_count += 1
        if proof_count != 4:
            return
        assert temporary_name is not None
        descriptor = os.open(fixture.output_parent, operator._directory_open_flags())
        try:
            competing_lease_acquired = _cooperative_exclusive_lease_acquired(descriptor)
            if competing_lease_acquired:
                os.mkdir(temporary_name, mode=0o700, dir_fd=descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(operator, "_rename_exclusive_at", capture_temporary_name)
    monkeypatch.setattr(
        operator,
        "_require_published_name_state",
        attempt_recreation_after_final_proof,
    )
    receipt = _build(fixture)

    assert proof_count == 4
    assert competing_lease_acquired is False
    assert receipt.input_custody.noncooperating_same_uid_mutation_excluded is True
    assert not _hidden_work_paths(fixture)


def test_final_source_proof_excludes_cooperative_output_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    moved_parent = (tmp_path / "leased-output-parent").resolve()
    original_source_proof = operator._require_source_binding
    proof_count = 0
    competing_lease_acquired = False

    def attempt_parent_replacement_after_final_source_proof(**arguments: Any) -> None:
        nonlocal competing_lease_acquired, proof_count
        original_source_proof(**arguments)
        proof_count += 1
        if proof_count != 2:
            return
        descriptor = os.open(fixture.output_parent, operator._directory_open_flags())
        try:
            competing_lease_acquired = _cooperative_exclusive_lease_acquired(descriptor)
            if competing_lease_acquired:
                os.rename(fixture.output_parent, moved_parent)
                fixture.output_parent.mkdir(mode=0o700)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(
        operator,
        "_require_source_binding",
        attempt_parent_replacement_after_final_source_proof,
    )
    receipt = _build(fixture)

    assert proof_count == 2
    assert competing_lease_acquired is False
    assert receipt.input_custody.noncooperating_same_uid_mutation_excluded is True
    assert fixture.output_root.is_dir()


def test_final_name_proof_excludes_cooperative_retained_writer_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_seal = operator._seal_temporary_tree
    original_name_proof = operator._require_published_name_state
    writer: int | None = None
    proof_count = 0
    competing_lease_acquired = False

    def retain_writer_before_read_only_seal(
        descriptor: int,
        receipt: operator.DevelopmentStagingViewReceipt,
    ) -> None:
        nonlocal writer
        writer = os.open("assignments.jsonl", os.O_RDWR, dir_fd=descriptor)
        original_seal(descriptor, receipt)

    def attempt_mutation_after_final_proof(**arguments: Any) -> None:
        nonlocal competing_lease_acquired, proof_count
        original_name_proof(**arguments)
        proof_count += 1
        if proof_count != 4:
            return
        assert writer is not None
        competing_lease_acquired = _cooperative_exclusive_lease_acquired(writer)
        if competing_lease_acquired:
            os.pwrite(writer, b"X", 0)
            os.fsync(writer)
            fcntl.flock(writer, fcntl.LOCK_UN)

    monkeypatch.setattr(operator, "_seal_temporary_tree", retain_writer_before_read_only_seal)
    monkeypatch.setattr(
        operator,
        "_require_published_name_state",
        attempt_mutation_after_final_proof,
    )
    try:
        receipt = _build(fixture)
    finally:
        if writer is not None:
            os.close(writer)

    assert proof_count == 4
    assert competing_lease_acquired is False
    assert receipt.input_custody.noncooperating_same_uid_mutation_excluded is True
    operator.verify_development_staging_view(
        fixture.output_root,
        expected_receipt_sha256=receipt.artifact_sha256,
    )


@pytest.mark.parametrize(
    "failure_kind",
    ("os-error", "keyboard-interrupt", "system-exit", "sigterm", "sighup"),
)
def test_postpublication_temporary_close_failure_is_indeterminate_with_recovery_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    fixture = _fixture(tmp_path)
    original_publish = operator._publish_exclusive
    original_close = os.close
    temporary_descriptor: int | None = None
    injected = False

    def capture_publication(**arguments: Any) -> None:
        nonlocal temporary_descriptor
        original_publish(**arguments)
        temporary_descriptor = arguments["temporary_descriptor"]

    def fail_proved_temporary_close(descriptor: int) -> None:
        nonlocal injected
        if not injected and descriptor == temporary_descriptor:
            injected = True
            if failure_kind == "os-error":
                raise OSError(errno.EIO, "injected close failure")
            if failure_kind == "keyboard-interrupt":
                raise KeyboardInterrupt
            if failure_kind == "system-exit":
                raise SystemExit(23)
            signum = signal.SIGTERM if failure_kind == "sigterm" else signal.SIGHUP
            os.kill(os.getpid(), signum)
            raise AssertionError("signal handler returned")
        original_close(descriptor)

    monkeypatch.setattr(operator, "_publish_exclusive", capture_publication)
    monkeypatch.setattr(operator.os, "close", fail_proved_temporary_close)
    try:
        with pytest.raises(
            operator.DevelopmentStagingPublicationIndeterminate,
            match=r"descriptor closure is indeterminate; public output=.*receipt_sha256=",
        ):
            _build(fixture)
    finally:
        monkeypatch.undo()
        if temporary_descriptor is not None:
            try:
                os.fstat(temporary_descriptor)
            except OSError:
                pass
            else:
                original_close(temporary_descriptor)

    assert injected
    receipt_path = fixture.output_root / operator.VIEW_RECEIPT_FILENAME
    receipt_bytes = receipt_path.read_bytes()
    operator.verify_development_staging_view(
        fixture.output_root,
        expected_receipt_sha256=_digest(receipt_bytes),
    )
    assert not _hidden_work_paths(fixture)


@pytest.mark.parametrize("signum", (signal.SIGINT, signal.SIGTERM, signal.SIGHUP))
def test_prepublication_signal_cleans_temporary_tree_and_preserves_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signum: int,
) -> None:
    fixture = _fixture(tmp_path)

    def interrupt_before_seal(
        _descriptor: int,
        _receipt: operator.DevelopmentStagingViewReceipt,
    ) -> None:
        os.kill(os.getpid(), signum)
        raise AssertionError("signal handler returned")

    monkeypatch.setattr(operator, "_seal_temporary_tree", interrupt_before_seal)
    with pytest.raises(operator.DevelopmentStagingInterruptedError) as failure:
        _build(fixture)

    assert failure.value.signum == signum
    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)


@pytest.mark.parametrize("signum", (signal.SIGTERM, signal.SIGHUP))
def test_postrename_signal_rolls_back_then_preserves_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signum: int,
) -> None:
    fixture = _fixture(tmp_path)
    original_rename = operator._rename_exclusive_at
    injected = False

    def rename_then_interrupt(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        nonlocal injected
        original_rename(parent_descriptor, source_name, destination_name)
        if not injected and destination_name == fixture.output_root.name:
            injected = True
            os.kill(os.getpid(), signum)
            raise AssertionError("signal handler returned")

    monkeypatch.setattr(operator, "_rename_exclusive_at", rename_then_interrupt)
    with pytest.raises(operator.DevelopmentStagingInterruptedError) as failure:
        _build(fixture)

    assert injected
    assert failure.value.signum == signum
    assert not fixture.output_root.exists()
    assert not _hidden_work_paths(fixture)
