from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from fractal_ann_diagnostics.corpora import (
    CorpusDocument,
    EvidenceQuery,
    NormalizedCorpus,
)
from fractal_ann_diagnostics.freeze_package import (
    FreezeArtifactLayout,
    FreezePackageError,
    _inspect_target,
)
from fractal_ann_diagnostics.partition_audit import (
    FROZEN_QUERY_PARTITION_CONFIG_SHA256,
    QueryPartitionLeakageError,
    audit_query_partitions,
)
from fractal_ann_diagnostics.scalable_partition_audit import (
    SCALABLE_PARTITION_ALGORITHM_SHA256,
    STRUCTURAL_EXCLUSION_REASON,
    STRUCTURAL_EXCLUSION_RULE_ID,
    STRUCTURAL_EXCLUSION_SCHEMA,
    ScalablePartitionAuditError,
    audit_staged_query_partitions,
    build_scalable_partition_audit,
    load_scalable_partition_audit,
    verify_scalable_partition_audit_against_staged,
)
from fractal_ann_diagnostics.study_data import (
    ASSIGNMENT_ALGORITHM,
    ASSIGNMENT_SCHEMA,
    INVENTORY_SCHEMA,
)

_DATASET = "demo"
_EXCLUSIONS_PATH = "partition-exclusions.jsonl"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_parts(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _normalize(text: str) -> str:
    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    return " ".join(tokens) if tokens else " ".join(normalized.split())


def _external_identity(document_id: str) -> str:
    return _sha256(_canonical([_DATASET, document_id]))


def _content_identity(*, title: str, text: str) -> str:
    return _sha256(
        _canonical(
            [
                "suite-global-canonical-document-content-v2",
                _hash_parts(title, text),
            ]
        )
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))


def _artifact(
    root: Path,
    relative_path: str,
    rows: list[dict[str, Any]],
    *,
    dataset: str | None,
    stage: str | None,
    role: str,
    visibility: str,
) -> dict[str, Any]:
    path = root / relative_path
    _write_jsonl(path, rows)
    encoded = path.read_bytes()
    return {
        "byte_count": len(encoded),
        "dataset": dataset,
        "path": relative_path,
        "record_count": len(rows),
        "role": role,
        "sha256": _sha256(encoded),
        "stage": stage,
        "visibility": visibility,
    }


def _component(*query_ids: str) -> str:
    return _sha256(_canonical(sorted(query_ids, key=lambda value: value.encode())))


def _structural_exclusion_rows(
    documents: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    specifications = (
        (
            "x-exact-a",
            "train",
            "alpha beta gamma delta epsilon zeta",
            "doc-x-a",
        ),
        (
            "x-exact-b",
            "train",
            "alpha beta gamma delta epsilon zeta",
            "doc-x-b",
        ),
        (
            "x-near",
            "dev",
            "alpha beta gamma delta epsilon eta",
            "doc-x-shared",
        ),
        (
            "x-positive",
            "dev",
            "a wholly separate excluded question with enough distinct words",
            "doc-x-shared",
        ),
    )
    component_sha256 = _component(*(row[0] for row in specifications))
    rows: list[dict[str, Any]] = []
    for query_id, source_split, text, document_id in specifications:
        title, document_text = documents[document_id]
        identities = sorted(
            {
                _external_identity(document_id),
                _content_identity(title=title, text=document_text),
            }
        )
        rows.append(
            {
                "dataset": _DATASET,
                "normalized_query_text_sha256": _sha256(_normalize(text).encode("utf-8")),
                "partition_component_sha256": component_sha256,
                "positive_relevance_identity_sha256s": identities,
                "query_id": query_id,
                "query_text_sha256": _sha256(text.encode("utf-8")),
                "reason": STRUCTURAL_EXCLUSION_REASON,
                "rule_id": STRUCTURAL_EXCLUSION_RULE_ID,
                "schema_version": STRUCTURAL_EXCLUSION_SCHEMA,
                "source_split": source_split,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["dataset"]).encode(),
            str(row["partition_component_sha256"]).encode(),
            str(row["source_split"]).encode(),
            str(row["query_id"]).encode(),
        ),
    )


def _stage(
    root: Path,
    *,
    structural_exclusions: bool = True,
) -> Path:
    root.mkdir()
    document_values = {
        "doc-cal": ("Calibration", "A calibration document."),
        "doc-fit-a": ("Fit alias", "Identical fit content."),
        "doc-fit-b": ("Fit alias", "Identical fit content."),
        "doc-negative": ("Negative", "A nonrelevant document."),
        "doc-sealed": ("Sealed", "A sealed document."),
        "doc-x-a": ("Excluded A", "First excluded evidence."),
        "doc-x-b": ("Excluded B", "Second excluded evidence."),
        "doc-x-shared": ("Excluded shared", "Shared excluded evidence."),
    }
    documents = [
        {"id": document_id, "text": text, "title": title}
        for document_id, (title, text) in sorted(document_values.items())
    ]
    stage_queries = {
        "fit": [
            {"id": "q-fit-a", "text": "How is fit evidence alpha selected?"},
            {"id": "q-fit-b", "text": "Which fit evidence beta is selected?"},
        ],
        "calibration": [{"id": "q-cal", "text": "How is calibration evidence scored?"}],
        "sealed": [{"id": "q-sealed", "text": "Which sealed outcome is evaluated?"}],
    }
    stage_qrels = {
        "fit": [
            {"document_id": "doc-fit-a", "query_id": "q-fit-a", "relevance": 1},
            {
                "document_id": "doc-negative",
                "query_id": "q-fit-a",
                "relevance": -1,
            },
            {"document_id": "doc-fit-b", "query_id": "q-fit-b", "relevance": 1},
        ],
        "calibration": [{"document_id": "doc-cal", "query_id": "q-cal", "relevance": 1}],
        "sealed": [
            {
                "document_id": "doc-sealed",
                "query_id": "q-sealed",
                "relevance": 1,
            }
        ],
    }
    component_by_query = {
        "q-fit-a": _component("q-fit-a", "q-fit-b"),
        "q-fit-b": _component("q-fit-a", "q-fit-b"),
        "q-cal": _component("q-cal"),
        "q-sealed": _component("q-sealed"),
    }
    stage_by_query = {row["id"]: stage for stage, rows in stage_queries.items() for row in rows}
    assignments = [
        {
            "assignment_key_sha256": _sha256(f"assignment:{query_id}".encode()),
            "dataset": _DATASET,
            "domain": None,
            "partition_component_sha256": component_by_query[query_id],
            "query_id": query_id,
            "query_text_sha256": _sha256(
                next(
                    row["text"]
                    for rows in stage_queries.values()
                    for row in rows
                    if row["id"] == query_id
                ).encode()
            ),
            "schema_version": ASSIGNMENT_SCHEMA,
            "source_split": f"fixture-{stage_by_query[query_id]}",
            "stage": stage_by_query[query_id],
        }
        for query_id in sorted(stage_by_query)
    ]

    artifacts = [
        _artifact(
            root,
            f"datasets/{_DATASET}/corpus.jsonl",
            documents,
            dataset=_DATASET,
            stage=None,
            role="corpus",
            visibility="online",
        ),
        _artifact(
            root,
            "assignments.jsonl",
            assignments,
            dataset=None,
            stage=None,
            role="assignments",
            visibility="online",
        ),
        _artifact(
            root,
            _EXCLUSIONS_PATH,
            (_structural_exclusion_rows(document_values) if structural_exclusions else []),
            dataset=None,
            stage=None,
            role="query-partition-structural-exclusions",
            visibility="protocol",
        ),
    ]
    for stage in ("fit", "calibration", "sealed"):
        query_path = (
            f"datasets/{_DATASET}/sealed/online/queries.jsonl"
            if stage == "sealed"
            else f"datasets/{_DATASET}/{stage}/queries.jsonl"
        )
        qrel_path = (
            f"datasets/{_DATASET}/sealed/custody/qrels.jsonl"
            if stage == "sealed"
            else f"datasets/{_DATASET}/{stage}/qrels.jsonl"
        )
        artifacts.extend(
            (
                _artifact(
                    root,
                    query_path,
                    stage_queries[stage],
                    dataset=_DATASET,
                    stage=stage,
                    role="queries",
                    visibility="online",
                ),
                _artifact(
                    root,
                    qrel_path,
                    stage_qrels[stage],
                    dataset=_DATASET,
                    stage=stage,
                    role="qrels",
                    visibility="custody" if stage == "sealed" else "online",
                ),
            )
        )
    inventory = {
        "artifacts": sorted(artifacts, key=lambda row: row["path"].encode()),
        "assignment_algorithm": {
            "component_edges": [
                "normalized-query-text-equality",
                "registered-near-duplicate-token-rule",
                "shared-positive-document-content",
                "shared-positive-relevance-document",
            ],
            "cross_source_split_policy": "exclude-entire-component-v1",
            "fit_calibration_component_ratio": "4:1",
            "name": ASSIGNMENT_ALGORITHM,
            "three_way_component_ratio": "3:1:1",
        },
        "assignment_seed_sha256": "1" * 64,
        "bright_document_identity": {},
        "bright_domains": [],
        "config_sha256": "2" * 64,
        "counts": {
            _DATASET: {
                "calibration_queries": 1,
                "documents": len(documents),
                "fit_queries": 2,
                "qrels": 5,
                "sealed_queries": 1,
                "partition_excluded_queries": (4 if structural_exclusions else 0),
            }
        },
        "hotpotqa_fullwiki_scope": {},
        "withhold_sealed_labels_from_online_process": True,
        "schema_version": INVENTORY_SCHEMA,
        "sources": [
            {
                "byte_count": 10,
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "sha256": "3" * 64,
                "source_id": "fixture-source",
            }
        ],
    }
    _write_inventory(root, inventory)
    return root


def _write_inventory(root: Path, inventory: dict[str, Any]) -> None:
    encoded = _canonical(inventory) + b"\n"
    (root / "inventory.json").write_bytes(encoded)
    (root / "inventory.sha256").write_text(
        f"{_sha256(encoded)}  inventory.json\n",
        encoding="ascii",
    )


def _inventory(root: Path) -> dict[str, Any]:
    return json.loads((root / "inventory.json").read_text(encoding="utf-8"))


def _repin_inventory(root: Path, inventory: dict[str, Any] | None = None) -> None:
    value = _inventory(root) if inventory is None else inventory
    for artifact in value["artifacts"]:
        encoded = (root / artifact["path"]).read_bytes()
        artifact["byte_count"] = len(encoded)
        artifact["record_count"] = encoded.count(b"\n")
        artifact["sha256"] = _sha256(encoded)
    value["artifacts"].sort(key=lambda row: row["path"].encode())
    _write_inventory(root, value)


def _rows(root: Path, relative_path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in (root / relative_path).read_text(encoding="utf-8").splitlines()
    ]


def _mutate_rows(
    root: Path,
    relative_path: str,
    mutation: Callable[[list[dict[str, Any]]], None],
) -> None:
    rows = _rows(root, relative_path)
    mutation(rows)
    _write_jsonl(root / relative_path, rows)
    _repin_inventory(root)


def _assignment_text_digest(rows: list[dict[str, Any]], query_id: str, text: str) -> None:
    for row in rows:
        if row["query_id"] == query_id:
            row["query_text_sha256"] = _sha256(text.encode("utf-8"))
            return
    raise AssertionError(query_id)


def test_build_load_and_recompute_inventory_derived_audit(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")
    output = tmp_path / "query-partition-audit.json"

    receipt = build_scalable_partition_audit(staged, output)
    loaded = load_scalable_partition_audit(
        output,
        expected_artifact_sha256=receipt.artifact_sha256,
        expected_inventory_sha256=receipt.staged_inventory_sha256,
    )
    recomputed = verify_scalable_partition_audit_against_staged(output, staged)

    assert loaded == receipt == recomputed
    assert receipt.algorithm_sha256 == SCALABLE_PARTITION_ALGORITHM_SHA256
    assert receipt.near_duplicate_config_sha256 == FROZEN_QUERY_PARTITION_CONFIG_SHA256
    assert receipt.query_count == receipt.assignment_count == 4
    assert receipt.qrel_count == 5
    assert receipt.assignment_component_count == receipt.audit_component_count == 3
    assert receipt.shared_positive_document_edge_count == 0
    assert receipt.shared_positive_document_content_edge_count == 1
    assert receipt.structural_exclusion_query_count == 4
    assert receipt.structural_exclusion_component_count == 1
    assert receipt.structural_exclusion_counts[0].dataset == _DATASET
    assert receipt.cross_stage_component_count == 0
    sealed_qrel = next(
        source
        for source in receipt.source_artifacts
        if source.stage == "sealed" and source.role == "qrels"
    )
    assert sealed_qrel.visibility == "custody"
    assert any(source.role == "corpus" for source in receipt.source_artifacts)
    assert b"x-exact-a" not in receipt.canonical_file_bytes()
    assert _sha256(output.read_bytes()) == receipt.artifact_sha256


def test_empty_structural_exclusion_artifact_is_still_required_and_bound(
    tmp_path: Path,
) -> None:
    staged = _stage(tmp_path / "staged", structural_exclusions=False)

    receipt = audit_staged_query_partitions(staged)

    assert receipt.structural_exclusion_query_count == 0
    assert receipt.structural_exclusion_component_count == 0
    assert receipt.structural_exclusion_counts == ()
    assert receipt.structural_exclusion_artifact_sha256 == _sha256(b"")
    assert receipt.structural_exclusion_membership_sha256 == _sha256(_canonical([]))


def test_missing_structural_exclusion_source_is_rejected(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged", structural_exclusions=False)
    inventory = _inventory(staged)
    inventory["artifacts"] = [
        row for row in inventory["artifacts"] if row["path"] != _EXCLUSIONS_PATH
    ]
    (staged / _EXCLUSIONS_PATH).unlink()
    _write_inventory(staged, inventory)

    with pytest.raises(ScalablePartitionAuditError, match="structural-exclusion"):
        audit_staged_query_partitions(staged)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rule_id", "unregistered-rule", "rule differs"),
        ("reason", "operator-choice", "reason differs"),
        ("schema_version", "unknown-schema", "schema differs"),
    ],
)
def test_structural_exclusion_contract_mutations_are_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    staged = _stage(tmp_path / "staged")

    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0][field] = value

    _mutate_rows(staged, _EXCLUSIONS_PATH, mutate)

    with pytest.raises(ScalablePartitionAuditError, match=message):
        audit_staged_query_partitions(staged)


def test_structural_exclusion_component_digest_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    staged = _stage(tmp_path / "staged")

    def mutate(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            row["partition_component_sha256"] = "9" * 64

    _mutate_rows(staged, _EXCLUSIONS_PATH, mutate)

    with pytest.raises(ScalablePartitionAuditError, match="component digest"):
        audit_staged_query_partitions(staged)


def test_structural_exclusion_must_span_source_splits(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")

    def mutate(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            row["source_split"] = "train"
        rows.sort(key=lambda row: str(row["query_id"]).encode())

    _mutate_rows(staged, _EXCLUSIONS_PATH, mutate)

    with pytest.raises(ScalablePartitionAuditError, match="does not span"):
        audit_staged_query_partitions(staged)


def test_assignment_omission_is_rejected_after_inventory_repin(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")

    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[:] = [row for row in rows if row["query_id"] != "q-cal"]

    _mutate_rows(staged, "assignments.jsonl", mutate)

    with pytest.raises(ScalablePartitionAuditError, match="absent from assignments"):
        audit_staged_query_partitions(staged)


def test_query_text_mutation_is_rejected_after_inventory_repin(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")
    path = f"datasets/{_DATASET}/calibration/queries.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["text"] = "A changed calibration question"

    _mutate_rows(staged, path, mutate)

    with pytest.raises(ScalablePartitionAuditError, match="text digest differs"):
        audit_staged_query_partitions(staged)


def test_assignment_component_cannot_span_stages(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")
    fit_component = _component("q-fit-a", "q-fit-b")

    def mutate(rows: list[dict[str, Any]]) -> None:
        next(row for row in rows if row["query_id"] == "q-cal")["partition_component_sha256"] = (
            fit_component
        )

    _mutate_rows(staged, "assignments.jsonl", mutate)

    with pytest.raises(ScalablePartitionAuditError, match="spans a corpus or stage"):
        audit_staged_query_partitions(staged)


def test_exact_normalized_cross_stage_query_is_rejected(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")
    calibration_path = f"datasets/{_DATASET}/calibration/queries.jsonl"
    sealed_path = f"datasets/{_DATASET}/sealed/online/queries.jsonl"
    text = _rows(staged, calibration_path)[0]["text"]

    def mutate_sealed(rows: list[dict[str, Any]]) -> None:
        rows[0]["text"] = text

    _mutate_rows(staged, sealed_path, mutate_sealed)

    def mutate_assignments(rows: list[dict[str, Any]]) -> None:
        _assignment_text_digest(rows, "q-sealed", text)

    _mutate_rows(staged, "assignments.jsonl", mutate_assignments)

    with pytest.raises(ScalablePartitionAuditError, match="crosses stages"):
        audit_staged_query_partitions(staged)


def test_shared_positive_external_document_crossing_is_rejected(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")
    sealed_qrels = f"datasets/{_DATASET}/sealed/custody/qrels.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[0]["document_id"] = "doc-cal"

    _mutate_rows(staged, sealed_qrels, mutate)

    with pytest.raises(ScalablePartitionAuditError, match="document crosses"):
        audit_staged_query_partitions(staged)


def test_shared_positive_document_content_crossing_is_rejected(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")
    corpus_path = f"datasets/{_DATASET}/corpus.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        calibration = next(row for row in rows if row["id"] == "doc-cal")
        sealed = next(row for row in rows if row["id"] == "doc-sealed")
        sealed["title"] = calibration["title"]
        sealed["text"] = calibration["text"]

    _mutate_rows(staged, corpus_path, mutate)

    with pytest.raises(ScalablePartitionAuditError, match="content crosses"):
        audit_staged_query_partitions(staged)


def test_shared_positive_content_across_corpora_and_stages_is_rejected(
    tmp_path: Path,
) -> None:
    staged = _stage(tmp_path / "staged")
    dataset = "other-corpus"
    query_id = "q-suite-sealed"
    query_text = "Which cross corpus evidence is evaluated?"
    calibration_document = next(
        row for row in _rows(staged, f"datasets/{_DATASET}/corpus.jsonl") if row["id"] == "doc-cal"
    )
    inventory = _inventory(staged)
    inventory["artifacts"].extend(
        [
            _artifact(
                staged,
                f"datasets/{dataset}/corpus/part-00000.jsonl",
                [
                    {
                        "id": "doc-suite-alias",
                        "text": calibration_document["text"],
                        "title": calibration_document["title"],
                    }
                ],
                dataset=dataset,
                stage=None,
                role="corpus-shard",
                visibility="online",
            ),
            _artifact(
                staged,
                f"datasets/{dataset}/sealed/online/queries.jsonl",
                [{"id": query_id, "text": query_text}],
                dataset=dataset,
                stage="sealed",
                role="queries",
                visibility="online",
            ),
            _artifact(
                staged,
                f"datasets/{dataset}/sealed/custody/qrels.jsonl",
                [
                    {
                        "document_id": "doc-suite-alias",
                        "query_id": query_id,
                        "relevance": 1,
                    }
                ],
                dataset=dataset,
                stage="sealed",
                role="qrels",
                visibility="custody",
            ),
        ]
    )
    assignments = _rows(staged, "assignments.jsonl")
    assignments.append(
        {
            "assignment_key_sha256": _sha256(f"assignment:{query_id}".encode()),
            "dataset": dataset,
            "domain": None,
            "partition_component_sha256": _component(query_id),
            "query_id": query_id,
            "query_text_sha256": _sha256(query_text.encode()),
            "schema_version": ASSIGNMENT_SCHEMA,
            "source_split": "fixture-test",
            "stage": "sealed",
        }
    )
    _write_jsonl(staged / "assignments.jsonl", assignments)
    inventory["counts"][dataset] = {
        "calibration_queries": 0,
        "documents": 1,
        "fit_queries": 0,
        "qrels": 1,
        "sealed_queries": 1,
        "partition_excluded_queries": 0,
    }
    _repin_inventory(staged, inventory)

    with pytest.raises(ScalablePartitionAuditError, match="content crosses"):
        audit_staged_query_partitions(staged)


def test_missing_qrel_stage_is_rejected_even_after_inventory_repin(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")
    missing_path = f"datasets/{_DATASET}/calibration/qrels.jsonl"
    inventory = _inventory(staged)
    inventory["artifacts"] = [row for row in inventory["artifacts"] if row["path"] != missing_path]
    (staged / missing_path).unlink()
    _write_inventory(staged, inventory)

    with pytest.raises(ScalablePartitionAuditError, match="same corpus/stage set"):
        audit_staged_query_partitions(staged)


def _legacy_corpus(*, stage: str, query_id: str, text: str) -> NormalizedCorpus:
    corpus_name = f"demo-{query_id}"
    document = CorpusDocument(
        document_id=0,
        external_id=f"doc-{query_id}",
        title="Fixture",
        text="Fixture body",
        source_uri=f"demo://document/{query_id}",
        content_hash="sha256:" + _sha256(query_id.encode()),
    )
    query = EvidenceQuery(
        query_id=query_id,
        query_family=query_id,
        text=text,
        corpus=corpus_name,
        stage=stage,
        answer=None,
        gold_evidence=None,
        relevant_document_ids=(),
    )
    return NormalizedCorpus(
        name=corpus_name,
        stage=stage,
        documents=(document,),
        queries=(query,),
    )


def test_registered_near_duplicate_rule_matches_existing_audit(tmp_path: Path) -> None:
    fit_text = "which policy revision grants authorized retrieval access today"
    sealed_text = "which policy revision denies authorized retrieval access today"
    staged = _stage(tmp_path / "staged")
    calibration_path = f"datasets/{_DATASET}/calibration/queries.jsonl"
    sealed_path = f"datasets/{_DATASET}/sealed/online/queries.jsonl"

    def mutate_calibration(rows: list[dict[str, Any]]) -> None:
        rows[0]["text"] = fit_text

    def mutate_sealed(rows: list[dict[str, Any]]) -> None:
        rows[0]["text"] = sealed_text

    _mutate_rows(staged, calibration_path, mutate_calibration)
    _mutate_rows(staged, sealed_path, mutate_sealed)

    def mutate_assignments(rows: list[dict[str, Any]]) -> None:
        _assignment_text_digest(rows, "q-cal", fit_text)
        _assignment_text_digest(rows, "q-sealed", sealed_text)

    _mutate_rows(staged, "assignments.jsonl", mutate_assignments)

    with pytest.raises(ScalablePartitionAuditError, match="crosses stages"):
        audit_staged_query_partitions(staged)
    with pytest.raises(QueryPartitionLeakageError) as captured:
        audit_query_partitions(
            (
                _legacy_corpus(
                    stage="development-calibration",
                    query_id="legacy-fit",
                    text=fit_text,
                ),
                _legacy_corpus(
                    stage="sealed",
                    query_id="legacy-sealed",
                    text=sealed_text,
                ),
            )
        )
    assert "normalized-text-near" in {edge.relation for edge in captured.value.audit.edges}


def test_receipt_mutation_and_noncanonical_bytes_are_rejected(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")
    output = tmp_path / "audit.json"
    receipt = build_scalable_partition_audit(staged, output)
    value = json.loads(output.read_text(encoding="utf-8"))
    value["unexpected"] = True
    output.write_bytes(_canonical(value) + b"\n")

    with pytest.raises(ScalablePartitionAuditError, match="fields differ"):
        load_scalable_partition_audit(output)

    output.write_bytes(json.dumps(receipt.to_dict(), indent=2).encode() + b"\n")
    with pytest.raises(ScalablePartitionAuditError, match="canonical JSON"):
        load_scalable_partition_audit(output)


def test_freeze_inspection_requires_the_typed_audit_receipt(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")
    artifact_root = tmp_path / "artifacts"
    target = artifact_root / "development" / "query-partition-audit.json"
    target.parent.mkdir(parents=True)
    receipt = build_scalable_partition_audit(staged, target)
    layout = FreezeArtifactLayout(
        artifact_id="query-partition-audit",
        role="query-partition-audit",
        relative_path="development/query-partition-audit.json",
        kind="file",
    )

    inspected = _inspect_target(layout, artifact_root, _REPOSITORY_ROOT)
    assert inspected["sha256"] == receipt.artifact_sha256
    assert inspected["revision"] == f"sha256:{receipt.artifact_sha256}"

    target.write_bytes(_canonical({"sha256": receipt.artifact_sha256}) + b"\n")
    with pytest.raises(FreezePackageError, match="invalid typed query-partition audit"):
        _inspect_target(layout, artifact_root, _REPOSITORY_ROOT)


def test_cli_build_verify_and_no_overwrite(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")
    output = tmp_path / "audit.json"
    command = [
        sys.executable,
        "-m",
        "fractal_ann_diagnostics.scalable_partition_audit",
    ]
    built = subprocess.run(
        [*command, "build", "--staged-root", str(staged), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    payload = json.loads(built.stdout)
    verified = subprocess.run(
        [
            *command,
            "verify-staged",
            "--audit",
            str(output),
            "--staged-root",
            str(staged),
            "--expected-sha256",
            payload["artifact_sha256"],
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr
    repeated = subprocess.run(
        [*command, "build", "--staged-root", str(staged), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_loader_rejects_symlinked_receipt(tmp_path: Path) -> None:
    staged = _stage(tmp_path / "staged")
    target = tmp_path / "target.json"
    build_scalable_partition_audit(staged, target)
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(ScalablePartitionAuditError):
        load_scalable_partition_audit(link)
