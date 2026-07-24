from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from fractal_ann_diagnostics.label_separation import load_sealed_label_artifact
from fractal_ann_diagnostics.scalable_custody import (
    CUSTODY_QUERY_KEY_ROW_SCHEMA,
    DOCUMENT_ROW_ORDER_ALGORITHM,
    FAMILY_SELECTION_ALGORITHM,
    FAMILY_SELECTION_DOMAIN,
    KEY_DERIVATION_ALGORITHM,
    NESTED_ROWS_PER_FAMILY,
    NESTED_TRIAL_SOURCE_DOMAIN,
    PROVENANCE_PATH,
    QUERY_KEY_MAP_PATH,
    REPRESENTATIVE_SELECTION_ALGORITHM,
    REPRESENTATIVE_SELECTION_DOMAIN,
    SCALABLE_CUSTODY_CONFIG_SCHEMA,
    SEALED_LABEL_PATH,
    ScalableCustodyError,
    ScalableCustodyPlan,
    build_scalable_custody_from_config,
    build_scalable_custody_package,
    load_scalable_custody_config,
    verify_query_trial_key_parity,
    verify_scalable_custody_package,
)
from fractal_ann_diagnostics.study_data import ASSIGNMENT_SCHEMA, INVENTORY_SCHEMA

_SECRET = bytes(range(32))
_EXECUTION_SHA256 = hashlib.sha256(b"fixture-frozen-execution").hexdigest()
_SELECTION_SEED_SHA256 = hashlib.sha256(b"fixture-family-selection-seed").hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _hash_parts(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _opaque(
    *,
    domain: str,
    corpus: str,
    source_value: str,
) -> str:
    digest = hmac.new(_SECRET, digestmod=hashlib.sha256)
    for value in (
        KEY_DERIVATION_ALGORITHM,
        domain,
        "fixture-custody-key",
        corpus,
        "sealed",
        source_value,
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _nested_trial_source(query_id: str, nested_index: int) -> str:
    return _canonical([NESTED_TRIAL_SOURCE_DOMAIN, query_id, nested_index]).decode("utf-8")


def _representative_query_id(
    *,
    plan: ScalableCustodyPlan,
    component: str,
    query_ids: tuple[str, ...],
) -> str:
    def rank(query_id: str) -> tuple[str, str]:
        query_id_sha256 = hashlib.sha256(query_id.encode()).hexdigest()
        rank_sha256 = _hash_parts(
            REPRESENTATIVE_SELECTION_DOMAIN,
            REPRESENTATIVE_SELECTION_ALGORITHM,
            plan.corpus,
            plan.stage,
            plan.selection_seed_sha256,
            component,
            query_id_sha256,
        )
        return rank_sha256, query_id_sha256

    return min(query_ids, key=rank)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> tuple[int, str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(_canonical(row) + b"\n" for row in rows)
    path.write_bytes(encoded)
    return len(encoded), hashlib.sha256(encoded).hexdigest(), len(rows)


def _artifact(
    root: Path,
    path: str,
    rows: list[dict[str, Any]],
    *,
    dataset: str | None,
    stage: str | None,
    role: str,
    visibility: str,
) -> dict[str, Any]:
    byte_count, sha256, record_count = _write_jsonl(root / path, rows)
    return {
        "byte_count": byte_count,
        "dataset": dataset,
        "path": path,
        "record_count": record_count,
        "role": role,
        "sha256": sha256,
        "stage": stage,
        "visibility": visibility,
    }


def _stage(
    root: Path,
    *,
    corpus: str,
    documents: list[dict[str, str]],
    queries: list[dict[str, str]],
    qrels: list[dict[str, Any]],
    evidence: list[dict[str, Any]] | None,
    components: dict[str, str] | None = None,
    sharded: bool = False,
    selected_families: int | None = None,
    available_families: int | None = None,
    selection_seed_sha256: str = _SELECTION_SEED_SHA256,
) -> tuple[ScalableCustodyPlan, dict[str, dict[str, Any]]]:
    root.mkdir()
    artifacts: list[dict[str, Any]] = []
    if sharded:
        split = max(1, len(documents) // 2)
        chunks = [documents[:split], documents[split:]]
        for part, rows in enumerate(chunk for chunk in chunks if chunk):
            artifacts.append(
                _artifact(
                    root,
                    f"datasets/{corpus}/corpus/part-{part:05d}.jsonl",
                    rows,
                    dataset=corpus,
                    stage=None,
                    role="corpus-shard",
                    visibility="online",
                )
            )
    else:
        artifacts.append(
            _artifact(
                root,
                f"datasets/{corpus}/corpus.jsonl",
                documents,
                dataset=corpus,
                stage=None,
                role="corpus",
                visibility="online",
            )
        )
    artifacts.append(
        _artifact(
            root,
            f"datasets/{corpus}/sealed/online/queries.jsonl",
            queries,
            dataset=corpus,
            stage="sealed",
            role="queries",
            visibility="online",
        )
    )
    artifacts.append(
        _artifact(
            root,
            f"datasets/{corpus}/sealed/custody/qrels.jsonl",
            qrels,
            dataset=corpus,
            stage="sealed",
            role="qrels",
            visibility="custody",
        )
    )
    if evidence is not None:
        artifacts.append(
            _artifact(
                root,
                f"datasets/{corpus}/sealed/custody/evidence-bundles.jsonl",
                evidence,
                dataset=corpus,
                stage="sealed",
                role="evidence-bundles",
                visibility="custody",
            )
        )
    component_by_query = components or {
        row["id"]: hashlib.sha256(f"component:{row['id']}".encode()).hexdigest() for row in queries
    }
    assignments = [
        {
            "assignment_key_sha256": hashlib.sha256(f"assignment:{row['id']}".encode()).hexdigest(),
            "dataset": corpus,
            "domain": None,
            "partition_component_sha256": component_by_query[row["id"]],
            "query_id": row["id"],
            "query_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            "schema_version": ASSIGNMENT_SCHEMA,
            "source_split": "fixture-sealed",
            "stage": "sealed",
        }
        for row in queries
    ]
    artifacts.append(
        _artifact(
            root,
            "assignments.jsonl",
            assignments,
            dataset=None,
            stage=None,
            role="assignments",
            visibility="online",
        )
    )
    artifacts.sort(key=lambda row: row["path"].encode())
    inventory = {
        "artifacts": artifacts,
        "assignment_algorithm": {},
        "assignment_seed_sha256": "1" * 64,
        "bright_document_identity": {},
        "bright_domains": [],
        "config_sha256": "2" * 64,
        "counts": {
            corpus: {
                "documents": len(documents),
                "sealed_queries": len(queries),
            }
        },
        "hotpotqa_fullwiki_scope": {},
        "withhold_sealed_labels_from_online_process": True,
        "schema_version": INVENTORY_SCHEMA,
        "sources": [
            {
                "byte_count": 1,
                "revision": "fixture-revision-1",
                "sha256": "3" * 64,
                "source_id": "fixture-source",
            }
        ],
    }
    inventory_bytes = _canonical(inventory) + b"\n"
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    (root / "inventory.json").write_bytes(inventory_bytes)
    (root / "inventory.sha256").write_text(
        f"{inventory_sha256}  inventory.json\n",
        encoding="ascii",
    )
    plan = ScalableCustodyPlan(
        corpus=corpus,
        staged_inventory_sha256=inventory_sha256,
        execution_artifact_sha256=_EXECUTION_SHA256,
        hmac_key_id="fixture-custody-key",
        expected_document_count=len(documents),
        available_families=(
            len(set(component_by_query.values()))
            if available_families is None
            else available_families
        ),
        selected_families=(
            len(set(component_by_query.values()))
            if selected_families is None
            else selected_families
        ),
        selection_seed_sha256=selection_seed_sha256,
        allowlisted_paths=tuple(row["path"] for row in artifacts),
    )
    return plan, {row["path"]: row for row in artifacts}


def _write_cli_config(
    root: Path,
    *,
    staged: Path,
    plan: ScalableCustodyPlan,
    source_rows: dict[str, dict[str, Any]],
    key_path: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    hmac_key_path = key_path or (root / "custody-hmac.key")
    if key_path is None:
        hmac_key_path.write_bytes(_SECRET)
        hmac_key_path.chmod(0o600)
    config = {
        "available_families": plan.available_families,
        "corpus": plan.corpus,
        "execution_artifact_sha256": plan.execution_artifact_sha256,
        "expected_document_count": plan.expected_document_count,
        "family_selection_algorithm": FAMILY_SELECTION_ALGORITHM,
        "hmac_key": {
            "byte_count": len(_SECRET),
            "path": str(hmac_key_path),
            "sha256": hashlib.sha256(_SECRET).hexdigest(),
        },
        "hmac_key_id": plan.hmac_key_id,
        "nested_rows_per_family": plan.nested_rows_per_family,
        "representative_selection_algorithm": (plan.representative_selection_algorithm),
        "schema_version": SCALABLE_CUSTODY_CONFIG_SCHEMA,
        "selected_families": plan.selected_families,
        "selection_seed_sha256": plan.selection_seed_sha256,
        "source_artifacts": [source_rows[path] for path in sorted(source_rows)],
        "stage": plan.stage,
        "staged_inventory_sha256": plan.staged_inventory_sha256,
        "staged_root": str(staged),
    }
    config_path = root / "scalable-custody-config.json"
    config_path.write_bytes(_canonical(config) + b"\n")
    config_path.chmod(0o600)
    return config_path, hmac_key_path, config


@pytest.mark.parametrize(
    ("corpus", "external_id", "title", "text", "locator", "expected_uri", "parts", "sharded"),
    [
        (
            "scifact",
            "17",
            "A scientific title",
            "Sentence zero.\nSentence one.",
            "sentence:1",
            "scifact://document/17",
            ("A scientific title", "Sentence zero.", "Sentence one."),
            False,
        ),
        (
            "hotpotqa-fullwiki",
            "Alpha/Beta",
            "Alpha/Beta",
            "First fact.\nSecond fact.",
            "sentence:0",
            "hotpotqa-fullwiki://title/Alpha%2FBeta",
            ("Alpha/Beta", "First fact.\nSecond fact."),
            True,
        ),
        (
            "t2-ragbench",
            "context/9",
            "filing.pdf",
            "Revenue was 42.",
            "document",
            "t2-ragbench://context/context%2F9",
            ("filing.pdf", "Revenue was 42."),
            False,
        ),
    ],
)
def test_streaming_package_maps_all_evidence_corpora_exactly(
    tmp_path: Path,
    corpus: str,
    external_id: str,
    title: str,
    text: str,
    locator: str,
    expected_uri: str,
    parts: tuple[str, ...],
    sharded: bool,
) -> None:
    query_id = f"custody-query-id-SENTINEL-{corpus}"
    answer = f"custody-answer-SENTINEL-{corpus}"
    component = hashlib.sha256(f"family:{corpus}".encode()).hexdigest()
    staged = tmp_path / "staged"
    output = tmp_path / "custody"
    plan, source_rows = _stage(
        staged,
        corpus=corpus,
        documents=[{"id": external_id, "text": text, "title": title}],
        queries=[{"id": query_id, "text": "Which claim is supported?"}],
        qrels=[{"document_id": external_id, "query_id": query_id, "relevance": 1}],
        evidence=[
            {
                "answer": answer,
                "evidence_bundles": [
                    {
                        "bundle_id": "gold-bundle",
                        "locations": [{"document_id": external_id, "locator": locator}],
                    }
                ],
                "label_metadata": [["evidence_class", "fixture"]],
                "query_id": query_id,
            }
        ],
        components={query_id: component},
        sharded=sharded,
    )

    receipt = build_scalable_custody_package(
        staged,
        output,
        plan=plan,
        hmac_secret=_SECRET,
    )

    assert receipt.execution_artifact_sha256 == _EXECUTION_SHA256
    assert receipt.staged_inventory_sha256 == plan.staged_inventory_sha256
    assert {row.path: row.sha256 for row in receipt.source_artifacts} == {
        path: row["sha256"] for path, row in source_rows.items()
    }
    expected_content_sha256 = _hash_parts(*parts)
    assert (output / PROVENANCE_PATH).read_bytes() == bytes.fromhex(expected_content_sha256)
    online_bytes = (output / QUERY_KEY_MAP_PATH).read_bytes()
    assert _canonical(query_id) not in online_bytes
    assert _canonical(answer) not in online_bytes
    assert _canonical(external_id) not in online_bytes
    online = [json.loads(line) for line in online_bytes.splitlines()]
    assert len(online) == NESTED_ROWS_PER_FAMILY
    assert all(
        set(row)
        == {
            "corpus",
            "family_key",
            "nested_index",
            "query_row",
            "schema_version",
            "stage",
            "text",
            "trial_key",
        }
        for row in online
    )
    assert [row["nested_index"] for row in online] == list(range(NESTED_ROWS_PER_FAMILY))
    assert all(row["schema_version"] == CUSTODY_QUERY_KEY_ROW_SCHEMA for row in online)
    assert {row["trial_key"] for row in online} == {
        _opaque(
            domain="trial",
            corpus=corpus,
            source_value=_nested_trial_source(query_id, nested_index),
        )
        for nested_index in range(NESTED_ROWS_PER_FAMILY)
    }
    assert {row["family_key"] for row in online} == {
        _opaque(domain="family", corpus=corpus, source_value=component)
    }

    sealed = load_sealed_label_artifact(output / SEALED_LABEL_PATH)
    assert sealed.execution_artifact_sha256 == _EXECUTION_SHA256
    assert sealed.document_count == 1
    assert len(sealed.labels) == NESTED_ROWS_PER_FAMILY
    for labels in sealed.labels:
        assert labels.answer == answer
        assert labels.relevant_document_ids == (0,)
        assert labels.label_metadata == (("evidence_class", "fixture"),)
        location = labels.evidence_bundles[0].locations[0]
        assert location.document_id == 0
        assert location.source_uri == expected_uri
        assert location.locator == locator
        assert location.content_hash == f"sha256:{expected_content_sha256}"
    assert receipt.query_count == NESTED_ROWS_PER_FAMILY
    assert (
        verify_scalable_custody_package(
            output,
            expected_execution_artifact_sha256=_EXECUTION_SHA256,
        )
        == receipt
    )


def test_bright_keeps_evidence_undefined_and_preserves_component_families(
    tmp_path: Path,
) -> None:
    corpus = "bright"
    query_ids = ("bright-query-SENTINEL-a", "bright-query-SENTINEL-b")
    component = hashlib.sha256(b"shared-partition-component").hexdigest()
    staged = tmp_path / "staged"
    output = tmp_path / "custody"
    plan, _ = _stage(
        staged,
        corpus=corpus,
        documents=[
            {"id": "10", "text": "First document.", "title": "Ten"},
            {"id": "2", "text": "Second document.", "title": "Two"},
        ],
        queries=[
            {"id": query_ids[0], "text": "First retrieval query?"},
            {"id": query_ids[1], "text": "Second retrieval query?"},
        ],
        qrels=[
            {"document_id": "10", "query_id": query_ids[0], "relevance": 1},
            {"document_id": "2", "query_id": query_ids[1], "relevance": 2},
        ],
        evidence=None,
        components={query_id: component for query_id in query_ids},
    )
    receipt = build_scalable_custody_package(
        staged,
        output,
        plan=plan,
        hmac_secret=_SECRET,
    )
    representative_query_id = _representative_query_id(
        plan=plan,
        component=component,
        query_ids=query_ids,
    )
    expected_text = {
        query_ids[0]: "First retrieval query?",
        query_ids[1]: "Second retrieval query?",
    }[representative_query_id]
    expected_relevant_row = {
        query_ids[0]: 0,
        query_ids[1]: 1,
    }[representative_query_id]

    online = [json.loads(line) for line in (output / QUERY_KEY_MAP_PATH).read_bytes().splitlines()]
    assert [row["query_row"] for row in online] == list(range(NESTED_ROWS_PER_FAMILY))
    assert [row["nested_index"] for row in online] == list(range(NESTED_ROWS_PER_FAMILY))
    assert {row["text"] for row in online} == {expected_text}
    assert {row["trial_key"] for row in online} == {
        _opaque(
            domain="trial",
            corpus=corpus,
            source_value=_nested_trial_source(
                representative_query_id,
                nested_index,
            ),
        )
        for nested_index in range(NESTED_ROWS_PER_FAMILY)
    }
    assert {row["family_key"] for row in online} == {
        _opaque(domain="family", corpus=corpus, source_value=component)
    }
    online_bytes = (output / QUERY_KEY_MAP_PATH).read_bytes()
    assert all(query_id.encode() not in online_bytes for query_id in query_ids)
    sealed = load_sealed_label_artifact(output / SEALED_LABEL_PATH)
    assert all(row.answer is None for row in sealed.labels)
    assert all(row.evidence_bundles == () for row in sealed.labels)
    assert {row.relevant_document_ids for row in sealed.labels} == {(expected_relevant_row,)}
    assert receipt.document_count == 2
    assert receipt.available_family_count == 1
    assert receipt.selected_family_count == 1
    assert receipt.nested_rows_per_family == NESTED_ROWS_PER_FAMILY
    assert receipt.query_count == NESTED_ROWS_PER_FAMILY
    assert (output / PROVENANCE_PATH).stat().st_size == 64

    row_order = hashlib.sha256()
    algorithm = DOCUMENT_ROW_ORDER_ALGORITHM.encode()
    row_order.update(len(algorithm).to_bytes(8, "big"))
    row_order.update(algorithm)
    for external_id in ("10", "2"):
        encoded = external_id.encode()
        row_order.update(len(encoded).to_bytes(8, "big"))
        row_order.update(encoded)
    assert receipt.ordered_document_row_sha256 == row_order.hexdigest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("family-key", "trial/family pairs"),
        ("label-payload", "identical sealed labels"),
        ("artifact-contract", "published artifact contract differs"),
    ],
)
def test_verifier_rejects_nested_label_and_receipt_contract_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    query_id = "bright-verifier-query"
    staged = tmp_path / "staged"
    plan, _ = _stage(
        staged,
        corpus="bright",
        documents=[{"id": "doc", "text": "Document.", "title": "Document"}],
        queries=[{"id": query_id, "text": "Question?"}],
        qrels=[{"document_id": "doc", "query_id": query_id, "relevance": 1}],
        evidence=None,
    )
    output = tmp_path / "custody"
    build_scalable_custody_package(
        staged,
        output,
        plan=plan,
        hmac_secret=_SECRET,
    )
    receipt_path = output / "receipt.json"
    receipt_value = json.loads(receipt_path.read_bytes())
    sealed_pin = next(row for row in receipt_value["artifacts"] if row["path"] == SEALED_LABEL_PATH)
    if mutation == "artifact-contract":
        sealed_pin["role"] = "forged-sealed-label-role"
    else:
        sealed_path = output / SEALED_LABEL_PATH
        sealed_value = json.loads(sealed_path.read_bytes())
        if mutation == "family-key":
            sealed_value["labels"][0]["family_key"] = "f" * 64
        else:
            sealed_value["labels"][0]["answer"] = "tampered nested answer"
        sealed_bytes = _canonical(sealed_value) + b"\n"
        sealed_path.write_bytes(sealed_bytes)
        sealed_pin["byte_count"] = len(sealed_bytes)
        sealed_pin["sha256"] = hashlib.sha256(sealed_bytes).hexdigest()
    receipt_path.write_bytes(_canonical(receipt_value) + b"\n")

    with pytest.raises(ScalableCustodyError, match=message):
        verify_scalable_custody_package(output)


def test_family_selection_is_deterministic_and_outcome_blind(tmp_path: Path) -> None:
    corpus = "bright"
    query_ids = tuple(f"bright-selection-query-{index}" for index in range(3))
    components = {
        query_id: hashlib.sha256(f"component:{query_id}".encode()).hexdigest()
        for query_id in query_ids
    }
    staged = tmp_path / "staged"
    plan, _ = _stage(
        staged,
        corpus=corpus,
        documents=[{"id": "doc", "text": "Document.", "title": "Document"}],
        queries=[
            {"id": query_id, "text": f"Question {index}?"}
            for index, query_id in enumerate(query_ids)
        ],
        qrels=[
            {"document_id": "doc", "query_id": query_id, "relevance": index + 1}
            for index, query_id in enumerate(query_ids)
        ],
        evidence=None,
        components=components,
        selected_families=2,
    )

    def rank(component: str) -> str:
        digest = hashlib.sha256()
        for value in (
            FAMILY_SELECTION_DOMAIN,
            FAMILY_SELECTION_ALGORITHM,
            corpus,
            "sealed",
            plan.selection_seed_sha256,
            component,
        ):
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    expected_components = set(
        sorted(components.values(), key=lambda value: (rank(value), value))[:2]
    )
    expected_trials = {
        _opaque(
            domain="trial",
            corpus=corpus,
            source_value=_nested_trial_source(query_id, nested_index),
        )
        for query_id, component in components.items()
        if component in expected_components
        for nested_index in range(NESTED_ROWS_PER_FAMILY)
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_receipt = build_scalable_custody_package(
        staged,
        first,
        plan=plan,
        hmac_secret=_SECRET,
    )
    second_receipt = build_scalable_custody_package(
        staged,
        second,
        plan=plan,
        hmac_secret=_SECRET,
    )
    first_rows = [
        json.loads(line) for line in (first / QUERY_KEY_MAP_PATH).read_bytes().splitlines()
    ]
    assert {row["trial_key"] for row in first_rows} == expected_trials
    assert [row["query_row"] for row in first_rows] == list(range(2 * NESTED_ROWS_PER_FAMILY))
    assert [row["nested_index"] for row in first_rows] == [
        nested_index for _ in range(2) for nested_index in range(NESTED_ROWS_PER_FAMILY)
    ]
    assert all(
        len({row["family_key"] for row in first_rows[offset : offset + NESTED_ROWS_PER_FAMILY]})
        == 1
        for offset in range(0, len(first_rows), NESTED_ROWS_PER_FAMILY)
    )
    assert first_receipt.available_family_count == 3
    assert first_receipt.selected_family_count == 2
    assert first_receipt.nested_rows_per_family == NESTED_ROWS_PER_FAMILY
    assert first_receipt.query_count == 2 * NESTED_ROWS_PER_FAMILY
    assert first_receipt == second_receipt
    for relative_path in (
        QUERY_KEY_MAP_PATH,
        PROVENANCE_PATH,
        SEALED_LABEL_PATH,
        "receipt.json",
    ):
        assert (first / relative_path).read_bytes() == (second / relative_path).read_bytes()


def test_label_byte_changes_cannot_change_family_or_representative_selection(
    tmp_path: Path,
) -> None:
    corpus = "bright"
    query_ids = tuple(
        f"bright-blind-family-{family}-candidate-{candidate}"
        for family in range(6)
        for candidate in range(4)
    )
    components = {
        query_id: hashlib.sha256(query_id.rsplit("-candidate-", 1)[0].encode()).hexdigest()
        for query_id in query_ids
    }
    documents = [
        {"id": "doc-a", "text": "Document A.", "title": "A"},
        {"id": "doc-b", "text": "Document B.", "title": "B"},
    ]
    queries = [{"id": query_id, "text": f"Question for {query_id}?"} for query_id in query_ids]
    staged_a = tmp_path / "staged-a"
    staged_b = tmp_path / "staged-b"
    plan_a, _ = _stage(
        staged_a,
        corpus=corpus,
        documents=documents,
        queries=queries,
        qrels=[
            {"document_id": "doc-a", "query_id": query_id, "relevance": 1} for query_id in query_ids
        ],
        evidence=None,
        components=components,
        selected_families=3,
    )
    plan_b, _ = _stage(
        staged_b,
        corpus=corpus,
        documents=documents,
        queries=queries,
        qrels=[
            {"document_id": "doc-b", "query_id": query_id, "relevance": 7} for query_id in query_ids
        ],
        evidence=None,
        components=components,
        selected_families=3,
    )
    assert plan_a.staged_inventory_sha256 != plan_b.staged_inventory_sha256

    output_a = tmp_path / "custody-a"
    output_b = tmp_path / "custody-b"
    receipt_a = build_scalable_custody_package(
        staged_a,
        output_a,
        plan=plan_a,
        hmac_secret=_SECRET,
    )
    receipt_b = build_scalable_custody_package(
        staged_b,
        output_b,
        plan=plan_b,
        hmac_secret=_SECRET,
    )
    assert (output_a / QUERY_KEY_MAP_PATH).read_bytes() == (
        output_b / QUERY_KEY_MAP_PATH
    ).read_bytes()
    assert (output_a / SEALED_LABEL_PATH).read_bytes() != (
        output_b / SEALED_LABEL_PATH
    ).read_bytes()
    assert receipt_a.available_family_count == receipt_b.available_family_count == 6
    assert receipt_a.selected_family_count == receipt_b.selected_family_count == 3
    assert receipt_a.query_count == receipt_b.query_count == 3 * NESTED_ROWS_PER_FAMILY


def test_verifier_rejects_interleaved_nested_family_blocks(tmp_path: Path) -> None:
    query_ids = ("bright-block-query-a", "bright-block-query-b")
    staged = tmp_path / "staged"
    plan, _ = _stage(
        staged,
        corpus="bright",
        documents=[{"id": "doc", "text": "Document.", "title": "Document"}],
        queries=[
            {"id": query_id, "text": f"Question {index}?"}
            for index, query_id in enumerate(query_ids)
        ],
        qrels=[
            {"document_id": "doc", "query_id": query_id, "relevance": 1} for query_id in query_ids
        ],
        evidence=None,
    )
    output = tmp_path / "custody"
    build_scalable_custody_package(
        staged,
        output,
        plan=plan,
        hmac_secret=_SECRET,
    )
    key_path = output / QUERY_KEY_MAP_PATH
    rows = [json.loads(line) for line in key_path.read_bytes().splitlines()]
    interleaved = [rows[index] for index in (0, 3, 1, 4, 2, 5)]
    for query_row, row in enumerate(interleaved):
        row["query_row"] = query_row
    key_bytes = b"".join(_canonical(row) + b"\n" for row in interleaved)
    key_path.write_bytes(key_bytes)

    receipt_path = output / "receipt.json"
    receipt_value = json.loads(receipt_path.read_bytes())
    key_pin = next(row for row in receipt_value["artifacts"] if row["path"] == QUERY_KEY_MAP_PATH)
    key_pin["byte_count"] = len(key_bytes)
    key_pin["sha256"] = hashlib.sha256(key_bytes).hexdigest()
    receipt_path.write_bytes(_canonical(receipt_value) + b"\n")

    with pytest.raises(ScalableCustodyError, match="contiguous block per family"):
        verify_scalable_custody_package(output)


def test_family_selection_fails_on_registered_count_underflow(tmp_path: Path) -> None:
    query_id = "bright-underflow-query"
    staged = tmp_path / "staged"
    plan, _ = _stage(
        staged,
        corpus="bright",
        documents=[{"id": "doc", "text": "Document.", "title": "Document"}],
        queries=[{"id": query_id, "text": "Question?"}],
        qrels=[{"document_id": "doc", "query_id": query_id, "relevance": 1}],
        evidence=None,
        selected_families=2,
        available_families=2,
    )
    output = tmp_path / "custody"
    with pytest.raises(ScalableCustodyError, match="underflows"):
        build_scalable_custody_package(
            staged,
            output,
            plan=plan,
            hmac_secret=_SECRET,
        )
    assert not output.exists()


def test_family_selection_fails_on_registered_available_count_drift(
    tmp_path: Path,
) -> None:
    query_ids = ("bright-count-query-a", "bright-count-query-b")
    staged = tmp_path / "staged"
    plan, _ = _stage(
        staged,
        corpus="bright",
        documents=[{"id": "doc", "text": "Document.", "title": "Document"}],
        queries=[
            {"id": query_id, "text": f"Question {index}?"}
            for index, query_id in enumerate(query_ids)
        ],
        qrels=[
            {"document_id": "doc", "query_id": query_id, "relevance": 1} for query_id in query_ids
        ],
        evidence=None,
        selected_families=1,
        available_families=3,
    )
    output = tmp_path / "custody"
    with pytest.raises(ScalableCustodyError, match="available family count differs"):
        build_scalable_custody_package(
            staged,
            output,
            plan=plan,
            hmac_secret=_SECRET,
        )
    assert not output.exists()


def test_runtime_query_store_must_match_custody_key_map_exactly(
    tmp_path: Path,
) -> None:
    staged, plan = _valid_scifact_stage(tmp_path)
    output = tmp_path / "custody"
    build_scalable_custody_package(
        staged,
        output,
        plan=plan,
        hmac_secret=_SECRET,
    )
    key_rows = [
        json.loads(line) for line in (output / QUERY_KEY_MAP_PATH).read_bytes().splitlines()
    ]
    runtime_rows = [
        {
            "corpus": key_row["corpus"],
            "family_key": key_row["family_key"],
            "query_row": key_row["query_row"],
            "schema_version": "fractal-query-trial-row-v1",
            "source": {
                "active_query_row_order_sha256": "1" * 64,
                "current_truth_query_row_order_sha256": "2" * 64,
                "embedding_query_row": 0,
                "source_file_sha256": "3" * 64,
                "source_path": "datasets/scifact/sealed/online/queries.jsonl",
                "source_query_id_sha256": "4" * 64,
                "source_record_sha256": "5" * 64,
                "source_row": 1,
            },
            "stage": key_row["stage"],
            "text": key_row["text"],
            "trial_key": key_row["trial_key"],
        }
        for key_row in key_rows
    ]
    runtime_path = tmp_path / "query-trials.jsonl"
    runtime_bytes = b"".join(_canonical(row) + b"\n" for row in runtime_rows)
    runtime_path.write_bytes(runtime_bytes)
    runtime_sha256 = hashlib.sha256(runtime_bytes).hexdigest()
    assert (
        verify_query_trial_key_parity(
            output,
            runtime_path,
            expected_runtime_sha256=runtime_sha256,
            expected_runtime_byte_count=len(runtime_bytes),
        )
        == runtime_sha256
    )

    runtime_rows[1]["family_key"] = "f" * 64
    mismatched = b"".join(_canonical(row) + b"\n" for row in runtime_rows)
    runtime_path.write_bytes(mismatched)
    with pytest.raises(ScalableCustodyError, match="key parity differs"):
        verify_query_trial_key_parity(
            output,
            runtime_path,
            expected_runtime_sha256=hashlib.sha256(mismatched).hexdigest(),
            expected_runtime_byte_count=len(mismatched),
        )


def _valid_scifact_stage(tmp_path: Path) -> tuple[Path, ScalableCustodyPlan]:
    staged = tmp_path / "staged"
    query_id = "scifact:sealed-SENTINEL"
    plan, _ = _stage(
        staged,
        corpus="scifact",
        documents=[{"id": "1", "text": "Sentence zero.\nSentence one.", "title": "Title"}],
        queries=[{"id": query_id, "text": "A claim."}],
        qrels=[{"document_id": "1", "query_id": query_id, "relevance": 1}],
        evidence=[
            {
                "answer": None,
                "evidence_bundles": [
                    {
                        "bundle_id": "rationale",
                        "locations": [{"document_id": "1", "locator": "sentence:0"}],
                    }
                ],
                "label_metadata": [["evidence_labels", "SUPPORT"]],
                "query_id": query_id,
            }
        ],
    )
    return staged, plan


@pytest.mark.parametrize("failure", ["allowlist", "mapping", "coverage", "ordering"])
def test_builder_fails_closed_on_input_drift(
    tmp_path: Path,
    failure: str,
) -> None:
    staged, plan = _valid_scifact_stage(tmp_path)
    if failure == "allowlist":
        plan = ScalableCustodyPlan(
            corpus=plan.corpus,
            staged_inventory_sha256=plan.staged_inventory_sha256,
            execution_artifact_sha256=plan.execution_artifact_sha256,
            hmac_key_id=plan.hmac_key_id,
            expected_document_count=plan.expected_document_count,
            available_families=plan.available_families,
            selected_families=plan.selected_families,
            selection_seed_sha256=plan.selection_seed_sha256,
            allowlisted_paths=plan.allowlisted_paths[:-1],
        )
    else:
        inventory = json.loads((staged / "inventory.json").read_bytes())
        if failure == "mapping":
            path = staged / "datasets/scifact/sealed/custody/qrels.jsonl"
            rows = [
                {
                    "document_id": "unknown-document",
                    "query_id": "scifact:sealed-SENTINEL",
                    "relevance": 1,
                }
            ]
        elif failure == "coverage":
            path = staged / "datasets/scifact/sealed/custody/evidence-bundles.jsonl"
            rows = []
        else:
            path = staged / "datasets/scifact/corpus.jsonl"
            rows = [
                {"id": "2", "text": "Second.", "title": "Second"},
                {"id": "1", "text": "First.", "title": "First"},
            ]
            inventory["counts"]["scifact"]["documents"] = 2
        relative = path.relative_to(staged).as_posix()
        byte_count, sha256, record_count = _write_jsonl(path, rows)
        for row in inventory["artifacts"]:
            if row["path"] == relative:
                row.update(
                    byte_count=byte_count,
                    sha256=sha256,
                    record_count=record_count,
                )
        inventory_bytes = _canonical(inventory) + b"\n"
        inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
        (staged / "inventory.json").write_bytes(inventory_bytes)
        (staged / "inventory.sha256").write_text(
            f"{inventory_sha256}  inventory.json\n",
            encoding="ascii",
        )
        plan = ScalableCustodyPlan(
            corpus=plan.corpus,
            staged_inventory_sha256=inventory_sha256,
            execution_artifact_sha256=plan.execution_artifact_sha256,
            hmac_key_id=plan.hmac_key_id,
            expected_document_count=2 if failure == "ordering" else 1,
            available_families=plan.available_families,
            selected_families=plan.selected_families,
            selection_seed_sha256=plan.selection_seed_sha256,
            allowlisted_paths=plan.allowlisted_paths,
        )
    output = tmp_path / "custody"
    with pytest.raises(ScalableCustodyError):
        build_scalable_custody_package(
            staged,
            output,
            plan=plan,
            hmac_secret=_SECRET,
        )
    assert not output.exists()


def test_builder_rejects_hardlinks_and_never_replaces_output(tmp_path: Path) -> None:
    staged, plan = _valid_scifact_stage(tmp_path)
    corpus = staged / "datasets/scifact/corpus.jsonl"
    os.link(corpus, tmp_path / "outside-hardlink.jsonl")
    output = tmp_path / "custody"
    with pytest.raises(ScalableCustodyError, match="unlinked regular file"):
        build_scalable_custody_package(
            staged,
            output,
            plan=plan,
            hmac_secret=_SECRET,
        )
    assert not output.exists()

    (tmp_path / "outside-hardlink.jsonl").unlink()
    build_scalable_custody_package(
        staged,
        output,
        plan=plan,
        hmac_secret=_SECRET,
    )
    original = (output / "receipt.json").read_bytes()
    with pytest.raises(ScalableCustodyError, match="already exists"):
        build_scalable_custody_package(
            staged,
            output,
            plan=plan,
            hmac_secret=_SECRET,
        )
    assert (output / "receipt.json").read_bytes() == original


def test_corpus_path_is_streamed_instead_of_read_as_a_control_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged, plan = _valid_scifact_stage(tmp_path)
    from fractal_ann_diagnostics import scalable_custody as custody

    original = custody._read_secure_file
    observed: list[str] = []

    def guarded_read(
        root_descriptor: int,
        relative_path: str,
        *,
        maximum_bytes: int,
        label: str,
    ) -> bytes:
        observed.append(relative_path)
        assert "corpus" not in relative_path
        return original(
            root_descriptor,
            relative_path,
            maximum_bytes=maximum_bytes,
            label=label,
        )

    monkeypatch.setattr(custody, "_read_secure_file", guarded_read)
    build_scalable_custody_package(
        staged,
        tmp_path / "custody",
        plan=plan,
        hmac_secret=_SECRET,
    )
    assert observed == ["inventory.json", "inventory.sha256", "receipt.json"]


def test_module_cli_build_verify_and_runtime_parity(tmp_path: Path) -> None:
    from fractal_ann_diagnostics.trial_runtime import (
        CanonicalQueryTrialRow,
        QueryTrialStoreReceipt,
        QueryVectorEpochBinding,
    )

    staged, plan = _valid_scifact_stage(tmp_path)
    inventory = json.loads((staged / "inventory.json").read_bytes())
    source_rows = {row["path"]: row for row in inventory["artifacts"]}
    config_path, key_path, config = _write_cli_config(
        tmp_path,
        staged=staged,
        plan=plan,
        source_rows=source_rows,
    )
    output = tmp_path / "custody-cli"
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "fractal_ann_diagnostics.scalable_custody",
            "build",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    build_result = json.loads(build.stdout)
    assert build_result["command"] == "build"
    assert build_result["selected_family_count"] == plan.selected_families
    receipt_bytes = (output / "receipt.json").read_bytes()
    assert _SECRET not in receipt_bytes
    assert str(key_path).encode() not in receipt_bytes
    assert config["hmac_key"]["sha256"].encode() not in receipt_bytes
    assert config["hmac_key"]["sha256"] not in build.stdout

    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "fractal_ann_diagnostics.scalable_custody",
            "verify",
            "--root",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["receipt_sha256"] == build_result["receipt_sha256"]

    key_rows = [
        json.loads(line) for line in (output / QUERY_KEY_MAP_PATH).read_bytes().splitlines()
    ]
    runtime_rows = tuple(
        CanonicalQueryTrialRow.from_dict(
            {
                "corpus": key_row["corpus"],
                "family_key": key_row["family_key"],
                "query_row": key_row["query_row"],
                "schema_version": "fractal-query-trial-row-v1",
                "source": {
                    "active_query_row_order_sha256": hashlib.sha256(b"query-order").hexdigest(),
                    "current_truth_query_row_order_sha256": hashlib.sha256(
                        b"query-order"
                    ).hexdigest(),
                    "embedding_query_row": 0,
                    "source_file_sha256": hashlib.sha256(b"query-source").hexdigest(),
                    "source_path": "datasets/scifact/sealed/online/queries.jsonl",
                    "source_query_id_sha256": hashlib.sha256(b"source-query-id").hexdigest(),
                    "source_record_sha256": hashlib.sha256(b"source-record").hexdigest(),
                    "source_row": 1,
                },
                "stage": key_row["stage"],
                "text": key_row["text"],
                "trial_key": key_row["trial_key"],
            }
        )
        for key_row in key_rows
    )
    runtime_bytes = b"".join(row.canonical_line() for row in runtime_rows)
    active_epoch = QueryVectorEpochBinding(
        role="active-migration",
        file_sha256=hashlib.sha256(b"active-vectors").hexdigest(),
        row_order_sha256=runtime_rows[0].source.active_query_row_order_sha256,
        model_tree_sha256=hashlib.sha256(b"old-model").hexdigest(),
        model_revision="old-model-revision",
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        dtype="float32",
        shape=(1, 2),
    )
    current_epoch = QueryVectorEpochBinding(
        role="current-exact-truth",
        file_sha256=hashlib.sha256(b"current-vectors").hexdigest(),
        row_order_sha256=runtime_rows[0].source.current_truth_query_row_order_sha256,
        model_tree_sha256=hashlib.sha256(b"current-model").hexdigest(),
        model_revision="current-model-revision",
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        dtype="float32",
        shape=(1, 2),
    )
    runtime_receipt = QueryTrialStoreReceipt(
        hmac_key_id=plan.hmac_key_id,
        corpus=plan.corpus,
        stage=plan.stage,
        staged_inventory_sha256=plan.staged_inventory_sha256,
        source_inventory_sha256=hashlib.sha256(b"source-inventory").hexdigest(),
        assignment_store_sha256=source_rows["assignments.jsonl"]["sha256"],
        query_partition_audit_sha256=hashlib.sha256(b"query-partition-audit").hexdigest(),
        selection_seed_sha256=plan.selection_seed_sha256,
        available_family_count=plan.available_families,
        selected_family_count=plan.selected_families,
        nested_rows_per_family=plan.nested_rows_per_family,
        embedding_store_receipt_sha256=hashlib.sha256(b"embedding-receipt").hexdigest(),
        active_query_epoch=active_epoch,
        current_truth_query_epoch=current_epoch,
        query_trial_store_sha256=hashlib.sha256(runtime_bytes).hexdigest(),
        query_trial_store_byte_count=len(runtime_bytes),
        record_count=NESTED_ROWS_PER_FAMILY,
        opaque_trials=tuple(row.opaque_row for row in runtime_rows),
    )
    runtime_root = tmp_path / "runtime-query-trials"
    runtime_root.mkdir(mode=0o700)
    (runtime_root / "query-trials.jsonl").write_bytes(runtime_bytes)
    (runtime_root / "query-trial-receipt.json").write_bytes(runtime_receipt.canonical_file_bytes())
    parity = subprocess.run(
        [
            sys.executable,
            "-m",
            "fractal_ann_diagnostics.scalable_custody",
            "verify-query-parity",
            "--custody-root",
            str(output),
            "--trial-root",
            str(runtime_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert parity.returncode == 0, parity.stderr
    assert json.loads(parity.stdout) == {
        "command": "verify-query-parity",
        "corpus": "scifact",
        "query_count": NESTED_ROWS_PER_FAMILY,
        "runtime_query_trial_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown-field",
        "relative-path",
        "placeholder-id",
        "placeholder-digest",
        "nested-count",
        "representative-algorithm",
    ],
)
def test_config_rejects_unknown_relative_and_placeholder_values(
    tmp_path: Path,
    mutation: str,
) -> None:
    staged, plan = _valid_scifact_stage(tmp_path)
    inventory = json.loads((staged / "inventory.json").read_bytes())
    source_rows = {row["path"]: row for row in inventory["artifacts"]}
    config_path, _, config = _write_cli_config(
        tmp_path,
        staged=staged,
        plan=plan,
        source_rows=source_rows,
    )
    if mutation == "unknown-field":
        config["unregistered"] = "value"
    elif mutation == "relative-path":
        config["staged_root"] = "relative/staged"
    elif mutation == "placeholder-id":
        config["hmac_key_id"] = "TBD"
    elif mutation == "placeholder-digest":
        config["selection_seed_sha256"] = "0" * 64
    elif mutation == "nested-count":
        config["nested_rows_per_family"] = NESTED_ROWS_PER_FAMILY + 1
    else:
        config["representative_selection_algorithm"] = "sha256-rank-v2"
    config_path.write_bytes(_canonical(config) + b"\n")
    config_path.chmod(0o600)
    with pytest.raises(ScalableCustodyError):
        load_scalable_custody_config(config_path)


def test_config_rejects_duplicate_keys_and_noncanonical_bytes(tmp_path: Path) -> None:
    staged, plan = _valid_scifact_stage(tmp_path)
    inventory = json.loads((staged / "inventory.json").read_bytes())
    source_rows = {row["path"]: row for row in inventory["artifacts"]}
    config_path, _, config = _write_cli_config(
        tmp_path,
        staged=staged,
        plan=plan,
        source_rows=source_rows,
    )
    duplicate = (_canonical(config) + b"\n").replace(
        b'"corpus":"scifact"',
        b'"corpus":"scifact","corpus":"scifact"',
        1,
    )
    config_path.write_bytes(duplicate)
    config_path.chmod(0o600)
    with pytest.raises(ScalableCustodyError, match="repeats key"):
        load_scalable_custody_config(config_path)

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    config_path.chmod(0o600)
    with pytest.raises(ScalableCustodyError, match="canonical JSON"):
        load_scalable_custody_config(config_path)


def test_config_and_key_symlinks_are_rejected(tmp_path: Path) -> None:
    staged, plan = _valid_scifact_stage(tmp_path)
    inventory = json.loads((staged / "inventory.json").read_bytes())
    source_rows = {row["path"]: row for row in inventory["artifacts"]}
    config_path, _, _ = _write_cli_config(
        tmp_path,
        staged=staged,
        plan=plan,
        source_rows=source_rows,
    )
    config_link = tmp_path / "config-link.json"
    config_link.symlink_to(config_path)
    with pytest.raises(ScalableCustodyError):
        load_scalable_custody_config(config_link)

    key_target = tmp_path / "private-key-target.bin"
    key_target.write_bytes(_SECRET)
    key_target.chmod(0o600)
    key_link = tmp_path / "key-link.bin"
    key_link.symlink_to(key_target)
    linked_config, _, _ = _write_cli_config(
        tmp_path,
        staged=staged,
        plan=plan,
        source_rows=source_rows,
        key_path=key_link,
    )
    with pytest.raises(ScalableCustodyError):
        build_scalable_custody_from_config(linked_config, tmp_path / "output")


def test_config_source_pins_must_match_inventory_exactly(tmp_path: Path) -> None:
    staged, plan = _valid_scifact_stage(tmp_path)
    inventory = json.loads((staged / "inventory.json").read_bytes())
    source_rows = {row["path"]: row for row in inventory["artifacts"]}
    config_path, _, config = _write_cli_config(
        tmp_path,
        staged=staged,
        plan=plan,
        source_rows=source_rows,
    )
    config["source_artifacts"][0]["sha256"] = hashlib.sha256(b"wrong-source-pin").hexdigest()
    config_path.write_bytes(_canonical(config) + b"\n")
    config_path.chmod(0o600)
    with pytest.raises(ScalableCustodyError, match="source_artifacts differ"):
        build_scalable_custody_from_config(config_path, tmp_path / "output")
