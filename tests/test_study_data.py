from __future__ import annotations

import bz2
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from fractal_ann_diagnostics.study_data import (
    BRIGHT_DOMAINS,
    CONFIG_SCHEMA,
    ONLINE_PROJECTION_RECEIPT_FILENAME,
    StudyDataError,
    compute_shard_tree_digest,
    load_staging_config,
    main,
    project_online_staging,
    stage_study_data,
    verify_online_staging_projection,
    verify_staged_data,
)


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(_canonical(payload) + b"\n")


@dataclass
class _Fixture:
    config_path: Path
    payload: dict[str, object]
    paths: dict[str, Path]
    pins: dict[str, dict[str, str]]

    def write_config(self) -> None:
        _write_json(self.config_path, self.payload)

    def repin(self, source_id: str) -> None:
        self.pins[source_id]["sha256"] = hashlib.sha256(
            self.paths[source_id].read_bytes()
        ).hexdigest()
        self.write_config()


def _study_fixture(tmp_path: Path) -> _Fixture:
    paths: dict[str, Path] = {}
    pins: dict[str, dict[str, str]] = {}

    def pin(source_id: str, rows: object, *, json_array: bool = False) -> dict[str, str]:
        path = tmp_path / (source_id.replace("/", "-") + (".json" if json_array else ".jsonl"))
        if json_array:
            _write_json(path, rows)
        else:
            assert isinstance(rows, list)
            _write_jsonl(path, rows)
        paths[source_id] = path
        value = {
            "path": path.name,
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        pins[source_id] = value
        return value

    def pin_bytes(source_id: str, encoded: bytes, *, suffix: str) -> dict[str, str]:
        path = tmp_path / (source_id.replace("/", "-") + suffix)
        path.write_bytes(encoded)
        paths[source_id] = path
        value = {
            "path": path.name,
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
        pins[source_id] = value
        return value

    scifact_documents: list[dict[str, object]] = []
    scifact_train: list[dict[str, object]] = []
    for index in range(5):
        document_id = f"s-train-{index}"
        scifact_documents.append(
            {
                "abstract": [f"SciFact sentence {index}."],
                "doc_id": document_id,
                "title": f"S {index}",
            }
        )
        scifact_train.append(
            {
                "claim": f"SciFact train proposition {index}",
                "evidence": {document_id: [{"label": "SUPPORT", "sentences": [0]}]},
                "id": f"train-{index}",
            }
        )
    scifact_documents.append(
        {"abstract": ["Sealed SciFact sentence."], "doc_id": "s-dev", "title": "S dev"}
    )
    scifact_dev = [
        {
            "claim": "SciFact sealed proposition",
            "evidence": {"s-dev": [{"label": "CONTRADICT", "sentences": [0]}]},
            "id": "dev-0",
        }
    ]
    scifact = {
        "corpus": pin("scifact/corpus", scifact_documents),
        "train_claims": pin("scifact/train_claims", scifact_train),
        "dev_claims": pin("scifact/dev_claims", scifact_dev),
    }

    t2: dict[str, dict[str, str]] = {}
    for split in ("train", "dev", "test"):
        row = {
            "context": f"T2 context for {split}.",
            "context_id": f"t2-context-{split}",
            "file_name": f"{split}.pdf",
            "id": f"finqa-{split}",
            "question": f"T2 {split} financial question?",
            "split": split,
            "subset": "FinQA",
        }
        t2[split] = pin(f"t2_finqa/{split}", [row], json_array=True)

    miracl_documents: list[dict[str, object]] = []
    miracl_train_queries: list[dict[str, object]] = []
    miracl_train_qrels: list[dict[str, object]] = []
    for index in range(5):
        document_id = f"sw-doc-{index}"
        query_id = f"sw-train-{index}"
        miracl_documents.append(
            {"id": document_id, "text": f"Maandishi ya Kiswahili {index}.", "title": f"SW {index}"}
        )
        miracl_train_queries.append({"id": query_id, "text": f"Swali Kiswahili mafunzo {index}?"})
        miracl_train_qrels.append(
            {"document_id": document_id, "query_id": query_id, "relevance": 1}
        )
    miracl_documents.append(
        {"id": "sw-doc-dev", "text": "Maandishi yaliyofungwa.", "title": "SW dev"}
    )
    dev_query = {"id": "sw-dev", "text": "Swali la tathmini lililofungwa?"}
    dev_qrel = {"document_id": "sw-doc-dev", "query_id": "sw-dev", "relevance": 1}
    document_bytes = b"".join(_canonical(row) + b"\n" for row in miracl_documents)
    train_topic_bytes = "".join(
        f"{row['id']}\t{row['text']}\n" for row in miracl_train_queries
    ).encode("utf-8")
    train_qrel_bytes = "".join(
        f"{row['query_id']}\tQ0\t{row['document_id']}\t{row['relevance']}\n"
        for row in miracl_train_qrels
    ).encode("utf-8")
    miracl = {
        "documents": pin_bytes(
            "miracl_sw/documents", gzip.compress(document_bytes), suffix=".jsonl.gz"
        ),
        "train_queries": pin_bytes("miracl_sw/train_queries", train_topic_bytes, suffix=".tsv"),
        "train_qrels": pin_bytes("miracl_sw/train_qrels", train_qrel_bytes, suffix=".tsv"),
        "dev_queries": pin_bytes(
            "miracl_sw/dev_queries",
            f"{dev_query['id']}\t{dev_query['text']}\n".encode("utf-8"),
            suffix=".tsv",
        ),
        "dev_qrels": pin_bytes(
            "miracl_sw/dev_qrels",
            (
                f"{dev_qrel['query_id']}\tQ0\t{dev_qrel['document_id']}\t{dev_qrel['relevance']}\n"
            ).encode("utf-8"),
            suffix=".tsv",
        ),
    }

    bright_domains: dict[str, object] = {}
    for domain in BRIGHT_DOMAINS:
        documents: list[dict[str, object]] = []
        examples: list[dict[str, object]] = []
        if domain in {"aops", "theoremqa_questions"}:
            documents.append({"content": "Shared BRIGHT pool row.", "id": "shared-doc"})
        for index in range(5):
            raw_document_id = f"{domain}-doc-{index}"
            documents.append(
                {
                    "content": f"BRIGHT {domain} document {index}.",
                    "id": raw_document_id,
                }
            )
            examples.append(
                {
                    "excluded_ids": ["N/A"],
                    "gold_answer": f"Answer {index}",
                    "gold_ids": [raw_document_id],
                    "gold_ids_long": ["N/A"],
                    "id": f"query-{index}",
                    "query": f"BRIGHT {domain} problem {index}?",
                    "reasoning": f"Reasoning {index}",
                }
            )
        prefix = f"bright/{domain}"
        bright_domains[domain] = {
            "documents": pin(f"{prefix}/documents", documents),
            "examples": pin(f"{prefix}/examples", examples),
        }

    hotpot_rows = [
        {"id": "hp-2", "text": ["Second article sentence."], "title": "Second Article"},
        {"id": "hp-1", "text": ["First article sentence."], "title": "First Article"},
        {"id": "hp-3", "text": ["Third article sentence."], "title": "Third Article"},
    ]
    hotpot_shards = tmp_path / "hotpot-shards"
    (hotpot_shards / "AA").mkdir(parents=True)
    for index, row in enumerate(hotpot_rows):
        (hotpot_shards / "AA" / f"wiki_{index:02d}.bz2").write_bytes(
            bz2.compress(_canonical(row) + b"\n")
        )
    shard_sha256, shard_count, _ = compute_shard_tree_digest(hotpot_shards)
    archive_path = tmp_path / "hotpot-official-archive.tar.bz2"
    archive_path.write_bytes(b"pinned official archive fixture")
    archive_pin = {
        "path": archive_path.name,
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }
    paths["hotpotqa_fullwiki/corpus_archive"] = archive_path
    pins["hotpotqa_fullwiki/corpus_archive"] = archive_pin
    hotpot_questions = [
        {
            "_id": "hp-question",
            "answer": "first",
            "question": "Which Hotpot article is first?",
            "supporting_facts": [["First Article", 0]],
        }
    ]
    hotpot_train_questions = [
        {
            "_id": "hp-train-second",
            "answer": "second",
            "question": "Which Hotpot article is second?",
            "supporting_facts": [["Second Article", 0]],
        },
        {
            "_id": "hp-train-third",
            "answer": "third",
            "question": "Which Hotpot article is third?",
            "supporting_facts": [["Third Article", 0]],
        },
    ]
    hotpot = {
        "corpus_archive": archive_pin,
        "corpus_shards": {
            "file_count": shard_count,
            "path": hotpot_shards.name,
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "sha256": shard_sha256,
        },
        "train_questions": [
            pin(
                "hotpotqa_fullwiki/train_questions/00000",
                [hotpot_train_questions[0]],
                json_array=True,
            ),
            pin(
                "hotpotqa_fullwiki/train_questions/00001",
                [hotpot_train_questions[1]],
                json_array=True,
            ),
        ],
        "dev_questions": pin("hotpotqa_fullwiki/dev_questions", hotpot_questions, json_array=True),
        "corpus_scope": {
            "expected_document_count": 3,
            "name": "fullwiki",
            "sampling": "none",
        },
    }

    payload: dict[str, object] = {
        "assignment_seed": "ab" * 32,
        "datasets": {
            "bright": {
                "document_id_collision_policy": "error",
                "domain_order": list(BRIGHT_DOMAINS),
                "domains": bright_domains,
            },
            "hotpotqa_fullwiki": hotpot,
            "miracl_sw": miracl,
            "scifact": scifact,
            "t2_finqa": t2,
        },
        "withhold_sealed_labels_from_online_process": True,
        "schema_version": CONFIG_SCHEMA,
    }
    fixture = _Fixture(
        config_path=tmp_path / "config.json",
        payload=payload,
        paths=paths,
        pins=pins,
    )
    fixture.write_config()
    return fixture


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stages_all_fixed_sources_and_withholds_sealed_qrels(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    output = tmp_path / "staged"

    receipt = stage_study_data(fixture.config_path, output)
    verified = verify_staged_data(output)

    assert verified.inventory_sha256 == receipt.inventory_sha256
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    assert len(inventory["sources"]) == 40
    assert inventory["bright_domains"] == list(BRIGHT_DOMAINS)
    assert inventory["bright_document_identity"]["duplicate_source_rows"] == 1
    assert inventory["withhold_sealed_labels_from_online_process"] is True
    paths = {row["path"] for row in inventory["artifacts"]}
    assert "partition-exclusions.jsonl" in paths
    assert _jsonl(output / "partition-exclusions.jsonl") == []
    for dataset in (
        "scifact",
        "hotpotqa-fullwiki",
        "t2-ragbench",
        "bright",
        "miracl-transfer",
    ):
        assert f"datasets/{dataset}/sealed/online/queries.jsonl" in paths
        assert f"datasets/{dataset}/sealed/custody/qrels.jsonl" in paths
        assert f"datasets/{dataset}/sealed/online/qrels.jsonl" not in paths
    for dataset in ("scifact", "hotpotqa-fullwiki", "t2-ragbench"):
        assert f"datasets/{dataset}/sealed/custody/evidence-bundles.jsonl" in paths
        assert f"datasets/{dataset}/sealed/online/evidence-bundles.jsonl" not in paths
    for dataset in ("bright", "miracl-transfer"):
        assert f"datasets/{dataset}/sealed/custody/evidence-bundles.jsonl" not in paths
    assert _jsonl(output / "datasets/hotpotqa-fullwiki/corpus/part-00000.jsonl") == [
        {"id": "hp-1", "text": "First article sentence.", "title": "First Article"},
        {"id": "hp-2", "text": "Second article sentence.", "title": "Second Article"},
        {"id": "hp-3", "text": "Third article sentence.", "title": "Third Article"},
    ]
    assignments = _jsonl(output / "assignments.jsonl")
    bright_stages = {row["stage"] for row in assignments if row["dataset"] == "bright"}
    assert bright_stages == {"fit", "calibration", "sealed"}
    hotpot_assignments = [row for row in assignments if row["dataset"] == "hotpotqa-fullwiki"]
    assert {row["stage"] for row in hotpot_assignments if row["source_split"] == "dev"} == {
        "sealed"
    }
    assert {row["stage"] for row in hotpot_assignments if row["source_split"] == "train"} == {
        "fit",
        "calibration",
    }
    t2_online = _jsonl(output / "datasets/t2-ragbench/sealed/online/queries.jsonl")
    assert all("answer" not in row for row in t2_online)
    assert _jsonl(output / "datasets/t2-ragbench/sealed/custody/evidence-bundles.jsonl") == [
        {
            "answer": None,
            "evidence_bundles": [
                {
                    "bundle_id": "source-context",
                    "locations": [{"document_id": "t2-context-test", "locator": "document"}],
                }
            ],
            "label_metadata": [["split", "test"], ["subset", "FinQA"]],
            "query_id": "t2-ragbench:finqa-test",
        }
    ]
    assert _jsonl(output / "datasets/hotpotqa-fullwiki/sealed/custody/evidence-bundles.jsonl") == [
        {
            "answer": "first",
            "evidence_bundles": [
                {
                    "bundle_id": "supporting-facts",
                    "locations": [{"document_id": "hp-1", "locator": "sentence:0"}],
                }
            ],
            "label_metadata": [],
            "query_id": "hotpotqa:hp-question",
        }
    ]
    assert inventory["counts"]["hotpotqa-fullwiki"]["evidence_locations"] == 3
    assert inventory["counts"]["t2-ragbench"]["evidence_label_rows"] == 3


def test_scifact_evidence_bearing_rule_emits_exact_exclusion_receipt(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    train_path = fixture.paths["scifact/train_claims"]
    train_rows = _jsonl(train_path)
    train_rows.append(
        {
            "claim": "A claim without an evidence-bearing retrieval target",
            "evidence": {},
            "id": "train-excluded",
        }
    )
    _write_jsonl(train_path, train_rows)
    fixture.repin("scifact/train_claims")

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    assert _jsonl(output / "exclusions.jsonl") == [
        {
            "dataset": "scifact",
            "query_id": "scifact:train-excluded",
            "reason": "no-nonempty-evidence-rationale-list",
            "rule_id": "scifact-evidence-bearing-v1",
            "source_split": "train",
        }
    ]
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    assert inventory["counts"]["scifact"]["excluded_queries"] == 1
    assert inventory["counts"]["scifact"]["excluded_train_queries"] == 1
    assignments = _jsonl(output / "assignments.jsonl")
    assert "scifact:train-excluded" not in {row["query_id"] for row in assignments}


def test_t2_empty_question_is_excluded_by_registered_rule(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    test_path = fixture.paths["t2_finqa/test"]
    test_rows = json.loads(test_path.read_text(encoding="utf-8"))
    test_rows.append(
        {
            "context": "An upstream row with no usable query.",
            "context_id": "t2-context-excluded",
            "file_name": "excluded.pdf",
            "id": "finqa-test-excluded",
            "question": "",
            "split": "test",
            "subset": "FinQA",
        }
    )
    _write_json(test_path, test_rows)
    fixture.repin("t2_finqa/test")

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    exclusion = _jsonl(output / "exclusions.jsonl")
    assert exclusion == [
        {
            "dataset": "t2-ragbench",
            "query_id": "t2-ragbench:finqa-test-excluded",
            "reason": "empty-question",
            "rule_id": "t2-finqa-nonempty-question-v1",
            "source_split": "test",
        }
    ]
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    assert inventory["counts"]["t2-ragbench"]["excluded_test_queries"] == 1


def test_t2_fixed_split_couples_duplicate_queries_and_positive_contexts(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    train_path = fixture.paths["t2_finqa/train"]
    train_rows = json.loads(train_path.read_text(encoding="utf-8"))
    duplicate = dict(train_rows[0])
    duplicate["id"] = "finqa-train-duplicate"
    duplicate["program_answer"] = "a distinct upstream answer"
    train_rows.append(duplicate)
    _write_json(train_path, train_rows)
    fixture.repin("t2_finqa/train")

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    assignments = {row["query_id"]: row for row in _jsonl(output / "assignments.jsonl")}
    original = assignments["t2-ragbench:finqa-train"]
    repeated = assignments["t2-ragbench:finqa-train-duplicate"]
    assert original["partition_component_sha256"] == repeated["partition_component_sha256"]
    assert original["assignment_key_sha256"] == repeated["assignment_key_sha256"]
    assert original["stage"] == repeated["stage"] == "fit"


def test_bright_duplicate_domain_views_share_one_assignment_component(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    first_path = fixture.paths["bright/theoremqa_questions/examples"]
    second_path = fixture.paths["bright/theoremqa_theorems/examples"]
    first_rows = _jsonl(first_path)
    second_rows = _jsonl(second_path)
    second_rows[0]["query"] = first_rows[0]["query"]
    _write_jsonl(second_path, second_rows)
    fixture.repin("bright/theoremqa_theorems/examples")

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    assignments = {row["query_id"]: row for row in _jsonl(output / "assignments.jsonl")}
    left = assignments["bright:theoremqa_questions:query-0"]
    right = assignments["bright:theoremqa_theorems:query-0"]
    assert left["partition_component_sha256"] == right["partition_component_sha256"]
    assert left["assignment_key_sha256"] == right["assignment_key_sha256"]
    assert left["stage"] == right["stage"]


def test_bright_repeated_binary_gold_id_is_deduplicated_with_count(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    examples_path = fixture.paths["bright/theoremqa_questions/examples"]
    rows = _jsonl(examples_path)
    rows[0]["gold_ids"] = [
        "theoremqa_questions-doc-0",
        "theoremqa_questions-doc-0",
    ]
    _write_jsonl(examples_path, rows)
    fixture.repin("bright/theoremqa_questions/examples")

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    assert inventory["counts"]["bright"]["duplicate_positive_judgments"] == 1
    qrel_paths = [path for path in output.glob("datasets/bright/*/qrels.jsonl") if path.is_file()]
    judgments = [row for path in qrel_paths for row in _jsonl(path)]
    matching = [row for row in judgments if row["query_id"] == "bright:theoremqa_questions:query-0"]
    assert len(matching) == 1


def test_scifact_shared_positive_content_couples_upstream_splits(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    dev_path = fixture.paths["scifact/dev_claims"]
    dev_rows = _jsonl(dev_path)
    dev_rows[0]["evidence"] = {"s-train-0": [{"label": "CONTRADICT", "sentences": [0]}]}
    _write_jsonl(dev_path, dev_rows)
    fixture.repin("scifact/dev_claims")

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    assignments = {row["query_id"]: row for row in _jsonl(output / "assignments.jsonl")}
    train = assignments["scifact:train-0"]
    development = assignments["scifact:dev-0"]
    assert train["partition_component_sha256"] == development["partition_component_sha256"]
    assert train["stage"] == development["stage"]


def test_miracl_content_aliases_are_preserved_and_couple_query_components(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    documents_path = fixture.paths["miracl_sw/documents"]
    documents = [
        json.loads(line)
        for line in gzip.decompress(documents_path.read_bytes()).decode("utf-8").splitlines()
    ]
    alias = dict(documents[0])
    alias["id"] = "sw-doc-alias"
    documents.append(alias)
    document_bytes = b"".join(_canonical(row) + b"\n" for row in documents)
    documents_path.write_bytes(gzip.compress(document_bytes, mtime=0))
    fixture.repin("miracl_sw/documents")

    dev_qrels_path = fixture.paths["miracl_sw/dev_qrels"]
    dev_qrels_path.write_text("sw-dev\tQ0\tsw-doc-alias\t1\n", encoding="utf-8")
    fixture.repin("miracl_sw/dev_qrels")

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    assert inventory["counts"]["miracl-transfer"]["duplicate_document_content_aliases"] == 1
    assignments = {row["query_id"]: row for row in _jsonl(output / "assignments.jsonl")}
    train = assignments["miracl-sw:sw-train-0"]
    development = assignments["miracl-sw:sw-dev"]
    assert train["partition_component_sha256"] == development["partition_component_sha256"]
    assert train["stage"] == development["stage"]


def test_hotpot_title_only_row_is_preserved_with_registered_text_fallback(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    shard_root = tmp_path / "hotpot-shards"
    shard_path = shard_root / "AA/wiki_01.bz2"
    title_only = {"id": "hp-1", "text": [""], "title": "First Article"}
    shard_path.write_bytes(bz2.compress(_canonical(title_only) + b"\n"))
    shard_sha256, shard_count, _ = compute_shard_tree_digest(shard_root)
    shard_config = fixture.payload["datasets"]["hotpotqa_fullwiki"][  # type: ignore[index]
        "corpus_shards"
    ]
    shard_config["file_count"] = shard_count  # type: ignore[index]
    shard_config["sha256"] = shard_sha256  # type: ignore[index]
    fixture.write_config()

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    corpus = _jsonl(output / "datasets/hotpotqa-fullwiki/corpus/part-00000.jsonl")
    assert corpus[0] == {
        "id": "hp-1",
        "text": "First Article",
        "title": "First Article",
    }
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    assert inventory["counts"]["hotpotqa-fullwiki"]["empty_document_text_fallbacks"] == 1


def test_hotpot_out_of_range_supporting_fact_emits_exact_exclusion(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    source_id = "hotpotqa_fullwiki/dev_questions"
    rows = json.loads(fixture.paths[source_id].read_text(encoding="utf-8"))
    rows.append(
        {
            "_id": "hp-invalid-support",
            "answer": "unscorable",
            "question": "Which sentence is structurally invalid?",
            "supporting_facts": [["First Article", 99]],
        }
    )
    _write_json(fixture.paths[source_id], rows)
    fixture.repin(source_id)

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)
    verify_staged_data(output)

    assert _jsonl(output / "exclusions.jsonl") == [
        {
            "dataset": "hotpotqa-fullwiki",
            "query_id": "hotpotqa:hp-invalid-support",
            "reason": "out-of-range-supporting-sentence",
            "rule_id": "hotpotqa-supporting-fact-range-v1",
            "source_split": "dev",
        }
    ]
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    counts = inventory["counts"]["hotpotqa-fullwiki"]
    assert counts["sealed_queries"] == 1
    assert counts["excluded_queries"] == 1
    assert counts["excluded_dev_queries"] == 1


def test_hotpot_train_exclusion_retains_source_split_provenance(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    source_id = "hotpotqa_fullwiki/train_questions/00000"
    rows = json.loads(fixture.paths[source_id].read_text(encoding="utf-8"))
    rows.append(
        {
            "_id": "hp-train-invalid-support",
            "answer": "unscorable",
            "question": "Which training sentence is structurally invalid?",
            "supporting_facts": [["Second Article", 99]],
        }
    )
    _write_json(fixture.paths[source_id], rows)
    fixture.repin(source_id)

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    assert _jsonl(output / "exclusions.jsonl") == [
        {
            "dataset": "hotpotqa-fullwiki",
            "query_id": "hotpotqa:hp-train-invalid-support",
            "reason": "out-of-range-supporting-sentence",
            "rule_id": "hotpotqa-supporting-fact-range-v1",
            "source_split": "train",
        }
    ]
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    counts = inventory["counts"]["hotpotqa-fullwiki"]
    assert counts["fit_queries"] == 1
    assert counts["calibration_queries"] == 1
    assert counts["sealed_queries"] == 1
    assert counts["excluded_train_queries"] == 1


def test_hotpot_train_source_is_required_by_closed_config(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    hotpot = fixture.payload["datasets"]["hotpotqa_fullwiki"]  # type: ignore[index]
    del hotpot["train_questions"]  # type: ignore[index]
    fixture.write_config()

    with pytest.raises(StudyDataError, match="train_questions"):
        load_staging_config(fixture.config_path)


@pytest.mark.parametrize("replacement", [[], [None], [None, None, None]])
def test_hotpot_train_release_requires_exactly_two_shard_pins(
    tmp_path: Path, replacement: list[object]
) -> None:
    fixture = _study_fixture(tmp_path)
    hotpot = fixture.payload["datasets"]["hotpotqa_fullwiki"]  # type: ignore[index]
    existing = hotpot["train_questions"]  # type: ignore[index]
    hotpot["train_questions"] = [  # type: ignore[index]
        existing[index % 2] if value is None else value  # type: ignore[index]
        for index, value in enumerate(replacement)
    ]
    fixture.write_config()

    with pytest.raises(StudyDataError, match="exactly two shard pins"):
        load_staging_config(fixture.config_path)


def test_hotpot_train_shard_digest_is_enforced(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    source_id = "hotpotqa_fullwiki/train_questions/00001"
    fixture.paths[source_id].write_bytes(b"tampered\n")

    with pytest.raises(StudyDataError, match="SHA-256 mismatch"):
        stage_study_data(fixture.config_path, tmp_path / "staged")


def test_hotpot_train_question_ids_must_be_unique_across_shards(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    first_id = "hotpotqa_fullwiki/train_questions/00000"
    second_id = "hotpotqa_fullwiki/train_questions/00001"
    first = json.loads(fixture.paths[first_id].read_text(encoding="utf-8"))
    second = json.loads(fixture.paths[second_id].read_text(encoding="utf-8"))
    second[0]["_id"] = first[0]["_id"]
    _write_json(fixture.paths[second_id], second)
    fixture.repin(second_id)

    with pytest.raises(StudyDataError, match="occurs in both"):
        stage_study_data(fixture.config_path, tmp_path / "staged")


def test_hotpot_cross_split_positive_component_is_excluded_before_allocation(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    source_id = "hotpotqa_fullwiki/train_questions/00000"
    rows = json.loads(fixture.paths[source_id].read_text(encoding="utf-8"))
    rows[0]["supporting_facts"] = [["First Article", 0]]
    rows.append(
        {
            "_id": "hp-train-safe-second",
            "answer": "second",
            "question": "Which independent training item names the second article?",
            "supporting_facts": [["Second Article", 0]],
        }
    )
    _write_json(fixture.paths[source_id], rows)
    fixture.repin(source_id)

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    exclusions = _jsonl(output / "partition-exclusions.jsonl")
    assert {row["query_id"] for row in exclusions} == {
        "hotpotqa:hp-question",
        "hotpotqa:hp-train-second",
    }
    assert {row["source_split"] for row in exclusions} == {"dev", "train"}
    assert len({row["partition_component_sha256"] for row in exclusions}) == 1
    assignments = {
        row["query_id"]: row
        for row in _jsonl(output / "assignments.jsonl")
        if row["dataset"] == "hotpotqa-fullwiki"
    }
    assert set(assignments) == {
        "hotpotqa:hp-train-safe-second",
        "hotpotqa:hp-train-third",
    }
    assert {row["stage"] for row in assignments.values()} == {"fit", "calibration"}


def test_staging_is_byte_deterministic(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    first = stage_study_data(fixture.config_path, tmp_path / "first")
    second = stage_study_data(fixture.config_path, tmp_path / "second")

    assert first.inventory_sha256 == second.inventory_sha256
    assert (tmp_path / "first/inventory.json").read_bytes() == (
        tmp_path / "second/inventory.json"
    ).read_bytes()
    assert (tmp_path / "first/assignments.jsonl").read_bytes() == (
        tmp_path / "second/assignments.jsonl"
    ).read_bytes()


def test_source_digest_mismatch_leaves_no_output(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    fixture.paths["scifact/train_claims"].write_bytes(b"tampered\n")
    output = tmp_path / "staged"

    with pytest.raises(StudyDataError, match="SHA-256 mismatch"):
        stage_study_data(fixture.config_path, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".staged.staging-*"))


def test_placeholder_revision_and_sampled_fullwiki_are_rejected(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    fixture.pins["scifact/corpus"]["revision"] = "latest"
    fixture.write_config()
    with pytest.raises(StudyDataError, match="movable placeholder"):
        load_staging_config(fixture.config_path)

    fixture.pins["scifact/corpus"]["revision"] = "0123456789abcdef0123456789abcdef01234567"
    scope = fixture.payload["datasets"]["hotpotqa_fullwiki"]["corpus_scope"]  # type: ignore[index]
    scope["sampling"] = "sampled"  # type: ignore[index]
    fixture.write_config()
    with pytest.raises(StudyDataError, match="cannot be relabeled as FullWiki"):
        load_staging_config(fixture.config_path)


def test_legacy_outcome_blind_label_claim_is_not_a_valid_public_field(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    fixture.payload["outcome_blind_sealed_labels"] = fixture.payload.pop(
        "withhold_sealed_labels_from_online_process"
    )
    fixture.write_config()

    with pytest.raises(StudyDataError, match="fields differ"):
        load_staging_config(fixture.config_path)


def test_registered_bright_domain_set_is_closed(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    bright = fixture.payload["datasets"]["bright"]  # type: ignore[index]
    removed_domain = BRIGHT_DOMAINS[-1]
    bright["domain_order"].remove(removed_domain)  # type: ignore[union-attr]
    del bright["domains"][removed_domain]  # type: ignore[index]
    fixture.write_config()

    with pytest.raises(StudyDataError, match="registered 12-domain set"):
        load_staging_config(fixture.config_path)


def test_bright_identity_conflict_is_rejected(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    source_id = "bright/theoremqa_questions/documents"
    rows = _jsonl(fixture.paths[source_id])
    rows[0]["content"] = "Conflicting content under the same upstream identifier."
    _write_jsonl(fixture.paths[source_id], rows)
    fixture.repin(source_id)

    with pytest.raises(StudyDataError, match="conflicting content"):
        stage_study_data(fixture.config_path, tmp_path / "staged")


def test_hotpot_shard_tree_digest_is_enforced(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    shard = tmp_path / "hotpot-shards/AA/wiki_00.bz2"
    shard.write_bytes(shard.read_bytes() + b"tampered")

    with pytest.raises(StudyDataError, match="SHA-256 mismatch"):
        stage_study_data(fixture.config_path, tmp_path / "staged")


def test_official_parquet_adapters(tmp_path: Path) -> None:
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    fixture = _study_fixture(tmp_path)

    for source_id in (
        "bright/biology/documents",
        "bright/biology/examples",
        "hotpotqa_fullwiki/dev_questions",
    ):
        rows = (
            json.loads(fixture.paths[source_id].read_text(encoding="utf-8"))
            if source_id == "hotpotqa_fullwiki/dev_questions"
            else _jsonl(fixture.paths[source_id])
        )
        if source_id == "hotpotqa_fullwiki/dev_questions":
            for row in rows:
                facts = row["supporting_facts"]
                row["supporting_facts"] = {
                    "sent_id": [fact[1] for fact in facts],
                    "title": [fact[0] for fact in facts],
                }
        parquet_path = fixture.paths[source_id].with_suffix(".parquet")
        parquet.write_table(arrow.Table.from_pylist(rows), parquet_path)
        fixture.paths[source_id] = parquet_path
        fixture.pins[source_id]["path"] = parquet_path.name
        fixture.pins[source_id]["sha256"] = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    fixture.write_config()

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)
    verify_staged_data(output)
    assert _jsonl(output / "datasets/hotpotqa-fullwiki/sealed/online/queries.jsonl") == [
        {"id": "hotpotqa:hp-question", "text": "Which Hotpot article is first?"}
    ]


def test_cross_source_split_component_is_excluded_as_a_complete_group(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    dev_rows = json.loads(fixture.paths["t2_finqa/dev"].read_text(encoding="utf-8"))
    test_rows = json.loads(fixture.paths["t2_finqa/test"].read_text(encoding="utf-8"))
    test_rows[0]["context_id"] = dev_rows[0]["context_id"]
    test_rows[0]["context"] = dev_rows[0]["context"]
    test_rows[0]["file_name"] = dev_rows[0]["file_name"]
    _write_json(fixture.paths["t2_finqa/test"], test_rows)
    fixture.repin("t2_finqa/test")

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    exclusions = _jsonl(output / "partition-exclusions.jsonl")
    assert {row["query_id"] for row in exclusions} == {
        "t2-ragbench:finqa-dev",
        "t2-ragbench:finqa-test",
    }
    assert {row["source_split"] for row in exclusions} == {"dev", "test"}
    assert {row["schema_version"] for row in exclusions} == {
        "fractal-study-query-partition-exclusion-v1"
    }
    assert {row["rule_id"] for row in exclusions} == {"source-split-component-isolation-v1"}
    assert {row["reason"] for row in exclusions} == {"cross-source-split-component"}
    assert len({row["partition_component_sha256"] for row in exclusions}) == 1
    assert all(row["positive_relevance_identity_sha256s"] for row in exclusions)
    assignments = _jsonl(output / "assignments.jsonl")
    assigned_ids = {row["query_id"] for row in assignments}
    assert not ({"t2-ragbench:finqa-dev", "t2-ragbench:finqa-test"} & assigned_ids)
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    counts = inventory["counts"]["t2-ragbench"]
    assert counts["partition_excluded_queries"] == 2
    assert counts["partition_excluded_dev_queries"] == 1
    assert counts["partition_excluded_test_queries"] == 1


def test_cross_split_normalized_duplicate_excludes_both_queries(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    train_path = fixture.paths["t2_finqa/train"]
    dev_path = fixture.paths["t2_finqa/dev"]
    train_rows = json.loads(train_path.read_text(encoding="utf-8"))
    dev_rows = json.loads(dev_path.read_text(encoding="utf-8"))
    train_rows[0]["question"] = "What was the reported net income in 2020?"
    dev_rows[0]["question"] = "WHAT was the reported net_income in 2020!!!"
    train_rows.append(
        {
            "context": "Independent fit context.",
            "context_id": "t2-context-train-safe",
            "file_name": "train-safe.pdf",
            "id": "finqa-train-safe",
            "question": "Which fit record is independent?",
            "split": "train",
            "subset": "FinQA",
        }
    )
    dev_rows.append(
        {
            "context": "Independent calibration context.",
            "context_id": "t2-context-dev-safe",
            "file_name": "dev-safe.pdf",
            "id": "finqa-dev-safe",
            "question": "Which calibration record is independent?",
            "split": "dev",
            "subset": "FinQA",
        }
    )
    _write_json(train_path, train_rows)
    _write_json(dev_path, dev_rows)
    fixture.repin("t2_finqa/train")
    fixture.repin("t2_finqa/dev")

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    exclusions = _jsonl(output / "partition-exclusions.jsonl")
    assert {row["query_id"] for row in exclusions} == {
        "t2-ragbench:finqa-train",
        "t2-ragbench:finqa-dev",
    }
    assert len({row["normalized_query_text_sha256"] for row in exclusions}) == 1
    assignment_ids = {row["query_id"] for row in _jsonl(output / "assignments.jsonl")}
    assert "t2-ragbench:finqa-train-safe" in assignment_ids
    assert "t2-ragbench:finqa-dev-safe" in assignment_ids


def test_transitive_exact_near_and_positive_edges_exclude_full_component(
    tmp_path: Path,
) -> None:
    fixture = _study_fixture(tmp_path)
    paths = {split: fixture.paths[f"t2_finqa/{split}"] for split in ("train", "dev", "test")}
    rows = {split: json.loads(path.read_text(encoding="utf-8")) for split, path in paths.items()}
    rows["train"][0]["question"] = "alpha beta gamma delta epsilon zeta theta"
    rows["dev"][0]["question"] = "alpha beta gamma delta epsilon zeta iota"
    rows["test"][0]["context_id"] = rows["dev"][0]["context_id"]
    rows["test"][0]["context"] = rows["dev"][0]["context"]
    rows["test"][0]["file_name"] = rows["dev"][0]["file_name"]
    for split in ("train", "dev", "test"):
        rows[split].append(
            {
                "context": f"Independent {split} context.",
                "context_id": f"t2-context-{split}-safe",
                "file_name": f"{split}-safe.pdf",
                "id": f"finqa-{split}-safe",
                "question": f"Independent {split} control question?",
                "split": split,
                "subset": "FinQA",
            }
        )
        _write_json(paths[split], rows[split])
        fixture.repin(f"t2_finqa/{split}")

    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    exclusions = _jsonl(output / "partition-exclusions.jsonl")
    assert {row["query_id"] for row in exclusions} == {
        "t2-ragbench:finqa-train",
        "t2-ragbench:finqa-dev",
        "t2-ragbench:finqa-test",
    }
    assert len({row["partition_component_sha256"] for row in exclusions}) == 1
    assert {row["source_split"] for row in exclusions} == {"train", "dev", "test"}
    assignment_ids = {row["query_id"] for row in _jsonl(output / "assignments.jsonl")}
    assert {
        "t2-ragbench:finqa-train-safe",
        "t2-ragbench:finqa-dev-safe",
        "t2-ragbench:finqa-test-safe",
    } <= assignment_ids


def test_fullwiki_count_is_exact(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    scope = fixture.payload["datasets"]["hotpotqa_fullwiki"]["corpus_scope"]  # type: ignore[index]
    scope["expected_document_count"] = 4  # type: ignore[index]
    fixture.write_config()

    with pytest.raises(StudyDataError, match="sampled corpora are not admissible"):
        stage_study_data(fixture.config_path, tmp_path / "staged")


def test_package_is_exclusive_and_verifier_detects_mutation(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)

    with pytest.raises(StudyDataError, match="will not be overwritten"):
        stage_study_data(fixture.config_path, output)
    queries = output / "datasets/scifact/sealed/online/queries.jsonl"
    queries.write_bytes(queries.read_bytes() + b"{}\n")
    with pytest.raises(StudyDataError, match="byte count changed"):
        verify_staged_data(output)


def test_verifier_recomputes_record_counts(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    inventory["artifacts"][0]["record_count"] += 1
    inventory_bytes = _canonical(inventory) + b"\n"
    (output / "inventory.json").write_bytes(inventory_bytes)
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    (output / "inventory.sha256").write_text(
        f"{inventory_sha256}  inventory.json\n", encoding="ascii"
    )

    with pytest.raises(StudyDataError, match="record count changed"):
        verify_staged_data(output)


def test_verifier_rejects_artifact_symlink_before_read(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    output = tmp_path / "staged"
    stage_study_data(fixture.config_path, output)
    target = output / "datasets/scifact/sealed/online/queries.jsonl"
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(StudyDataError, match="contains symlink"):
        verify_staged_data(output)


def test_module_cli_stages_and_verifies(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _study_fixture(tmp_path)
    output = tmp_path / "staged"

    assert main(["stage", "--config", str(fixture.config_path), "--output", str(output)]) == 0
    staged_stdout = json.loads(capsys.readouterr().out)
    assert staged_stdout["source_count"] == 40
    assert main(["verify", "--root", str(output)]) == 0
    verify_stdout = json.loads(capsys.readouterr().out)
    assert verify_stdout["inventory_sha256"] == staged_stdout["inventory_sha256"]


def test_online_projection_has_exact_non_label_payload_membership(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    staged = tmp_path / "staged"
    projected = tmp_path / "online-projection"
    source_receipt = stage_study_data(fixture.config_path, staged)

    receipt = project_online_staging(staged, projected)
    verified = verify_online_staging_projection(
        projected,
        expected_inventory_sha256=source_receipt.inventory_sha256,
    )

    assert receipt == verified
    assert (projected / "inventory.json").read_bytes() == (staged / "inventory.json").read_bytes()
    assert (projected / "inventory.sha256").read_bytes() == (
        staged / "inventory.sha256"
    ).read_bytes()
    assert (projected / ONLINE_PROJECTION_RECEIPT_FILENAME).is_file()
    inventory = json.loads((staged / "inventory.json").read_text(encoding="utf-8"))
    for row in inventory["artifacts"]:
        target = projected / row["path"]
        if row["role"] in {"qrels", "evidence-bundles"}:
            assert not target.exists()
        else:
            assert target.is_file()
            assert hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]
    assert receipt.projected_artifact_count < receipt.source_artifact_count


def test_online_projection_rejects_injected_outcome_payload(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    staged = tmp_path / "staged"
    projected = tmp_path / "online-projection"
    stage_study_data(fixture.config_path, staged)
    project_online_staging(staged, projected)
    relative = "datasets/scifact/sealed/custody/qrels.jsonl"
    target = projected / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((staged / relative).read_bytes())

    with pytest.raises(StudyDataError, match="projection membership changed"):
        verify_online_staging_projection(projected)


def test_online_projection_rejects_mutation_and_overwrite(tmp_path: Path) -> None:
    fixture = _study_fixture(tmp_path)
    staged = tmp_path / "staged"
    projected = tmp_path / "online-projection"
    stage_study_data(fixture.config_path, staged)
    project_online_staging(staged, projected)
    with pytest.raises(StudyDataError, match="already exists"):
        project_online_staging(staged, projected)

    target = projected / "datasets/scifact/sealed/online/queries.jsonl"
    encoded = bytearray(target.read_bytes())
    encoded[0] ^= 1
    target.write_bytes(encoded)
    with pytest.raises(StudyDataError, match="differs from its source pin"):
        verify_online_staging_projection(projected)


def test_module_cli_projects_and_verifies_online_tree(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _study_fixture(tmp_path)
    staged = tmp_path / "staged"
    projected = tmp_path / "online-projection"
    source = stage_study_data(fixture.config_path, staged)
    assert (
        main(
            [
                "project-online",
                "--source-root",
                str(staged),
                "--output",
                str(projected),
            ]
        )
        == 0
    )
    project_stdout = json.loads(capsys.readouterr().out)
    assert project_stdout["inventory_sha256"] == source.inventory_sha256
    assert (
        main(
            [
                "verify-online",
                "--root",
                str(projected),
                "--expected-inventory-sha256",
                source.inventory_sha256,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == project_stdout
