from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from fractal_ann_diagnostics.corpora import (
    CorpusFormatError,
    NormalizedCorpus,
    assert_query_families_disjoint,
    normalize_hotpotqa,
    normalize_hotpotqa_fullwiki,
    normalize_qrels_corpus,
    normalize_scifact,
    normalize_t2_ragbench,
)


def test_scifact_rationales_become_alternative_complete_bundles(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    claims = tmp_path / "claims.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": 20,
                        "title": "Second",
                        "abstract": ["A", "B", "C"],
                        "structured": False,
                    }
                ),
                json.dumps(
                    {
                        "doc_id": 10,
                        "title": "First",
                        "abstract": ["D", "E"],
                        "structured": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    claims.write_text(
        json.dumps(
            {
                "id": 7,
                "claim": "A scientific claim",
                "evidence": {
                    "20": [
                        {"label": "SUPPORT", "sentences": [0, 2]},
                        {"label": "SUPPORT", "sentences": [1]},
                    ]
                },
                "cited_doc_ids": [20],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    normalized = normalize_scifact(corpus, claims, stage="sealed")
    assert [document.external_id for document in normalized.documents] == ["10", "20"]
    gold = normalized.queries[0].gold_evidence
    assert gold is not None
    assert len(gold.alternatives) == 2
    assert len(gold.alternatives[0].locations) == 2
    assert gold.alternatives[0].locations[0].content_hash is not None


def test_hotpot_supporting_facts_form_one_joint_bundle(tmp_path: Path) -> None:
    source = tmp_path / "hotpot.json"
    source.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "question": "Which two facts are needed?",
                    "answer": "answer",
                    "supporting_facts": [["Alpha", 1], ["Beta", 0]],
                    "context": [
                        ["Alpha", ["a0", "a1"]],
                        ["Beta", ["b0", "b1"]],
                    ],
                    "type": "bridge",
                    "level": "hard",
                }
            ]
        ),
        encoding="utf-8",
    )
    normalized = normalize_hotpotqa(source, stage="development")
    gold = normalized.queries[0].gold_evidence
    assert gold is not None
    assert len(gold.alternatives) == 1
    assert len(gold.alternatives[0].locations) == 2
    assert {location.locator for location in gold.alternatives[0].locations} == {
        "sentence:0",
        "sentence:1",
    }


def test_hotpot_missing_supporting_paragraph_fails_loudly(tmp_path: Path) -> None:
    source = tmp_path / "hotpot.json"
    source.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "question": "question",
                    "answer": "answer",
                    "supporting_facts": [["Missing", 0]],
                    "context": [["Present", ["text"]]],
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(CorpusFormatError, match="absent from context"):
        normalize_hotpotqa(source, stage="development")


def test_hotpot_out_of_range_supporting_sentence_fails_loudly(tmp_path: Path) -> None:
    source = tmp_path / "hotpot.json"
    source.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "question": "question",
                    "answer": "answer",
                    "supporting_facts": [["Present", 3]],
                    "context": [["Present", ["only sentence"]]],
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(CorpusFormatError, match="out-of-range supporting sentence"):
        normalize_hotpotqa(source, stage="development")


def test_query_family_cannot_cross_study_stages(tmp_path: Path) -> None:
    source = tmp_path / "hotpot.json"
    source.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "question": "question",
                    "answer": "answer",
                    "supporting_facts": [["Title", 0]],
                    "context": [["Title", ["text"]]],
                }
            ]
        ),
        encoding="utf-8",
    )
    development = normalize_hotpotqa(source, stage="development")
    calibration = normalize_hotpotqa(source, stage="calibration")
    with pytest.raises(CorpusFormatError, match="crosses stages"):
        assert_query_families_disjoint((development, calibration))

    duplicate_same_stage = NormalizedCorpus(
        name=development.name,
        stage=development.stage,
        documents=development.documents,
        queries=development.queries,
    )
    assert_query_families_disjoint((development, duplicate_same_stage))


def test_hotpot_context_corpus_is_rejected_for_sealed_retrieval(tmp_path: Path) -> None:
    source = tmp_path / "hotpot.json"
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(CorpusFormatError, match="cannot be a sealed retrieval corpus"):
        normalize_hotpotqa(source, stage="sealed")


def test_hotpot_fullwiki_uses_separate_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "wiki.jsonl"
    questions = tmp_path / "questions.json"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {"id": "alpha", "title": "Alpha", "sentences": ["a0", "a1"]}
                ),
                json.dumps({"id": "beta", "title": "Beta", "sentences": ["b0"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    questions.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "question": "question",
                    "answer": "answer",
                    "supporting_facts": [["Alpha", 1], ["Beta", 0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    normalized = normalize_hotpotqa_fullwiki(corpus, questions, stage="sealed")
    assert normalized.name == "hotpotqa-fullwiki"
    assert normalized.queries[0].relevant_document_ids == (0, 1)
    gold = normalized.queries[0].gold_evidence
    assert gold is not None
    assert len(gold.alternatives[0].locations) == 2


def test_t2_rows_deduplicate_contexts_and_bind_gold_document(tmp_path: Path) -> None:
    source = tmp_path / "t2.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "context_id": "report-1",
                        "question": "What changed?",
                        "program_answer": "12",
                        "context": "Narrative\n| Year | Value |",
                        "file_name": "report.pdf",
                        "split": "test",
                    }
                ),
                json.dumps(
                    {
                        "id": "q2",
                        "context_id": "report-1",
                        "question": "What is the value?",
                        "original_answer": "12",
                        "context": "Narrative\n| Year | Value |",
                        "file_name": "report.pdf",
                        "split": "test",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    normalized = normalize_t2_ragbench(source, stage="sealed")
    assert len(normalized.documents) == 1
    assert len(normalized.queries) == 2
    assert all(query.relevant_document_ids == (0,) for query in normalized.queries)


def test_qrels_adapter_preserves_relevance_without_claiming_evidence(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    documents.write_text(
        json.dumps({"id": "d1", "title": "Title", "text": "Body"}) + "\n",
        encoding="utf-8",
    )
    queries.write_text(
        json.dumps({"id": "q1", "text": "Query"}) + "\n",
        encoding="utf-8",
    )
    qrels.write_text(
        json.dumps({"query_id": "q1", "document_id": "d1", "relevance": 2}) + "\n",
        encoding="utf-8",
    )
    normalized = normalize_qrels_corpus(
        documents,
        queries,
        qrels,
        corpus_name="bright",
        stage="sealed",
    )
    assert normalized.queries[0].relevant_document_ids == (0,)
    assert normalized.queries[0].gold_evidence is None


def test_normalized_corpus_rejects_forged_gold_provenance(tmp_path: Path) -> None:
    source = tmp_path / "hotpot.json"
    source.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "question": "question",
                    "answer": "answer",
                    "supporting_facts": [["Title", 0]],
                    "context": [["Title", ["sentence"]]],
                }
            ]
        ),
        encoding="utf-8",
    )
    normalized = normalize_hotpotqa(source, stage="development")
    query = normalized.queries[0]
    assert query.gold_evidence is not None
    bundle = query.gold_evidence.alternatives[0]
    location = bundle.locations[0]
    forged_location = replace(location, content_hash="sha256:" + "0" * 64)
    forged_bundle = replace(bundle, locations=(forged_location,))
    forged_gold = replace(query.gold_evidence, alternatives=(forged_bundle,))

    with pytest.raises(ValueError, match="content hash does not match"):
        NormalizedCorpus(
            name=normalized.name,
            stage=normalized.stage,
            documents=normalized.documents,
            queries=(replace(query, gold_evidence=forged_gold),),
        )


def test_normalized_corpus_binds_gold_to_the_exact_query(tmp_path: Path) -> None:
    source = tmp_path / "hotpot.json"
    source.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "question": "question",
                    "answer": "answer",
                    "supporting_facts": [["Title", 0]],
                    "context": [["Title", ["sentence"]]],
                }
            ]
        ),
        encoding="utf-8",
    )
    normalized = normalize_hotpotqa(source, stage="development")
    query = normalized.queries[0]
    assert query.gold_evidence is not None

    with pytest.raises(ValueError, match="query ID must match"):
        NormalizedCorpus(
            name=normalized.name,
            stage=normalized.stage,
            documents=normalized.documents,
            queries=(
                replace(
                    query,
                    gold_evidence=replace(query.gold_evidence, query_id="hotpotqa:other"),
                ),
            ),
        )
