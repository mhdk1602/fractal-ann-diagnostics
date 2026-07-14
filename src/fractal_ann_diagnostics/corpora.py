"""Normalization adapters for evidence-bearing confirmatory corpora."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from urllib.parse import quote

from .evidence import CompleteEvidenceBundle, EvidenceLocation, GoldEvidence


class CorpusFormatError(ValueError):
    """Raised when source data cannot satisfy the registered evidence schema."""


def _content_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusFormatError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise CorpusFormatError(f"line {line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _json_array(path: str | Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusFormatError(f"invalid JSON array: {exc}") from exc
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise CorpusFormatError("source file must contain an array of objects")
    return payload


@dataclass(frozen=True)
class CorpusDocument:
    """One normalized retrieval document with immutable source provenance."""

    document_id: int
    external_id: str
    title: str
    text: str
    source_uri: str
    content_hash: str

    def __post_init__(self) -> None:
        if self.document_id < 0:
            raise ValueError("document_id must be non-negative")
        for name in ("external_id", "title", "text", "source_uri"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.content_hash) is None:
            raise ValueError("content_hash must use sha256:<lowercase hex>")


@dataclass(frozen=True)
class EvidenceQuery:
    """One normalized query, answer, and optional complete-evidence annotation."""

    query_id: str
    query_family: str
    text: str
    corpus: str
    stage: str
    answer: str | None
    gold_evidence: GoldEvidence | None
    relevant_document_ids: tuple[int, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("query_id", "query_family", "text", "corpus", "stage"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        relevant = tuple(int(document_id) for document_id in self.relevant_document_ids)
        if any(document_id < 0 for document_id in relevant) or len(set(relevant)) != len(
            relevant
        ):
            raise ValueError("relevant_document_ids must be unique and non-negative")
        object.__setattr__(self, "relevant_document_ids", relevant)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType({str(key): str(value) for key, value in self.metadata.items()}),
        )


@dataclass(frozen=True)
class NormalizedCorpus:
    name: str
    stage: str
    documents: tuple[CorpusDocument, ...]
    queries: tuple[EvidenceQuery, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.stage:
            raise ValueError("normalized corpus name and stage must be non-empty")
        documents = tuple(self.documents)
        queries = tuple(self.queries)
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "queries", queries)
        if not documents or not queries:
            raise ValueError("a normalized corpus needs documents and queries")
        document_ids = [document.document_id for document in documents]
        if document_ids != list(range(len(document_ids))):
            raise ValueError("normalized document IDs must be contiguous and ordered")
        external_ids = [document.external_id for document in documents]
        if len(set(external_ids)) != len(external_ids):
            raise ValueError("normalized external document IDs must be unique")
        query_ids = [query.query_id for query in queries]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("normalized query IDs must be unique")
        for query in queries:
            if query.corpus != self.name or query.stage != self.stage:
                raise ValueError("every query must match the normalized corpus and stage")
            if any(
                document_id >= len(documents)
                for document_id in query.relevant_document_ids
            ):
                raise ValueError("query relevance judgment names an unknown document")
            gold = query.gold_evidence
            if gold is None:
                continue
            if gold.query_id != query.query_id:
                raise ValueError("gold evidence query ID must match its normalized query")
            gold_document_ids: set[int] = set()
            for bundle in gold.alternatives:
                for location in bundle.locations:
                    if location.document_id >= len(documents):
                        raise ValueError("gold evidence names an unknown document")
                    document = documents[location.document_id]
                    if location.source_uri != document.source_uri:
                        raise ValueError("gold evidence source URI does not match its document")
                    if location.content_hash != document.content_hash:
                        raise ValueError("gold evidence content hash does not match its document")
                    sentence = re.fullmatch(r"sentence:(\d+)", location.locator)
                    if sentence is not None and int(sentence.group(1)) >= len(
                        document.text.splitlines()
                    ):
                        raise ValueError("gold evidence sentence locator is out of range")
                    gold_document_ids.add(location.document_id)
            if not gold_document_ids.issubset(query.relevant_document_ids):
                raise ValueError("gold evidence documents must be relevance-judged")


class DocumentRegistry:
    """Deterministic external-to-internal document ID mapping."""

    def __init__(self, external_ids: Iterable[str]) -> None:
        ordered = sorted({str(external_id) for external_id in external_ids})
        self._mapping = {external_id: index for index, external_id in enumerate(ordered)}

    def __len__(self) -> int:
        return len(self._mapping)

    def id_for(self, external_id: str | int) -> int:
        key = str(external_id)
        try:
            return self._mapping[key]
        except KeyError as exc:
            raise CorpusFormatError(f"evidence names unknown document {key!r}") from exc


def normalize_scifact(
    corpus_path: str | Path,
    claims_path: str | Path,
    *,
    stage: str,
) -> NormalizedCorpus:
    """Normalize SciFact abstracts and rationale alternatives."""
    corpus_rows = _jsonl(corpus_path)
    claim_rows = _jsonl(claims_path)
    external_ids = [str(row.get("doc_id")) for row in corpus_rows]
    if any(external_id == "None" for external_id in external_ids):
        raise CorpusFormatError("every SciFact corpus row needs doc_id")
    registry = DocumentRegistry(external_ids)
    documents: list[CorpusDocument] = []
    by_external: dict[str, CorpusDocument] = {}
    for row in corpus_rows:
        external_id = str(row["doc_id"])
        title = row.get("title")
        abstract = row.get("abstract")
        if not isinstance(title, str) or not isinstance(abstract, list) or not all(
            isinstance(sentence, str) for sentence in abstract
        ):
            raise CorpusFormatError(f"SciFact document {external_id} has invalid text fields")
        source_uri = f"scifact://document/{quote(external_id, safe='')}"
        content_hash = _content_hash(title, *abstract)
        document = CorpusDocument(
            document_id=registry.id_for(external_id),
            external_id=external_id,
            title=title,
            text="\n".join(abstract),
            source_uri=source_uri,
            content_hash=content_hash,
        )
        documents.append(document)
        by_external[external_id] = document
    documents.sort(key=lambda document: document.document_id)

    queries: list[EvidenceQuery] = []
    for row in claim_rows:
        claim_id = row.get("id")
        claim = row.get("claim")
        evidence = row.get("evidence", {})
        if claim_id is None or not isinstance(claim, str) or not isinstance(evidence, dict):
            raise CorpusFormatError("SciFact claim needs id, claim, and evidence object")
        alternatives: list[CompleteEvidenceBundle] = []
        labels: set[str] = set()
        for external_id, rationales in sorted(evidence.items()):
            if not isinstance(rationales, list):
                raise CorpusFormatError(f"claim {claim_id} has invalid rationale list")
            document = by_external.get(str(external_id))
            if document is None:
                raise CorpusFormatError(
                    f"claim {claim_id} names unknown document {external_id!r}"
                )
            for rationale_index, rationale in enumerate(rationales):
                if not isinstance(rationale, dict):
                    raise CorpusFormatError(f"claim {claim_id} has invalid rationale")
                sentences = rationale.get("sentences")
                label = rationale.get("label")
                if not isinstance(sentences, list) or not sentences or not all(
                    type(sentence) is int and sentence >= 0 for sentence in sentences
                ):
                    raise CorpusFormatError(
                        f"claim {claim_id} rationale must name non-negative sentences"
                    )
                if label not in {"SUPPORT", "CONTRADICT"}:
                    raise CorpusFormatError(f"claim {claim_id} has unknown evidence label")
                if any(sentence >= len(document.text.splitlines()) for sentence in sentences):
                    raise CorpusFormatError(
                        f"claim {claim_id} rationale sentence is out of range"
                    )
                labels.add(str(label))
                locations = tuple(
                    EvidenceLocation(
                        document_id=document.document_id,
                        source_uri=document.source_uri,
                        locator=f"sentence:{sentence}",
                        content_hash=document.content_hash,
                    )
                    for sentence in sentences
                )
                alternatives.append(
                    CompleteEvidenceBundle(
                        bundle_id=(
                            f"document-{external_id}-rationale-{rationale_index}"
                        ),
                        locations=locations,
                    )
                )
        query_id = f"scifact:{claim_id}"
        gold = (
            GoldEvidence(query_id=query_id, alternatives=tuple(alternatives))
            if alternatives
            else None
        )
        queries.append(
            EvidenceQuery(
                query_id=query_id,
                query_family=query_id,
                text=claim,
                corpus="scifact",
                stage=stage,
                answer=None,
                gold_evidence=gold,
                relevant_document_ids=tuple(
                    sorted(
                        {
                            location.document_id
                            for bundle in alternatives
                            for location in bundle.locations
                        }
                    )
                ),
                metadata={"evidence_labels": ",".join(sorted(labels))},
            )
        )
    return NormalizedCorpus(
        name="scifact",
        stage=stage,
        documents=tuple(documents),
        queries=tuple(queries),
    )


def normalize_hotpotqa(path: str | Path, *, stage: str) -> NormalizedCorpus:
    """Normalize HotpotQA's per-question context as a development fixture.

    This is not a FullWiki retrieval corpus. Sealed use is rejected because
    supplied contexts preselect candidate paragraphs for each query.
    """
    if stage == "sealed":
        raise CorpusFormatError(
            "HotpotQA context files cannot be a sealed retrieval corpus; "
            "use normalize_hotpotqa_fullwiki with an external corpus"
        )
    rows = _json_array(path)
    paragraphs: dict[str, tuple[str, ...]] = {}
    for row in rows:
        context = row.get("context")
        if not isinstance(context, list):
            raise CorpusFormatError("HotpotQA row needs context")
        for paragraph in context:
            if (
                not isinstance(paragraph, list)
                or len(paragraph) != 2
                or not isinstance(paragraph[0], str)
                or not isinstance(paragraph[1], list)
                or not all(isinstance(sentence, str) for sentence in paragraph[1])
            ):
                raise CorpusFormatError("HotpotQA paragraph must be [title, sentences]")
            title, sentences = paragraph
            observed = tuple(sentences)
            if title in paragraphs and paragraphs[title] != observed:
                raise CorpusFormatError(f"HotpotQA title {title!r} has conflicting text")
            paragraphs[title] = observed
    registry = DocumentRegistry(paragraphs)
    documents = tuple(
        CorpusDocument(
            document_id=registry.id_for(title),
            external_id=title,
            title=title,
            text="\n".join(sentences),
            source_uri=f"hotpotqa://title/{quote(title, safe='')}",
            content_hash=_content_hash(title, *sentences),
        )
        for title, sentences in sorted(
            paragraphs.items(), key=lambda item: registry.id_for(item[0])
        )
    )
    by_title = {document.external_id: document for document in documents}

    queries: list[EvidenceQuery] = []
    for row in rows:
        query_id_value = row.get("_id")
        question = row.get("question")
        supporting_facts = row.get("supporting_facts")
        if (
            not isinstance(query_id_value, str)
            or not isinstance(question, str)
            or not isinstance(supporting_facts, list)
        ):
            raise CorpusFormatError("HotpotQA row needs _id, question, and supporting_facts")
        locations: list[EvidenceLocation] = []
        for fact in supporting_facts:
            if (
                not isinstance(fact, list)
                or len(fact) != 2
                or not isinstance(fact[0], str)
                or type(fact[1]) is not int
                or fact[1] < 0
            ):
                raise CorpusFormatError("supporting fact must be [title, sentence_id]")
            title, sentence_id = fact
            document = by_title.get(title)
            if document is None:
                raise CorpusFormatError(
                    f"question {query_id_value!r} has supporting title absent from context"
                )
            if sentence_id >= len(paragraphs[title]):
                raise CorpusFormatError(
                    f"question {query_id_value!r} has an out-of-range supporting sentence"
                )
            locations.append(
                EvidenceLocation(
                    document_id=document.document_id,
                    source_uri=document.source_uri,
                    locator=f"sentence:{sentence_id}",
                    content_hash=document.content_hash,
                )
            )
        query_id = f"hotpotqa:{query_id_value}"
        gold = (
            GoldEvidence(
                query_id=query_id,
                alternatives=(
                    CompleteEvidenceBundle(
                        bundle_id="supporting-facts",
                        locations=tuple(locations),
                    ),
                ),
            )
            if locations
            else None
        )
        queries.append(
            EvidenceQuery(
                query_id=query_id,
                query_family=query_id,
                text=question,
                corpus="hotpotqa",
                stage=stage,
                answer=(str(row["answer"]) if "answer" in row else None),
                gold_evidence=gold,
                relevant_document_ids=tuple(
                    sorted({location.document_id for location in locations})
                ),
                metadata={
                    "type": str(row.get("type", "")),
                    "level": str(row.get("level", "")),
                },
            )
        )
    return NormalizedCorpus(
        name="hotpotqa",
        stage=stage,
        documents=documents,
        queries=tuple(queries),
    )


def _text_from_document_row(row: Mapping[str, Any]) -> tuple[str, str, str]:
    external_id_value = row.get("id", row.get("docid", row.get("_id", row.get("title"))))
    title_value = row.get("title", external_id_value)
    text_value = row.get("text", row.get("contents", row.get("sentences")))
    if isinstance(text_value, list) and all(isinstance(item, str) for item in text_value):
        text_value = "\n".join(text_value)
    if external_id_value is None or not isinstance(title_value, str) or not isinstance(
        text_value, str
    ):
        raise CorpusFormatError("document rows need an ID, title, and text")
    return str(external_id_value), title_value, text_value


def normalize_hotpotqa_fullwiki(
    corpus_path: str | Path,
    questions_path: str | Path,
    *,
    stage: str,
) -> NormalizedCorpus:
    """Normalize HotpotQA questions against a separately acquired wiki corpus."""
    parsed_documents = [_text_from_document_row(row) for row in _jsonl(corpus_path)]
    registry = DocumentRegistry(external_id for external_id, _, _ in parsed_documents)
    documents = tuple(
        CorpusDocument(
            document_id=registry.id_for(external_id),
            external_id=external_id,
            title=title,
            text=text,
            source_uri=f"hotpotqa-fullwiki://title/{quote(external_id, safe='')}",
            content_hash=_content_hash(title, text),
        )
        for external_id, title, text in sorted(
            parsed_documents,
            key=lambda item: registry.id_for(item[0]),
        )
    )
    by_title = {document.title: document for document in documents}
    queries: list[EvidenceQuery] = []
    for row in _json_array(questions_path):
        query_id_value = row.get("_id")
        question = row.get("question")
        supporting_facts = row.get("supporting_facts")
        if (
            not isinstance(query_id_value, str)
            or not isinstance(question, str)
            or not isinstance(supporting_facts, list)
        ):
            raise CorpusFormatError(
                "HotpotQA FullWiki rows need _id, question, and supporting_facts"
            )
        locations: list[EvidenceLocation] = []
        for fact in supporting_facts:
            if (
                not isinstance(fact, list)
                or len(fact) != 2
                or not isinstance(fact[0], str)
                or type(fact[1]) is not int
                or fact[1] < 0
            ):
                raise CorpusFormatError("supporting fact must be [title, sentence_id]")
            document = by_title.get(fact[0])
            if document is None:
                raise CorpusFormatError(
                    f"question {query_id_value!r} names a title absent from the external corpus"
                )
            if fact[1] >= len(document.text.splitlines()):
                raise CorpusFormatError(
                    f"question {query_id_value!r} has an out-of-range supporting sentence"
                )
            locations.append(
                EvidenceLocation(
                    document_id=document.document_id,
                    source_uri=document.source_uri,
                    locator=f"sentence:{fact[1]}",
                    content_hash=document.content_hash,
                )
            )
        query_id = f"hotpotqa:{query_id_value}"
        queries.append(
            EvidenceQuery(
                query_id=query_id,
                query_family=query_id,
                text=question,
                corpus="hotpotqa-fullwiki",
                stage=stage,
                answer=(str(row["answer"]) if "answer" in row else None),
                gold_evidence=GoldEvidence(
                    query_id=query_id,
                    alternatives=(
                        CompleteEvidenceBundle(
                            bundle_id="supporting-facts",
                            locations=tuple(locations),
                        ),
                    ),
                ),
                relevant_document_ids=tuple(
                    sorted({location.document_id for location in locations})
                ),
                metadata={
                    "type": str(row.get("type", "")),
                    "level": str(row.get("level", "")),
                },
            )
        )
    return NormalizedCorpus(
        name="hotpotqa-fullwiki",
        stage=stage,
        documents=documents,
        queries=tuple(queries),
    )


def normalize_t2_ragbench(path: str | Path, *, stage: str) -> NormalizedCorpus:
    """Normalize exported T2-RAGBench rows into unique financial documents."""
    source = Path(path)
    rows = _jsonl(source) if source.suffix == ".jsonl" else _json_array(source)
    contexts: dict[str, tuple[str, str]] = {}
    for row in rows:
        context_id = row.get("context_id")
        context = row.get("context")
        file_name = row.get("file_name", context_id)
        if context_id is None or not isinstance(context, str) or not isinstance(file_name, str):
            raise CorpusFormatError("T2-RAGBench rows need context_id, context, and file_name")
        key = str(context_id)
        value = (file_name, context)
        if key in contexts and contexts[key] != value:
            raise CorpusFormatError(f"T2-RAGBench context {key!r} has conflicting content")
        contexts[key] = value
    registry = DocumentRegistry(contexts)
    documents = tuple(
        CorpusDocument(
            document_id=registry.id_for(context_id),
            external_id=context_id,
            title=file_name,
            text=context,
            source_uri=f"t2-ragbench://context/{quote(context_id, safe='')}",
            content_hash=_content_hash(file_name, context),
        )
        for context_id, (file_name, context) in sorted(
            contexts.items(),
            key=lambda item: registry.id_for(item[0]),
        )
    )
    by_external = {document.external_id: document for document in documents}
    queries: list[EvidenceQuery] = []
    for row in rows:
        query_id_value = row.get("id")
        question = row.get("question")
        context_id = str(row.get("context_id"))
        if query_id_value is None or not isinstance(question, str):
            raise CorpusFormatError("T2-RAGBench rows need id and question")
        document = by_external[context_id]
        query_id = f"t2-ragbench:{query_id_value}"
        location = EvidenceLocation(
            document_id=document.document_id,
            source_uri=document.source_uri,
            locator="document",
            content_hash=document.content_hash,
        )
        answer_value = row.get("program_answer", row.get("original_answer"))
        queries.append(
            EvidenceQuery(
                query_id=query_id,
                query_family=query_id,
                text=question,
                corpus="t2-ragbench",
                stage=stage,
                answer=(str(answer_value) if answer_value is not None else None),
                gold_evidence=GoldEvidence(
                    query_id=query_id,
                    alternatives=(
                        CompleteEvidenceBundle(
                            bundle_id="source-context",
                            locations=(location,),
                        ),
                    ),
                ),
                relevant_document_ids=(document.document_id,),
                metadata={
                    "split": str(row.get("split", "")),
                    "subset": str(row.get("subset", "")),
                },
            )
        )
    return NormalizedCorpus(
        name="t2-ragbench",
        stage=stage,
        documents=documents,
        queries=tuple(queries),
    )


def normalize_qrels_corpus(
    documents_path: str | Path,
    queries_path: str | Path,
    qrels_path: str | Path,
    *,
    corpus_name: str,
    stage: str,
) -> NormalizedCorpus:
    """Normalize BRIGHT/MIRACL-style JSONL documents, queries, and qrels."""
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", corpus_name) is None:
        raise ValueError("corpus_name must be a lowercase URI-safe identifier")
    parsed_documents = [_text_from_document_row(row) for row in _jsonl(documents_path)]
    registry = DocumentRegistry(external_id for external_id, _, _ in parsed_documents)
    documents = tuple(
        CorpusDocument(
            document_id=registry.id_for(external_id),
            external_id=external_id,
            title=title,
            text=text,
            source_uri=(
                f"{quote(corpus_name, safe='')}://document/{quote(external_id, safe='')}"
            ),
            content_hash=_content_hash(title, text),
        )
        for external_id, title, text in sorted(
            parsed_documents,
            key=lambda item: registry.id_for(item[0]),
        )
    )
    relevance: dict[str, set[int]] = {}
    for row in _jsonl(qrels_path):
        query_id = row.get("query_id", row.get("qid"))
        document_id = row.get("document_id", row.get("docid"))
        score = row.get("relevance", row.get("score"))
        if (
            query_id is None
            or document_id is None
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise CorpusFormatError(
                "qrels rows need query_id, document_id, and numeric relevance"
            )
        if score > 0:
            relevance.setdefault(str(query_id), set()).add(
                registry.id_for(str(document_id))
            )

    queries: list[EvidenceQuery] = []
    for row in _jsonl(queries_path):
        query_id_value = row.get("id", row.get("query_id", row.get("qid")))
        text = row.get("text", row.get("query"))
        if query_id_value is None or not isinstance(text, str):
            raise CorpusFormatError("query rows need an ID and text")
        external_query_id = str(query_id_value)
        query_id = f"{corpus_name}:{external_query_id}"
        queries.append(
            EvidenceQuery(
                query_id=query_id,
                query_family=query_id,
                text=text,
                corpus=corpus_name,
                stage=stage,
                answer=None,
                gold_evidence=None,
                relevant_document_ids=tuple(
                    sorted(relevance.get(external_query_id, set()))
                ),
                metadata={},
            )
        )
    known_query_ids = {
        query.query_id.removeprefix(f"{corpus_name}:") for query in queries
    }
    unknown_queries = set(relevance) - known_query_ids
    if unknown_queries:
        raise CorpusFormatError(f"qrels name unknown queries: {sorted(unknown_queries)}")
    return NormalizedCorpus(
        name=corpus_name,
        stage=stage,
        documents=documents,
        queries=tuple(queries),
    )


def assert_query_families_disjoint(corpora: Iterable[NormalizedCorpus]) -> None:
    """Compatibility wrapper for the graph-based query partition audit."""
    from .partition_audit import audit_query_partitions

    audit_query_partitions(corpora)
