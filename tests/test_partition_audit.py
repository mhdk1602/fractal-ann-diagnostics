from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from fractal_ann_diagnostics.corpora import (
    CorpusDocument,
    CorpusFormatError,
    EvidenceQuery,
    NormalizedCorpus,
    assert_query_families_disjoint,
)
from fractal_ann_diagnostics.evidence import (
    CompleteEvidenceBundle,
    EvidenceLocation,
    GoldEvidence,
)
from fractal_ann_diagnostics.partition_audit import (
    FROZEN_QUERY_PARTITION_CONFIG,
    FROZEN_QUERY_PARTITION_CONFIG_SHA256,
    QUERY_PARTITION_AUDIT_SCHEMA,
    QueryPartitionLeakageError,
    audit_query_partitions,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _documents(corpus: str, external_ids: tuple[str, ...]) -> tuple[CorpusDocument, ...]:
    return tuple(
        CorpusDocument(
            document_id=index,
            external_id=external_id,
            title=f"Title {external_id}",
            text=f"Sentence for {external_id}",
            source_uri=f"{corpus}://document/{external_id}",
            content_hash=_digest(f"Title {external_id}\nSentence for {external_id}"),
        )
        for index, external_id in enumerate(external_ids)
    )


def _corpus(
    *,
    corpus: str,
    stage: str,
    query_id: str,
    family: str,
    text: str,
    external_ids: tuple[str, ...] = ("unused",),
    relevant_external_ids: tuple[str, ...] = (),
    gold_external_ids: tuple[str, ...] = (),
) -> NormalizedCorpus:
    documents = _documents(corpus, external_ids)
    by_external = {document.external_id: document for document in documents}
    relevant_ids = tuple(
        sorted(by_external[external_id].document_id for external_id in relevant_external_ids)
    )
    gold = None
    if gold_external_ids:
        locations = tuple(
            EvidenceLocation(
                document_id=by_external[external_id].document_id,
                source_uri=by_external[external_id].source_uri,
                locator="sentence:0",
                content_hash=by_external[external_id].content_hash,
            )
            for external_id in gold_external_ids
        )
        gold = GoldEvidence(
            query_id=query_id,
            alternatives=(CompleteEvidenceBundle(bundle_id="gold", locations=locations),),
        )
    query = EvidenceQuery(
        query_id=query_id,
        query_family=family,
        text=text,
        corpus=corpus,
        stage=stage,
        answer=None,
        gold_evidence=gold,
        relevant_document_ids=relevant_ids,
    )
    return NormalizedCorpus(
        name=corpus,
        stage=stage,
        documents=documents,
        queries=(query,),
    )


def _relations(error: QueryPartitionLeakageError) -> set[str]:
    return {edge.relation for edge in error.audit.edges}


def test_frozen_config_and_canonical_audit_are_order_invariant() -> None:
    first = _corpus(
        corpus="alpha",
        stage="development-fit",
        query_id="alpha:q1",
        family="family-1",
        text="How is authorization evaluated?",
    )
    second = _corpus(
        corpus="beta",
        stage="development-fit",
        query_id="beta:q2",
        family="family-2",
        text="Which evidence is returned?",
    )

    forward = audit_query_partitions((first, second))
    reverse = audit_query_partitions((second, first, first))

    assert forward.schema == QUERY_PARTITION_AUDIT_SCHEMA
    assert forward.config is FROZEN_QUERY_PARTITION_CONFIG
    assert forward.config_sha256 == FROZEN_QUERY_PARTITION_CONFIG_SHA256
    assert FROZEN_QUERY_PARTITION_CONFIG_SHA256 == (
        "sha256:f85961157428295f2d254e172a8a9582ce8d48dcfe31bb06f8248b4b6f1bbd9f"
    )
    assert forward.canonical_bytes() == reverse.canonical_bytes()
    assert forward.sha256 == reverse.sha256
    assert forward.passed
    assert forward.component_count == 2
    assert [row["stage"] for row in json.loads(forward.canonical_bytes())["queries"]] == [
        "development-fit",
        "development-fit",
    ]
    with pytest.raises(FrozenInstanceError):
        FROZEN_QUERY_PARTITION_CONFIG.minimum_near_duplicate_tokens = 5  # type: ignore[misc]


def test_declared_dataset_family_connects_different_query_ids_across_stages() -> None:
    fit = _corpus(
        corpus="shared",
        stage="development-fit",
        query_id="shared:fit",
        family="case-19",
        text="Fit-only wording with no lexical counterpart",
        external_ids=("fit-doc",),
    )
    sealed = _corpus(
        corpus="shared",
        stage="sealed",
        query_id="shared:sealed",
        family="case-19",
        text="Unrelated sealed prompt",
        external_ids=("sealed-doc",),
    )

    with pytest.raises(QueryPartitionLeakageError, match="crosses stages") as captured:
        audit_query_partitions((fit, sealed))

    assert "declared-dataset-family" in _relations(captured.value)
    assert captured.value.audit.cross_stage_components[0].stages == (
        "development-fit",
        "sealed",
    )
    assert not captured.value.audit.passed


def test_shared_relevant_and_gold_external_document_ids_cross_the_boundary() -> None:
    fit = _corpus(
        corpus="shared",
        stage="development-fit",
        query_id="shared:fit",
        family="fit-family",
        text="What fact is present in the development record?",
        external_ids=("doc-7", "other"),
        relevant_external_ids=("doc-7",),
        gold_external_ids=("doc-7",),
    )
    calibration = _corpus(
        corpus="shared",
        stage="development-calibration",
        query_id="shared:calibration",
        family="calibration-family",
        text="Name a separate calibrated fact without copied words",
        external_ids=("other", "doc-7"),
        relevant_external_ids=("doc-7",),
        gold_external_ids=("doc-7",),
    )

    with pytest.raises(QueryPartitionLeakageError) as captured:
        audit_query_partitions((fit, calibration))

    assert {
        "shared-gold-document",
        "shared-relevant-document",
    }.issubset(_relations(captured.value))


def test_normalized_exact_text_edge_is_global_across_corpora() -> None:
    fit = _corpus(
        corpus="alpha",
        stage="development-fit",
        query_id="alpha:q1",
        family="alpha-family",
        text="WHO approved this policy revision?",
    )
    sealed = _corpus(
        corpus="beta",
        stage="sealed",
        query_id="beta:q8",
        family="beta-family",
        text="who approved this policy revision",
    )

    with pytest.raises(QueryPartitionLeakageError) as captured:
        audit_query_partitions((fit, sealed))

    assert _relations(captured.value) == {"normalized-text-exact"}


@pytest.mark.parametrize(
    ("fit_text", "sealed_text"),
    [
        (
            "which policy revision grants authorized retrieval access today",
            "which policy revision denies authorized retrieval access today",
        ),
        (
            "which policy revision grants retrieval access today",
            "which current policy revision grants retrieval access today",
        ),
    ],
)
def test_one_token_edit_is_a_conservative_near_duplicate_edge(
    fit_text: str,
    sealed_text: str,
) -> None:
    fit = _corpus(
        corpus="alpha",
        stage="development-fit",
        query_id="alpha:q1",
        family="alpha-family",
        text=fit_text,
    )
    sealed = _corpus(
        corpus="beta",
        stage="sealed",
        query_id="beta:q2",
        family="beta-family",
        text=sealed_text,
    )

    with pytest.raises(QueryPartitionLeakageError) as captured:
        audit_query_partitions((fit, sealed))

    assert "normalized-text-near" in _relations(captured.value)


def test_short_or_two_edit_texts_do_not_receive_near_duplicate_edges() -> None:
    short_fit = _corpus(
        corpus="alpha",
        stage="development-fit",
        query_id="alpha:short",
        family="short-fit",
        text="which policy grants access today",
    )
    short_sealed = _corpus(
        corpus="beta",
        stage="sealed",
        query_id="beta:short",
        family="short-sealed",
        text="which policy denies access today",
    )
    two_edit_fit = _corpus(
        corpus="gamma",
        stage="development-fit",
        query_id="gamma:two",
        family="two-fit",
        text="which policy revision grants authorized retrieval access today",
    )
    two_edit_sealed = _corpus(
        corpus="delta",
        stage="sealed",
        query_id="delta:two",
        family="two-sealed",
        text="which policy version denies authorized retrieval access today",
    )

    audit = audit_query_partitions(
        (short_fit, short_sealed, two_edit_fit, two_edit_sealed)
    )

    assert audit.passed
    assert not audit.edges
    assert audit.component_count == 4


def test_transitive_evidence_closure_rejects_a_stage_spanning_component() -> None:
    fit_one = _corpus(
        corpus="shared",
        stage="development-fit",
        query_id="shared:fit-one",
        family="family-link",
        text="A development prompt unlike the sealed prompt",
        external_ids=("fit-only",),
    )
    fit_two = _corpus(
        corpus="shared",
        stage="development-fit",
        query_id="shared:fit-two",
        family="family-link",
        text="which controller records the final authorization decision now",
        external_ids=("fit-only",),
    )
    sealed = _corpus(
        corpus="sealed-corpus",
        stage="sealed",
        query_id="sealed:q1",
        family="sealed-family",
        text="Which controller records the final authorization decision now?",
    )

    with pytest.raises(QueryPartitionLeakageError) as captured:
        audit_query_partitions((fit_one, fit_two, sealed))

    relations = _relations(captured.value)
    assert "declared-dataset-family" in relations
    assert "normalized-text-exact" in relations
    crossing = captured.value.audit.cross_stage_components[0]
    assert len(crossing.node_ids) == 3


def test_same_node_with_conflicting_records_is_rejected() -> None:
    original = _corpus(
        corpus="alpha",
        stage="development-fit",
        query_id="alpha:q1",
        family="family-1",
        text="Original query text",
    )
    conflicting = NormalizedCorpus(
        name=original.name,
        stage=original.stage,
        documents=original.documents,
        queries=(replace(original.queries[0], text="Conflicting query text"),),
    )

    with pytest.raises(CorpusFormatError, match="conflicting normalized records"):
        audit_query_partitions((original, conflicting))


def test_legacy_assertion_delegates_to_complete_partition_audit() -> None:
    fit = _corpus(
        corpus="alpha",
        stage="development-fit",
        query_id="alpha:q1",
        family="one",
        text="which evidence bundle satisfies the registered retrieval target",
    )
    sealed = _corpus(
        corpus="beta",
        stage="sealed",
        query_id="beta:q2",
        family="two",
        text="which evidence bundle satisfies the registered retrieval target",
    )

    with pytest.raises(QueryPartitionLeakageError):
        assert_query_families_disjoint((fit, sealed))
