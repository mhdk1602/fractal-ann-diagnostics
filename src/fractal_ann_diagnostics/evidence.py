"""Gold evidence bundles and answer-level confirmatory outcomes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EvidenceLocation:
    """One citable evidence location with stable document provenance."""

    document_id: int
    source_uri: str
    locator: str
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if self.document_id < 0:
            raise ValueError("document_id must be non-negative")
        if not self.source_uri.strip():
            raise ValueError("source_uri must be non-empty")
        if not self.locator.strip():
            raise ValueError("locator must be non-empty")
        if self.content_hash is not None and not self.content_hash.strip():
            raise ValueError("content_hash must be non-empty when provided")

    def matches(self, observed: EvidenceLocation) -> bool:
        """Return whether an observed location satisfies this gold location."""
        if (
            self.document_id != observed.document_id
            or self.source_uri != observed.source_uri
            or self.locator != observed.locator
        ):
            return False
        return self.content_hash is None or self.content_hash == observed.content_hash


@dataclass(frozen=True)
class CompleteEvidenceBundle:
    """One complete accepted route to answering a query."""

    bundle_id: str
    locations: tuple[EvidenceLocation, ...]

    def __post_init__(self) -> None:
        if not self.bundle_id.strip():
            raise ValueError("bundle_id must be non-empty")
        locations = tuple(self.locations)
        object.__setattr__(self, "locations", locations)
        if not locations:
            raise ValueError("an evidence bundle must contain at least one location")
        if len(set(locations)) != len(locations):
            raise ValueError("an evidence bundle cannot repeat a location")


@dataclass(frozen=True)
class GoldEvidence:
    """All accepted alternative complete evidence bundles for one query."""

    query_id: str
    alternatives: tuple[CompleteEvidenceBundle, ...]

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must be non-empty")
        alternatives = tuple(self.alternatives)
        object.__setattr__(self, "alternatives", alternatives)
        if not alternatives:
            raise ValueError("gold evidence must define at least one accepted bundle")
        bundle_ids = [bundle.bundle_id for bundle in alternatives]
        if len(set(bundle_ids)) != len(bundle_ids):
            raise ValueError("gold evidence bundle IDs must be unique")


@dataclass(frozen=True)
class EvidenceAssessment:
    """Evidence availability and retrieval sufficiency for one trial."""

    authorized_solution_exists: bool
    evidence_sufficient: bool
    authorized_bundle_ids: tuple[str, ...]
    complete_bundle_ids: tuple[str, ...]


@dataclass(frozen=True)
class AnswerOutcomes:
    """Emission-policy outcomes kept separate from answer correctness."""

    answered: bool
    evidence_supported_emission: bool
    false_permit: bool
    false_denial: bool


def assess_evidence(
    gold: GoldEvidence,
    returned: Iterable[EvidenceLocation],
    authorized_document_ids: Iterable[int],
) -> EvidenceAssessment:
    """Score returned provenance against alternative authorized gold bundles."""
    observed = tuple(returned)
    if len(set(observed)) != len(observed):
        raise ValueError("returned evidence cannot repeat a location")
    raw_authorized = tuple(authorized_document_ids)
    if any(type(document_id) is not int or document_id < 0 for document_id in raw_authorized):
        raise ValueError("authorized document IDs must be non-negative integers")
    authorized = frozenset(raw_authorized)
    authorized_bundle_ids: list[str] = []
    complete_bundle_ids: list[str] = []

    for bundle in gold.alternatives:
        bundle_is_authorized = all(
            location.document_id in authorized for location in bundle.locations
        )
        if not bundle_is_authorized:
            continue
        authorized_bundle_ids.append(bundle.bundle_id)
        bundle_is_complete = all(
            any(required.matches(candidate) for candidate in observed)
            for required in bundle.locations
        )
        if bundle_is_complete:
            complete_bundle_ids.append(bundle.bundle_id)

    return EvidenceAssessment(
        authorized_solution_exists=bool(authorized_bundle_ids),
        evidence_sufficient=bool(complete_bundle_ids),
        authorized_bundle_ids=tuple(authorized_bundle_ids),
        complete_bundle_ids=tuple(complete_bundle_ids),
    )


def evaluate_answer(assessment: EvidenceAssessment, *, answered: bool) -> AnswerOutcomes:
    """Classify emission or abstention from evidence sufficiency alone.

    This does not score answer correctness, entailment, citation faithfulness,
    or generated-text leakage. It therefore makes no general safety claim.
    """
    sufficient = assessment.evidence_sufficient
    return AnswerOutcomes(
        answered=answered,
        evidence_supported_emission=answered and sufficient,
        false_permit=answered and not sufficient,
        false_denial=not answered and sufficient,
    )
