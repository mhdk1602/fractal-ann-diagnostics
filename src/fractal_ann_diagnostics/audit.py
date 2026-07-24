"""Privacy-minimized integrity records for governed retrieval.

The record chain detects changes relative to a trusted chain head or expected
length. It does not provide immutable storage, signatures, or protection when
an attacker can rewrite the full chain and its external anchor.
"""

from __future__ import annotations

import hmac
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .artifact_integrity import ArtifactVerificationReceipt
from .controller import GovernedResult
from .corpora import NormalizedCorpus
from .policy import PolicyDecision, policy_document_universe_sha256
from .retrieval import SearchWork

AUDIT_SCHEMA_VERSION = "fractalguard-audit-v2"
GENESIS_RECORD_SHA256 = sha256(b"fractalguard-audit-chain-v1").hexdigest()
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_COMPONENTS = {
    "application",
    "controller",
    "corpus",
    "embedding",
    "index",
    "policy",
}
_ALLOWED_COMPONENTS = _REQUIRED_COMPONENTS | {
    "evaluator",
    "manifest",
    "normalizer",
    "runner",
}


@runtime_checkable
class AdmittedProvenanceRegistry(Protocol):
    """Digest-only registry surface admitted by the sealed orchestrator."""

    corpus_name: str
    corpus_stage: str
    document_count: int
    document_universe_sha256: str
    verification_receipt_sha256: str
    component_revisions: tuple[tuple[str, str], ...]

    def content_sha256(self, document_id: int) -> str: ...


def _require_nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_nonnegative_finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    return sha256(data).hexdigest()


def _normalize_timestamp(value: datetime | str | None) -> str:
    if value is None:
        instant = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        instant = value
    else:
        _require_nonempty("occurred_at", value)
        try:
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("occurred_at must be an ISO 8601 timestamp") from exc
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("occurred_at must include a timezone")
    return (
        instant.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def pseudonymize_subject(subject: str, *, key: bytes, key_id: str) -> str:
    """Return a scoped HMAC pseudonym without retaining the raw subject."""
    _require_nonempty("subject", subject)
    _require_nonempty("key_id", key_id)
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("pseudonym key must contain at least 32 bytes")
    scoped_subject = key_id.encode("utf-8") + b"\x00" + subject.encode("utf-8")
    return hmac.new(key, scoped_subject, sha256).hexdigest()


@dataclass(frozen=True)
class AuthorizationAudit:
    """Identifiers and a mask digest for one policy decision."""

    decision_id: str
    action: str
    policy_revision: str
    mask_sha256: str
    mask_size: int
    available: bool
    environment_sha256: str
    document_universe_sha256: str
    request_nonce: str
    request_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty("decision_id", self.decision_id)
        _require_nonempty("action", self.action)
        _require_nonempty("policy_revision", self.policy_revision)
        _require_sha256("mask_sha256", self.mask_sha256)
        _require_sha256("environment_sha256", self.environment_sha256)
        _require_sha256("document_universe_sha256", self.document_universe_sha256)
        _require_nonempty("request_nonce", self.request_nonce)
        _require_sha256("request_sha256", self.request_sha256)
        if self.mask_size < 0:
            raise ValueError("mask_size must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "available": self.available,
            "decision_id": self.decision_id,
            "document_universe_sha256": self.document_universe_sha256,
            "environment_sha256": self.environment_sha256,
            "mask_sha256": self.mask_sha256,
            "mask_size": self.mask_size,
            "policy_revision": self.policy_revision,
            "request_nonce": self.request_nonce,
            "request_sha256": self.request_sha256,
        }


@dataclass(frozen=True)
class WorkAudit:
    """Observed latency and backend work, distinct from configured effort."""

    latency_ms: float
    returned_candidates: int
    visited_candidates: int | None
    distance_evaluations: int | None
    configured_ef_search: int | None

    def __post_init__(self) -> None:
        _require_nonnegative_finite("latency_ms", self.latency_ms)
        for name in (
            "returned_candidates",
            "visited_candidates",
            "distance_evaluations",
            "configured_ef_search",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured_ef_search": self.configured_ef_search,
            "distance_evaluations": self.distance_evaluations,
            "latency_ms": self.latency_ms,
            "returned_candidates": self.returned_candidates,
            "visited_candidates": self.visited_candidates,
        }


@dataclass(frozen=True)
class IndexRefreshAudit:
    """Policy-bound index refresh work for one request."""

    policy_revision: str
    mask_sha256: str
    rebuilt: bool
    latency_ms: float
    authorized_count: int

    def __post_init__(self) -> None:
        _require_nonempty("policy_revision", self.policy_revision)
        _require_sha256("mask_sha256", self.mask_sha256)
        _require_nonnegative_finite("latency_ms", self.latency_ms)
        if self.authorized_count < 0:
            raise ValueError("authorized_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized_count": self.authorized_count,
            "latency_ms": self.latency_ms,
            "mask_sha256": self.mask_sha256,
            "policy_revision": self.policy_revision,
            "rebuilt": self.rebuilt,
        }


@dataclass(frozen=True)
class EvidenceAudit:
    """One emitted evidence identifier and precomputed content digest."""

    document_id: int
    content_sha256: str

    def __post_init__(self) -> None:
        if self.document_id < 0:
            raise ValueError("document_id must be non-negative")
        _require_sha256("content_sha256", self.content_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "document_id": self.document_id,
        }


@dataclass(frozen=True, init=False)
class VerifiedProvenanceRegistry:
    """Digest-only audit provenance derived from verified local artifacts.

    Construction accepts normalized corpus data, a closed artifact-verification
    receipt, and component-to-artifact identifiers. It deliberately accepts no
    caller-supplied digest. The registry retains document content hashes but no
    document text, source URI, external identifier, query, or label.
    """

    corpus_name: str
    corpus_stage: str
    document_count: int
    document_universe_sha256: str
    verification_receipt_sha256: str
    component_revisions: tuple[tuple[str, str], ...]
    _document_content_sha256: tuple[str, ...]

    def __init__(
        self,
        *,
        corpus: NormalizedCorpus,
        verification_receipt: ArtifactVerificationReceipt,
        component_artifact_ids: Mapping[str, str],
    ) -> None:
        if not isinstance(corpus, NormalizedCorpus):
            raise TypeError("corpus must be a NormalizedCorpus")
        if not isinstance(verification_receipt, ArtifactVerificationReceipt):
            raise TypeError("verification_receipt must be an ArtifactVerificationReceipt")
        if not isinstance(component_artifact_ids, Mapping):
            raise TypeError("component_artifact_ids must be a mapping")

        binding = dict(component_artifact_ids)
        if any(not isinstance(name, str) for name in binding):
            raise ValueError("component names must be strings")
        if any(
            not isinstance(artifact_id, str) or not artifact_id.strip()
            for artifact_id in binding.values()
        ):
            raise ValueError("component artifact IDs must be non-empty strings")
        component_names = set(binding)
        missing_components = _REQUIRED_COMPONENTS - component_names
        unknown_components = component_names - _ALLOWED_COMPONENTS
        if missing_components:
            raise ValueError(f"component_artifact_ids are missing {sorted(missing_components)}")
        if unknown_components:
            raise ValueError(
                f"component_artifact_ids contain unknown names {sorted(unknown_components)}"
            )
        if len(set(binding.values())) != len(binding):
            raise ValueError("each component must bind a distinct artifact ID")

        artifacts_by_id = {
            artifact.artifact_id: artifact for artifact in verification_receipt.artifacts
        }
        unknown_artifacts = set(binding.values()) - set(artifacts_by_id)
        if unknown_artifacts:
            raise ValueError(
                f"component_artifact_ids name unverified artifacts {sorted(unknown_artifacts)}"
            )
        non_exact = sorted(
            artifact_id
            for artifact_id in binding.values()
            if not artifacts_by_id[artifact_id].exact
        )
        if non_exact:
            raise ValueError(
                f"audit components require exact verified artifacts; non-exact IDs are {non_exact}"
            )

        document_identities = tuple(
            _canonical_bytes(
                {
                    "content_hash": document.content_hash,
                    "document_id": document.document_id,
                    "external_id": document.external_id,
                    "source_uri": document.source_uri,
                }
            ).decode("utf-8")
            for document in corpus.documents
        )
        document_universe_sha256 = policy_document_universe_sha256(document_identities)
        content_digests = tuple(
            document.content_hash.removeprefix("sha256:") for document in corpus.documents
        )
        component_revisions = tuple(
            sorted(
                (
                    name,
                    artifacts_by_id[artifact_id].verified_sha256,
                )
                for name, artifact_id in binding.items()
            )
        )

        object.__setattr__(self, "corpus_name", corpus.name)
        object.__setattr__(self, "corpus_stage", corpus.stage)
        object.__setattr__(self, "document_count", len(corpus.documents))
        object.__setattr__(self, "document_universe_sha256", document_universe_sha256)
        object.__setattr__(
            self,
            "verification_receipt_sha256",
            verification_receipt.receipt_sha256,
        )
        object.__setattr__(self, "component_revisions", component_revisions)
        object.__setattr__(self, "_document_content_sha256", content_digests)

    def content_sha256(self, document_id: int) -> str:
        """Return a verified-corpus content digest for one positional ID."""
        if type(document_id) is not int or not 0 <= document_id < self.document_count:
            raise ValueError("returned document ID is outside the verified corpus")
        return self._document_content_sha256[document_id]


@dataclass(frozen=True)
class AuditRecord:
    """Canonical, self-hashed record linked to its predecessor."""

    schema_version: str
    sequence: int
    occurred_at: str
    request_id: str
    trace_id: str
    trial_sha256: str
    subject_pseudonym: str
    pseudonym_key_id: str
    provenance_receipt_sha256: str
    initial_authorization: AuthorizationAudit | None
    final_authorization: AuthorizationAudit | None
    component_revisions: tuple[tuple[str, str], ...]
    returned_evidence: tuple[EvidenceAudit, ...]
    controller_action: str
    controller_reasons: tuple[str, ...]
    controller_risk_score: float
    controller_policy_revision: str
    authorization_latency_ms: float
    controller_latency_ms: float
    probe_work: WorkAudit | None
    geometry_feature_latency_ms: float | None
    search_strategy: str | None
    search_work: WorkAudit | None
    search_reused_probe: bool
    index_refresh: IndexRefreshAudit | None
    total_online_latency_ms: float
    abstained: bool
    output_emitted: bool
    output_sha256: str | None
    previous_record_sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty("schema_version", self.schema_version)
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        _normalize_timestamp(self.occurred_at)
        _require_nonempty("request_id", self.request_id)
        _require_nonempty("trace_id", self.trace_id)
        _require_sha256("trial_sha256", self.trial_sha256)
        _require_sha256("subject_pseudonym", self.subject_pseudonym)
        _require_nonempty("pseudonym_key_id", self.pseudonym_key_id)
        _require_sha256("provenance_receipt_sha256", self.provenance_receipt_sha256)
        _require_nonempty("controller_action", self.controller_action)
        _require_nonempty("controller_policy_revision", self.controller_policy_revision)
        reasons_are_empty = any(not reason.strip() for reason in self.controller_reasons)
        if not self.controller_reasons or reasons_are_empty:
            raise ValueError("controller_reasons must contain non-empty strings")
        if not math.isfinite(self.controller_risk_score):
            raise ValueError("controller_risk_score must be finite")
        _require_nonnegative_finite("authorization_latency_ms", self.authorization_latency_ms)
        _require_nonnegative_finite("controller_latency_ms", self.controller_latency_ms)
        _require_nonnegative_finite("total_online_latency_ms", self.total_online_latency_ms)
        if self.geometry_feature_latency_ms is not None:
            _require_nonnegative_finite(
                "geometry_feature_latency_ms", self.geometry_feature_latency_ms
            )
        if not self.component_revisions:
            raise ValueError("component_revisions must not be empty")
        if tuple(sorted(self.component_revisions)) != self.component_revisions:
            raise ValueError("component_revisions must be sorted")
        component_names = [name for name, _ in self.component_revisions]
        if len(set(component_names)) != len(component_names):
            raise ValueError("component revision names must be unique")
        component_name_set = set(component_names)
        missing_components = _REQUIRED_COMPONENTS - component_name_set
        unknown_components = component_name_set - _ALLOWED_COMPONENTS
        if missing_components:
            raise ValueError(f"component_revisions are missing {sorted(missing_components)}")
        if unknown_components:
            raise ValueError(
                f"component_revisions contain unknown names {sorted(unknown_components)}"
            )
        for name, revision in self.component_revisions:
            _require_nonempty("component name", name)
            _require_sha256("component digest", revision)
        evidence_ids = [item.document_id for item in self.returned_evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("returned evidence IDs must be unique")
        if self.abstained and self.returned_evidence:
            raise ValueError("an abstention cannot emit evidence")
        if self.abstained and self.output_emitted:
            raise ValueError("an abstention cannot emit output")
        if self.output_emitted != (self.output_sha256 is not None):
            raise ValueError("output_emitted must match output_sha256 presence")
        if self.output_sha256 is not None:
            _require_sha256("output_sha256", self.output_sha256)
        _require_sha256("previous_record_sha256", self.previous_record_sha256)
        _require_sha256("record_sha256", self.record_sha256)

    def hash_payload(self) -> dict[str, Any]:
        """Return all persisted fields except the record's self-hash."""
        payload = self.to_dict()
        del payload["record_sha256"]
        return payload

    def canonical_bytes(self, *, include_record_hash: bool = True) -> bytes:
        """Serialize with sorted keys, UTF-8, and no insignificant whitespace."""
        payload = self.to_dict() if include_record_hash else self.hash_payload()
        return _canonical_bytes(payload)

    def computed_record_sha256(self) -> str:
        return sha256(self.canonical_bytes(include_record_hash=False)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstained": self.abstained,
            "component_revisions": [
                {"component": name, "revision": revision}
                for name, revision in self.component_revisions
            ],
            "controller_action": self.controller_action,
            "authorization_latency_ms": self.authorization_latency_ms,
            "controller_latency_ms": self.controller_latency_ms,
            "controller_policy_revision": self.controller_policy_revision,
            "controller_reasons": list(self.controller_reasons),
            "controller_risk_score": self.controller_risk_score,
            "final_authorization": (
                None if self.final_authorization is None else self.final_authorization.to_dict()
            ),
            "geometry_feature_latency_ms": self.geometry_feature_latency_ms,
            "index_refresh": None if self.index_refresh is None else self.index_refresh.to_dict(),
            "initial_authorization": (
                None if self.initial_authorization is None else self.initial_authorization.to_dict()
            ),
            "occurred_at": self.occurred_at,
            "output_emitted": self.output_emitted,
            "output_sha256": self.output_sha256,
            "previous_record_sha256": self.previous_record_sha256,
            "probe_work": None if self.probe_work is None else self.probe_work.to_dict(),
            "provenance_receipt_sha256": self.provenance_receipt_sha256,
            "pseudonym_key_id": self.pseudonym_key_id,
            "record_sha256": self.record_sha256,
            "request_id": self.request_id,
            "returned_evidence": [item.to_dict() for item in self.returned_evidence],
            "schema_version": self.schema_version,
            "search_reused_probe": self.search_reused_probe,
            "search_strategy": self.search_strategy,
            "search_work": None if self.search_work is None else self.search_work.to_dict(),
            "sequence": self.sequence,
            "subject_pseudonym": self.subject_pseudonym,
            "total_online_latency_ms": self.total_online_latency_ms,
            "trace_id": self.trace_id,
            "trial_sha256": self.trial_sha256,
        }


@dataclass(frozen=True)
class ChainVerification:
    """Result of checking sequence, links, self-hashes, and optional anchors."""

    valid: bool
    checked_records: int
    head_sha256: str
    errors: tuple[str, ...]


def _authorization_audit(decision: PolicyDecision | None) -> AuthorizationAudit | None:
    if decision is None:
        return None
    mask = np.asarray(decision.authorized_mask, dtype=bool)
    return AuthorizationAudit(
        decision_id=decision.decision_id,
        action=decision.action,
        policy_revision=decision.policy_version,
        mask_sha256=sha256(mask.tobytes(order="C")).hexdigest(),
        mask_size=int(mask.size),
        available=bool(decision.available),
        environment_sha256=decision.environment_sha256,
        document_universe_sha256=decision.document_universe_sha256,
        request_nonce=decision.request_nonce,
        request_sha256=decision.request_sha256,
    )


def _work_audit(latency_ms: float, work: SearchWork | None, returned: int) -> WorkAudit:
    observed = work or SearchWork(returned_candidates=returned)
    return WorkAudit(
        latency_ms=float(latency_ms),
        returned_candidates=observed.returned_candidates,
        visited_candidates=observed.visited_candidates,
        distance_evaluations=observed.distance_evaluations,
        configured_ef_search=observed.configured_ef_search,
    )


def _evidence_audit(
    result: GovernedResult,
    provenance_registry: AdmittedProvenanceRegistry,
) -> tuple[EvidenceAudit, ...]:
    if result.search is None:
        return ()

    returned_ids = tuple(int(document_id) for document_id in result.search.ids)
    if len(set(returned_ids)) != len(returned_ids):
        raise ValueError("search result contains duplicate document IDs")
    return tuple(
        EvidenceAudit(
            document_id=document_id,
            content_sha256=provenance_registry.content_sha256(document_id),
        )
        for document_id in returned_ids
    )


def audit_record_from_governed_result(
    result: GovernedResult,
    *,
    request_id: str,
    trace_id: str,
    trial_sha256: str,
    subject: str,
    pseudonym_key: bytes,
    pseudonym_key_id: str,
    provenance_registry: AdmittedProvenanceRegistry,
    output: str | bytes | None = None,
    occurred_at: datetime | str | None = None,
    previous_record: AuditRecord | None = None,
) -> AuditRecord:
    """Create a privacy-minimized audit record from a governed result.

    Evidence and component digests come only from ``provenance_registry``. Query
    text, vectors, distances, policy masks, raw subjects, and generated output
    are not persisted. ``output`` is consumed only to compute its SHA-256 digest.
    """
    _require_nonempty("request_id", request_id)
    _require_nonempty("trace_id", trace_id)
    _require_sha256("trial_sha256", trial_sha256)
    _require_nonempty("subject", subject)
    _require_nonempty("pseudonym_key_id", pseudonym_key_id)
    if not isinstance(provenance_registry, AdmittedProvenanceRegistry):
        raise TypeError("provenance_registry lacks the admitted digest-only interface")

    for authorization in (result.initial_authorization, result.final_authorization):
        if authorization is None:
            continue
        if authorization.subject != subject:
            raise ValueError("policy decision subject does not match the audited subject")
        if authorization.document_universe_sha256 != provenance_registry.document_universe_sha256:
            raise ValueError("policy decision document universe does not match verified provenance")
        if authorization.authorized_mask.size != provenance_registry.document_count:
            raise ValueError("policy decision mask size does not match the verified corpus")

    abstained = result.decision.action == "abstain"
    if abstained and result.search is not None:
        raise ValueError("an abstention cannot contain a search result")
    if not abstained and result.search is None:
        raise ValueError("a non-abstaining result must contain a search result")
    if not abstained and result.final_authorization is None:
        raise ValueError("a non-abstaining result requires final authorization")
    if abstained and output is not None:
        raise ValueError("an abstention cannot emit output")
    if not abstained:
        initial = result.initial_authorization
        final = result.final_authorization
        if initial is None or final is None:
            raise ValueError("an emitted result requires initial and final authorization")
        if not initial.available or not final.available:
            raise ValueError("an emitted result requires available authorization decisions")
        if initial.action != final.action:
            raise ValueError("initial and final authorization actions must match")
        if initial.environment_sha256 != final.environment_sha256:
            raise ValueError("initial and final authorization environments must match")
        if initial.document_universe_sha256 != final.document_universe_sha256:
            raise ValueError("initial and final document universes must match")
        if initial.policy_version != final.policy_version or not np.array_equal(
            initial.authorized_mask,
            final.authorized_mask,
        ):
            raise ValueError("initial and final authorization snapshots must match")
        if result.decision.policy_version != final.policy_version:
            raise ValueError("controller and final policy revisions must match")
        if result.search is None or not final.permits(result.search.ids):
            raise ValueError("final authorization does not permit every returned document")
        if result.search.unauthorized_candidates or result.search.unauthorized_context:
            raise ValueError("a governed result cannot contain unauthorized material")

    if previous_record is None:
        sequence = 0
        previous_hash = GENESIS_RECORD_SHA256
    else:
        if previous_record.computed_record_sha256() != previous_record.record_sha256:
            raise ValueError("previous record self-hash is invalid")
        sequence = previous_record.sequence + 1
        previous_hash = previous_record.record_sha256

    initial_authorization = _authorization_audit(result.initial_authorization)
    final_authorization = _authorization_audit(result.final_authorization)
    evidence = _evidence_audit(result, provenance_registry)

    probe_work = None
    if result.probe is not None:
        probe_work = _work_audit(
            result.probe.search_latency_ms,
            result.probe.work,
            len(result.probe.ids),
        )

    search_work = None
    search_strategy = None
    if result.search is not None:
        search_strategy = result.search.strategy
        search_work = _work_audit(
            result.search.latency_ms,
            result.search.work,
            len(result.search.ids),
        )

    index_refresh = None
    if result.index_refresh is not None:
        index_refresh = IndexRefreshAudit(
            policy_revision=result.index_refresh.policy_version,
            mask_sha256=result.index_refresh.mask_sha256,
            rebuilt=result.index_refresh.rebuilt,
            latency_ms=result.index_refresh.latency_ms,
            authorized_count=result.index_refresh.authorized_count,
        )

    output_sha256 = None if output is None else _sha256_bytes(output)
    placeholder_hash = "0" * 64
    record = AuditRecord(
        schema_version=AUDIT_SCHEMA_VERSION,
        sequence=sequence,
        occurred_at=_normalize_timestamp(occurred_at),
        request_id=request_id,
        trace_id=trace_id,
        trial_sha256=trial_sha256,
        subject_pseudonym=pseudonymize_subject(
            subject,
            key=pseudonym_key,
            key_id=pseudonym_key_id,
        ),
        pseudonym_key_id=pseudonym_key_id,
        provenance_receipt_sha256=(provenance_registry.verification_receipt_sha256),
        initial_authorization=initial_authorization,
        final_authorization=final_authorization,
        component_revisions=provenance_registry.component_revisions,
        returned_evidence=evidence,
        controller_action=result.decision.action,
        controller_reasons=result.decision.reasons,
        controller_risk_score=result.decision.risk_score,
        controller_policy_revision=result.decision.policy_version,
        authorization_latency_ms=result.authorization_latency_ms,
        controller_latency_ms=result.controller_latency_ms,
        probe_work=probe_work,
        geometry_feature_latency_ms=(
            None if result.geometry is None else result.geometry.feature_latency_ms
        ),
        search_strategy=search_strategy,
        search_work=search_work,
        search_reused_probe=(
            result.probe is not None
            and result.search is not None
            and result.search.strategy == "hnsw-low"
        ),
        index_refresh=index_refresh,
        total_online_latency_ms=result.total_online_latency_ms,
        abstained=abstained,
        output_emitted=output is not None,
        output_sha256=output_sha256,
        previous_record_sha256=previous_hash,
        record_sha256=placeholder_hash,
    )
    return AuditRecord(**{**record.__dict__, "record_sha256": record.computed_record_sha256()})


def verify_audit_chain(
    records: tuple[AuditRecord, ...] | list[AuditRecord],
    *,
    expected_head_sha256: str | None = None,
    expected_length: int | None = None,
) -> ChainVerification:
    """Verify record order, links, self-hashes, and optional trusted anchors.

    A trusted head or length is required to detect deletion from the end of a
    chain. Middle deletion and reordering break sequence or predecessor links.
    """
    if expected_head_sha256 is not None:
        _require_sha256("expected_head_sha256", expected_head_sha256)
    if expected_length is not None and (type(expected_length) is not int or expected_length < 0):
        raise ValueError("expected_length must be a non-negative integer")

    errors: list[str] = []
    previous_hash = GENESIS_RECORD_SHA256
    for position, record in enumerate(records):
        if record.sequence != position:
            errors.append(f"record {position}: sequence is {record.sequence}, expected {position}")
        if record.previous_record_sha256 != previous_hash:
            errors.append(f"record {position}: previous-record hash mismatch")
        computed_hash = record.computed_record_sha256()
        if record.record_sha256 != computed_hash:
            errors.append(f"record {position}: record hash mismatch")
        previous_hash = record.record_sha256

    if expected_length is not None and len(records) != expected_length:
        errors.append(f"chain length is {len(records)}, expected {expected_length}")
    if expected_head_sha256 is not None and previous_hash != expected_head_sha256:
        errors.append("chain head does not match the trusted head")
    return ChainVerification(
        valid=not errors,
        checked_records=len(records),
        head_sha256=previous_hash,
        errors=tuple(errors),
    )
