"""Versioned corpus, embedding, and policy drift interventions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal

import numpy as np

from .policy import AuthorizationPolicy


@dataclass(frozen=True)
class CorpusRecord:
    external_id: str
    content_hash: str
    chunking_revision: str

    def __post_init__(self) -> None:
        if not self.external_id or not self.content_hash or not self.chunking_revision:
            raise ValueError("corpus record fields must be non-empty")


@dataclass(frozen=True)
class CorpusSnapshot:
    snapshot_id: str
    observed_at: datetime
    records: tuple[CorpusRecord, ...]

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        identifiers = [record.external_id for record in self.records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("corpus snapshot external IDs must be unique")


@dataclass(frozen=True)
class CorpusSnapshotDiff:
    previous_snapshot: str
    current_snapshot: str
    added_ids: tuple[str, ...]
    removed_ids: tuple[str, ...]
    content_changed_ids: tuple[str, ...]
    rechunked_ids: tuple[str, ...]
    previous_count: int
    current_count: int

    @property
    def affected_fraction(self) -> float:
        denominator = max(self.previous_count + len(self.added_ids), 1)
        affected = set(self.added_ids)
        affected.update(self.removed_ids)
        affected.update(self.content_changed_ids)
        affected.update(self.rechunked_ids)
        return len(affected) / denominator


def diff_corpus_snapshots(
    previous: CorpusSnapshot,
    current: CorpusSnapshot,
) -> CorpusSnapshotDiff:
    """Measure genuine source, deletion, and chunking changes between snapshots."""
    if current.observed_at <= previous.observed_at:
        raise ValueError("current snapshot must be observed after previous snapshot")
    before = {record.external_id: record for record in previous.records}
    after = {record.external_id: record for record in current.records}
    shared = set(before).intersection(after)
    return CorpusSnapshotDiff(
        previous_snapshot=previous.snapshot_id,
        current_snapshot=current.snapshot_id,
        added_ids=tuple(sorted(set(after) - set(before))),
        removed_ids=tuple(sorted(set(before) - set(after))),
        content_changed_ids=tuple(
            sorted(
                external_id
                for external_id in shared
                if before[external_id].content_hash != after[external_id].content_hash
            )
        ),
        rechunked_ids=tuple(
            sorted(
                external_id
                for external_id in shared
                if before[external_id].chunking_revision != after[external_id].chunking_revision
            )
        ),
        previous_count=len(before),
        current_count=len(after),
    )


@dataclass(frozen=True)
class EmbeddingRevisionBinding:
    """One document/query encoder pair that defines a comparable vector space."""

    document_revision: str
    query_revision: str

    def __post_init__(self) -> None:
        for name in ("document_revision", "query_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class EmbeddingMigration:
    old_binding: EmbeddingRevisionBinding
    new_binding: EmbeddingRevisionBinding
    migrated_fraction: float
    migrated_mask: np.ndarray
    active_vectors: np.ndarray
    current_truth_vectors: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.old_binding, EmbeddingRevisionBinding) or not isinstance(
            self.new_binding, EmbeddingRevisionBinding
        ):
            raise TypeError("old_binding and new_binding must be revision bindings")
        if self.old_binding.document_revision == self.new_binding.document_revision:
            raise ValueError("old and new document revisions must be distinct")
        if self.old_binding.query_revision == self.new_binding.query_revision:
            raise ValueError("old and new query revisions must be distinct")
        if not np.isfinite(self.migrated_fraction) or not (0.0 <= self.migrated_fraction <= 1.0):
            raise ValueError("migrated_fraction must be finite and in [0, 1]")

        raw_mask = np.asarray(self.migrated_mask)
        if raw_mask.dtype != np.bool_:
            raise ValueError("migrated_mask must contain booleans")
        mask = raw_mask.astype(bool, copy=True)
        active = np.array(self.active_vectors, dtype=np.float32, copy=True)
        truth = np.array(self.current_truth_vectors, dtype=np.float32, copy=True)
        if (
            active.ndim != 2
            or len(active) == 0
            or active.shape[1] == 0
            or truth.shape != active.shape
            or mask.shape != (len(active),)
        ):
            raise ValueError("embedding migration arrays have incompatible shapes")
        if not np.all(np.isfinite(active)) or not np.all(np.isfinite(truth)):
            raise ValueError("embedding migration contains non-finite values")
        observed_fraction = float(mask.mean())
        if not np.isclose(
            float(self.migrated_fraction),
            observed_fraction,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("migrated_fraction does not match migrated_mask")
        mask.setflags(write=False)
        active.setflags(write=False)
        truth.setflags(write=False)
        object.__setattr__(self, "migrated_fraction", observed_fraction)
        object.__setattr__(self, "migrated_mask", mask)
        object.__setattr__(self, "active_vectors", active)
        object.__setattr__(self, "current_truth_vectors", truth)


def _migration_order(document_ids: Iterable[str], seed: int) -> list[int]:
    scored: list[tuple[bytes, int]] = []
    for index, document_id in enumerate(document_ids):
        payload = f"{seed}\0{document_id}".encode("utf-8")
        scored.append((hashlib.sha256(payload).digest(), index))
    return [index for _, index in sorted(scored)]


def build_embedding_migration(
    old_vectors: np.ndarray,
    new_vectors: np.ndarray,
    document_ids: Iterable[str],
    *,
    old_document_revision: str,
    new_document_revision: str,
    old_query_revision: str,
    new_query_revision: str,
    migrated_fraction: float,
    seed: int,
) -> EmbeddingMigration:
    """Build a deterministic partial migration while retaining current exact truth."""
    old = np.asarray(old_vectors, dtype=np.float32)
    new = np.asarray(new_vectors, dtype=np.float32)
    identifiers = tuple(str(document_id) for document_id in document_ids)
    if (
        old.ndim != 2
        or len(old) == 0
        or old.shape[1] == 0
        or old.shape != new.shape
        or len(identifiers) != len(old)
    ):
        raise ValueError("old, new, and document IDs must describe the same matrix")
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ValueError("embedding migration contains non-finite values")
    if any(not identifier for identifier in identifiers):
        raise ValueError("document IDs must be non-empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("document IDs must be unique")
    if not np.isfinite(migrated_fraction) or not 0.0 <= migrated_fraction <= 1.0:
        raise ValueError("migrated_fraction must be finite and in [0, 1]")
    old_binding = EmbeddingRevisionBinding(
        document_revision=old_document_revision,
        query_revision=old_query_revision,
    )
    new_binding = EmbeddingRevisionBinding(
        document_revision=new_document_revision,
        query_revision=new_query_revision,
    )
    count = int(round(migrated_fraction * len(old)))
    migrated_mask = np.zeros(len(old), dtype=bool)
    migrated_mask[_migration_order(identifiers, seed)[:count]] = True
    active = old.copy()
    active[migrated_mask] = new[migrated_mask]
    return EmbeddingMigration(
        old_binding=old_binding,
        new_binding=new_binding,
        migrated_fraction=(float(migrated_mask.mean()) if len(old) else 0.0),
        migrated_mask=migrated_mask,
        active_vectors=active,
        current_truth_vectors=new,
    )


PolicyMutationAction = Literal["grant", "revoke"]


@dataclass(frozen=True)
class PolicyMutation:
    role: str
    document_id: int
    action: PolicyMutationAction

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("policy mutation role must be a non-empty string")
        if isinstance(self.document_id, bool) or not isinstance(
            self.document_id, (int, np.integer)
        ):
            raise ValueError("policy mutation document ID must be an integer")
        if self.document_id < 0:
            raise ValueError("policy mutation document ID must be non-negative")
        if self.action not in {"grant", "revoke"}:
            raise ValueError("policy mutation action must be 'grant' or 'revoke'")


def apply_policy_mutations(
    policy: AuthorizationPolicy,
    mutations: Iterable[PolicyMutation],
    *,
    version: str,
) -> AuthorizationPolicy:
    """Apply registered grants or revocations as a new authoritative revision."""
    if not version or version == policy.version:
        raise ValueError("a policy mutation requires a distinct non-empty version")
    visibility = policy.visibility.copy()
    observed: set[tuple[str, int]] = set()
    for mutation in mutations:
        key = (mutation.role, mutation.document_id)
        if key in observed:
            raise ValueError(f"multiple mutations target {key!r} in one revision")
        observed.add(key)
        role_index = policy.role_index(mutation.role)
        if not 0 <= mutation.document_id < policy.n_documents:
            raise ValueError("policy mutation document ID is out of range")
        visibility[role_index, mutation.document_id] = mutation.action == "grant"
    return AuthorizationPolicy(
        roles=policy.roles,
        visibility=visibility,
        version=version,
        document_universe_sha256=policy.document_universe_sha256,
    )
