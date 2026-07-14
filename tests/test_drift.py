from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from fractal_ann_diagnostics.drift import (
    CorpusRecord,
    CorpusSnapshot,
    EmbeddingMigration,
    EmbeddingRevisionBinding,
    PolicyMutation,
    apply_policy_mutations,
    build_embedding_migration,
    diff_corpus_snapshots,
)
from fractal_ann_diagnostics.policy import (
    AuthorizationPolicy,
    policy_document_universe_sha256,
)


def _snapshot(
    snapshot_id: str,
    hour: int,
    records: tuple[CorpusRecord, ...],
) -> CorpusSnapshot:
    return CorpusSnapshot(
        snapshot_id=snapshot_id,
        observed_at=datetime(2026, 1, 1, hour, tzinfo=timezone.utc),
        records=records,
    )


def test_corpus_diff_separates_source_change_from_rechunking() -> None:
    before = _snapshot(
        "s1",
        1,
        (
            CorpusRecord("a", "h1", "chunk-v1"),
            CorpusRecord("b", "h2", "chunk-v1"),
            CorpusRecord("c", "h3", "chunk-v1"),
        ),
    )
    after = _snapshot(
        "s2",
        2,
        (
            CorpusRecord("a", "h1-new", "chunk-v1"),
            CorpusRecord("b", "h2", "chunk-v2"),
            CorpusRecord("d", "h4", "chunk-v1"),
        ),
    )
    observed = diff_corpus_snapshots(before, after)
    assert observed.added_ids == ("d",)
    assert observed.removed_ids == ("c",)
    assert observed.content_changed_ids == ("a",)
    assert observed.rechunked_ids == ("b",)
    assert observed.affected_fraction == 1.0


def test_embedding_migration_is_deterministic_and_truth_is_fully_current() -> None:
    old = np.zeros((8, 3), dtype=np.float32)
    new = np.ones((8, 3), dtype=np.float32)
    identifiers = [f"d-{index}" for index in range(8)]
    first = build_embedding_migration(
        old,
        new,
        identifiers,
        old_document_revision="documents-old",
        new_document_revision="documents-new",
        old_query_revision="queries-old",
        new_query_revision="queries-new",
        migrated_fraction=0.5,
        seed=17,
    )
    second = build_embedding_migration(
        old,
        new,
        identifiers,
        old_document_revision="documents-old",
        new_document_revision="documents-new",
        old_query_revision="queries-old",
        new_query_revision="queries-new",
        migrated_fraction=0.5,
        seed=17,
    )
    assert np.array_equal(first.migrated_mask, second.migrated_mask)
    assert first.migrated_mask.sum() == 4
    assert np.all(first.active_vectors[first.migrated_mask] == 1)
    assert np.all(first.active_vectors[~first.migrated_mask] == 0)
    assert np.all(first.current_truth_vectors == 1)
    assert first.old_binding == EmbeddingRevisionBinding(
        document_revision="documents-old",
        query_revision="queries-old",
    )
    assert first.new_binding == EmbeddingRevisionBinding(
        document_revision="documents-new",
        query_revision="queries-new",
    )
    assert not first.active_vectors.flags.writeable
    assert not first.current_truth_vectors.flags.writeable
    assert not first.migrated_mask.flags.writeable


def _direct_migration(**overrides: object) -> EmbeddingMigration:
    values: dict[str, object] = {
        "old_binding": EmbeddingRevisionBinding("documents-old", "queries-old"),
        "new_binding": EmbeddingRevisionBinding("documents-new", "queries-new"),
        "migrated_fraction": 0.5,
        "migrated_mask": np.asarray([True, False], dtype=bool),
        "active_vectors": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "current_truth_vectors": np.ones((2, 2), dtype=np.float32),
    }
    values.update(overrides)
    return EmbeddingMigration(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("fraction", [float("nan"), -0.1, 1.1, 0.25])
def test_embedding_migration_rejects_invalid_or_inconsistent_fraction(
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match="migrated_fraction"):
        _direct_migration(migrated_fraction=fraction)


@pytest.mark.parametrize(
    ("old_binding", "new_binding", "message"),
    [
        (
            EmbeddingRevisionBinding("documents-one", "queries-old"),
            EmbeddingRevisionBinding("documents-one", "queries-new"),
            "document revisions",
        ),
        (
            EmbeddingRevisionBinding("documents-old", "queries-one"),
            EmbeddingRevisionBinding("documents-new", "queries-one"),
            "query revisions",
        ),
    ],
)
def test_embedding_migration_requires_distinct_query_and_document_revisions(
    old_binding: EmbeddingRevisionBinding,
    new_binding: EmbeddingRevisionBinding,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _direct_migration(old_binding=old_binding, new_binding=new_binding)


@pytest.mark.parametrize(
    "binding",
    [
        ("", "queries-old"),
        ("documents-old", " "),
    ],
)
def test_embedding_revision_binding_rejects_blank_identifiers(
    binding: tuple[str, str],
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        EmbeddingRevisionBinding(*binding)


def test_embedding_migration_rejects_non_boolean_mask_and_non_finite_vectors() -> None:
    with pytest.raises(ValueError, match="booleans"):
        _direct_migration(migrated_mask=np.asarray([1, 0], dtype=np.int8))

    active = np.asarray([[float("nan"), 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        _direct_migration(active_vectors=active)

    truth = np.asarray([[1.0, 1.0], [1.0, float("inf")]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        _direct_migration(current_truth_vectors=truth)


def test_embedding_migration_owns_immutable_array_snapshots() -> None:
    mask = np.asarray([True, False], dtype=bool)
    active = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    truth = np.ones((2, 2), dtype=np.float32)
    migration = _direct_migration(
        migrated_mask=mask,
        active_vectors=active,
        current_truth_vectors=truth,
    )

    mask[:] = False
    active[:] = 99.0
    truth[:] = 99.0

    assert migration.migrated_mask.tolist() == [True, False]
    np.testing.assert_array_equal(
        migration.active_vectors,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(migration.current_truth_vectors, np.ones((2, 2)))


def test_embedding_migration_builder_rejects_non_finite_source_vectors() -> None:
    old = np.zeros((2, 2), dtype=np.float32)
    new = np.ones((2, 2), dtype=np.float32)
    old[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_embedding_migration(
            old,
            new,
            ("a", "b"),
            old_document_revision="documents-old",
            new_document_revision="documents-new",
            old_query_revision="queries-old",
            new_query_revision="queries-new",
            migrated_fraction=0.5,
            seed=1,
        )


def test_policy_drift_requires_new_version_and_applies_grant_and_revoke() -> None:
    universe_digest = policy_document_universe_sha256(
        ("stable-document-a", "stable-document-b", "stable-document-c")
    )
    policy = AuthorizationPolicy(
        roles=("reader",),
        visibility=np.asarray([[True, False, True]], dtype=bool),
        version="v1",
        document_universe_sha256=universe_digest,
    )
    changed = apply_policy_mutations(
        policy,
        (
            PolicyMutation("reader", 0, "revoke"),
            PolicyMutation("reader", 1, "grant"),
        ),
        version="v2",
    )
    assert changed.authorized_ids("reader").tolist() == [1, 2]
    assert changed.document_universe_sha256 == universe_digest
    assert policy.authorized_ids("reader").tolist() == [0, 2]
    with pytest.raises(ValueError, match="distinct"):
        apply_policy_mutations(policy, (), version="v1")


def test_policy_drift_rejects_duplicate_target() -> None:
    policy = AuthorizationPolicy(
        roles=("reader",),
        visibility=np.asarray([[True]], dtype=bool),
        version="v1",
    )
    with pytest.raises(ValueError, match="multiple mutations"):
        apply_policy_mutations(
            policy,
            (
                PolicyMutation("reader", 0, "grant"),
                PolicyMutation("reader", 0, "revoke"),
            ),
            version="v2",
        )


@pytest.mark.parametrize("action", ["typo", "", "GRANT"])
def test_policy_mutation_rejects_unknown_actions(action: str) -> None:
    with pytest.raises(ValueError, match="grant.*revoke"):
        PolicyMutation("reader", 0, action)  # type: ignore[arg-type]
