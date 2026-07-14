from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, RLock, get_ident

import numpy as np
import pytest

from fractal_ann_diagnostics.controller import (
    ControllerConfig,
    ControllerDecision,
    GovernedRetriever,
    RuleController,
)
from fractal_ann_diagnostics.geometry import QueryGeometry
from fractal_ann_diagnostics.policy import (
    AuthorizationPolicy,
    InMemoryPolicyDecisionPoint,
    PolicyDecision,
    policy_document_universe_sha256,
)
from fractal_ann_diagnostics.retrieval import (
    AuthorizedExactIndex,
    AuthorizedHNSWIndex,
    HNSWSearchIndex,
    authorized_exact_search,
    authorized_hnsw_probe,
    authorized_hnsw_search,
    snapshot_query,
    unsafe_unfiltered_search,
)


def _fixture() -> tuple[np.ndarray, AuthorizationPolicy]:
    rng = np.random.default_rng(21)
    public = rng.normal(0, 0.2, size=(40, 8))
    private = rng.normal(3, 0.2, size=(40, 8))
    vectors = np.vstack([public, private]).astype(np.float32)
    visibility = np.asarray(
        [np.r_[np.ones(40, dtype=bool), np.zeros(40, dtype=bool)]],
        dtype=bool,
    )
    universe_digest = policy_document_universe_sha256(
        f"fixture-document-{index}" for index in range(len(vectors))
    )
    return vectors, AuthorizationPolicy(
        roles=("reader",),
        visibility=visibility,
        document_universe_sha256=universe_digest,
    )


class _SequencedDecisionPoint:
    def __init__(
        self,
        decisions: list[PolicyDecision],
        *,
        document_universe_sha256: str | None = None,
    ) -> None:
        self.decisions = decisions
        self.calls = 0
        self._document_universe_sha256 = document_universe_sha256

    @property
    def n_documents(self) -> int:
        return int(self.decisions[0].authorized_mask.size)

    @property
    def document_universe_sha256(self) -> str:
        return (
            self._document_universe_sha256
            or self.decisions[0].document_universe_sha256
        )

    def decide(
        self,
        subject: str,
        *,
        action: str = "retrieve",
        environment: Mapping[str, object] | None = None,
    ) -> PolicyDecision:
        del subject, action, environment
        decision = self.decisions[min(self.calls, len(self.decisions) - 1)]
        self.calls += 1
        return decision


def test_authorized_exact_and_hnsw_never_cross_boundary() -> None:
    vectors, policy = _fixture()
    mask = policy.authorized_mask("reader")
    query = vectors[0]
    exact_index = AuthorizedExactIndex(vectors, mask)
    exact = authorized_exact_search(exact_index, query, 10)
    hnsw_index = AuthorizedHNSWIndex(vectors, mask, ef_search=40, seed=3)
    hnsw = authorized_hnsw_search(
        hnsw_index,
        query,
        mask,
        10,
        ef_search=40,
        strategy="hnsw-high",
    )
    assert exact.unauthorized_context == 0
    assert hnsw.unauthorized_context == 0
    assert mask[exact.ids].all()
    assert mask[hnsw.ids].all()
    assert exact.work is not None
    assert exact.work.visited_candidates == 40
    assert exact.work.distance_evaluations == 40
    assert hnsw.work is not None
    assert hnsw.work.configured_ef_search == 40
    assert hnsw.work.visited_candidates is None
    assert hnsw.work.distance_evaluations is None


class _DeniedReadGuard(np.ndarray):
    """Array test double that raises when a forbidden source row is selected."""

    denied_rows: frozenset[int]
    accessed_rows: list[int]

    def __new__(
        cls,
        values: np.ndarray,
        denied_rows: set[int],
    ) -> _DeniedReadGuard:
        instance = np.asarray(values, dtype=np.float32).view(cls)
        instance.denied_rows = frozenset(denied_rows)
        instance.accessed_rows = []
        return instance

    def __array_finalize__(self, source: np.ndarray | None) -> None:
        if source is None:
            return
        self.denied_rows = getattr(source, "denied_rows", frozenset())
        self.accessed_rows = getattr(source, "accessed_rows", [])

    def __getitem__(self, key: object) -> np.ndarray:
        row_key = key[0] if isinstance(key, tuple) else key
        selected = np.asarray(np.arange(self.shape[0])[row_key]).reshape(-1)
        rows = [int(row) for row in selected]
        forbidden = self.denied_rows.intersection(rows)
        if forbidden:
            raise AssertionError(f"exact index read denied rows: {sorted(forbidden)}")
        self.accessed_rows.extend(rows)
        return super().__getitem__(key)


def test_authorized_exact_index_never_reads_denied_source_rows() -> None:
    vectors = np.arange(24, dtype=np.float32).reshape(6, 4)
    mask = np.asarray([True, False, True, False, False, True], dtype=bool)
    guarded = _DeniedReadGuard(vectors, set(np.flatnonzero(~mask)))

    index = AuthorizedExactIndex(guarded, mask)
    result = authorized_exact_search(index, np.zeros(4, dtype=np.float32), 2)

    assert guarded.accessed_rows == [0, 2, 5]
    assert index.original_ids.tolist() == [0, 2, 5]
    assert set(result.ids).issubset({0, 2, 5})


def test_authorized_exact_work_counts_only_the_authorized_slice() -> None:
    vectors = np.arange(20, dtype=np.float32).reshape(5, 4)
    mask = np.asarray([False, True, False, True, False], dtype=bool)
    index = AuthorizedExactIndex(vectors, mask)

    result = authorized_exact_search(index, vectors[1], 4)

    assert result.ids.tolist() == [1, 3]
    assert result.requested_k == 4
    assert result.shortfall == 2
    assert result.candidates_examined == 2
    assert result.work is not None
    assert result.work.returned_candidates == 2
    assert result.work.visited_candidates == 2
    assert result.work.distance_evaluations == 2


def test_authorized_probe_is_bounded_and_does_not_call_ef_measured_work() -> None:
    rng = np.random.default_rng(23)
    vectors = rng.normal(size=(180, 16)).astype(np.float32)
    mask = np.r_[np.ones(140, dtype=bool), np.zeros(40, dtype=bool)]
    index = AuthorizedHNSWIndex(vectors, mask, ef_search=128, seed=5)
    probe = authorized_hnsw_probe(
        index,
        vectors[0],
        mask,
        probe_k=101,
        ef_search=128,
        max_neighbors=101,
    )
    assert len(probe.ids) == 101
    assert mask[probe.ids].all()
    assert probe.work.returned_candidates == 101
    assert probe.work.configured_ef_search == 128
    assert probe.work.visited_candidates is None
    assert probe.work.distance_evaluations is None


def test_unsafe_comparator_detects_context_exposure() -> None:
    vectors, policy = _fixture()
    mask = policy.authorized_mask("reader")
    query = vectors[-1]
    global_index = HNSWSearchIndex(vectors, ef_search=40, seed=3)
    result = unsafe_unfiltered_search(global_index, query, mask, 10)
    assert result.unauthorized_context > 0


def test_controller_fails_closed_on_policy_mismatch() -> None:
    vectors, policy = _fixture()
    retriever = GovernedRetriever(
        vectors,
        policy,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        hnsw_seed=4,
    )
    result = retriever.query(vectors[0], expected_policy_version="future-policy")
    assert result.decision.action == "abstain"
    assert result.search is None


def test_retriever_rejects_same_sized_corpus_identity_substitution() -> None:
    vectors, expected_policy = _fixture()
    substituted_digest = policy_document_universe_sha256(
        f"substituted-document-{index}" for index in range(len(vectors))
    )
    substituted_policy = AuthorizationPolicy(
        roles=expected_policy.roles,
        visibility=expected_policy.visibility,
        version=expected_policy.version,
        document_universe_sha256=substituted_digest,
    )

    with pytest.raises(ValueError, match="does not match.*expected ordered"):
        GovernedRetriever(
            vectors,
            substituted_policy,
            "reader",
            expected_document_universe_sha256=(
                expected_policy.document_universe_sha256
            ),
        )


def test_retriever_requires_an_explicit_stable_document_universe_digest() -> None:
    vectors, policy = _fixture()

    with pytest.raises(TypeError, match="expected_document_universe_sha256"):
        GovernedRetriever(vectors, policy, "reader")  # type: ignore[call-arg]


def test_governed_query_returns_only_authorized_ids() -> None:
    vectors, policy = _fixture()
    retriever = GovernedRetriever(
        vectors,
        policy,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        hnsw_seed=4,
    )
    assert retriever._authorized_exact is None
    result = retriever.query(vectors[0], expected_policy_version=policy.version)
    assert result.search is not None
    assert result.search.unauthorized_context == 0
    assert policy.authorized_mask("reader")[result.search.ids].all()
    assert result.initial_authorization is not None
    assert result.final_authorization is not None
    assert result.initial_authorization.decision_id != result.final_authorization.decision_id
    assert result.request_latency_ms is not None
    assert result.request_latency_ms > 0.0
    assert result.total_online_latency_ms == result.request_latency_ms
    assert result.authorization_latency_ms > 0.0
    assert result.controller_latency_ms > 0.0
    assert retriever._authorized_exact is not None
    assert retriever._authorized_exact.n_authorized == 40
    assert retriever._authorized_exact._inner.n_documents == 40
    assert not result.search.ids.flags.writeable
    with np.testing.assert_raises(ValueError):
        result.search.ids[0] = 79


def test_retriever_owns_an_immutable_vector_snapshot() -> None:
    vectors, policy = _fixture()
    original_first = vectors[0].copy()
    retriever = GovernedRetriever(
        vectors,
        policy,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        hnsw_seed=4,
    )

    vectors[0] = 999.0

    np.testing.assert_array_equal(retriever.vectors[0], original_first)
    assert not np.shares_memory(retriever.vectors, vectors)
    assert not retriever.vectors.flags.writeable


def test_query_snapshot_owns_one_immutable_request_vector() -> None:
    query = np.arange(8, dtype=np.float32)
    frozen = snapshot_query(query, 8)

    query[:] = -1.0

    np.testing.assert_array_equal(frozen, np.arange(8, dtype=np.float32))
    assert not np.shares_memory(frozen, query)
    assert not frozen.flags.writeable


class _CallerQueryMutatingController(RuleController):
    def __init__(self, caller_query: np.ndarray, replacement: np.ndarray) -> None:
        super().__init__(
            ControllerConfig(
                low_ef=20,
                high_ef=40,
                probe_k=10,
                exact_scan_threshold=100,
            )
        )
        self.caller_query = caller_query
        self.replacement = replacement

    def decide(
        self,
        features: QueryGeometry,
        *,
        n_authorized: int,
        policy_version: str,
        policy_available: bool = True,
        expected_policy_version: str | None = None,
    ) -> ControllerDecision:
        self.caller_query[:] = self.replacement
        return super().decide(
            features,
            n_authorized=n_authorized,
            policy_version=policy_version,
            policy_available=policy_available,
            expected_policy_version=expected_policy_version,
        )


def test_governed_request_uses_one_snapshot_after_caller_mutates_query() -> None:
    vectors, policy = _fixture()
    query = vectors[0].copy()
    controller = _CallerQueryMutatingController(query, vectors[39])
    retriever = GovernedRetriever(
        vectors,
        policy,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        controller=controller,
        hnsw_seed=4,
    )

    result = retriever.query(query, k=5)

    np.testing.assert_array_equal(query, vectors[39])
    assert result.search is not None
    assert result.search.strategy == "exact-authorized"
    assert int(result.search.ids[0]) == 0


class _OwnershipTrackingLock:
    """RLock wrapper that exposes the current critical-section generation."""

    def __init__(self) -> None:
        self._lock = RLock()
        self.owner: int | None = None
        self.generation = 0

    def __enter__(self) -> _OwnershipTrackingLock:
        self._lock.acquire()
        self.owner = get_ident()
        self.generation += 1
        return self

    def __exit__(self, *_: object) -> None:
        self.owner = None
        self._lock.release()

    def assert_owned(self) -> None:
        assert self.owner == get_ident()


class _LockCheckingHNSWBackend:
    """Backend double that fails unless set_ef and query share one lock hold."""

    def __init__(self, lock: _OwnershipTrackingLock) -> None:
        self.lock = lock
        self.events: list[tuple[str, int, int, int]] = []
        self._thread_effort: dict[int, tuple[int, int]] = {}

    def set_ef(self, ef_search: int) -> None:
        self.lock.assert_owned()
        thread = get_ident()
        self._thread_effort[thread] = (ef_search, self.lock.generation)
        self.events.append(("set", thread, ef_search, self.lock.generation))

    def knn_query(
        self,
        query: np.ndarray,
        *,
        k: int,
        num_threads: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del query, k, num_threads
        self.lock.assert_owned()
        thread = get_ident()
        effort, generation = self._thread_effort[thread]
        assert generation == self.lock.generation
        self.events.append(("query", thread, effort, generation))
        label = 0 if effort < 50 else 1
        return (
            np.asarray([[label]], dtype=np.int64),
            np.asarray([[float(effort)]], dtype=np.float32),
        )


def test_hnsw_effort_and_query_are_atomic_across_concurrent_requests() -> None:
    inner = object.__new__(HNSWSearchIndex)
    inner.n_documents = 2
    inner.dimension = 1
    inner.metric = "euclidean"
    lock = _OwnershipTrackingLock()
    backend = _LockCheckingHNSWBackend(lock)
    inner._query_lock = lock
    inner._index = backend

    index = object.__new__(AuthorizedHNSWIndex)
    index.original_ids = np.asarray([0, 1], dtype=np.int64)
    index._inner = inner
    index.n_documents = 2
    index.n_authorized = 2
    index.metric = "euclidean"
    mask = np.ones(2, dtype=bool)
    start = Barrier(3)

    def search(ef_search: int) -> tuple[int, float]:
        start.wait(timeout=5)
        result = authorized_hnsw_search(
            index,
            np.asarray([0.0], dtype=np.float32),
            mask,
            1,
            ef_search=ef_search,
            strategy="hnsw-high",
        )
        return int(result.ids[0]), float(result.distances[0])

    with ThreadPoolExecutor(max_workers=2) as executor:
        low = executor.submit(search, 11)
        high = executor.submit(search, 97)
        start.wait(timeout=5)
        observed = {low.result(timeout=5), high.result(timeout=5)}

    assert observed == {(0, 11.0), (1, 97.0)}
    assert [event[0] for event in backend.events] == ["set", "query", "set", "query"]
    for set_event, query_event in zip(backend.events[::2], backend.events[1::2]):
        assert set_event[1:] == query_event[1:]


def test_query_time_policy_update_cannot_reuse_cached_authority() -> None:
    vectors, policy = _fixture()
    initial_visibility = policy.visibility.copy()
    initial_visibility[0, 40:50] = True
    policy = AuthorizationPolicy(
        roles=policy.roles,
        visibility=initial_visibility,
        version="v1",
        document_universe_sha256=policy.document_universe_sha256,
    )
    pdp = InMemoryPolicyDecisionPoint(policy)
    controller = RuleController(
        ControllerConfig(
            low_ef=40,
            high_ef=80,
            probe_k=20,
            exact_scan_threshold=0,
            high_effort_threshold=0.99,
            exact_threshold=1.0,
        )
    )
    retriever = GovernedRetriever(
        vectors,
        pdp,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        controller=controller,
        hnsw_seed=4,
    )
    before = retriever.query(vectors[0], expected_policy_version="v1")
    assert before.search is not None
    assert 0 in before.search.ids

    visibility = policy.visibility.copy()
    visibility[0, 0] = False
    pdp.replace(
        AuthorizationPolicy(
            roles=policy.roles,
            visibility=visibility,
            version="v2",
            document_universe_sha256=policy.document_universe_sha256,
        )
    )
    after = retriever.query(vectors[0], expected_policy_version="v2")
    assert after.search is not None
    assert 0 not in after.search.ids
    assert after.final_authorization is not None
    assert after.final_authorization.policy_version == "v2"


def test_policy_change_between_search_and_emission_abstains() -> None:
    vectors, policy = _fixture()
    before = AuthorizationPolicy(
        roles=policy.roles,
        visibility=policy.visibility,
        version="v1",
        document_universe_sha256=policy.document_universe_sha256,
    ).decide("reader")
    visibility = policy.visibility.copy()
    visibility[0, 0] = False
    after = AuthorizationPolicy(
        roles=policy.roles,
        visibility=visibility,
        version="v2",
        document_universe_sha256=policy.document_universe_sha256,
    ).decide("reader")
    pdp = _SequencedDecisionPoint([before, after])
    retriever = GovernedRetriever(
        vectors,
        pdp,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        hnsw_seed=4,
    )

    result = retriever.query(vectors[0])
    assert pdp.calls == 2
    assert result.decision.action == "abstain"
    assert result.search is None
    assert "policy changed" in result.decision.reasons[0]


def test_silent_mask_change_between_search_and_emission_abstains() -> None:
    vectors, policy = _fixture()
    before = AuthorizationPolicy(
        roles=policy.roles,
        visibility=policy.visibility,
        version="v1",
        document_universe_sha256=policy.document_universe_sha256,
    ).decide("reader")
    visibility = policy.visibility.copy()
    visibility[0, 0] = False
    after = AuthorizationPolicy(
        roles=policy.roles,
        visibility=visibility,
        version="v1",
        document_universe_sha256=policy.document_universe_sha256,
    ).decide("reader")
    pdp = _SequencedDecisionPoint([before, after])
    retriever = GovernedRetriever(
        vectors,
        pdp,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        hnsw_seed=4,
    )

    result = retriever.query(vectors[0])
    assert result.decision.action == "abstain"
    assert result.search is None
    assert "policy changed" in result.decision.reasons[0]


def test_policy_outage_during_final_authorization_abstains() -> None:
    vectors, policy = _fixture()
    initial = policy.decide("reader")
    outage = PolicyDecision(
        subject="reader",
        action="retrieve",
        policy_version=policy.version,
        authorized_mask=np.zeros(policy.n_documents, dtype=bool),
        available=False,
        reason="policy decision point unavailable; deny by default",
        document_universe_sha256=policy.document_universe_sha256,
    )
    pdp = _SequencedDecisionPoint([initial, outage])
    retriever = GovernedRetriever(
        vectors,
        pdp,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        hnsw_seed=4,
    )

    result = retriever.query(vectors[0])
    assert result.decision.action == "abstain"
    assert result.search is None
    assert "unavailable" in result.decision.reasons[0]


@pytest.mark.parametrize(
    "mismatch",
    ["subject", "action", "environment", "universe"],
)
def test_policy_response_must_bind_the_exact_request(mismatch: str) -> None:
    vectors, policy = _fixture()
    mask = policy.authorized_mask("reader")
    kwargs = {
        "subject": "reader",
        "action": "retrieve",
        "policy_version": "v1",
        "authorized_mask": mask,
        "environment_sha256": policy.decide(
            "reader", environment={"tenant": "a"}
        ).environment_sha256,
        "document_universe_sha256": policy.document_universe_sha256,
    }
    if mismatch == "subject":
        kwargs["subject"] = "admin"
    elif mismatch == "action":
        kwargs["action"] = "delete"
    elif mismatch == "environment":
        kwargs["environment_sha256"] = "a" * 64
    else:
        kwargs["document_universe_sha256"] = "b" * 64
    pdp = _SequencedDecisionPoint(
        [PolicyDecision(**kwargs)],  # type: ignore[arg-type]
        document_universe_sha256=(
            policy.document_universe_sha256 if mismatch == "universe" else None
        ),
    )
    retriever = GovernedRetriever(
        vectors,
        pdp,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        hnsw_seed=4,
    )

    result = retriever.query(vectors[0], environment={"tenant": "a"})

    assert result.decision.action == "abstain"
    assert result.search is None
    assert "does not match" in result.decision.reasons[0]


def test_replayed_final_policy_decision_abstains() -> None:
    vectors, policy = _fixture()
    decision = policy.decide("reader")
    pdp = _SequencedDecisionPoint([decision, decision])
    retriever = GovernedRetriever(
        vectors,
        pdp,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        hnsw_seed=4,
    )

    result = retriever.query(vectors[0])

    assert result.decision.action == "abstain"
    assert result.search is None
    assert "replayed" in result.decision.reasons[0]
