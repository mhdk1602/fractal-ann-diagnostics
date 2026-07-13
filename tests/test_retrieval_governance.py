from __future__ import annotations

import numpy as np

from fractal_ann_diagnostics.controller import GovernedRetriever
from fractal_ann_diagnostics.policy import AuthorizationPolicy
from fractal_ann_diagnostics.retrieval import (
    AuthorizedHNSWIndex,
    ExactSearchIndex,
    HNSWSearchIndex,
    authorized_hnsw_search,
    exact_authorized_search,
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
    return vectors, AuthorizationPolicy(roles=("reader",), visibility=visibility)


def test_authorized_exact_and_hnsw_never_cross_boundary() -> None:
    vectors, policy = _fixture()
    mask = policy.authorized_mask("reader")
    query = vectors[0]
    exact = exact_authorized_search(ExactSearchIndex(vectors), query, mask, 10)
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


def test_unsafe_comparator_detects_context_exposure() -> None:
    vectors, policy = _fixture()
    mask = policy.authorized_mask("reader")
    query = vectors[-1]
    global_index = HNSWSearchIndex(vectors, ef_search=40, seed=3)
    result = unsafe_unfiltered_search(global_index, query, mask, 10)
    assert result.unauthorized_context > 0


def test_controller_fails_closed_on_policy_mismatch() -> None:
    vectors, policy = _fixture()
    retriever = GovernedRetriever(vectors, policy, "reader", hnsw_seed=4)
    result = retriever.query(vectors[0], expected_policy_version="future-policy")
    assert result.decision.action == "abstain"
    assert result.search is None


def test_governed_query_returns_only_authorized_ids() -> None:
    vectors, policy = _fixture()
    retriever = GovernedRetriever(vectors, policy, "reader", hnsw_seed=4)
    result = retriever.query(vectors[0], expected_policy_version=policy.version)
    assert result.search is not None
    assert result.search.unauthorized_context == 0
    assert policy.authorized_mask("reader")[result.search.ids].all()
