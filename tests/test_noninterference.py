from __future__ import annotations

import numpy as np

from fractal_ann_diagnostics.controller import (
    ControllerConfig,
    GovernedRetriever,
    RuleController,
)
from fractal_ann_diagnostics.noninterference import (
    compare_governed_observations,
    compare_search_observations,
    observe_governed_result,
    observe_search,
)
from fractal_ann_diagnostics.policy import (
    AuthorizationPolicy,
    policy_document_universe_sha256,
)
from fractal_ann_diagnostics.retrieval import (
    ExactSearchIndex,
    HNSWSearchIndex,
    unsafe_unfiltered_search,
)


def _worlds() -> tuple[np.ndarray, np.ndarray, AuthorizationPolicy, np.ndarray]:
    rng = np.random.default_rng(210)
    authorized = rng.normal(0, 0.5, size=(140, 16)).astype(np.float32)
    denied_a = rng.normal(8, 0.5, size=(40, 16)).astype(np.float32)
    denied_b = rng.normal(-8, 0.5, size=(40, 16)).astype(np.float32)
    first = np.vstack([authorized, denied_a])
    second = np.vstack([authorized, denied_b])
    mask = np.r_[np.ones(140, dtype=bool), np.zeros(40, dtype=bool)]
    policy = AuthorizationPolicy(
        roles=("reader",),
        visibility=mask.reshape(1, -1),
        version="paired-world-v1",
        document_universe_sha256=policy_document_universe_sha256(
            f"paired-world-document-{index}" for index in range(len(first))
        ),
    )
    return first, second, policy, authorized[0] + 0.01


def test_governed_observation_is_unchanged_when_only_denied_vectors_change() -> None:
    first, second, policy, query = _worlds()
    controller = RuleController(
        ControllerConfig(
            low_ef=128,
            high_ef=256,
            probe_k=101,
            exact_scan_threshold=0,
            high_effort_threshold=0.99,
            exact_threshold=1.0,
        )
    )
    first_result = GovernedRetriever(
        first,
        policy,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        controller=controller,
        hnsw_seed=71,
    ).query(query)
    second_result = GovernedRetriever(
        second,
        policy,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        controller=controller,
        hnsw_seed=71,
    ).query(query)
    report = compare_governed_observations(
        observe_governed_result(first_result),
        observe_governed_result(second_result),
    )
    assert report.equivalent, report.differences
    assert first_result.geometry is not None
    assert first_result.geometry.source == "bounded-probe"


def test_governed_exact_path_is_invariant_to_adversarial_denied_vectors() -> None:
    first, second, policy, query = _worlds()
    first[140:] = query
    second[140:] = query + 1_000.0

    global_first_ids, _ = ExactSearchIndex(first).query(query, 10)
    global_second_ids, _ = ExactSearchIndex(second).query(query, 10)
    assert np.any(global_first_ids >= 140)
    assert not np.array_equal(global_first_ids, global_second_ids)

    controller = RuleController(
        ControllerConfig(
            low_ef=128,
            high_ef=256,
            probe_k=101,
            exact_scan_threshold=140,
            high_effort_threshold=0.99,
            exact_threshold=1.0,
        )
    )
    first_retriever = GovernedRetriever(
        first,
        policy,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        controller=controller,
        hnsw_seed=71,
    )
    second_retriever = GovernedRetriever(
        second,
        policy,
        "reader",
        expected_document_universe_sha256=policy.document_universe_sha256,
        controller=controller,
        hnsw_seed=71,
    )

    first_result = first_retriever.query(query)
    second_result = second_retriever.query(query)
    report = compare_governed_observations(
        observe_governed_result(first_result),
        observe_governed_result(second_result),
    )

    assert report.equivalent, report.differences
    assert first_result.decision.action == "exact-authorized"
    assert second_result.decision.action == "exact-authorized"
    assert first_result.search is not None
    assert second_result.search is not None
    assert first_result.search.work is not None
    assert second_result.search.work is not None
    assert first_result.search.work.visited_candidates == 140
    assert second_result.search.work.visited_candidates == 140
    assert first_retriever._authorized_exact is not None
    assert second_retriever._authorized_exact is not None
    assert first_retriever._authorized_exact.original_ids.tolist() == list(range(140))
    assert second_retriever._authorized_exact.original_ids.tolist() == list(range(140))


def test_unsafe_global_comparator_is_a_sensitive_positive_control() -> None:
    first, second, policy, _ = _worlds()
    query = first[-1]
    mask = policy.authorized_mask("reader")
    first_search = unsafe_unfiltered_search(
        HNSWSearchIndex(first, ef_search=128, seed=19),
        query,
        mask,
        10,
    )
    second_search = unsafe_unfiltered_search(
        HNSWSearchIndex(second, ef_search=128, seed=19),
        query,
        mask,
        10,
    )
    differences = compare_search_observations(
        observe_search(first_search),
        observe_search(second_search),
    )
    assert differences
    assert first_search.unauthorized_context > 0
