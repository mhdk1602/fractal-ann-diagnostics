from __future__ import annotations

import numpy as np
import pytest

from fractal_ann_diagnostics.policy import (
    AuthorizationPolicy,
    InMemoryPolicyDecisionPoint,
    PolicyDecision,
    policy_churn,
    policy_document_universe_sha256,
)


def test_document_role_policy_and_selectivity() -> None:
    policy = AuthorizationPolicy.from_document_roles(
        [{"*"}, {"analyst"}, {"admin"}, {"analyst", "admin"}],
        roles=("analyst", "admin"),
    )
    assert policy.authorized_ids("analyst").tolist() == [0, 1, 3]
    assert policy.authorized_ids("admin").tolist() == [0, 2, 3]
    assert policy.selectivity("analyst") == 0.75
    assert not policy.visibility.flags.writeable


def test_unknown_role_fails_closed() -> None:
    policy = AuthorizationPolicy(roles=("a",), visibility=np.ones((1, 3), dtype=bool))
    with pytest.raises(KeyError, match="unknown role"):
        policy.authorized_mask("missing")


def test_policy_decision_is_read_only_and_auditable() -> None:
    policy = AuthorizationPolicy(
        roles=("reader",),
        visibility=np.asarray([[1, 0, 1]], dtype=bool),
        version="iam-v4",
    )
    decision = policy.decide(
        "reader",
        action="retrieve",
        environment={"region": "us-east"},
    )
    assert decision.subject == "reader"
    assert decision.action == "retrieve"
    assert decision.policy_version == "iam-v4"
    assert decision.authorized_count == 2
    assert decision.decision_id
    assert decision.request_nonce
    assert len(decision.request_sha256) == 64
    assert decision.permits(np.asarray([0, 2]))
    assert not decision.permits(np.asarray([1]))
    with pytest.raises(ValueError, match="read-only"):
        decision.authorized_mask[0] = False


def test_each_policy_evaluation_has_a_fresh_request_binding() -> None:
    policy = AuthorizationPolicy(
        roles=("reader",),
        visibility=np.ones((1, 2), dtype=bool),
    )
    first = policy.decide("reader", environment={"tenant": "research"})
    second = policy.decide("reader", environment={"tenant": "research"})
    assert first.request_nonce != second.request_nonce
    assert first.request_sha256 != second.request_sha256


def test_policy_decision_rejects_a_forged_request_digest() -> None:
    with pytest.raises(ValueError, match="does not bind"):
        PolicyDecision(
            subject="reader",
            action="retrieve",
            policy_version="v1",
            authorized_mask=np.ones(2, dtype=bool),
            request_nonce="fresh-nonce",
            request_sha256="0" * 64,
        )


def test_mutable_pdp_requires_version_change_for_new_decisions() -> None:
    initial = AuthorizationPolicy(
        roles=("reader",),
        visibility=np.asarray([[1, 1, 0]], dtype=bool),
        version="v1",
    )
    pdp = InMemoryPolicyDecisionPoint(initial)
    changed_without_version = AuthorizationPolicy(
        roles=("reader",),
        visibility=np.asarray([[1, 0, 0]], dtype=bool),
        version="v1",
    )
    with pytest.raises(ValueError, match="require a new version"):
        pdp.replace(changed_without_version)

    changed = AuthorizationPolicy(
        roles=("reader",),
        visibility=np.asarray([[1, 0, 0]], dtype=bool),
        version="v2",
    )
    pdp.replace(changed)
    assert pdp.decide("reader").policy_version == "v2"


def test_mutable_pdp_rejects_same_sized_document_universe_substitution() -> None:
    initial = AuthorizationPolicy(
        roles=("reader",),
        visibility=np.asarray([[1, 1, 0]], dtype=bool),
        version="v1",
        document_universe_sha256=policy_document_universe_sha256(
            ("document-a", "document-b", "document-c")
        ),
    )
    substitution = AuthorizationPolicy(
        roles=("reader",),
        visibility=initial.visibility,
        version="v2",
        document_universe_sha256=policy_document_universe_sha256(("other-a", "other-b", "other-c")),
    )

    with pytest.raises(ValueError, match="preserve document universe identity"):
        InMemoryPolicyDecisionPoint(initial).replace(substitution)


def test_unavailable_pdp_denies_every_document() -> None:
    policy = AuthorizationPolicy(
        roles=("reader",),
        visibility=np.ones((1, 3), dtype=bool),
    )
    pdp = InMemoryPolicyDecisionPoint(policy)
    pdp.set_available(False)
    decision = pdp.decide("reader")
    assert not decision.available
    assert decision.authorized_count == 0


def test_policy_churn_is_role_specific() -> None:
    before = AuthorizationPolicy(
        roles=("a", "b"),
        visibility=np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool),
    )
    after = AuthorizationPolicy(
        roles=("a", "b"),
        visibility=np.asarray([[1, 0, 1, 0], [0, 0, 1, 1]], dtype=bool),
    )
    assert policy_churn(before, after, "a") == 0.5
    assert policy_churn(before, after, "b") == 0.0
