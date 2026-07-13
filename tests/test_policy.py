from __future__ import annotations

import numpy as np
import pytest

from fractal_ann_diagnostics.policy import AuthorizationPolicy, policy_churn


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
