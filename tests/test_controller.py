from __future__ import annotations

import pytest

from fractal_ann_diagnostics.controller import ControllerConfig, RuleController
from fractal_ann_diagnostics.geometry import QueryGeometry


def _features(*, drift: float = 0.0, churn: float = 0.0) -> QueryGeometry:
    return QueryGeometry(
        lid=5.0,
        lid_scale_instability=0.05,
        authorized_selectivity=0.5,
        relative_contrast=8.0,
        radius_expansion=1.1,
        policy_churn=churn,
        embedding_drift=drift,
    )


def test_controller_exercises_low_high_exact_and_abstain_paths() -> None:
    controller = RuleController(
        ControllerConfig(
            low_ef=128,
            high_ef=256,
            probe_k=101,
            exact_scan_threshold=10,
            high_effort_threshold=0.15,
            exact_threshold=0.25,
        )
    )
    low = controller.decide(_features(), n_authorized=100, policy_version="v1")
    high = controller.decide(
        _features(drift=1.0), n_authorized=100, policy_version="v1"
    )
    exact = controller.decide(
        _features(drift=1.0, churn=0.1),
        n_authorized=100,
        policy_version="v1",
    )
    abstain = controller.decide(
        _features(),
        n_authorized=100,
        policy_version="v1",
        policy_available=False,
    )
    assert low.action == "hnsw-low"
    assert high.action == "hnsw-high"
    assert exact.action == "exact-authorized"
    assert abstain.action == "abstain"


@pytest.mark.parametrize(
    "config",
    [
        {"low_ef": 0},
        {"low_ef": 40, "high_ef": 10},
        {"low_ef": 100, "high_ef": 200, "probe_k": 101},
        {"exact_scan_threshold": -1},
        {"high_effort_threshold": 0.4, "exact_threshold": 0.3},
    ],
)
def test_controller_rejects_incoherent_configuration(config: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ControllerConfig(**config)  # type: ignore[arg-type]
