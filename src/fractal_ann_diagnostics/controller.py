"""Fail-closed action selection inside an already authorized universe."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .geometry import QueryGeometry, query_geometry
from .policy import AuthorizationPolicy
from .retrieval import (
    AuthorizedHNSWIndex,
    DistanceMetric,
    ExactSearchIndex,
    SearchResult,
    authorized_hnsw_search,
    exact_authorized_search,
)

ControllerAction = Literal["hnsw-low", "hnsw-high", "exact-authorized", "abstain"]


@dataclass(frozen=True)
class ControllerConfig:
    """Locked development thresholds for the reference rule controller."""

    low_ef: int = 32
    high_ef: int = 160
    exact_scan_threshold: int = 256
    high_effort_threshold: float = 0.24
    exact_threshold: float = 0.36

    def __post_init__(self) -> None:
        if self.low_ef <= 0 or self.high_ef < self.low_ef:
            raise ValueError("high_ef must be greater than or equal to positive low_ef")
        if self.exact_scan_threshold < 0:
            raise ValueError("exact_scan_threshold must be non-negative")
        if not 0.0 <= self.high_effort_threshold < self.exact_threshold <= 1.0:
            raise ValueError(
                "risk thresholds must satisfy 0 <= high_effort < exact <= 1"
            )


@dataclass(frozen=True)
class ControllerDecision:
    action: ControllerAction
    risk_score: float
    reasons: tuple[str, ...]
    policy_version: str


@dataclass(frozen=True)
class GovernedResult:
    decision: ControllerDecision
    geometry: QueryGeometry | None
    search: SearchResult | None


def geometry_risk_score(features: QueryGeometry) -> float:
    """Transparent development score; not a calibrated probability."""
    lid = 1.0 if not np.isfinite(features.lid) else features.lid / (features.lid + 20.0)
    instability = (
        1.0
        if not np.isfinite(features.lid_scale_instability)
        else min(features.lid_scale_instability / 0.5, 1.0)
    )
    contrast_pressure = (
        1.0
        if not np.isfinite(features.relative_contrast)
        else 1.0 / max(features.relative_contrast, 1.0)
    )
    expansion = (
        1.0
        if not np.isfinite(features.radius_expansion)
        else min(max(features.radius_expansion - 1.0, 0.0) / 2.0, 1.0)
    )
    selectivity_pressure = 1.0 - min(features.authorized_selectivity / 0.25, 1.0)
    churn = min(features.policy_churn / 0.10, 1.0)
    drift = min(features.embedding_drift / 1.0, 1.0)
    return float(
        0.25 * lid
        + 0.15 * instability
        + 0.10 * contrast_pressure
        + 0.10 * expansion
        + 0.15 * selectivity_pressure
        + 0.15 * churn
        + 0.10 * drift
    )


class RuleController:
    """Reference controller whose actions cannot expand document permissions."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()

    def decide(
        self,
        features: QueryGeometry,
        *,
        n_authorized: int,
        policy_version: str,
        policy_available: bool = True,
        expected_policy_version: str | None = None,
    ) -> ControllerDecision:
        if not policy_available:
            return ControllerDecision(
                action="abstain",
                risk_score=1.0,
                reasons=("live policy engine unavailable; fail closed",),
                policy_version=policy_version,
            )
        if expected_policy_version is not None and policy_version != expected_policy_version:
            return ControllerDecision(
                action="abstain",
                risk_score=1.0,
                reasons=("policy version mismatch; fail closed",),
                policy_version=policy_version,
            )
        if n_authorized == 0:
            return ControllerDecision(
                action="abstain",
                risk_score=1.0,
                reasons=("authorized universe is empty",),
                policy_version=policy_version,
            )

        score = geometry_risk_score(features)
        if n_authorized <= self.config.exact_scan_threshold:
            action: ControllerAction = "exact-authorized"
            reasons = ("authorized subset is below the exact-scan threshold",)
        elif score >= self.config.exact_threshold:
            action = "exact-authorized"
            reasons = ("geometry score exceeds the exact-search threshold",)
        elif score >= self.config.high_effort_threshold:
            action = "hnsw-high"
            reasons = ("geometry score calls for widened authorized HNSW",)
        else:
            action = "hnsw-low"
            reasons = ("geometry score permits the low-effort authorized path",)
        return ControllerDecision(
            action=action,
            risk_score=score,
            reasons=reasons,
            policy_version=policy_version,
        )


class GovernedRetriever:
    """Reference execution path with deterministic IAM before geometry or ANN."""

    def __init__(
        self,
        vectors: np.ndarray,
        policy: AuthorizationPolicy,
        role: str,
        *,
        metric: DistanceMetric = "euclidean",
        controller: RuleController | None = None,
        policy_churn: float = 0.0,
        embedding_drift: float = 0.0,
        hnsw_seed: int = 42,
    ) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.policy = policy
        self.role = role
        self.metric = metric
        self.controller = controller or RuleController()
        self.policy_churn = policy_churn
        self.embedding_drift = embedding_drift
        self.mask = policy.authorized_mask(role)
        self.exact = ExactSearchIndex(self.vectors, metric=metric)
        self.authorized_hnsw = AuthorizedHNSWIndex(
            self.vectors,
            self.mask,
            metric=metric,
            ef_search=self.controller.config.low_ef,
            seed=hnsw_seed,
        )

    def query(
        self,
        query: np.ndarray,
        *,
        k: int = 10,
        policy_available: bool = True,
        expected_policy_version: str | None = None,
    ) -> GovernedResult:
        if not policy_available or (
            expected_policy_version is not None
            and self.policy.version != expected_policy_version
        ):
            placeholder = QueryGeometry(
                lid=float("nan"),
                lid_scale_instability=float("nan"),
                authorized_selectivity=self.policy.selectivity(self.role),
                relative_contrast=float("nan"),
                radius_expansion=float("nan"),
                policy_churn=self.policy_churn,
                embedding_drift=self.embedding_drift,
            )
            decision = self.controller.decide(
                placeholder,
                n_authorized=int(self.mask.sum()),
                policy_version=self.policy.version,
                policy_available=policy_available,
                expected_policy_version=expected_policy_version,
            )
            return GovernedResult(decision=decision, geometry=None, search=None)

        geometry = query_geometry(
            self.vectors,
            query,
            self.mask,
            metric=self.metric,
            policy_churn=self.policy_churn,
            embedding_drift=self.embedding_drift,
        )
        decision = self.controller.decide(
            geometry,
            n_authorized=int(self.mask.sum()),
            policy_version=self.policy.version,
            policy_available=policy_available,
            expected_policy_version=expected_policy_version,
        )
        if decision.action == "abstain":
            return GovernedResult(decision=decision, geometry=geometry, search=None)
        if decision.action == "exact-authorized":
            search = exact_authorized_search(self.exact, query, self.mask, k)
        else:
            ef = (
                self.controller.config.low_ef
                if decision.action == "hnsw-low"
                else self.controller.config.high_ef
            )
            search = authorized_hnsw_search(
                self.authorized_hnsw,
                query,
                self.mask,
                k,
                ef_search=ef,
                strategy=decision.action,
            )
        return GovernedResult(decision=decision, geometry=geometry, search=search)
