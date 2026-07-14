"""Fail-closed action selection inside an already authorized universe."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from threading import RLock
from time import perf_counter_ns
from typing import Literal

import numpy as np

from .geometry import QueryGeometry, query_geometry_from_probe
from .policy import PolicyDecision, PolicyDecisionPoint, policy_environment_sha256
from .retrieval import (
    AuthorizedExactIndex,
    AuthorizedHNSWIndex,
    DistanceMetric,
    ProbeTelemetry,
    SearchResult,
    authorized_exact_search,
    authorized_hnsw_probe,
    authorized_hnsw_search,
    search_result_from_probe,
    snapshot_query,
)

ControllerAction = Literal["hnsw-low", "hnsw-high", "exact-authorized", "abstain"]


@dataclass(frozen=True)
class ControllerConfig:
    """Locked development thresholds for the reference rule controller."""

    low_ef: int = 128
    high_ef: int = 512
    probe_k: int = 101
    exact_scan_threshold: int = 256
    high_effort_threshold: float = 0.24
    exact_threshold: float = 0.36

    def __post_init__(self) -> None:
        if self.probe_k <= 0:
            raise ValueError("probe_k must be positive")
        if self.low_ef < self.probe_k or self.high_ef < self.low_ef:
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
    initial_authorization: PolicyDecision | None = None
    final_authorization: PolicyDecision | None = None
    probe: ProbeTelemetry | None = None
    index_refresh: IndexRefreshWork | None = None
    authorization_latency_ms: float = 0.0
    controller_latency_ms: float = 0.0
    request_latency_ms: float | None = None

    @property
    def total_online_latency_ms(self) -> float:
        """Measured end-to-end request latency, with a compositional fallback."""
        if self.request_latency_ms is not None:
            return self.request_latency_ms
        if self.geometry is None:
            return self.authorization_latency_ms + self.controller_latency_ms
        total = (
            self.authorization_latency_ms
            + self.controller_latency_ms
            + self.geometry.accounted_latency_ms
        )
        if self.search is not None and self.search.strategy != "hnsw-low":
            total += self.search.latency_ms
        if self.index_refresh is not None:
            total += self.index_refresh.latency_ms
        return total


@dataclass(frozen=True)
class IndexRefreshWork:
    """Observed construction or cache-reuse work for an authorized index."""

    policy_version: str
    mask_sha256: str
    rebuilt: bool
    latency_ms: float
    authorized_count: int


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
        policy: PolicyDecisionPoint,
        role: str,
        *,
        expected_document_universe_sha256: str,
        metric: DistanceMetric = "euclidean",
        controller: RuleController | None = None,
        policy_churn: float = 0.0,
        embedding_drift: float = 0.0,
        hnsw_seed: int = 42,
    ) -> None:
        self.vectors = np.array(vectors, dtype=np.float32, copy=True)
        if self.vectors.ndim != 2 or len(self.vectors) == 0:
            raise ValueError("vectors must have shape (n_documents, dimension), n > 0")
        if not np.all(np.isfinite(self.vectors)):
            raise ValueError("vectors contain non-finite values")
        self.vectors.setflags(write=False)
        self.policy = policy
        self.role = role
        self.metric = metric
        self.controller = controller or RuleController()
        if not np.isfinite(policy_churn) or not 0.0 <= policy_churn <= 1.0:
            raise ValueError("policy_churn must be finite and in [0, 1]")
        if not np.isfinite(embedding_drift) or embedding_drift < 0.0:
            raise ValueError("embedding_drift must be finite and non-negative")
        self.policy_churn = float(policy_churn)
        self.embedding_drift = float(embedding_drift)
        if policy.n_documents != len(self.vectors):
            raise ValueError("policy and vector document counts differ")
        if (
            not isinstance(expected_document_universe_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_document_universe_sha256) is None
        ):
            raise ValueError(
                "expected_document_universe_sha256 must be an explicit lowercase "
                "SHA-256 digest of the ordered stable document IDs"
            )
        if policy.document_universe_sha256 != expected_document_universe_sha256:
            raise ValueError(
                "policy document universe does not match the retriever's expected "
                "ordered document universe"
            )
        self.document_universe_sha256 = expected_document_universe_sha256
        self.hnsw_seed = hnsw_seed
        self._authorized_exact: AuthorizedExactIndex | None = None
        self._authorized_hnsw: AuthorizedHNSWIndex | None = None
        self._authorized_hnsw_key: tuple[str, str] | None = None
        self._index_lock = RLock()

    def _policy_decision(
        self,
        *,
        action: str,
        environment: Mapping[str, object] | None,
    ) -> PolicyDecision:
        try:
            return self.policy.decide(
                self.role,
                action=action,
                environment=environment,
            )
        except Exception as exc:
            return PolicyDecision(
                subject=self.role,
                action=action,
                policy_version="unavailable",
                authorized_mask=np.zeros(len(self.vectors), dtype=bool),
                available=False,
                reason=(
                    f"policy decision point raised {type(exc).__name__}; deny by default"
                ),
                document_universe_sha256=self.document_universe_sha256,
            )

    def _abstain(
        self,
        reason: str,
        *,
        policy_version: str,
        geometry: QueryGeometry | None = None,
        initial_authorization: PolicyDecision | None = None,
        final_authorization: PolicyDecision | None = None,
        probe: ProbeTelemetry | None = None,
        index_refresh: IndexRefreshWork | None = None,
    ) -> GovernedResult:
        return GovernedResult(
            decision=ControllerDecision(
                action="abstain",
                risk_score=1.0,
                reasons=(reason,),
                policy_version=policy_version,
            ),
            geometry=geometry,
            search=None,
            initial_authorization=initial_authorization,
            final_authorization=final_authorization,
            probe=probe,
            index_refresh=index_refresh,
        )

    def _validate_decision(
        self,
        decision: PolicyDecision,
        *,
        action: str,
        environment_sha256: str,
    ) -> str | None:
        if not decision.available:
            return decision.reason
        if decision.subject != self.role:
            return "policy decision subject does not match the request; fail closed"
        if decision.action != action:
            return "policy decision action does not match the request; fail closed"
        if decision.environment_sha256 != environment_sha256:
            return "policy decision environment does not match the request; fail closed"
        if decision.document_universe_sha256 != self.document_universe_sha256:
            return "policy decision document universe does not match the request; fail closed"
        if decision.authorized_mask.shape != (len(self.vectors),):
            return "policy decision has the wrong document universe; fail closed"
        return None

    def _authorized_index(
        self,
        authorization: PolicyDecision,
    ) -> tuple[AuthorizedExactIndex, AuthorizedHNSWIndex, IndexRefreshWork]:
        mask_digest = sha256(authorization.authorized_mask.tobytes()).hexdigest()
        key = (authorization.policy_version, mask_digest)
        with self._index_lock:
            rebuilt = (
                self._authorized_exact is None
                or self._authorized_hnsw is None
                or self._authorized_hnsw_key != key
            )
            start = perf_counter_ns()
            if rebuilt:
                authorized_exact = AuthorizedExactIndex(
                    self.vectors,
                    authorization.authorized_mask,
                    metric=self.metric,
                )
                authorized_hnsw = AuthorizedHNSWIndex(
                    self.vectors,
                    authorization.authorized_mask,
                    metric=self.metric,
                    ef_search=self.controller.config.low_ef,
                    seed=self.hnsw_seed,
                )
                self._authorized_exact = authorized_exact
                self._authorized_hnsw = authorized_hnsw
                self._authorized_hnsw_key = key
            latency_ms = (perf_counter_ns() - start) / 1_000_000
            assert self._authorized_exact is not None
            assert self._authorized_hnsw is not None
            return self._authorized_exact, self._authorized_hnsw, IndexRefreshWork(
                policy_version=authorization.policy_version,
                mask_sha256=mask_digest,
                rebuilt=rebuilt,
                latency_ms=latency_ms,
                authorized_count=authorization.authorized_count,
            )

    def query(
        self,
        query: np.ndarray,
        *,
        k: int = 10,
        policy_available: bool = True,
        expected_policy_version: str | None = None,
        action: str = "retrieve",
        environment: Mapping[str, object] | None = None,
    ) -> GovernedResult:
        request_started = perf_counter_ns()
        authorization_latency_ms = 0.0
        controller_latency_ms = 0.0

        def finish(result: GovernedResult) -> GovernedResult:
            return replace(
                result,
                authorization_latency_ms=authorization_latency_ms,
                controller_latency_ms=controller_latency_ms,
                request_latency_ms=(perf_counter_ns() - request_started) / 1_000_000,
            )

        if k <= 0:
            raise ValueError("k must be positive")
        if k > self.controller.config.probe_k:
            raise ValueError("k cannot exceed the frozen probe_k bound")
        query_snapshot = snapshot_query(query, self.vectors.shape[1])
        if not policy_available:
            return finish(
                self._abstain(
                    "live policy decision point unavailable; fail closed",
                    policy_version=expected_policy_version or "unavailable",
                )
            )

        try:
            environment_digest = policy_environment_sha256(environment)
        except ValueError:
            return finish(
                self._abstain(
                    "policy environment is not finite JSON; fail closed",
                    policy_version=expected_policy_version or "unavailable",
                )
            )

        authorization_started = perf_counter_ns()
        initial_authorization = self._policy_decision(
            action=action,
            environment=environment,
        )
        authorization_latency_ms += (
            perf_counter_ns() - authorization_started
        ) / 1_000_000
        initial_error = self._validate_decision(
            initial_authorization,
            action=action,
            environment_sha256=environment_digest,
        )
        if initial_error is not None:
            return finish(
                self._abstain(
                    initial_error,
                    policy_version=initial_authorization.policy_version,
                    initial_authorization=initial_authorization,
                )
            )
        if (
            expected_policy_version is not None
            and initial_authorization.policy_version != expected_policy_version
        ):
            return finish(
                self._abstain(
                    "policy version mismatch; fail closed",
                    policy_version=initial_authorization.policy_version,
                    initial_authorization=initial_authorization,
                )
            )

        mask = initial_authorization.authorized_mask
        if initial_authorization.authorized_count == 0:
            return finish(
                self._abstain(
                    "authorized universe is empty",
                    policy_version=initial_authorization.policy_version,
                    initial_authorization=initial_authorization,
                )
            )

        authorized_exact, authorized_index, index_refresh = self._authorized_index(
            initial_authorization
        )
        probe = authorized_hnsw_probe(
            authorized_index,
            query_snapshot,
            mask,
            probe_k=self.controller.config.probe_k,
            ef_search=self.controller.config.low_ef,
            max_neighbors=self.controller.config.probe_k,
        )
        geometry = query_geometry_from_probe(
            probe,
            policy_churn=self.policy_churn,
            embedding_drift=self.embedding_drift,
        )
        controller_started = perf_counter_ns()
        decision = self.controller.decide(
            geometry,
            n_authorized=initial_authorization.authorized_count,
            policy_version=initial_authorization.policy_version,
            policy_available=True,
            expected_policy_version=expected_policy_version,
        )
        controller_latency_ms += (perf_counter_ns() - controller_started) / 1_000_000
        if decision.action == "abstain":
            return finish(
                GovernedResult(
                    decision=decision,
                    geometry=geometry,
                    search=None,
                    initial_authorization=initial_authorization,
                    probe=probe,
                    index_refresh=index_refresh,
                )
            )
        if decision.action == "exact-authorized":
            search = authorized_exact_search(authorized_exact, query_snapshot, k)
        elif decision.action == "hnsw-low":
            search = search_result_from_probe(probe, k, strategy="hnsw-low")
        else:
            search = authorized_hnsw_search(
                authorized_index,
                query_snapshot,
                mask,
                k,
                ef_search=self.controller.config.high_ef,
                strategy=decision.action,
            )

        # Re-query the authoritative PDP immediately before any IDs cross the
        # emission/context boundary. A cached index or mask is never final authority.
        authorization_started = perf_counter_ns()
        final_authorization = self._policy_decision(
            action=action,
            environment=environment,
        )
        authorization_latency_ms += (
            perf_counter_ns() - authorization_started
        ) / 1_000_000
        final_error = self._validate_decision(
            final_authorization,
            action=action,
            environment_sha256=environment_digest,
        )
        if final_error is not None:
            return finish(
                self._abstain(
                    final_error,
                    policy_version=final_authorization.policy_version,
                    geometry=geometry,
                    initial_authorization=initial_authorization,
                    final_authorization=final_authorization,
                    probe=probe,
                    index_refresh=index_refresh,
                )
            )
        if (
            expected_policy_version is not None
            and final_authorization.policy_version != expected_policy_version
        ):
            return finish(
                self._abstain(
                    "policy version mismatch during final authorization; fail closed",
                    policy_version=final_authorization.policy_version,
                    geometry=geometry,
                    initial_authorization=initial_authorization,
                    final_authorization=final_authorization,
                    probe=probe,
                    index_refresh=index_refresh,
                )
            )
        if (
            final_authorization.policy_version != initial_authorization.policy_version
            or final_authorization.decision_id == initial_authorization.decision_id
            or final_authorization.request_nonce == initial_authorization.request_nonce
            or final_authorization.request_sha256 == initial_authorization.request_sha256
            or not np.array_equal(
                final_authorization.authorized_mask,
                initial_authorization.authorized_mask,
            )
        ):
            return finish(
                self._abstain(
                    "policy changed or replayed a decision during retrieval; fail closed",
                    policy_version=final_authorization.policy_version,
                    geometry=geometry,
                    initial_authorization=initial_authorization,
                    final_authorization=final_authorization,
                    probe=probe,
                    index_refresh=index_refresh,
                )
            )
        if not final_authorization.permits(search.ids):
            return finish(
                self._abstain(
                    "final authorization revoked a returned document; fail closed",
                    policy_version=final_authorization.policy_version,
                    geometry=geometry,
                    initial_authorization=initial_authorization,
                    final_authorization=final_authorization,
                    probe=probe,
                    index_refresh=index_refresh,
                )
            )
        return finish(
            GovernedResult(
                decision=decision,
                geometry=geometry,
                search=search,
                initial_authorization=initial_authorization,
                final_authorization=final_authorization,
                probe=probe,
                index_refresh=index_refresh,
            )
        )
