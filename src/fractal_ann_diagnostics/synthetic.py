"""Deterministic governed-vector scenarios for the development pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import AuthorizationPolicy


@dataclass(frozen=True)
class GovernedScenario:
    name: str
    vectors: np.ndarray
    queries: np.ndarray
    query_roles: tuple[str, ...]
    policy: AuthorizationPolicy
    baseline_policy: AuthorizationPolicy
    embedding_drift: float
    description: str


def _base_corpus(
    *,
    n_documents: int,
    dimension: int,
    n_roles: int,
    n_queries_per_role: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    rng = np.random.default_rng(seed)
    roles = tuple(f"role-{i}" for i in range(n_roles))
    centers = rng.normal(0.0, 4.0, size=(n_roles, dimension))
    document_roles = np.arange(n_documents) % n_roles
    rng.shuffle(document_roles)
    scales = rng.uniform(0.55, 1.35, size=n_roles)
    vectors = centers[document_roles] + rng.normal(
        0.0, scales[document_roles, None], size=(n_documents, dimension)
    )

    queries: list[np.ndarray] = []
    query_roles: list[str] = []
    for role_id, role in enumerate(roles):
        candidates = np.flatnonzero(document_roles == role_id)
        chosen = rng.choice(candidates, size=n_queries_per_role, replace=False)
        queries.extend(vectors[chosen] + rng.normal(0.0, 0.12, size=(len(chosen), dimension)))
        query_roles.extend([role] * len(chosen))
    return (
        vectors.astype(np.float32),
        np.asarray(queries, dtype=np.float32),
        tuple(query_roles),
        document_roles,
    )


def _aligned_policy(
    document_roles: np.ndarray,
    roles: tuple[str, ...],
    *,
    public_fraction: float,
    seed: int,
    version: str,
) -> AuthorizationPolicy:
    rng = np.random.default_rng(seed)
    visibility = np.zeros((len(roles), len(document_roles)), dtype=bool)
    public = rng.random(len(document_roles)) < public_fraction
    visibility[:, public] = True
    for role_id in range(len(roles)):
        visibility[role_id, document_roles == role_id] = True
    return AuthorizationPolicy(roles=roles, visibility=visibility, version=version)


def _scramble_policy(
    policy: AuthorizationPolicy,
    *,
    seed: int,
    version: str,
) -> AuthorizationPolicy:
    """Destroy policy-geometry alignment while preserving role selectivity."""
    rng = np.random.default_rng(seed)
    visibility = np.empty_like(policy.visibility)
    for role_id in range(len(policy.roles)):
        visibility[role_id] = rng.permutation(policy.visibility[role_id])
    return AuthorizationPolicy(roles=policy.roles, visibility=visibility, version=version)


def make_governed_scenarios(
    *,
    n_documents: int = 2400,
    dimension: int = 24,
    n_roles: int = 4,
    n_queries_per_role: int = 20,
    seed: int = 20260713,
) -> tuple[GovernedScenario, ...]:
    """Create development scenarios with controlled geometry and policy drift."""
    vectors, queries, query_roles, document_roles = _base_corpus(
        n_documents=n_documents,
        dimension=dimension,
        n_roles=n_roles,
        n_queries_per_role=n_queries_per_role,
        seed=seed,
    )
    roles = tuple(sorted(set(query_roles)))
    aligned = _aligned_policy(
        document_roles,
        roles,
        public_fraction=0.08,
        seed=seed + 1,
        version="aligned-v1",
    )
    scrambled = _scramble_policy(aligned, seed=seed + 2, version="scrambled-v2")

    rng = np.random.default_rng(seed + 3)
    drifted_vectors = vectors + rng.normal(0.0, 0.65, size=vectors.shape).astype(np.float32)
    drifted_queries = queries + rng.normal(0.0, 0.65, size=queries.shape).astype(np.float32)

    return (
        GovernedScenario(
            name="aligned",
            vectors=vectors,
            queries=queries,
            query_roles=query_roles,
            policy=aligned,
            baseline_policy=aligned,
            embedding_drift=0.0,
            description="Role policy follows the latent mixture geometry.",
        ),
        GovernedScenario(
            name="policy-scrambled",
            vectors=vectors,
            queries=queries,
            query_roles=query_roles,
            policy=scrambled,
            baseline_policy=aligned,
            embedding_drift=0.0,
            description="Role selectivity is fixed while local policy alignment is destroyed.",
        ),
        GovernedScenario(
            name="embedding-drift",
            vectors=drifted_vectors,
            queries=drifted_queries,
            query_roles=query_roles,
            policy=aligned,
            baseline_policy=aligned,
            embedding_drift=0.65,
            description="Corpus and query embeddings move under a controlled distribution shift.",
        ),
    )
