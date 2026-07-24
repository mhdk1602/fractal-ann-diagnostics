"""Paired-world observations for authorization noninterference tests."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .controller import GovernedResult
from .retrieval import SearchResult


@dataclass(frozen=True)
class SearchObservation:
    strategy: str | None
    returned_ids: tuple[int, ...]
    distances: tuple[float, ...]
    unauthorized_candidates: int
    unauthorized_context: int
    shortfall: int


@dataclass(frozen=True)
class GovernedObservation:
    action: str
    risk_score: float
    reasons: tuple[str, ...]
    policy_version: str
    initial_mask_sha256: str | None
    final_mask_sha256: str | None
    geometry_source: str | None
    geometry_values: tuple[float, ...]
    lid_by_scale: tuple[tuple[int, float], ...]
    search: SearchObservation


@dataclass(frozen=True)
class NoninterferenceReport:
    equivalent: bool
    differences: tuple[str, ...]


def observe_search(search: SearchResult | None) -> SearchObservation:
    if search is None:
        return SearchObservation(
            strategy=None,
            returned_ids=(),
            distances=(),
            unauthorized_candidates=0,
            unauthorized_context=0,
            shortfall=0,
        )
    return SearchObservation(
        strategy=search.strategy,
        returned_ids=tuple(int(document_id) for document_id in search.ids),
        distances=tuple(float(distance) for distance in search.distances),
        unauthorized_candidates=search.unauthorized_candidates,
        unauthorized_context=search.unauthorized_context,
        shortfall=search.shortfall,
    )


def _mask_digest(mask: np.ndarray | None) -> str | None:
    if mask is None:
        return None
    return sha256(np.asarray(mask, dtype=bool).tobytes()).hexdigest()


def observe_governed_result(result: GovernedResult) -> GovernedObservation:
    """Project a run onto deterministic fields allowed to depend on authorized data."""
    geometry = result.geometry
    initial = result.initial_authorization
    final = result.final_authorization
    if result.search is not None and final is None:
        raise ValueError("emitted search results require a final authorization decision")
    return GovernedObservation(
        action=result.decision.action,
        risk_score=result.decision.risk_score,
        reasons=result.decision.reasons,
        policy_version=result.decision.policy_version,
        initial_mask_sha256=(
            _mask_digest(initial.authorized_mask) if initial is not None else None
        ),
        final_mask_sha256=(_mask_digest(final.authorized_mask) if final is not None else None),
        geometry_source=(geometry.source if geometry is not None else None),
        geometry_values=(
            tuple(float(value) for value in geometry.as_array()) if geometry is not None else ()
        ),
        lid_by_scale=(geometry.lid_by_scale if geometry is not None else ()),
        search=observe_search(result.search),
    )


def compare_search_observations(
    first: SearchObservation,
    second: SearchObservation,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> tuple[str, ...]:
    differences: list[str] = []
    for field in (
        "strategy",
        "returned_ids",
        "unauthorized_candidates",
        "unauthorized_context",
        "shortfall",
    ):
        if getattr(first, field) != getattr(second, field):
            differences.append(f"search.{field}")
    if len(first.distances) != len(second.distances) or not np.allclose(
        first.distances,
        second.distances,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
        differences.append("search.distances")
    return tuple(differences)


def compare_governed_observations(
    first: GovernedObservation,
    second: GovernedObservation,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> NoninterferenceReport:
    """Compare paired worlds after excluding timestamps, IDs, and raw timing noise."""
    differences: list[str] = []
    for field in (
        "action",
        "reasons",
        "policy_version",
        "initial_mask_sha256",
        "final_mask_sha256",
        "geometry_source",
    ):
        if getattr(first, field) != getattr(second, field):
            differences.append(field)
    if not np.isclose(
        first.risk_score,
        second.risk_score,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
        differences.append("risk_score")
    if len(first.geometry_values) != len(second.geometry_values) or not np.allclose(
        first.geometry_values,
        second.geometry_values,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
        differences.append("geometry_values")
    first_lids = np.asarray([value for _, value in first.lid_by_scale], dtype=float)
    second_lids = np.asarray([value for _, value in second.lid_by_scale], dtype=float)
    if [scale for scale, _ in first.lid_by_scale] != [
        scale for scale, _ in second.lid_by_scale
    ] or not np.allclose(
        first_lids,
        second_lids,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
        differences.append("lid_by_scale")
    differences.extend(
        compare_search_observations(first.search, second.search, rtol=rtol, atol=atol)
    )
    return NoninterferenceReport(
        equivalent=not differences,
        differences=tuple(differences),
    )
