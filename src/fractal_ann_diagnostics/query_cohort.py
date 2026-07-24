"""Shared outcome-blind query-family cohort selection primitives.

Both the online trial builder and the separated label custodian must derive the
same representative query and nested trial identities without exchanging raw
query identifiers.  Keeping the ranking and source-value functions here makes
that equality a code-level invariant instead of a duplicated convention.
"""

from __future__ import annotations

import hashlib
import json

FAMILY_SELECTION_ALGORITHM = "sha256-rank-v1"
FAMILY_SELECTION_DOMAIN = "fractal-custody-family-selection-v1"
REPRESENTATIVE_SELECTION_ALGORITHM = "sha256-rank-v1"
REPRESENTATIVE_SELECTION_DOMAIN = "fractal-custody-representative-selection-v1"
NESTED_TRIAL_SOURCE_DOMAIN = "fractal-custody-nested-trial-v1"
NESTED_ROWS_PER_FAMILY = 3


def _length_prefixed_sha256(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def family_selection_rank(
    *,
    corpus: str,
    stage: str,
    selection_seed_sha256: str,
    component_sha256: str,
) -> str:
    """Return the registered outcome-blind rank for one assignment family."""

    return _length_prefixed_sha256(
        (
            FAMILY_SELECTION_DOMAIN,
            FAMILY_SELECTION_ALGORITHM,
            corpus,
            stage,
            selection_seed_sha256,
            component_sha256,
        )
    )


def representative_selection_rank(
    *,
    corpus: str,
    stage: str,
    selection_seed_sha256: str,
    component_sha256: str,
    query_id_sha256: str,
) -> str:
    """Return the registered rank for one representative-query candidate."""

    return _length_prefixed_sha256(
        (
            REPRESENTATIVE_SELECTION_DOMAIN,
            REPRESENTATIVE_SELECTION_ALGORITHM,
            corpus,
            stage,
            selection_seed_sha256,
            component_sha256,
            query_id_sha256,
        )
    )


def nested_trial_source_value(source_id: str, nested_index: int) -> str:
    """Encode the custody/runtime HMAC source value for one nested trial."""

    return json.dumps(
        [NESTED_TRIAL_SOURCE_DOMAIN, source_id, nested_index],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
