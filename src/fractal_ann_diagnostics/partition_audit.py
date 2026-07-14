"""Deterministic query-family partition audits for confirmatory execution.

The audit treats partition independence as a graph property.  Queries are
connected when their identifiers, declared families, judged documents, or
normalized texts show that they may represent the same statistical unit.  A
connected component may belong to exactly one study stage.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Mapping
from urllib.parse import quote

from .corpora import CorpusFormatError, NormalizedCorpus

QUERY_PARTITION_AUDIT_SCHEMA = "fractal-query-partition-audit-v1"
QUERY_PARTITION_CONFIG_SCHEMA = "fractal-query-partition-config-v1"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_EDGE_RELATIONS = frozenset(
    {
        "declared-dataset-family",
        "normalized-text-exact",
        "normalized-text-near",
        "query-identifier",
        "shared-gold-document",
        "shared-relevant-document",
    }
)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must use sha256:<lowercase hex>")


@dataclass(frozen=True)
class QueryPartitionAuditConfig:
    """Pinned normalization and conservative near-duplicate thresholds."""

    schema: str = QUERY_PARTITION_CONFIG_SCHEMA
    text_normalization: str = "unicode-nfkc-casefold-alphanumeric-token-v1"
    exact_match_rule: str = "normalized-token-sequence-equality"
    near_duplicate_rule: str = "single-token-insertion-deletion-or-substitution-v1"
    minimum_near_duplicate_tokens: int = 6
    maximum_token_edit_distance: int = 1
    minimum_length_ratio_numerator: int = 17
    minimum_length_ratio_denominator: int = 20
    family_identity: str = "corpus-name-plus-declared-family"
    document_identity: str = "corpus-name-plus-external-document-id"
    stage_rule: str = "one-exact-stage-per-connected-component"

    def __post_init__(self) -> None:
        if self.schema != QUERY_PARTITION_CONFIG_SCHEMA:
            raise ValueError("unknown query partition config schema")
        if self.minimum_near_duplicate_tokens < 2:
            raise ValueError("minimum_near_duplicate_tokens must be at least two")
        if self.maximum_token_edit_distance != 1:
            raise ValueError("the pinned near-duplicate join supports one token edit")
        if (
            self.minimum_length_ratio_numerator <= 0
            or self.minimum_length_ratio_denominator <= 0
            or self.minimum_length_ratio_numerator
            > self.minimum_length_ratio_denominator
        ):
            raise ValueError("minimum token-length ratio must be in (0, 1]")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "document_identity": self.document_identity,
            "exact_match_rule": self.exact_match_rule,
            "family_identity": self.family_identity,
            "maximum_token_edit_distance": self.maximum_token_edit_distance,
            "minimum_length_ratio_denominator": self.minimum_length_ratio_denominator,
            "minimum_length_ratio_numerator": self.minimum_length_ratio_numerator,
            "minimum_near_duplicate_tokens": self.minimum_near_duplicate_tokens,
            "near_duplicate_rule": self.near_duplicate_rule,
            "schema": self.schema,
            "stage_rule": self.stage_rule,
            "text_normalization": self.text_normalization,
        }

    @property
    def sha256(self) -> str:
        """Digest of the complete threshold and algorithm configuration."""
        return _sha256(self.to_dict())


FROZEN_QUERY_PARTITION_CONFIG = QueryPartitionAuditConfig()
FROZEN_QUERY_PARTITION_CONFIG_SHA256 = FROZEN_QUERY_PARTITION_CONFIG.sha256


@dataclass(frozen=True, order=True)
class QueryPartitionMembership:
    """Canonical assignment of one normalized query to one graph component."""

    node_id: str
    corpus: str
    query_id: str
    query_family: str
    stage: str
    component_sha256: str

    def __post_init__(self) -> None:
        for name in ("node_id", "corpus", "query_id", "query_family", "stage"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        _require_sha256("component_sha256", self.component_sha256)

    def to_dict(self) -> dict[str, str]:
        return {
            "component_sha256": self.component_sha256,
            "corpus": self.corpus,
            "node_id": self.node_id,
            "query_family": self.query_family,
            "query_id": self.query_id,
            "stage": self.stage,
        }


@dataclass(frozen=True, order=True)
class QueryPartitionEdge:
    """One deterministic edge used to construct a query-family component."""

    left_node_id: str
    right_node_id: str
    relation: str
    basis_sha256: str

    def __post_init__(self) -> None:
        if not self.left_node_id or not self.right_node_id:
            raise ValueError("partition edge node IDs must be non-empty")
        if self.left_node_id >= self.right_node_id:
            raise ValueError("partition edge node IDs must be strictly ordered")
        if self.relation not in _EDGE_RELATIONS:
            raise ValueError(f"unknown query partition relation {self.relation!r}")
        _require_sha256("basis_sha256", self.basis_sha256)

    def to_dict(self) -> dict[str, str]:
        return {
            "basis_sha256": self.basis_sha256,
            "left_node_id": self.left_node_id,
            "relation": self.relation,
            "right_node_id": self.right_node_id,
        }


@dataclass(frozen=True, order=True)
class CrossStageQueryComponent:
    """A component that violates the locked partition boundary."""

    component_sha256: str
    stages: tuple[str, ...]
    node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256("component_sha256", self.component_sha256)
        if len(self.stages) < 2 or tuple(sorted(set(self.stages))) != self.stages:
            raise ValueError("a cross-stage component needs at least two sorted stages")
        if not self.node_ids or tuple(sorted(set(self.node_ids))) != self.node_ids:
            raise ValueError("cross-stage component node IDs must be unique and sorted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_sha256": self.component_sha256,
            "node_ids": list(self.node_ids),
            "stages": list(self.stages),
        }


@dataclass(frozen=True)
class QueryFamilyPartitionAudit:
    """Canonical graph audit binding every query to one stage-local component."""

    source_sha256: str
    memberships: tuple[QueryPartitionMembership, ...]
    edges: tuple[QueryPartitionEdge, ...]
    cross_stage_components: tuple[CrossStageQueryComponent, ...]
    config: QueryPartitionAuditConfig = FROZEN_QUERY_PARTITION_CONFIG
    schema: str = QUERY_PARTITION_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUERY_PARTITION_AUDIT_SCHEMA:
            raise ValueError("unknown query partition audit schema")
        if self.config != FROZEN_QUERY_PARTITION_CONFIG:
            raise ValueError("query partition audits require the frozen config")
        _require_sha256("source_sha256", self.source_sha256)
        memberships = tuple(self.memberships)
        edges = tuple(self.edges)
        crossings = tuple(self.cross_stage_components)
        object.__setattr__(self, "memberships", memberships)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "cross_stage_components", crossings)
        if not memberships:
            raise ValueError("query partition audit needs at least one query")
        if memberships != tuple(sorted(memberships)):
            raise ValueError("query partition memberships must be canonically sorted")
        if edges != tuple(sorted(edges)):
            raise ValueError("query partition edges must be canonically sorted")
        if crossings != tuple(sorted(crossings)):
            raise ValueError("cross-stage components must be canonically sorted")
        node_ids = [membership.node_id for membership in memberships]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("query partition memberships contain duplicate node IDs")
        known_nodes = set(node_ids)
        if any(
            edge.left_node_id not in known_nodes or edge.right_node_id not in known_nodes
            for edge in edges
        ):
            raise ValueError("query partition edge names an unknown node")
        components = {membership.component_sha256 for membership in memberships}
        if any(crossing.component_sha256 not in components for crossing in crossings):
            raise ValueError("cross-stage audit names an unknown component")

    @property
    def passed(self) -> bool:
        return not self.cross_stage_components

    @property
    def component_count(self) -> int:
        return len({membership.component_sha256 for membership in self.memberships})

    @property
    def config_sha256(self) -> str:
        return self.config.sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_count": self.component_count,
            "config": self.config.to_dict(),
            "config_sha256": self.config_sha256,
            "cross_stage_components": [
                crossing.to_dict() for crossing in self.cross_stage_components
            ],
            "edges": [edge.to_dict() for edge in self.edges],
            "passed": self.passed,
            "queries": [membership.to_dict() for membership in self.memberships],
            "schema": self.schema,
            "source_sha256": self.source_sha256,
        }

    def canonical_bytes(self) -> bytes:
        """Return the newline-terminated canonical JSON audit artifact."""
        return _canonical_json(self.to_dict()) + b"\n"

    @property
    def sha256(self) -> str:
        return f"sha256:{hashlib.sha256(self.canonical_bytes()).hexdigest()}"


class QueryPartitionLeakageError(CorpusFormatError):
    """Raised when a graph component spans two or more study stages."""

    def __init__(self, audit: QueryFamilyPartitionAudit) -> None:
        self.audit = audit
        first = audit.cross_stage_components[0]
        super().__init__(
            "query-family component "
            f"{first.component_sha256} crosses stages {', '.join(first.stages)}"
        )


@dataclass(frozen=True)
class _QueryNode:
    node_id: str
    corpus: str
    stage: str
    query_id: str
    query_family: str
    text_sha256: str
    normalized_text: str
    tokens: tuple[str, ...]
    relevant_documents: tuple[tuple[str, str, str], ...]
    gold_documents: tuple[tuple[str, str, str], ...]

    def source_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "gold_documents": [list(document) for document in self.gold_documents],
            "node_id": self.node_id,
            "normalized_text_sha256": _sha256(self.normalized_text),
            "query_family": self.query_family,
            "query_id": self.query_id,
            "relevant_documents": [
                list(document) for document in self.relevant_documents
            ],
            "stage": self.stage,
            "text_sha256": self.text_sha256,
        }


class _DisjointSet:
    def __init__(self, node_ids: Iterable[str]) -> None:
        self._parent = {node_id: node_id for node_id in node_ids}

    def find(self, node_id: str) -> str:
        parent = self._parent[node_id]
        while parent != self._parent[parent]:
            parent = self._parent[parent]
        while node_id != parent:
            next_node = self._parent[node_id]
            self._parent[node_id] = parent
            node_id = next_node
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self._parent[second] = first


def _node_id(corpus: str, stage: str, query_id: str) -> str:
    return "query://" + "/".join(
        quote(value, safe="") for value in (corpus, stage, query_id)
    )


def _normalize_query_text(text: str) -> tuple[str, tuple[str, ...]]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))
    if tokens:
        return " ".join(tokens), tokens
    return " ".join(normalized.split()), tokens


def _documents_for_query(
    corpus: NormalizedCorpus,
    document_ids: Iterable[int],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            {
                (
                    corpus.documents[document_id].external_id,
                    corpus.documents[document_id].source_uri,
                    corpus.documents[document_id].content_hash,
                )
                for document_id in document_ids
            }
        )
    )


def _nodes(corpora: Iterable[NormalizedCorpus]) -> tuple[_QueryNode, ...]:
    observed: dict[str, _QueryNode] = {}
    for corpus in corpora:
        if not isinstance(corpus, NormalizedCorpus):
            raise TypeError("corpora must contain NormalizedCorpus instances")
        for query in corpus.queries:
            gold_document_ids: set[int] = set()
            if query.gold_evidence is not None:
                for bundle in query.gold_evidence.alternatives:
                    gold_document_ids.update(
                        location.document_id for location in bundle.locations
                    )
            normalized_text, tokens = _normalize_query_text(query.text)
            node = _QueryNode(
                node_id=_node_id(query.corpus, query.stage, query.query_id),
                corpus=query.corpus,
                stage=query.stage,
                query_id=query.query_id,
                query_family=query.query_family,
                text_sha256=_sha256(query.text),
                normalized_text=normalized_text,
                tokens=tokens,
                relevant_documents=_documents_for_query(
                    corpus, query.relevant_document_ids
                ),
                gold_documents=_documents_for_query(corpus, gold_document_ids),
            )
            prior = observed.setdefault(node.node_id, node)
            if prior != node:
                raise CorpusFormatError(
                    f"query node {node.node_id!r} has conflicting normalized records"
                )
    if not observed:
        raise CorpusFormatError("query partition audit needs at least one query")
    return tuple(sorted(observed.values(), key=lambda node: node.node_id))


def _add_edge(
    left: str,
    right: str,
    *,
    relation: str,
    basis: Any,
    disjoint_set: _DisjointSet,
    edges: set[QueryPartitionEdge],
) -> None:
    if left == right:
        return
    left, right = sorted((left, right))
    edges.add(
        QueryPartitionEdge(
            left_node_id=left,
            right_node_id=right,
            relation=relation,
            basis_sha256=_sha256({"basis": basis, "relation": relation}),
        )
    )
    disjoint_set.union(left, right)


def _connect_groups(
    groups: Mapping[Hashable, Iterable[str]],
    *,
    relation: str,
    disjoint_set: _DisjointSet,
    edges: set[QueryPartitionEdge],
) -> None:
    for basis, values in sorted(groups.items(), key=lambda item: repr(item[0])):
        node_ids = sorted(set(values))
        if len(node_ids) < 2:
            continue
        anchor = node_ids[0]
        for other in node_ids[1:]:
            _add_edge(
                anchor,
                other,
                relation=relation,
                basis=basis,
                disjoint_set=disjoint_set,
                edges=edges,
            )


def _append_group(
    groups: dict[Hashable, list[str]], key: Hashable, node_id: str
) -> None:
    groups.setdefault(key, []).append(node_id)


def _connect_near_duplicates(
    nodes: tuple[_QueryNode, ...],
    *,
    config: QueryPartitionAuditConfig,
    disjoint_set: _DisjointSet,
    edges: set[QueryPartitionEdge],
) -> None:
    eligible = [
        node for node in nodes if len(node.tokens) >= config.minimum_near_duplicate_tokens
    ]
    exact_tokens: dict[tuple[str, ...], list[str]] = {}
    substitutions: dict[
        tuple[int, int, tuple[str, ...]], dict[str, list[str]]
    ] = {}
    for node in eligible:
        _append_group(exact_tokens, node.tokens, node.node_id)
        for position, token in enumerate(node.tokens):
            deleted = node.tokens[:position] + node.tokens[position + 1 :]
            variants = substitutions.setdefault(
                (len(node.tokens), position, deleted), {}
            )
            variants.setdefault(token, []).append(node.node_id)

    for signature, variants in sorted(substitutions.items(), key=lambda item: repr(item[0])):
        if len(variants) < 2:
            continue
        representatives = [min(node_ids) for node_ids in variants.values()]
        representatives.sort()
        anchor = representatives[0]
        for other in representatives[1:]:
            _add_edge(
                anchor,
                other,
                relation="normalized-text-near",
                basis={"edit": "substitution", "signature": signature},
                disjoint_set=disjoint_set,
                edges=edges,
            )

    numerator = config.minimum_length_ratio_numerator
    denominator = config.minimum_length_ratio_denominator
    for node in eligible:
        shorter_length = len(node.tokens) - 1
        if shorter_length < config.minimum_near_duplicate_tokens:
            continue
        if shorter_length * denominator < len(node.tokens) * numerator:
            continue
        seen_deletions: set[tuple[str, ...]] = set()
        for position in range(len(node.tokens)):
            deleted = node.tokens[:position] + node.tokens[position + 1 :]
            if deleted in seen_deletions:
                continue
            seen_deletions.add(deleted)
            shorter_nodes = exact_tokens.get(deleted)
            if not shorter_nodes:
                continue
            _add_edge(
                node.node_id,
                min(shorter_nodes),
                relation="normalized-text-near",
                basis={"edit": "insertion-deletion", "shorter_tokens": deleted},
                disjoint_set=disjoint_set,
                edges=edges,
            )


def _build_query_partition_audit(
    corpora: Iterable[NormalizedCorpus],
) -> QueryFamilyPartitionAudit:
    config = FROZEN_QUERY_PARTITION_CONFIG
    nodes = _nodes(tuple(corpora))
    disjoint_set = _DisjointSet(node.node_id for node in nodes)
    edges: set[QueryPartitionEdge] = set()

    identifiers: dict[tuple[str, str], list[str]] = {}
    families: dict[tuple[str, str], list[str]] = {}
    relevant_documents: dict[tuple[str, str], list[str]] = {}
    gold_documents: dict[tuple[str, str], list[str]] = {}
    exact_texts: dict[str, list[str]] = {}
    for node in nodes:
        _append_group(identifiers, (node.corpus, node.query_id), node.node_id)
        _append_group(families, (node.corpus, node.query_family), node.node_id)
        for external_id, _, _ in node.relevant_documents:
            _append_group(
                relevant_documents, (node.corpus, external_id), node.node_id
            )
        for external_id, _, _ in node.gold_documents:
            _append_group(gold_documents, (node.corpus, external_id), node.node_id)
        if node.normalized_text:
            _append_group(exact_texts, node.normalized_text, node.node_id)

    for groups, relation in (
        (identifiers, "query-identifier"),
        (families, "declared-dataset-family"),
        (relevant_documents, "shared-relevant-document"),
        (gold_documents, "shared-gold-document"),
        (exact_texts, "normalized-text-exact"),
    ):
        _connect_groups(
            groups,
            relation=relation,
            disjoint_set=disjoint_set,
            edges=edges,
        )
    _connect_near_duplicates(
        nodes,
        config=config,
        disjoint_set=disjoint_set,
        edges=edges,
    )

    by_root: dict[str, list[_QueryNode]] = {}
    for node in nodes:
        by_root.setdefault(disjoint_set.find(node.node_id), []).append(node)
    component_by_node: dict[str, str] = {}
    crossing_rows: list[CrossStageQueryComponent] = []
    for members in by_root.values():
        node_ids = tuple(sorted(node.node_id for node in members))
        component_sha256 = _sha256({"node_ids": list(node_ids)})
        for node_id in node_ids:
            component_by_node[node_id] = component_sha256
        stages = tuple(sorted({node.stage for node in members}))
        if len(stages) > 1:
            crossing_rows.append(
                CrossStageQueryComponent(
                    component_sha256=component_sha256,
                    stages=stages,
                    node_ids=node_ids,
                )
            )

    memberships = tuple(
        sorted(
            QueryPartitionMembership(
                node_id=node.node_id,
                corpus=node.corpus,
                query_id=node.query_id,
                query_family=node.query_family,
                stage=node.stage,
                component_sha256=component_by_node[node.node_id],
            )
            for node in nodes
        )
    )
    return QueryFamilyPartitionAudit(
        source_sha256=_sha256([node.source_dict() for node in nodes]),
        memberships=memberships,
        edges=tuple(sorted(edges)),
        cross_stage_components=tuple(sorted(crossing_rows)),
        config=config,
    )


def audit_query_partitions(
    corpora: Iterable[NormalizedCorpus],
) -> QueryFamilyPartitionAudit:
    """Build the frozen audit and reject every cross-stage component.

    The raised :class:`QueryPartitionLeakageError` retains the canonical failed
    audit in its ``audit`` attribute so a custodian can preserve diagnostic
    evidence without allowing the study to proceed.
    """
    audit = _build_query_partition_audit(corpora)
    if not audit.passed:
        raise QueryPartitionLeakageError(audit)
    return audit
