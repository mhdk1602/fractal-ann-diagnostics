"""Authorization policy primitives for governed retrieval."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AuthorizationPolicy:
    """A versioned role-by-document visibility matrix.

    ``visibility[r, i]`` is true only when role ``r`` may place document ``i``
    in model context. The matrix is the policy oracle used by the benchmark;
    production adapters should construct it from the source system's IAM data.
    """

    roles: tuple[str, ...]
    visibility: np.ndarray
    version: str = "policy-v1"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.visibility, dtype=bool)
        if matrix.ndim != 2:
            raise ValueError("visibility must have shape (n_roles, n_documents)")
        if matrix.shape[0] != len(self.roles):
            raise ValueError("roles must match the first visibility dimension")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("roles must be unique")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "visibility", matrix)

    @property
    def n_documents(self) -> int:
        return int(self.visibility.shape[1])

    def role_index(self, role: str) -> int:
        try:
            return self.roles.index(role)
        except ValueError as exc:
            raise KeyError(f"unknown role: {role!r}") from exc

    def authorized_mask(self, role: str) -> np.ndarray:
        """Return a read-only mask of documents visible to ``role``."""
        return self.visibility[self.role_index(role)]

    def authorized_ids(self, role: str) -> np.ndarray:
        return np.flatnonzero(self.authorized_mask(role))

    def selectivity(self, role: str) -> float:
        if self.n_documents == 0:
            return 0.0
        return float(self.authorized_mask(role).mean())

    @classmethod
    def from_document_roles(
        cls,
        document_roles: list[set[str] | frozenset[str]],
        roles: tuple[str, ...],
        *,
        public_token: str = "*",
        version: str = "policy-v1",
    ) -> "AuthorizationPolicy":
        """Build a visibility matrix from document ACL role sets."""
        visibility = np.zeros((len(roles), len(document_roles)), dtype=bool)
        for document_id, acl in enumerate(document_roles):
            if public_token in acl:
                visibility[:, document_id] = True
                continue
            for role in acl:
                if role not in roles:
                    raise ValueError(f"document {document_id} names unknown role {role!r}")
                visibility[roles.index(role), document_id] = True
        return cls(roles=roles, visibility=visibility, version=version)


def policy_churn(
    previous: AuthorizationPolicy,
    current: AuthorizationPolicy,
    role: str,
) -> float:
    """Fraction of role-specific decisions changed between policy versions."""
    if previous.roles != current.roles:
        raise ValueError("policy roles differ")
    if previous.n_documents != current.n_documents:
        raise ValueError("policy document counts differ")
    old = previous.authorized_mask(role)
    new = current.authorized_mask(role)
    if old.size == 0:
        return 0.0
    return float(np.not_equal(old, new).mean())
