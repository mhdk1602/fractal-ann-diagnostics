"""Authorization policy primitives for governed retrieval."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Protocol
from uuid import uuid4

import numpy as np

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def policy_environment_sha256(environment: Mapping[str, object] | None) -> str:
    """Bind a decision to the exact JSON policy environment supplied by the caller."""
    if environment is not None and not isinstance(environment, Mapping):
        raise ValueError("policy environment must be a mapping")
    try:
        payload = json.dumps(
            dict(environment or {}),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("policy environment must be finite JSON data") from exc
    return hashlib.sha256(payload).hexdigest()


EMPTY_POLICY_ENVIRONMENT_SHA256 = policy_environment_sha256(None)


def policy_document_universe_sha256(document_ids: Iterable[object]) -> str:
    """Hash the ordered, immutable document identities governed by a policy."""
    identifiers = tuple(str(identifier) for identifier in document_ids)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("document_ids must be unique")
    digest = hashlib.sha256()
    for identifier in identifiers:
        encoded = identifier.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def positional_document_universe_sha256(n_documents: int) -> str:
    if type(n_documents) is not int or n_documents < 0:
        raise ValueError("n_documents must be a non-negative integer")
    return policy_document_universe_sha256(range(n_documents))


def policy_request_sha256(
    *,
    subject: str,
    action: str,
    environment_sha256: str,
    document_universe_sha256: str,
    request_nonce: str,
) -> str:
    """Bind one policy response to the exact request and a fresh nonce."""
    payload = {
        "action": action,
        "document_universe_sha256": document_universe_sha256,
        "environment_sha256": environment_sha256,
        "request_nonce": request_nonce,
        "subject": subject,
    }
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("policy request binding contains invalid values") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PolicyDecision:
    """One auditable authorization decision over a document universe."""

    subject: str
    action: str
    policy_version: str
    authorized_mask: np.ndarray
    available: bool = True
    reason: str = "policy evaluated"
    decision_id: str = ""
    environment_sha256: str = EMPTY_POLICY_ENVIRONMENT_SHA256
    document_universe_sha256: str = ""
    request_nonce: str = ""
    request_sha256: str = ""

    def __post_init__(self) -> None:
        mask = np.asarray(self.authorized_mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("authorized_mask must have shape (n_documents,)")
        mask = mask.copy()
        mask.setflags(write=False)
        object.__setattr__(self, "authorized_mask", mask)
        for name in ("subject", "action", "policy_version", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if _SHA256.fullmatch(self.environment_sha256) is None:
            raise ValueError("environment_sha256 must be a lowercase SHA-256 digest")
        universe_digest = self.document_universe_sha256 or positional_document_universe_sha256(
            len(mask)
        )
        if _SHA256.fullmatch(universe_digest) is None:
            raise ValueError("document_universe_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "document_universe_sha256", universe_digest)
        request_nonce = self.request_nonce or uuid4().hex
        if not isinstance(request_nonce, str) or not request_nonce.strip():
            raise ValueError("request_nonce must be a non-empty string")
        object.__setattr__(self, "request_nonce", request_nonce)
        expected_request_digest = policy_request_sha256(
            subject=self.subject,
            action=self.action,
            environment_sha256=self.environment_sha256,
            document_universe_sha256=universe_digest,
            request_nonce=request_nonce,
        )
        request_digest = self.request_sha256 or expected_request_digest
        if _SHA256.fullmatch(request_digest) is None:
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest")
        if not hmac.compare_digest(request_digest, expected_request_digest):
            raise ValueError("request_sha256 does not bind this policy decision")
        object.__setattr__(self, "request_sha256", request_digest)
        if not self.decision_id:
            object.__setattr__(self, "decision_id", str(uuid4()))

    @property
    def authorized_count(self) -> int:
        return int(self.authorized_mask.sum())

    @property
    def selectivity(self) -> float:
        if self.authorized_mask.size == 0:
            return 0.0
        return float(self.authorized_mask.mean())

    def permits(self, document_ids: np.ndarray) -> bool:
        ids = np.asarray(document_ids, dtype=np.int64)
        if ids.size == 0:
            return True
        if ids.min() < 0 or ids.max() >= self.authorized_mask.size:
            return False
        return bool(self.authorized_mask[ids].all())


class PolicyDecisionPoint(Protocol):
    """Runtime authorization interface consumed by ``GovernedRetriever``."""

    @property
    def n_documents(self) -> int: ...

    @property
    def document_universe_sha256(self) -> str: ...

    def decide(
        self,
        subject: str,
        *,
        action: str = "retrieve",
        environment: Mapping[str, object] | None = None,
    ) -> PolicyDecision: ...


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
    document_universe_sha256: str = ""

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
        universe_digest = self.document_universe_sha256 or positional_document_universe_sha256(
            matrix.shape[1]
        )
        if _SHA256.fullmatch(universe_digest) is None:
            raise ValueError("document_universe_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "document_universe_sha256", universe_digest)

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

    def decide(
        self,
        subject: str,
        *,
        action: str = "retrieve",
        environment: Mapping[str, object] | None = None,
    ) -> PolicyDecision:
        """Evaluate this immutable snapshot as a policy decision point."""
        environment_digest = policy_environment_sha256(environment)
        try:
            mask = self.authorized_mask(subject)
            reason = "policy evaluated"
        except KeyError:
            mask = np.zeros(self.n_documents, dtype=bool)
            reason = "unknown subject; deny by default"
        return PolicyDecision(
            subject=subject,
            action=action,
            policy_version=self.version,
            authorized_mask=mask,
            reason=reason,
            environment_sha256=environment_digest,
            document_universe_sha256=self.document_universe_sha256,
        )

    @classmethod
    def from_document_roles(
        cls,
        document_roles: list[set[str] | frozenset[str]],
        roles: tuple[str, ...],
        *,
        public_token: str = "*",
        version: str = "policy-v1",
        document_universe_sha256: str = "",
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
        return cls(
            roles=roles,
            visibility=visibility,
            version=version,
            document_universe_sha256=document_universe_sha256,
        )


class InMemoryPolicyDecisionPoint:
    """Thread-safe mutable PDP for conformance tests and local adapters.

    Production integrations should implement :class:`PolicyDecisionPoint` by
    consulting their authoritative IAM service. Replacing a policy with changed
    decisions requires a new version so callers can reject time-of-check/time-of-use
    races.
    """

    def __init__(self, policy: AuthorizationPolicy) -> None:
        self._policy = policy
        self._available = True
        self._lock = RLock()

    @property
    def n_documents(self) -> int:
        with self._lock:
            return self._policy.n_documents

    @property
    def version(self) -> str:
        with self._lock:
            return self._policy.version

    @property
    def document_universe_sha256(self) -> str:
        with self._lock:
            return self._policy.document_universe_sha256

    def replace(self, policy: AuthorizationPolicy) -> None:
        with self._lock:
            if policy.n_documents != self._policy.n_documents:
                raise ValueError("replacement policy must preserve the document universe")
            if policy.document_universe_sha256 != self._policy.document_universe_sha256:
                raise ValueError("replacement policy must preserve document universe identity")
            changed = policy.roles != self._policy.roles or not np.array_equal(
                policy.visibility,
                self._policy.visibility,
            )
            if changed and policy.version == self._policy.version:
                raise ValueError("changed authorization decisions require a new version")
            self._policy = policy

    def set_available(self, available: bool) -> None:
        with self._lock:
            self._available = bool(available)

    def decide(
        self,
        subject: str,
        *,
        action: str = "retrieve",
        environment: Mapping[str, object] | None = None,
    ) -> PolicyDecision:
        with self._lock:
            policy = self._policy
            available = self._available
        if not available:
            environment_digest = policy_environment_sha256(environment)
            return PolicyDecision(
                subject=subject,
                action=action,
                policy_version=policy.version,
                authorized_mask=np.zeros(policy.n_documents, dtype=bool),
                available=False,
                reason="policy decision point unavailable; deny by default",
                environment_sha256=environment_digest,
                document_universe_sha256=policy.document_universe_sha256,
            )
        return policy.decide(subject, action=action, environment=environment)


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
    if previous.document_universe_sha256 != current.document_universe_sha256:
        raise ValueError("policy document universe identities differ")
    old = previous.authorized_mask(role)
    new = current.authorized_mask(role)
    if old.size == 0:
        return 0.0
    return float(np.not_equal(old, new).mean())
