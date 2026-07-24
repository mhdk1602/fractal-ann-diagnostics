"""Offline admission of the fixed GitHub environment-control configuration.

This module never authenticates, opens a socket, or invokes ``gh``.  It admits
five caller-retained GitHub REST response bodies, verifies the closed control
contract, and publishes one canonical receipt without replacing an existing
file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY = "mhdk1602/fractal-ann-diagnostics"
REPOSITORY_API_URL = f"https://api.github.com/repos/{REPOSITORY}"
REVIEWER_LOGIN = "mhdk1602"
REVIEWER_USER_ID = 9_646_005
RECEIPT_SCHEMA = "fractal-github-environment-control-v1"
ADMIN_BYPASS_REST_ATTESTATION = "not-attestable-via-github-rest-environments-api"
APPROVAL_CLASSIFICATION = "sole-operator-recorded-self-approval"
MAX_API_RESPONSE_BYTES = 2 * 1024 * 1024

ENVIRONMENT_NAMES = ("confirmatory", "confirmatory-rehearsal")
EXPECTED_DEPLOYMENT_POLICIES = {
    "confirmatory": (
        ("tag", "confirmatory-apparatus-c0"),
        ("tag", "confirmatory-freeze-c1"),
    ),
    "confirmatory-rehearsal": (("branch", "c0-candidate/*"),),
}
API_RESPONSE_ROLES = (
    "environments-list",
    "environment-confirmatory",
    "deployment-policies-confirmatory",
    "environment-confirmatory-rehearsal",
    "deployment-policies-confirmatory-rehearsal",
)


class GitHubEnvironmentControlError(ValueError):
    """The retained API evidence or canonical receipt is inadmissible."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise GitHubEnvironmentControlError("value cannot be encoded as canonical JSON") from exc


def _strict_json(encoded: bytes, *, label: str) -> object:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise GitHubEnvironmentControlError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise GitHubEnvironmentControlError(f"{label} contains non-finite number {value!r}")

    try:
        return json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitHubEnvironmentControlError(f"cannot decode {label}: {exc}") from exc


def _object(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise GitHubEnvironmentControlError(f"{label} must be a JSON object")
    return value


def _array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise GitHubEnvironmentControlError(f"{label} must be a JSON array")
    return value


def _closed_object(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    payload = _object(value, label=label)
    observed = set(payload)
    if observed != fields:
        missing = sorted(fields - observed)
        extra = sorted(observed - fields)
        raise GitHubEnvironmentControlError(
            f"{label} must use the closed schema; missing={missing}, extra={extra}"
        )
    return payload


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise GitHubEnvironmentControlError(f"{label} must be a canonical non-empty string")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise GitHubEnvironmentControlError(f"{label} must be a positive integer")
    return value


def _digest(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise GitHubEnvironmentControlError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _secure_file_bytes(path: str | Path, *, label: str, maximum: int) -> bytes:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.name in {"", ".", ".."}:
        raise GitHubEnvironmentControlError(f"{label} must be an absolute file path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise GitHubEnvironmentControlError(f"cannot open {label} without following links") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise GitHubEnvironmentControlError(
                f"{label} must be one nonempty bounded singly linked regular file"
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum:
                raise GitHubEnvironmentControlError(f"{label} exceeds its byte limit")
        after = os.fstat(descriptor)
    except OSError as exc:
        raise GitHubEnvironmentControlError(f"cannot read {label}") from exc
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise GitHubEnvironmentControlError(f"{label} changed while it was read")
    try:
        current = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise GitHubEnvironmentControlError(f"cannot reobserve {label}") from exc
    if (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
        raise GitHubEnvironmentControlError(f"{label} path changed while it was read")
    return b"".join(chunks)


@dataclass(frozen=True)
class ApiResponseDigest:
    role: str
    canonical_sha256: str
    canonical_byte_count: int

    def __post_init__(self) -> None:
        if self.role not in API_RESPONSE_ROLES:
            raise GitHubEnvironmentControlError("API response digest has an unknown role")
        _digest(self.canonical_sha256, label=f"{self.role} canonical_sha256")
        if type(self.canonical_byte_count) is not int or self.canonical_byte_count <= 0:
            raise GitHubEnvironmentControlError(
                f"{self.role} canonical_byte_count must be a positive integer"
            )

    @classmethod
    def from_dict(cls, value: object) -> ApiResponseDigest:
        payload = _closed_object(
            value,
            frozenset({"canonical_byte_count", "canonical_sha256", "role"}),
            label="API response digest",
        )
        return cls(
            role=payload["role"],
            canonical_sha256=payload["canonical_sha256"],
            canonical_byte_count=payload["canonical_byte_count"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_byte_count": self.canonical_byte_count,
            "canonical_sha256": self.canonical_sha256,
            "role": self.role,
        }


@dataclass(frozen=True)
class RequiredReviewer:
    reviewer_type: str
    login: str
    user_id: int

    def __post_init__(self) -> None:
        if (
            self.reviewer_type != "User"
            or self.login != REVIEWER_LOGIN
            or self.user_id != REVIEWER_USER_ID
        ):
            raise GitHubEnvironmentControlError("required reviewer differs from mhdk1602")

    @classmethod
    def from_dict(cls, value: object) -> RequiredReviewer:
        payload = _closed_object(
            value,
            frozenset({"login", "reviewer_type", "user_id"}),
            label="required reviewer",
        )
        return cls(
            reviewer_type=payload["reviewer_type"],
            login=payload["login"],
            user_id=payload["user_id"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "login": self.login,
            "reviewer_type": self.reviewer_type,
            "user_id": self.user_id,
        }


@dataclass(frozen=True)
class ProtectionRule:
    rule_type: str
    prevent_self_review: bool
    reviewers: tuple[RequiredReviewer, ...]

    def __post_init__(self) -> None:
        if (
            self.rule_type != "required_reviewers"
            or self.prevent_self_review is not False
            or self.reviewers != (RequiredReviewer("User", REVIEWER_LOGIN, REVIEWER_USER_ID),)
        ):
            raise GitHubEnvironmentControlError(
                "confirmatory protection must be one self-review-permitted mhdk1602 reviewer rule"
            )

    @classmethod
    def from_dict(cls, value: object) -> ProtectionRule:
        payload = _closed_object(
            value,
            frozenset({"prevent_self_review", "reviewers", "rule_type"}),
            label="protection rule",
        )
        return cls(
            rule_type=payload["rule_type"],
            prevent_self_review=payload["prevent_self_review"],
            reviewers=tuple(
                RequiredReviewer.from_dict(row)
                for row in _array(payload["reviewers"], label="protection rule reviewers")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "prevent_self_review": self.prevent_self_review,
            "reviewers": [reviewer.to_dict() for reviewer in self.reviewers],
            "rule_type": self.rule_type,
        }


@dataclass(frozen=True)
class DeploymentPolicy:
    policy_id: int
    policy_type: str
    name: str

    def __post_init__(self) -> None:
        _positive_integer(self.policy_id, label="deployment policy ID")
        if self.policy_type not in {"branch", "tag"}:
            raise GitHubEnvironmentControlError("deployment policy type must be branch or tag")
        _text(self.name, label="deployment policy name")

    @classmethod
    def from_dict(cls, value: object) -> DeploymentPolicy:
        payload = _closed_object(
            value,
            frozenset({"name", "policy_id", "policy_type"}),
            label="deployment policy",
        )
        return cls(
            policy_id=payload["policy_id"],
            policy_type=payload["policy_type"],
            name=payload["name"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "policy_id": self.policy_id,
            "policy_type": self.policy_type,
        }


@dataclass(frozen=True)
class EnvironmentControl:
    environment_id: int
    name: str
    protected_branches: bool
    custom_branch_policies: bool
    protection_rules: tuple[ProtectionRule, ...]
    deployment_policies: tuple[DeploymentPolicy, ...]

    def __post_init__(self) -> None:
        _positive_integer(self.environment_id, label="environment ID")
        if self.name not in ENVIRONMENT_NAMES:
            raise GitHubEnvironmentControlError("receipt has an unknown environment")
        if self.protected_branches is not False or self.custom_branch_policies is not True:
            raise GitHubEnvironmentControlError(
                f"{self.name} must use only explicit custom deployment policies"
            )
        expected_rules: tuple[ProtectionRule, ...] = ()
        if self.name == "confirmatory":
            expected_rules = (
                ProtectionRule(
                    "required_reviewers",
                    False,
                    (RequiredReviewer("User", REVIEWER_LOGIN, REVIEWER_USER_ID),),
                ),
            )
        if self.protection_rules != expected_rules:
            raise GitHubEnvironmentControlError(f"{self.name} protection rules differ")
        observed = tuple((row.policy_type, row.name) for row in self.deployment_policies)
        if observed != EXPECTED_DEPLOYMENT_POLICIES[self.name]:
            raise GitHubEnvironmentControlError(f"{self.name} deployment policies differ")
        policy_ids = [row.policy_id for row in self.deployment_policies]
        if len(set(policy_ids)) != len(policy_ids):
            raise GitHubEnvironmentControlError(f"{self.name} repeats a deployment policy ID")

    @classmethod
    def from_dict(cls, value: object) -> EnvironmentControl:
        payload = _closed_object(
            value,
            frozenset(
                {
                    "custom_branch_policies",
                    "deployment_policies",
                    "environment_id",
                    "name",
                    "protected_branches",
                    "protection_rules",
                }
            ),
            label="environment control",
        )
        return cls(
            environment_id=payload["environment_id"],
            name=payload["name"],
            protected_branches=payload["protected_branches"],
            custom_branch_policies=payload["custom_branch_policies"],
            protection_rules=tuple(
                ProtectionRule.from_dict(row)
                for row in _array(payload["protection_rules"], label="protection rules")
            ),
            deployment_policies=tuple(
                DeploymentPolicy.from_dict(row)
                for row in _array(payload["deployment_policies"], label="deployment policies")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "custom_branch_policies": self.custom_branch_policies,
            "deployment_policies": [row.to_dict() for row in self.deployment_policies],
            "environment_id": self.environment_id,
            "name": self.name,
            "protected_branches": self.protected_branches,
            "protection_rules": [rule.to_dict() for rule in self.protection_rules],
        }


@dataclass(frozen=True)
class GitHubEnvironmentControlReceipt:
    environments: tuple[EnvironmentControl, ...]
    api_responses: tuple[ApiResponseDigest, ...]
    repository: str = REPOSITORY
    repository_api_url: str = REPOSITORY_API_URL
    reviewer_login: str = REVIEWER_LOGIN
    reviewer_user_id: int = REVIEWER_USER_ID
    approval_classification: str = APPROVAL_CLASSIFICATION
    independent_custody: bool = False
    admin_bypass_rest_attestation: str = ADMIN_BYPASS_REST_ATTESTATION
    schema_version: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.repository != REPOSITORY or self.repository_api_url != REPOSITORY_API_URL:
            raise GitHubEnvironmentControlError(
                "receipt repository differs from the fixed repository"
            )
        if self.reviewer_login != REVIEWER_LOGIN or self.reviewer_user_id != REVIEWER_USER_ID:
            raise GitHubEnvironmentControlError("receipt reviewer differs from mhdk1602")
        if self.approval_classification != APPROVAL_CLASSIFICATION:
            raise GitHubEnvironmentControlError("receipt approval classification differs")
        if self.independent_custody is not False:
            raise GitHubEnvironmentControlError(
                "environment approval cannot claim independent custody"
            )
        if self.admin_bypass_rest_attestation != ADMIN_BYPASS_REST_ATTESTATION:
            raise GitHubEnvironmentControlError("receipt overstates REST admin-bypass evidence")
        if self.schema_version != RECEIPT_SCHEMA:
            raise GitHubEnvironmentControlError("environment-control receipt schema differs")
        if tuple(row.name for row in self.environments) != ENVIRONMENT_NAMES:
            raise GitHubEnvironmentControlError(
                "receipt must contain exactly the two fixed environments"
            )
        if tuple(row.role for row in self.api_responses) != API_RESPONSE_ROLES:
            raise GitHubEnvironmentControlError("receipt must bind exactly the five API responses")

    @classmethod
    def from_dict(cls, value: object) -> GitHubEnvironmentControlReceipt:
        payload = _closed_object(
            value,
            frozenset(
                {
                    "admin_bypass_rest_attestation",
                    "api_responses",
                    "approval_classification",
                    "environments",
                    "independent_custody",
                    "repository",
                    "repository_api_url",
                    "reviewer_login",
                    "reviewer_user_id",
                    "schema_version",
                }
            ),
            label="GitHub environment-control receipt",
        )
        return cls(
            environments=tuple(
                EnvironmentControl.from_dict(row)
                for row in _array(payload["environments"], label="receipt environments")
            ),
            api_responses=tuple(
                ApiResponseDigest.from_dict(row)
                for row in _array(payload["api_responses"], label="receipt API responses")
            ),
            repository=payload["repository"],
            repository_api_url=payload["repository_api_url"],
            reviewer_login=payload["reviewer_login"],
            reviewer_user_id=payload["reviewer_user_id"],
            approval_classification=payload["approval_classification"],
            independent_custody=payload["independent_custody"],
            admin_bypass_rest_attestation=payload["admin_bypass_rest_attestation"],
            schema_version=payload["schema_version"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "admin_bypass_rest_attestation": self.admin_bypass_rest_attestation,
            "api_responses": [row.to_dict() for row in self.api_responses],
            "approval_classification": self.approval_classification,
            "environments": [row.to_dict() for row in self.environments],
            "independent_custody": self.independent_custody,
            "repository": self.repository,
            "repository_api_url": self.repository_api_url,
            "reviewer_login": self.reviewer_login,
            "reviewer_user_id": self.reviewer_user_id,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()


@dataclass(frozen=True)
class GitHubEnvironmentApiSnapshots:
    environments_list: Path
    confirmatory_environment: Path
    confirmatory_deployment_policies: Path
    rehearsal_environment: Path
    rehearsal_deployment_policies: Path

    def role_paths(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("environments-list", self.environments_list),
            ("environment-confirmatory", self.confirmatory_environment),
            ("deployment-policies-confirmatory", self.confirmatory_deployment_policies),
            ("environment-confirmatory-rehearsal", self.rehearsal_environment),
            (
                "deployment-policies-confirmatory-rehearsal",
                self.rehearsal_deployment_policies,
            ),
        )


def _load_api_snapshot(path: Path, *, role: str) -> tuple[Mapping[str, Any], ApiResponseDigest]:
    encoded = _secure_file_bytes(path, label=role, maximum=MAX_API_RESPONSE_BYTES)
    payload = _object(_strict_json(encoded, label=role), label=role)
    canonical = _canonical_bytes(payload)
    return payload, ApiResponseDigest(
        role=role,
        canonical_sha256=hashlib.sha256(canonical).hexdigest(),
        canonical_byte_count=len(canonical),
    )


def _environment_url(name: str) -> str:
    return f"{REPOSITORY_API_URL}/environments/{name}"


def _environment_from_api(
    *,
    name: str,
    summary: Mapping[str, Any],
    detail: Mapping[str, Any],
    policy_response: Mapping[str, Any],
) -> EnvironmentControl:
    expected_url = _environment_url(name)
    environment_id = _positive_integer(detail.get("id"), label=f"{name} environment ID")
    if (
        detail.get("name") != name
        or detail.get("url") != expected_url
        or summary.get("name") != name
        or summary.get("id") != environment_id
        or summary.get("url") != expected_url
    ):
        raise GitHubEnvironmentControlError(
            f"{name} environment identity differs from the fixed repository"
        )
    branch_policy = _object(
        detail.get("deployment_branch_policy"),
        label=f"{name} deployment_branch_policy",
    )
    protected_branches = branch_policy.get("protected_branches")
    custom_branch_policies = branch_policy.get("custom_branch_policies")
    if type(protected_branches) is not bool or type(custom_branch_policies) is not bool:
        raise GitHubEnvironmentControlError(f"{name} deployment branch policy is incomplete")

    protection_values = _array(
        detail.get("protection_rules"),
        label=f"{name} protection_rules",
    )
    protection_rules: tuple[ProtectionRule, ...] = ()
    if name == "confirmatory":
        if len(protection_values) != 1:
            raise GitHubEnvironmentControlError(
                "confirmatory must have exactly one required-reviewers rule"
            )
        rule = _object(protection_values[0], label="confirmatory protection rule")
        if rule.get("type") != "required_reviewers" or rule.get("prevent_self_review") is not False:
            raise GitHubEnvironmentControlError(
                "confirmatory must permit sole-operator self-review"
            )
        reviewer_values = _array(rule.get("reviewers"), label="confirmatory reviewers")
        if len(reviewer_values) != 1:
            raise GitHubEnvironmentControlError("confirmatory must have exactly one reviewer")
        reviewer_binding = _object(reviewer_values[0], label="confirmatory reviewer binding")
        reviewer = _object(
            reviewer_binding.get("reviewer"),
            label="confirmatory reviewer",
        )
        protection_rules = (
            ProtectionRule(
                rule_type="required_reviewers",
                prevent_self_review=False,
                reviewers=(
                    RequiredReviewer(
                        reviewer_type=reviewer_binding.get("type"),
                        login=reviewer.get("login"),
                        user_id=reviewer.get("id"),
                    ),
                ),
            ),
        )
    elif protection_values:
        raise GitHubEnvironmentControlError("confirmatory-rehearsal must have no protection rules")

    total_count = policy_response.get("total_count")
    policy_values = _array(
        policy_response.get("branch_policies"),
        label=f"{name} branch_policies",
    )
    if type(total_count) is not int or total_count != len(policy_values):
        raise GitHubEnvironmentControlError(f"{name} deployment policy count is incomplete")
    policies: list[DeploymentPolicy] = []
    for value in policy_values:
        policy = _object(value, label=f"{name} deployment policy")
        policy_id = _positive_integer(policy.get("id"), label=f"{name} deployment policy ID")
        expected_policy_url = f"{expected_url}/deployment-branch-policies/{policy_id}"
        if policy.get("url") != expected_policy_url:
            raise GitHubEnvironmentControlError(
                f"{name} deployment policy URL differs from the fixed repository"
            )
        policies.append(
            DeploymentPolicy(
                policy_id=policy_id,
                policy_type=policy.get("type"),
                name=policy.get("name"),
            )
        )
    policies.sort(key=lambda row: (row.policy_type, row.name))
    return EnvironmentControl(
        environment_id=environment_id,
        name=name,
        protected_branches=protected_branches,
        custom_branch_policies=custom_branch_policies,
        protection_rules=protection_rules,
        deployment_policies=tuple(policies),
    )


def verify_github_environment_controls(
    snapshots: GitHubEnvironmentApiSnapshots,
) -> GitHubEnvironmentControlReceipt:
    """Admit the exact offline REST readback and return its deterministic receipt."""

    if not isinstance(snapshots, GitHubEnvironmentApiSnapshots):
        raise GitHubEnvironmentControlError("snapshots must use GitHubEnvironmentApiSnapshots")
    payloads: dict[str, Mapping[str, Any]] = {}
    digests: list[ApiResponseDigest] = []
    for role, path in snapshots.role_paths():
        payload, digest = _load_api_snapshot(path, role=role)
        payloads[role] = payload
        digests.append(digest)

    environment_list = payloads["environments-list"]
    summaries = _array(environment_list.get("environments"), label="environments list")
    if environment_list.get("total_count") != 2 or len(summaries) != 2:
        raise GitHubEnvironmentControlError("repository must contain exactly two environments")
    indexed: dict[str, Mapping[str, Any]] = {}
    for value in summaries:
        summary = _object(value, label="environment summary")
        name = _text(summary.get("name"), label="environment summary name")
        if name in indexed:
            raise GitHubEnvironmentControlError(f"environments list repeats {name!r}")
        indexed[name] = summary
    if tuple(sorted(indexed)) != ENVIRONMENT_NAMES:
        raise GitHubEnvironmentControlError("repository environment names differ from the contract")

    environments = (
        _environment_from_api(
            name="confirmatory",
            summary=indexed["confirmatory"],
            detail=payloads["environment-confirmatory"],
            policy_response=payloads["deployment-policies-confirmatory"],
        ),
        _environment_from_api(
            name="confirmatory-rehearsal",
            summary=indexed["confirmatory-rehearsal"],
            detail=payloads["environment-confirmatory-rehearsal"],
            policy_response=payloads["deployment-policies-confirmatory-rehearsal"],
        ),
    )
    return GitHubEnvironmentControlReceipt(
        environments=environments,
        api_responses=tuple(digests),
    )


def _receipt_parent(path: Path) -> int:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise GitHubEnvironmentControlError("receipt target must be an absolute file path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError as exc:
        raise GitHubEnvironmentControlError("cannot open the receipt parent without links") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise GitHubEnvironmentControlError("receipt parent must be one real directory")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        os.close(descriptor)
        raise GitHubEnvironmentControlError("receipt parent must be owned by the runner")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        os.close(descriptor)
        raise GitHubEnvironmentControlError("receipt parent cannot be group- or other-writable")
    return descriptor


def write_github_environment_control_receipt(
    receipt: GitHubEnvironmentControlReceipt,
    target: str | Path,
) -> None:
    """Atomically link one closed receipt into place without replacement."""

    if not isinstance(receipt, GitHubEnvironmentControlReceipt):
        raise GitHubEnvironmentControlError("receipt must be a GitHubEnvironmentControlReceipt")
    path = Path(target)
    encoded = receipt.canonical_file_bytes()
    parent = _receipt_parent(path)
    temporary_name = f".{path.name}.{secrets.token_hex(12)}.tmp"
    temporary_created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent)
        temporary_created = True
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise GitHubEnvironmentControlError("receipt write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise GitHubEnvironmentControlError(
                "environment-control receipt already exists"
            ) from exc
        os.unlink(temporary_name, dir_fd=parent)
        temporary_created = False
        os.fsync(parent)
    except OSError as exc:
        raise GitHubEnvironmentControlError(
            "cannot atomically publish environment-control receipt"
        ) from exc
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent)
            except FileNotFoundError:
                pass
        os.close(parent)
    if (
        _secure_file_bytes(
            path,
            label="published environment-control receipt",
            maximum=MAX_API_RESPONSE_BYTES,
        )
        != encoded
    ):
        raise GitHubEnvironmentControlError("published environment-control receipt changed")


def load_github_environment_control_receipt(
    path: str | Path,
) -> GitHubEnvironmentControlReceipt:
    """Load and revalidate one exact canonical receipt file."""

    encoded = _secure_file_bytes(
        path,
        label="GitHub environment-control receipt",
        maximum=MAX_API_RESPONSE_BYTES,
    )
    payload = _strict_json(encoded, label="GitHub environment-control receipt")
    receipt = GitHubEnvironmentControlReceipt.from_dict(payload)
    if encoded != receipt.canonical_file_bytes():
        raise GitHubEnvironmentControlError(
            "GitHub environment-control receipt is not canonical JSON plus one LF"
        )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fractal-github-environment-control")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser(
        "verify",
        help="verify five retained REST response bodies and publish one receipt",
    )
    verify.add_argument("--environments-list", type=Path, required=True)
    verify.add_argument("--confirmatory-environment", type=Path, required=True)
    verify.add_argument("--confirmatory-deployment-policies", type=Path, required=True)
    verify.add_argument("--rehearsal-environment", type=Path, required=True)
    verify.add_argument("--rehearsal-deployment-policies", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    readback = subparsers.add_parser(
        "readback",
        help="revalidate and print one canonical environment-control receipt",
    )
    readback.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        receipt = verify_github_environment_controls(
            GitHubEnvironmentApiSnapshots(
                environments_list=args.environments_list,
                confirmatory_environment=args.confirmatory_environment,
                confirmatory_deployment_policies=args.confirmatory_deployment_policies,
                rehearsal_environment=args.rehearsal_environment,
                rehearsal_deployment_policies=args.rehearsal_deployment_policies,
            )
        )
        write_github_environment_control_receipt(receipt, args.receipt)
        print(f"environment-control receipt: {args.receipt}")
        print(f"receipt sha256: {receipt.receipt_sha256}")
        print(f"receipt file sha256: {receipt.file_sha256}")
        return 0
    if args.command == "readback":
        receipt = load_github_environment_control_receipt(args.receipt)
        print(receipt.canonical_bytes().decode("ascii"))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
