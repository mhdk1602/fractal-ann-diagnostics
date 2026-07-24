"""Neutral public-C1 schema for the five production workload specifications."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

PRODUCTION_WORKLOAD_SPEC_SCHEMA = "fractal-production-corpus-workload-spec-v1"
PRODUCTION_WORKLOADS_UNRESOLVED = "unresolved-before-c1"
FIXED_PRODUCTION_CORPORA = (
    "scifact",
    "hotpotqa-fullwiki",
    "t2-ragbench",
    "bright",
    "miracl-transfer",
)

PRODUCTION_WORKLOAD_REGISTRATION_FIELDS = frozenset({"canonical_file_sha256", "corpus_id", "spec"})
PRODUCTION_WORKLOAD_SPEC_FIELDS = frozenset(
    {
        "artifact_root",
        "artifact_tree_sha256",
        "authorized_index_store_root",
        "authorized_index_store_tree_sha256",
        "available_family_count",
        "code_commit",
        "corpus_id",
        "embedding_store_root",
        "embedding_store_tree_sha256",
        "expected_authorized_index_store_receipt_sha256",
        "expected_policy_intervention_receipt_sha256",
        "expected_pseudonym_key_sha256",
        "factory_artifact_tree_sha256",
        "factory_config_sha256",
        "factory_suite_receipt_sha256",
        "feature_bindings",
        "index_bundle_receipt_path",
        "index_bundle_receipt_sha256",
        "online_execution_plan_sha256",
        "online_execution_tree_sha256",
        "partition_audit_file_sha256",
        "partition_audit_path",
        "partition_audit_sha256",
        "policy_bundle_receipt_path",
        "policy_bundle_receipt_sha256",
        "policy_intervention_root",
        "policy_intervention_tree_sha256",
        "pseudonym_key_path",
        "query_package_root",
        "query_package_tree_sha256",
        "query_receipt_sha256",
        "runner_identity",
        "runner_image",
        "runner_platform",
        "schema_version",
        "selected_family_count",
        "sharded_execution_plan_file_sha256",
        "staged_root",
        "staged_tree_sha256",
        "trial_runtime_admission_receipt_file_sha256",
    }
)
PRODUCTION_WORKLOAD_FEATURE_BINDING_FIELDS = frozenset(
    {
        "backend",
        "drift_family",
        "group_order",
        "policy_complexity",
        "policy_state",
        "repetition",
        "subject",
        "version_lag",
    }
)

_PATH_FIELDS = frozenset(
    {
        "artifact_root",
        "authorized_index_store_root",
        "embedding_store_root",
        "index_bundle_receipt_path",
        "partition_audit_path",
        "policy_bundle_receipt_path",
        "policy_intervention_root",
        "pseudonym_key_path",
        "query_package_root",
        "staged_root",
    }
)
_SHA256_FIELDS = frozenset(
    field for field in PRODUCTION_WORKLOAD_SPEC_FIELDS if field.endswith("_sha256")
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OCI_IMAGE = re.compile(r"^[^\s/@]+(?:/[^\s/@]+)+@sha256:[0-9a-f]{64}$")


class ProductionWorkloadRegistrationError(ValueError):
    """Raised when a public C1 workload registration is not exact."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the one registered JSON encoding, excluding the terminal newline."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionWorkloadRegistrationError(
            "production workload must be finite canonical JSON"
        ) from exc


def canonical_workload_file_bytes(spec: Mapping[str, Any]) -> bytes:
    """Return the exact bytes written to a production WorkloadSpec file."""
    return canonical_json_bytes(spec) + b"\n"


def production_workload_file_sha256(spec: Mapping[str, Any]) -> str:
    """Hash the registered WorkloadSpec file, including its terminal newline."""
    return hashlib.sha256(canonical_workload_file_bytes(spec)).hexdigest()


def _closed_mapping(value: object, *, fields: frozenset[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ProductionWorkloadRegistrationError(f"{path} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise ProductionWorkloadRegistrationError(
            f"{path} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _canonical_text(value: object, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ProductionWorkloadRegistrationError(f"{path} must be canonical non-empty text")
    return value


def _canonical_absolute_path(value: object, *, path: str) -> str:
    text = _canonical_text(value, path=path)
    pure = PurePosixPath(text)
    if (
        not text.startswith("/")
        or "\\" in text
        or pure.as_posix() != text
        or any(part in {"", ".", ".."} for part in pure.parts[1:])
    ):
        raise ProductionWorkloadRegistrationError(f"{path} must be a canonical absolute POSIX path")
    return text


def _positive_integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionWorkloadRegistrationError(f"{path} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionWorkloadRegistrationError(f"{path} must be a non-negative integer")
    return value


def _nonnegative_number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionWorkloadRegistrationError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ProductionWorkloadRegistrationError(f"{path} must be finite and non-negative")
    return result


def _validate_feature_bindings(value: object, *, path: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ProductionWorkloadRegistrationError(f"{path} must be a non-empty array")
    block_keys: set[tuple[int, str, int, str]] = set()
    for position, item in enumerate(value):
        item_path = f"{path}[{position}]"
        row = _closed_mapping(
            item,
            fields=PRODUCTION_WORKLOAD_FEATURE_BINDING_FIELDS,
            path=item_path,
        )
        group_order = _nonnegative_integer(row["group_order"], path=f"{item_path}.group_order")
        repetition = _nonnegative_integer(row["repetition"], path=f"{item_path}.repetition")
        subject = _canonical_text(row["subject"], path=f"{item_path}.subject")
        policy_state = _canonical_text(row["policy_state"], path=f"{item_path}.policy_state")
        _canonical_text(row["backend"], path=f"{item_path}.backend")
        _canonical_text(row["drift_family"], path=f"{item_path}.drift_family")
        _nonnegative_number(row["version_lag"], path=f"{item_path}.version_lag")
        _nonnegative_number(row["policy_complexity"], path=f"{item_path}.policy_complexity")
        block_key = (group_order, subject, repetition, policy_state)
        if block_key in block_keys:
            raise ProductionWorkloadRegistrationError(f"{path} repeats a schedule block key")
        block_keys.add(block_key)


def _validate_workload_spec(
    value: object,
    *,
    corpus_id: str,
    registered_selected_family_count: int,
    sealed_execution: Mapping[str, Any],
    path: str,
) -> Mapping[str, Any]:
    spec = _closed_mapping(value, fields=PRODUCTION_WORKLOAD_SPEC_FIELDS, path=path)
    if spec["schema_version"] != PRODUCTION_WORKLOAD_SPEC_SCHEMA:
        raise ProductionWorkloadRegistrationError(f"{path}.schema_version differs")
    if spec["corpus_id"] != corpus_id:
        raise ProductionWorkloadRegistrationError(f"{path}.corpus_id differs from its wrapper")
    available = _positive_integer(
        spec["available_family_count"], path=f"{path}.available_family_count"
    )
    selected = _positive_integer(
        spec["selected_family_count"], path=f"{path}.selected_family_count"
    )
    if selected > available:
        raise ProductionWorkloadRegistrationError(
            f"{path}.selected_family_count exceeds available_family_count"
        )
    if selected != registered_selected_family_count:
        raise ProductionWorkloadRegistrationError(
            f"{path}.selected_family_count differs from analysis.power.selected_families_per_corpus"
        )
    for field in _SHA256_FIELDS:
        if not isinstance(spec[field], str) or _SHA256.fullmatch(spec[field]) is None:
            raise ProductionWorkloadRegistrationError(
                f"{path}.{field} must be a lowercase SHA-256 digest"
            )
    for field in _PATH_FIELDS:
        _canonical_absolute_path(spec[field], path=f"{path}.{field}")
    if spec["runner_platform"] != "linux/arm64":
        raise ProductionWorkloadRegistrationError(f"{path}.runner_platform must equal linux/arm64")
    if (
        not isinstance(spec["runner_image"], str)
        or _OCI_IMAGE.fullmatch(spec["runner_image"]) is None
    ):
        raise ProductionWorkloadRegistrationError(f"{path}.runner_image must be digest-qualified")
    if (
        not isinstance(spec["code_commit"], str)
        or _GIT_COMMIT.fullmatch(spec["code_commit"]) is None
    ):
        raise ProductionWorkloadRegistrationError(f"{path}.code_commit must be one full Git commit")
    _canonical_text(spec["runner_identity"], path=f"{path}.runner_identity")
    for field in ("runner_image", "runner_identity", "code_commit"):
        if spec[field] != sealed_execution.get(field):
            raise ProductionWorkloadRegistrationError(
                f"{path}.{field} differs from sealed_execution.{field}"
            )
    _validate_feature_bindings(spec["feature_bindings"], path=f"{path}.feature_bindings")
    canonical_json_bytes(spec)
    return spec


def validate_production_workload_registrations(
    value: object,
    *,
    frozen: bool,
    registered_selected_family_count: int,
    sealed_execution: Mapping[str, Any],
    fixed_corpora: Sequence[str] = FIXED_PRODUCTION_CORPORA,
) -> tuple[Mapping[str, Any], ...]:
    """Validate the exact ordered public workload set bound by C1."""
    expected_corpora = tuple(fixed_corpora)
    if value == PRODUCTION_WORKLOADS_UNRESOLVED:
        if frozen:
            raise ProductionWorkloadRegistrationError(
                "production_workloads must be resolved before freeze"
            )
        return ()
    registered_selected_family_count = _positive_integer(
        registered_selected_family_count,
        path="analysis.power.selected_families_per_corpus",
    )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProductionWorkloadRegistrationError(
            "production_workloads must be the explicit unresolved sentinel or an array"
        )
    if len(value) != len(expected_corpora):
        raise ProductionWorkloadRegistrationError(
            f"production_workloads must contain exactly {len(expected_corpora)} rows"
        )
    registrations: list[Mapping[str, Any]] = []
    for position, (item, expected_corpus) in enumerate(zip(value, expected_corpora, strict=True)):
        path = f"production_workloads[{position}]"
        row = _closed_mapping(
            item,
            fields=PRODUCTION_WORKLOAD_REGISTRATION_FIELDS,
            path=path,
        )
        if row["corpus_id"] != expected_corpus:
            raise ProductionWorkloadRegistrationError(
                f"{path}.corpus_id must equal {expected_corpus!r}"
            )
        spec = _validate_workload_spec(
            row["spec"],
            corpus_id=expected_corpus,
            registered_selected_family_count=registered_selected_family_count,
            sealed_execution=sealed_execution,
            path=f"{path}.spec",
        )
        observed_sha256 = row["canonical_file_sha256"]
        if not isinstance(observed_sha256, str) or _SHA256.fullmatch(observed_sha256) is None:
            raise ProductionWorkloadRegistrationError(
                f"{path}.canonical_file_sha256 must be a lowercase SHA-256 digest"
            )
        expected_sha256 = production_workload_file_sha256(spec)
        if observed_sha256 != expected_sha256:
            raise ProductionWorkloadRegistrationError(
                f"{path}.canonical_file_sha256 differs from the canonical spec file"
            )
        registrations.append(row)
    return tuple(registrations)
