"""Closed C1 binding for the immutable GitHub C0 evidence release.

The evidence archive is a GitHub Release asset, not a Zenodo registration
member. C1 embeds this compact binding and retains a fresh public-verification
receipt so the fixed 27-file package records both release identity and readback.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

C0_EVIDENCE_RELEASE_SCHEMA = "fractal-c0-evidence-release-binding-v2"
C0_EVIDENCE_VERIFICATION_SCHEMA = "fractal-c0-evidence-release-verification-v2"
C0_APPARATUS_EVIDENCE_SCHEMA = "fractal-c0-apparatus-evidence-closure-v5"
C0_EVIDENCE_RELEASE_TAG = "confirmatory-apparatus-c0"
C0_EVIDENCE_REPOSITORY = "mhdk1602/fractal-ann-diagnostics"
C0_CANDIDATE_MANIFEST_ARCHIVE_MEMBER_PATH = (
    "production-control-instantiation/candidate-manifest-package/candidate-study-manifest.json"
)
C0_CANDIDATE_ASSEMBLY_RECEIPT_ARCHIVE_MEMBER_PATH = (
    "production-control-instantiation/candidate-manifest-package/"
    "candidate-manifest-assembly-receipt.json"
)
C0_EVIDENCE_RELEASE_URL = (
    "https://github.com/mhdk1602/fractal-ann-diagnostics/releases/tag/confirmatory-apparatus-c0"
)

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ASSET_NAME = re.compile(r"^fractal-ann-diagnostics-c0-evidence-([0-9a-f]{40})\.tar\.gz$")
_BINDING_FIELDS = frozenset(
    {
        "apparatus_evidence",
        "apparatus_evidence_sha256",
        "asset_name",
        "asset_sha256",
        "asset_size",
        "asset_url",
        "checksum_asset_name",
        "checksum_asset_sha256",
        "checksum_asset_size",
        "checksum_asset_url",
        "immutable_release",
        "release_tag",
        "release_url",
        "repository",
        "schema_version",
        "target_commit",
        "verification_receipt",
        "verification_receipt_sha256",
    }
)
_APPARATUS_EVIDENCE_FIELDS = frozenset(
    {
        "build_context_tree_sha256",
        "c0_commit",
        "candidate_bootstrap_closure_sha256",
        "candidate_image_closure_sha256",
        "candidate_image_run_id",
        "candidate_image_source_commit",
        "candidate_manifest_archive_member_path",
        "candidate_manifest_assembly_receipt_archive_member_path",
        "candidate_manifest_assembly_receipt_file_sha256",
        "candidate_manifest_file_sha256",
        "github_environment_control_receipt_file_sha256",
        "oci_promotion_receipt_sha256",
        "production_image_run_id",
        "production_control_instantiation_receipt_file_sha256",
        "provider_phase_plan_closure_sha256",
        "provider_rehearsal_gate_sha256",
        "provider_rehearsal_receipt_sha256",
        "provider_rehearsal_run_id",
        "rehearsal_attestation_verification_sha256",
        "rehearsal_manifest_sha256",
        "release_image_index_digest",
        "schema_version",
        "scientific_image_index_digest",
    }
)
_VERIFICATION_FIELDS = frozenset(
    {
        "anonymous_asset_sha256",
        "anonymous_asset_size",
        "anonymous_checksum_sha256",
        "anonymous_checksum_size",
        "asset_attestation_output_sha256",
        "asset_attestation_verified",
        "release_api_output_sha256",
        "release_attestation_output_sha256",
        "release_attestation_verified",
        "release_tag_readback_sha256",
        "release_tag_target_commit",
        "release_tag_target_verified",
        "schema_version",
    }
)


class C0EvidenceReleaseError(ValueError):
    """The C0 release binding is incomplete or inconsistent."""


def _closed(value: object, fields: frozenset[str], *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise C0EvidenceReleaseError(f"{path} must be an object")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise C0EvidenceReleaseError(f"{path} keys must be strings")
    if keys != fields:
        missing = sorted(fields - keys)
        extra = sorted(keys - fields)
        raise C0EvidenceReleaseError(
            f"{path} has another field set: missing={missing}, extra={extra}"
        )
    return value


def _digest(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise C0EvidenceReleaseError(f"{path} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, *, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise C0EvidenceReleaseError(f"{path} must be a positive integer")
    return value


def canonical_verification_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the canonical file bytes used by the C1 digest binding."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def canonical_apparatus_evidence_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the canonical bytes that bind the pre-C0 rehearsal to production."""

    return canonical_verification_bytes(value)


def validate_c0_evidence_release_binding(
    value: object,
    *,
    frozen: bool,
    code_commit: object,
) -> None:
    """Validate the draft sentinel or one exact immutable-release binding."""

    path = "sealed_execution.c0_evidence_release"
    if not frozen:
        if value != "tbd":
            raise C0EvidenceReleaseError(f"{path} must be the literal 'tbd' before freeze")
        return

    if not isinstance(code_commit, str) or _SHA1.fullmatch(code_commit) is None:
        raise C0EvidenceReleaseError(f"{path} requires the frozen C0 commit")
    binding = _closed(value, _BINDING_FIELDS, path=path)
    expected_name = f"fractal-ann-diagnostics-c0-evidence-{code_commit}.tar.gz"
    expected_checksum_name = f"{expected_name}.sha256"
    expected_asset_url = (
        "https://github.com/mhdk1602/fractal-ann-diagnostics/releases/download/"
        f"{C0_EVIDENCE_RELEASE_TAG}/{expected_name}"
    )
    expected_checksum_url = f"{expected_asset_url}.sha256"
    exact = {
        "schema_version": C0_EVIDENCE_RELEASE_SCHEMA,
        "repository": C0_EVIDENCE_REPOSITORY,
        "release_tag": C0_EVIDENCE_RELEASE_TAG,
        "release_url": C0_EVIDENCE_RELEASE_URL,
        "target_commit": code_commit,
        "asset_name": expected_name,
        "checksum_asset_name": expected_checksum_name,
        "asset_url": expected_asset_url,
        "checksum_asset_url": expected_checksum_url,
    }
    for field, expected in exact.items():
        if binding[field] != expected:
            raise C0EvidenceReleaseError(f"{path}.{field} differs from C0")
    if _ASSET_NAME.fullmatch(str(binding["asset_name"])) is None:
        raise C0EvidenceReleaseError(f"{path}.asset_name is not canonical")
    if binding["immutable_release"] is not True:
        raise C0EvidenceReleaseError(f"{path}.immutable_release must be true")

    apparatus = _closed(
        binding["apparatus_evidence"],
        _APPARATUS_EVIDENCE_FIELDS,
        path=f"{path}.apparatus_evidence",
    )
    if apparatus["schema_version"] != C0_APPARATUS_EVIDENCE_SCHEMA:
        raise C0EvidenceReleaseError(f"{path}.apparatus_evidence.schema_version differs")
    if apparatus["c0_commit"] != code_commit:
        raise C0EvidenceReleaseError(f"{path}.apparatus_evidence.c0_commit differs from C0")
    exact_members = {
        "candidate_manifest_archive_member_path": C0_CANDIDATE_MANIFEST_ARCHIVE_MEMBER_PATH,
        "candidate_manifest_assembly_receipt_archive_member_path": (
            C0_CANDIDATE_ASSEMBLY_RECEIPT_ARCHIVE_MEMBER_PATH
        ),
    }
    for field, expected_member in exact_members.items():
        if apparatus[field] != expected_member:
            raise C0EvidenceReleaseError(
                f"{path}.apparatus_evidence.{field} differs from the fixed C0 archive member"
            )
    source_commit = apparatus["candidate_image_source_commit"]
    if not isinstance(source_commit, str) or _SHA1.fullmatch(source_commit) is None:
        raise C0EvidenceReleaseError(
            f"{path}.apparatus_evidence.candidate_image_source_commit must be a Git commit"
        )
    for field in (
        "candidate_image_run_id",
        "production_image_run_id",
        "provider_rehearsal_run_id",
    ):
        _positive_integer(apparatus[field], path=f"{path}.apparatus_evidence.{field}")
    for field in (
        "build_context_tree_sha256",
        "candidate_bootstrap_closure_sha256",
        "candidate_image_closure_sha256",
        "candidate_manifest_assembly_receipt_file_sha256",
        "candidate_manifest_file_sha256",
        "github_environment_control_receipt_file_sha256",
        "oci_promotion_receipt_sha256",
        "production_control_instantiation_receipt_file_sha256",
        "provider_phase_plan_closure_sha256",
        "provider_rehearsal_gate_sha256",
        "provider_rehearsal_receipt_sha256",
        "rehearsal_attestation_verification_sha256",
        "rehearsal_manifest_sha256",
    ):
        _digest(apparatus[field], path=f"{path}.apparatus_evidence.{field}")
    for field in ("release_image_index_digest", "scientific_image_index_digest"):
        value = apparatus[field]
        if not isinstance(value, str) or not value.startswith("sha256:"):
            raise C0EvidenceReleaseError(
                f"{path}.apparatus_evidence.{field} must be an OCI SHA-256 digest"
            )
        _digest(value.removeprefix("sha256:"), path=f"{path}.apparatus_evidence.{field}")
    apparatus_sha256 = hashlib.sha256(canonical_apparatus_evidence_bytes(apparatus)).hexdigest()
    if binding["apparatus_evidence_sha256"] != apparatus_sha256:
        raise C0EvidenceReleaseError(
            f"{path}.apparatus_evidence_sha256 differs from canonical apparatus evidence"
        )

    asset_sha256 = _digest(binding["asset_sha256"], path=f"{path}.asset_sha256")
    checksum_sha256 = _digest(
        binding["checksum_asset_sha256"],
        path=f"{path}.checksum_asset_sha256",
    )
    asset_size = _positive_integer(binding["asset_size"], path=f"{path}.asset_size")
    checksum_size = _positive_integer(
        binding["checksum_asset_size"],
        path=f"{path}.checksum_asset_size",
    )

    verification = _closed(
        binding["verification_receipt"],
        _VERIFICATION_FIELDS,
        path=f"{path}.verification_receipt",
    )
    if verification["schema_version"] != C0_EVIDENCE_VERIFICATION_SCHEMA:
        raise C0EvidenceReleaseError(f"{path}.verification_receipt.schema_version differs")
    for field in (
        "anonymous_asset_sha256",
        "anonymous_checksum_sha256",
        "asset_attestation_output_sha256",
        "release_api_output_sha256",
        "release_attestation_output_sha256",
        "release_tag_readback_sha256",
    ):
        _digest(verification[field], path=f"{path}.verification_receipt.{field}")
    for field in (
        "release_attestation_verified",
        "asset_attestation_verified",
        "release_tag_target_verified",
    ):
        if verification[field] is not True:
            raise C0EvidenceReleaseError(f"{path}.verification_receipt.{field} must be true")
    release_tag_target = verification["release_tag_target_commit"]
    if not isinstance(release_tag_target, str) or _SHA1.fullmatch(release_tag_target) is None:
        raise C0EvidenceReleaseError(
            f"{path}.verification_receipt.release_tag_target_commit must be a Git commit"
        )
    if release_tag_target != binding["target_commit"]:
        raise C0EvidenceReleaseError(
            f"{path}.verification_receipt.release_tag_target_commit differs from target_commit"
        )
    if (
        verification["anonymous_asset_sha256"] != asset_sha256
        or verification["anonymous_asset_size"] != asset_size
        or verification["anonymous_checksum_sha256"] != checksum_sha256
        or verification["anonymous_checksum_size"] != checksum_size
    ):
        raise C0EvidenceReleaseError(
            f"{path}.verification_receipt anonymous readback differs from the assets"
        )
    _positive_integer(
        verification["anonymous_asset_size"],
        path=f"{path}.verification_receipt.anonymous_asset_size",
    )
    _positive_integer(
        verification["anonymous_checksum_size"],
        path=f"{path}.verification_receipt.anonymous_checksum_size",
    )
    receipt_sha256 = hashlib.sha256(canonical_verification_bytes(verification)).hexdigest()
    if binding["verification_receipt_sha256"] != receipt_sha256:
        raise C0EvidenceReleaseError(
            f"{path}.verification_receipt_sha256 differs from canonical receipt bytes"
        )
