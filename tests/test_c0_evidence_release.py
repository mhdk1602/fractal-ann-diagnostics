from __future__ import annotations

import copy
import hashlib

import pytest

from fractal_ann_diagnostics.c0_evidence_release import (
    C0_APPARATUS_EVIDENCE_SCHEMA,
    C0_CANDIDATE_ASSEMBLY_RECEIPT_ARCHIVE_MEMBER_PATH,
    C0_CANDIDATE_MANIFEST_ARCHIVE_MEMBER_PATH,
    C0_EVIDENCE_RELEASE_SCHEMA,
    C0_EVIDENCE_RELEASE_TAG,
    C0_EVIDENCE_RELEASE_URL,
    C0_EVIDENCE_REPOSITORY,
    C0_EVIDENCE_VERIFICATION_SCHEMA,
    C0EvidenceReleaseError,
    canonical_apparatus_evidence_bytes,
    canonical_verification_bytes,
    validate_c0_evidence_release_binding,
)

COMMIT = "1" * 40


def _binding() -> dict[str, object]:
    name = f"fractal-ann-diagnostics-c0-evidence-{COMMIT}.tar.gz"
    url = (
        "https://github.com/mhdk1602/fractal-ann-diagnostics/releases/download/"
        f"{C0_EVIDENCE_RELEASE_TAG}/{name}"
    )
    verification: dict[str, object] = {
        "anonymous_asset_sha256": "2" * 64,
        "anonymous_asset_size": 4096,
        "anonymous_checksum_sha256": "3" * 64,
        "anonymous_checksum_size": 128,
        "asset_attestation_output_sha256": "4" * 64,
        "asset_attestation_verified": True,
        "release_api_output_sha256": "5" * 64,
        "release_attestation_output_sha256": "6" * 64,
        "release_attestation_verified": True,
        "release_tag_readback_sha256": "7" * 64,
        "release_tag_target_commit": COMMIT,
        "release_tag_target_verified": True,
        "schema_version": C0_EVIDENCE_VERIFICATION_SCHEMA,
    }
    apparatus: dict[str, object] = {
        "build_context_tree_sha256": "b" * 64,
        "c0_commit": COMMIT,
        "candidate_bootstrap_closure_sha256": "c" * 64,
        "candidate_image_closure_sha256": "d" * 64,
        "candidate_image_run_id": 101,
        "candidate_image_source_commit": "2" * 40,
        "candidate_manifest_archive_member_path": C0_CANDIDATE_MANIFEST_ARCHIVE_MEMBER_PATH,
        "candidate_manifest_assembly_receipt_archive_member_path": (
            C0_CANDIDATE_ASSEMBLY_RECEIPT_ARCHIVE_MEMBER_PATH
        ),
        "candidate_manifest_assembly_receipt_file_sha256": "a" * 64,
        "candidate_manifest_file_sha256": "9" * 64,
        "github_environment_control_receipt_file_sha256": "8" * 64,
        "oci_promotion_receipt_sha256": "e" * 64,
        "production_image_run_id": 202,
        "production_control_instantiation_receipt_file_sha256": "7" * 64,
        "provider_phase_plan_closure_sha256": "f" * 64,
        "provider_rehearsal_gate_sha256": "1" * 64,
        "provider_rehearsal_receipt_sha256": "2" * 64,
        "provider_rehearsal_run_id": 303,
        "rehearsal_attestation_verification_sha256": "3" * 64,
        "rehearsal_manifest_sha256": "4" * 64,
        "release_image_index_digest": f"sha256:{'5' * 64}",
        "schema_version": C0_APPARATUS_EVIDENCE_SCHEMA,
        "scientific_image_index_digest": f"sha256:{'6' * 64}",
    }
    return {
        "apparatus_evidence": apparatus,
        "apparatus_evidence_sha256": hashlib.sha256(
            canonical_apparatus_evidence_bytes(apparatus)
        ).hexdigest(),
        "asset_name": name,
        "asset_sha256": "2" * 64,
        "asset_size": 4096,
        "asset_url": url,
        "checksum_asset_name": f"{name}.sha256",
        "checksum_asset_sha256": "3" * 64,
        "checksum_asset_size": 128,
        "checksum_asset_url": f"{url}.sha256",
        "immutable_release": True,
        "release_tag": C0_EVIDENCE_RELEASE_TAG,
        "release_url": C0_EVIDENCE_RELEASE_URL,
        "repository": C0_EVIDENCE_REPOSITORY,
        "schema_version": C0_EVIDENCE_RELEASE_SCHEMA,
        "target_commit": COMMIT,
        "verification_receipt": verification,
        "verification_receipt_sha256": hashlib.sha256(
            canonical_verification_bytes(verification)
        ).hexdigest(),
    }


def test_draft_requires_one_literal_sentinel() -> None:
    validate_c0_evidence_release_binding("tbd", frozen=False, code_commit="tbd")
    with pytest.raises(C0EvidenceReleaseError, match="literal 'tbd'"):
        validate_c0_evidence_release_binding({}, frozen=False, code_commit="tbd")


def test_frozen_binding_closes_release_and_verification_identity() -> None:
    validate_c0_evidence_release_binding(_binding(), frozen=True, code_commit=COMMIT)


def test_frozen_binding_rejects_non_string_schema_keys() -> None:
    binding = _binding()
    binding[1] = binding.pop("repository")  # type: ignore[index]
    with pytest.raises(C0EvidenceReleaseError, match="keys must be strings"):
        validate_c0_evidence_release_binding(binding, frozen=True, code_commit=COMMIT)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("repository",), "someone/else", "repository differs"),
        (("release_tag",), "mutable", "release_tag differs"),
        (("target_commit",), "0" * 40, "target_commit differs"),
        (("immutable_release",), False, "immutable_release must be true"),
        (("apparatus_evidence", "c0_commit"), "0" * 40, "differs from C0"),
        (
            ("apparatus_evidence", "candidate_image_source_commit"),
            "short",
            "must be a Git commit",
        ),
        (("apparatus_evidence", "candidate_image_run_id"), 0, "positive integer"),
        (
            ("apparatus_evidence", "candidate_manifest_archive_member_path"),
            "production-control-instantiation/substituted.json",
            "fixed C0 archive member",
        ),
        (
            (
                "apparatus_evidence",
                "candidate_manifest_assembly_receipt_archive_member_path",
            ),
            "production-control-instantiation/substituted-receipt.json",
            "fixed C0 archive member",
        ),
        (
            ("apparatus_evidence", "github_environment_control_receipt_file_sha256"),
            "short",
            "lowercase SHA-256",
        ),
        (
            ("apparatus_evidence", "scientific_image_index_digest"),
            "6" * 64,
            "OCI SHA-256 digest",
        ),
        (
            ("apparatus_evidence_sha256",),
            "0" * 64,
            "canonical apparatus evidence",
        ),
        (("asset_sha256",), "x", "asset_sha256 must be"),
        (("asset_size",), 0, "asset_size must be"),
        (
            ("verification_receipt", "release_attestation_verified"),
            False,
            "release_attestation_verified must be true",
        ),
        (
            ("verification_receipt", "release_tag_readback_sha256"),
            "short",
            "lowercase SHA-256",
        ),
        (
            ("verification_receipt", "release_tag_target_verified"),
            False,
            "release_tag_target_verified must be true",
        ),
        (
            ("verification_receipt", "release_tag_target_commit"),
            "short",
            "must be a Git commit",
        ),
        (
            ("verification_receipt", "release_tag_target_commit"),
            "0" * 40,
            "differs from target_commit",
        ),
        (
            ("verification_receipt", "anonymous_asset_sha256"),
            "0" * 64,
            "anonymous readback differs",
        ),
        (("verification_receipt_sha256",), "0" * 64, "canonical receipt bytes"),
    ),
)
def test_frozen_binding_rejects_every_release_substitution(
    path: tuple[str, ...], replacement: object, message: str
) -> None:
    binding = copy.deepcopy(_binding())
    target: dict[str, object] = binding
    for component in path[:-1]:
        target = target[component]  # type: ignore[assignment]
    target[path[-1]] = replacement
    with pytest.raises(C0EvidenceReleaseError, match=message):
        validate_c0_evidence_release_binding(binding, frozen=True, code_commit=COMMIT)


def test_receipt_digest_covers_canonical_embedded_receipt() -> None:
    binding = _binding()
    receipt = binding["verification_receipt"]
    assert isinstance(receipt, dict)
    assert (
        binding["verification_receipt_sha256"]
        == hashlib.sha256(canonical_verification_bytes(receipt)).hexdigest()
    )
    apparatus = binding["apparatus_evidence"]
    assert isinstance(apparatus, dict)
    assert (
        binding["apparatus_evidence_sha256"]
        == hashlib.sha256(canonical_apparatus_evidence_bytes(apparatus)).hexdigest()
    )
