from __future__ import annotations

import hashlib
from collections.abc import Sequence

from fractal_ann_diagnostics.c0_evidence_release import (
    C0_APPARATUS_EVIDENCE_SCHEMA,
    C0_CANDIDATE_ASSEMBLY_RECEIPT_ARCHIVE_MEMBER_PATH,
    C0_CANDIDATE_MANIFEST_ARCHIVE_MEMBER_PATH,
    C0_EVIDENCE_RELEASE_SCHEMA,
    C0_EVIDENCE_RELEASE_TAG,
    C0_EVIDENCE_RELEASE_URL,
    C0_EVIDENCE_REPOSITORY,
    C0_EVIDENCE_VERIFICATION_SCHEMA,
    canonical_apparatus_evidence_bytes,
    canonical_verification_bytes,
)
from fractal_ann_diagnostics.production_workload_registration import (
    PRODUCTION_WORKLOAD_SPEC_SCHEMA,
    production_workload_file_sha256,
)


def registered_c0_evidence_release(*, code_commit: str) -> dict[str, object]:
    """Build one internally consistent frozen C0 release fixture."""

    asset_name = f"fractal-ann-diagnostics-c0-evidence-{code_commit}.tar.gz"
    asset_url = (
        "https://github.com/mhdk1602/fractal-ann-diagnostics/releases/download/"
        f"{C0_EVIDENCE_RELEASE_TAG}/{asset_name}"
    )
    verification: dict[str, object] = {
        "anonymous_asset_sha256": "6" * 64,
        "anonymous_asset_size": 100,
        "anonymous_checksum_sha256": "7" * 64,
        "anonymous_checksum_size": 120,
        "asset_attestation_output_sha256": "8" * 64,
        "asset_attestation_verified": True,
        "release_api_output_sha256": "9" * 64,
        "release_attestation_output_sha256": "a" * 64,
        "release_attestation_verified": True,
        "release_tag_readback_sha256": "b" * 64,
        "release_tag_target_commit": code_commit,
        "release_tag_target_verified": True,
        "schema_version": C0_EVIDENCE_VERIFICATION_SCHEMA,
    }
    apparatus: dict[str, object] = {
        "build_context_tree_sha256": "b" * 64,
        "c0_commit": code_commit,
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
        "asset_name": asset_name,
        "asset_sha256": "6" * 64,
        "asset_size": 100,
        "asset_url": asset_url,
        "checksum_asset_name": f"{asset_name}.sha256",
        "checksum_asset_sha256": "7" * 64,
        "checksum_asset_size": 120,
        "checksum_asset_url": f"{asset_url}.sha256",
        "immutable_release": True,
        "release_tag": C0_EVIDENCE_RELEASE_TAG,
        "release_url": C0_EVIDENCE_RELEASE_URL,
        "repository": C0_EVIDENCE_REPOSITORY,
        "schema_version": C0_EVIDENCE_RELEASE_SCHEMA,
        "target_commit": code_commit,
        "verification_receipt": verification,
        "verification_receipt_sha256": hashlib.sha256(
            canonical_verification_bytes(verification)
        ).hexdigest(),
    }


def registered_production_workloads(
    *,
    fixed_corpora: Sequence[str],
    runner_image: str,
    runner_identity: str,
    code_commit: str,
    selected_family_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position, corpus_id in enumerate(fixed_corpora):

        def digest(name: str) -> str:
            return hashlib.sha256(f"{corpus_id}:{name}".encode("utf-8")).hexdigest()

        root = f"/opt/study/corpora/{corpus_id}"
        spec: dict[str, object] = {
            "artifact_root": f"{root}/factory",
            "artifact_tree_sha256": digest("artifact-tree"),
            "authorized_index_store_root": f"{root}/authorized-index",
            "authorized_index_store_tree_sha256": digest("authorized-index-tree"),
            "available_family_count": max(75, selected_family_count),
            "code_commit": code_commit,
            "corpus_id": corpus_id,
            "embedding_store_root": f"{root}/embedding-store",
            "embedding_store_tree_sha256": digest("embedding-store-tree"),
            "expected_authorized_index_store_receipt_sha256": digest("authorized-index-receipt"),
            "expected_policy_intervention_receipt_sha256": digest("policy-receipt"),
            "expected_pseudonym_key_sha256": digest("pseudonym-key"),
            "factory_artifact_tree_sha256": digest("factory-tree"),
            "factory_config_sha256": digest("factory-config"),
            "factory_suite_receipt_sha256": digest("factory-suite-receipt"),
            "feature_bindings": [
                {
                    "backend": "hnsw",
                    "drift_family": "current-truth",
                    "group_order": position,
                    "policy_complexity": 1.0,
                    "policy_state": "current",
                    "repetition": 0,
                    "subject": corpus_id,
                    "version_lag": 0.0,
                }
            ],
            "index_bundle_receipt_path": f"{root}/bundles/index-receipt.json",
            "index_bundle_receipt_sha256": digest("index-bundle-receipt"),
            "online_execution_plan_sha256": digest("online-execution-plan"),
            "online_execution_tree_sha256": digest("online-execution-tree"),
            "partition_audit_file_sha256": digest("partition-audit-file"),
            "partition_audit_path": f"{root}/query-partition-audit.json",
            "partition_audit_sha256": digest("partition-audit-logical"),
            "policy_bundle_receipt_path": f"{root}/bundles/policy-receipt.json",
            "policy_bundle_receipt_sha256": digest("policy-bundle-receipt"),
            "policy_intervention_root": f"{root}/policy-intervention",
            "policy_intervention_tree_sha256": digest("policy-intervention-tree"),
            "pseudonym_key_path": f"{root}/custody/pseudonym.key",
            "query_package_root": f"{root}/query-package",
            "query_package_tree_sha256": digest("query-package-tree"),
            "query_receipt_sha256": digest("query-receipt"),
            "runner_identity": runner_identity,
            "runner_image": runner_image,
            "runner_platform": "linux/arm64",
            "schema_version": PRODUCTION_WORKLOAD_SPEC_SCHEMA,
            "selected_family_count": selected_family_count,
            "sharded_execution_plan_file_sha256": digest("sharded-plan-file"),
            "staged_root": f"{root}/online-staging",
            "staged_tree_sha256": digest("online-staging-tree"),
            "trial_runtime_admission_receipt_file_sha256": digest(
                "trial-runtime-admission-receipt-file"
            ),
        }
        rows.append(
            {
                "canonical_file_sha256": production_workload_file_sha256(spec),
                "corpus_id": corpus_id,
                "spec": spec,
            }
        )
    return rows
