from __future__ import annotations

import copy
import hashlib
import json
import ssl
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import Request

import pytest
from production_workload_fixtures import registered_production_workloads

import fractal_ann_diagnostics.study as study_module
from fractal_ann_diagnostics.artifact_integrity import (
    ArtifactVerificationReceipt,
    VerifiedArtifact,
    load_verification_receipt,
    write_verification_receipt,
)
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
from fractal_ann_diagnostics.execution_claim import (
    ExecutionClaimError,
    assert_normalized_provider_phase_plan_closure,
    derive_phase_runner_label,
    load_provider_phase_plans,
    materialize_provider_phase_plan,
    provider_phase_plan_templates_sha256,
)
from fractal_ann_diagnostics.production_workload_registration import (
    PRODUCTION_WORKLOAD_SPEC_FIELDS,
    PRODUCTION_WORKLOADS_UNRESOLVED,
    production_workload_file_sha256,
)
from fractal_ann_diagnostics.provider_contract import (
    DOCKER_SERVER_PROBE_SCHEMA,
    OFFICIAL_ACTIONS_RUNNER_CONFIG_SHA256,
    OFFICIAL_ACTIONS_RUNNER_LISTENER_DLL_SHA256,
    OFFICIAL_ACTIONS_RUNNER_LISTENER_SHA256,
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_BYTE_COUNT,
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_URI,
    OFFICIAL_ACTIONS_RUNNER_RUN_SHA256,
    OFFICIAL_ACTIONS_RUNNER_VERSION,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_BYTE_COUNT,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_SHA256,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_URI,
    OFFICIAL_GH_OSX_ARM64_BINARY_SHA256,
    OFFICIAL_GH_VERSION,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_BYTE_COUNT,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_SHA256,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_URI,
    OFFICIAL_PYTHON_BUILD_STANDALONE_BINARY_SHA256,
    OFFICIAL_PYTHON_BUILD_STANDALONE_VERSION,
    PHASE_HOST_PROBE_SCHEMA,
    PHASE_HOST_TOOL_CONTRACT_SCHEMA,
    REGISTERED_DOCKER_CLIENT_BUILD,
    REGISTERED_DOCKER_CLIENT_SHA256,
    REGISTERED_DOCKER_CLIENT_VERSION,
    REGISTERED_HOST_PYTHON_LAUNCHER_SHA256,
    SOURCE_BUILT_LINUX_ARM64_TLE_SHA256,
)
from fractal_ann_diagnostics.study import (
    C0_COMMIT_SENTINEL,
    C0_COMMIT_SENTINEL_PATHS,
    EVIDENCE_CORPORA,
    FIXED_CORPORA,
    MAX_PROTOCOL_REGISTRY_RECORD_BYTES,
    PROVIDER_APPROVAL_ENVIRONMENT,
    PROVIDER_PHASE_COMMAND_IDS,
    PROVIDER_PHASE_JOB_NAMES,
    PROVIDER_PHASE_PLAN_TEMPLATE_SCHEMA,
    PROVIDER_PHASE_RUNTIME_BINDINGS,
    PROVIDER_PHASE_RUNTIME_CEILINGS,
    PROVIDER_PHASE_WORKFLOWS,
    PROVIDER_PLAN_ACTIVATION_OUTPUT_BINDING,
    PROVIDER_PLAN_C1_COMMIT_BINDING,
    PROVIDER_PLAN_CLAIM_RECEIPT_BINDING,
    PROVIDER_PLAN_GITHUB_OUTPUT_BINDING,
    PROVIDER_PLAN_LAUNCHER_SOURCE_BINDING,
    PROVIDER_PLAN_MANIFEST_BINDING,
    PROVIDER_PLAN_PHASE_INPUT_BINDING,
    PROVIDER_PLAN_PHASE_OUTPUT_BINDING,
    PROVIDER_PLAN_PREDECESSOR_BINDING,
    PROVIDER_PLAN_SUITE_BINDING,
    PROVIDER_RUNNER_IDENTITY,
    REGISTERED_ACTION_SET,
    REGISTERED_POWER_ENDPOINTS,
    REGISTERED_POWER_FAMILY_CANDIDATES,
    REGISTERED_PRIMARY_CLAIM,
    ProtocolRegistrationReceipt,
    ProtocolRegistryRecord,
    SealedRunReceipt,
    StudyManifestError,
    VerifiedC1ProtocolRegistration,
    begin_sealed_run,
    load_protocol_registration_receipt,
    load_sealed_run_receipt,
    load_study_manifest,
    manifest_sha256,
    resolve_candidate_c0_commit_sentinels,
    sealed_receipt_uri,
    validate_candidate_rehearsal_manifest,
    validate_candidate_rehearsal_to_frozen_transition,
    validate_study_manifest,
)

_ROLE_KINDS = (
    ("study-data-package", "dataset-package"),
    ("online-staging-package", "dataset-package"),
    ("development-freeze-package", "development-package"),
    ("development-fit-data", "dataset"),
    ("development-calibration-data", "dataset"),
    ("query-partition-audit", "partition-audit"),
    ("primary-embedding", "embedding"),
    ("stale-embedding", "embedding"),
    ("exact-authorized-oracle", "backend"),
    ("strict-authorized-hnsw", "backend"),
    ("opa-pdp", "policy"),
    ("opa-runtime-binary", "tool"),
    ("frozen-controller", "controller"),
    ("static-comparator", "comparator"),
    ("h1-predictive-model", "model"),
    ("h2-model-suite", "model"),
    ("power-analysis-report", "analysis"),
    ("analysis-runner", "analysis"),
    ("source-code", "source"),
    ("custody-seal-receipt", "custody"),
    ("tlock-release-provenance", "custody"),
    ("timelock-tool", "tool"),
    ("custody-builder", "execution"),
    ("suite-attestation-descriptor", "attestation"),
)
_COMMIT = "1" * 40
_RUNNER_IDENTITY = "github-actions:environment:confirmatory"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _c0_evidence_release_binding(commit: str) -> dict[str, object]:
    asset_name = f"fractal-ann-diagnostics-c0-evidence-{commit}.tar.gz"
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
        "release_tag_target_commit": commit,
        "release_tag_target_verified": True,
        "schema_version": C0_EVIDENCE_VERIFICATION_SCHEMA,
    }
    apparatus: dict[str, object] = {
        "build_context_tree_sha256": "b" * 64,
        "c0_commit": commit,
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
        "target_commit": commit,
        "verification_receipt": verification,
        "verification_receipt_sha256": hashlib.sha256(
            canonical_verification_bytes(verification)
        ).hexdigest(),
    }


def _provider_host_tools() -> dict[str, object]:
    root = "/opt/fractal-confirmatory/host-tools"
    host_probe = {
        "architecture": "ARM64",
        "kernel_release": "24.5.0",
        "logical_cpu_count": 12,
        "operating_system": "macOS",
        "operating_system_version": "15.5",
        "physical_memory_bytes": 64 * 1024**3,
        "schema_version": PHASE_HOST_PROBE_SCHEMA,
    }
    docker_probe = {
        "architecture": "arm64",
        "cpu_count": 12,
        "engine_build": "fixture-server-build",
        "engine_version": "28.3.2",
        "kernel_version": "6.10.14-linuxkit",
        "memory_bytes": 48 * 1024**3,
        "operating_system": "linux",
        "schema_version": DOCKER_SERVER_PROBE_SCHEMA,
    }
    return {
        "controlled_root": root,
        "docker_client_build": REGISTERED_DOCKER_CLIENT_BUILD,
        "docker_client_version": REGISTERED_DOCKER_CLIENT_VERSION,
        "docker_executable": "/usr/local/bin/docker",
        "docker_executable_sha256": REGISTERED_DOCKER_CLIENT_SHA256,
        "docker_resolved_executable": ("/Applications/Docker.app/Contents/Resources/bin/docker"),
        "docker_server_probe": docker_probe,
        "docker_server_probe_receipt_sha256": hashlib.sha256(
            _canonical(docker_probe) + b"\n"
        ).hexdigest(),
        "gh_archive_byte_count": OFFICIAL_GH_OSX_ARM64_ARCHIVE_BYTE_COUNT,
        "gh_archive_sha256": OFFICIAL_GH_OSX_ARM64_ARCHIVE_SHA256,
        "gh_archive_uri": OFFICIAL_GH_OSX_ARM64_ARCHIVE_URI,
        "gh_executable": f"{root}/gh/bin/gh",
        "gh_executable_sha256": OFFICIAL_GH_OSX_ARM64_BINARY_SHA256,
        "gh_version": OFFICIAL_GH_VERSION,
        "host_architecture": "ARM64",
        "host_operating_system": "macOS",
        "host_probe": host_probe,
        "host_probe_receipt_sha256": hashlib.sha256(_canonical(host_probe) + b"\n").hexdigest(),
        "python_archive_byte_count": OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_BYTE_COUNT,
        "python_archive_sha256": OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_SHA256,
        "python_archive_uri": OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_URI,
        "python_executable": f"{root}/python/bin/python3.12",
        "python_executable_sha256": OFFICIAL_PYTHON_BUILD_STANDALONE_BINARY_SHA256,
        "python_import_root": f"{root}/venv/lib/python3.12/site-packages",
        "python_import_tree_sha256": hashlib.sha256(b"python import tree").hexdigest(),
        "python_launcher_sha256": REGISTERED_HOST_PYTHON_LAUNCHER_SHA256,
        "python_package_content_sha256": hashlib.sha256(b"python package content").hexdigest(),
        "python_package_source_commit": _COMMIT,
        "python_package_source_tree": "f" * 40,
        "python_package_tree_sha256": hashlib.sha256(b"python package tree").hexdigest(),
        "python_version": OFFICIAL_PYTHON_BUILD_STANDALONE_VERSION,
        "runner_archive_byte_count": OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_BYTE_COUNT,
        "runner_archive_sha256": OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
        "runner_archive_uri": OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_URI,
        "runner_config_executable": f"{root}/runner/config.sh",
        "runner_config_sha256": OFFICIAL_ACTIONS_RUNNER_CONFIG_SHA256,
        "runner_disable_update": True,
        "runner_ephemeral": True,
        "runner_listener_dll": f"{root}/runner/bin/Runner.Listener.dll",
        "runner_listener_dll_sha256": OFFICIAL_ACTIONS_RUNNER_LISTENER_DLL_SHA256,
        "runner_listener_executable": f"{root}/runner/bin/Runner.Listener",
        "runner_listener_sha256": OFFICIAL_ACTIONS_RUNNER_LISTENER_SHA256,
        "runner_run_executable": f"{root}/runner/run.sh",
        "runner_run_sha256": OFFICIAL_ACTIONS_RUNNER_RUN_SHA256,
        "runner_unattended": True,
        "runner_version": OFFICIAL_ACTIONS_RUNNER_VERSION,
        "schema_version": PHASE_HOST_TOOL_CONTRACT_SCHEMA,
        "venv_root": f"{root}/venv",
        "venv_symlink_inventory_sha256": hashlib.sha256(b"venv symlinks").hexdigest(),
        "venv_tree_sha256": hashlib.sha256(b"venv tree").hexdigest(),
    }


def _provider_phase_plans(
    *,
    runner_image: str,
    timelock_sha256: str,
    code_commit: str = _COMMIT,
) -> dict[str, object]:
    host_tools = _provider_host_tools()
    plans: dict[str, object] = {}
    for position, phase in enumerate(("online", "label-release", "analysis"), start=1):
        workflow = PROVIDER_PHASE_WORKFLOWS[phase]
        claim_job, execute_job = PROVIDER_PHASE_JOB_NAMES[phase]
        platform, image_role, index_role = PROVIDER_PHASE_RUNTIME_BINDINGS[phase]
        self_hosted_path = f"/opt/fractal-confirmatory/host-tools/provider-plans/{phase}.json"
        runtime_image = (
            runner_image
            if phase != "label-release"
            else f"ghcr.io/example/timelock@sha256:{'9' * 64}"
        )
        claim_nonce = hashlib.sha256(f"nonce:{phase}".encode()).hexdigest()
        runner_label = derive_phase_runner_label(claim_nonce, phase)  # type: ignore[arg-type]
        bootstrap_receipt = {
            "approval_environment": PROVIDER_APPROVAL_ENVIRONMENT,
            "disable_update": True,
            "ephemeral": True,
            "phase": phase,
            "registered_at_utc": "2026-07-17T12:00:00+00:00",
            "repository": "mhdk1602/fractal-ann-diagnostics",
            "repository_runner_inventory_sha256": hashlib.sha256(
                f"inventory:{phase}".encode()
            ).hexdigest(),
            "runner_archive_sha256": OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
            "runner_group_id": 7,
            "runner_id": position,
            "runner_identity": PROVIDER_RUNNER_IDENTITY,
            "runner_label": runner_label,
            "runner_name": f"fractal-confirmatory-{phase}",
            "runner_version": OFFICIAL_ACTIONS_RUNNER_VERSION,
            "schema_version": "fractal-provider-runner-bootstrap-v2",
            "unattended": True,
            "workflow_sha": code_commit,
        }
        bootstrap_sha256 = hashlib.sha256(
            json.dumps(
                bootstrap_receipt,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        ).hexdigest()
        plans[phase] = {
            "activation_argv_template": [
                host_tools["python_executable"],
                "-I",
                "-S",
                "-P",
                "-s",
                "-c",
                PROVIDER_PLAN_LAUNCHER_SOURCE_BINDING,
                "fractal-host-python-verified-launcher-v1",
                "fractal_ann_diagnostics.execution_claim",
                "verify-prerequisites",
                "--phase",
                phase,
                "--suite-attempt-id",
                PROVIDER_PLAN_SUITE_BINDING,
                "--claim-receipt",
                PROVIDER_PLAN_CLAIM_RECEIPT_BINDING,
                "--activate-and-execute",
                "--output-dir",
                PROVIDER_PLAN_ACTIVATION_OUTPUT_BINDING,
                "--github-output",
                PROVIDER_PLAN_GITHUB_OUTPUT_BINDING,
            ],
            "activation_command_id": PROVIDER_PHASE_COMMAND_IDS[phase],
            "activation_environment": {
                "HOST_CONTROLLED_ROOT": host_tools["controlled_root"],
                "HOST_PYTHON_IMPORT_ROOT": host_tools["python_import_root"],
                "HOST_PYTHON_IMPORT_TREE_SHA256": host_tools["python_import_tree_sha256"],
                "HOST_PYTHON_PACKAGE_CONTENT_SHA256": host_tools["python_package_content_sha256"],
                "HOST_PYTHON_PACKAGE_TREE_SHA256": host_tools["python_package_tree_sha256"],
                "HOST_PYTHON_VENV_ROOT": host_tools["venv_root"],
                "HOST_PYTHON_VENV_SYMLINK_INVENTORY_SHA256": host_tools[
                    "venv_symlink_inventory_sha256"
                ],
                "HOST_PYTHON_VENV_TREE_SHA256": host_tools["venv_tree_sha256"],
                "HOST_PYTHON_VERIFIED_LAUNCHER_SHA256": host_tools["python_launcher_sha256"],
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "approval_environment": PROVIDER_APPROVAL_ENVIRONMENT,
            "c1_commit_binding": PROVIDER_PLAN_C1_COMMIT_BINDING,
            "claim_job_name": claim_job,
            "claim_nonce": claim_nonce,
            "claim_predecessor_binding": PROVIDER_PLAN_PREDECESSOR_BINDING,
            "claim_receipt_path_template": (
                f"/var/fractal-confirmatory/claims/{{suite_attempt_id}}/{phase}/claim-receipt.json"
            ),
            "execute_job_name": execute_job,
            "execution_claim_inputs": (
                {
                    "beacon": {
                        "chain_genesis_unix_seconds": 1_595_431_050,
                        "chain_hash": "a" * 64,
                        "chain_period_seconds": 3,
                        "chain_public_key": "ab" * 48,
                        "chain_scheme_id": "bls-unchained-g1-rfc9380",
                        "drand_network": "https://api.drand.sh",
                        "execution_round": 100,
                        "label_release_round": 120,
                        "minimum_label_release_safety_rounds": 10,
                        "schema_version": "fractal-execution-beacon-contract-v1",
                        "seed_derivation": "sha256-fractal-execution-seed-v1-u64be",
                        "verification_identity": "b" * 64,
                    },
                    "design_seed_sha256": "c" * 64,
                    "registered_online_runtime_budget_seconds": 68_000,
                }
                if phase == "online"
                else None
            ),
            "host_tools": host_tools,
            "manifest_sha256_binding": PROVIDER_PLAN_MANIFEST_BINDING,
            "maximum_runtime_seconds": PROVIDER_PHASE_RUNTIME_CEILINGS[phase],
            "oci_index_digest": runtime_image.rsplit("@", 1)[1],
            "oci_platform_manifest_digest": f"sha256:{str(position + 3) * 64}",
            "phase": phase,
            "phase_evidence_root_template": (
                f"/var/fractal-confirmatory/evidence/{{suite_attempt_id}}/{phase}"
            ),
            "phase_input_binding": PROVIDER_PLAN_PHASE_INPUT_BINDING,
            "phase_output_binding": PROVIDER_PLAN_PHASE_OUTPUT_BINDING,
            "provider_architecture": "ARM64",
            "provider_operating_system": "macOS",
            "provider_plan_path": self_hosted_path,
            "repository": "mhdk1602/fractal-ann-diagnostics",
            "run_head_branch": "confirmatory-apparatus-c0",
            "runner_archive_sha256": OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
            "runner_bootstrap_receipt": bootstrap_receipt,
            "runner_bootstrap_receipt_file_sha256": bootstrap_sha256,
            "runner_bootstrap_receipt_path": (
                f"{host_tools['controlled_root']}/production/runners/{phase}/"
                f"{runner_label}/bootstrap-receipt.json"
            ),
            "runner_group_id": 7,
            "runner_id": position,
            "runner_identity": PROVIDER_RUNNER_IDENTITY,
            "runner_name": f"fractal-confirmatory-{phase}",
            "runner_registration_bundle_path": (
                f"{host_tools['controlled_root']}/production/runner-registrations/"
                f"{phase}/{runner_label}"
            ),
            "runner_registration_bundle_sha256": hashlib.sha256(
                f"registration-bundle:{phase}".encode()
            ).hexdigest(),
            "runner_registration_evidence_file_sha256": hashlib.sha256(
                f"registration-evidence:{phase}".encode()
            ).hexdigest(),
            "runner_version": OFFICIAL_ACTIONS_RUNNER_VERSION,
            "runtime_image": runtime_image,
            "runtime_image_role": image_role,
            "runtime_index_role": index_role,
            "runtime_platform": platform,
            "runtime_probe_receipt_sha256": hashlib.sha256(f"probe:{phase}".encode()).hexdigest(),
            "schema_version": PROVIDER_PHASE_PLAN_TEMPLATE_SCHEMA,
            "suite_attempt_id_binding": PROVIDER_PLAN_SUITE_BINDING,
            "tle_binary_sha256": timelock_sha256 if phase == "label-release" else None,
            "tle_build_provenance_sha256": (
                hashlib.sha256(b"tle build").hexdigest() if phase == "label-release" else None
            ),
            "tle_interoperability_receipt_sha256": (
                hashlib.sha256(b"tle interop").hexdigest() if phase == "label-release" else None
            ),
            "tle_vulnerability_scan_sha256": (
                hashlib.sha256(b"tle scan").hexdigest() if phase == "label-release" else None
            ),
            "workflow_path": workflow,
            "workflow_ref": (
                f"mhdk1602/fractal-ann-diagnostics/{workflow}@refs/tags/confirmatory-apparatus-c0"
            ),
            "workflow_sha": code_commit,
        }
    return plans


def _artifact(
    role: str,
    kind: str,
    *,
    frozen: bool,
    corpus_id: str | None = None,
) -> dict[str, object]:
    identifier = f"{corpus_id}-{role}" if corpus_id is not None else role
    artifact_sha256 = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    if frozen and role == "timelock-tool":
        artifact_sha256 = SOURCE_BUILT_LINUX_ARM64_TLE_SHA256
    if frozen and role == "source-code":
        revision = _COMMIT
    elif frozen and role == "online-execution":
        revision = f"sha256:{hashlib.sha256((identifier + '-logical').encode()).hexdigest()}"
    elif frozen and role == "opa-runtime-binary":
        revision = f"sha256:{artifact_sha256}"
    else:
        revision = "v1.0.0"
    artifact: dict[str, object] = {
        "kind": kind,
        "id": identifier,
        "uri": f"https://example.test/{identifier}",
        "revision": revision if frozen else "tbd",
        "sha256": artifact_sha256 if frozen else "tbd",
        "license": "MIT",
        "role": role,
    }
    if corpus_id is not None:
        artifact["corpus_id"] = corpus_id
    return artifact


def _artifacts(*, frozen: bool) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for role, kind in (
        ("sealed-inputs", "dataset"),
        ("sealed-labels", "dataset"),
        ("sealed-label-ciphertext", "encrypted-dataset"),
        ("timelock-encryption-receipt", "custody"),
        ("online-execution", "execution"),
        ("corpus-normalizer", "normalizer"),
        ("policy-workload", "policy-data"),
        ("embedding-store", "embedding-store"),
        ("authorized-index-store", "index-store"),
        ("trial-runtime-package", "runtime-input"),
        ("runtime-attestation-plan-template", "execution"),
    ):
        artifacts.extend(
            _artifact(role, kind, frozen=frozen, corpus_id=corpus_id) for corpus_id in FIXED_CORPORA
        )
    artifacts.extend(_artifact(role, kind, frozen=frozen) for role, kind in _ROLE_KINDS)
    return artifacts


def _manifest(
    *,
    frozen: bool = False,
    receipt_root: Path | None = None,
) -> dict[str, object]:
    receipt_directory = receipt_root or Path("/tmp/fractal-ann-confirmatory-receipts")
    return {
        "schema_version": "1.0",
        "protocol_version": "0.3.0" if frozen else "0.3.0-draft",
        "status": "frozen" if frozen else "draft",
        "claim_scope": "suite-conditional-retrieval-control",
        "primary_claim": REGISTERED_PRIMARY_CLAIM,
        "freeze_blockers": [] if frozen else ["artifact hashes and power design remain tbd"],
        "analysis": {
            "k": 10,
            "failure_recall_threshold": 0.9,
            "alpha": 0.05,
            "bootstrap_seed": 20260713,
            "h1_minimum_risk_increase": 0.0,
            "power_target": 0.9,
            "retrieval_target_noninferiority_margin": 0.01,
            "evidence_sufficiency_noninferiority_margin": 0.01,
            "minimum_cost_reduction": 0.1,
            "maximum_p95_latency_ratio": 1.25,
            "maximum_entitlement_violations": 0,
            "minimum_corpora_with_geometry_gain": 4,
            "geometry_reference_model": "system-policy",
            "geometry_candidate_model": "full",
            "geometry_gain_metrics": [
                "log_loss_reduction",
                "brier_score_reduction",
                "auprc_gain",
            ],
            "geometry_gain_thresholds": {
                "log_loss_reduction": 0.0 if frozen else "tbd",
                "brier_score_reduction": 0.0 if frozen else "tbd",
                "auprc_gain": 0.0 if frozen else "tbd",
            },
            "low_geometry": ({"instability": 0.1, "lid": 1.0} if frozen else "tbd"),
            "high_geometry": ({"instability": 0.9, "lid": 9.0} if frozen else "tbd"),
            "cluster_unit": "query_family",
            "corpus_weighting": "equal",
            "interval_construction": "directional-one-sided-95",
            "gatekeeping": "intersection-union-primary-gates",
            "cost_estimand": "end-to-end-request-latency-family-relative-reduction",
            "bootstrap_replicates": 10_000,
            "nested_rows_per_family": 3 if frozen else "tbd",
            "fixed_corpora": list(FIXED_CORPORA),
            "evidence_corpora": list(EVIDENCE_CORPORA),
            "action_set": list(REGISTERED_ACTION_SET),
            "static_comparator_action": "hnsw-high" if frozen else "tbd",
            "power": {
                "model": "development-family-cluster-resampling",
                "joint_success_event": "h2-and-h3-all-gates-pass",
                "registered_endpoints": list(REGISTERED_POWER_ENDPOINTS),
                "dependence_source": (
                    "development query-family endpoint vectors" if frozen else "tbd"
                ),
                "effect_scenarios": (
                    ["registered-minimum-effects", "development-observed-effects"]
                    if frozen
                    else ["tbd-expected-effect", "tbd-conservative-effect"]
                ),
                "candidate_families_per_corpus": list(REGISTERED_POWER_FAMILY_CANDIDATES),
                "selection_cell_alpha": 0.05 / 12,
                "selection_exact_blocking_failures": 445,
                "selection_exact_qualifying_passes": 4_556,
                "selection_family_size": 12,
                "selection_familywise_confidence": 0.95,
                "selection_multiplicity_method": (
                    "bonferroni-fixed-required-scenario-candidate-grid-v1"
                ),
                "selected_families_per_corpus": 75 if frozen else "tbd",
                "simulation_seed": 71 if frozen else "tbd",
                "simulation_count": 5_000,
                "selected_joint_power_lower_bound": 0.91 if frozen else "tbd",
            },
        },
        "artifacts": _artifacts(frozen=frozen),
        "production_workloads": (
            registered_production_workloads(
                fixed_corpora=FIXED_CORPORA,
                runner_image=f"ghcr.io/example/study@sha256:{'2' * 64}",
                runner_identity=_RUNNER_IDENTITY,
                code_commit=_COMMIT,
                selected_family_count=75,
            )
            if frozen
            else PRODUCTION_WORKLOADS_UNRESOLVED
        ),
        "sealed_execution": {
            "reserve_fraction": 0.0,
            "custodian": "custodian@example.test" if frozen else "unassigned",
            "approval_environment": "confirmatory" if frozen else "tbd",
            "results_store": "file:///controlled/immutable-results" if frozen else "tbd",
            "runner_identity": _RUNNER_IDENTITY if frozen else "tbd",
            "code_commit": _COMMIT if frozen else "tbd",
            "c0_evidence_release": (_c0_evidence_release_binding(_COMMIT) if frozen else "tbd"),
            "runner_image": (f"ghcr.io/example/study@sha256:{'2' * 64}" if frozen else "tbd"),
            "provider_phase_plans": (
                _provider_phase_plans(
                    runner_image=f"ghcr.io/example/study@sha256:{'2' * 64}",
                    timelock_sha256=SOURCE_BUILT_LINUX_ARM64_TLE_SHA256,
                )
                if frozen
                else "tbd"
            ),
            "production_controls": {
                "materialization_config_file_sha256": "3" * 64 if frozen else "tbd",
                "blueprint_receipt_sha256": "4" * 64 if frozen else "tbd",
                "blueprint_receipt_file_sha256": "5" * 64 if frozen else "tbd",
            },
            "hardware": {
                "provider": "aws" if frozen else "tbd",
                "instance_type": "c7i.4xlarge" if frozen else "tbd",
                "cpu_model": "Intel Xeon Platinum 8488C" if frozen else "tbd",
                "logical_cores": 16 if frozen else "tbd",
                "memory_gib": 32 if frozen else "tbd",
                "accelerator": "none" if frozen else "tbd",
                "region": "us-east-1" if frozen else "tbd",
                "operating_system": "ubuntu-24.04" if frozen else "tbd",
            },
            "receipt_uri_template": (
                receipt_directory.resolve().as_uri() + "/{manifest_sha256}.json"
                if frozen
                else "tbd"
            ),
            "label_artifacts_withheld_until_prediction_receipt": True,
            "public_query_reidentification_risk": ("accepted-public-benchmark-limitation"),
            "runner_network_access": "disabled",
            "interactive_access": "disabled",
        },
    }


def _candidate_rehearsal_manifest() -> dict[str, object]:
    payload = _manifest(frozen=True)
    payload["status"] = "draft"
    payload["protocol_version"] = "0.3.0-draft"
    payload["freeze_blockers"] = ["the immutable C0 evidence release remains unresolved"]
    sealed = payload["sealed_execution"]
    assert isinstance(sealed, dict)
    sealed["c0_evidence_release"] = "tbd"
    sealed["code_commit"] = C0_COMMIT_SENTINEL

    source = _artifact_for(payload, "source-code")
    source["revision"] = C0_COMMIT_SENTINEL

    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    for row in workloads:
        assert isinstance(row, dict)
        spec = row["spec"]
        assert isinstance(spec, dict)
        spec["code_commit"] = C0_COMMIT_SENTINEL
        row["canonical_file_sha256"] = production_workload_file_sha256(spec)

    plans = sealed["provider_phase_plans"]
    assert isinstance(plans, dict)
    for plan in plans.values():
        assert isinstance(plan, dict)
        plan["workflow_sha"] = C0_COMMIT_SENTINEL
        bootstrap = plan["runner_bootstrap_receipt"]
        assert isinstance(bootstrap, dict)
        bootstrap["workflow_sha"] = C0_COMMIT_SENTINEL
        plan["runner_bootstrap_receipt_file_sha256"] = hashlib.sha256(
            _canonical(bootstrap) + b"\n"
        ).hexdigest()
    return payload


def _artifact_for(payload: dict[str, object], role: str) -> dict[str, object]:
    return next(
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] == role
    )


def _workload_spec_for(payload: dict[str, object], position: int = 0) -> dict[str, object]:
    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    row = workloads[position]
    assert isinstance(row, dict)
    spec = row["spec"]
    assert isinstance(spec, dict)
    return spec


def _verification_receipt_path(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    name: str = "artifact-verification.json",
    manifest_digest: str | None = None,
    omit_id: str | None = None,
    digest_override: tuple[str, str] | None = None,
    exact_override: tuple[str, bool] | None = None,
    add_unexpected: bool = False,
) -> Path:
    rows: list[VerifiedArtifact] = []
    for position, artifact in enumerate(payload["artifacts"]):  # type: ignore[union-attr]
        artifact_id = str(artifact["id"])
        if artifact_id == omit_id:
            continue
        digest = str(artifact["sha256"])
        if digest_override is not None and artifact_id == digest_override[0]:
            digest = digest_override[1]
        exact = not (
            exact_override is not None
            and artifact_id == exact_override[0]
            and exact_override[1] is False
        )
        rows.append(
            VerifiedArtifact(
                artifact_id=artifact_id,
                relative_path=f"objects/{position}.bin",
                kind="file" if exact else "directory",
                exact=exact,
                expected_sha256=digest,
                verified_sha256=digest,
                file_count=1,
                directory_count=0,
                byte_count=1,
                observed_file_count=1,
                observed_directory_count=0,
                observed_byte_count=1,
            )
        )
    if add_unexpected:
        rows.append(
            VerifiedArtifact(
                artifact_id="unexpected-artifact",
                relative_path="objects/unexpected.bin",
                kind="file",
                exact=True,
                expected_sha256="e" * 64,
                verified_sha256="e" * 64,
                file_count=1,
                directory_count=0,
                byte_count=1,
                observed_file_count=1,
                observed_directory_count=0,
                observed_byte_count=1,
            )
        )
    receipt = ArtifactVerificationReceipt(
        manifest_sha256=manifest_digest or manifest_sha256(payload),
        artifacts=tuple(rows),
    )
    receipt_root = tmp_path / "artifact-receipts"
    receipt_root.mkdir(exist_ok=True)
    target = receipt_root / name
    write_verification_receipt(receipt, target)
    return target


def _registration_receipt_path(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    manifest_digest: str | None = None,
    registered_at_utc: str = "2026-07-13T12:00:00+00:00",
    extra_field: bool = False,
) -> Path:
    manifest_value = manifest_digest or manifest_sha256(payload)
    registry_identity = "osf-registration:test-2026-07-13"
    registry_uri = "https://osf.io/registries/test-registration"
    record = tmp_path / "protocol-registration-record.json"
    registry_record = ProtocolRegistryRecord(
        manifest_sha256=manifest_value,
        protocol_version="0.3.0",
        registered_at_utc=registered_at_utc,
        registry_identity=registry_identity,
        registry_uri=registry_uri,
    )
    record.write_bytes(registry_record.canonical_bytes() + b"\n")
    receipt = ProtocolRegistrationReceipt(
        manifest_sha256=manifest_value,
        protocol_version="0.3.0",
        registered_at_utc=registered_at_utc,
        registry_identity=registry_identity,
        registry_uri=registry_uri,
        registry_record_sha256=registry_record.record_sha256,
    ).to_dict()
    if extra_field:
        receipt["unregistered_field"] = "forbidden"
    target = tmp_path / "protocol-registration.json"
    target.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return target


def _verified_registration(
    tmp_path: Path,
    registration_receipt_path: Path,
    *,
    fresh_revalidator: object | None = None,
) -> VerifiedC1ProtocolRegistration:
    """Mint the private test bridge without weakening the production entry point."""

    registration_record_path = tmp_path / "protocol-registration-record.json"
    record = study_module.load_protocol_registry_record(registration_record_path)
    receipt = load_protocol_registration_receipt(registration_receipt_path)
    position = 0
    while True:
        package_root = tmp_path / f"verified-c1-test-package-{position}"
        try:
            package_root.mkdir(mode=0o700)
        except FileExistsError:
            position += 1
            continue
        break
    package_record = package_root / "protocol-registry-record.json"
    package_record.write_bytes(registration_record_path.read_bytes())
    callback = fresh_revalidator if fresh_revalidator is not None else (lambda: None)
    assert callable(callback)
    return study_module._mint_verified_c1_protocol_registration(
        record=record,
        receipt=receipt,
        package_root=package_root,
        registration_record_path=registration_record_path,
        registration_receipt_path=registration_receipt_path,
        c0_commit="0" * 40,
        c1_commit="1" * 40,
        package_file_sha256s=(("protocol-registry-record.json", record.record_sha256),),
        fresh_revalidator=callback,
    )


def _begin_with_test_registration(
    tmp_path: Path,
    manifest_path: Path,
    lock_path: Path,
    *,
    runner_identity: str,
    artifact_verification_receipt_path: Path,
    registration_receipt_path: Path,
) -> SealedRunReceipt:
    """Open through the typed boundary while stubbing unrelated artifact I/O."""

    verified_registration = _verified_registration(tmp_path, registration_receipt_path)
    admitted_receipt = load_verification_receipt(artifact_verification_receipt_path)
    artifact_root = tmp_path / "test-artifact-root"
    artifact_root.mkdir(exist_ok=True)
    artifact_map = tmp_path / "test-artifact-map.json"
    artifact_map.write_text("{}\n", encoding="utf-8")
    with (
        patch.object(study_module, "load_local_artifact_map", return_value=()),
        patch.object(
            study_module,
            "verify_local_artifacts",
            return_value=admitted_receipt,
        ),
    ):
        return begin_sealed_run(
            manifest_path,
            lock_path,
            runner_identity=runner_identity,
            artifact_verification_receipt_path=artifact_verification_receipt_path,
            artifact_root=artifact_root,
            local_artifact_map_path=artifact_map,
            verified_protocol_registration=verified_registration,
        )


def test_repository_draft_manifest_validates_with_explicit_blockers() -> None:
    payload = load_study_manifest("research/study-manifest.json")
    validate_study_manifest(payload)
    assert payload["status"] == "draft"
    assert payload["freeze_blockers"]
    power = payload["analysis"]["power"]
    assert power["selection_family_size"] == 12
    assert power["selection_cell_alpha"] == pytest.approx(0.05 / 12)
    assert power["selection_exact_qualifying_passes"] == 4_556
    assert power["selection_exact_blocking_failures"] == 445
    assert payload["production_workloads"] == PRODUCTION_WORKLOADS_UNRESOLVED


def test_frozen_provider_plans_resolve_only_manifest_and_c1_commit(
    tmp_path: Path,
) -> None:
    payload = _manifest(frozen=True)
    manifest_path = tmp_path / "study-manifest.json"
    manifest_path.write_bytes(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    c1_commit = "9" * 40
    plans = load_provider_phase_plans(manifest_path, c1_commit=c1_commit)
    assert set(plans) == {"online", "label-release", "analysis"}
    assert len({plan.provider_plan_path for plan in plans.values()}) == 3
    template_digest = provider_phase_plan_templates_sha256(payload)
    assert len(template_digest) == 64
    for phase, plan in plans.items():
        assert plan.phase == phase
        assert plan.manifest_sha256 == manifest_sha256(payload)
        assert plan.c1_commit == c1_commit
        assert plan.workflow_sha == _COMMIT
        assert plan.runner_group_id == 7
        assert plan.oci_index_digest == plan.runtime_image.rsplit("@", 1)[1]
        assert PROVIDER_PLAN_MANIFEST_BINDING not in plan.canonical_file_bytes().decode()
        assert PROVIDER_PLAN_C1_COMMIT_BINDING in plan.canonical_file_bytes().decode()
        if phase == "online":
            assert plan.execution_claim_inputs is not None
            assert plan.execution_claim_inputs.registered_online_runtime_budget_seconds == 68_000
        else:
            assert plan.execution_claim_inputs is None


def test_candidate_rehearsal_admits_only_the_exact_c0_sentinel_paths(
    tmp_path: Path,
) -> None:
    payload = _candidate_rehearsal_manifest()
    validate_candidate_rehearsal_manifest(payload, c0_commit=_COMMIT)
    resolved = resolve_candidate_c0_commit_sentinels(payload, c0_commit=_COMMIT)
    encoded = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    assert C0_COMMIT_SENTINEL not in encoded
    assert C0_COMMIT_SENTINEL_PATHS == {
        "sealed_execution.code_commit",
        "artifacts[role=source-code].revision",
        *(
            f"production_workloads[corpus_id={corpus_id}].spec.code_commit"
            for corpus_id in FIXED_CORPORA
        ),
        *(
            f"sealed_execution.provider_phase_plans.{phase}.workflow_sha"
            for phase in ("online", "label-release", "analysis")
        ),
        *(
            f"sealed_execution.provider_phase_plans.{phase}.runner_bootstrap_receipt.workflow_sha"
            for phase in ("online", "label-release", "analysis")
        ),
    }

    manifest_path = tmp_path / "candidate-study-manifest.json"
    manifest_path.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(ExecutionClaimError, match="status='frozen'"):
        load_provider_phase_plans(manifest_path, c1_commit="9" * 40)

    plans = load_provider_phase_plans(
        manifest_path,
        c1_commit=_COMMIT,
        validation_mode="candidate-rehearsal",
        c0_commit=_COMMIT,
    )
    assert set(plans) == {"online", "label-release", "analysis"}
    assert {plan.workflow_sha for plan in plans.values()} == {_COMMIT}
    assert {plan.runner_bootstrap_receipt.workflow_sha for plan in plans.values()} == {_COMMIT}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["sealed_execution"].__setitem__(  # type: ignore[union-attr]
                "runner_identity", C0_COMMIT_SENTINEL
            ),
            "sentinel path set differs",
        ),
        (
            lambda payload: payload["sealed_execution"].__setitem__(  # type: ignore[union-attr]
                "code_commit", _COMMIT
            ),
            "sentinel path set differs",
        ),
        (
            lambda payload: payload["analysis"].__setitem__(  # type: ignore[union-attr]
                "static_comparator_action", "tbd"
            ),
            "must be pinned before freeze",
        ),
    ],
)
def test_candidate_rehearsal_rejects_extra_missing_and_unresolved_values(
    mutate: object,
    match: str,
) -> None:
    payload = _candidate_rehearsal_manifest()
    assert callable(mutate)
    mutate(payload)
    with pytest.raises(StudyManifestError, match=match):
        validate_candidate_rehearsal_manifest(payload, c0_commit=_COMMIT)


def test_normalized_provider_plan_closure_and_c1_transition_reject_mutation() -> None:
    candidate = _candidate_rehearsal_manifest()
    frozen = _manifest(frozen=True)
    binding = _c0_evidence_release_binding(_COMMIT)
    apparatus = binding["apparatus_evidence"]
    assert isinstance(apparatus, dict)
    apparatus["rehearsal_manifest_sha256"] = manifest_sha256(candidate)
    apparatus["provider_phase_plan_closure_sha256"] = provider_phase_plan_templates_sha256(
        candidate,
        validation_mode="candidate-rehearsal",
        c0_commit=_COMMIT,
    )
    binding["apparatus_evidence_sha256"] = hashlib.sha256(
        canonical_apparatus_evidence_bytes(apparatus)
    ).hexdigest()
    frozen["sealed_execution"]["c0_evidence_release"] = binding  # type: ignore[index]
    closure = assert_normalized_provider_phase_plan_closure(
        candidate,
        frozen,
        c0_commit=_COMMIT,
    )
    assert len(closure) == 64
    validate_candidate_rehearsal_to_frozen_transition(
        candidate,
        frozen,
        c0_commit=_COMMIT,
    )

    changed = _manifest(frozen=True)
    changed["sealed_execution"]["c0_evidence_release"] = copy.deepcopy(binding)  # type: ignore[index]
    changed_plans = changed["sealed_execution"]["provider_phase_plans"]  # type: ignore[index]
    changed_plans["online"]["execution_claim_inputs"][
        "registered_online_runtime_budget_seconds"
    ] = 67_000
    with pytest.raises(ExecutionClaimError, match="normalized provider-plan closure differs"):
        assert_normalized_provider_phase_plan_closure(
            candidate,
            changed,
            c0_commit=_COMMIT,
        )
    with pytest.raises(StudyManifestError, match="outside the registered candidate transition"):
        validate_candidate_rehearsal_to_frozen_transition(
            candidate,
            changed,
            c0_commit=_COMMIT,
        )


@pytest.mark.parametrize("phase", ["label-release", "analysis"])
def test_only_online_plan_can_carry_execution_claim_inputs(phase: str) -> None:
    payload = _manifest(frozen=True)
    plans = payload["sealed_execution"]["provider_phase_plans"]  # type: ignore[index]
    plans[phase]["execution_claim_inputs"] = plans["online"]["execution_claim_inputs"]
    with pytest.raises(StudyManifestError, match="must be null outside"):
        validate_study_manifest(payload)


def test_online_execution_claim_inputs_are_closed_and_bounded() -> None:
    payload = _manifest(frozen=True)
    inputs = payload["sealed_execution"]["provider_phase_plans"]["online"][  # type: ignore[index]
        "execution_claim_inputs"
    ]
    inputs["caller_argv"] = ["docker", "run", "mutable:latest"]
    with pytest.raises(StudyManifestError, match="schema mismatch"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    inputs = payload["sealed_execution"]["provider_phase_plans"]["online"][  # type: ignore[index]
        "execution_claim_inputs"
    ]
    inputs["registered_online_runtime_budget_seconds"] = 72_001
    with pytest.raises(StudyManifestError, match="exceeds the phase ceiling"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    inputs = payload["sealed_execution"]["provider_phase_plans"]["online"][  # type: ignore[index]
        "execution_claim_inputs"
    ]
    inputs["measured_full_suite_runtime_seconds"] = 68_000
    with pytest.raises(StudyManifestError, match="schema mismatch"):
        validate_study_manifest(payload)


def test_provider_plans_reject_bootstrap_path_substitution_and_collision() -> None:
    payload = _manifest(frozen=True)
    plans = payload["sealed_execution"]["provider_phase_plans"]  # type: ignore[index]
    plans["analysis"]["runner_bootstrap_receipt_path"] = plans["online"][
        "runner_bootstrap_receipt_path"
    ]
    with pytest.raises(StudyManifestError, match="bootstrap_receipt_path differs"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    plans = payload["sealed_execution"]["provider_phase_plans"]  # type: ignore[index]
    plans["online"]["runner_bootstrap_receipt_file_sha256"] = "mutable"
    with pytest.raises(StudyManifestError, match="not a SHA-256 binding"):
        validate_study_manifest(payload)


def test_hosted_plan_materialization_is_distinct_and_byte_exact(tmp_path: Path) -> None:
    payload = _manifest(frozen=True)
    manifest_path = tmp_path / "study-manifest.json"
    manifest_path.write_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    )
    plan = load_provider_phase_plans(manifest_path, c1_commit="9" * 40)["online"]
    materialized = materialize_provider_phase_plan(plan, tmp_path / "hosted-evidence")
    assert materialized != Path(plan.provider_plan_path)
    assert materialized.read_bytes() == plan.canonical_file_bytes()
    assert hashlib.sha256(materialized.read_bytes()).hexdigest() == plan.file_sha256
    with pytest.raises(ExecutionClaimError, match="cannot create"):
        materialize_provider_phase_plan(plan, tmp_path / "hosted-evidence")


def test_provider_plan_loader_rejects_noncanonical_manifest_bytes(tmp_path: Path) -> None:
    payload = _manifest(frozen=True)
    manifest_path = tmp_path / "study-manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ExecutionClaimError, match="not canonical"):
        load_provider_phase_plans(manifest_path, c1_commit="9" * 40)


def test_frozen_manifest_requires_resolved_public_workloads() -> None:
    payload = _manifest(frozen=True)
    payload["production_workloads"] = PRODUCTION_WORKLOADS_UNRESOLVED
    with pytest.raises(StudyManifestError, match="must be resolved before freeze"):
        validate_study_manifest(payload)


def test_public_workloads_are_exactly_five_rows_in_registered_order() -> None:
    payload = _manifest(frozen=True)
    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    workloads.reverse()
    with pytest.raises(StudyManifestError, match="corpus_id must equal 'scifact'"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    workloads.pop()
    with pytest.raises(StudyManifestError, match="exactly 5 rows"):
        validate_study_manifest(payload)


@pytest.mark.parametrize("location", ("wrapper", "spec"))
def test_public_workload_schema_is_closed(location: str) -> None:
    payload = _manifest(frozen=True)
    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    row = workloads[0]
    assert isinstance(row, dict)
    target = row if location == "wrapper" else row["spec"]
    assert isinstance(target, dict)
    target["unregistered_override"] = True
    with pytest.raises(StudyManifestError, match="unknown=.*unregistered_override"):
        validate_study_manifest(payload)


def test_public_workload_hash_binds_canonical_file_and_terminal_newline() -> None:
    payload = _manifest(frozen=True)
    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    row = workloads[0]
    assert isinstance(row, dict)
    spec = row["spec"]
    assert isinstance(spec, dict)
    canonical_without_newline = json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert (
        row["canonical_file_sha256"]
        == hashlib.sha256(canonical_without_newline + b"\n").hexdigest()
    )
    assert row["canonical_file_sha256"] != hashlib.sha256(canonical_without_newline).hexdigest()

    spec["factory_config_sha256"] = "f" * 64
    with pytest.raises(StudyManifestError, match="differs from the canonical spec file"):
        validate_study_manifest(payload)


def test_public_workload_family_count_matches_registered_power_design() -> None:
    payload = _manifest(frozen=True)
    spec = _workload_spec_for(payload)
    spec["selected_family_count"] = 74
    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    row = workloads[0]
    assert isinstance(row, dict)
    row["canonical_file_sha256"] = production_workload_file_sha256(spec)
    with pytest.raises(
        StudyManifestError,
        match="differs from analysis.power.selected_families_per_corpus",
    ):
        validate_study_manifest(payload)


def test_public_workload_schema_and_runtime_parser_cannot_drift() -> None:
    from fractal_ann_diagnostics.production_corpus_run import ProductionCorpusWorkloadSpec

    payload = _manifest(frozen=True)
    spec = _workload_spec_for(payload)
    parsed = ProductionCorpusWorkloadSpec.from_dict(spec)
    assert set(parsed.__dataclass_fields__) == PRODUCTION_WORKLOAD_SPEC_FIELDS
    assert parsed.to_dict() == spec
    assert parsed.file_sha256 == production_workload_file_sha256(spec)


@pytest.mark.parametrize("field", ("runner_image", "runner_identity", "code_commit"))
def test_public_workload_runner_identity_matches_sealed_execution(field: str) -> None:
    payload = _manifest(frozen=True)
    spec = _workload_spec_for(payload)
    if field == "runner_image":
        spec[field] = f"ghcr.io/example/other@sha256:{'3' * 64}"
    elif field == "code_commit":
        spec[field] = "4" * 40
    else:
        spec[field] = "github-actions:environment:other"
    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    row = workloads[0]
    assert isinstance(row, dict)
    row["canonical_file_sha256"] = production_workload_file_sha256(spec)
    with pytest.raises(StudyManifestError, match=rf"{field} differs from sealed_execution.{field}"):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("approval_environment", "confirmatory-other"),
        ("runner_identity", "github-actions:environment:confirmatory-other"),
        ("runner_identity", "github-actions:ref:refs/heads/master"),
    ),
)
def test_runner_identity_is_derived_from_approval_environment(
    field: str,
    value: str,
) -> None:
    payload = _manifest(frozen=True)
    sealed = payload["sealed_execution"]
    assert isinstance(sealed, dict)
    sealed[field] = value

    with pytest.raises(
        StudyManifestError,
        match=(
            r"sealed_execution\.runner_identity must equal "
            r"github-actions:environment:\{approval_environment\}"
        ),
    ):
        validate_study_manifest(payload)


def test_public_workload_rejects_noncanonical_path_and_duplicate_feature_block() -> None:
    payload = _manifest(frozen=True)
    spec = _workload_spec_for(payload)
    spec["artifact_root"] = "/opt/study/../escape"
    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    row = workloads[0]
    assert isinstance(row, dict)
    row["canonical_file_sha256"] = production_workload_file_sha256(spec)
    with pytest.raises(StudyManifestError, match="canonical absolute POSIX path"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    spec = _workload_spec_for(payload)
    features = spec["feature_bindings"]
    assert isinstance(features, list)
    features.append(dict(features[0]))
    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    row = workloads[0]
    assert isinstance(row, dict)
    row["canonical_file_sha256"] = production_workload_file_sha256(spec)
    with pytest.raises(StudyManifestError, match="repeats a schedule block key"):
        validate_study_manifest(payload)


def test_public_workload_rejects_nonfinite_or_boolean_numeric_fields() -> None:
    payload = _manifest(frozen=True)
    spec = _workload_spec_for(payload)
    features = spec["feature_bindings"]
    assert isinstance(features, list)
    feature = features[0]
    assert isinstance(feature, dict)
    feature["version_lag"] = float("nan")
    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    row = workloads[0]
    assert isinstance(row, dict)
    row["canonical_file_sha256"] = "0" * 64
    with pytest.raises(StudyManifestError, match="finite and non-negative"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    spec = _workload_spec_for(payload)
    spec["selected_family_count"] = True
    workloads = payload["production_workloads"]
    assert isinstance(workloads, list)
    row = workloads[0]
    assert isinstance(row, dict)
    row["canonical_file_sha256"] = "0" * 64
    with pytest.raises(StudyManifestError, match="positive integer"):
        validate_study_manifest(payload)


def test_draft_manifest_is_valid_but_cannot_open_sealed_run() -> None:
    payload = _manifest()
    validate_study_manifest(payload)
    with pytest.raises(StudyManifestError, match="status='frozen'"):
        validate_study_manifest(payload, require_frozen=True)
    payload["freeze_blockers"] = []
    with pytest.raises(StudyManifestError, match="must state its explicit freeze blockers"):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("location", "field"),
    (
        ("root", "surprise"),
        ("analysis", "unregistered_gate"),
        ("power", "unregistered_assumption"),
        ("artifact", "mutable_tag"),
        ("hardware", "benchmark_mode"),
    ),
)
def test_closed_schema_rejects_unknown_fields(location: str, field: str) -> None:
    payload = _manifest()
    if location == "root":
        payload[field] = True
    elif location == "analysis":
        payload["analysis"][field] = True  # type: ignore[index]
    elif location == "power":
        payload["analysis"]["power"][field] = True  # type: ignore[index]
    elif location == "artifact":
        payload["artifacts"][0][field] = True  # type: ignore[index]
    else:
        payload["sealed_execution"]["hardware"][field] = True  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="unknown"):
        validate_study_manifest(payload)


def test_exact_artifact_roles_and_corpus_coverage_are_required() -> None:
    payload = _manifest(frozen=True)
    payload["artifacts"] = [  # type: ignore[assignment]
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] != "h1-predictive-model"
    ]
    with pytest.raises(StudyManifestError, match="h1-predictive-model.*exactly 1"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    normalizers = [
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] == "corpus-normalizer"
    ]
    normalizers[0]["corpus_id"] = normalizers[1]["corpus_id"]
    with pytest.raises(StudyManifestError, match="cover every fixed corpus exactly once"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    _artifact_for(payload, "sealed-labels")["role"] = "sealed-inputs"
    with pytest.raises(StudyManifestError, match="sealed-inputs.*exactly 5"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["artifacts"] = [  # type: ignore[assignment]
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] != "online-execution"
    ]
    with pytest.raises(StudyManifestError, match="online-execution.*exactly 5"):
        validate_study_manifest(payload)


def test_frozen_analysis_requires_registered_seed_and_geometry_profiles() -> None:
    payload = _manifest(frozen=True)
    payload["analysis"]["bootstrap_seed"] = 20260714  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="bootstrap_seed"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["analysis"]["low_geometry"] = "tbd"  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="low_geometry.*pinned"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["analysis"]["nested_rows_per_family"] = "tbd"  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="nested_rows_per_family.*pinned"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["analysis"]["nested_rows_per_family"] = 0  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="nested_rows_per_family.*at least 1"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["analysis"]["high_geometry"] = {  # type: ignore[index]
        "different-feature": 0.9
    }
    with pytest.raises(StudyManifestError, match="name identical features"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    input_artifact = _artifact_for(payload, "sealed-inputs")
    label_artifact = next(
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] == "sealed-labels"
        and artifact["corpus_id"] == input_artifact["corpus_id"]
    )
    label_artifact["uri"] = input_artifact["uri"]
    label_artifact["sha256"] = input_artifact["sha256"]
    with pytest.raises(StudyManifestError, match="must be separately pinned"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    input_artifact = _artifact_for(payload, "sealed-inputs")
    execution_artifact = next(
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] == "online-execution"
        and artifact["corpus_id"] == input_artifact["corpus_id"]
    )
    execution_artifact["uri"] = input_artifact["uri"]
    execution_artifact["sha256"] = input_artifact["sha256"]
    with pytest.raises(StudyManifestError, match="online execution.*separately pinned"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    label_artifact = _artifact_for(payload, "sealed-labels")
    ciphertext_artifact = next(
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] == "sealed-label-ciphertext"
        and artifact["corpus_id"] == label_artifact["corpus_id"]
    )
    ciphertext_artifact["uri"] = label_artifact["uri"]
    ciphertext_artifact["sha256"] = label_artifact["sha256"]
    with pytest.raises(
        StudyManifestError,
        match="sealed-label ciphertext.*separately pinned",
    ):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    fit_data = _artifact_for(payload, "development-fit-data")
    calibration_data = _artifact_for(payload, "development-calibration-data")
    calibration_data["uri"] = fit_data["uri"]
    with pytest.raises(StudyManifestError, match="fit and calibration"):
        validate_study_manifest(payload)


def test_frozen_status_always_enforces_pins_and_rejects_minimal_artifacts() -> None:
    payload = _manifest(frozen=True)
    _artifact_for(payload, "primary-embedding")["sha256"] = "tbd"
    with pytest.raises(StudyManifestError, match="pinned sha256"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["artifacts"] = payload["artifacts"][:4]  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="requires exactly"):
        validate_study_manifest(payload)


def test_action_set_and_noninferiority_gates_are_exact() -> None:
    payload = _manifest()
    payload["analysis"]["action_set"].append("rerank")  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="registered action set"):
        validate_study_manifest(payload)

    for field in (
        "retrieval_target_noninferiority_margin",
        "evidence_sufficiency_noninferiority_margin",
    ):
        payload = _manifest()
        payload["analysis"][field] = 0.02  # type: ignore[index]
        with pytest.raises(StudyManifestError, match=field):
            validate_study_manifest(payload)


def test_registered_claim_and_geometry_baseline_are_exact() -> None:
    assert REGISTERED_PRIMARY_CLAIM == (
        "On the fixed five-corpus suite, a frozen full model that adds LID at k=50, "
        "LID-CV, relative contrast, and radius expansion to the frozen system-policy "
        "baseline improves held-out prediction of intent-to-treat low-effort action "
        "failure beyond the frozen H2 thresholds; and a frozen adaptive controller "
        "achieves an equal-corpus mean family-level relative end-to-end request-latency "
        "reduction greater than 10% relative to a frozen static action while authorized "
        "retrieval-target attainment and complete-evidence sufficiency remain noninferior "
        "within one percentage point, the equal-corpus mean of within-corpus "
        "proposed-to-comparator p95 ratios of family-mean end-to-end request latency "
        "remains below 1.25, and no denied item is emitted at the controlled retrieval "
        "boundary."
    )

    payload = _manifest()
    payload["analysis"]["geometry_reference_model"] = "policy-only"  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="system-policy"):
        validate_study_manifest(payload)


@pytest.mark.parametrize("mutation", ("missing", "legacy"))
def test_joint_power_contract_is_closed(mutation: str) -> None:
    payload = _manifest()
    power = payload["analysis"]["power"]  # type: ignore[index]
    if mutation == "missing":
        del power["dependence_source"]
    else:
        power["favorable_event"] = "low-effort-retrieval-success"
    with pytest.raises(StudyManifestError, match="schema mismatch"):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model", "beta-binomial", "development-family-cluster-resampling"),
        ("joint_success_event", "h2-only", "h2-and-h3-all-gates-pass"),
    ),
)
def test_joint_power_design_literals_are_exact(
    field: str,
    value: str,
    message: str,
) -> None:
    payload = _manifest()
    payload["analysis"]["power"][field] = value  # type: ignore[index]
    with pytest.raises(StudyManifestError, match=message):
        validate_study_manifest(payload)


@pytest.mark.parametrize("mutation", ("removed", "reordered", "appended"))
def test_joint_power_registered_endpoint_order_is_exact(mutation: str) -> None:
    payload = _manifest()
    endpoints = payload["analysis"]["power"]["registered_endpoints"]  # type: ignore[index]
    if mutation == "removed":
        endpoints.pop()
    elif mutation == "reordered":
        endpoints[0], endpoints[1] = endpoints[1], endpoints[0]
    else:
        endpoints.append("unregistered-endpoint")
    with pytest.raises(StudyManifestError, match="registered ordered joint endpoint"):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    "candidate_grid",
    (
        [1],
        [2],
        [50, 25, 75, 100, 150, 200],
        [25, 50],
        [25, 50, 75, 100],
        [25, 50, 75, 100, 150, 150],
        [25, 50, 75, 100, 150, 201],
    ),
)
def test_joint_power_candidate_grid_is_exact(candidate_grid: list[int]) -> None:
    payload = _manifest()
    payload["analysis"]["power"][  # type: ignore[index]
        "candidate_families_per_corpus"
    ] = candidate_grid
    with pytest.raises(StudyManifestError, match="registered candidate grid"):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ([], "non-empty array"),
        (["one-scenario"], "exactly two"),
        (["scenario", "scenario"], "duplicates"),
        ([""], "non-empty string"),
    ),
)
def test_joint_power_effect_scenarios_are_nonempty_unique_draftable_text(
    value: list[str],
    message: str,
) -> None:
    payload = _manifest()
    payload["analysis"]["power"]["effect_scenarios"] = value  # type: ignore[index]
    with pytest.raises(StudyManifestError, match=message):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("selection_multiplicity_method", "pointwise", "Bonferroni grid method"),
        ("selection_familywise_confidence", 0.90, "must equal 0.95"),
        ("selection_cell_alpha", 0.05, "must equal"),
        ("selection_family_size", 6, "fixed 12-cell grid"),
        ("selection_exact_qualifying_passes", 4_555, "must equal 4556"),
        ("selection_exact_blocking_failures", 446, "must equal 445"),
    ),
)
def test_joint_power_simultaneous_selection_contract_is_exact(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _manifest()
    payload["analysis"]["power"][field] = value  # type: ignore[index]
    with pytest.raises(StudyManifestError, match=message):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dependence_source", "tbd", "must be pinned"),
        ("effect_scenarios", ["tbd"], "must be pinned"),
        ("simulation_seed", "tbd", "must be pinned"),
        ("selected_families_per_corpus", "tbd", "must be pinned"),
        ("selected_families_per_corpus", 1, "at least 2"),
        ("selected_families_per_corpus", 60, "registered candidate"),
        ("selected_joint_power_lower_bound", "tbd", "must be pinned"),
        ("selected_joint_power_lower_bound", 0.89, "power_target"),
        ("simulation_count", 4_999, "at least 5000"),
    ),
)
def test_frozen_power_assumptions_and_selected_family_count_are_enforced(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _manifest(frozen=True)
    payload["analysis"]["power"][field] = value  # type: ignore[index]
    with pytest.raises(StudyManifestError, match=message):
        validate_study_manifest(payload)


@pytest.mark.parametrize("reserve_fraction", (0.2, -0.01, 0.01))
def test_one_shot_sealed_execution_has_no_reserve_rescue(
    reserve_fraction: object,
) -> None:
    payload = _manifest()
    payload["sealed_execution"]["reserve_fraction"] = reserve_fraction  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="reserve_fraction.*0.0"):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("runner_identity", "must be pinned"),
        ("code_commit", "full lowercase Git commit"),
        ("runner_image", "OCI SHA-256 digest"),
        ("production_controls", "must be pinned"),
        ("hardware", "logical_cores.*pinned"),
        ("source_revision", "source-code artifact revision"),
        ("receipt", "manifest_sha256"),
    ),
)
def test_frozen_runner_and_receipt_contract_is_fully_pinned(
    mutation: str,
    message: str,
) -> None:
    payload = _manifest(frozen=True)
    sealed = payload["sealed_execution"]  # type: ignore[assignment]
    if mutation == "runner_identity":
        sealed["runner_identity"] = "tbd"
    elif mutation == "code_commit":
        sealed["code_commit"] = "short"
    elif mutation == "runner_image":
        sealed["runner_image"] = "ghcr.io/example/study:latest"
    elif mutation == "production_controls":
        sealed["production_controls"]["blueprint_receipt_sha256"] = "tbd"
    elif mutation == "hardware":
        sealed["hardware"]["logical_cores"] = "tbd"
    elif mutation == "source_revision":
        _artifact_for(payload, "source-code")["revision"] = "v1.0.0"
    else:
        sealed["receipt_uri_template"] = "file:///tmp/receipt.json"
    with pytest.raises(StudyManifestError, match=message):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "protocol_registration_receipt_uri",
            "https://example.test/registration",
            "absolute file URI",
        ),
        (
            "protocol_registration_receipt_sha256",
            "not-a-digest",
            "lowercase SHA-256",
        ),
        ("verification_receipt_uri", "https://example.test/receipt", "absolute file URI"),
        ("verification_receipt_uri", "file:relative.json", "absolute file URI"),
        ("verification_receipt_sha256", "not-a-digest", "lowercase SHA-256"),
    ),
)
def test_sealed_run_receipt_requires_a_canonical_verification_pointer(
    field: str,
    value: str,
    message: str,
) -> None:
    arguments = {
        "manifest_sha256": "a" * 64,
        "protocol_version": "0.3.0",
        "started_at_utc": "2026-07-13T12:00:00+00:00",
        "runner_identity": _RUNNER_IDENTITY,
        "code_commit": _COMMIT,
        "runner_image": f"ghcr.io/example/study@sha256:{'2' * 64}",
        "protocol_registration_receipt_uri": "file:///tmp/registration.json",
        "protocol_registration_receipt_sha256": "c" * 64,
        "protocol_registration_record_uri": "file:///tmp/registration-record.json",
        "verification_receipt_uri": "file:///tmp/verification.json",
        "verification_receipt_sha256": "b" * 64,
        "receipt_uri": "file:///tmp/run.json",
    }
    arguments[field] = value
    with pytest.raises(StudyManifestError, match=message):
        SealedRunReceipt(**arguments)


def test_protocol_registration_receipt_is_closed_and_externally_addressed(
    tmp_path: Path,
) -> None:
    payload = _manifest(frozen=True)
    path = _registration_receipt_path(tmp_path, payload)
    receipt = load_protocol_registration_receipt(path)
    assert receipt.manifest_sha256 == manifest_sha256(payload)
    assert receipt.registry_uri.startswith("https://")
    assert len(receipt.receipt_sha256) == 64

    _registration_receipt_path(tmp_path, payload, extra_field=True)
    with pytest.raises(StudyManifestError, match="unknown"):
        load_protocol_registration_receipt(path)


def test_builtin_registry_fetch_uses_verified_https_and_one_bounded_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_uri = "https://registry.example.test/records/protocol.json"
    expected = b'{"record":"exact"}\n'
    observed: dict[str, object] = {}

    class Response:
        headers = {"Content-Length": str(len(expected))}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def getcode(self) -> int:
            return 200

        def geturl(self) -> str:
            return registry_uri

        def read(self, limit: int) -> bytes:
            observed["read_limit"] = limit
            return expected

    class Opener:
        def open(self, request: Request, *, timeout: float) -> Response:
            observed["request"] = request
            observed["timeout"] = timeout
            return Response()

    def build_opener(*handlers: object) -> Opener:
        observed["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(study_module.urllib_request, "build_opener", build_opener)
    fetched = study_module._fetch_protocol_registry_record(
        registry_uri,
        MAX_PROTOCOL_REGISTRY_RECORD_BYTES,
    )

    assert fetched == expected
    assert observed["read_limit"] == MAX_PROTOCOL_REGISTRY_RECORD_BYTES + 1
    request = observed["request"]
    assert isinstance(request, Request)
    assert request.full_url == registry_uri
    assert request.get_method() == "GET"
    assert request.get_header("Accept-encoding") == "identity"
    handlers = observed["handlers"]
    assert isinstance(handlers, tuple)
    assert any(
        isinstance(handler, study_module._NoProtocolRegistryRedirects) for handler in handlers
    )
    https_handler = next(
        handler
        for handler in handlers
        if isinstance(handler, study_module.urllib_request.HTTPSHandler)
    )
    assert https_handler._context.check_hostname is True
    assert https_handler._context.verify_mode == ssl.CERT_REQUIRED


def test_builtin_registry_fetch_refuses_redirects() -> None:
    handler = study_module._NoProtocolRegistryRedirects()
    with pytest.raises(StudyManifestError, match="refused HTTP redirect status 302"):
        handler.redirect_request(
            Request("https://registry.example.test/original"),
            None,
            302,
            "Found",
            {},
            "https://registry.example.test/substitute",
        )


@pytest.mark.parametrize("declared_oversize", (True, False))
def test_builtin_registry_fetch_rejects_oversize_headers_and_bodies(
    monkeypatch: pytest.MonkeyPatch,
    declared_oversize: bool,
) -> None:
    registry_uri = "https://registry.example.test/record.json"

    class Response:
        headers = {
            "Content-Length": str(
                MAX_PROTOCOL_REGISTRY_RECORD_BYTES + 1
                if declared_oversize
                else MAX_PROTOCOL_REGISTRY_RECORD_BYTES
            )
        }

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def getcode(self) -> int:
            return 200

        def geturl(self) -> str:
            return registry_uri

        def read(self, limit: int) -> bytes:
            assert not declared_oversize
            return b"x" * limit

    class Opener:
        def open(self, request: Request, *, timeout: float) -> Response:
            del request, timeout
            return Response()

    monkeypatch.setattr(
        study_module.urllib_request,
        "build_opener",
        lambda *handlers: Opener(),
    )
    with pytest.raises(StudyManifestError, match="maximum byte limit"):
        study_module._fetch_protocol_registry_record(
            registry_uri,
            MAX_PROTOCOL_REGISTRY_RECORD_BYTES,
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        (URLError("offline"), "verified HTTPS"),
        (TimeoutError("timed out"), "verified HTTPS"),
    ),
)
def test_builtin_registry_fetch_fails_closed_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    message: str,
) -> None:
    class Opener:
        def open(self, request: Request, *, timeout: float) -> None:
            del request, timeout
            raise failure

    monkeypatch.setattr(
        study_module.urllib_request,
        "build_opener",
        lambda *handlers: Opener(),
    )
    with pytest.raises(StudyManifestError, match=message):
        study_module._fetch_protocol_registry_record(
            "https://registry.example.test/record.json",
            MAX_PROTOCOL_REGISTRY_RECORD_BYTES,
        )


def test_arbitrary_https_registry_objects_cannot_construct_c1_admission(
    tmp_path: Path,
) -> None:
    payload = _manifest(frozen=True)
    registration_receipt_path = _registration_receipt_path(tmp_path, payload)
    registration_record_path = tmp_path / "protocol-registration-record.json"
    record = study_module.load_protocol_registry_record(registration_record_path)
    receipt = load_protocol_registration_receipt(registration_receipt_path)

    with pytest.raises(StudyManifestError, match="only come from the production verifier"):
        VerifiedC1ProtocolRegistration(
            record=record,
            receipt=receipt,
            package_root=tmp_path,
            registration_record_path=registration_record_path,
            registration_receipt_path=registration_receipt_path,
            c0_commit="0" * 40,
            c1_commit="1" * 40,
            package_file_sha256s=(("protocol-registry-record.json", record.record_sha256),),
            _fresh_revalidator=lambda: None,
            _capability=object(),
        )


def test_verified_c1_capability_rejects_package_changes(tmp_path: Path) -> None:
    payload = _manifest(frozen=True)
    registration_receipt_path = _registration_receipt_path(tmp_path, payload)
    verified = _verified_registration(tmp_path, registration_receipt_path)
    package_record = verified.package_root / "protocol-registry-record.json"
    package_record.write_bytes(package_record.read_bytes() + b" ")

    with pytest.raises(StudyManifestError, match="changed after verification"):
        verified.assert_current()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("manifest", "different manifest digest"),
        ("future", "cannot be in the future"),
        ("record", "local record or receipt changed"),
    ),
)
def test_sealed_run_requires_a_prior_external_protocol_registration(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    receipt_root = tmp_path / "registration-gated-receipts"
    receipt_root.mkdir()
    payload = _manifest(frozen=True, receipt_root=receipt_root)
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    digest = manifest_sha256(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    lock.write_text(digest + "\n", encoding="utf-8")
    artifact_receipt = _verification_receipt_path(tmp_path, payload)
    registration_receipt = _registration_receipt_path(
        tmp_path,
        payload,
        manifest_digest="f" * 64 if mutation == "manifest" else None,
        registered_at_utc=(
            "2999-01-01T00:00:00+00:00" if mutation == "future" else "2026-07-13T12:00:00+00:00"
        ),
    )
    registration_record = tmp_path / "protocol-registration-record.json"
    verified_registration = _verified_registration(tmp_path, registration_receipt)
    if mutation == "record":
        changed_record = ProtocolRegistryRecord(
            manifest_sha256=digest,
            protocol_version="0.3.0",
            registered_at_utc="2026-07-13T12:00:00+00:00",
            registry_identity="osf-registration:substituted",
            registry_uri="https://osf.io/registries/test-registration",
        )
        registration_record.write_bytes(changed_record.canonical_bytes() + b"\n")

    with pytest.raises(StudyManifestError, match=message):
        begin_sealed_run(
            manifest,
            lock,
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=artifact_receipt,
            artifact_root=tmp_path,
            local_artifact_map_path=tmp_path / "unused-artifact-map.json",
            verified_protocol_registration=verified_registration,
        )
    assert not tuple(receipt_root.iterdir())


def test_sealed_run_uses_digest_derived_manifest_receipt_and_pinned_identity(
    tmp_path: Path,
) -> None:
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    payload = _manifest(frozen=True, receipt_root=receipt_root)
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    digest = manifest_sha256(payload)
    receipt_path = receipt_root / f"{digest}.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    lock.write_text(digest + "\n", encoding="utf-8")
    artifact_receipt = _verification_receipt_path(tmp_path, payload)
    registration_receipt = _registration_receipt_path(tmp_path, payload)
    revalidation_calls: list[str] = []

    def revalidate() -> None:
        revalidation_calls.append("fresh")

    verified_registration = _verified_registration(
        tmp_path,
        registration_receipt,
        fresh_revalidator=revalidate,
    )
    artifact_root = tmp_path / "test-artifact-root"
    artifact_root.mkdir()
    artifact_map = tmp_path / "test-artifact-map.json"
    artifact_map.write_text("{}\n", encoding="utf-8")
    admitted_receipt = load_verification_receipt(artifact_receipt)

    assert sealed_receipt_uri(payload) == receipt_path.resolve().as_uri()
    with pytest.raises(StudyManifestError, match="does not equal"):
        begin_sealed_run(
            manifest,
            lock,
            runner_identity="different-runner",
            artifact_verification_receipt_path=artifact_receipt,
            artifact_root=artifact_root,
            local_artifact_map_path=artifact_map,
            verified_protocol_registration=verified_registration,
        )
    assert not receipt_path.exists()

    with (
        patch.object(study_module, "load_local_artifact_map", return_value=()),
        patch.object(
            study_module,
            "verify_local_artifacts",
            return_value=admitted_receipt,
        ),
    ):
        observed = begin_sealed_run(
            manifest,
            lock,
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=artifact_receipt,
            artifact_root=artifact_root,
            local_artifact_map_path=artifact_map,
            verified_protocol_registration=verified_registration,
        )
    assert revalidation_calls == ["fresh"]
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert observed.manifest_sha256 == digest
    assert observed.receipt_uri == receipt_path.resolve().as_uri()
    assert observed.verification_receipt_uri == artifact_receipt.as_uri()
    assert (
        observed.verification_receipt_sha256
        == load_verification_receipt(artifact_receipt).receipt_sha256
    )
    assert stored["runner_identity"] == _RUNNER_IDENTITY
    assert stored["code_commit"] == _COMMIT
    assert stored["verification_receipt_uri"] == artifact_receipt.as_uri()
    assert stored["protocol_registration_receipt_uri"] == registration_receipt.as_uri()
    assert (
        stored["protocol_registration_record_uri"]
        == (tmp_path / "protocol-registration-record.json").as_uri()
    )
    assert (
        stored["protocol_registration_receipt_sha256"]
        == load_protocol_registration_receipt(registration_receipt).receipt_sha256
    )
    assert stored["verification_receipt_sha256"] == observed.verification_receipt_sha256
    assert load_sealed_run_receipt(receipt_path.resolve()) == observed
    relocated = (tmp_path / "relocated-run.json").resolve()
    relocated.write_bytes(receipt_path.read_bytes())
    with pytest.raises(StudyManifestError, match="manifest-derived receipt_uri"):
        load_sealed_run_receipt(relocated)
    with pytest.raises(
        StudyManifestError,
        match="one-shot execution has already been consumed",
    ) as error:
        with (
            patch.object(study_module, "load_local_artifact_map", return_value=()),
            patch.object(
                study_module,
                "verify_local_artifacts",
                return_value=admitted_receipt,
            ),
        ):
            begin_sealed_run(
                manifest,
                lock,
                runner_identity=_RUNNER_IDENTITY,
                artifact_verification_receipt_path=artifact_receipt,
                artifact_root=artifact_root,
                local_artifact_map_path=artifact_map,
                verified_protocol_registration=verified_registration,
            )
    assert revalidation_calls == ["fresh", "fresh"]
    assert "reserve_fraction is 0.0" in str(error.value)
    assert "no rerun or rescue is permitted" in str(error.value)
    assert "use the registered reserve set" not in str(error.value)


def test_production_run_requires_verified_fixed_c1_registration(
    tmp_path: Path,
) -> None:
    with pytest.raises(StudyManifestError, match="verified fixed C1 registration"):
        begin_sealed_run(
            tmp_path / "missing-manifest.json",
            tmp_path / "missing-lock.txt",
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=tmp_path / "missing-receipt.json",
            artifact_root=tmp_path,
            local_artifact_map_path=tmp_path / "missing-map.json",
            verified_protocol_registration=object(),  # type: ignore[arg-type]
        )


def test_sealed_run_reopens_every_local_artifact_before_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_root = tmp_path / "fresh-verification-receipts"
    receipt_root.mkdir()
    payload = _manifest(frozen=True, receipt_root=receipt_root)
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    digest = manifest_sha256(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    lock.write_text(digest + "\n", encoding="utf-8")
    artifact_receipt_path = _verification_receipt_path(tmp_path, payload)
    admitted_receipt = load_verification_receipt(artifact_receipt_path)
    registration_receipt = _registration_receipt_path(tmp_path, payload)
    verified_registration = _verified_registration(tmp_path, registration_receipt)
    artifact_root = tmp_path / "artifact-root"
    artifact_root.mkdir()
    artifact_map = tmp_path / "artifact-map.json"
    artifact_map.write_text("{}", encoding="utf-8")
    expected_pins = {
        str(artifact["id"]): str(artifact["sha256"]) for artifact in payload["artifacts"]
    }
    specs = (object(),)
    calls: list[tuple[object, ...]] = []

    def load_map(path: Path, *, expected_sha256_by_id: object) -> object:
        calls.append(("map", path, expected_sha256_by_id))
        return specs

    def verify(
        root: Path,
        *,
        manifest_sha256: str,
        artifacts: object,
    ) -> ArtifactVerificationReceipt:
        calls.append(("verify", root, manifest_sha256, artifacts))
        return admitted_receipt

    monkeypatch.setattr(study_module, "load_local_artifact_map", load_map)
    monkeypatch.setattr(study_module, "verify_local_artifacts", verify)
    observed = begin_sealed_run(
        manifest,
        lock,
        runner_identity=_RUNNER_IDENTITY,
        artifact_verification_receipt_path=artifact_receipt_path,
        artifact_root=artifact_root,
        local_artifact_map_path=artifact_map,
        verified_protocol_registration=verified_registration,
    )
    assert observed.verification_receipt_sha256 == admitted_receipt.receipt_sha256
    assert calls == [
        ("map", artifact_map, expected_pins),
        ("verify", artifact_root, digest, specs),
    ]

    mismatched = ArtifactVerificationReceipt(
        manifest_sha256=digest,
        artifacts=admitted_receipt.artifacts[:-1],
    )
    monkeypatch.setattr(
        study_module,
        "verify_local_artifacts",
        lambda *args, **kwargs: mismatched,
    )
    with pytest.raises(StudyManifestError, match="differs from the admitted receipt"):
        begin_sealed_run(
            manifest,
            lock,
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=artifact_receipt_path,
            artifact_root=artifact_root,
            local_artifact_map_path=artifact_map,
            verified_protocol_registration=verified_registration,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("manifest", "different manifest digest"),
        ("missing", "cover every manifest artifact exactly"),
        ("extra", "cover every manifest artifact exactly"),
        ("digest", "digest mismatch"),
        ("inexact", "must be exact"),
    ),
)
def test_sealed_run_rejects_unbound_or_incomplete_artifact_receipts(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    receipt_root = tmp_path / "run-receipts"
    receipt_root.mkdir()
    payload = _manifest(frozen=True, receipt_root=receipt_root)
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    digest = manifest_sha256(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    lock.write_text(digest + "\n", encoding="utf-8")
    first_id = str(payload["artifacts"][0]["id"])  # type: ignore[index]
    options: dict[str, object] = {"name": f"{mutation}.json"}
    if mutation == "manifest":
        options["manifest_digest"] = "f" * 64
    elif mutation == "missing":
        options["omit_id"] = first_id
    elif mutation == "extra":
        options["add_unexpected"] = True
    elif mutation == "digest":
        options["digest_override"] = (first_id, "d" * 64)
    else:
        options["exact_override"] = (first_id, False)
    artifact_receipt = _verification_receipt_path(tmp_path, payload, **options)
    registration_receipt = _registration_receipt_path(tmp_path, payload)

    with pytest.raises(StudyManifestError, match=message):
        _begin_with_test_registration(
            tmp_path,
            manifest,
            lock,
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=artifact_receipt,
            registration_receipt_path=registration_receipt,
        )
    assert not (receipt_root / f"{digest}.json").exists()


def test_sealed_run_refuses_a_symlinked_receipt_parent(tmp_path: Path) -> None:
    real_receipts = tmp_path / "real-receipts"
    real_receipts.mkdir()
    linked_receipts = tmp_path / "linked-receipts"
    linked_receipts.symlink_to(real_receipts, target_is_directory=True)
    payload = _manifest(frozen=True, receipt_root=linked_receipts)
    payload["sealed_execution"]["receipt_uri_template"] = (  # type: ignore[index]
        linked_receipts.as_uri() + "/{manifest_sha256}.json"
    )
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    digest = manifest_sha256(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    lock.write_text(digest + "\n", encoding="utf-8")
    artifact_receipt = _verification_receipt_path(tmp_path, payload)
    registration_receipt = _registration_receipt_path(tmp_path, payload)

    with pytest.raises(StudyManifestError, match="cannot write sealed run receipt safely"):
        _begin_with_test_registration(
            tmp_path,
            manifest,
            lock,
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=artifact_receipt,
            registration_receipt_path=registration_receipt,
        )
    assert not tuple(real_receipts.iterdir())
