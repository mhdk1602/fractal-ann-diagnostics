from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest

import fractal_ann_diagnostics.production_corpus_run as production_entry
from fractal_ann_diagnostics.artifact_integrity import (
    ArtifactVerificationReceipt,
    VerifiedArtifact,
    digest_directory_tree,
    digest_regular_file,
    write_exclusive_receipt_bytes,
)
from fractal_ann_diagnostics.cli import main as cli_main
from fractal_ann_diagnostics.custody import OnlineCustodyAdmissionReceipt
from fractal_ann_diagnostics.embedding_store import (
    EmbeddingStoreConfig,
    LocalModelSpec,
    StagedEmbeddingSources,
    build_embedding_store,
)
from fractal_ann_diagnostics.execution_claim import ExecutionClaimError, RuntimeClaimReceipt
from fractal_ann_diagnostics.policy_intervention import (
    PolicyInterventionConfig,
    compile_policy_intervention,
)
from fractal_ann_diagnostics.production_corpus_run import (
    ONLINE_CUSTODY_ADMISSION_FILENAME,
    PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME,
    PRODUCTION_CORPUS_CONFIG_FILENAME,
    PRODUCTION_CORPUS_WORKLOAD_ID,
    PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME,
    REQUIRED_ARTIFACT_BINDINGS_FILENAME,
    RUNTIME_ATTESTATION_PLAN_FILENAME,
    RUNTIME_ATTESTATION_RECEIPT_FILENAME,
    RUNTIME_INVOCATION_MARKER_FILENAME,
    SEALED_RUN_RECEIPT_FILENAME,
    SHARDED_EXECUTION_PLAN_FILENAME,
    TRIAL_RUNTIME_RECEIPT_FILENAME,
    ProductionCorpusRunConfig,
    ProductionCorpusRunError,
    ProductionCorpusWorkloadSpec,
    load_admitted_production_corpus_controls,
    load_production_corpus_command_attempt,
)
from fractal_ann_diagnostics.query_cohort import (
    NESTED_ROWS_PER_FAMILY,
    nested_trial_source_value,
    representative_selection_rank,
)
from fractal_ann_diagnostics.runtime_attestation import (
    RuntimeArtifactMount,
    RuntimeAttestationError,
    RuntimeAttestationPlan,
    RuntimeAttestationReceipt,
    RuntimeFilePin,
    argv_sha256,
    environment_sha256,
)
from fractal_ann_diagnostics.scalable_custody import (
    QUERY_KEY_MAP_PATH,
    ScalableCustodyPlan,
    build_scalable_custody_package,
    verify_query_trial_key_parity,
)
from fractal_ann_diagnostics.scalable_execution import (
    ImmutableArtifactPin,
    IndexArtifactDescriptor,
    ProvenanceSidecarDescriptor,
    ShardedOnlineExecutionPlan,
    VectorStoreDescriptor,
    write_sharded_online_execution_plan,
)
from fractal_ann_diagnostics.scalable_partition_audit import (
    build_scalable_partition_audit,
    load_scalable_partition_audit,
)
from fractal_ann_diagnostics.sealed_orchestrator import RequiredArtifactIdBindings
from fractal_ann_diagnostics.study import SealedRunReceipt
from fractal_ann_diagnostics.trial_runtime import (
    QUERY_TRIAL_FILENAME,
    CanonicalQueryTrialRow,
    RuntimeFeatureBinding,
    TrialRuntimeAdmission,
    TrialRuntimeError,
    admit_trial_runtime,
    build_query_trial_store,
    load_trial_runtime,
    load_trial_runtime_admission,
    load_trial_runtime_block,
    load_trial_runtime_receipt,
    verify_query_trial_store,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


_SELECTION_SEED_SHA256 = _sha(b"registered query-family cohort")
_REAL_WORKLOAD_SOURCE_VERIFIER = production_entry._verify_workload_spec_sources


@pytest.fixture(autouse=True)
def _isolate_synthetic_production_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep command-boundary tests focused on ordering, not factory construction."""

    monkeypatch.setattr(production_entry, "_verify_workload_spec_sources", lambda _spec: None)


def _opaque_v2(
    secret: bytes,
    *,
    domain: str,
    source_value: str,
) -> str:
    digest = hmac.new(secret, digestmod=hashlib.sha256)
    for value in (
        "fractal-label-separation-v2",
        domain,
        "sealed-query-hmac-2026-07",
        "demo",
        "sealed",
        source_value,
    ):
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(_canonical(row) + b"\n" for row in rows)
    path.write_bytes(encoded)
    return encoded


def _artifact(
    path: str,
    encoded: bytes,
    *,
    dataset: str | None,
    role: str,
    stage: str | None,
    visibility: str = "online",
) -> dict[str, object]:
    return {
        "byte_count": len(encoded),
        "dataset": dataset,
        "path": path,
        "record_count": encoded.count(b"\n"),
        "role": role,
        "sha256": _sha(encoded),
        "stage": stage,
        "visibility": visibility,
    }


class _Encoder:
    implementation_id = "trial-runtime-test-encoder-v1"

    def encode(
        self,
        texts: Sequence[str],
        *,
        model_path: Path,
        prompt: str,
        max_sequence_length: int,
        output_dimension: int,
        normalize: bool,
        device: str,
        seed: int,
    ) -> np.ndarray:
        del max_sequence_length, normalize, device
        model = (model_path / "config.json").read_bytes()
        rows = []
        for text in texts:
            digest = hashlib.sha256(
                model + prompt.encode() + seed.to_bytes(8, "big") + text.encode()
            ).digest()
            vector = np.frombuffer(digest, dtype=np.uint8)[:output_dimension].astype(np.float32)
            vector += 1.0
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.stack(rows)


@dataclass(frozen=True)
class _Fixture:
    staged_root: Path
    partition_audit_path: Path
    embedding_root: Path
    package_root: Path
    schedule_path: Path
    plan: ShardedOnlineExecutionPlan
    feature_bindings: tuple[RuntimeFeatureBinding, ...]
    secret: bytes


def _model(tmp_path: Path, name: str, payload: bytes) -> LocalModelSpec:
    root = (tmp_path / name).resolve()
    root.mkdir()
    (root / "config.json").write_bytes(payload)
    return LocalModelSpec(
        path=root,
        revision=hashlib.sha1(payload).hexdigest(),
        tree_sha256=digest_directory_tree(root).sha256,
    )


def _pin(artifact_id: str, relative_path: str, payload: bytes) -> ImmutableArtifactPin:
    return ImmutableArtifactPin(
        artifact_id=artifact_id,
        relative_path=relative_path,
        kind="file",
        byte_count=len(payload),
        sha256=_sha(payload),
    )


def _stage(tmp_path: Path) -> tuple[Path, StagedEmbeddingSources]:
    root = (tmp_path / "staged").resolve()
    root.mkdir()
    document_path = "datasets/demo/corpus/part-00000.jsonl"
    fit_path = "datasets/demo/fit/queries.jsonl"
    calibration_path = "datasets/demo/calibration/queries.jsonl"
    sealed_path = "datasets/demo/sealed/online/queries.jsonl"
    fit_qrels_path = "datasets/demo/fit/qrels.jsonl"
    calibration_qrels_path = "datasets/demo/calibration/qrels.jsonl"
    sealed_qrels_path = "datasets/demo/sealed/custody/qrels.jsonl"
    assignment_path = "assignments.jsonl"
    exclusion_path = "partition-exclusions.jsonl"
    documents = _write_jsonl(
        root / document_path,
        [
            {"id": "doc-0", "text": "alpha", "title": "Alpha"},
            {"id": "doc-1", "text": "beta", "title": "Beta"},
            {"id": "doc-2", "text": "gamma", "title": "Gamma"},
            {"id": "doc-3", "text": "delta", "title": "Delta"},
            {
                "id": "doc-calibration",
                "text": "calibration evidence",
                "title": "Calibration",
            },
            {"id": "doc-fit", "text": "fit evidence", "title": "Fit"},
        ],
    )
    fit = _write_jsonl(root / fit_path, [{"id": "fit-0", "text": "fit question"}])
    calibration = _write_jsonl(
        root / calibration_path,
        [{"id": "calibration-0", "text": "calibration question"}],
    )
    sealed_rows = [
        {"id": "sealed-0", "text": "find alpha"},
        {"id": "sealed-1", "text": "find delta"},
        {"id": "sealed-2", "text": "find gamma"},
    ]
    sealed = _write_jsonl(root / sealed_path, sealed_rows)
    fit_qrels = _write_jsonl(
        root / fit_qrels_path,
        [{"document_id": "doc-fit", "query_id": "fit-0", "relevance": 1}],
    )
    calibration_qrels = _write_jsonl(
        root / calibration_qrels_path,
        [
            {
                "document_id": "doc-calibration",
                "query_id": "calibration-0",
                "relevance": 1,
            }
        ],
    )
    sealed_qrels = _write_jsonl(
        root / sealed_qrels_path,
        [
            {"document_id": "doc-0", "query_id": "sealed-0", "relevance": 1},
            {"document_id": "doc-3", "query_id": "sealed-1", "relevance": 1},
            {"document_id": "doc-2", "query_id": "sealed-2", "relevance": 1},
        ],
    )
    exclusions = _write_jsonl(root / exclusion_path, [])
    assignment_rows = []
    staged_queries = [
        ("fit", {"id": "fit-0", "text": "fit question"}),
        (
            "calibration",
            {"id": "calibration-0", "text": "calibration question"},
        ),
        *(("sealed", row) for row in sealed_rows),
    ]
    component_by_stage = {
        "fit": _sha(_canonical(["fit-0"])),
        "calibration": _sha(_canonical(["calibration-0"])),
        "sealed": _sha(_canonical(sorted((str(row["id"]) for row in sealed_rows)))),
    }
    for position, (stage, row) in enumerate(staged_queries):
        assignment_rows.append(
            {
                "assignment_key_sha256": _sha(f"assignment-{position}".encode()),
                "dataset": "demo",
                "domain": None,
                "partition_component_sha256": component_by_stage[stage],
                "query_id": row["id"],
                "query_text_sha256": _sha(str(row["text"]).encode()),
                "schema_version": "fractal-study-query-assignment-v1",
                "source_split": "fixture",
                "stage": stage,
            }
        )
    assignments = _write_jsonl(root / assignment_path, assignment_rows)
    artifacts = [
        _artifact(document_path, documents, dataset="demo", role="corpus-shard", stage=None),
        _artifact(fit_path, fit, dataset="demo", role="queries", stage="fit"),
        _artifact(
            calibration_path,
            calibration,
            dataset="demo",
            role="queries",
            stage="calibration",
        ),
        _artifact(sealed_path, sealed, dataset="demo", role="queries", stage="sealed"),
        _artifact(
            fit_qrels_path,
            fit_qrels,
            dataset="demo",
            role="qrels",
            stage="fit",
        ),
        _artifact(
            calibration_qrels_path,
            calibration_qrels,
            dataset="demo",
            role="qrels",
            stage="calibration",
        ),
        _artifact(
            sealed_qrels_path,
            sealed_qrels,
            dataset="demo",
            role="qrels",
            stage="sealed",
            visibility="custody",
        ),
        _artifact(assignment_path, assignments, dataset=None, role="assignments", stage=None),
        _artifact(
            exclusion_path,
            exclusions,
            dataset=None,
            role="query-partition-structural-exclusions",
            stage=None,
            visibility="protocol",
        ),
    ]
    artifacts.sort(key=lambda row: str(row["path"]).encode())
    inventory = (
        _canonical(
            {
                "artifacts": artifacts,
                "assignment_algorithm": {
                    "component_edges": [
                        "normalized-query-text-equality",
                        "registered-near-duplicate-token-rule",
                        "shared-positive-document-content",
                        "shared-positive-relevance-document",
                    ],
                    "cross_source_split_policy": "exclude-entire-component-v1",
                    "fit_calibration_component_ratio": "4:1",
                    "name": "component-ranked-sha256-v2",
                    "three_way_component_ratio": "3:1:1",
                },
                "assignment_seed_sha256": "1" * 64,
                "bright_document_identity": {},
                "bright_domains": [],
                "config_sha256": "2" * 64,
                "counts": {
                    "demo": {
                        "calibration_queries": 1,
                        "documents": 6,
                        "fit_queries": 1,
                        "qrels": 5,
                        "sealed_queries": 3,
                    }
                },
                "hotpotqa_fullwiki_scope": {},
                "schema_version": "fractal-study-data-inventory-v2",
                "withhold_sealed_labels_from_online_process": True,
                "sources": [
                    {
                        "byte_count": 1,
                        "revision": "0123456789abcdef0123456789abcdef01234567",
                        "sha256": "3" * 64,
                        "source_id": "fixture-source",
                    }
                ],
            }
        )
        + b"\n"
    )
    (root / "inventory.json").write_bytes(inventory)
    (root / "inventory.sha256").write_text(
        f"{_sha(inventory)}  inventory.json\n",
        encoding="ascii",
    )
    return root, StagedEmbeddingSources(
        root=root,
        inventory_sha256=_sha(inventory),
        document_paths=(document_path,),
        query_paths=tuple(sorted((fit_path, calibration_path, sealed_path))),
    )


def _plan(receipt: Any) -> ShardedOnlineExecutionPlan:
    document_count = 64
    dimension = 3
    universe = _sha(b"document universe")
    inventory_pin = _pin("demo-shards", "corpus/shards.json", b"shards\n")
    active_payload = bytes(value % 256 for value in range(document_count * dimension * 4))
    truth_payload = bytes((value + 97) % 256 for value in range(document_count * dimension * 4))
    active_pin = _pin("demo-active-docs", "vectors/active.raw", active_payload)
    truth_pin = _pin("demo-truth-docs", "vectors/truth.raw", truth_payload)
    sidecar_pin = _pin(
        "demo-provenance",
        "provenance/sha256.bin",
        bytes(32 * document_count),
    )
    index_pin = _pin("demo-hnsw", "indexes/active.bin", b"index")
    return ShardedOnlineExecutionPlan(
        key_id=receipt.hmac_key_id,
        corpus="demo",
        stage="sealed",
        document_count=document_count,
        ordered_document_universe_sha256=universe,
        permutation_seed=20260714,
        trials=receipt.opaque_trials,
        query_partition_audit_sha256=receipt.query_partition_audit_sha256,
        corpus_shard_inventory=inventory_pin,
        query_trial_store=receipt.store_descriptor(
            artifact_id="demo-query-trials",
            relative_path="query-package/query-trials.jsonl",
            receipt_artifact_id="demo-query-trial-receipt",
            receipt_relative_path="query-package/query-trial-receipt.json",
        ),
        active_vector_store=VectorStoreDescriptor(
            artifact=active_pin,
            role="active-migration",
            dtype="<f4",
            shape=(document_count, dimension),
            document_universe_sha256=universe,
        ),
        current_truth_vector_store=VectorStoreDescriptor(
            artifact=truth_pin,
            role="current-exact-truth",
            dtype="<f4",
            shape=(document_count, dimension),
            document_universe_sha256=universe,
        ),
        provenance_sha256_sidecar=ProvenanceSidecarDescriptor(
            artifact=sidecar_pin,
            record_count=document_count,
            document_universe_sha256=universe,
        ),
        hnsw_index=IndexArtifactDescriptor(
            artifact=index_pin,
            document_count=document_count,
            document_universe_sha256=universe,
            source_vector_sha256=active_pin.sha256,
            format_revision="hnswlib-0.8.0-cosine-v1",
        ),
    )


@pytest.fixture
def runtime_fixture(tmp_path: Path) -> _Fixture:
    staged_root, sources = _stage(tmp_path)
    partition_audit_path = (tmp_path / "query-partition-audit.json").resolve()
    build_scalable_partition_audit(staged_root, partition_audit_path)
    old_model = _model(tmp_path, "old-model", b"old")
    current_model = _model(tmp_path, "current-model", b"current")
    embedding_root = (tmp_path / "embedding-store").resolve()
    build_embedding_store(
        sources,
        embedding_root,
        current_model=current_model,
        current_encoder=_Encoder(),
        old_model=old_model,
        old_encoder=_Encoder(),
        config=EmbeddingStoreConfig(
            query_prompt="Query: ",
            document_prompt="",
            max_sequence_length=64,
            output_dimension=3,
            normalize=True,
            batch_size=2,
            output_dtype="float32",
            device="cpu",
            deterministic_seed=20260714,
        ),
    )
    secret = bytes(range(32))
    package_root = (tmp_path / "query-package").resolve()
    receipt = build_query_trial_store(
        staged_root,
        embedding_root,
        package_root,
        partition_audit_path=partition_audit_path,
        corpus="demo",
        stage="sealed",
        hmac_key_id="sealed-query-hmac-2026-07",
        hmac_secret=secret,
        selection_seed_sha256=_SELECTION_SEED_SHA256,
        available_family_count=1,
        selected_family_count=1,
    )
    plan = _plan(receipt)
    compiled = compile_policy_intervention(
        plan,
        PolicyInterventionConfig(
            seed_sha256=_sha(b"policy seed"),
            baseline_seed_sha256=_sha(b"baseline policy seed"),
            policy_bundle_revision=f"sha256:{_sha(b'policy bundle')}",
            baseline_policy_revision=f"sha256:{_sha(b'baseline policy bundle')}",
            subject_ids=("analyst",),
            assignment_repetitions=1,
        ),
    )
    schedule_path = (tmp_path / "trial-schedule.json").resolve()
    write_exclusive_receipt_bytes(compiled.schedule.canonical_file_bytes(), schedule_path)
    bindings: list[RuntimeFeatureBinding] = []
    seen: set[tuple[int, str, int, str]] = set()
    for row in compiled.schedule.rows:
        key = (row.group_order, row.subject, row.repetition, row.policy_state)
        if key in seen:
            continue
        seen.add(key)
        bindings.append(
            RuntimeFeatureBinding(
                group_order=row.group_order,
                subject=row.subject,
                repetition=row.repetition,
                policy_state=row.policy_state,
                version_lag=1.0,
                backend="hnsw",
                drift_family="model-revision-lag",
                policy_complexity=row.realized_allow_rate,
            )
        )
    return _Fixture(
        staged_root=staged_root,
        partition_audit_path=partition_audit_path,
        embedding_root=embedding_root,
        package_root=package_root,
        schedule_path=schedule_path,
        plan=plan,
        feature_bindings=tuple(bindings),
        secret=secret,
    )


def _admit(fixture: _Fixture, tmp_path: Path) -> TrialRuntimeAdmission:
    return admit_trial_runtime(
        fixture.plan,
        fixture.package_root,
        fixture.staged_root,
        fixture.embedding_root,
        fixture.schedule_path,
        fixture.feature_bindings,
        partition_audit_path=fixture.partition_audit_path,
        receipt_target=(tmp_path / "runtime-receipt.json").resolve(),
    )


def _verified_artifact(artifact_id: str, digest: str, position: int) -> VerifiedArtifact:
    return VerifiedArtifact(
        artifact_id=artifact_id,
        relative_path=f"objects/{position:02d}-{artifact_id}.bin",
        kind="file",
        exact=True,
        expected_sha256=digest,
        verified_sha256=digest,
        file_count=1,
        directory_count=0,
        byte_count=1,
        observed_file_count=1,
        observed_directory_count=0,
        observed_byte_count=1,
    )


@dataclass(frozen=True)
class _ProductionCommandPackage:
    config: ProductionCorpusRunConfig
    config_path: Path
    config_sha256: str
    output_root: Path
    attestation_plan: RuntimeAttestationPlan
    attestation_receipt: RuntimeAttestationReceipt
    marker_payload: bytes
    runtime_claim: RuntimeClaimReceipt


def _production_command_package(
    fixture: _Fixture,
    tmp_path: Path,
) -> _ProductionCommandPackage:
    runtime = _admit(fixture, tmp_path)
    control_root = (tmp_path / "production-controls").resolve()
    output_root = (tmp_path / "production-output").resolve()
    control_root.mkdir(mode=0o700)
    output_root.mkdir(mode=0o700)

    artifact_root = (tmp_path / "provenance-artifacts").resolve()
    index_root = (tmp_path / "authorized-indexes").resolve()
    policy_root = (tmp_path / "policy-intervention").resolve()
    for root in (artifact_root, index_root, policy_root):
        root.mkdir(mode=0o700)
    (policy_root / "trial-schedule.json").write_bytes(fixture.schedule_path.read_bytes())
    pseudonym_key = (tmp_path / "pseudonym.key").resolve()
    pseudonym_payload = b"production-test-pseudonym-key-material"
    pseudonym_key.write_bytes(pseudonym_payload)

    component_names = ("application", "controller", "corpus", "embedding", "index", "policy")
    component_ids = {name: f"{name}-artifact" for name in component_names}
    artifact_digests = {
        "execution-artifact": fixture.plan.artifact_sha256,
        "runner-artifact": _sha(b"runner"),
        "source-artifact": _sha(b"source"),
        **{artifact_id: _sha(name.encode()) for name, artifact_id in component_ids.items()},
    }
    verification = ArtifactVerificationReceipt(
        manifest_sha256="a" * 64,
        artifacts=tuple(
            _verified_artifact(artifact_id, digest, position)
            for position, (artifact_id, digest) in enumerate(artifact_digests.items())
        ),
    )
    provenance_bindings = tuple(sorted(component_ids.items(), key=lambda item: item[0].encode()))
    required = RequiredArtifactIdBindings(
        verification_receipt=verification,
        execution_artifact_id="execution-artifact",
        execution_revision_sha256=fixture.plan.artifact_sha256,
        runner_artifact_ids=("runner-artifact",),
        source_artifact_ids=("source-artifact",),
        retriever_artifact_ids=tuple(
            sorted(component_ids.values(), key=lambda value: value.encode())
        ),
        provenance_component_artifact_ids=provenance_bindings,
    )
    run_path = (tmp_path / SEALED_RUN_RECEIPT_FILENAME).resolve()
    run = SealedRunReceipt(
        manifest_sha256=verification.manifest_sha256,
        protocol_version="0.3.0",
        started_at_utc="2026-07-14T12:00:00+00:00",
        runner_identity="runner-65532",
        code_commit="b" * 40,
        runner_image=("ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@sha256:" + "c" * 64),
        protocol_registration_receipt_uri="file:///frozen/protocol-receipt.json",
        protocol_registration_receipt_sha256="d" * 64,
        protocol_registration_record_uri="file:///frozen/protocol-record.json",
        verification_receipt_uri="file:///frozen/artifact-verification.json",
        verification_receipt_sha256=verification.receipt_sha256,
        receipt_uri=run_path.as_uri(),
    )
    derived_seed_sha256 = _sha(b"fixture runtime claim derived seed")
    runtime_claim = RuntimeClaimReceipt(
        manifest_sha256=run.manifest_sha256,
        run_receipt_sha256=run.binding_sha256,
        c1_commit="1" * 40,
        claim_contract_sha256=_sha(b"fixture claim contract"),
        claim_state_sha256=_sha(b"fixture claim state"),
        claim_ledger_commit="2" * 40,
        provider_identity_sha256=_sha(b"fixture provider identity"),
        live_execute_job_receipt_sha256=_sha(b"fixture live execute job"),
        execute_job_id=1729,
        beacon_receipt_sha256=_sha(b"fixture beacon receipt"),
        beacon_bytes_sha256=_sha(b"fixture beacon bytes"),
        design_seed_sha256=_sha(b"fixture design seed"),
        derived_seed_sha256=derived_seed_sha256,
        permutation_seed=int.from_bytes(bytes.fromhex(derived_seed_sha256)[:8], "big"),
        output_aggregate_identity=_sha(b"fixture output aggregate"),
    )
    admission = OnlineCustodyAdmissionReceipt(
        manifest_sha256=verification.manifest_sha256,
        run_receipt_sha256=run.binding_sha256,
        artifact_verification_receipt_sha256=verification.receipt_sha256,
        custody_seal_receipt_sha256="e" * 64,
        online_artifact_verification_receipt_sha256="f" * 64,
        runner_identity=run.runner_identity,
        verified_artifact_ids=tuple(sorted(artifact_digests, key=lambda value: value.encode())),
    )

    controls = {
        ONLINE_CUSTODY_ADMISSION_FILENAME: admission.canonical_bytes() + b"\n",
        REQUIRED_ARTIFACT_BINDINGS_FILENAME: required.canonical_file_bytes(),
        SHARDED_EXECUTION_PLAN_FILENAME: fixture.plan.canonical_file_bytes(),
        TRIAL_RUNTIME_RECEIPT_FILENAME: runtime.receipt.canonical_file_bytes(),
    }
    run_path.write_bytes(run.canonical_bytes() + b"\n")
    for filename, encoded in controls.items():
        (control_root / filename).write_bytes(encoded)

    factory_artifact_root = (tmp_path / "factory-artifacts").resolve()
    factory_artifact_root.mkdir(mode=0o700)
    policy_bundle_path = (tmp_path / "policy-stage-bundle.json").resolve()
    index_bundle_path = (tmp_path / "index-stage-bundle.json").resolve()
    policy_bundle_path.write_bytes(b'{"fixture":"policy-bundle"}\n')
    index_bundle_path.write_bytes(b'{"fixture":"index-bundle"}\n')
    partition_audit = load_scalable_partition_audit(fixture.partition_audit_path)
    workload_spec = ProductionCorpusWorkloadSpec(
        corpus_id="scifact",
        available_family_count=1,
        selected_family_count=1,
        factory_config_sha256="8" * 64,
        factory_suite_receipt_sha256="9" * 64,
        factory_artifact_tree_sha256=digest_directory_tree(factory_artifact_root).sha256,
        runner_image=run.runner_image,
        runner_platform="linux/arm64",
        runner_identity=run.runner_identity,
        code_commit=run.code_commit,
        artifact_root=artifact_root,
        artifact_tree_sha256=digest_directory_tree(artifact_root).sha256,
        authorized_index_store_root=index_root,
        authorized_index_store_tree_sha256=digest_directory_tree(index_root).sha256,
        embedding_store_root=fixture.embedding_root,
        embedding_store_tree_sha256=digest_directory_tree(fixture.embedding_root).sha256,
        partition_audit_path=fixture.partition_audit_path,
        partition_audit_file_sha256=digest_regular_file(
            fixture.partition_audit_path,
            label="fixture partition audit",
        ),
        partition_audit_sha256=partition_audit.artifact_sha256,
        policy_intervention_root=policy_root,
        policy_intervention_tree_sha256=digest_directory_tree(policy_root).sha256,
        pseudonym_key_path=pseudonym_key,
        expected_pseudonym_key_sha256=_sha(pseudonym_payload),
        query_package_root=fixture.package_root,
        query_package_tree_sha256=digest_directory_tree(fixture.package_root).sha256,
        staged_root=fixture.staged_root,
        staged_tree_sha256=digest_directory_tree(fixture.staged_root).sha256,
        expected_authorized_index_store_receipt_sha256="1" * 64,
        expected_policy_intervention_receipt_sha256="2" * 64,
        policy_bundle_receipt_sha256=digest_regular_file(
            policy_bundle_path,
            label="fixture policy bundle",
        ),
        index_bundle_receipt_sha256=digest_regular_file(
            index_bundle_path,
            label="fixture index bundle",
        ),
        policy_bundle_receipt_path=policy_bundle_path,
        index_bundle_receipt_path=index_bundle_path,
        query_receipt_sha256=digest_regular_file(
            fixture.package_root / "query-trial-receipt.json",
            label="fixture query receipt",
        ),
        online_execution_plan_sha256=fixture.plan.artifact_sha256,
        online_execution_tree_sha256=digest_directory_tree(artifact_root).sha256,
        sharded_execution_plan_file_sha256=_sha(controls[SHARDED_EXECUTION_PLAN_FILENAME]),
        trial_runtime_admission_receipt_file_sha256=_sha(controls[TRIAL_RUNTIME_RECEIPT_FILENAME]),
        feature_bindings=fixture.feature_bindings,
    )
    workload_spec_path = control_root / PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME
    workload_spec_path.write_bytes(workload_spec.canonical_file_bytes())
    launcher_control_root = (tmp_path / "launcher-control").resolve()
    launcher_control_root.mkdir(mode=0o700)
    runtime_plan_path = launcher_control_root / RUNTIME_ATTESTATION_PLAN_FILENAME
    config = ProductionCorpusRunConfig(
        control_root=control_root,
        output_root=output_root,
        sealed_run_receipt_path=run_path,
        runtime_attestation_plan_path=runtime_plan_path,
        workload_spec_file_sha256=workload_spec.file_sha256,
        online_custody_admission_file_sha256=_sha(controls[ONLINE_CUSTODY_ADMISSION_FILENAME]),
        required_artifact_bindings_file_sha256=_sha(controls[REQUIRED_ARTIFACT_BINDINGS_FILENAME]),
        sealed_run_receipt_file_sha256=digest_regular_file(
            run_path,
            label="fixture sealed run receipt",
        ),
    )
    config_path = control_root / PRODUCTION_CORPUS_CONFIG_FILENAME
    config_path.write_bytes(config.canonical_file_bytes())
    config_sha256 = config.file_sha256

    argv = (
        "/opt/venv/bin/python",
        "-m",
        "fractal_ann_diagnostics.cli",
        "run-sealed-corpus",
        "--config",
        str(config_path),
    )
    pins = {
        "opa": RuntimeFilePin(path="/usr/local/bin/opa", sha256="3" * 64),
        "python": RuntimeFilePin(path="/opt/venv/bin/python", sha256="5" * 64),
        "uv": RuntimeFilePin(path="/opt/app/uv.lock", sha256="6" * 64),
        "launcher": RuntimeFilePin(path="/input/launcher.json", sha256="7" * 64),
    }
    mounts = tuple(
        sorted(
            (
                *(
                    RuntimeArtifactMount(
                        root=pin.path,
                        role=f"{name}-control",
                        kind="file",
                        artifact_sha256=pin.sha256,
                    )
                    for name, pin in pins.items()
                ),
                RuntimeArtifactMount(
                    root=str(artifact_root),
                    role="sealed-online-artifact",
                    kind="directory",
                    artifact_sha256=workload_spec.artifact_tree_sha256,
                ),
                RuntimeArtifactMount(
                    root=str(index_root),
                    role="authorized-index-store",
                    kind="directory",
                    artifact_sha256=workload_spec.authorized_index_store_tree_sha256,
                ),
                RuntimeArtifactMount(
                    root=str(fixture.embedding_root),
                    role="embedding-store",
                    kind="directory",
                    artifact_sha256=workload_spec.embedding_store_tree_sha256,
                ),
                RuntimeArtifactMount(
                    root=str(fixture.partition_audit_path),
                    role="partition-audit",
                    kind="file",
                    artifact_sha256=workload_spec.partition_audit_file_sha256,
                ),
                RuntimeArtifactMount(
                    root=str(policy_root),
                    role="policy-intervention",
                    kind="directory",
                    artifact_sha256=workload_spec.policy_intervention_tree_sha256,
                ),
                RuntimeArtifactMount(
                    root=str(pseudonym_key),
                    role="pseudonym-key",
                    kind="file",
                    artifact_sha256=workload_spec.expected_pseudonym_key_sha256,
                ),
                RuntimeArtifactMount(
                    root=str(fixture.package_root),
                    role="query-package",
                    kind="directory",
                    artifact_sha256=workload_spec.query_package_tree_sha256,
                ),
                RuntimeArtifactMount(
                    root=str(fixture.staged_root),
                    role="staged-inputs",
                    kind="directory",
                    artifact_sha256=workload_spec.staged_tree_sha256,
                ),
                RuntimeArtifactMount(
                    root=str(policy_bundle_path),
                    role="policy-stage-bundle",
                    kind="file",
                    artifact_sha256=workload_spec.policy_bundle_receipt_sha256,
                ),
                RuntimeArtifactMount(
                    root=str(index_bundle_path),
                    role="index-stage-bundle",
                    kind="file",
                    artifact_sha256=workload_spec.index_bundle_receipt_sha256,
                ),
            ),
            key=lambda row: row.root.encode(),
        )
    )
    attestation_plan = RuntimeAttestationPlan(
        attestation_id="confirmatory-corpus-v1",
        manifest_sha256=run.manifest_sha256,
        runner_identity=run.runner_identity,
        oci_image_digest=run.runner_image,
        code_commit=run.code_commit,
        operating_system_id="debian",
        operating_system_version_id="12",
        kernel_release="6.12.0-linuxkit",
        architecture="x86_64",
        cpu_model="AMD EPYC 7763",
        logical_cpu_count=8,
        memory_limit_bytes=16 * 1024**3,
        mount_namespace_sha256="8" * 64,
        mounts=mounts,
        argv=argv,
        argv_sha256=argv_sha256(argv),
        environment_allowlist=(),
        environment_sha256=environment_sha256({}),
        opa_binary=pins["opa"],
        python_binary=pins["python"],
        python_version="3.12.11",
        uv_lock=pins["uv"],
        launcher_identity=pins["launcher"],
        workload_id=PRODUCTION_CORPUS_WORKLOAD_ID,
        workload_sha256=workload_spec.file_sha256,
        invocation_marker_path=str(output_root / RUNTIME_INVOCATION_MARKER_FILENAME),
    )
    marker_payload = (
        _canonical(
            {
                "plan_sha256": attestation_plan.plan_sha256,
                "schema_version": "fractal-runtime-invocation-marker-v1",
                "workload_id": attestation_plan.workload_id,
                "workload_sha256": attestation_plan.workload_sha256,
            }
        )
        + b"\n"
    )
    attestation_receipt = RuntimeAttestationReceipt(
        attestation_id=attestation_plan.attestation_id,
        plan_sha256=attestation_plan.plan_sha256,
        manifest_sha256=attestation_plan.manifest_sha256,
        runner_identity=attestation_plan.runner_identity,
        oci_image_digest=attestation_plan.oci_image_digest,
        code_commit=attestation_plan.code_commit,
        operating_system_id=attestation_plan.operating_system_id,
        operating_system_version_id=attestation_plan.operating_system_version_id,
        kernel_release=attestation_plan.kernel_release,
        architecture=attestation_plan.architecture,
        cpu_model=attestation_plan.cpu_model,
        logical_cpu_count=attestation_plan.logical_cpu_count,
        memory_limit_bytes=attestation_plan.memory_limit_bytes,
        mount_namespace_sha256=attestation_plan.mount_namespace_sha256,
        mount_namespace_raw_sha256="a" * 64,
        mounts=attestation_plan.mounts,
        network={
            "interfaces": ("lo",),
            "mode": "none",
            "namespace_inode": 42,
            "non_loopback_route_count": 0,
            "route_tables_sha256": "9" * 64,
        },
        process={
            "argument_count": len(argv),
            "argv_sha256": attestation_plan.argv_sha256,
            "environment_allowlist": (),
            "environment_sha256": attestation_plan.environment_sha256,
        },
        opa_binary=attestation_plan.opa_binary,
        python_binary=attestation_plan.python_binary,
        python_version=attestation_plan.python_version,
        uv_lock=attestation_plan.uv_lock,
        launcher_identity=attestation_plan.launcher_identity,
        workload_id=attestation_plan.workload_id,
        workload_sha256=attestation_plan.workload_sha256,
        invocation_marker_path=attestation_plan.invocation_marker_path,
        invocation_marker_sha256=_sha(marker_payload),
    )
    runtime_plan_path.write_bytes(attestation_plan.canonical_file_bytes())
    return _ProductionCommandPackage(
        config=config,
        config_path=config_path,
        config_sha256=config_sha256,
        output_root=output_root,
        attestation_plan=attestation_plan,
        attestation_receipt=attestation_receipt,
        marker_payload=marker_payload,
        runtime_claim=runtime_claim,
    )


def _invoke_production_command(
    package: _ProductionCommandPackage,
    monkeypatch: pytest.MonkeyPatch,
    *,
    arguments: Sequence[str] | None = None,
    claim_bytes: bytes | None = None,
) -> int:
    encoded = package.runtime_claim.canonical_file_bytes() if claim_bytes is None else claim_bytes
    stream = io.TextIOWrapper(io.BytesIO(encoded), encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stream)
    return cli_main(
        list(arguments)
        if arguments is not None
        else ["run-sealed-corpus", "--config", str(package.config_path)]
    )


def _publish_test_runtime_attestation(package: _ProductionCommandPackage) -> None:
    write_exclusive_receipt_bytes(
        package.marker_payload,
        package.output_root / RUNTIME_INVOCATION_MARKER_FILENAME,
    )
    write_exclusive_receipt_bytes(
        package.attestation_receipt.canonical_file_bytes(),
        package.output_root / RUNTIME_ATTESTATION_RECEIPT_FILENAME,
    )


def test_builds_opaque_source_bound_rows_without_outcome_fields(
    runtime_fixture: _Fixture,
) -> None:
    receipt = verify_query_trial_store(
        runtime_fixture.package_root,
        runtime_fixture.staged_root,
        runtime_fixture.embedding_root,
        partition_audit_path=runtime_fixture.partition_audit_path,
        secret=runtime_fixture.secret,
    )
    assert receipt.record_count == 3
    assert receipt.available_family_count == 1
    assert receipt.selected_family_count == 1
    assert receipt.nested_rows_per_family == NESTED_ROWS_PER_FAMILY
    assert receipt.active_query_epoch.role == "active-migration"
    assert receipt.current_truth_query_epoch.role == "current-exact-truth"
    assert receipt.active_query_epoch.file_sha256 != receipt.current_truth_query_epoch.file_sha256
    encoded = (runtime_fixture.package_root / QUERY_TRIAL_FILENAME).read_bytes()
    rows = [CanonicalQueryTrialRow.from_dict(json.loads(line)) for line in encoded.splitlines()]
    component_sha256 = _sha(_canonical(["sealed-0", "sealed-1", "sealed-2"]))
    expected_query = min(
        (f"sealed-{index}" for index in range(3)),
        key=lambda query_id: (
            representative_selection_rank(
                corpus="demo",
                stage="sealed",
                selection_seed_sha256=_SELECTION_SEED_SHA256,
                component_sha256=component_sha256,
                query_id_sha256=_sha(query_id.encode()),
            ),
            _sha(query_id.encode()),
        ),
    )
    expected_embedding_row = 2 + int(expected_query.rsplit("-", 1)[1])
    assert [row.source.embedding_query_row for row in rows] == [expected_embedding_row] * 3
    assert len({row.text for row in rows}) == 1
    assert len({row.family_key for row in rows}) == 1
    assert tuple(row.opaque_row for row in rows) == receipt.opaque_trials
    assert rows[0].trial_key == _opaque_v2(
        runtime_fixture.secret,
        domain="trial",
        source_value=nested_trial_source_value(expected_query, 0),
    )
    assert rows[0].family_key == _opaque_v2(
        runtime_fixture.secret,
        domain="family",
        source_value=_sha(_canonical(["sealed-0", "sealed-1", "sealed-2"])),
    )
    assert b"sealed-0" not in encoded
    lowered = encoded.lower()
    assert all(token not in lowered for token in (b"qrel", b"answer", b"gold"))


def test_production_builder_matches_independent_custody_cohort_exactly(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    inventory = json.loads((runtime_fixture.staged_root / "inventory.json").read_bytes())
    allowlisted_paths = tuple(
        row["path"]
        for row in inventory["artifacts"]
        if row["path"]
        in {
            "assignments.jsonl",
            "datasets/demo/corpus/part-00000.jsonl",
            "datasets/demo/sealed/custody/qrels.jsonl",
            "datasets/demo/sealed/online/queries.jsonl",
        }
    )
    custody_plan = ScalableCustodyPlan(
        corpus="demo",
        staged_inventory_sha256=_sha((runtime_fixture.staged_root / "inventory.json").read_bytes()),
        execution_artifact_sha256=runtime_fixture.plan.artifact_sha256,
        hmac_key_id="sealed-query-hmac-2026-07",
        expected_document_count=6,
        available_families=1,
        selected_families=1,
        selection_seed_sha256=_SELECTION_SEED_SHA256,
        allowlisted_paths=allowlisted_paths,
    )
    custody_root = (tmp_path / "custody-package").resolve()
    custody_receipt = build_scalable_custody_package(
        runtime_fixture.staged_root,
        custody_root,
        plan=custody_plan,
        hmac_secret=runtime_fixture.secret,
    )
    runtime_receipt = verify_query_trial_store(
        runtime_fixture.package_root,
        runtime_fixture.staged_root,
        runtime_fixture.embedding_root,
        partition_audit_path=runtime_fixture.partition_audit_path,
        secret=runtime_fixture.secret,
    )
    assert custody_receipt.query_count == runtime_receipt.record_count == 3
    assert (
        verify_query_trial_key_parity(
            custody_root,
            (runtime_fixture.package_root / QUERY_TRIAL_FILENAME).resolve(),
            expected_runtime_sha256=runtime_receipt.query_trial_store_sha256,
            expected_runtime_byte_count=runtime_receipt.query_trial_store_byte_count,
        )
        == runtime_receipt.query_trial_store_sha256
    )
    custody_rows = [
        json.loads(line) for line in (custody_root / QUERY_KEY_MAP_PATH).read_bytes().splitlines()
    ]
    runtime_rows = [
        json.loads(line)
        for line in (runtime_fixture.package_root / QUERY_TRIAL_FILENAME).read_bytes().splitlines()
    ]
    assert [
        (
            row["query_row"],
            row["text"],
            row["trial_key"],
            row["family_key"],
        )
        for row in runtime_rows
    ] == [
        (
            row["query_row"],
            row["text"],
            row["trial_key"],
            row["family_key"],
        )
        for row in custody_rows
    ]


def test_receipt_rejects_family_denominator_and_block_forgery(
    runtime_fixture: _Fixture,
) -> None:
    receipt = verify_query_trial_store(
        runtime_fixture.package_root,
        runtime_fixture.staged_root,
        runtime_fixture.embedding_root,
        partition_audit_path=runtime_fixture.partition_audit_path,
    )
    with pytest.raises(TrialRuntimeError, match="record_count must equal"):
        replace(
            receipt,
            available_family_count=2,
            selected_family_count=2,
        )
    forged_rows = list(receipt.opaque_trials)
    forged_rows[1] = replace(forged_rows[1], family_key="f" * 64)
    with pytest.raises(TrialRuntimeError, match="one exact nested block"):
        replace(receipt, opaque_trials=tuple(forged_rows))


def test_builder_rejects_registered_available_family_mismatch(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    target = (tmp_path / "wrong-family-count").resolve()
    with pytest.raises(TrialRuntimeError, match="available family count differs"):
        build_query_trial_store(
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            target,
            partition_audit_path=runtime_fixture.partition_audit_path,
            corpus="demo",
            stage="sealed",
            hmac_key_id="sealed-query-hmac-2026-07",
            hmac_secret=runtime_fixture.secret,
            selection_seed_sha256=_SELECTION_SEED_SHA256,
            available_family_count=2,
            selected_family_count=1,
        )
    assert not target.exists()


def test_admission_has_deterministic_policy_blocks_and_exclusive_receipt(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    admission = _admit(runtime_fixture, tmp_path)
    receipt_path = (tmp_path / "runtime-receipt.json").resolve()
    assert load_trial_runtime_receipt(receipt_path) == admission.receipt
    assert admission.receipt.query_count == 3
    assert admission.receipt.query_partition_audit_sha256 == (
        runtime_fixture.plan.query_partition_audit_sha256
    )
    assert admission.receipt.permutation_seed == 20260714
    assert [row.block_order for row in admission.receipt.groups] == list(
        range(len(admission.receipt.groups))
    )
    assert len(admission.receipt.groups) == 3
    assert admission.receipt.assignment_map_sha256
    assert admission.receipt.trial_state_assignment_algorithm == (
        "sha256-config-seed-family-trial-rank-v1"
    )
    assert "action" not in admission.receipt.canonical_bytes().decode().casefold()
    with pytest.raises(TrialRuntimeError, match="already exists"):
        admit_trial_runtime(
            runtime_fixture.plan,
            runtime_fixture.package_root,
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            runtime_fixture.schedule_path,
            runtime_fixture.feature_bindings,
            partition_audit_path=runtime_fixture.partition_audit_path,
            receipt_target=receipt_path,
        )


def test_reconstructs_lazy_admission_from_canonical_controls_without_source_io(
    runtime_fixture: _Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _admit(runtime_fixture, tmp_path)
    plan_path = (tmp_path / "sharded-plan.json").resolve()
    write_sharded_online_execution_plan(runtime_fixture.plan, plan_path)

    def forbidden_source(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("workload source opened during control reconstruction")

    monkeypatch.setattr(
        "fractal_ann_diagnostics.trial_runtime.verify_query_trial_store",
        forbidden_source,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.trial_runtime.load_canonical_trial_schedule",
        forbidden_source,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.trial_runtime.verify_embedding_store",
        forbidden_source,
    )
    restored = load_trial_runtime_admission(
        plan_path=plan_path,
        receipt_path=(tmp_path / "runtime-receipt.json").resolve(),
        partition_audit_path=runtime_fixture.partition_audit_path,
        query_package_root=runtime_fixture.package_root,
        staged_root=runtime_fixture.staged_root,
        embedding_store_root=runtime_fixture.embedding_root,
        schedule_path=runtime_fixture.schedule_path,
        feature_bindings=runtime_fixture.feature_bindings,
    )

    assert restored == admitted
    with pytest.raises(TrialRuntimeError, match="feature bindings differ"):
        load_trial_runtime_admission(
            plan_path=plan_path,
            receipt_path=(tmp_path / "runtime-receipt.json").resolve(),
            partition_audit_path=runtime_fixture.partition_audit_path,
            query_package_root=runtime_fixture.package_root,
            staged_root=runtime_fixture.staged_root,
            embedding_store_root=runtime_fixture.embedding_root,
            schedule_path=runtime_fixture.schedule_path,
            feature_bindings=(
                replace(runtime_fixture.feature_bindings[0], version_lag=2.0),
                *runtime_fixture.feature_bindings[1:],
            ),
        )


def test_two_argument_command_loads_the_tiny_closed_package_and_consumes_replay(
    runtime_fixture: _Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package = _production_command_package(runtime_fixture, tmp_path)
    assert list(package.output_root.iterdir()) == []
    process_id = os.getpid()
    events: list[tuple[str, int]] = []
    calls: list[dict[str, object]] = []
    probe = object()

    def probe_factory() -> object:
        events.append(("probe-constructed", os.getpid()))
        return probe

    def fake_attest(
        plan: RuntimeAttestationPlan,
        *,
        probe: object,
        receipt_target: str | Path | None = None,
    ) -> RuntimeAttestationReceipt:
        events.append(("attestation-entered", os.getpid()))
        assert probe is probe_instance
        assert plan == package.attestation_plan
        assert receipt_target == package.config.runtime_attestation_receipt_path
        assert list(package.output_root.iterdir()) == []
        write_exclusive_receipt_bytes(
            package.marker_payload,
            package.config.runtime_invocation_marker_path,
        )
        events.append(("runtime-marker-created", os.getpid()))
        write_exclusive_receipt_bytes(
            package.attestation_receipt.canonical_file_bytes(),
            package.config.runtime_attestation_receipt_path,
        )
        events.append(("runtime-receipt-created", os.getpid()))
        return package.attestation_receipt

    probe_instance = probe
    monkeypatch.setattr(production_entry, "LinuxRuntimeProbe", probe_factory)
    monkeypatch.setattr(production_entry, "attest_runtime_once", fake_attest)

    original_read_pinned_control = production_entry._read_pinned_control

    def guarded_read_pinned_control(
        path: Path,
        expected_sha256: str,
        *,
        label: str,
    ) -> bytes:
        if label != "production corpus workload spec":
            events.append((f"control:{label}", os.getpid()))
            assert package.config.runtime_invocation_marker_path.is_file()
        return original_read_pinned_control(
            path,
            expected_sha256,
            label=label,
        )

    monkeypatch.setattr(
        production_entry,
        "_read_pinned_control",
        guarded_read_pinned_control,
    )

    original_reconstruct = production_entry.reconstruct_trial_runtime_admission

    def guarded_reconstruct(**kwargs: object) -> TrialRuntimeAdmission:
        events.append(("runtime-admission", os.getpid()))
        assert package.config.runtime_invocation_marker_path.is_file()
        return original_reconstruct(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        production_entry,
        "reconstruct_trial_runtime_admission",
        guarded_reconstruct,
    )

    def fake_run(**kwargs: object) -> SimpleNamespace:
        events.append(("scientific-boundary", os.getpid()))
        calls.append(kwargs)
        assert package.config.runtime_invocation_marker_path.is_file()
        assert package.config.runtime_attestation_receipt_path.is_file()
        assert (package.output_root / PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME).is_file()
        runtime_admission = kwargs["runtime_admission"]
        assert isinstance(runtime_admission, TrialRuntimeAdmission)
        assert kwargs["expected_runtime_receipt_sha256"] == (
            runtime_admission.receipt.receipt_sha256
        )
        assert kwargs["expected_policy_intervention_receipt_sha256"] == "2" * 64
        assert "k" not in kwargs
        assert "policy_action" not in kwargs
        return SimpleNamespace(
            output_root=package.output_root,
            attempt_receipt=SimpleNamespace(receipt_sha256="a" * 64),
            result_receipt=SimpleNamespace(receipt_sha256="b" * 64),
        )

    monkeypatch.setattr(
        "fractal_ann_diagnostics.production_corpus_run.run_sealed_online_once",
        fake_run,
    )
    assert _invoke_production_command(package, monkeypatch) == 0
    assert len(calls) == 1
    assert "completed sealed corpus attempt" in capsys.readouterr().out
    command_attempt_path = package.output_root / PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME
    assert command_attempt_path.is_file()
    command_attempt = load_production_corpus_command_attempt(command_attempt_path)
    assert command_attempt.config_file_sha256 == package.config_sha256
    assert command_attempt.manifest_sha256 == package.attestation_plan.manifest_sha256
    assert command_attempt.runtime_attestation_plan_sha256 == package.attestation_plan.plan_sha256
    assert (
        command_attempt.runtime_attestation_receipt_sha256
        == package.attestation_receipt.receipt_sha256
    )
    event_names = [name for name, _ in events]
    assert event_names[:4] == [
        "probe-constructed",
        "attestation-entered",
        "runtime-marker-created",
        "runtime-receipt-created",
    ]
    assert event_names.index("runtime-marker-created") < event_names.index(
        "control:online custody admission"
    )
    assert event_names.index("runtime-marker-created") < event_names.index("runtime-admission")
    assert event_names.index("runtime-marker-created") < event_names.index("scientific-boundary")
    assert {pid for _, pid in events} == {process_id}

    with pytest.raises(ProductionCorpusRunError, match="membership differs"):
        _invoke_production_command(package, monkeypatch)
    assert len(calls) == 1
    assert [name for name, _ in events].count("attestation-entered") == 1


def test_production_command_attempt_loader_rejects_schema_extension(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "substituted-command-attempt.json").resolve()
    target.write_bytes(
        _canonical(
            {
                "config_file_sha256": "1" * 64,
                "manifest_sha256": "2" * 64,
                "runtime_attestation_plan_sha256": "3" * 64,
                "runtime_attestation_receipt_sha256": "4" * 64,
                "schema_version": "fractal-production-corpus-command-attempt-v1",
                "unregistered_retry": True,
                "workload_id": PRODUCTION_CORPUS_WORKLOAD_ID,
            }
        )
        + b"\n"
    )
    with pytest.raises(ProductionCorpusRunError, match="fields differ"):
        load_production_corpus_command_attempt(target)


@pytest.mark.parametrize(
    "preexisting_name",
    [
        RUNTIME_ATTESTATION_RECEIPT_FILENAME,
        RUNTIME_INVOCATION_MARKER_FILENAME,
        "unregistered-output.json",
    ],
)
def test_production_command_rejects_any_preexisting_output_before_attestation(
    runtime_fixture: _Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting_name: str,
) -> None:
    package = _production_command_package(runtime_fixture, tmp_path)
    (package.output_root / preexisting_name).write_bytes(b"occupied\n")

    def forbidden_attestation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("attestation reached with a non-empty output directory")

    monkeypatch.setattr(production_entry, "attest_runtime_once", forbidden_attestation)
    with pytest.raises(ProductionCorpusRunError, match="membership differs"):
        _invoke_production_command(package, monkeypatch)


@pytest.mark.parametrize(
    ("binding", "expected_error"),
    [
        ("argv", "argv differs"),
        ("workload", "frozen workload spec or marker"),
        ("marker", "frozen workload spec or marker"),
    ],
)
def test_runtime_plan_invocation_substitution_fails_before_probe(
    runtime_fixture: _Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: str,
    expected_error: str,
) -> None:
    package = _production_command_package(runtime_fixture, tmp_path)
    plan = package.attestation_plan
    if binding == "argv":
        changed_argv = (*plan.argv, "--k", "100")
        plan = replace(
            plan,
            argv=changed_argv,
            argv_sha256=argv_sha256(changed_argv),
        )
    elif binding == "workload":
        plan = replace(plan, workload_sha256="f" * 64)
    else:
        plan = replace(
            plan,
            invocation_marker_path=str(package.output_root / "alternate-marker.json"),
        )
    package.config.runtime_attestation_plan_path.write_bytes(plan.canonical_file_bytes())

    def forbidden_probe() -> None:
        raise AssertionError("Linux probe constructed before plan binding rejection")

    monkeypatch.setattr(production_entry, "LinuxRuntimeProbe", forbidden_probe)
    with pytest.raises(ProductionCorpusRunError, match=expected_error):
        _invoke_production_command(package, monkeypatch)


def test_failed_same_process_attestation_consumes_the_runtime_marker(
    runtime_fixture: _Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _production_command_package(runtime_fixture, tmp_path)
    probe = object()
    calls = 0

    monkeypatch.setattr(production_entry, "LinuxRuntimeProbe", lambda: probe)

    def failed_attestation(
        plan: RuntimeAttestationPlan,
        *,
        probe: object,
        receipt_target: str | Path | None = None,
    ) -> RuntimeAttestationReceipt:
        nonlocal calls
        calls += 1
        assert probe is probe_instance
        assert receipt_target == package.config.runtime_attestation_receipt_path
        write_exclusive_receipt_bytes(
            package.marker_payload,
            plan.invocation_marker_path,
        )
        raise RuntimeAttestationError("hostile runtime observation")

    probe_instance = probe
    command = [
        "run-sealed-corpus",
        "--config",
        str(package.config_path),
    ]
    monkeypatch.setattr(production_entry, "attest_runtime_once", failed_attestation)
    with pytest.raises(RuntimeAttestationError, match="hostile runtime observation"):
        _invoke_production_command(package, monkeypatch, arguments=command)
    assert package.config.runtime_invocation_marker_path.is_file()
    assert not package.config.runtime_attestation_receipt_path.exists()

    with pytest.raises(ProductionCorpusRunError, match="membership differs"):
        _invoke_production_command(package, monkeypatch, arguments=command)
    assert calls == 1


def test_production_command_rejects_unknown_cli_scientific_overrides(
    runtime_fixture: _Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _production_command_package(runtime_fixture, tmp_path)
    with pytest.raises(SystemExit) as caught:
        _invoke_production_command(
            package,
            monkeypatch,
            arguments=[
                "run-sealed-corpus",
                "--config",
                str(package.config_path),
                "--k",
                "100",
            ],
        )
    assert caught.value.code == 2


@pytest.mark.parametrize(
    ("case", "expected_error"),
    (
        ("empty", "must end with one newline"),
        ("noncanonical", "bytes are not canonical"),
        ("oversized", "exceeds its fixed byte limit"),
    ),
)
def test_production_command_rejects_invalid_runtime_claim_stdin_before_admission(
    runtime_fixture: _Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_error: str,
) -> None:
    package = _production_command_package(runtime_fixture, tmp_path)
    payloads = {
        "empty": b"",
        "noncanonical": (
            json.dumps(
                package.runtime_claim.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        ),
        "oversized": b"x" * (256 * 1024 + 1),
    }

    with pytest.raises(ExecutionClaimError, match=expected_error):
        _invoke_production_command(
            package,
            monkeypatch,
            claim_bytes=payloads[case],
        )
    assert list(package.output_root.iterdir()) == []


def test_production_control_root_rejects_extra_files_and_symlink_substitution(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    package = _production_command_package(runtime_fixture, tmp_path)
    _publish_test_runtime_attestation(package)
    extra = package.config.control_root / "alternate-plan.json"
    extra.write_bytes(b"{}\n")
    with pytest.raises(ProductionCorpusRunError, match="extra"):
        load_admitted_production_corpus_controls(
            package.config_path,
        )
    extra.unlink()

    bindings = package.config.control_root / REQUIRED_ARTIFACT_BINDINGS_FILENAME
    replacement = (tmp_path / "replacement-bindings.json").resolve()
    replacement.write_bytes(bindings.read_bytes())
    bindings.unlink()
    bindings.symlink_to(replacement)
    with pytest.raises(ProductionCorpusRunError, match="singly linked regular file"):
        load_admitted_production_corpus_controls(
            package.config_path,
        )


def test_production_command_rejects_digest_drift_and_alternate_config_path(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    package = _production_command_package(runtime_fixture, tmp_path)
    _publish_test_runtime_attestation(package)
    alternate = (tmp_path / "alternate-config.json").resolve()
    alternate.write_bytes(package.config_path.read_bytes())
    with pytest.raises(ProductionCorpusRunError, match="self-declared fixed path"):
        load_admitted_production_corpus_controls(
            alternate,
        )

    admission_path = package.config.control_root / ONLINE_CUSTODY_ADMISSION_FILENAME
    admission_path.write_bytes(
        admission_path.read_bytes().replace(b"runner-65532", b"runner-65533")
    )
    with pytest.raises(ProductionCorpusRunError, match="differs from its config pin"):
        load_admitted_production_corpus_controls(
            package.config_path,
        )


def test_production_command_rejects_config_substitution_even_with_new_cli_hash(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    package = _production_command_package(runtime_fixture, tmp_path)
    _publish_test_runtime_attestation(package)
    payload = json.loads(package.config_path.read_bytes())
    payload["sealed_run_receipt_file_sha256"] = "f" * 64
    substituted = _canonical(payload) + b"\n"
    package.config_path.write_bytes(substituted)

    with pytest.raises(ProductionCorpusRunError, match="sealed run receipt differs"):
        load_admitted_production_corpus_controls(
            package.config_path,
        )


def test_production_config_rejects_unknown_fields_before_any_attempt(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    package = _production_command_package(runtime_fixture, tmp_path)
    payload = json.loads(package.config_path.read_bytes())
    payload["alternate_partition"] = "reserve"
    substituted = _canonical(payload) + b"\n"
    package.config_path.write_bytes(substituted)

    with pytest.raises(ProductionCorpusRunError, match="unknown"):
        load_admitted_production_corpus_controls(
            package.config_path,
        )


def test_loads_one_read_only_dual_epoch_query_map(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    block = load_trial_runtime_block(_admit(runtime_fixture, tmp_path), 0)
    assert len(block.trial_runtimes) == 1
    assert block.descriptor.subject == "analyst"
    for runtime in block.trial_runtimes.values():
        assert not runtime.active_query_vector.flags.writeable
        assert not runtime.current_truth_query_vector.flags.writeable
        assert not np.array_equal(
            runtime.active_query_vector,
            runtime.current_truth_query_vector,
        )
        assert runtime.environment["assignment_repetition"] == 0
        with pytest.raises(ValueError):
            runtime.active_query_vector[0] = 0.0
    with pytest.raises(TypeError):
        block.trial_runtimes["new"] = next(iter(block.trial_runtimes.values()))  # type: ignore[index]


def test_combines_disjoint_blocks_into_one_plan_bound_runtime(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    admission = _admit(runtime_fixture, tmp_path)
    loaded = load_trial_runtime(admission)

    assert loaded.execution.artifact_sha256 == runtime_fixture.plan.artifact_sha256
    assert loaded.execution.trial_keys == runtime_fixture.plan.trial_keys
    assert set(loaded.trial_runtimes) == set(runtime_fixture.plan.trial_keys)
    assert len(loaded.descriptors) == 3
    policy_states = {
        str(runtime.environment["policy_state"]) for runtime in loaded.trial_runtimes.values()
    }
    assert policy_states == {"low", "medium", "high"}
    assert (
        len(
            {
                (trial_key, action)
                for trial_key in loaded.trial_runtimes
                for action in ("hnsw-low", "hnsw-high", "exact")
            }
        )
        == len(loaded.trial_runtimes) * 3
    )


@pytest.mark.parametrize("matrix", ["old_queries", "current_queries"])
def test_loaded_runtime_rejects_same_shape_query_epoch_substitution(
    runtime_fixture: _Fixture,
    tmp_path: Path,
    matrix: str,
) -> None:
    admission = _admit(runtime_fixture, tmp_path)
    embedding_receipt = json.loads((runtime_fixture.embedding_root / "receipt.json").read_bytes())
    descriptor = embedding_receipt["vectors"][matrix]
    target = runtime_fixture.embedding_root / descriptor["relative_path"]
    original = np.load(target, allow_pickle=False)
    replacement = tmp_path / f"substituted-{matrix}.npy"
    mapped = np.lib.format.open_memmap(
        replacement,
        mode="w+",
        dtype=original.dtype,
        shape=original.shape,
        fortran_order=False,
        version=(2, 0),
    )
    mapped[:] = -original
    mapped.flush()
    del mapped
    replacement.replace(target)

    with pytest.raises(TrialRuntimeError, match="embedding store verification failed"):
        load_trial_runtime(admission)


def test_loaded_runtime_rejects_feature_context_substitution(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    admission = _admit(runtime_fixture, tmp_path)
    changed = replace(
        admission.feature_bindings[0],
        backend="substituted-backend",
        policy_complexity=0.999,
    )
    changed_admission = replace(
        admission,
        feature_bindings=(changed, *admission.feature_bindings[1:]),
    )

    with pytest.raises(TrialRuntimeError, match="feature contexts changed"):
        load_trial_runtime(changed_admission)


def test_loaded_runtime_rejects_environment_assignment_substitution(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    admission = _admit(runtime_fixture, tmp_path)
    original = compile_policy_intervention(
        runtime_fixture.plan,
        PolicyInterventionConfig(
            seed_sha256=_sha(b"policy seed"),
            baseline_seed_sha256=_sha(b"baseline policy seed"),
            policy_bundle_revision=f"sha256:{_sha(b'policy bundle')}",
            baseline_policy_revision=f"sha256:{_sha(b'baseline policy bundle')}",
            subject_ids=("analyst",),
            assignment_repetitions=1,
        ),
    ).schedule
    original_environments = {row.trial_key: row.environment_sha256 for row in original.rows}
    changed_schedule = None
    for index in range(1, 64):
        candidate = compile_policy_intervention(
            runtime_fixture.plan,
            PolicyInterventionConfig(
                seed_sha256=_sha(f"changed policy seed {index}".encode()),
                baseline_seed_sha256=_sha(b"baseline policy seed"),
                policy_bundle_revision=f"sha256:{_sha(b'policy bundle')}",
                baseline_policy_revision=f"sha256:{_sha(b'baseline policy bundle')}",
                subject_ids=("analyst",),
                assignment_repetitions=1,
            ),
        ).schedule
        if {
            row.trial_key: row.environment_sha256 for row in candidate.rows
        } != original_environments:
            changed_schedule = candidate
            break
    assert changed_schedule is not None
    changed_path = (tmp_path / "changed-environment-schedule.json").resolve()
    write_exclusive_receipt_bytes(
        changed_schedule.canonical_file_bytes(),
        changed_path,
    )

    with pytest.raises(TrialRuntimeError, match="runtime group changed"):
        load_trial_runtime(replace(admission, schedule_path=changed_path))


def test_loaded_runtime_rejects_receipt_seed_substitution(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    admission = _admit(runtime_fixture, tmp_path)
    changed_receipt = replace(
        admission.receipt,
        permutation_seed=admission.receipt.permutation_seed + 1,
    )

    with pytest.raises(TrialRuntimeError, match="runtime source changed"):
        load_trial_runtime(replace(admission, receipt=changed_receipt))


def test_loaded_runtime_rejects_query_receipt_pin_substitution(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    admission = _admit(runtime_fixture, tmp_path)
    changed_plan = replace(
        admission.plan,
        query_trial_store=replace(
            admission.plan.query_trial_store,
            receipt=replace(
                admission.plan.query_trial_store.receipt,
                sha256="7" * 64,
            ),
        ),
    )

    with pytest.raises(TrialRuntimeError, match="runtime source changed"):
        load_trial_runtime(replace(admission, plan=changed_plan))


def test_source_mutation_and_hard_links_are_rejected(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    source = runtime_fixture.staged_root / "datasets/demo/sealed/online/queries.jsonl"
    source_bytes = source.read_bytes()
    source.write_bytes(source_bytes.replace(b"find alpha", b"find omega"))
    with pytest.raises(TrialRuntimeError, match="digest differs"):
        verify_query_trial_store(
            runtime_fixture.package_root,
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            partition_audit_path=runtime_fixture.partition_audit_path,
        )
    source.write_bytes(source_bytes)

    original = runtime_fixture.package_root / QUERY_TRIAL_FILENAME
    copy = tmp_path / "query-copy.jsonl"
    copy.write_bytes(original.read_bytes())
    original.unlink()
    os.link(copy, original)
    with pytest.raises(TrialRuntimeError, match="hard-linked"):
        verify_query_trial_store(
            runtime_fixture.package_root,
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            partition_audit_path=runtime_fixture.partition_audit_path,
        )


def test_missing_schedule_row_and_feature_block_are_rejected(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    compiled = compile_policy_intervention(
        runtime_fixture.plan,
        PolicyInterventionConfig(
            seed_sha256=_sha(b"policy seed"),
            baseline_seed_sha256=_sha(b"baseline policy seed"),
            policy_bundle_revision=f"sha256:{_sha(b'policy bundle')}",
            baseline_policy_revision=f"sha256:{_sha(b'baseline policy bundle')}",
            subject_ids=("analyst",),
            assignment_repetitions=1,
        ),
    )
    incomplete = compiled.schedule.to_dict()
    incomplete_rows = incomplete["rows"]
    assert isinstance(incomplete_rows, list)
    incomplete["rows"] = incomplete_rows[1:]
    for position, row in enumerate(incomplete["rows"]):
        assert isinstance(row, dict)
        row["schedule_order"] = position
    incomplete_path = (tmp_path / "incomplete-schedule.json").resolve()
    write_exclusive_receipt_bytes(_canonical(incomplete) + b"\n", incomplete_path)
    with pytest.raises(TrialRuntimeError, match="cannot load policy schedule"):
        admit_trial_runtime(
            runtime_fixture.plan,
            runtime_fixture.package_root,
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            incomplete_path,
            runtime_fixture.feature_bindings,
            partition_audit_path=runtime_fixture.partition_audit_path,
        )
    with pytest.raises(TrialRuntimeError, match="exact schedule blocks"):
        admit_trial_runtime(
            runtime_fixture.plan,
            runtime_fixture.package_root,
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            runtime_fixture.schedule_path,
            runtime_fixture.feature_bindings[:-1],
            partition_audit_path=runtime_fixture.partition_audit_path,
        )


def test_admission_rejects_schedule_family_mapping_substitution(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    class _ChangedFamilies:
        corpus = runtime_fixture.plan.corpus
        stage = runtime_fixture.plan.stage
        document_count = runtime_fixture.plan.document_count
        document_universe_sha256 = runtime_fixture.plan.document_universe_sha256
        artifact_sha256 = runtime_fixture.plan.artifact_sha256
        trials = tuple(
            replace(row, family_key=_sha(b"substituted-family"))
            for row in runtime_fixture.plan.trials
        )

    compiled = compile_policy_intervention(
        _ChangedFamilies(),
        PolicyInterventionConfig(
            seed_sha256=_sha(b"policy seed"),
            baseline_seed_sha256=_sha(b"baseline policy seed"),
            policy_bundle_revision=f"sha256:{_sha(b'policy bundle')}",
            baseline_policy_revision=f"sha256:{_sha(b'baseline policy bundle')}",
            subject_ids=("analyst",),
        ),
    )
    schedule_path = (tmp_path / "substituted-family-schedule.json").resolve()
    write_exclusive_receipt_bytes(compiled.schedule.canonical_file_bytes(), schedule_path)
    with pytest.raises(TrialRuntimeError, match="family mapping differs"):
        admit_trial_runtime(
            runtime_fixture.plan,
            runtime_fixture.package_root,
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            schedule_path,
            runtime_fixture.feature_bindings,
            partition_audit_path=runtime_fixture.partition_audit_path,
        )


def test_typed_audit_and_permutation_seed_substitution_are_rejected(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    changed_audit = (tmp_path / "changed-audit.json").resolve()
    changed_audit.write_bytes(runtime_fixture.partition_audit_path.read_bytes() + b"\n")
    with pytest.raises(TrialRuntimeError, match="typed query-partition audit"):
        verify_query_trial_store(
            runtime_fixture.package_root,
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            partition_audit_path=changed_audit,
        )

    changed_plan = replace(
        runtime_fixture.plan,
        permutation_seed=runtime_fixture.plan.permutation_seed + 1,
    )
    with pytest.raises(TrialRuntimeError, match="policy schedule differs"):
        admit_trial_runtime(
            changed_plan,
            runtime_fixture.package_root,
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            runtime_fixture.schedule_path,
            runtime_fixture.feature_bindings,
            partition_audit_path=runtime_fixture.partition_audit_path,
        )

    changed_digest_plan = replace(
        runtime_fixture.plan,
        query_partition_audit_sha256="9" * 64,
    )
    with pytest.raises(TrialRuntimeError, match="query/trial package"):
        admit_trial_runtime(
            changed_digest_plan,
            runtime_fixture.package_root,
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            runtime_fixture.schedule_path,
            runtime_fixture.feature_bindings,
            partition_audit_path=runtime_fixture.partition_audit_path,
        )

    changed_receipt_plan = replace(
        runtime_fixture.plan,
        query_trial_store=replace(
            runtime_fixture.plan.query_trial_store,
            receipt=replace(
                runtime_fixture.plan.query_trial_store.receipt,
                sha256="8" * 64,
            ),
        ),
    )
    with pytest.raises(TrialRuntimeError, match="query/trial package"):
        admit_trial_runtime(
            changed_receipt_plan,
            runtime_fixture.package_root,
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            runtime_fixture.schedule_path,
            runtime_fixture.feature_bindings,
            partition_audit_path=runtime_fixture.partition_audit_path,
        )


def test_hmac_key_and_package_publication_are_frozen(
    runtime_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    with pytest.raises(TrialRuntimeError, match="already exists"):
        build_query_trial_store(
            runtime_fixture.staged_root,
            runtime_fixture.embedding_root,
            runtime_fixture.package_root,
            partition_audit_path=runtime_fixture.partition_audit_path,
            corpus="demo",
            stage="sealed",
            hmac_key_id="sealed-query-hmac-2026-07",
            hmac_secret=runtime_fixture.secret,
            selection_seed_sha256=_SELECTION_SEED_SHA256,
            available_family_count=1,
            selected_family_count=1,
        )
    second = (tmp_path / "second-query-package").resolve()
    changed = build_query_trial_store(
        runtime_fixture.staged_root,
        runtime_fixture.embedding_root,
        second,
        partition_audit_path=runtime_fixture.partition_audit_path,
        corpus="demo",
        stage="sealed",
        hmac_key_id="sealed-query-hmac-2026-07",
        hmac_secret=bytes(range(31, -1, -1)),
        selection_seed_sha256=_SELECTION_SEED_SHA256,
        available_family_count=1,
        selected_family_count=1,
    )
    first = verify_query_trial_store(
        runtime_fixture.package_root,
        runtime_fixture.staged_root,
        runtime_fixture.embedding_root,
        partition_audit_path=runtime_fixture.partition_audit_path,
    )
    assert {row.trial_key for row in first.opaque_trials}.isdisjoint(
        {row.trial_key for row in changed.opaque_trials}
    )
