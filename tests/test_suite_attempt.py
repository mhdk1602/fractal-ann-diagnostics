from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_execution_claim import (
    _claim as _base_execution_claim,
)
from test_execution_claim import (
    _live_job,
    _phase_provider,
    _provider,
    _zenodo,
)
from test_execution_claim import (
    _phase_contract as _base_phase_claim,
)
from test_execution_claim import (
    _Verifier as _BeaconVerifier,
)

import fractal_ann_diagnostics.confirmatory_execution as confirmatory_execution_module
import fractal_ann_diagnostics.offline_analysis_contract as offline_contract_module
import fractal_ann_diagnostics.suite_attempt as suite_attempt_module
import fractal_ann_diagnostics.timelock_release as timelock_release_module
from fractal_ann_diagnostics.artifact_integrity import (
    digest_directory_tree,
    digest_regular_file,
)
from fractal_ann_diagnostics.execution_claim import (
    ClaimCorpusBinding,
    CorpusOutputTree,
    FailedExecuteJobReceipt,
    PhaseCorpusBinding,
    ProviderPhaseFailure,
    RunOutputAggregate,
)
from fractal_ann_diagnostics.production_corpus_run import (
    PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME,
    RUNTIME_ATTESTATION_RECEIPT_FILENAME,
    RUNTIME_INVOCATION_MARKER_FILENAME,
)
from fractal_ann_diagnostics.provider_phase_runtime import (
    LabelReleaseOutputAuthority,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA
from fractal_ann_diagnostics.suite_attempt import (
    AnalysisClosure,
    CorpusDigest,
    CorpusNamespace,
    CorpusOutputTransfer,
    CorpusRuntimePlanBinding,
    LabelCorpusClosure,
    OnlineCorpusClosure,
    OnlineSuiteClosure,
    PhaseClaimBindings,
    RunClaimBindings,
    SuiteAttemptError,
    SuiteAttestationDescriptor,
    SuiteAttestationEvidence,
    SuiteOpenBindings,
    SuiteOutputTransferReceipt,
    SuiteProviderClaims,
    SuiteStateRecord,
    TransferFileBinding,
    VerifiedSuiteAnalysisComplete,
    VerifiedSuiteLabelReleaseClaimed,
    VerifiedSuiteOnlineCompletion,
    _admit_registered_execution_artifacts,
    _expected_online_names,
    admit_label_release_claim_beacon,
    complete_label_release,
    fail_suite_attempt,
    require_verified_online_completion,
    suite_attempt_id,
    verify_suite_state,
    write_suite_attestation_evidence,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_corpora() -> tuple[str, ...]:
    return tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))


def _synthetic_transfer_members(
    corpus_id: str,
) -> tuple[tuple[TransferFileBinding, bytes], ...]:
    manifest_digest = _digest("frozen-manifest")
    filenames = {
        "action-panel": f"{manifest_digest}.action-panel.json",
        "action-panel-admission": f"{manifest_digest}.action-panel-admission.json",
        "audit-chain": f"{manifest_digest}.audit-chain.jsonl",
        "cache-preparation": f"{manifest_digest}.cache-preparation.json",
        "execution-order": f"{manifest_digest}.execution-order.json",
        "predictions": f"{manifest_digest}.predictions.json",
        "production-command-attempt": PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME,
        "runtime-attestation-receipt": RUNTIME_ATTESTATION_RECEIPT_FILENAME,
        "runtime-invocation-marker": RUNTIME_INVOCATION_MARKER_FILENAME,
        "sealed-online-attempt": f"{manifest_digest}.sealed-online-attempt.json",
        "sealed-online-result": f"{manifest_digest}.sealed-online-result-receipt.json",
    }
    members = []
    for role, filename in filenames.items():
        encoded = f"{corpus_id}:{role}\n".encode()
        members.append(
            (
                TransferFileBinding(
                    role=role,
                    relative_path=filename,
                    file_sha256=hashlib.sha256(encoded).hexdigest(),
                    byte_count=len(encoded),
                ),
                encoded,
            )
        )
    return tuple(sorted(members, key=lambda item: item[0].relative_path.encode("utf-8")))


def _descriptor() -> SuiteAttestationDescriptor:
    return SuiteAttestationDescriptor(
        expected_signer_identity=(
            "https://github.com/mhdk1602/fractal-ann-diagnostics/"
            ".github/workflows/suite-transition-attestation.yml@refs/heads/main"
        ),
        expected_oidc_issuer="https://token.actions.githubusercontent.com",
        expected_repository="mhdk1602/fractal-ann-diagnostics",
        expected_workflow=".github/workflows/suite-transition-attestation.yml",
        expected_git_ref="refs/heads/main",
        expected_signer_digest="1" * 40,
        transparency_log_identity="sigstore-public-good",
        transparency_log_uri="https://rekor.sigstore.dev/api/v1/log",
        transparency_log_public_key_sha256=_digest("rekor-key"),
        timestamp_authority_identity="sigstore-public-good-tsa",
        timestamp_authority_uri="https://timestamp.sigstore.dev/api/v1/timestamp",
        timestamp_authority_public_key_sha256=_digest("tsa-key"),
        state_service_identity="github-artifact-attestations",
        state_service_uri="https://api.github.com/repos/mhdk1602/fractal-ann-diagnostics",
        state_key_prefix="confirmatory-suite",
    )


def _open_payload(namespace: Path, descriptor: SuiteAttestationDescriptor) -> SuiteOpenBindings:
    manifest_digest = _digest("frozen-manifest")
    provisional_closure = _digest("provisional-closure")
    instantiated_closure = _digest("instantiated-closure")
    finalization_request = _digest("finalization-request")
    closure_entries = ["control"]
    closure_bindings = {
        corpus_id: {
            "corpus_id": corpus_id,
            "entries": closure_entries,
            "instantiated_closure_tree_sha256": instantiated_closure,
            "manifest_sha256": manifest_digest,
            "provisional_closure_tree_sha256": provisional_closure,
        }
        for corpus_id in FIXED_CORPORA
    }
    closure_sha256s = {
        corpus_id: hashlib.sha256(
            json.dumps(
                binding,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for corpus_id, binding in closure_bindings.items()
    }
    finalization_path = namespace.parent / f"{namespace.name}.finalization.json"
    finalization_bytes = (
        json.dumps(
            {
                "canonical_suite_namespace": str(namespace),
                "corpora": [
                    {
                        "closure_binding": closure_bindings[corpus_id],
                        "corpus_id": corpus_id,
                    }
                    for corpus_id in FIXED_CORPORA
                ],
                "finalization_request_sha256": finalization_request,
                "instantiated_closure_entries": closure_entries,
                "instantiated_closure_tree_sha256": instantiated_closure,
                "manifest_sha256": manifest_digest,
                "pre_c1_output_staging_root": str(namespace.parent / f".{namespace.name}.pre-c1"),
                "provisional_closure_tree_sha256": provisional_closure,
                "suite_attempt_id": suite_attempt_id(manifest_digest),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    if not finalization_path.exists():
        finalization_path.write_bytes(finalization_bytes)
    finalization_sha256 = hashlib.sha256(finalization_bytes).hexdigest()
    return SuiteOpenBindings(
        protocol_registration_receipt_sha256=_digest("registration"),
        protocol_registration_receipt_file_sha256=_digest("registration-file"),
        protocol_registry_record_sha256=_digest("registry-record"),
        registered_at_utc="2026-07-14T12:00:00+00:00",
        run_receipt_file_sha256=_digest("run-file"),
        run_started_at_utc="2026-07-14T12:01:00+00:00",
        code_commit="1" * 40,
        runner_image=f"ghcr.io/example/runner@sha256:{'2' * 64}",
        attestation_descriptor_sha256=descriptor.descriptor_sha256,
        production_finalization_receipt_uri=finalization_path.as_uri(),
        production_finalization_receipt_file_sha256=finalization_sha256,
        production_finalization_request_sha256=finalization_request,
        provisional_closure_tree_sha256=provisional_closure,
        instantiated_closure_tree_sha256=instantiated_closure,
        runtime_attestation_plans=tuple(
            CorpusRuntimePlanBinding(
                corpus_id=corpus_id,
                plan_sha256=_digest(f"runtime-plan:{corpus_id}"),
                file_sha256=_digest(f"runtime-plan-file:{corpus_id}"),
                production_run_closure_binding_receipt_sha256=closure_sha256s[corpus_id],
                registered_plan_instantiation_receipt_sha256=_digest(
                    f"plan-instantiation:{corpus_id}"
                ),
                registered_plan_instantiation_file_sha256=_digest(
                    f"plan-instantiation-file:{corpus_id}"
                ),
                sealed_launch_contract_uri=(
                    namespace.parent
                    / "sealed-contracts"
                    / corpus_id
                    / "sealed-launch-contract.json"
                ).as_uri(),
                sealed_launch_contract_sha256=_digest(f"sealed-launch-contract:{corpus_id}"),
                sealed_launch_contract_file_sha256=_digest(
                    f"sealed-launch-contract-file:{corpus_id}"
                ),
            )
            for corpus_id in _ordered_corpora()
        ),
        execution_artifacts=tuple(
            CorpusDigest(corpus_id, _digest(f"execution:{corpus_id}"))
            for corpus_id in _ordered_corpora()
        ),
        staging_namespaces=tuple(
            CorpusNamespace(
                corpus_id,
                (namespace.parent / f".{namespace.name}.pre-c1" / "online" / corpus_id).as_uri(),
            )
            for corpus_id in _ordered_corpora()
        ),
        output_namespaces=tuple(
            CorpusNamespace(corpus_id, (namespace / "online" / corpus_id).as_uri())
            for corpus_id in _ordered_corpora()
        ),
    )


def _online_row(namespace: Path, corpus_id: str) -> OnlineCorpusClosure:
    values = {
        name: _digest(f"{name}:{corpus_id}")
        for name in OnlineCorpusClosure.__dataclass_fields__
        if name.endswith("sha256")
    }
    transfer_files = tuple(item[0] for item in _synthetic_transfer_members(corpus_id))
    file_fields = {
        "action-panel": "action_panel_file_sha256",
        "action-panel-admission": "action_panel_admission_file_sha256",
        "audit-chain": "audit_file_sha256",
        "cache-preparation": "cache_preparation_file_sha256",
        "execution-order": "execution_order_file_sha256",
        "predictions": "prediction_file_sha256",
        "production-command-attempt": "production_command_attempt_file_sha256",
        "runtime-attestation-receipt": "runtime_attestation_receipt_file_sha256",
        "runtime-invocation-marker": "runtime_invocation_marker_file_sha256",
        "sealed-online-attempt": "attempt_file_sha256",
        "sealed-online-result": "result_file_sha256",
    }
    for binding in transfer_files:
        values[file_fields[binding.role]] = binding.file_sha256
    values["runtime_attestation_plan_sha256"] = _digest(f"runtime-plan:{corpus_id}")
    values["runtime_attestation_plan_file_sha256"] = _digest(f"runtime-plan-file:{corpus_id}")
    values["execution_artifact_sha256"] = _digest(f"execution:{corpus_id}")
    values["sealed_launch_contract_sha256"] = _digest(f"sealed-launch-contract:{corpus_id}")
    marker_sha256 = values["runtime_invocation_marker_file_sha256"]
    values["runtime_invocation_marker_sha256"] = marker_sha256
    staging_output = namespace.parent / f".{namespace.name}.pre-c1" / "online" / corpus_id
    return OnlineCorpusClosure(
        corpus_id=corpus_id,
        staging_output_uri=staging_output.as_uri(),
        output_uri=(namespace / "online" / corpus_id).as_uri(),
        sealed_launch_receipt_uri=(
            namespace.parent / "launch-evidence" / corpus_id / "sealed-launch-receipt.json"
        ).as_uri(),
        sealed_launch_copy_output_uri=staging_output.as_uri(),
        audit_record_count=4,
        transfer_files=transfer_files,
        **values,
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "bind each exact role and filename once"),
        ("duplicate", "bind each exact role and filename once"),
        ("unknown", "role is not registered"),
    ],
)
def test_online_closure_parser_requires_all_eleven_transfer_roles(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    payload = _online_row(tmp_path / "suite", FIXED_CORPORA[0]).to_dict()
    transfer_files = payload["transfer_files"]
    assert isinstance(transfer_files, list)
    if mutation == "missing":
        transfer_files.pop()
    elif mutation == "duplicate":
        assert isinstance(transfer_files[-1], dict)
        assert isinstance(transfer_files[0], dict)
        transfer_files[-1]["role"] = transfer_files[0]["role"]
    else:
        assert isinstance(transfer_files[-1], dict)
        transfer_files[-1]["role"] = "unregistered-output"

    with pytest.raises(SuiteAttemptError, match=match):
        OnlineCorpusClosure.from_dict(payload)


def _registered_execution_manifest() -> tuple[dict[str, object], dict[str, str]]:
    registered = {
        corpus_id: _digest(f"registered execution:{corpus_id}") for corpus_id in FIXED_CORPORA
    }
    manifest: dict[str, object] = {
        "production_workloads": [
            {
                "corpus_id": corpus_id,
                "spec": {"online_execution_plan_sha256": registered[corpus_id]},
            }
            for corpus_id in FIXED_CORPORA
        ]
    }
    return manifest, registered


def test_registered_execution_map_reordering_cannot_change_persisted_order() -> None:
    manifest, registered = _registered_execution_manifest()
    reordered = dict(reversed(tuple(registered.items())))

    admitted = _admit_registered_execution_artifacts(manifest, reordered)

    assert tuple(admitted) == FIXED_CORPORA
    assert admitted == registered


@pytest.mark.parametrize("mutation", ["substituted", "missing", "numeric", "type-alias"])
def test_registered_execution_map_rejects_non_c1_value(mutation: str) -> None:
    manifest, registered = _registered_execution_manifest()
    supplied: dict[str, object] = dict(registered)
    corpus_id = FIXED_CORPORA[0]
    if mutation == "substituted":
        supplied[corpus_id] = _digest("substituted execution")
    elif mutation == "missing":
        supplied.pop(corpus_id)
    elif mutation == "numeric":
        supplied[corpus_id] = 0
    else:
        alias = type("DigestAlias", (str,), {})
        supplied[corpus_id] = alias(registered[corpus_id])

    with pytest.raises(SuiteAttemptError, match="execution"):
        _admit_registered_execution_artifacts(manifest, supplied)  # type: ignore[arg-type]


def _label_row(corpus_id: str, *, release_root: Path = Path("/released")) -> LabelCorpusClosure:
    release_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt_path = release_root / f"{corpus_id}-decryption.json"
    receipt_bytes = f'{{"corpus_id":"{corpus_id}","fixture":"decryption"}}\n'.encode()
    receipt_path.write_bytes(receipt_bytes)
    plaintext_path = release_root / f"{corpus_id}-labels.json"
    plaintext_bytes = f'{{"corpus_id":"{corpus_id}","fixture":"labels"}}\n'.encode()
    plaintext_path.write_bytes(plaintext_bytes)
    return LabelCorpusClosure(
        corpus_id=corpus_id,
        decryption_receipt_uri=receipt_path.as_uri(),
        decryption_receipt_sha256=_digest(f"release:{corpus_id}"),
        decryption_receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        decryption_receipt_byte_count=len(receipt_bytes),
        plaintext_uri=plaintext_path.as_uri(),
        plaintext_sha256=hashlib.sha256(plaintext_bytes).hexdigest(),
        plaintext_byte_count=len(plaintext_bytes),
    )


def _run_output_aggregate(
    rows: tuple[OnlineCorpusClosure, ...],
    *,
    claim_state_sha256: str,
    claim_ledger_commit: str,
    provider_identity_sha256: str,
    output_aggregate_identity: str,
    execute_job_id: int = 9901,
) -> RunOutputAggregate:
    trees = tuple(
        CorpusOutputTree(
            corpus_id=row.corpus_id,
            output_namespace_uri=row.output_uri,
            tree_sha256=row.sealed_launch_output_tree_sha256,
        )
        for row in rows
    )
    aggregate_payload = {
        "claim_ledger_commit": claim_ledger_commit,
        "claim_state_sha256": claim_state_sha256,
        "corpus_trees": [row.to_dict() for row in trees],
        "derivation": "sha256-five-canonical-output-trees-v1",
        "execute_job_id": execute_job_id,
        "output_aggregate_identity": output_aggregate_identity,
        "provider_identity_sha256": provider_identity_sha256,
    }
    return RunOutputAggregate(
        claim_state_sha256=claim_state_sha256,
        claim_ledger_commit=claim_ledger_commit,
        provider_identity_sha256=provider_identity_sha256,
        execute_job_id=execute_job_id,
        output_aggregate_identity=output_aggregate_identity,
        corpus_trees=trees,
        aggregate_sha256=_canonical_digest(aggregate_payload),
    )


def _online_payload(
    namespace: Path,
    transfer: SuiteOutputTransferReceipt | None = None,
    *,
    claim_state_sha256: str | None = None,
    claim_ledger_commit: str = "4" * 40,
    provider_identity_sha256: str | None = None,
    output_aggregate_identity: str | None = None,
    execute_job_id: int = 9901,
) -> OnlineSuiteClosure:
    transfer_path = namespace.parent / f"{namespace.name}.output-transfer.json"
    rows = tuple(_online_row(namespace, corpus_id) for corpus_id in _ordered_corpora())
    if transfer is not None:
        transfer_by_corpus = {row.corpus_id: row for row in transfer.corpora}
        rows = tuple(
            replace(
                row,
                sealed_launch_output_tree_sha256=(
                    transfer_by_corpus[row.corpus_id].source_tree_sha256
                ),
            )
            for row in rows
        )
    aggregate = _run_output_aggregate(
        rows,
        claim_state_sha256=claim_state_sha256 or _digest("synthetic-claim-state"),
        claim_ledger_commit=claim_ledger_commit,
        provider_identity_sha256=(
            provider_identity_sha256 or _digest("synthetic-provider-identity")
        ),
        output_aggregate_identity=(
            output_aggregate_identity or _digest("synthetic-output-aggregate-identity")
        ),
        execute_job_id=execute_job_id,
    )
    return OnlineSuiteClosure(
        corpora=rows,
        output_transfer_receipt_uri=transfer_path.as_uri(),
        output_transfer_receipt_sha256=(
            transfer.receipt_sha256 if transfer is not None else _digest("output-transfer")
        ),
        output_transfer_receipt_file_sha256=(
            transfer.file_sha256 if transfer is not None else _digest("output-transfer-file")
        ),
        source_online_tree_sha256=(
            transfer.source_online_tree_sha256
            if transfer is not None
            else _digest("source-online-tree")
        ),
        canonical_online_tree_sha256=(
            transfer.canonical_online_tree_sha256
            if transfer is not None
            else _digest("source-online-tree")
        ),
        run_output_aggregate=aggregate,
    )


def _analysis(root: Path) -> AnalysisClosure:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    execution_path = root / "offline-execution-receipt.json"
    attempt_path = root / "attempt.json"
    receipt_path = root / "result-receipt.json"
    result_path = root / "final-result.json"
    execution_bytes = b'{"fixture":"offline-execution-receipt"}\n'
    attempt_bytes = b'{"fixture":"analysis-attempt"}\n'
    receipt_bytes = b'{"fixture":"analysis-result-receipt"}\n'
    result_bytes = b'{"fixture":"analysis-result"}\n'
    for path, encoded in (
        (execution_path, execution_bytes),
        (attempt_path, attempt_bytes),
        (receipt_path, receipt_bytes),
        (result_path, result_bytes),
    ):
        path.write_bytes(encoded)
    return AnalysisClosure(
        confirmatory_input_artifact_sha256=_digest("confirmatory-input"),
        analysis_execution_receipt_uri=execution_path.as_uri(),
        analysis_execution_receipt_sha256=_digest("offline-execution-receipt"),
        analysis_execution_receipt_file_sha256=hashlib.sha256(execution_bytes).hexdigest(),
        analysis_execution_receipt_byte_count=len(execution_bytes),
        analysis_attempt_receipt_uri=attempt_path.as_uri(),
        analysis_attempt_receipt_sha256=_digest("attempt"),
        analysis_attempt_file_sha256=hashlib.sha256(attempt_bytes).hexdigest(),
        analysis_attempt_byte_count=len(attempt_bytes),
        analysis_result_receipt_uri=receipt_path.as_uri(),
        analysis_result_receipt_sha256=_digest("result-receipt"),
        analysis_result_receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        analysis_result_receipt_byte_count=len(receipt_bytes),
        final_result_uri=result_path.as_uri(),
        final_result_artifact_sha256=_digest("final-result"),
        final_result_file_sha256=hashlib.sha256(result_bytes).hexdigest(),
        final_result_byte_count=len(result_bytes),
    )


class _Verifier:
    def __init__(self, *, invalid: str | None = None) -> None:
        self.invalid = invalid

    def verify(
        self,
        *,
        bundle: bytes,
        evidence: SuiteAttestationEvidence,
        descriptor: SuiteAttestationDescriptor,
        state_record_bytes: bytes,
    ) -> SuiteProviderClaims:
        claims = SuiteProviderClaims(
            subject_sha256=hashlib.sha256(state_record_bytes).hexdigest(),
            bundle_sha256=hashlib.sha256(bundle).hexdigest(),
            signer_identity=evidence.signer_identity,
            oidc_issuer=evidence.oidc_issuer,
            repository=evidence.repository,
            workflow=evidence.workflow,
            git_ref=evidence.git_ref,
            signer_digest=evidence.signer_digest,
            github_hosted_runner=(evidence.github_hosted_runner and self.invalid != "runner"),
            transparency_log_identity=evidence.transparency_log_identity,
            transparency_entry_id=evidence.transparency_entry_id,
            transparency_log_index=evidence.transparency_log_index,
            integrated_at_utc=evidence.integrated_at_utc,
            timestamp_authority_identity=evidence.timestamp_authority_identity,
            timestamp_token_sha256=evidence.timestamp_token_sha256,
            signed_at_utc=evidence.signed_at_utc,
            state_service_identity=evidence.state_service_identity,
            state_key=evidence.state_key,
            transition_id=evidence.transition_id,
            previous_transition_id=evidence.previous_transition_id,
            signature_verified=self.invalid != "signer",
            transparency_verified=self.invalid != "log",
            timestamp_verified=self.invalid != "time",
            exclusive_transition=self.invalid != "cas",
        )
        if self.invalid == "workflow":
            return replace(claims, workflow=".github/workflows/wrong.yml")
        return claims


def _synthetic_transfer(namespace: Path) -> SuiteOutputTransferReceipt:
    staging_online = namespace.parent / f".{namespace.name}.pre-c1" / "online"
    retained = namespace.parent / f".{namespace.name}.online-transfer"
    staging_online.mkdir(mode=0o700, parents=True)
    retained.mkdir(mode=0o700)
    corpus_rows: list[CorpusOutputTransfer] = []
    for corpus_id in _ordered_corpora():
        source = staging_online / corpus_id
        target = namespace / "online" / corpus_id
        source.mkdir(mode=0o700)
        target.mkdir(mode=0o700)
        files: list[TransferFileBinding] = []
        for binding, encoded in _synthetic_transfer_members(corpus_id):
            (source / binding.relative_path).write_bytes(encoded)
            (target / binding.relative_path).write_bytes(encoded)
            (target / binding.relative_path).chmod(0o600)
            files.append(binding)
        source_tree = digest_directory_tree(source)
        target_tree = digest_directory_tree(target)
        assert source_tree == target_tree
        corpus_rows.append(
            CorpusOutputTransfer(
                corpus_id=corpus_id,
                staging_output_uri=source.as_uri(),
                canonical_output_uri=target.as_uri(),
                source_tree_sha256=source_tree.sha256,
                canonical_tree_sha256=target_tree.sha256,
                entries=source_tree.entries,
                files=tuple(files),
            )
        )
    source_online = digest_directory_tree(staging_online)
    canonical_online = digest_directory_tree(namespace / "online")
    assert source_online == canonical_online
    empty = digest_directory_tree(retained)
    finalization_path = namespace.parent / f"{namespace.name}.finalization.json"
    receipt = SuiteOutputTransferReceipt(
        suite_attempt_id=suite_attempt_id(_digest("frozen-manifest")),
        manifest_sha256=_digest("frozen-manifest"),
        production_finalization_receipt_file_sha256=digest_regular_file(
            finalization_path,
            label="synthetic finalization receipt",
        ),
        staging_online_root_uri=staging_online.as_uri(),
        canonical_online_root_uri=(namespace / "online").as_uri(),
        retained_empty_placeholder_uri=retained.as_uri(),
        empty_placeholder_tree_sha256=empty.sha256,
        source_online_tree_sha256=source_online.sha256,
        canonical_online_tree_sha256=canonical_online.sha256,
        entries=source_online.entries,
        corpora=tuple(corpus_rows),
    )
    (namespace.parent / f"{namespace.name}.output-transfer.json").write_bytes(
        receipt.canonical_file_bytes()
    )
    return receipt


def _transition_id(sequence: int) -> str:
    return f"{sequence + 1:040x}"


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _online_failure_evidence(
    claim: SuiteStateRecord,
    *,
    evidence_root: Path,
) -> tuple[ProviderPhaseFailure, FailedExecuteJobReceipt]:
    assert isinstance(claim.payload, RunClaimBindings)
    provider = claim.payload.provider_identity
    incident = evidence_root / f"{claim.suite_attempt_id}.online-incident.json"
    incident_bytes = b'{"exit_status":1,"phase":"online"}\n'
    incident.write_bytes(incident_bytes)
    labels = tuple(
        sorted(
            ("ARM64", "macOS", provider.runner_label, "self-hosted"),
            key=lambda item: item.encode("utf-8"),
        )
    )
    failed_job = FailedExecuteJobReceipt(
        provider_identity_sha256=provider.identity_sha256,
        repository=provider.repository,
        workflow_path=provider.workflow_path,
        workflow_sha=provider.workflow_sha,
        run_head_branch=provider.run_head_branch,
        run_id=provider.run_id,
        run_attempt=provider.run_attempt,
        execute_job_id=9901,
        execute_job_name=provider.execute_job_name,
        conclusion="failure",
        runner_assigned=True,
        runner_id=provider.runner_id,
        runner_name=provider.runner_name,
        runner_group_id=provider.runner_group_id,
        runner_labels=labels,
        verified_at_utc="2026-07-14T12:04:00+00:00",
    )
    failure = ProviderPhaseFailure(
        phase="online",
        claim_state_sha256=claim.record_sha256,
        claim_ledger_commit=_transition_id(claim.sequence),
        provider_identity_sha256=provider.identity_sha256,
        failed_execute_job_receipt_sha256=failed_job.receipt_sha256,
        execute_job_id=failed_job.execute_job_id,
        phase_input_sha256=claim.payload.execution_claim.contract_sha256,
        exit_status=1,
        termination_signal=None,
        incident_uri=incident.as_uri(),
        incident_byte_count=len(incident_bytes),
        incident_file_sha256=hashlib.sha256(incident_bytes).hexdigest(),
        partial_evidence=(),
    )
    return failure, failed_job


def _run_claim_payload(opened: SuiteStateRecord) -> RunClaimBindings:
    assert isinstance(opened.payload, SuiteOpenBindings)
    base = _base_execution_claim()
    plans = {row.corpus_id: row for row in opened.payload.runtime_attestation_plans}
    staging = {row.corpus_id: row for row in opened.payload.staging_namespaces}
    canonical = {row.corpus_id: row for row in opened.payload.output_namespaces}
    corpora = tuple(
        ClaimCorpusBinding(
            corpus_id=corpus_id,
            staging_namespace_uri=staging[corpus_id].output_uri,
            canonical_namespace_uri=canonical[corpus_id].output_uri,
            runtime_plan_sha256=plans[corpus_id].plan_sha256,
            runtime_plan_file_sha256=plans[corpus_id].file_sha256,
        )
        for corpus_id in _ordered_corpora()
    )
    output_identity = _canonical_digest(
        {
            "corpora": [
                {
                    "canonical_namespace_uri": row.canonical_namespace_uri,
                    "corpus_id": row.corpus_id,
                    "staging_namespace_uri": row.staging_namespace_uri,
                }
                for row in corpora
            ],
            "derivation": "sha256-five-canonical-output-trees-v1",
            "manifest_sha256": opened.manifest_sha256,
        }
    )
    execution_claim = replace(
        base,
        beacon=replace(base.beacon, chain_genesis_unix_seconds=1_784_073_600),
        manifest_sha256=opened.manifest_sha256,
        run_receipt_sha256=opened.run_receipt_sha256,
        run_receipt_file_sha256=opened.payload.run_receipt_file_sha256,
        corpora=corpora,
        output_aggregate_identity=output_identity,
    )
    zenodo = replace(
        _zenodo(),
        published_at_utc="2026-07-14T12:01:00+00:00",
        verified_at_utc="2026-07-14T12:01:01+00:00",
    )
    return RunClaimBindings(
        opened_state_sha256=opened.record_sha256,
        execution_claim=execution_claim,
        provider_identity=_provider(execution_claim),
        zenodo_admission=zenodo,
        c1_manifest_rekor_integrated_at_utc="2026-07-14T12:00:00+00:00",
        c1_registry_rekor_integrated_at_utc="2026-07-14T12:00:01+00:00",
        workload_inputs_opened_before_claim=False,
        public_benchmark_labels_accessible=True,
        human_outcome_blindness=False,
        independent_organizational_custody=False,
    )


def _phase_claim_payload(
    predecessor: SuiteStateRecord,
    *,
    execution_claim: RunClaimBindings,
    phase: str,
    online: OnlineSuiteClosure,
    labels: tuple[LabelCorpusClosure, ...] | None = None,
) -> PhaseClaimBindings:
    base = _base_phase_claim(phase)
    online_rows = {row.corpus_id: row for row in online.corpora}
    label_rows = {} if labels is None else {row.corpus_id: row for row in labels}
    if phase == "label-release":
        release_root = Path(predecessor.namespace_uri.removeprefix("file://")).parent / "released"
        corpora = tuple(
            PhaseCorpusBinding(
                corpus_id=corpus_id,
                input_uri=f"file:///sealed/{corpus_id}.ciphertext",
                input_sha256=_digest(f"ciphertext:{corpus_id}"),
                supporting_input_uri=f"file:///sealed/{corpus_id}.encryption-receipt.json",
                supporting_input_sha256=_digest(f"encryption-receipt:{corpus_id}"),
                output_uri=(release_root / f"{corpus_id}-labels.json").as_uri(),
            )
            for corpus_id in _ordered_corpora()
        )
    else:
        corpora = tuple(
            PhaseCorpusBinding(
                corpus_id=corpus_id,
                input_uri=label_rows[corpus_id].plaintext_uri,
                input_sha256=label_rows[corpus_id].plaintext_sha256,
                supporting_input_uri=online_rows[corpus_id].output_uri,
                supporting_input_sha256=(online_rows[corpus_id].sealed_launch_output_tree_sha256),
                output_uri=f"file:///analysis/{corpus_id}.json",
            )
            for corpus_id in _ordered_corpora()
        )
    input_identity = _canonical_digest(
        {
            "corpora": [
                {
                    "corpus_id": row.corpus_id,
                    "input_sha256": row.input_sha256,
                    "input_uri": row.input_uri,
                    "supporting_input_sha256": row.supporting_input_sha256,
                    "supporting_input_uri": row.supporting_input_uri,
                }
                for row in corpora
            ],
            "manifest_sha256": predecessor.manifest_sha256,
            "phase": phase,
            "predecessor_state_sha256": predecessor.record_sha256,
        }
    )
    output_identity = _canonical_digest(
        {
            "corpora": [
                {"corpus_id": row.corpus_id, "output_uri": row.output_uri} for row in corpora
            ],
            "manifest_sha256": predecessor.manifest_sha256,
            "phase": phase,
        }
    )
    root = execution_claim.execution_claim
    is_release = phase == "label-release"
    contract = replace(
        base,
        c1_commit=root.c1_commit,
        manifest_sha256=predecessor.manifest_sha256,
        c1_provider_plan_uri=(
            root.label_release_provider_plan_uri if is_release else root.analysis_provider_plan_uri
        ),
        c1_provider_plan_sha256=(
            root.label_release_provider_plan_sha256
            if is_release
            else root.analysis_provider_plan_sha256
        ),
        run_receipt_sha256=predecessor.run_receipt_sha256,
        oci_index_digest=(root.release_oci_index_digest if is_release else root.oci_index_digest),
        oci_platform_manifest_digest=(
            root.release_oci_platform_manifest_digest
            if is_release
            else root.analysis_oci_platform_manifest_digest
        ),
        tle_binary_sha256=root.release_tle_binary_sha256 if is_release else None,
        online_execution_claim_contract_sha256=root.contract_sha256,
        predecessor_state_sha256=predecessor.record_sha256,
        predecessor_ledger_commit=_transition_id(predecessor.sequence),
        corpora=corpora,
        phase_input_aggregate_sha256=input_identity,
        phase_output_identity=output_identity,
        label_release_beacon=root.beacon if is_release else None,
    )
    return PhaseClaimBindings(
        predecessor_state_sha256=predecessor.record_sha256,
        phase_claim=contract,
        provider_identity=_phase_provider(contract),
        phase_inputs_opened_before_claim=False,
        public_benchmark_labels_accessible=True,
        human_outcome_blindness=False,
        independent_organizational_custody=False,
    )


def _state_chain(tmp_path: Path, *, through: str) -> tuple[Path, list[SuiteStateRecord]]:
    manifest_digest = _digest("frozen-manifest")
    attempt_id = suite_attempt_id(manifest_digest)
    namespace = tmp_path / f"suite-attempt-{attempt_id}"
    namespace.mkdir(mode=0o700)
    (namespace / "online").mkdir(mode=0o700)
    descriptor = _descriptor()
    (namespace / "attestation-descriptor.json").write_bytes(descriptor.canonical_bytes() + b"\n")
    opened = SuiteStateRecord(
        suite_attempt_id=attempt_id,
        manifest_sha256=manifest_digest,
        run_receipt_sha256=_digest("sealed-run"),
        namespace_uri=namespace.as_uri(),
        sequence=0,
        state="OPENED",
        previous_state_record_sha256=None,
        payload=_open_payload(namespace, descriptor),
    )
    records = [opened]
    target = {
        "OPENED": 0,
        "RUN_CLAIMED": 1,
        "ONLINE_COMPLETE": 2,
        "LABEL_RELEASE_CLAIMED": 3,
        "LABELS_RELEASED": 4,
        "ANALYSIS_CLAIMED": 5,
        "ANALYSIS_COMPLETE": 6,
    }[through]
    if target >= 1:
        records.append(
            SuiteStateRecord(
                suite_attempt_id=attempt_id,
                manifest_sha256=manifest_digest,
                run_receipt_sha256=opened.run_receipt_sha256,
                namespace_uri=namespace.as_uri(),
                sequence=1,
                state="RUN_CLAIMED",
                previous_state_record_sha256=opened.record_sha256,
                payload=_run_claim_payload(opened),
            )
        )
    if target >= 2:
        transfer = _synthetic_transfer(namespace)
        claim = records[1]
        assert isinstance(claim.payload, RunClaimBindings)
        records.append(
            SuiteStateRecord(
                suite_attempt_id=attempt_id,
                manifest_sha256=manifest_digest,
                run_receipt_sha256=_digest("sealed-run"),
                namespace_uri=namespace.as_uri(),
                sequence=2,
                state="ONLINE_COMPLETE",
                previous_state_record_sha256=records[-1].record_sha256,
                payload=_online_payload(
                    namespace,
                    transfer,
                    claim_state_sha256=claim.record_sha256,
                    claim_ledger_commit=_transition_id(claim.sequence),
                    provider_identity_sha256=(claim.payload.provider_identity.identity_sha256),
                    output_aggregate_identity=(
                        claim.payload.execution_claim.output_aggregate_identity
                    ),
                ),
            )
        )
    if target >= 3:
        claim = records[1].payload
        online = records[2].payload
        assert isinstance(claim, RunClaimBindings)
        assert isinstance(online, OnlineSuiteClosure)
        records.append(
            SuiteStateRecord(
                suite_attempt_id=attempt_id,
                manifest_sha256=manifest_digest,
                run_receipt_sha256=_digest("sealed-run"),
                namespace_uri=namespace.as_uri(),
                sequence=3,
                state="LABEL_RELEASE_CLAIMED",
                previous_state_record_sha256=records[-1].record_sha256,
                payload=_phase_claim_payload(
                    records[-1],
                    execution_claim=claim,
                    phase="label-release",
                    online=online,
                ),
            )
        )
    if target >= 4:
        release_root = namespace.parent / "released"
        records.append(
            SuiteStateRecord(
                suite_attempt_id=attempt_id,
                manifest_sha256=manifest_digest,
                run_receipt_sha256=_digest("sealed-run"),
                namespace_uri=namespace.as_uri(),
                sequence=4,
                state="LABELS_RELEASED",
                previous_state_record_sha256=records[-1].record_sha256,
                payload=tuple(
                    _label_row(corpus_id, release_root=release_root)
                    for corpus_id in _ordered_corpora()
                ),
            )
        )
    if target >= 5:
        claim = records[1].payload
        online = records[2].payload
        labels = records[4].payload
        assert isinstance(claim, RunClaimBindings)
        assert isinstance(online, OnlineSuiteClosure)
        assert isinstance(labels, tuple)
        records.append(
            SuiteStateRecord(
                suite_attempt_id=attempt_id,
                manifest_sha256=manifest_digest,
                run_receipt_sha256=_digest("sealed-run"),
                namespace_uri=namespace.as_uri(),
                sequence=5,
                state="ANALYSIS_CLAIMED",
                previous_state_record_sha256=records[-1].record_sha256,
                payload=_phase_claim_payload(
                    records[-1],
                    execution_claim=claim,
                    phase="analysis",
                    online=online,
                    labels=labels,
                ),
            )
        )
    if target >= 6:
        records.append(
            SuiteStateRecord(
                suite_attempt_id=attempt_id,
                manifest_sha256=manifest_digest,
                run_receipt_sha256=_digest("sealed-run"),
                namespace_uri=namespace.as_uri(),
                sequence=6,
                state="ANALYSIS_COMPLETE",
                previous_state_record_sha256=records[-1].record_sha256,
                payload=_analysis(namespace.parent / "analysis"),
            )
        )
    for record in records:
        (namespace / f"{record.sequence:03d}.state.json").write_bytes(
            record.canonical_bytes() + b"\n"
        )
    return namespace, records


def _attest(
    namespace: Path,
    records: list[SuiteStateRecord],
    *,
    omit_last: bool = False,
    backdate_sequence: int | None = None,
    previous_transition: str | None = None,
) -> None:
    descriptor = _descriptor()
    previous = previous_transition
    selected = records[:-1] if omit_last else records
    for record in selected:
        policy = descriptor
        if isinstance(record.payload, RunClaimBindings):
            contract = record.payload.execution_claim
            policy = replace(
                descriptor,
                expected_signer_identity=f"https://github.com/{contract.claim_workflow_ref}",
                expected_repository=contract.repository,
                expected_workflow=contract.claim_workflow_path,
                expected_git_ref="refs/tags/confirmatory-apparatus-c0",
                expected_signer_digest=contract.claim_workflow_sha,
            )
        elif isinstance(record.payload, PhaseClaimBindings):
            contract = record.payload.phase_claim
            policy = replace(
                descriptor,
                expected_signer_identity=f"https://github.com/{contract.claim_workflow_ref}",
                expected_repository=contract.repository,
                expected_workflow=contract.claim_workflow_path,
                expected_git_ref="refs/tags/confirmatory-apparatus-c0",
                expected_signer_digest=contract.claim_workflow_sha,
            )
        bundle = f"signed-bundle:{record.sequence}".encode("ascii")
        minute = record.sequence + 2
        signed_at = f"2026-07-14T12:{minute:02d}:00+00:00"
        if record.sequence == backdate_sequence:
            signed_at = "2026-07-14T11:59:00+00:00"
        transition = _transition_id(record.sequence)
        evidence = SuiteAttestationEvidence(
            suite_attempt_id=record.suite_attempt_id,
            state_sequence=record.sequence,
            state_name=record.state,
            state_record_sha256=record.record_sha256,
            descriptor_sha256=descriptor.descriptor_sha256,
            bundle_sha256=hashlib.sha256(bundle).hexdigest(),
            bundle_byte_count=len(bundle),
            signer_identity=policy.expected_signer_identity,
            oidc_issuer=policy.expected_oidc_issuer,
            repository=policy.expected_repository,
            workflow=policy.expected_workflow,
            git_ref=policy.expected_git_ref,
            signer_digest=policy.expected_signer_digest,
            github_hosted_runner=True,
            transparency_log_identity=policy.transparency_log_identity,
            transparency_entry_id=f"rekor-{record.sequence}",
            transparency_log_index=100 + record.sequence,
            integrated_at_utc=f"2026-07-14T12:{minute:02d}:30+00:00",
            timestamp_authority_identity=policy.timestamp_authority_identity,
            timestamp_token_sha256=_digest(f"timestamp:{record.sequence}"),
            signed_at_utc=signed_at,
            state_service_identity=policy.state_service_identity,
            state_key=f"{policy.state_key_prefix}/{record.suite_attempt_id}",
            transition_id=transition,
            previous_transition_id=previous,
        )
        write_suite_attestation_evidence(evidence, namespace=namespace, bundle=bundle)
        previous = transition


def test_full_chain_is_externally_attestable_only_with_final_anchor(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="ANALYSIS_COMPLETE")
    _attest(namespace, records)
    verified = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="ANALYSIS_COMPLETE",
    )
    assert isinstance(verified, VerifiedSuiteAnalysisComplete)
    assert isinstance(verified.records[2].payload, OnlineSuiteClosure)
    assert len(verified.records[2].payload.corpora) == 5


def test_provider_predecessor_mints_only_from_verified_chain_and_builds_exact_claim(
    tmp_path: Path,
) -> None:
    namespace, records = _state_chain(tmp_path, through="RUN_CLAIMED")
    target = records[1]
    assert isinstance(target.payload, RunClaimBindings)
    (namespace / "001.state.json").unlink()
    _attest(namespace, [records[0]])
    opened = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="OPENED",
    )
    live = {"current": True}

    def revalidate() -> None:
        if not live["current"]:
            raise SuiteAttemptError("provider evidence advanced")

    predecessor = suite_attempt_module._mint_verified_provider_predecessor(
        records=opened.records,
        evidences=opened.evidences,
        control_inventory_sha256=_digest("control-inventory"),
        artifact_receipt_sha256=_digest("artifact-receipt"),
        fresh_revalidator=revalidate,
    )
    candidate = suite_attempt_module.claim_online_provider_candidate(
        predecessor,
        execution_claim=target.payload.execution_claim,
        provider_identity=target.payload.provider_identity,
        zenodo_admission=target.payload.zenodo_admission,
        c1_manifest_rekor_integrated_at_utc=(target.payload.c1_manifest_rekor_integrated_at_utc),
        c1_registry_rekor_integrated_at_utc=(target.payload.c1_registry_rekor_integrated_at_utc),
    )
    assert candidate == target
    assert not (namespace / "001.state.json").exists()

    live["current"] = False
    with pytest.raises(SuiteAttemptError, match="provider evidence advanced"):
        suite_attempt_module.claim_online_provider_candidate(
            predecessor,
            execution_claim=target.payload.execution_claim,
            provider_identity=target.payload.provider_identity,
            zenodo_admission=target.payload.zenodo_admission,
            c1_manifest_rekor_integrated_at_utc=(
                target.payload.c1_manifest_rekor_integrated_at_utc
            ),
            c1_registry_rekor_integrated_at_utc=(
                target.payload.c1_registry_rekor_integrated_at_utc
            ),
        )


def _provider_token(
    verified: suite_attempt_module.VerifiedSuiteState,
) -> suite_attempt_module.VerifiedProviderPredecessor:
    return suite_attempt_module._mint_verified_provider_predecessor(
        records=verified.records,
        evidences=verified.evidences,
        control_inventory_sha256=_digest("control-inventory"),
        artifact_receipt_sha256=_digest("artifact-receipt"),
        fresh_revalidator=lambda: None,
    )


def test_provider_claim_admissions_use_exact_claimed_state_and_fresh_provider(
    tmp_path: Path,
) -> None:
    namespace, records = _state_chain(tmp_path, through="RUN_CLAIMED")
    _attest(namespace, records)
    local_run = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="RUN_CLAIMED",
    )
    provider_run = _provider_token(local_run)
    run_payload = provider_run.state.payload
    assert isinstance(run_payload, RunClaimBindings)
    run_capability = suite_attempt_module.admit_run_claim_beacon(
        provider_run,
        beacon_bytes=b'{"round":101,"randomness":"synthetic"}',
        beacon_verifier=_BeaconVerifier(),
        live_execute_job_receipt=_live_job(
            run_payload.execution_claim,
            run_payload.provider_identity,
        ),
        verified_at_utc="2026-07-15T00:06:00+00:00",
        fresh_state_revalidator=lambda: provider_run,
    )
    assert run_capability.claim_state_sha256 == provider_run.state.record_sha256

    analysis_root = tmp_path / "analysis-claim"
    analysis_root.mkdir(mode=0o700)
    analysis_namespace, analysis_records = _state_chain(
        analysis_root,
        through="ANALYSIS_CLAIMED",
    )
    _attest(analysis_namespace, analysis_records)
    local_analysis = verify_suite_state(
        analysis_namespace,
        verifier=_Verifier(),
        expected_state="ANALYSIS_CLAIMED",
    )
    provider_analysis = _provider_token(local_analysis)
    analysis_payload = provider_analysis.state.payload
    assert isinstance(analysis_payload, PhaseClaimBindings)
    analysis_capability = suite_attempt_module.admit_analysis_claim(
        provider_analysis,
        live_execute_job_receipt=_live_job(
            analysis_payload.phase_claim,
            analysis_payload.provider_identity,
        ),
        fresh_state_revalidator=lambda: provider_analysis,
    )
    assert analysis_capability.phase_claim_state_sha256 == (provider_analysis.state.record_sha256)

    opened_root = tmp_path / "wrong-state"
    opened_root.mkdir(mode=0o700)
    opened_namespace, opened_records = _state_chain(opened_root, through="OPENED")
    _attest(opened_namespace, opened_records)
    opened = verify_suite_state(
        opened_namespace,
        verifier=_Verifier(),
        expected_state="OPENED",
    )
    wrong = _provider_token(opened)
    with pytest.raises(SuiteAttemptError, match="verified ANALYSIS_CLAIMED"):
        suite_attempt_module.admit_analysis_claim(
            wrong,
            live_execute_job_receipt=_live_job(
                analysis_payload.phase_claim,
                analysis_payload.provider_identity,
            ),
            fresh_state_revalidator=lambda: wrong,
        )


def test_provider_predecessor_rejects_public_construction(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="OPENED")
    _attest(namespace, records)
    opened = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="OPENED",
    )
    with pytest.raises(SuiteAttemptError, match="can only come from provider evidence"):
        suite_attempt_module.VerifiedProviderPredecessor(
            records=opened.records,
            evidences=opened.evidences,
            control_inventory_sha256=_digest("control-inventory"),
            artifact_receipt_sha256=_digest("artifact-receipt"),
            _fresh_revalidator=lambda: None,
            _capability=object(),
        )


def test_provider_analysis_candidate_preserves_full_verified_lineage(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="ANALYSIS_CLAIMED")
    target = records[5]
    assert isinstance(target.payload, PhaseClaimBindings)
    (namespace / "005.state.json").unlink()
    _attest(namespace, records[:5])
    labels = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="LABELS_RELEASED",
    )
    predecessor = suite_attempt_module._mint_verified_provider_predecessor(
        records=labels.records,
        evidences=labels.evidences,
        control_inventory_sha256=_digest("control-inventory"),
        artifact_receipt_sha256=_digest("artifact-receipt"),
        fresh_revalidator=lambda: None,
    )
    candidate = suite_attempt_module.claim_analysis_provider_candidate(
        predecessor,
        phase_contract=target.payload.phase_claim,
        provider_identity=target.payload.provider_identity,
    )
    assert candidate == target
    assert not (namespace / "005.state.json").exists()


def test_synthetic_token_and_missing_canonical_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SuiteAttemptError, match="file verification"):
        VerifiedSuiteOnlineCompletion(  # type: ignore[call-arg]
            namespace=tmp_path,
            records=(),
            evidences=(),
            descriptor_sha256=_digest("descriptor"),
            _file_sha256s=(),
            _capability=object(),
        )
    namespace, _ = _state_chain(tmp_path, through="OPENED")
    with pytest.raises(SuiteAttemptError, match="attestation"):
        verify_suite_state(namespace, verifier=_Verifier())
    with pytest.raises(SuiteAttemptError, match="canonical files"):
        require_verified_online_completion(
            object(),
            manifest_digest=_digest("frozen-manifest"),
            corpus_id=FIXED_CORPORA[0],
            online_result_receipt_sha256=_digest("result"),
        )


def test_provider_label_claim_admits_its_live_verified_online_predecessor(
    tmp_path: Path,
) -> None:
    namespace, records = _state_chain(tmp_path, through="LABEL_RELEASE_CLAIMED")
    _attest(namespace, records)
    verified_claimed = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="LABEL_RELEASE_CLAIMED",
    )
    revalidations: list[bool] = []
    predecessor = suite_attempt_module._mint_verified_provider_predecessor(
        records=verified_claimed.records,
        evidences=verified_claimed.evidences,
        control_inventory_sha256=_digest("control-inventory"),
        artifact_receipt_sha256=_digest("artifact-receipt"),
        fresh_revalidator=lambda: revalidations.append(True),
    )
    online = records[2]
    assert isinstance(online.payload, OnlineSuiteClosure)
    corpus = online.payload.corpora[0]

    closure = require_verified_online_completion(
        predecessor,
        manifest_digest=online.manifest_sha256,
        corpus_id=corpus.corpus_id,
        online_result_receipt_sha256=corpus.result_receipt_sha256,
    )

    assert closure == corpus
    assert revalidations


def test_provider_online_gate_rejects_a_non_label_claim_tip(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="ONLINE_COMPLETE")
    _attest(namespace, records)
    verified_online = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="ONLINE_COMPLETE",
    )
    predecessor = suite_attempt_module._mint_verified_provider_predecessor(
        records=verified_online.records,
        evidences=verified_online.evidences,
        control_inventory_sha256=_digest("control-inventory"),
        artifact_receipt_sha256=_digest("artifact-receipt"),
        fresh_revalidator=lambda: None,
    )
    online = records[2]
    assert isinstance(online.payload, OnlineSuiteClosure)
    corpus = online.payload.corpora[0]

    with pytest.raises(SuiteAttemptError, match="LABEL_RELEASE_CLAIMED"):
        require_verified_online_completion(
            predecessor,
            manifest_digest=online.manifest_sha256,
            corpus_id=corpus.corpus_id,
            online_result_receipt_sha256=corpus.result_receipt_sha256,
        )


def test_online_complete_requires_exact_five_without_duplicates(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="OPENED")
    common = dict(
        suite_attempt_id=records[0].suite_attempt_id,
        manifest_sha256=records[0].manifest_sha256,
        run_receipt_sha256=records[0].run_receipt_sha256,
        namespace_uri=namespace.as_uri(),
        sequence=1,
        state="ONLINE_COMPLETE",
        previous_state_record_sha256=records[0].record_sha256,
    )
    four = tuple(_online_row(namespace, corpus_id) for corpus_id in _ordered_corpora()[:4])
    with pytest.raises(SuiteAttemptError, match="five typed"):
        SuiteStateRecord(
            payload=replace(_online_payload(namespace), corpora=four),
            **common,  # type: ignore[arg-type]
        )
    duplicate = tuple(
        _online_row(namespace, corpus_id)
        for corpus_id in (*_ordered_corpora()[:4], _ordered_corpora()[0])
    )
    with pytest.raises(SuiteAttemptError, match="each fixed corpus once"):
        SuiteStateRecord(
            payload=replace(_online_payload(namespace), corpora=duplicate),
            **common,  # type: ignore[arg-type]
        )


def test_opened_requires_five_distinct_corpus_runtime_plans(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="OPENED")
    payload = records[0].payload
    assert isinstance(payload, SuiteOpenBindings)
    with pytest.raises(SuiteAttemptError, match="five typed"):
        replace(payload, runtime_attestation_plans=payload.runtime_attestation_plans[:4])

    repeated = tuple(
        replace(row, plan_sha256=_digest("reused-runtime-plan"))
        for row in payload.runtime_attestation_plans
    )
    with pytest.raises(SuiteAttemptError, match="reuse one runtime plan digest"):
        replace(payload, runtime_attestation_plans=repeated)


def test_online_closure_rejects_reused_runtime_evidence(tmp_path: Path) -> None:
    payload = _online_payload(tmp_path)
    with pytest.raises(SuiteAttemptError, match="marker semantic and file digests"):
        replace(
            payload.corpora[0],
            runtime_invocation_marker_file_sha256=_digest("another-marker-file"),
        )

    rows = list(payload.corpora)
    rows[1] = replace(
        rows[1],
        runtime_attestation_receipt_sha256=rows[0].runtime_attestation_receipt_sha256,
    )
    with pytest.raises(
        SuiteAttemptError,
        match="repeat runtime_attestation_receipt_sha256",
    ):
        replace(payload, corpora=tuple(rows))

    rows = list(payload.corpora)
    rows[1] = replace(
        rows[1],
        production_command_attempt_sha256=rows[0].production_command_attempt_sha256,
    )
    with pytest.raises(
        SuiteAttemptError,
        match="repeat production_command_attempt_sha256",
    ):
        replace(payload, corpora=tuple(rows))


def test_production_online_closure_has_exactly_eleven_registered_files() -> None:
    manifest_digest = _digest("manifest")
    attempt_path = Path(f"/{manifest_digest}.sealed-online-attempt.json")
    result_path = Path(f"/{manifest_digest}.sealed-online-result.json")
    output_names = tuple(
        f"{manifest_digest}.{name}.json"
        for name in (
            "action-panel",
            "action-panel-admission",
            "audit-chain",
            "cache-preparation",
            "execution-order",
            "predictions",
        )
    )
    result = SimpleNamespace(outputs=tuple(SimpleNamespace(filename=name) for name in output_names))
    names = _expected_online_names(attempt_path, result_path, result)
    assert len(names) == 11
    assert {
        PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME,
        RUNTIME_ATTESTATION_RECEIPT_FILENAME,
        RUNTIME_INVOCATION_MARKER_FILENAME,
    }.issubset(names)


def test_label_transition_rehashes_each_frozen_plaintext_before_consuming_state_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, records = _state_chain(tmp_path, through="LABEL_RELEASE_CLAIMED")
    _attest(namespace, records)
    verified = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="LABEL_RELEASE_CLAIMED",
    )
    assert isinstance(verified, VerifiedSuiteLabelReleaseClaimed)
    phase_payload = verified.state.payload
    assert isinstance(phase_payload, PhaseClaimBindings)
    phase_claim = admit_label_release_claim_beacon(
        verified,
        beacon_bytes=b'{"round":101,"randomness":"synthetic"}',
        beacon_verifier=_BeaconVerifier(),
        live_execute_job_receipt=_live_job(
            phase_payload.phase_claim,
            phase_payload.provider_identity,
        ),
        verified_at_utc="2026-07-15T00:06:00+00:00",
        fresh_state_revalidator=lambda: verified,
    )
    provider_verified = _provider_token(verified)
    provider_phase_claim = admit_label_release_claim_beacon(
        provider_verified,
        beacon_bytes=b'{"round":101,"randomness":"synthetic"}',
        beacon_verifier=_BeaconVerifier(),
        live_execute_job_receipt=_live_job(
            phase_payload.phase_claim,
            phase_payload.provider_identity,
        ),
        verified_at_utc="2026-07-15T00:06:00+00:00",
        fresh_state_revalidator=lambda: provider_verified,
    )
    online = records[2].payload
    assert isinstance(online, OnlineSuiteClosure)
    online_by_corpus = {row.corpus_id: row for row in online.corpora}

    plaintext_paths: dict[str, Path] = {}
    receipt_paths: dict[str, Path] = {}
    receipts: dict[Path, SimpleNamespace] = {}
    action_authorities: dict[str, LabelReleaseOutputAuthority] = {}
    post_online_aggregate_file_sha256 = _digest("post-online-completion-aggregate")
    artifacts: list[dict[str, str]] = []
    tool_sha256 = _digest("timelock-tool")
    artifacts.append({"role": "timelock-tool", "sha256": tool_sha256})
    for corpus_id in _ordered_corpora():
        plaintext = f"labels:{corpus_id}\n".encode()
        phase_rows = {row.corpus_id: row for row in phase_payload.phase_claim.corpora}
        plaintext_path = Path(phase_rows[corpus_id].output_uri.removeprefix("file://")).resolve()
        plaintext_path.parent.mkdir(parents=True, exist_ok=True)
        plaintext_path.write_bytes(plaintext)
        plaintext_paths[corpus_id] = plaintext_path
        receipt_path = (tmp_path / f"{corpus_id}-decryption.json").resolve()
        receipt_path.write_bytes(f"receipt:{corpus_id}\n".encode())
        receipt_paths[corpus_id] = receipt_path
        plaintext_sha256 = hashlib.sha256(plaintext).hexdigest()
        ciphertext_sha256 = _digest(f"ciphertext:{corpus_id}")
        encryption_file_sha256 = _digest(f"encryption-file:{corpus_id}")
        artifacts.extend(
            (
                {
                    "corpus_id": corpus_id,
                    "role": "sealed-labels",
                    "sha256": plaintext_sha256,
                },
                {
                    "corpus_id": corpus_id,
                    "role": "sealed-label-ciphertext",
                    "sha256": ciphertext_sha256,
                },
                {
                    "corpus_id": corpus_id,
                    "role": "timelock-encryption-receipt",
                    "sha256": encryption_file_sha256,
                },
            )
        )
        receipts[receipt_path] = SimpleNamespace(
            manifest_sha256=verified.state.manifest_sha256,
            corpus_id=corpus_id,
            online_execution_result_receipt_sha256=(
                online_by_corpus[corpus_id].result_receipt_sha256
            ),
            plaintext_sha256=plaintext_sha256,
            plaintext_byte_count=len(plaintext),
            ciphertext_sha256=ciphertext_sha256,
            timelock_encryption_receipt_file_sha256=encryption_file_sha256,
            tle_binary_sha256=tool_sha256,
            post_online_completion_aggregate_file_sha256=(post_online_aggregate_file_sha256),
            label_release_claim_state_sha256=phase_claim.phase_claim_state_sha256,
            label_release_claim_ledger_commit=phase_claim.phase_claim_ledger_commit,
            label_release_phase_claim_contract_sha256=(phase_claim.contract.contract_sha256),
            label_release_phase_beacon_receipt_sha256=(
                phase_claim.phase_beacon_receipt.receipt_sha256
            ),
            label_release_live_execute_job_receipt_sha256=(
                phase_claim.live_execute_job_receipt.receipt_sha256
            ),
            label_release_provider_identity_sha256=(phase_claim.provider_identity.identity_sha256),
            receipt_sha256=_digest(f"decryption:{corpus_id}"),
        )
        action_authorities[corpus_id] = LabelReleaseOutputAuthority(
            corpus_id=corpus_id,
            post_online_completion_aggregate_file_sha256=(post_online_aggregate_file_sha256),
            label_release_claim_state_sha256=phase_claim.phase_claim_state_sha256,
            label_release_claim_ledger_commit=phase_claim.phase_claim_ledger_commit,
            label_release_phase_claim_contract_sha256=(phase_claim.contract.contract_sha256),
            label_release_phase_beacon_receipt_sha256=(
                phase_claim.phase_beacon_receipt.receipt_sha256
            ),
            label_release_live_execute_job_receipt_sha256=(
                phase_claim.live_execute_job_receipt.receipt_sha256
            ),
            label_release_provider_identity_sha256=(phase_claim.provider_identity.identity_sha256),
            label_release_phase_beacon_receipt=phase_claim.phase_beacon_receipt,
            label_release_live_execute_job_receipt=(phase_claim.live_execute_job_receipt),
        )

    manifest = {"artifacts": artifacts}
    monkeypatch.setattr(
        suite_attempt_module,
        "validate_study_manifest",
        lambda value, *, require_frozen: None,
    )
    monkeypatch.setattr(
        suite_attempt_module,
        "manifest_sha256",
        lambda value: verified.state.manifest_sha256,
    )
    monkeypatch.setattr(
        timelock_release_module,
        "load_timelock_decryption_receipt",
        lambda path: receipts[Path(path)],
    )
    completion_authority_kwargs = {
        "post_online_completion_aggregate_file_sha256": (post_online_aggregate_file_sha256),
        "label_release_authorities": action_authorities,
    }

    first = _ordered_corpora()[0]
    first_receipt_path = receipt_paths[first]
    original_receipt = receipts[first_receipt_path]
    for field in (
        "post_online_completion_aggregate_file_sha256",
        "label_release_claim_state_sha256",
        "label_release_claim_ledger_commit",
        "label_release_phase_claim_contract_sha256",
        "label_release_phase_beacon_receipt_sha256",
        "label_release_live_execute_job_receipt_sha256",
        "label_release_provider_identity_sha256",
    ):
        changed = dict(vars(original_receipt))
        changed[field] = (
            "b" * 40
            if field == "label_release_claim_ledger_commit"
            else _digest(f"changed:{field}")
        )
        receipts[first_receipt_path] = SimpleNamespace(**changed)
        with pytest.raises(SuiteAttemptError, match="decryption receipt differs"):
            complete_label_release(
                verified,
                phase_claim=phase_claim,
                manifest=manifest,
                decryption_receipt_paths=receipt_paths,
                plaintext_paths=plaintext_paths,
                **completion_authority_kwargs,
            )
    receipts[first_receipt_path] = original_receipt

    same_path_plaintexts = dict(plaintext_paths)
    same_path_plaintexts[first] = receipt_paths[first]
    with pytest.raises(SuiteAttemptError, match="pairwise-distinct real regular files"):
        complete_label_release(
            verified,
            phase_claim=phase_claim,
            manifest=manifest,
            decryption_receipt_paths=receipt_paths,
            plaintext_paths=same_path_plaintexts,
            **completion_authority_kwargs,
        )

    expected_plaintext = plaintext_paths[first].read_bytes()
    plaintext_paths[first].unlink()
    plaintext_paths[first].symlink_to(receipt_paths[first])
    with pytest.raises(SuiteAttemptError, match="pairwise-distinct real regular files"):
        complete_label_release(
            verified,
            phase_claim=phase_claim,
            manifest=manifest,
            decryption_receipt_paths=receipt_paths,
            plaintext_paths=plaintext_paths,
            **completion_authority_kwargs,
        )
    plaintext_paths[first].unlink()
    plaintext_paths[first].write_bytes(expected_plaintext)

    plaintext_paths[first].unlink()
    plaintext_paths[first].hardlink_to(receipt_paths[first])
    with pytest.raises(SuiteAttemptError, match="pairwise-distinct real regular files"):
        complete_label_release(
            verified,
            phase_claim=phase_claim,
            manifest=manifest,
            decryption_receipt_paths=receipt_paths,
            plaintext_paths=plaintext_paths,
            **completion_authority_kwargs,
        )
    plaintext_paths[first].unlink()
    plaintext_paths[first].write_bytes(expected_plaintext)

    plaintext_paths[first].write_bytes(b"x" * len(expected_plaintext))
    with pytest.raises(SuiteAttemptError, match="released plaintext differs"):
        complete_label_release(
            verified,
            phase_claim=phase_claim,
            manifest=manifest,
            decryption_receipt_paths=receipt_paths,
            plaintext_paths=plaintext_paths,
            **completion_authority_kwargs,
        )
    assert not (namespace / "004.state.json").exists()

    plaintext_paths[first].write_bytes(expected_plaintext)

    with monkeypatch.context() as mutation_patch:
        final_receipt = receipt_paths[_ordered_corpora()[-1]]

        def mutate_after_initial_verification(path: str | Path) -> object:
            receipt = receipts[Path(path)]
            if Path(path) == final_receipt:
                plaintext_paths[first].write_bytes(b"z" * len(expected_plaintext))
            return receipt

        mutation_patch.setattr(
            timelock_release_module,
            "load_timelock_decryption_receipt",
            mutate_after_initial_verification,
        )
        with pytest.raises(SuiteAttemptError, match="changed before candidate creation"):
            complete_label_release(
                provider_verified,
                phase_claim=provider_phase_claim,
                manifest=manifest,
                decryption_receipt_paths=receipt_paths,
                plaintext_paths=plaintext_paths,
                **completion_authority_kwargs,
            )
    plaintext_paths[first].write_bytes(expected_plaintext)
    assert not (namespace / "004.state.json").exists()

    import fractal_ann_diagnostics.execution_claim as execution_claim_module

    with monkeypatch.context() as renewal_patch:
        monotonic = [execution_claim_module.time.monotonic_ns()]
        renewal_patch.setattr(
            execution_claim_module.time,
            "monotonic_ns",
            lambda: monotonic[0],
        )
        renewal_calls = 0
        observation_serial = 0

        def renew_after_boundary() -> object:
            nonlocal observation_serial, renewal_calls
            renewal_calls += 1
            observation_serial += 1
            observed_at = f"2026-07-15T00:06:00.{observation_serial:06d}+00:00"
            return admit_label_release_claim_beacon(
                provider_verified,
                beacon_bytes=b'{"round":101,"randomness":"synthetic"}',
                beacon_verifier=_BeaconVerifier(),
                live_execute_job_receipt=replace(
                    provider_phase_claim.live_execute_job_receipt,
                    verified_at_utc=observed_at,
                ),
                verified_at_utc=observed_at,
                fresh_state_revalidator=lambda: provider_verified,
            )

        slow_initial = renew_after_boundary()
        renewal_calls = 0

        def slow_receipt(path: str | Path) -> object:
            receipt = receipts[Path(path)]
            monotonic[0] += 5 * 60 * 1_000_000_000 + 1
            return receipt

        renewal_patch.setattr(
            timelock_release_module,
            "load_timelock_decryption_receipt",
            slow_receipt,
        )
        slow_candidate = complete_label_release(
            provider_verified,
            phase_claim=slow_initial,
            phase_claim_factory=renew_after_boundary,
            manifest=manifest,
            decryption_receipt_paths=receipt_paths,
            plaintext_paths=plaintext_paths,
            **completion_authority_kwargs,
        )
        assert slow_candidate.state == "LABELS_RELEASED"
        assert renewal_calls == len(FIXED_CORPORA) + 1

    changed_provider = suite_attempt_module._mint_verified_provider_predecessor(
        records=provider_verified.records,
        evidences=provider_verified.evidences,
        control_inventory_sha256=provider_verified.control_inventory_sha256,
        artifact_receipt_sha256=_digest("changed-artifact-receipt"),
        fresh_revalidator=lambda: None,
    )
    changing_calls = 0

    def change_tip_before_third_corpus() -> object:
        nonlocal changing_calls
        changing_calls += 1
        fresh = changed_provider if changing_calls == 3 else provider_verified
        observed_at = f"2026-07-15T00:06:00.{changing_calls:06d}+00:00"
        return admit_label_release_claim_beacon(
            provider_verified,
            beacon_bytes=b'{"round":101,"randomness":"synthetic"}',
            beacon_verifier=_BeaconVerifier(),
            live_execute_job_receipt=replace(
                provider_phase_claim.live_execute_job_receipt,
                verified_at_utc=observed_at,
            ),
            verified_at_utc=observed_at,
            fresh_state_revalidator=lambda: fresh,
        )

    with pytest.raises(SuiteAttemptError, match="provider authority changed"):
        complete_label_release(
            provider_verified,
            phase_claim=provider_phase_claim,
            phase_claim_factory=change_tip_before_third_corpus,
            manifest=manifest,
            decryption_receipt_paths=receipt_paths,
            plaintext_paths=plaintext_paths,
            **completion_authority_kwargs,
        )
    assert changing_calls == 3
    assert not (namespace / "004.state.json").exists()

    state = complete_label_release(
        verified,
        phase_claim=phase_claim,
        manifest=manifest,
        decryption_receipt_paths=receipt_paths,
        plaintext_paths=plaintext_paths,
        **completion_authority_kwargs,
    )
    assert state.state == "LABELS_RELEASED"
    assert isinstance(state.payload, tuple)
    assert {row.plaintext_uri for row in state.payload} == {
        path.as_uri() for path in plaintext_paths.values()
    }
    provider_state = complete_label_release(
        provider_verified,
        phase_claim=provider_phase_claim,
        manifest=manifest,
        decryption_receipt_paths=receipt_paths,
        plaintext_paths=plaintext_paths,
        **completion_authority_kwargs,
    )
    assert provider_state == state


def test_analysis_evidence_rejects_same_symlink_and_hardlink_aliases(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt.json"
    receipt = tmp_path / "receipt.json"
    result = tmp_path / "result.json"
    attempt.write_text("attempt\n")
    receipt.write_text("receipt\n")
    result.write_text("result\n")
    rows = (
        ("analysis attempt", attempt),
        ("analysis receipt", receipt),
        ("analysis result", result),
    )
    suite_attempt_module._assert_distinct_regular_files(
        rows,
        label="ANALYSIS_COMPLETE evidence",
    )
    with pytest.raises(SuiteAttemptError, match="pairwise-distinct real regular files"):
        suite_attempt_module._assert_distinct_regular_files(
            (rows[0], rows[0], rows[2]),
            label="ANALYSIS_COMPLETE evidence",
        )

    result.unlink()
    result.symlink_to(attempt)
    with pytest.raises(SuiteAttemptError, match="pairwise-distinct real regular files"):
        suite_attempt_module._assert_distinct_regular_files(
            rows,
            label="ANALYSIS_COMPLETE evidence",
        )
    result.unlink()
    result.hardlink_to(attempt)
    with pytest.raises(SuiteAttemptError, match="pairwise-distinct real regular files"):
        suite_attempt_module._assert_distinct_regular_files(
            rows,
            label="ANALYSIS_COMPLETE evidence",
        )


def test_provider_analysis_completion_returns_candidate_without_local_state_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, records = _state_chain(tmp_path, through="ANALYSIS_CLAIMED")
    _attest(namespace, records)
    local = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="ANALYSIS_CLAIMED",
    )
    provider = _provider_token(local)
    payload = provider.state.payload
    assert isinstance(payload, PhaseClaimBindings)
    phase_claim = suite_attempt_module.admit_analysis_claim(
        provider,
        live_execute_job_receipt=_live_job(
            payload.phase_claim,
            payload.provider_identity,
        ),
        fresh_state_revalidator=lambda: provider,
    )

    evidence_root = tmp_path / "analysis-evidence"
    evidence_root.mkdir(mode=0o700)
    execution_path = evidence_root / "offline-execution.json"
    attempt_path = evidence_root / "attempt.json"
    receipt_path = evidence_root / "receipt.json"
    result_path = evidence_root / "result.json"
    execution_bytes = b'{"offline-execution":true}\n'
    attempt_bytes = b'{"attempt":true}\n'
    receipt_bytes = b'{"receipt":true}\n'
    result_bytes = b'{"result":true}\n'
    execution_path.write_bytes(execution_bytes)
    attempt_path.write_bytes(attempt_bytes)
    receipt_path.write_bytes(receipt_bytes)
    result_path.write_bytes(result_bytes)
    input_digest = _digest("confirmatory-input")
    attempt = SimpleNamespace(
        manifest_sha256=provider.state.manifest_sha256,
        run_receipt_sha256=provider.state.run_receipt_sha256,
        confirmatory_input_artifact_sha256=input_digest,
        receipt_sha256=_digest("analysis-attempt-receipt"),
    )
    receipt = SimpleNamespace(
        attempt_receipt_sha256=attempt.receipt_sha256,
        result_artifact_sha256=hashlib.sha256(result_bytes).hexdigest(),
        receipt_sha256=_digest("analysis-result-receipt"),
    )
    execution_sha256 = _digest("offline-execution")
    execution_file_sha256 = hashlib.sha256(execution_bytes).hexdigest()
    execution = SimpleNamespace(
        receipt_sha256=execution_sha256,
        suite_attempt_id=provider.state.suite_attempt_id,
        manifest_sha256=provider.state.manifest_sha256,
        run_receipt_sha256=provider.state.run_receipt_sha256,
        provider_state_record_sha256=provider.state.record_sha256,
        provider_ledger_commit=provider.ledger_commit,
        phase_claim_contract_sha256=phase_claim.contract.contract_sha256,
        phase_claim_state_sha256=phase_claim.phase_claim_state_sha256,
        phase_claim_ledger_commit=phase_claim.phase_claim_ledger_commit,
        provider_identity_sha256=phase_claim.provider_identity.identity_sha256,
        attempt_uri=attempt_path.as_uri(),
        attempt_receipt_sha256=attempt.receipt_sha256,
        attempt_file_sha256=hashlib.sha256(attempt_bytes).hexdigest(),
        result_receipt_uri=receipt_path.as_uri(),
        result_receipt_sha256=receipt.receipt_sha256,
        result_receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        result_uri=result_path.as_uri(),
        result_artifact_sha256=receipt.result_artifact_sha256,
        result_file_sha256=hashlib.sha256(result_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        offline_contract_module,
        "load_offline_analysis_execution_receipt",
        lambda path, **kwargs: execution,
    )
    monkeypatch.setattr(
        confirmatory_execution_module,
        "load_confirmatory_analysis_attempt_receipt",
        lambda path: attempt,
    )
    monkeypatch.setattr(
        confirmatory_execution_module,
        "load_confirmatory_analysis_result_receipt",
        lambda path: receipt,
    )
    monkeypatch.setattr(
        confirmatory_execution_module,
        "load_confirmatory_result_artifact_bytes",
        lambda path, **kwargs: result_bytes,
    )

    candidate = suite_attempt_module.complete_confirmatory_analysis(
        provider,
        phase_claim=phase_claim,
        confirmatory_input_artifact_sha256=input_digest,
        execution_receipt_path=execution_path,
        execution_receipt_sha256=execution_sha256,
        execution_receipt_file_sha256=execution_file_sha256,
        attempt_receipt_path=attempt_path,
        result_receipt_path=receipt_path,
        final_result_path=result_path,
    )
    assert candidate.state == "ANALYSIS_COMPLETE"
    assert candidate.sequence == 6
    assert not (namespace / "006.state.json").exists()


def test_verified_analysis_token_rejects_post_mint_external_mutation(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="ANALYSIS_COMPLETE")
    _attest(namespace, records)
    verified = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="ANALYSIS_COMPLETE",
    )
    closure = records[-1].payload
    assert isinstance(closure, AnalysisClosure)
    final_path = Path(closure.final_result_uri.removeprefix("file://"))
    final_path.write_bytes(b"mutated after verified token mint\n")
    with pytest.raises(SuiteAttemptError, match="changed after verification"):
        verified.assert_current()


def test_mixed_run_chain_is_rejected_before_provider_verification(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="ONLINE_COMPLETE")
    mixed = replace(records[1], run_receipt_sha256=_digest("another-run"))
    (namespace / "001.state.json").write_bytes(mixed.canonical_bytes() + b"\n")
    _attest(namespace, [records[0]])
    with pytest.raises(SuiteAttemptError, match="identity or predecessor"):
        verify_suite_state(namespace, verifier=_Verifier())


def test_online_chain_cannot_replace_a_corpus_runtime_plan(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="ONLINE_COMPLETE")
    payload = records[2].payload
    assert isinstance(payload, OnlineSuiteClosure)
    rows = list(payload.corpora)
    rows[0] = replace(
        rows[0],
        runtime_attestation_plan_sha256=_digest("substituted-runtime-plan"),
    )
    changed = replace(records[2], payload=replace(payload, corpora=tuple(rows)))
    (namespace / "002.state.json").write_bytes(changed.canonical_bytes() + b"\n")
    _attest(namespace, [records[0], records[1], changed])

    with pytest.raises(SuiteAttemptError, match="differs from RUN_CLAIMED"):
        verify_suite_state(namespace, verifier=_Verifier())


def test_backdated_provider_timestamp_is_rejected(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="ONLINE_COMPLETE")
    _attest(namespace, records, backdate_sequence=1)
    with pytest.raises(SuiteAttemptError, match="backdated"):
        verify_suite_state(namespace, verifier=_Verifier())


@pytest.mark.parametrize(
    "invalid",
    ["signer", "workflow", "runner", "log", "time", "cas"],
)
def test_invalid_signer_log_timestamp_or_cas_is_rejected(
    tmp_path: Path,
    invalid: str,
) -> None:
    namespace, records = _state_chain(tmp_path, through="OPENED")
    _attest(namespace, records)
    with pytest.raises(SuiteAttemptError, match="provider"):
        verify_suite_state(namespace, verifier=_Verifier(invalid=invalid))


def test_rerun_cannot_consume_the_same_transition_slot(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="RUN_CLAIMED")
    _attest(namespace, records)
    claimed = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="RUN_CLAIMED",
    )
    failure, failed_job = _online_failure_evidence(
        records[1],
        evidence_root=tmp_path,
    )
    fail_suite_attempt(
        claimed,
        provider_failure=failure,
        failed_execute_job_receipt=failed_job,
    )
    with pytest.raises(SuiteAttemptError, match="cannot write"):
        fail_suite_attempt(
            claimed,
            provider_failure=failure,
            failed_execute_job_receipt=failed_job,
        )


def test_analysis_complete_and_failed_are_both_terminal(tmp_path: Path) -> None:
    complete_root = tmp_path / "complete"
    complete_root.mkdir()
    complete_namespace, complete_records = _state_chain(
        complete_root,
        through="ANALYSIS_COMPLETE",
    )
    _attest(complete_namespace, complete_records)
    complete = verify_suite_state(
        complete_namespace,
        verifier=_Verifier(),
        expected_state="ANALYSIS_COMPLETE",
    )
    complete_failure, complete_failed_job = _online_failure_evidence(
        complete_records[1],
        evidence_root=complete_root,
    )
    with pytest.raises(SuiteAttemptError, match="ANALYSIS_COMPLETE is terminal"):
        fail_suite_attempt(
            complete,
            provider_failure=complete_failure,
            failed_execute_job_receipt=complete_failed_job,
        )

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed_namespace, failed_records = _state_chain(
        failed_root,
        through="RUN_CLAIMED",
    )
    _attest(failed_namespace, failed_records)
    claimed = verify_suite_state(
        failed_namespace,
        verifier=_Verifier(),
        expected_state="RUN_CLAIMED",
    )
    failure, failed_job = _online_failure_evidence(
        failed_records[1],
        evidence_root=failed_root,
    )
    failed_record = fail_suite_attempt(
        claimed,
        provider_failure=failure,
        failed_execute_job_receipt=failed_job,
    )
    _attest(
        failed_namespace,
        [failed_record],
        previous_transition=_transition_id(1),
    )
    failed = verify_suite_state(
        failed_namespace,
        verifier=_Verifier(),
        expected_state="FAILED",
    )
    with pytest.raises(SuiteAttemptError, match="FAILED is terminal"):
        fail_suite_attempt(
            failed,
            provider_failure=failure,
            failed_execute_job_receipt=failed_job,
        )


def test_missing_final_anchor_cannot_claim_analysis_complete(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="ANALYSIS_COMPLETE")
    _attest(namespace, records, omit_last=True)
    with pytest.raises(SuiteAttemptError, match="attestation"):
        verify_suite_state(
            namespace,
            verifier=_Verifier(),
            expected_state="ANALYSIS_COMPLETE",
        )


def _opened_transfer_fixture(
    tmp_path: Path,
) -> tuple[Path, object, tuple[OnlineCorpusClosure, ...], dict[str, list[Path]]]:
    namespace, records = _state_chain(tmp_path, through="RUN_CLAIMED")
    _attest(namespace, records)
    opened = verify_suite_state(
        namespace,
        verifier=_Verifier(),
        expected_state="RUN_CLAIMED",
    )
    closures = _online_payload(namespace).corpora
    files: dict[str, list[Path]] = {}
    for closure in closures:
        root = Path(closure.staging_output_uri.removeprefix("file://"))
        root.mkdir(mode=0o700, parents=True)
        files[closure.corpus_id] = []
        for binding, encoded in _synthetic_transfer_members(closure.corpus_id):
            path = root / binding.relative_path
            path.write_bytes(encoded)
            files[closure.corpus_id].append(path)
    closures = tuple(
        replace(
            closure,
            sealed_launch_output_tree_sha256=digest_directory_tree(
                Path(closure.staging_output_uri.removeprefix("file://"))
            ).sha256,
        )
        for closure in closures
    )
    return namespace, opened, closures, files


def test_staged_outputs_transfer_atomically_and_preserve_source(tmp_path: Path) -> None:
    namespace, opened, closures, _ = _opened_transfer_fixture(tmp_path)
    receipt = suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    assert digest_directory_tree(
        Path(receipt.staging_online_root_uri.removeprefix("file://"))
    ) == digest_directory_tree(namespace / "online")
    assert (
        digest_directory_tree(
            Path(receipt.retained_empty_placeholder_uri.removeprefix("file://"))
        ).entries
        == ()
    )
    assert (
        namespace.parent / f"{namespace.name}.output-transfer.json"
    ).read_bytes() == receipt.canonical_file_bytes()

    recovered = suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    assert recovered == receipt
    assert (
        namespace.parent / f"{namespace.name}.output-transfer.json"
    ).read_bytes() == receipt.canonical_file_bytes()

    state = suite_attempt_module._write_transition(
        opened,
        state="ONLINE_COMPLETE",
        payload=OnlineSuiteClosure(
            corpora=closures,
            output_transfer_receipt_uri=(
                namespace.parent / f"{namespace.name}.output-transfer.json"
            ).as_uri(),
            output_transfer_receipt_sha256=receipt.receipt_sha256,
            output_transfer_receipt_file_sha256=receipt.file_sha256,
            source_online_tree_sha256=receipt.source_online_tree_sha256,
            canonical_online_tree_sha256=receipt.canonical_online_tree_sha256,
            run_output_aggregate=_run_output_aggregate(
                closures,
                claim_state_sha256=opened.state.record_sha256,
                claim_ledger_commit=_transition_id(opened.state.sequence),
                provider_identity_sha256=(opened.state.payload.provider_identity.identity_sha256),
                output_aggregate_identity=(
                    opened.state.payload.execution_claim.output_aggregate_identity
                ),
            ),
        ),
    )
    assert state.state == "ONLINE_COMPLETE"


def test_online_transition_binds_host_launch_contract_to_opened_plan(tmp_path: Path) -> None:
    namespace, records = _state_chain(tmp_path, through="ONLINE_COMPLETE")
    _, _, online = records
    assert isinstance(online.payload, OnlineSuiteClosure)
    rows = list(online.payload.corpora)
    rows[0] = replace(
        rows[0],
        sealed_launch_contract_sha256=_digest("substituted sealed launch contract"),
    )
    hostile = replace(online, payload=replace(online.payload, corpora=tuple(rows)))
    records[2] = hostile
    (namespace / "002.state.json").write_bytes(hostile.canonical_bytes() + b"\n")
    _attest(namespace, records)
    with pytest.raises(SuiteAttemptError, match="differs from OPENED after RUN_CLAIMED"):
        verify_suite_state(namespace, verifier=_Verifier())


def test_online_transfer_binds_host_launch_output_tree(tmp_path: Path) -> None:
    _, records = _state_chain(tmp_path, through="ONLINE_COMPLETE")
    _, _, online = records
    assert isinstance(online.payload, OnlineSuiteClosure)
    rows = list(online.payload.corpora)
    rows[0] = replace(
        rows[0],
        sealed_launch_output_tree_sha256=_digest("substituted launch output tree"),
    )
    with pytest.raises(SuiteAttemptError, match="aggregate changes"):
        replace(online.payload, corpora=tuple(rows))


@pytest.mark.parametrize("mutation", ["extra", "omission"])
def test_transfer_rejects_extra_or_missing_staging_member(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, opened, closures, files = _opened_transfer_fixture(tmp_path)
    first = closures[0].corpus_id
    if mutation == "extra":
        files[first][0].parent.joinpath("unexpected.json").write_bytes(b"extra\n")
    else:
        files[first][0].unlink()
    with pytest.raises(SuiteAttemptError, match="eleven"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_transfer_rejects_symlink_or_hardlink(
    tmp_path: Path,
    link_kind: str,
) -> None:
    _, opened, closures, files = _opened_transfer_fixture(tmp_path)
    first = closures[0].corpus_id
    source = files[first][0]
    replacement = files[first][1]
    source.unlink()
    if link_kind == "symlink":
        source.symlink_to(replacement.name)
    else:
        source.hardlink_to(replacement)
    with pytest.raises(SuiteAttemptError, match="symlink|hard-linked"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)


def test_transfer_rejects_wrong_corpus_root(tmp_path: Path) -> None:
    _, opened, closures, _ = _opened_transfer_fixture(tmp_path)
    hostile = list(closures)
    hostile[0] = replace(
        hostile[0],
        staging_output_uri=closures[1].staging_output_uri,
        sealed_launch_copy_output_uri=closures[1].staging_output_uri,
    )
    with pytest.raises(SuiteAttemptError, match="wrong corpus"):
        suite_attempt_module._transfer_staged_online_outputs(opened, tuple(hostile))


def test_transfer_detects_source_mutation_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, opened, closures, files = _opened_transfer_fixture(tmp_path)
    original = suite_attempt_module._copy_transfer_file
    victim = files[closures[0].corpus_id][-1]
    mutated = False

    def mutating_copy(source: Path, target: Path, expected: TransferFileBinding) -> None:
        nonlocal mutated
        original(source, target, expected)
        if not mutated:
            victim.write_bytes(b"mutated-after-admission\n")
            mutated = True

    monkeypatch.setattr(suite_attempt_module, "_copy_transfer_file", mutating_copy)
    with pytest.raises(SuiteAttemptError, match="changed|admitted"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)


def test_transfer_rejects_mutation_between_closure_and_copy(tmp_path: Path) -> None:
    _, opened, closures, files = _opened_transfer_fixture(tmp_path)
    first = closures[0].corpus_id
    files[first][0].write_bytes(b"mutated after closure verification\n")

    with pytest.raises(SuiteAttemptError, match="verified online closure"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)


def test_transfer_rejects_undeclared_partial_candidate(tmp_path: Path) -> None:
    namespace, opened, closures, _ = _opened_transfer_fixture(tmp_path)
    partial = namespace.parent / f".{namespace.name}.online-transfer"
    partial.mkdir(mode=0o700)
    partial.joinpath("partial.json").write_bytes(b"partial\n")
    with pytest.raises(SuiteAttemptError, match="undeclared corpus member"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)


def test_transfer_resumes_an_exact_partial_file_prefix(tmp_path: Path) -> None:
    namespace, opened, closures, files = _opened_transfer_fixture(tmp_path)
    first_corpus = closures[0].corpus_id
    source = files[first_corpus][0]
    source_bytes = source.read_bytes()
    staging_before = digest_directory_tree(source.parents[1])
    candidate = namespace.parent / f".{namespace.name}.online-transfer"
    candidate_root = candidate / first_corpus
    candidate.mkdir(mode=0o700)
    candidate_root.mkdir(mode=0o700)
    partial = candidate_root / source.name
    partial.write_bytes(source_bytes[: len(source_bytes) // 2])
    partial.chmod(0o600)

    receipt = suite_attempt_module._transfer_staged_online_outputs(opened, closures)

    assert digest_directory_tree(source.parents[1]) == staging_before
    assert (namespace / "online" / first_corpus / source.name).read_bytes() == source_bytes
    assert digest_directory_tree(candidate).entries == ()
    assert (
        receipt.canonical_file_bytes()
        == (namespace.parent / f"{namespace.name}.output-transfer.json").read_bytes()
    )


def test_transfer_rejects_a_wrong_partial_file_prefix(tmp_path: Path) -> None:
    namespace, opened, closures, files = _opened_transfer_fixture(tmp_path)
    first_corpus = closures[0].corpus_id
    source = files[first_corpus][0]
    source_bytes = source.read_bytes()
    candidate = namespace.parent / f".{namespace.name}.online-transfer"
    candidate_root = candidate / first_corpus
    candidate.mkdir(mode=0o700)
    candidate_root.mkdir(mode=0o700)
    wrong_prefix = bytes([source_bytes[0] ^ 0xFF]) + source_bytes[1:4]
    partial = candidate_root / source.name
    partial.write_bytes(wrong_prefix)
    partial.chmod(0o600)

    with pytest.raises(SuiteAttemptError, match="exact source prefix"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)


@pytest.mark.parametrize("mutation", ["directory-mode", "file-mode", "file-symlink"])
def test_transfer_rejects_uncontrolled_candidate_objects(
    tmp_path: Path,
    mutation: str,
) -> None:
    namespace, opened, closures, files = _opened_transfer_fixture(tmp_path)
    first_corpus = closures[0].corpus_id
    source = files[first_corpus][0]
    candidate = namespace.parent / f".{namespace.name}.online-transfer"
    candidate.mkdir(mode=0o700)
    candidate_root = candidate / first_corpus
    candidate_root.mkdir(mode=0o700)
    partial = candidate_root / source.name
    if mutation == "directory-mode":
        candidate.chmod(0o755)
    elif mutation == "file-mode":
        partial.write_bytes(source.read_bytes()[:2])
        partial.chmod(0o644)
    else:
        partial.symlink_to(source)

    with pytest.raises(SuiteAttemptError, match="private|copy staged output|classify"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)


def test_transfer_rejects_group_writable_source_file(tmp_path: Path) -> None:
    _, opened, closures, files = _opened_transfer_fixture(tmp_path)
    files[closures[0].corpus_id][0].chmod(0o666)
    with pytest.raises(SuiteAttemptError, match="group-or-other-writable"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)


def test_transfer_lock_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    namespace, opened, closures, _ = _opened_transfer_fixture(tmp_path)
    target = tmp_path / "unrelated-lock-target"
    target.write_bytes(b"unchanged\n")
    lock_path = namespace.parent / f".{namespace.name}.online-transfer.lock"
    lock_path.symlink_to(target)

    with pytest.raises(SuiteAttemptError, match="cannot open"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    assert target.read_bytes() == b"unchanged\n"


def test_transfer_lock_detects_path_replacement_while_held(tmp_path: Path) -> None:
    namespace, opened, closures, _ = _opened_transfer_fixture(tmp_path)
    lock_path = namespace.parent / f".{namespace.name}.online-transfer.lock"
    with pytest.raises(SuiteAttemptError, match="changed while held"):
        with suite_attempt_module._output_transfer_lock(namespace):
            lock_path.unlink()
            lock_path.write_bytes(b"replacement\n")
            lock_path.chmod(0o600)

    receipt = suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    assert (
        receipt.canonical_file_bytes()
        == (namespace.parent / f"{namespace.name}.output-transfer.json").read_bytes()
    )


def test_transfer_recovers_after_crash_before_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, opened, closures, _ = _opened_transfer_fixture(tmp_path)
    staging = Path(closures[0].staging_output_uri.removeprefix("file://")).parent
    staging_before = digest_directory_tree(staging)
    candidate = namespace.parent / f".{namespace.name}.online-transfer"
    transfer_path = namespace.parent / f"{namespace.name}.output-transfer.json"
    original_exchange = suite_attempt_module._atomic_exchange_directories

    def crash_before_exchange(first: Path, second: Path) -> None:
        raise RuntimeError("injected crash before exchange")

    monkeypatch.setattr(
        suite_attempt_module,
        "_atomic_exchange_directories",
        crash_before_exchange,
    )
    with pytest.raises(RuntimeError, match="before exchange"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    assert digest_directory_tree(candidate) == staging_before
    assert digest_directory_tree(namespace / "online").entries == ()
    assert not transfer_path.exists()

    monkeypatch.setattr(
        suite_attempt_module,
        "_atomic_exchange_directories",
        original_exchange,
    )
    receipt = suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    assert digest_directory_tree(staging) == staging_before
    assert digest_directory_tree(namespace / "online") == staging_before
    assert digest_directory_tree(candidate).entries == ()
    assert transfer_path.read_bytes() == receipt.canonical_file_bytes()


def test_transfer_recovers_after_crash_after_exchange_without_reexchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, opened, closures, _ = _opened_transfer_fixture(tmp_path)
    staging = Path(closures[0].staging_output_uri.removeprefix("file://")).parent
    staging_before = digest_directory_tree(staging)
    candidate = namespace.parent / f".{namespace.name}.online-transfer"
    transfer_path = namespace.parent / f"{namespace.name}.output-transfer.json"
    original_exchange = suite_attempt_module._atomic_exchange_directories

    def crash_after_exchange(first: Path, second: Path) -> None:
        original_exchange(first, second)
        raise RuntimeError("injected crash after exchange")

    monkeypatch.setattr(
        suite_attempt_module,
        "_atomic_exchange_directories",
        crash_after_exchange,
    )
    with pytest.raises(RuntimeError, match="after exchange"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    assert digest_directory_tree(namespace / "online") == staging_before
    assert digest_directory_tree(candidate).entries == ()
    assert not transfer_path.exists()

    def reject_second_exchange(first: Path, second: Path) -> None:
        raise AssertionError(f"unexpected second exchange: {first}, {second}")

    monkeypatch.setattr(
        suite_attempt_module,
        "_atomic_exchange_directories",
        reject_second_exchange,
    )
    receipt = suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    assert digest_directory_tree(staging) == staging_before
    assert transfer_path.read_bytes() == receipt.canonical_file_bytes()


def test_transfer_recovers_exact_receipt_after_write_then_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, opened, closures, _ = _opened_transfer_fixture(tmp_path)
    transfer_path = namespace.parent / f"{namespace.name}.output-transfer.json"
    original_write = suite_attempt_module._write_once

    def write_then_crash(encoded: bytes, path: Path, *, label: str) -> None:
        original_write(encoded, path, label=label)
        if path == transfer_path:
            raise RuntimeError("injected crash after receipt write")

    monkeypatch.setattr(suite_attempt_module, "_write_once", write_then_crash)
    with pytest.raises(RuntimeError, match="after receipt write"):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    persisted_bytes = transfer_path.read_bytes()
    persisted = suite_attempt_module.load_suite_output_transfer_receipt(transfer_path)
    assert persisted_bytes == persisted.canonical_file_bytes()

    def reject_second_write(encoded: bytes, path: Path, *, label: str) -> None:
        raise AssertionError(f"unexpected second receipt write: {path} ({label})")

    monkeypatch.setattr(suite_attempt_module, "_write_once", reject_second_write)
    recovered = suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    assert recovered == persisted
    assert transfer_path.read_bytes() == persisted_bytes


def test_transfer_lock_rejects_live_duplicate_and_releases_descriptor(
    tmp_path: Path,
) -> None:
    namespace, opened, closures, _ = _opened_transfer_fixture(tmp_path)
    with suite_attempt_module._output_transfer_lock(namespace):
        with pytest.raises(SuiteAttemptError, match="live worker"):
            suite_attempt_module._transfer_staged_online_outputs(opened, closures)

    receipt = suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    lock_path = namespace.parent / f".{namespace.name}.online-transfer.lock"
    assert lock_path.stat().st_mode & 0o777 == 0o600
    assert (
        receipt.canonical_file_bytes()
        == (namespace.parent / f"{namespace.name}.output-transfer.json").read_bytes()
    )


@pytest.mark.parametrize("mutation", ["source", "canonical", "placeholder"])
def test_transfer_recovery_rejects_post_receipt_tree_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    namespace, opened, closures, _ = _opened_transfer_fixture(tmp_path)
    receipt = suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    first = receipt.corpora[0]
    if mutation == "source":
        root = Path(first.staging_output_uri.removeprefix("file://"))
        (root / first.files[0].relative_path).write_bytes(b"mutated source\n")
    elif mutation == "canonical":
        root = Path(first.canonical_output_uri.removeprefix("file://"))
        (root / first.files[0].relative_path).write_bytes(b"mutated canonical\n")
    else:
        retained = Path(receipt.retained_empty_placeholder_uri.removeprefix("file://"))
        (retained / "unexpected.json").write_bytes(b"mutated placeholder\n")

    with pytest.raises(SuiteAttemptError):
        suite_attempt_module._transfer_staged_online_outputs(opened, closures)


def test_transfer_receipt_rejects_wrong_closure_inventory(tmp_path: Path) -> None:
    _, opened, closures, _ = _opened_transfer_fixture(tmp_path)
    receipt = suite_attempt_module._transfer_staged_online_outputs(opened, closures)
    with pytest.raises(SuiteAttemptError, match="inventory"):
        replace(receipt, entries=receipt.entries[:-1])


@pytest.mark.parametrize("mutation", ["filename", "byte-count"])
def test_online_verification_rejects_closure_transfer_mapping_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    namespace, records = _state_chain(tmp_path, through="ONLINE_COMPLETE")
    payload = records[2].payload
    assert isinstance(payload, OnlineSuiteClosure)
    first = payload.corpora[0]
    bindings = list(first.transfer_files)
    target = next(item for item in bindings if item.role == "action-panel")
    position = bindings.index(target)
    bindings[position] = (
        replace(target, relative_path="changed-action-panel.json")
        if mutation == "filename"
        else replace(target, byte_count=target.byte_count + 1)
    )
    mutated_first = replace(
        first,
        transfer_files=tuple(sorted(bindings, key=lambda item: item.relative_path.encode("utf-8"))),
    )
    mutated_payload = replace(
        payload,
        corpora=(mutated_first, *payload.corpora[1:]),
    )
    records[2] = replace(records[2], payload=mutated_payload)
    (namespace / "002.state.json").write_bytes(records[2].canonical_bytes() + b"\n")
    _attest(namespace, records)

    with pytest.raises(SuiteAttemptError, match="role, filename, digest, and count"):
        verify_suite_state(
            namespace,
            verifier=_Verifier(),
            expected_state="ONLINE_COMPLETE",
        )


@pytest.mark.parametrize(
    "mutation",
    ["source", "canonical", "receipt", "retained-placeholder"],
)
def test_online_verification_rejects_post_transfer_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    namespace, records = _state_chain(tmp_path, through="ONLINE_COMPLETE")
    _attest(namespace, records)
    online = records[2].payload
    assert isinstance(online, OnlineSuiteClosure)
    transfer_path = Path(online.output_transfer_receipt_uri.removeprefix("file://"))
    transfer = suite_attempt_module.load_suite_output_transfer_receipt(transfer_path)
    row = transfer.corpora[0]
    if mutation == "source":
        root = Path(row.staging_output_uri.removeprefix("file://"))
        root.joinpath(row.files[0].relative_path).write_bytes(b"source mutation\n")
    elif mutation == "canonical":
        root = Path(row.canonical_output_uri.removeprefix("file://"))
        root.joinpath(row.files[0].relative_path).write_bytes(b"canonical mutation\n")
    elif mutation == "receipt":
        transfer_path.write_bytes(transfer_path.read_bytes() + b"\n")
    else:
        retained = Path(transfer.retained_empty_placeholder_uri.removeprefix("file://"))
        retained.joinpath("unexpected.json").write_bytes(b"unexpected\n")
    with pytest.raises(SuiteAttemptError):
        verify_suite_state(
            namespace,
            verifier=_Verifier(),
            expected_state="ONLINE_COMPLETE",
        )
