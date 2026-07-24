"""Closed derivation and publication boundary for hosted provider claims.

The GitHub publication receipt is evidence about a successful compare-and-swap.
It is never accepted as state authority.  Authority remains the freshly
verified C1 registration, provider predecessor, and exact GitHub readback made
by :func:`publish_candidate_ledger_transition`.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import unquote, urlsplit

from .execution_claim import (
    AnonymousZenodoAdmission,
    ExecutionClaimContract,
    PhaseClaimContract,
    ProviderExecutionIdentity,
    ProviderPhasePlan,
    ProviderRunnerReadinessReceipt,
    load_provider_phase_plans,
    verify_provider_runner_ready,
)
from .github_state_attestation import (
    GitHubWriteApi,
    LedgerPublicationReceipt,
    publish_candidate_ledger_transition,
)
from .provider_workflow_orchestration import (
    DerivedPhaseClaim,
    ProviderWorkflowContext,
    derive_post_online_phase_claim,
    verify_provider_execution_identity,
)
from .study import FIXED_CORPORA, VerifiedC1ProtocolRegistration, load_study_manifest
from .suite_attempt import (
    RunClaimBindings,
    SuiteStateRecord,
    VerifiedProviderPredecessor,
    claim_analysis_provider_candidate,
    claim_label_release_provider_candidate,
    claim_online_provider_candidate,
    derive_execution_claim_contract_from_provider_opened,
)

ProviderPhase = Literal["online", "label-release", "analysis"]
ProviderClaimContract = ExecutionClaimContract | PhaseClaimContract

PROVIDER_CLAIM_PREDICATE_TYPE = (
    "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/provider-claim/v1"
)
PROVIDER_CLAIM_PREDICATE_SCHEMA = "fractal-provider-claim-attestation-v1"


class ProviderClaimPublicationError(ValueError):
    """A hosted provider claim failed its closed publication policy."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ProviderClaimPublicationError(
            "provider claim evidence is not canonical JSON"
        ) from exc


def _digest(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _controlled_output_dir(value: str | Path) -> Path:
    root = Path(value)
    if (
        not root.is_absolute()
        or root.anchor != "/"
        or any(part in {"", ".", ".."} for part in root.parts[1:])
    ):
        raise ProviderClaimPublicationError("provider claim output directory is not canonical")
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ProviderClaimPublicationError(
            "provider claim output directory is not readable"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or resolved != root
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise ProviderClaimPublicationError(
            "provider claim output directory is not controlled by this identity"
        )
    return root


def _write_exclusive(path: Path, encoded: bytes, *, label: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise ProviderClaimPublicationError(f"cannot create {label} once") from exc


def _file_uri_path(uri: str, *, label: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
        raise ProviderClaimPublicationError(f"{label} is not a local file URI")
    try:
        path = Path(unquote(parsed.path, errors="strict"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProviderClaimPublicationError(f"{label} has invalid URI encoding") from exc
    if not path.is_absolute() or path.as_uri() != uri:
        raise ProviderClaimPublicationError(f"{label} is not a canonical file URI")
    return path


def _current_c1_plan(
    registration: VerifiedC1ProtocolRegistration,
    plan: ProviderPhasePlan,
) -> ProviderPhasePlan:
    plans = load_provider_phase_plans(
        registration.package_root / "study-manifest.json",
        c1_commit=registration.c1_commit,
    )
    observed = plans.get(plan.phase)
    if observed is None or observed.canonical_file_bytes() != plan.canonical_file_bytes():
        raise ProviderClaimPublicationError("provider plan differs from the current verified C1")
    return observed


def _root_online_bindings(predecessor: VerifiedProviderPredecessor) -> RunClaimBindings | None:
    matches = [
        record.payload
        for record in predecessor.records
        if record.state == "RUN_CLAIMED" and isinstance(record.payload, RunClaimBindings)
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ProviderClaimPublicationError("provider predecessor repeats RUN_CLAIMED")
    return matches[0]


def _assert_registration_evidence(
    *,
    phase: ProviderPhase,
    predecessor: VerifiedProviderPredecessor,
    zenodo_admission: AnonymousZenodoAdmission,
    c1_manifest_rekor_integrated_at_utc: str,
    c1_registry_rekor_integrated_at_utc: str,
) -> None:
    if not isinstance(zenodo_admission, AnonymousZenodoAdmission):
        raise ProviderClaimPublicationError("provider claim lacks typed Zenodo admission")
    if phase == "online":
        return
    root = _root_online_bindings(predecessor)
    if root is None:
        raise ProviderClaimPublicationError("post-online predecessor lacks RUN_CLAIMED evidence")
    if (
        root.zenodo_admission != zenodo_admission
        or root.c1_manifest_rekor_integrated_at_utc != c1_manifest_rekor_integrated_at_utc
        or root.c1_registry_rekor_integrated_at_utc != c1_registry_rekor_integrated_at_utc
    ):
        raise ProviderClaimPublicationError(
            "post-online registration evidence differs from the root provider claim"
        )


def _stable_readiness(receipt: ProviderRunnerReadinessReceipt) -> Mapping[str, object]:
    row = receipt.to_dict()
    row.pop("verified_at_utc")
    return row


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def provider_claim_predicate(
    *,
    plan: ProviderPhasePlan,
    context: ProviderWorkflowContext,
    predecessor: VerifiedProviderPredecessor,
    contract: ProviderClaimContract,
    target: SuiteStateRecord,
    provider_identity: ProviderExecutionIdentity,
    runner_readiness: ProviderRunnerReadinessReceipt,
    zenodo_admission: AnonymousZenodoAdmission,
    c1_manifest_rekor_integrated_at_utc: str,
    c1_registry_rekor_integrated_at_utc: str,
) -> Mapping[str, object]:
    """Return the closed custom predicate for one prospective provider CAS."""

    return {
        "c1": {
            "c1_commit": plan.c1_commit,
            "manifest_rekor_integrated_at_utc": c1_manifest_rekor_integrated_at_utc,
            "manifest_sha256": plan.manifest_sha256,
            "registry_rekor_integrated_at_utc": c1_registry_rekor_integrated_at_utc,
            "zenodo_admission_sha256": zenodo_admission.receipt_sha256,
        },
        "claim": {
            "contract_sha256": contract.contract_sha256,
            "phase": plan.phase,
            "state": target.state,
            "state_record_sha256": target.record_sha256,
            "state_sequence": target.sequence,
            "suite_attempt_id": target.suite_attempt_id,
        },
        "predecessor": {
            "ledger_commit": predecessor.ledger_commit,
            "state": predecessor.state.state,
            "state_record_sha256": predecessor.state.record_sha256,
            "state_sequence": predecessor.state.sequence,
        },
        "provider": {
            "claim_job_id": provider_identity.claim_job_id,
            "provider_identity_sha256": provider_identity.identity_sha256,
            "provider_plan_sha256": plan.plan_sha256,
            "run_attempt": provider_identity.run_attempt,
            "run_id": provider_identity.run_id,
            "runner_label": provider_identity.runner_label,
            "runner_readiness_sha256": runner_readiness.receipt_sha256,
            "workflow_context_sha256": context.identity_sha256,
        },
        "schema_version": PROVIDER_CLAIM_PREDICATE_SCHEMA,
    }


def _path_map(value: Mapping[str, Path]) -> Mapping[str, Path]:
    ordered = dict(sorted(value.items(), key=lambda row: row[0].encode("utf-8")))
    if set(ordered) not in (set(), set(FIXED_CORPORA)) or any(
        type(name) is not str or not isinstance(path, Path) or not path.is_absolute()
        for name, path in ordered.items()
    ):
        raise ProviderClaimPublicationError("provider claim path map is malformed")
    return MappingProxyType(ordered)


@dataclass(frozen=True)
class ProviderClaimPublicationResult:
    """Typed result of one exact hosted provider claim publication."""

    phase: ProviderPhase
    contract: ProviderClaimContract
    state: SuiteStateRecord
    provider_identity: ProviderExecutionIdentity
    runner_readiness: ProviderRunnerReadinessReceipt
    publication_receipt: LedgerPublicationReceipt
    published: bool
    publication_receipt_path: Path
    subject_path: Path
    subject_sha256: str
    predicate_path: Path
    predicate_sha256: str
    input_paths: Mapping[str, Path]
    supporting_input_paths: Mapping[str, Path]
    output_paths: Mapping[str, Path]
    runner_label: str
    suite_namespace: Path
    predicate_type: str = PROVIDER_CLAIM_PREDICATE_TYPE

    def __post_init__(self) -> None:
        if not isinstance(self.contract, (ExecutionClaimContract, PhaseClaimContract)):
            raise ProviderClaimPublicationError("provider claim result has an untyped contract")
        if not isinstance(self.state, SuiteStateRecord) or not isinstance(
            self.publication_receipt, LedgerPublicationReceipt
        ):
            raise ProviderClaimPublicationError("provider claim result has untyped state evidence")
        if self.publication_receipt.state_record_sha256 != self.state.record_sha256:
            raise ProviderClaimPublicationError("publication receipt names another claim state")
        if self.subject_sha256 != self.state.record_sha256:
            raise ProviderClaimPublicationError("claim subject digest differs from its state")
        if self.runner_label != self.provider_identity.runner_label:
            raise ProviderClaimPublicationError("claim result runner label differs")
        if self.predicate_type != PROVIDER_CLAIM_PREDICATE_TYPE:
            raise ProviderClaimPublicationError("provider claim predicate type differs")
        for name in ("publication_receipt_path", "subject_path", "predicate_path"):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                raise ProviderClaimPublicationError(f"{name} must be absolute")
            object.__setattr__(self, name, path)
        if not self.suite_namespace.is_absolute():
            raise ProviderClaimPublicationError("suite namespace must be absolute")
        for name in ("input_paths", "supporting_input_paths", "output_paths"):
            object.__setattr__(self, name, _path_map(getattr(self, name)))


def derive_and_publish_provider_claim(
    *,
    registration: VerifiedC1ProtocolRegistration,
    plan: ProviderPhasePlan,
    predecessor: VerifiedProviderPredecessor,
    zenodo_admission: AnonymousZenodoAdmission,
    context: ProviderWorkflowContext,
    c1_manifest_rekor_integrated_at_utc: str,
    c1_registry_rekor_integrated_at_utc: str,
    output_dir: str | Path,
    github_api: GitHubWriteApi,
) -> ProviderClaimPublicationResult:
    """Derive, bind, publish, and exactly read back one provider claim."""

    if not isinstance(registration, VerifiedC1ProtocolRegistration):
        raise ProviderClaimPublicationError("provider claim requires verified C1")
    if not isinstance(plan, ProviderPhasePlan):
        raise ProviderClaimPublicationError("provider claim requires a typed C1 plan")
    if not isinstance(predecessor, VerifiedProviderPredecessor):
        raise ProviderClaimPublicationError("provider claim requires a verified predecessor")
    if (
        not isinstance(context, ProviderWorkflowContext)
        or context.job != "claim"
        or context.phase != plan.phase
    ):
        raise ProviderClaimPublicationError("provider claim context differs from its phase plan")
    root = _controlled_output_dir(output_dir)
    registration.assert_current()
    predecessor.assert_current()
    current_plan = _current_c1_plan(registration, plan)
    _assert_registration_evidence(
        phase=plan.phase,
        predecessor=predecessor,
        zenodo_admission=zenodo_admission,
        c1_manifest_rekor_integrated_at_utc=c1_manifest_rekor_integrated_at_utc,
        c1_registry_rekor_integrated_at_utc=c1_registry_rekor_integrated_at_utc,
    )

    input_paths: dict[str, Path] = {}
    supporting_paths: dict[str, Path] = {}
    output_paths: dict[str, Path] = {}
    derived_phase: DerivedPhaseClaim | None = None
    if plan.phase == "online":
        contract: ProviderClaimContract = derive_execution_claim_contract_from_provider_opened(
            registration=registration,
            opened_state=predecessor.state,
        )
        if not isinstance(contract, ExecutionClaimContract):
            raise ProviderClaimPublicationError("online derivation returned another contract type")
        output_paths = {
            row.corpus_id: _file_uri_path(
                row.staging_namespace_uri,
                label=f"{row.corpus_id} online staging namespace",
            )
            for row in contract.corpora
        }
    else:
        derived_phase = derive_post_online_phase_claim(
            phase=plan.phase,
            registration=registration,
            predecessor=predecessor,
            plan=current_plan,
        )
        contract = derived_phase.contract
        input_paths = {name: Path(path) for name, path in derived_phase.input_paths}
        supporting_paths = {name: Path(path) for name, path in derived_phase.supporting_input_paths}
        output_paths = {name: Path(path) for name, path in derived_phase.output_paths}

    identity = verify_provider_execution_identity(
        context=context,
        plan=current_plan,
        api=github_api,
    )
    if isinstance(contract, ExecutionClaimContract):
        identity.matches_contract(contract)
    else:
        identity.matches_phase_contract(contract)
    readiness = verify_provider_runner_ready(
        plan=current_plan,
        api=github_api,
        verified_at_utc=_utc_now(),
    )
    if plan.phase == "online":
        target = claim_online_provider_candidate(
            predecessor,
            execution_claim=contract,
            provider_identity=identity,
            zenodo_admission=zenodo_admission,
            c1_manifest_rekor_integrated_at_utc=c1_manifest_rekor_integrated_at_utc,
            c1_registry_rekor_integrated_at_utc=c1_registry_rekor_integrated_at_utc,
        )
    elif plan.phase == "label-release":
        manifest = load_study_manifest(registration.package_root / "study-manifest.json")
        target = claim_label_release_provider_candidate(
            predecessor,
            phase_contract=contract,
            provider_identity=identity,
            manifest=manifest,
            ciphertext_paths=input_paths,
            encryption_receipt_paths=supporting_paths,
        )
    else:
        target = claim_analysis_provider_candidate(
            predecessor,
            phase_contract=contract,
            provider_identity=identity,
        )

    subject_bytes = target.canonical_bytes() + b"\n"
    predicate = provider_claim_predicate(
        plan=current_plan,
        context=context,
        predecessor=predecessor,
        contract=contract,
        target=target,
        provider_identity=identity,
        runner_readiness=readiness,
        zenodo_admission=zenodo_admission,
        c1_manifest_rekor_integrated_at_utc=c1_manifest_rekor_integrated_at_utc,
        c1_registry_rekor_integrated_at_utc=c1_registry_rekor_integrated_at_utc,
    )
    predicate_bytes = _canonical_bytes(predicate) + b"\n"
    subject_path = root / "claim-subject.json"
    predicate_path = root / "claim-predicate.json"
    publication_path = root / "ledger-publication-receipt.json"
    _write_exclusive(subject_path, subject_bytes, label="provider claim subject")
    _write_exclusive(predicate_path, predicate_bytes, label="provider claim predicate")

    # These are the last operations before the provider CAS.  The second live
    # identity and runner reads close the gap introduced by evidence writes.
    registration.assert_current()
    predecessor.assert_current()
    final_plan = _current_c1_plan(registration, current_plan)
    final_identity = verify_provider_execution_identity(
        context=context,
        plan=final_plan,
        api=github_api,
    )
    final_readiness = verify_provider_runner_ready(
        plan=final_plan,
        api=github_api,
        verified_at_utc=_utc_now(),
    )
    if final_identity != identity or _stable_readiness(final_readiness) != _stable_readiness(
        readiness
    ):
        raise ProviderClaimPublicationError("provider identity or runner changed before CAS")

    publication, published = publish_candidate_ledger_transition(
        target=target,
        expected_predecessor_commit=predecessor.ledger_commit,
        receipt_path=publication_path,
        api=github_api,
    )
    if (
        publication.previous_commit_oid != predecessor.ledger_commit
        or publication.state_record_sha256 != target.record_sha256
        or publication.state_sequence != target.sequence
        or publication.suite_attempt_id != target.suite_attempt_id
    ):
        raise ProviderClaimPublicationError("exact provider publication readback differs")

    suite_namespace = _file_uri_path(target.namespace_uri, label="suite namespace")
    return ProviderClaimPublicationResult(
        phase=plan.phase,
        contract=contract,
        state=target,
        provider_identity=identity,
        runner_readiness=readiness,
        publication_receipt=publication,
        published=published,
        publication_receipt_path=publication_path,
        subject_path=subject_path,
        subject_sha256=_digest(subject_bytes),
        predicate_path=predicate_path,
        predicate_sha256=_digest(predicate_bytes),
        input_paths=input_paths,
        supporting_input_paths=supporting_paths,
        output_paths=output_paths,
        runner_label=identity.runner_label,
        suite_namespace=suite_namespace,
    )
