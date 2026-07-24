"""Typed admission of governed executions into pre-label action panels."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np

from .audit import (
    AUDIT_SCHEMA_VERSION,
    AuditRecord,
    AuthorizationAudit,
    verify_audit_chain,
)
from .confirmatory_analysis import (
    ActionPanelAdmissionReceipt,
    ActionPanelAdmissionRecord,
    ActionPanelArtifact,
    ConfirmatoryAnalysisError,
    PreLabelActionRow,
)
from .controller import ControllerDecision, GovernedResult
from .label_separation import (
    OnlineTrial,
    sealed_run_receipt_sha256,
)
from .policy import PolicyDecision
from .scalable_execution import execution_artifact_sha256, execution_document_count
from .study import SealedRunReceipt

FailureCode = Literal[
    "backend-error",
    "backend-timeout",
    "invalid-result",
    "resource-exhausted",
    "runner-interruption",
]
REGISTERED_FAILURE_CODES = frozenset(
    {
        "backend-error",
        "backend-timeout",
        "invalid-result",
        "resource-exhausted",
        "runner-interruption",
    }
)


def _same_number(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def _canonical_sha256(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConfirmatoryAnalysisError(
            "admission evidence must contain canonical finite JSON values"
        ) from exc
    return sha256(encoded).hexdigest()


def _controller_decision_sha256(decision: ControllerDecision) -> str:
    return _canonical_sha256(
        {
            "action": decision.action,
            "policy_version": decision.policy_version,
            "reasons": list(decision.reasons),
            "risk_score": decision.risk_score,
        }
    )


def _authorization_fields(decision: PolicyDecision) -> dict[str, object]:
    mask = np.asarray(decision.authorized_mask, dtype=bool)
    mask_sha256 = sha256(mask.tobytes(order="C")).hexdigest()
    fields: dict[str, object] = {
        "available": bool(decision.available),
        "decision_id": decision.decision_id,
        "document_universe_sha256": decision.document_universe_sha256,
        "environment_sha256": decision.environment_sha256,
        "mask_sha256": mask_sha256,
        "mask_size": int(mask.size),
        "policy_version": decision.policy_version,
        "request_sha256": decision.request_sha256,
    }
    fields["decision_sha256"] = _canonical_sha256(fields)
    return fields


def _failure_timing_sha256(
    *,
    trial: OnlineTrial,
    decision: ControllerDecision,
    authorization_decision_sha256: str,
    failure_code: FailureCode,
    started_monotonic_ns: int,
    finished_monotonic_ns: int,
    runner_identity: str,
) -> str:
    return _canonical_sha256(
        {
            "action": decision.action,
            "authorization_decision_sha256": authorization_decision_sha256,
            "controller_decision_sha256": _controller_decision_sha256(decision),
            "failure_code": failure_code,
            "family_key": trial.family_key,
            "finished_monotonic_ns": finished_monotonic_ns,
            "runner_identity": runner_identity,
            "schema_version": "fractal-runner-failure-timing-v1",
            "started_monotonic_ns": started_monotonic_ns,
            "trial_key": trial.trial_key,
        }
    )


def _authorization_matches(
    observed: AuthorizationAudit | None,
    decision: PolicyDecision | None,
) -> bool:
    if decision is None:
        return observed is None
    if observed is None:
        return False
    mask = np.asarray(decision.authorized_mask, dtype=bool)
    return (
        observed.decision_id == decision.decision_id
        and observed.action == decision.action
        and observed.policy_revision == decision.policy_version
        and observed.mask_sha256 == sha256(mask.tobytes(order="C")).hexdigest()
        and observed.mask_size == int(mask.size)
        and observed.available is bool(decision.available)
        and observed.environment_sha256 == decision.environment_sha256
        and observed.document_universe_sha256 == decision.document_universe_sha256
        and observed.request_nonce == decision.request_nonce
        and observed.request_sha256 == decision.request_sha256
    )


@dataclass(frozen=True)
class GovernedActionExecution:
    """One actual governed action paired with its self-hashed audit record.

    This type admits completed retrievals and governed abstentions. A backend
    failure has no ``AuditRecord`` in the current audit schema and therefore
    cannot claim this admission path.
    """

    trial: OnlineTrial
    result: GovernedResult
    audit_record: AuditRecord
    feature_values: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trial, OnlineTrial):
            raise ConfirmatoryAnalysisError("trial must be an OnlineTrial")
        if not isinstance(self.result, GovernedResult):
            raise ConfirmatoryAnalysisError("result must be a GovernedResult")
        if not isinstance(self.audit_record, AuditRecord):
            raise ConfirmatoryAnalysisError("audit_record must be an AuditRecord")
        record = self.audit_record
        result = self.result
        if record.schema_version != AUDIT_SCHEMA_VERSION:
            raise ConfirmatoryAnalysisError("audit record schema is not registered")
        if record.record_sha256 != record.computed_record_sha256():
            raise ConfirmatoryAnalysisError("audit record self-hash is invalid")
        if record.trial_sha256 != self.trial.trial_key:
            raise ConfirmatoryAnalysisError("audit record belongs to another trial")
        if record.controller_action != result.decision.action:
            raise ConfirmatoryAnalysisError("audit and governed-result actions differ")
        if record.controller_policy_revision != result.decision.policy_version:
            raise ConfirmatoryAnalysisError("audit and governed-result policy revisions differ")
        if record.controller_reasons != result.decision.reasons or not _same_number(
            record.controller_risk_score,
            result.decision.risk_score,
        ):
            raise ConfirmatoryAnalysisError("audit and governed-result controller decisions differ")
        if not _same_number(
            record.total_online_latency_ms,
            result.total_online_latency_ms,
        ):
            raise ConfirmatoryAnalysisError("audit and governed-result request latencies differ")
        if not _authorization_matches(
            record.initial_authorization,
            result.initial_authorization,
        ) or not _authorization_matches(
            record.final_authorization,
            result.final_authorization,
        ):
            raise ConfirmatoryAnalysisError(
                "audit and governed-result authorization records differ"
            )

        abstained = result.decision.action == "abstain"
        if record.abstained is not abstained:
            raise ConfirmatoryAnalysisError("audit and governed-result execution states differ")
        if abstained:
            if result.search is not None or record.returned_evidence:
                raise ConfirmatoryAnalysisError(
                    "a governed abstention cannot contain retrieved documents"
                )
        else:
            if result.search is None or result.final_authorization is None:
                raise ConfirmatoryAnalysisError(
                    "a completed action needs search and final authorization records"
                )
            if result.search.strategy != result.decision.action:
                raise ConfirmatoryAnalysisError(
                    "search strategy does not match the executed action"
                )
            returned = tuple(int(value) for value in result.search.ids)
            audited = tuple(item.document_id for item in record.returned_evidence)
            if audited != returned:
                raise ConfirmatoryAnalysisError(
                    "audit and governed-result returned document IDs differ"
                )
            if record.search_strategy != result.search.strategy:
                raise ConfirmatoryAnalysisError(
                    "audit and governed-result search strategies differ"
                )
            if record.search_work is None or not _same_number(
                record.search_work.latency_ms,
                result.search.latency_ms,
            ):
                raise ConfirmatoryAnalysisError(
                    "audit and governed-result search measurements differ"
                )

        features = None if self.feature_values is None else tuple(self.feature_values)
        object.__setattr__(self, "feature_values", features)

    @property
    def returned_document_ids(self) -> tuple[int, ...]:
        """Return IDs from the governed search result, never from caller scalars."""

        if self.result.search is None:
            return ()
        return tuple(int(value) for value in self.result.search.ids)

    @property
    def entitlement_violations(self) -> int:
        """Count emitted IDs denied by the recorded final policy decision."""

        if not self.returned_document_ids:
            return 0
        authorization = self.result.final_authorization
        if authorization is None:
            raise ConfirmatoryAnalysisError(
                "returned documents have no final authorization decision"
            )
        mask = np.asarray(authorization.authorized_mask, dtype=bool)
        return sum(
            document_id < 0 or document_id >= mask.size or not bool(mask[document_id])
            for document_id in self.returned_document_ids
        )

    def to_prelabel_row(
        self,
        *,
        action_order: int,
        execution_position: int,
        controller_selected: bool,
    ) -> PreLabelActionRow:
        """Derive one serializable row from the admitted execution pair."""

        abstained = self.result.decision.action == "abstain"
        return PreLabelActionRow(
            trial_key=self.trial.trial_key,
            family_key=self.trial.family_key,
            action=self.result.decision.action,
            action_order=action_order,
            execution_position=execution_position,
            audit_record_sha256=self.audit_record.record_sha256,
            execution_state="abstained" if abstained else "completed",
            failure_state="governed-abstention" if abstained else None,
            controller_selected=controller_selected,
            request_latency_ms=self.audit_record.total_online_latency_ms,
            entitlement_violations=self.entitlement_violations,
            returned_document_ids=self.returned_document_ids,
            feature_values=self.feature_values,
        )


@dataclass(frozen=True)
class FailedActionExecution:
    """One runner-timed attempted action with no emitted retrieval or audit claim."""

    trial: OnlineTrial
    decision: ControllerDecision
    authorization: PolicyDecision
    failure_code: FailureCode
    started_monotonic_ns: int
    finished_monotonic_ns: int
    runner_identity: str
    feature_values: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trial, OnlineTrial):
            raise ConfirmatoryAnalysisError("trial must be an OnlineTrial")
        if not isinstance(self.decision, ControllerDecision):
            raise ConfirmatoryAnalysisError("failed action decision must be a ControllerDecision")
        if not isinstance(self.authorization, PolicyDecision):
            raise ConfirmatoryAnalysisError("failed action authorization must be a PolicyDecision")
        if self.decision.policy_version != self.authorization.policy_version:
            raise ConfirmatoryAnalysisError(
                "failed action decision and authorization policy revisions differ"
            )
        if not self.authorization.available:
            raise ConfirmatoryAnalysisError(
                "an unavailable policy decision must abstain before backend execution"
            )
        if self.failure_code not in REGISTERED_FAILURE_CODES:
            raise ConfirmatoryAnalysisError("failure_code is not registered")
        for name, value in (
            ("started_monotonic_ns", self.started_monotonic_ns),
            ("finished_monotonic_ns", self.finished_monotonic_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfirmatoryAnalysisError(f"{name} must be a non-negative integer")
        if self.finished_monotonic_ns <= self.started_monotonic_ns:
            raise ConfirmatoryAnalysisError("failure timing must have positive monotonic duration")
        if (
            not isinstance(self.runner_identity, str)
            or not self.runner_identity
            or self.runner_identity != self.runner_identity.strip()
            or any(
                ord(character) < 32 or ord(character) == 127 for character in self.runner_identity
            )
        ):
            raise ConfirmatoryAnalysisError("runner_identity must be a canonical non-empty string")
        features = None if self.feature_values is None else tuple(self.feature_values)
        object.__setattr__(self, "feature_values", features)

    @property
    def action(self) -> str:
        return self.decision.action

    @property
    def request_latency_ms(self) -> float:
        return (self.finished_monotonic_ns - self.started_monotonic_ns) / 1_000_000.0

    @property
    def timing_receipt_sha256(self) -> str:
        authorization = _authorization_fields(self.authorization)
        return _failure_timing_sha256(
            trial=self.trial,
            decision=self.decision,
            authorization_decision_sha256=str(authorization["decision_sha256"]),
            failure_code=self.failure_code,
            started_monotonic_ns=self.started_monotonic_ns,
            finished_monotonic_ns=self.finished_monotonic_ns,
            runner_identity=self.runner_identity,
        )

    def to_prelabel_row(
        self,
        *,
        action_order: int,
        execution_position: int,
        controller_selected: bool,
    ) -> PreLabelActionRow:
        """Emit an explicit intention-to-treat failure row without invented output."""

        return PreLabelActionRow(
            trial_key=self.trial.trial_key,
            family_key=self.trial.family_key,
            action=self.action,
            action_order=action_order,
            execution_position=execution_position,
            audit_record_sha256=None,
            execution_state="failed",
            failure_state=self.failure_code,
            controller_selected=controller_selected,
            request_latency_ms=self.request_latency_ms,
            entitlement_violations=0,
            returned_document_ids=(),
            feature_values=self.feature_values,
        )


@dataclass(frozen=True)
class AdmittedActionPanel:
    """One pre-label panel and the detached receipt that admits its bytes."""

    panel: ActionPanelArtifact
    admission_receipt: ActionPanelAdmissionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.panel, ActionPanelArtifact):
            raise ConfirmatoryAnalysisError("panel must be an ActionPanelArtifact")
        if not isinstance(self.admission_receipt, ActionPanelAdmissionReceipt):
            raise ConfirmatoryAnalysisError(
                "admission_receipt must be an ActionPanelAdmissionReceipt"
            )
        self.admission_receipt.validate_panel(self.panel)


def _admission_record(
    item: GovernedActionExecution | FailedActionExecution,
    *,
    action_order: int,
    execution_position: int,
    controller_selected: bool,
) -> ActionPanelAdmissionRecord:
    if isinstance(item, GovernedActionExecution):
        decision = item.result.decision
        authorization = item.result.final_authorization or item.result.initial_authorization
        record = item.audit_record
        execution_state: Literal["completed", "failed", "abstained"] = (
            "abstained" if decision.action == "abstain" else "completed"
        )
        failure_code = None
        failure_started = None
        failure_finished = None
        failure_runner = None
        failure_timing = None
        audit_sequence = record.sequence
        audit_previous = record.previous_record_sha256
        audit_sha256 = record.record_sha256
    else:
        decision = item.decision
        authorization = item.authorization
        execution_state = "failed"
        failure_code = item.failure_code
        failure_started = item.started_monotonic_ns
        failure_finished = item.finished_monotonic_ns
        failure_runner = item.runner_identity
        failure_timing = item.timing_receipt_sha256
        audit_sequence = None
        audit_previous = None
        audit_sha256 = None
    authorization_fields = _authorization_fields(authorization)
    return ActionPanelAdmissionRecord(
        trial_key=item.trial.trial_key,
        family_key=item.trial.family_key,
        action=decision.action,
        action_order=action_order,
        execution_position=execution_position,
        controller_selected=controller_selected,
        execution_state=execution_state,
        controller_risk_score=decision.risk_score,
        controller_reasons=decision.reasons,
        controller_policy_version=decision.policy_version,
        controller_decision_sha256=_controller_decision_sha256(decision),
        authorization_decision_id=str(authorization_fields["decision_id"]),
        authorization_request_sha256=str(authorization_fields["request_sha256"]),
        authorization_mask_sha256=str(authorization_fields["mask_sha256"]),
        authorization_mask_size=int(authorization_fields["mask_size"]),
        authorization_decision_sha256=str(authorization_fields["decision_sha256"]),
        policy_available=bool(authorization_fields["available"]),
        environment_sha256=str(authorization_fields["environment_sha256"]),
        document_universe_sha256=str(authorization_fields["document_universe_sha256"]),
        audit_sequence=audit_sequence,
        audit_previous_record_sha256=audit_previous,
        audit_record_sha256=audit_sha256,
        failure_code=failure_code,
        failure_started_monotonic_ns=failure_started,
        failure_finished_monotonic_ns=failure_finished,
        failure_runner_identity=failure_runner,
        failure_timing_receipt_sha256=failure_timing,
    )


def action_panel_from_governed_executions(
    *,
    execution: object,
    run_receipt: SealedRunReceipt,
    governed_executions: Iterable[GovernedActionExecution],
    failed_executions: Iterable[FailedActionExecution] = (),
    selected_decisions: Mapping[str, ControllerDecision],
    action_set: Sequence[str],
    execution_orders: Mapping[str, Sequence[str]],
    expected_audit_head_sha256: str,
    query_partition_audit_sha256: str,
    partition_label: Literal["primary", "reserve"],
) -> AdmittedActionPanel:
    """Build a complete panel and its provenance receipt from typed executions."""

    if not isinstance(run_receipt, SealedRunReceipt):
        raise ConfirmatoryAnalysisError("run_receipt must be a SealedRunReceipt")
    try:
        stage = execution.stage  # type: ignore[attr-defined]
        corpus = execution.corpus  # type: ignore[attr-defined]
        execution_trials = tuple(execution.trials)  # type: ignore[attr-defined]
        document_count = execution_document_count(execution)
        execution_sha256 = execution_artifact_sha256(execution)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ConfirmatoryAnalysisError(
            "execution lacks the admitted online artifact interface"
        ) from exc
    if stage != "sealed":
        raise ConfirmatoryAnalysisError("action-panel execution stage must be sealed")
    actions = tuple(action_set)
    if not actions or len(actions) != len(set(actions)):
        raise ConfirmatoryAnalysisError("action_set must be non-empty and unique")
    expected_trials = {trial.trial_key: trial for trial in execution_trials}
    if len(expected_trials) != len(execution_trials):
        raise ConfirmatoryAnalysisError("execution contains duplicate trial keys")
    orders = {trial_key: tuple(order) for trial_key, order in execution_orders.items()}
    if set(orders) != set(expected_trials):
        raise ConfirmatoryAnalysisError("execution_orders must cover the exact execution trial set")
    for trial_key, order in orders.items():
        if len(order) != len(actions) or set(order) != set(actions):
            raise ConfirmatoryAnalysisError(
                f"execution order for trial {trial_key!r} is not one complete permutation"
            )
    selections = dict(selected_decisions)
    if set(selections) != set(expected_trials):
        raise ConfirmatoryAnalysisError(
            "selected decisions must cover the exact execution trial set"
        )
    if not all(isinstance(value, ControllerDecision) for value in selections.values()):
        raise ConfirmatoryAnalysisError("selected decisions must contain ControllerDecision values")

    governed = tuple(governed_executions)
    failed = tuple(failed_executions)
    if not all(isinstance(item, GovernedActionExecution) for item in governed):
        raise ConfirmatoryAnalysisError(
            "governed_executions must contain GovernedActionExecution values"
        )
    if not all(isinstance(item, FailedActionExecution) for item in failed):
        raise ConfirmatoryAnalysisError(
            "failed_executions must contain FailedActionExecution values"
        )
    admitted: tuple[GovernedActionExecution | FailedActionExecution, ...] = governed + failed
    if not admitted:
        raise ConfirmatoryAnalysisError("action executions must not be empty")

    record_hashes = [item.audit_record.record_sha256 for item in governed]
    if len(record_hashes) != len(set(record_hashes)):
        raise ConfirmatoryAnalysisError(
            "one audit record cannot be reused across action executions"
        )
    ordered_records = tuple(
        sorted((item.audit_record for item in governed), key=lambda row: row.sequence)
    )
    if not ordered_records:
        raise ConfirmatoryAnalysisError("action-panel admission requires governed audit records")
    try:
        verification = verify_audit_chain(
            ordered_records,
            expected_head_sha256=expected_audit_head_sha256,
            expected_length=len(ordered_records),
        )
    except ValueError as exc:
        raise ConfirmatoryAnalysisError("expected audit-chain head is invalid") from exc
    if not verification.valid:
        raise ConfirmatoryAnalysisError(
            "governed action audit chain is invalid: " + "; ".join(verification.errors)
        )

    keyed: dict[
        tuple[str, str],
        GovernedActionExecution | FailedActionExecution,
    ] = {}
    for item in admitted:
        trial = expected_trials.get(item.trial.trial_key)
        if trial is None:
            raise ConfirmatoryAnalysisError(
                "action execution belongs to another execution artifact"
            )
        try:
            expected_family_key = trial.family_key
        except AttributeError as exc:
            raise ConfirmatoryAnalysisError("execution trial lacks a family-key binding") from exc
        if (
            item.trial.family_key != expected_family_key
            or item.trial.corpus != corpus
            or item.trial.stage != stage
        ):
            raise ConfirmatoryAnalysisError(
                "action execution belongs to another execution artifact"
            )
        if isinstance(trial, OnlineTrial) and item.trial != trial:
            raise ConfirmatoryAnalysisError(
                "inline execution trial content differs from its artifact"
            )
        if isinstance(item, FailedActionExecution):
            if item.runner_identity != run_receipt.runner_identity:
                raise ConfirmatoryAnalysisError(
                    "failed action timing receipt belongs to another runner"
                )
            if item.authorization.authorized_mask.size != document_count:
                raise ConfirmatoryAnalysisError(
                    "failed action authorization mask does not match document_count"
                )
        else:
            authorization = item.result.final_authorization or item.result.initial_authorization
            if authorization.authorized_mask.size != document_count:
                raise ConfirmatoryAnalysisError(
                    "governed action authorization mask does not match document_count"
                )
        action = (
            item.result.decision.action
            if isinstance(item, GovernedActionExecution)
            else item.action
        )
        key = (item.trial.trial_key, action)
        if key in keyed:
            raise ConfirmatoryAnalysisError(
                "action executions contain a duplicate trial-action pair"
            )
        keyed[key] = item

    rows: list[PreLabelActionRow] = []
    admission_records: list[ActionPanelAdmissionRecord] = []
    for trial_key, trial in expected_trials.items():
        observed_actions = {
            action for candidate_trial, action in keyed if candidate_trial == trial_key
        }
        if observed_actions != set(actions):
            raise ConfirmatoryAnalysisError(
                f"trial {trial_key!r} lacks the complete admitted action set"
            )
        selection = selections[trial_key]
        if selection.action not in actions:
            raise ConfirmatoryAnalysisError(
                "selected controller action is outside the registered action set"
            )
        trial_executions = [keyed[(trial_key, action)] for action in actions]
        attempted_decisions = [
            item.result.decision if isinstance(item, GovernedActionExecution) else item.decision
            for item in trial_executions
        ]
        if any(
            decision.policy_version != selection.policy_version for decision in attempted_decisions
        ):
            raise ConfirmatoryAnalysisError("selected and counterfactual policy revisions differ")
        selected = keyed[(trial_key, selection.action)]
        selected_decision = (
            selected.result.decision
            if isinstance(selected, GovernedActionExecution)
            else selected.decision
        )
        if selected_decision != selection:
            raise ConfirmatoryAnalysisError(
                "selected action is not bound to the frozen controller decision"
            )

        authorization_decisions = [
            (
                item.result.final_authorization or item.result.initial_authorization
                if isinstance(item, GovernedActionExecution)
                else item.authorization
            )
            for item in trial_executions
        ]
        reference = authorization_decisions[0]
        if any(
            decision.policy_version != reference.policy_version
            or decision.environment_sha256 != reference.environment_sha256
            or decision.document_universe_sha256 != reference.document_universe_sha256
            or decision.available is not reference.available
            or not np.array_equal(
                decision.authorized_mask,
                reference.authorized_mask,
            )
            for decision in authorization_decisions[1:]
        ):
            raise ConfirmatoryAnalysisError(
                "paired actions do not share one final authorization universe"
            )

        for action_order, action in enumerate(actions):
            item = keyed[(trial.trial_key, action)]
            controller_selected = action == selection.action
            execution_position = orders[trial.trial_key].index(action)
            rows.append(
                item.to_prelabel_row(
                    action_order=action_order,
                    execution_position=execution_position,
                    controller_selected=controller_selected,
                )
            )
            admission_records.append(
                _admission_record(
                    item,
                    action_order=action_order,
                    execution_position=execution_position,
                    controller_selected=controller_selected,
                )
            )

    panel = ActionPanelArtifact(
        manifest_sha256=run_receipt.manifest_sha256,
        run_receipt_sha256=sealed_run_receipt_sha256(run_receipt),
        execution_artifact_sha256=execution_sha256,
        corpus=corpus,
        stage=stage,
        document_count=document_count,
        action_set=actions,
        rows=tuple(rows),
    )
    receipt = ActionPanelAdmissionReceipt(
        manifest_sha256=run_receipt.manifest_sha256,
        run_receipt_sha256=sealed_run_receipt_sha256(run_receipt),
        execution_artifact_sha256=execution_sha256,
        action_panel_artifact_sha256=panel.artifact_sha256,
        corpus=corpus,
        query_partition_audit_sha256=query_partition_audit_sha256,
        partition_label=partition_label,
        audit_head_sha256=expected_audit_head_sha256,
        audit_chain_length=len(ordered_records),
        audit_record_sha256s=tuple(record.record_sha256 for record in ordered_records),
        records=tuple(admission_records),
    )
    return AdmittedActionPanel(panel=panel, admission_receipt=receipt)
