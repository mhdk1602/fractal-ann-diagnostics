from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import fractal_ann_diagnostics.label_separation as separation
from fractal_ann_diagnostics.corpora import (
    CorpusDocument,
    EvidenceQuery,
    NormalizedCorpus,
)
from fractal_ann_diagnostics.evidence import (
    CompleteEvidenceBundle,
    EvidenceLocation,
    GoldEvidence,
)
from fractal_ann_diagnostics.label_separation import (
    ActionPanelBinding,
    LabelSeparationError,
    OnlineExecutionArtifact,
    OnlinePrediction,
    PredictionCompletionReceipt,
    SealedLabelArtifact,
    create_prediction_completion_receipt,
    emit_online_predictions,
    join_predictions_after_receipt,
    split_custodian_corpus,
    write_prediction_completion_receipt,
)
from fractal_ann_diagnostics.study import SealedRunReceipt

MANIFEST_SHA256 = "a" * 64
OTHER_MANIFEST_SHA256 = "b" * 64
HMAC_KEY = bytes(range(32))
KEY_ID = "custodian-hmac-2026-01"
ANCHOR_IDENTITY = "rekor-log-entry-018f6f8c"
ANCHOR_URI = "https://rekor.example.test/api/v1/log/entries/018f6f8c"
ANCHORED_AT_UTC = "2026-07-13T22:00:01+00:00"
ACTION_PANEL_SHA256 = "9" * 64


def _document(document_id: int) -> CorpusDocument:
    return CorpusDocument(
        document_id=document_id,
        external_id=f"doc-{document_id}",
        title=f"Document {document_id}",
        text=f"Sealed document text {document_id}",
        source_uri=f"corpus://document/{document_id}",
        content_hash=f"sha256:{document_id + 1:064x}",
    )


def _gold(query_id: str) -> GoldEvidence:
    return GoldEvidence(
        query_id=query_id,
        alternatives=(
            CompleteEvidenceBundle(
                bundle_id="SENTINEL_GOLD_BUNDLE",
                locations=(
                    EvidenceLocation(
                        document_id=1,
                        source_uri="corpus://document/1",
                        locator="sentence:SENTINEL_GOLD_LOCATION",
                        content_hash=f"sha256:{2:064x}",
                    ),
                ),
            ),
        ),
    )


def _corpus(
    *,
    name: str = "sealed-corpus",
    stage: str = "sealed",
    reverse_queries: bool = False,
    answer_override: str | None = None,
) -> NormalizedCorpus:
    first_id = "raw-query-alpha-secret"
    first = EvidenceQuery(
        query_id=first_id,
        query_family="raw-family-shared-secret",
        text="Which document supports the alpha claim?",
        corpus=name,
        stage=stage,
        answer=(answer_override if answer_override is not None else "SENTINEL_GOLD_ANSWER"),
        gold_evidence=_gold(first_id),
        relevant_document_ids=(1,),
        metadata={
            "evidence_labels": "SENTINEL_SUPPORT_LABEL",
            "type": "bridge",
        },
    )
    second = EvidenceQuery(
        query_id="raw-query-beta-secret",
        query_family="raw-family-shared-secret",
        text="Which document supports the beta claim?",
        corpus=name,
        stage=stage,
        answer="SENTINEL_BETA_ANSWER",
        gold_evidence=None,
        relevant_document_ids=(2,),
        metadata={
            "correctness_label": "SENTINEL_CORRECT",
            "level": "hard",
        },
    )
    queries = (second, first) if reverse_queries else (first, second)
    return NormalizedCorpus(
        name=name,
        stage=stage,
        documents=tuple(_document(index) for index in range(3)),
        queries=queries,
    )


def _split(
    *,
    corpus: NormalizedCorpus | None = None,
    hmac_key: bytes = HMAC_KEY,
    key_id: str = KEY_ID,
):
    return split_custodian_corpus(
        corpus or _corpus(),
        hmac_key=hmac_key,
        key_id=key_id,
    )


def _receipt(
    *,
    manifest_sha256: str = MANIFEST_SHA256,
    runner_identity: str = "sealed-runner",
    started_at_utc: str = "2026-07-13T22:00:00+00:00",
) -> SealedRunReceipt:
    return SealedRunReceipt(
        manifest_sha256=manifest_sha256,
        protocol_version="0.3.0",
        started_at_utc=started_at_utc,
        runner_identity=runner_identity,
        code_commit="c" * 40,
        runner_image=f"ghcr.io/example/runner@sha256:{'d' * 64}",
        protocol_registration_receipt_uri=(
            "file:///sealed/receipts/protocol-registration.json"
        ),
        protocol_registration_receipt_sha256="f" * 64,
        protocol_registration_record_uri=(
            "file:///sealed/receipts/protocol-registration-record.json"
        ),
        receipt_uri=f"file:///sealed/receipts/{manifest_sha256}.json",
        verification_receipt_uri="file:///sealed/receipts/artifacts.json",
        verification_receipt_sha256="e" * 64,
    )


def _online_rows(execution: OnlineExecutionArtifact) -> tuple[OnlinePrediction, ...]:
    return tuple(
        OnlinePrediction(
            trial_key=trial.trial_key,
            family_key=trial.family_key,
            returned_document_ids=(1,),
            emitted_answer="generated answer",
        )
        for trial in execution.trials
    )


def _panel_binding(
    split,
    receipt: SealedRunReceipt,
) -> ActionPanelBinding:
    return ActionPanelBinding(
        manifest_sha256=MANIFEST_SHA256,
        run_receipt_sha256=separation.sealed_run_receipt_sha256(receipt),
        execution_artifact_sha256=split.execution.artifact_sha256,
        corpus=split.execution.corpus,
        stage=split.execution.stage,
        action_panel_artifact_sha256=ACTION_PANEL_SHA256,
    )


def _prediction_artifact(split=None, *, receipt: SealedRunReceipt | None = None):
    split = split or _split()
    return emit_online_predictions(
        split.execution,
        _online_rows(split.execution),
        receipt=receipt or _receipt(),
        manifest_sha256=MANIFEST_SHA256,
    )


def _completion_receipt(
    split=None,
    predictions=None,
    *,
    receipt: SealedRunReceipt | None = None,
) -> PredictionCompletionReceipt:
    split = split or _split()
    receipt = receipt or _receipt()
    predictions = predictions or _prediction_artifact(split, receipt=receipt)
    return create_prediction_completion_receipt(
        predictions,
        execution=split.execution,
        receipt=receipt,
        manifest_sha256=MANIFEST_SHA256,
        action_panel_binding=_panel_binding(split, receipt),
        external_anchor_identity=ANCHOR_IDENTITY,
        external_anchor_uri=ANCHOR_URI,
        anchored_at_utc=ANCHORED_AT_UTC,
    )


def test_custodian_split_removes_every_query_label_and_original_identifier() -> None:
    split = _split()
    execution_bytes = split.execution.canonical_bytes()
    execution_json = json.loads(execution_bytes)
    labels_bytes = split.sealed_labels.canonical_bytes()

    assert len(split.execution.documents) == 3
    assert len(split.execution.trials) == 2
    assert split.execution.key_id == KEY_ID
    assert split.sealed_labels.key_id == KEY_ID
    assert split.sealed_labels.execution_artifact_sha256 == split.execution.artifact_sha256
    assert "manifest_sha256" not in execution_json
    assert "manifest_sha256" not in json.loads(labels_bytes)
    assert execution_json["trials"][0].keys() == {
        "corpus",
        "family_key",
        "stage",
        "text",
        "trial_key",
    }
    for forbidden in (
        b"raw-query-alpha-secret",
        b"raw-query-beta-secret",
        b"raw-family-shared-secret",
        b"SENTINEL_GOLD_ANSWER",
        b"SENTINEL_BETA_ANSWER",
        b"SENTINEL_GOLD_BUNDLE",
        b"SENTINEL_GOLD_LOCATION",
        b"SENTINEL_SUPPORT_LABEL",
        b'"answer"',
        b'"gold_evidence"',
        b'"query_id"',
        b'"relevant_document_ids"',
    ):
        assert forbidden not in execution_bytes
    assert HMAC_KEY.hex().encode() not in execution_bytes
    assert b"SENTINEL_GOLD_ANSWER" in labels_bytes
    assert b"SENTINEL_SUPPORT_LABEL" in labels_bytes
    assert b"SENTINEL_CORRECT" in labels_bytes
    assert b'"type"' not in labels_bytes
    assert b'"level"' not in labels_bytes
    assert b"raw-query-alpha-secret" not in labels_bytes

    trials = split.execution.trials
    assert trials[0].trial_key != trials[1].trial_key
    assert trials[0].family_key == trials[1].family_key
    assert all(len(trial.trial_key) == 64 for trial in trials)
    assert not hasattr(trials[0], "answer")
    assert not hasattr(trials[0], "relevant_document_ids")
    assert not hasattr(trials[0], "gold_evidence")


def test_label_changes_cannot_change_the_online_execution_artifact() -> None:
    original = _split(corpus=_corpus())
    changed = _split(corpus=_corpus(answer_override="A DIFFERENT SEALED ANSWER"))

    assert original.execution.canonical_bytes() == changed.execution.canonical_bytes()
    assert original.execution.artifact_sha256 == changed.execution.artifact_sha256
    assert original.sealed_labels.canonical_bytes() != changed.sealed_labels.canonical_bytes()
    assert original.sealed_labels.artifact_sha256 != changed.sealed_labels.artifact_sha256


def test_split_is_deterministic_across_source_query_order() -> None:
    first = _split(corpus=_corpus(reverse_queries=False))
    second = _split(corpus=_corpus(reverse_queries=True))

    assert first.execution.canonical_bytes() == second.execution.canonical_bytes()
    assert first.sealed_labels.canonical_bytes() == second.sealed_labels.canonical_bytes()
    assert first.execution.artifact_sha256 == second.execution.artifact_sha256
    assert first.sealed_labels.artifact_sha256 == second.sealed_labels.artifact_sha256


def test_key_and_key_id_are_bound_into_opaque_keys() -> None:
    baseline = _split()
    other_key = _split(hmac_key=bytes(range(1, 33)))
    other_key_id = _split(key_id="custodian-hmac-2026-02")

    baseline_keys = {trial.trial_key for trial in baseline.execution.trials}
    assert baseline_keys.isdisjoint({trial.trial_key for trial in other_key.execution.trials})
    assert baseline_keys.isdisjoint(
        {trial.trial_key for trial in other_key_id.execution.trials}
    )


@pytest.mark.parametrize(
    "weak_key",
    [
        b"too-short",
        b"\x00" * 32,
        b"ab" * 16,
        bytearray(range(32)),
    ],
)
def test_weak_or_mutable_hmac_keys_are_rejected(weak_key: object) -> None:
    with pytest.raises(LabelSeparationError, match="HMAC key"):
        split_custodian_corpus(
            _corpus(),
            hmac_key=weak_key,  # type: ignore[arg-type]
            key_id=KEY_ID,
        )


def test_only_sealed_corpora_can_cross_the_custodian_boundary() -> None:
    with pytest.raises(LabelSeparationError, match="stage='sealed'"):
        _split(corpus=_corpus(stage="development"))


def test_nested_artifact_data_is_immutable() -> None:
    split = _split()
    trial = split.execution.trials[0]
    label = split.sealed_labels.labels[0]

    with pytest.raises(FrozenInstanceError):
        trial.text = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        label.answer = "mutated"  # type: ignore[misc]
    assert isinstance(split.execution.trials, tuple)
    assert isinstance(split.sealed_labels.labels, tuple)
    assert isinstance(label.relevant_document_ids, tuple)
    assert isinstance(label.evidence_bundles, tuple)
    assert isinstance(label.label_metadata, tuple)


def test_online_types_and_emitter_have_no_ground_truth_label_surface() -> None:
    forbidden = {
        "answer",
        "correctness",
        "gold_evidence",
        "labels",
        "relevance",
        "relevant_document_ids",
    }
    prediction_parameters = set(inspect.signature(OnlinePrediction).parameters)
    emitter_parameters = set(inspect.signature(emit_online_predictions).parameters)
    assert not prediction_parameters.intersection(forbidden)
    assert not emitter_parameters.intersection(forbidden)

    split = _split()
    trial = split.execution.trials[0]
    with pytest.raises(TypeError):
        OnlinePrediction(
            trial_key=trial.trial_key,
            family_key=trial.family_key,
            returned_document_ids=(1,),
            correctness=True,  # type: ignore[call-arg]
        )
    with pytest.raises(LabelSeparationError, match="only OnlinePrediction"):
        emit_online_predictions(
            split.execution,
            split.sealed_labels.labels,
            receipt=_receipt(),
            manifest_sha256=MANIFEST_SHA256,
        )


def test_online_prediction_emission_is_exact_canonical_and_label_free() -> None:
    split = _split()
    rows = list(_online_rows(split.execution))
    first = emit_online_predictions(
        split.execution,
        rows,
        receipt=_receipt(),
        manifest_sha256=MANIFEST_SHA256,
    )
    rows.reverse()
    second = emit_online_predictions(
        split.execution,
        rows,
        receipt=_receipt(),
        manifest_sha256=MANIFEST_SHA256,
    )

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.execution_artifact_sha256 == split.execution.artifact_sha256
    assert first.run_receipt_sha256 == separation.sealed_run_receipt_sha256(_receipt())
    payload = first.canonical_bytes()
    assert b"SENTINEL_GOLD_ANSWER" not in payload
    assert b'"relevant_document_ids"' not in payload
    assert b'"gold_evidence"' not in payload
    with pytest.raises(FrozenInstanceError):
        first.predictions = ()  # type: ignore[misc]


def test_online_emission_rejects_missing_extra_and_duplicate_trial_keys() -> None:
    split = _split()
    rows = _online_rows(split.execution)
    with pytest.raises(LabelSeparationError, match="missing trial keys"):
        emit_online_predictions(
            split.execution,
            rows[:-1],
            receipt=_receipt(),
            manifest_sha256=MANIFEST_SHA256,
        )

    extra = OnlinePrediction(
        trial_key="e" * 64,
        family_key="f" * 64,
        returned_document_ids=(1,),
    )
    with pytest.raises(LabelSeparationError, match="extra trial keys"):
        emit_online_predictions(
            split.execution,
            (*rows, extra),
            receipt=_receipt(),
            manifest_sha256=MANIFEST_SHA256,
        )
    with pytest.raises(LabelSeparationError, match="duplicate trial keys"):
        emit_online_predictions(
            split.execution,
            (*rows, rows[0]),
            receipt=_receipt(),
            manifest_sha256=MANIFEST_SHA256,
        )


def test_online_emission_rejects_family_and_document_binding_mismatches() -> None:
    split = _split()
    rows = list(_online_rows(split.execution))
    rows[0] = replace(rows[0], family_key="f" * 64)
    with pytest.raises(LabelSeparationError, match="family key"):
        emit_online_predictions(
            split.execution,
            rows,
            receipt=_receipt(),
            manifest_sha256=MANIFEST_SHA256,
        )

    with pytest.raises(LabelSeparationError, match="unknown document"):
        emit_online_predictions(
            split.execution,
            (
                replace(_online_rows(split.execution)[0], returned_document_ids=(99,)),
                _online_rows(split.execution)[1],
            ),
            receipt=_receipt(),
            manifest_sha256=MANIFEST_SHA256,
        )


def test_online_emission_rejects_cross_manifest_receipt() -> None:
    split = _split()
    rows = _online_rows(split.execution)
    with pytest.raises(LabelSeparationError, match="receipt belongs to another manifest"):
        emit_online_predictions(
            split.execution,
            rows,
            receipt=_receipt(manifest_sha256=OTHER_MANIFEST_SHA256),
            manifest_sha256=MANIFEST_SHA256,
        )


def test_prediction_completion_receipt_is_canonical_and_externally_anchored() -> None:
    split = _split()
    receipt = _receipt()
    predictions = _prediction_artifact(split, receipt=receipt)
    completion = _completion_receipt(split, predictions, receipt=receipt)

    assert completion.to_dict() == {
        "anchored_at_utc": ANCHORED_AT_UTC,
        "action_panel_binding": _panel_binding(split, receipt).to_dict(),
        "corpus": split.execution.corpus,
        "execution_artifact_sha256": split.execution.artifact_sha256,
        "external_anchor_identity": ANCHOR_IDENTITY,
        "external_anchor_uri": ANCHOR_URI,
        "manifest_sha256": MANIFEST_SHA256,
        "prediction_artifact_sha256": predictions.artifact_sha256,
        "prediction_count": len(predictions.predictions),
        "run_receipt_sha256": separation.sealed_run_receipt_sha256(receipt),
        "schema_version": separation.PREDICTION_COMPLETION_SCHEMA,
        "stage": split.execution.stage,
    }
    assert completion.canonical_bytes() == json.dumps(
        completion.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert completion.receipt_sha256 == hashlib.sha256(
        completion.canonical_bytes()
    ).hexdigest()


def test_prediction_completion_requires_a_canonical_prelabel_action_panel() -> None:
    split = _split()
    receipt = _receipt()
    predictions = _prediction_artifact(split, receipt=receipt)
    with pytest.raises(LabelSeparationError, match="action_panel_binding"):
        create_prediction_completion_receipt(
            predictions,
            execution=split.execution,
            receipt=receipt,
            manifest_sha256=MANIFEST_SHA256,
            action_panel_binding="not-a-binding",  # type: ignore[arg-type]
            external_anchor_identity=ANCHOR_IDENTITY,
            external_anchor_uri=ANCHOR_URI,
            anchored_at_utc=ANCHORED_AT_UTC,
        )


def test_prediction_completion_writer_is_exclusive_and_does_not_follow_links(
    tmp_path: Path,
) -> None:
    completion = _completion_receipt()
    target = (tmp_path / "prediction-completion.json").resolve()
    write_prediction_completion_receipt(completion, target)

    assert target.read_bytes() == completion.canonical_bytes() + b"\n"
    with pytest.raises(LabelSeparationError, match="already exists"):
        write_prediction_completion_receipt(completion, target)

    outside = (tmp_path / "outside.json").resolve()
    outside.write_text("do-not-replace", encoding="utf-8")
    link = (tmp_path / "linked-receipt.json").resolve()
    link.symlink_to(outside)
    with pytest.raises(LabelSeparationError, match="safely"):
        write_prediction_completion_receipt(completion, link)
    assert outside.read_text(encoding="utf-8") == "do-not-replace"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"manifest_sha256": OTHER_MANIFEST_SHA256}, "manifest_sha256"),
        ({"run_receipt_sha256": "c" * 64}, "run_receipt_sha256"),
        ({"execution_artifact_sha256": "c" * 64}, "execution_artifact_sha256"),
        ({"prediction_artifact_sha256": "c" * 64}, "prediction_artifact_sha256"),
        ({"prediction_count": 3}, "prediction_count"),
        ({"corpus": "another-corpus"}, "corpus"),
    ],
)
def test_join_rejects_mismatched_prediction_completion_bindings(
    updates: dict[str, object],
    message: str,
) -> None:
    split = _split()
    receipt = _receipt()
    predictions = _prediction_artifact(split, receipt=receipt)
    original = _completion_receipt(split, predictions, receipt=receipt)
    panel_bound_fields = {
        "manifest_sha256",
        "run_receipt_sha256",
        "execution_artifact_sha256",
        "corpus",
    }
    panel_updates = {
        field: value for field, value in updates.items() if field in panel_bound_fields
    }
    completion = replace(
        original,
        action_panel_binding=replace(original.action_panel_binding, **panel_updates),
        **updates,
    )

    with pytest.raises(LabelSeparationError, match=message):
        join_predictions_after_receipt(
            predictions,
            split.sealed_labels,
            execution=split.execution,
            receipt=receipt,
            completion_receipt=completion,
            action_panel_binding=completion.action_panel_binding,
            manifest_sha256=MANIFEST_SHA256,
        )


def test_completion_rejects_a_nonsealed_panel_stage_before_join() -> None:
    original = _completion_receipt()

    with pytest.raises(LabelSeparationError, match="stage"):
        replace(original, stage="another-stage")


@pytest.mark.parametrize(
    ("anchor_uri", "anchored_at_utc", "message"),
    [
        (
            "http://rekor.example.test/entry/018f6f8c",
            ANCHORED_AT_UTC,
            "absolute HTTPS URI",
        ),
        (ANCHOR_URI, "2026-07-13T21:59:59+00:00", "must postdate"),
        (ANCHOR_URI, "2026-07-13T22:00:00+00:00", "must postdate"),
        (ANCHOR_URI, "2026-07-13T22:00:01Z", "canonical ISO 8601"),
    ],
)
def test_completion_receipt_rejects_nonexternal_or_noncanonical_anchor(
    anchor_uri: str,
    anchored_at_utc: str,
    message: str,
) -> None:
    split = _split()
    receipt = _receipt()
    predictions = _prediction_artifact(split, receipt=receipt)
    with pytest.raises(LabelSeparationError, match=message):
        create_prediction_completion_receipt(
            predictions,
            execution=split.execution,
            receipt=receipt,
            manifest_sha256=MANIFEST_SHA256,
            action_panel_binding=_panel_binding(split, receipt),
            external_anchor_identity=ANCHOR_IDENTITY,
            external_anchor_uri=anchor_uri,
            anchored_at_utc=anchored_at_utc,
        )


def test_bare_prediction_artifact_cannot_unlock_labels() -> None:
    split = _split()
    receipt = _receipt()
    predictions = _prediction_artifact(split, receipt=receipt)

    with pytest.raises(TypeError, match="completion_receipt"):
        join_predictions_after_receipt(
            predictions,
            split.sealed_labels,
            execution=split.execution,
            receipt=receipt,
            manifest_sha256=MANIFEST_SHA256,
        )


def test_label_release_rejects_a_panel_not_bound_before_completion() -> None:
    split = _split()
    receipt = _receipt()
    predictions = _prediction_artifact(split, receipt=receipt)
    completion = _completion_receipt(split, predictions, receipt=receipt)
    later_panel = replace(
        completion.action_panel_binding,
        action_panel_artifact_sha256="8" * 64,
    )

    with pytest.raises(LabelSeparationError, match="different action panel"):
        join_predictions_after_receipt(
            predictions,
            split.sealed_labels,
            execution=split.execution,
            receipt=receipt,
            completion_receipt=completion,
            action_panel_binding=later_panel,
            manifest_sha256=MANIFEST_SHA256,
        )


def test_post_completion_join_is_exact_and_exposes_labels_only_offline() -> None:
    split = _split()
    receipt = _receipt()
    predictions = _prediction_artifact(split, receipt=receipt)
    completion = _completion_receipt(split, predictions, receipt=receipt)

    joined = join_predictions_after_receipt(
        predictions,
        split.sealed_labels,
        execution=split.execution,
        receipt=receipt,
        completion_receipt=completion,
        action_panel_binding=completion.action_panel_binding,
        manifest_sha256=MANIFEST_SHA256,
    )

    assert len(joined.trials) == len(split.execution.trials)
    assert joined.execution_artifact_sha256 == split.execution.artifact_sha256
    assert joined.prediction_artifact_sha256 == predictions.artifact_sha256
    assert joined.prediction_completion_receipt_sha256 == completion.receipt_sha256
    assert joined.sealed_label_artifact_sha256 == split.sealed_labels.artifact_sha256
    assert any(row.labels.answer == "SENTINEL_GOLD_ANSWER" for row in joined.trials)
    assert all(
        row.prediction.trial_key == row.labels.trial_key for row in joined.trials
    )
    assert all(
        row.prediction.family_key == row.labels.family_key for row in joined.trials
    )
    assert not hasattr(predictions.predictions[0], "labels")


def test_join_requires_an_immutable_prediction_artifact() -> None:
    split = _split()
    predictions = _prediction_artifact(split)
    completion = _completion_receipt(split, predictions)
    with pytest.raises(LabelSeparationError, match="PredictionArtifact"):
        join_predictions_after_receipt(
            _online_rows(split.execution),  # type: ignore[arg-type]
            split.sealed_labels,
            execution=split.execution,
            receipt=_receipt(),
            completion_receipt=completion,
            action_panel_binding=completion.action_panel_binding,
            manifest_sha256=MANIFEST_SHA256,
        )


def test_join_rejects_missing_extra_and_duplicate_label_keys() -> None:
    split = _split()
    predictions = _prediction_artifact(split)
    labels = split.sealed_labels
    completion = _completion_receipt(split, predictions)

    missing = replace(labels, labels=labels.labels[:-1])
    with pytest.raises(LabelSeparationError, match="missing trial keys"):
        join_predictions_after_receipt(
            predictions,
            missing,
            execution=split.execution,
            receipt=_receipt(),
            completion_receipt=completion,
            action_panel_binding=completion.action_panel_binding,
            manifest_sha256=MANIFEST_SHA256,
        )

    extra_row = replace(labels.labels[0], trial_key="e" * 64)
    extra = replace(labels, labels=(*labels.labels, extra_row))
    with pytest.raises(LabelSeparationError, match="extra trial keys"):
        join_predictions_after_receipt(
            predictions,
            extra,
            execution=split.execution,
            receipt=_receipt(),
            completion_receipt=completion,
            action_panel_binding=completion.action_panel_binding,
            manifest_sha256=MANIFEST_SHA256,
        )

    with pytest.raises(LabelSeparationError, match="duplicate trial keys"):
        replace(labels, labels=(*labels.labels, labels.labels[0]))

    both_truncated_predictions = replace(
        predictions,
        predictions=predictions.predictions[:-1],
    )
    both_truncated_labels = replace(labels, labels=labels.labels[:-1])
    with pytest.raises(LabelSeparationError, match="prediction_artifact_sha256"):
        join_predictions_after_receipt(
            both_truncated_predictions,
            both_truncated_labels,
            execution=split.execution,
            receipt=_receipt(),
            completion_receipt=completion,
            action_panel_binding=completion.action_panel_binding,
            manifest_sha256=MANIFEST_SHA256,
        )


def test_join_rejects_cross_manifest_corpus_stage_execution_and_run_bindings() -> None:
    split = _split()
    predictions = _prediction_artifact(split)
    labels = split.sealed_labels
    completion = _completion_receipt(split, predictions)

    with pytest.raises(LabelSeparationError, match="mismatched corpus"):
        join_predictions_after_receipt(
            predictions,
            replace(labels, corpus="another-corpus"),
            execution=split.execution,
            receipt=_receipt(),
            completion_receipt=completion,
            action_panel_binding=completion.action_panel_binding,
            manifest_sha256=MANIFEST_SHA256,
        )
    with pytest.raises(LabelSeparationError, match="mismatched stage"):
        join_predictions_after_receipt(
            predictions,
            replace(labels, stage="another-stage"),
            execution=split.execution,
            receipt=_receipt(),
            completion_receipt=completion,
            action_panel_binding=completion.action_panel_binding,
            manifest_sha256=MANIFEST_SHA256,
        )
    with pytest.raises(LabelSeparationError, match="different execution artifacts"):
        join_predictions_after_receipt(
            predictions,
            replace(labels, execution_artifact_sha256="e" * 64),
            execution=split.execution,
            receipt=_receipt(),
            completion_receipt=completion,
            action_panel_binding=completion.action_panel_binding,
            manifest_sha256=MANIFEST_SHA256,
        )
    with pytest.raises(LabelSeparationError, match="another sealed run"):
        join_predictions_after_receipt(
            predictions,
            labels,
            execution=split.execution,
            receipt=_receipt(
                runner_identity="another-runner",
                started_at_utc="2026-07-13T22:00:01+00:00",
            ),
            completion_receipt=completion,
            action_panel_binding=completion.action_panel_binding,
            manifest_sha256=MANIFEST_SHA256,
        )


def test_join_rejects_family_key_rebinding() -> None:
    split = _split()
    predictions = _prediction_artifact(split)
    labels = split.sealed_labels
    completion = _completion_receipt(split, predictions)
    changed_row = replace(labels.labels[0], family_key="f" * 64)
    rebound = replace(labels, labels=(changed_row, *labels.labels[1:]))

    with pytest.raises(LabelSeparationError, match="mismatched family keys"):
        join_predictions_after_receipt(
            predictions,
            rebound,
            execution=split.execution,
            receipt=_receipt(),
            completion_receipt=completion,
            action_panel_binding=completion.action_panel_binding,
            manifest_sha256=MANIFEST_SHA256,
        )


def test_execution_serializer_rejects_a_forged_label_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = _split()

    def leaking_dict(self: object) -> dict[str, object]:
        return {"gold_answer_payload": "forged gold label"}

    monkeypatch.setattr(separation.OnlineTrial, "to_dict", leaking_dict)
    with pytest.raises(LabelSeparationError, match="field leaked"):
        OnlineExecutionArtifact(
            key_id=split.execution.key_id,
            corpus=split.execution.corpus,
            stage=split.execution.stage,
            documents=split.execution.documents,
            trials=split.execution.trials,
        )


def test_hmac_collision_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(separation, "_derive_opaque_key", lambda *args, **kwargs: "f" * 64)
    with pytest.raises(LabelSeparationError, match="duplicate trial keys"):
        _split()


def test_sealed_label_artifact_rejects_duplicate_rows_directly() -> None:
    labels = _split().sealed_labels
    with pytest.raises(LabelSeparationError, match="duplicate trial keys"):
        SealedLabelArtifact(
            execution_artifact_sha256=labels.execution_artifact_sha256,
            key_id=labels.key_id,
            corpus=labels.corpus,
            stage=labels.stage,
            document_count=labels.document_count,
            labels=(*labels.labels, labels.labels[0]),
        )
