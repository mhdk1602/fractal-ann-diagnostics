from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

import fractal_ann_diagnostics.label_separation as separation
from fractal_ann_diagnostics.artifact_integrity import ArtifactIntegrityError
from fractal_ann_diagnostics.label_separation import (
    ActionPanelBinding,
    JoinedEvaluationTrial,
    LabelSeparationError,
    OfflineEvaluationArtifact,
    OnlineDocument,
    OnlineExecutionArtifact,
    OnlinePrediction,
    OnlineTrial,
    PredictionArtifact,
    PredictionCompletionReceipt,
    SealedEvidenceBundle,
    SealedEvidenceLocation,
    SealedLabelArtifact,
    SealedTrialLabels,
)

MANIFEST_SHA256 = "a" * 64
TRIAL_KEY = "b" * 64
FAMILY_KEY = "c" * 64


def _artifact_set() -> dict[str, Any]:
    execution = OnlineExecutionArtifact(
        key_id="custodian-key-1",
        corpus="sealed-corpus",
        stage="sealed",
        documents=(
            OnlineDocument(
                document_id=0,
                external_id="doc-0",
                title="Document zero",
                text="Evidence text",
                source_uri="corpus://sealed/doc-0",
                content_hash=f"sha256:{'d' * 64}",
            ),
        ),
        trials=(
            OnlineTrial(
                trial_key=TRIAL_KEY,
                family_key=FAMILY_KEY,
                text="Which record supports the claim?",
                corpus="sealed-corpus",
                stage="sealed",
            ),
        ),
    )
    trial_labels = SealedTrialLabels(
        trial_key=TRIAL_KEY,
        family_key=FAMILY_KEY,
        answer="Document zero",
        relevant_document_ids=(0,),
        evidence_bundles=(
            SealedEvidenceBundle(
                bundle_id="bundle-0",
                locations=(
                    SealedEvidenceLocation(
                        document_id=0,
                        source_uri="corpus://sealed/doc-0",
                        locator="sentence:1",
                        content_hash=f"sha256:{'d' * 64}",
                    ),
                ),
            ),
        ),
        label_metadata=(("correctness_label", "supported"),),
    )
    sealed_labels = SealedLabelArtifact(
        execution_artifact_sha256=execution.artifact_sha256,
        key_id=execution.key_id,
        corpus=execution.corpus,
        stage=execution.stage,
        document_count=1,
        labels=(trial_labels,),
    )
    prediction_row = OnlinePrediction(
        trial_key=TRIAL_KEY,
        family_key=FAMILY_KEY,
        returned_document_ids=(0,),
        emitted_answer="Document zero",
    )
    predictions = PredictionArtifact(
        manifest_sha256=MANIFEST_SHA256,
        run_receipt_sha256="e" * 64,
        execution_artifact_sha256=execution.artifact_sha256,
        key_id=execution.key_id,
        corpus=execution.corpus,
        stage=execution.stage,
        document_count=1,
        predictions=(prediction_row,),
    )
    completion = PredictionCompletionReceipt(
        manifest_sha256=MANIFEST_SHA256,
        run_receipt_sha256=predictions.run_receipt_sha256,
        execution_artifact_sha256=execution.artifact_sha256,
        prediction_artifact_sha256=predictions.artifact_sha256,
        action_panel_binding=ActionPanelBinding(
            manifest_sha256=MANIFEST_SHA256,
            run_receipt_sha256=predictions.run_receipt_sha256,
            execution_artifact_sha256=execution.artifact_sha256,
            corpus=execution.corpus,
            stage=execution.stage,
            action_panel_artifact_sha256="f" * 64,
        ),
        prediction_count=1,
        corpus=execution.corpus,
        stage=execution.stage,
        external_anchor_identity="rekor-entry-1",
        external_anchor_uri="https://rekor.example.test/entries/1",
        anchored_at_utc="2026-07-13T22:00:01+00:00",
    )
    offline = OfflineEvaluationArtifact(
        manifest_sha256=MANIFEST_SHA256,
        run_receipt_sha256=predictions.run_receipt_sha256,
        execution_artifact_sha256=execution.artifact_sha256,
        prediction_artifact_sha256=predictions.artifact_sha256,
        prediction_completion_receipt_sha256=completion.receipt_sha256,
        sealed_label_artifact_sha256=sealed_labels.artifact_sha256,
        corpus=execution.corpus,
        stage=execution.stage,
        trials=(
            JoinedEvaluationTrial(
                prediction=prediction_row,
                labels=trial_labels,
            ),
        ),
    )
    return {
        "execution": execution,
        "sealed_labels": sealed_labels,
        "predictions": predictions,
        "completion": completion,
        "offline": offline,
    }


IO_CASES = (
    (
        "execution",
        separation.write_online_execution_artifact,
        separation.load_online_execution_artifact,
    ),
    (
        "sealed_labels",
        separation.write_sealed_label_artifact,
        separation.load_sealed_label_artifact,
    ),
    (
        "predictions",
        separation.write_prediction_artifact,
        separation.load_prediction_artifact,
    ),
    (
        "completion",
        separation.write_prediction_completion_receipt,
        separation.load_prediction_completion_receipt,
    ),
    (
        "offline",
        separation.write_offline_evaluation_artifact,
        separation.load_offline_evaluation_artifact,
    ),
)


@pytest.mark.parametrize(("name", "writer", "loader"), IO_CASES)
def test_artifact_io_round_trip_is_canonical_and_exclusive(
    tmp_path: Path,
    name: str,
    writer: Any,
    loader: Any,
) -> None:
    artifact = _artifact_set()[name]
    target = tmp_path / f"{name}.json"

    writer(artifact, target)

    assert target.read_bytes() == artifact.canonical_bytes() + b"\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert loader(target) == artifact
    with pytest.raises(LabelSeparationError, match="already exists"):
        writer(artifact, target)


def test_from_dict_recursively_restores_typed_rows() -> None:
    artifacts = _artifact_set()
    execution = OnlineExecutionArtifact.from_dict(artifacts["execution"].to_dict())
    sealed = SealedLabelArtifact.from_dict(artifacts["sealed_labels"].to_dict())
    predictions = PredictionArtifact.from_dict(artifacts["predictions"].to_dict())
    completion = PredictionCompletionReceipt.from_dict(
        artifacts["completion"].to_dict()
    )
    offline = OfflineEvaluationArtifact.from_dict(artifacts["offline"].to_dict())

    assert execution == artifacts["execution"]
    assert isinstance(execution.documents[0], OnlineDocument)
    assert isinstance(execution.trials[0], OnlineTrial)
    assert sealed == artifacts["sealed_labels"]
    assert isinstance(sealed.labels[0], SealedTrialLabels)
    assert isinstance(sealed.labels[0].evidence_bundles[0], SealedEvidenceBundle)
    assert isinstance(
        sealed.labels[0].evidence_bundles[0].locations[0],
        SealedEvidenceLocation,
    )
    assert predictions == artifacts["predictions"]
    assert isinstance(predictions.predictions[0], OnlinePrediction)
    assert completion == artifacts["completion"]
    assert offline == artifacts["offline"]
    assert isinstance(offline.trials[0], JoinedEvaluationTrial)
    assert isinstance(offline.trials[0].prediction, OnlinePrediction)
    assert isinstance(offline.trials[0].labels, SealedTrialLabels)


@pytest.mark.parametrize(
    ("name", "artifact_type"),
    (
        ("execution", OnlineExecutionArtifact),
        ("sealed_labels", SealedLabelArtifact),
        ("predictions", PredictionArtifact),
        ("completion", PredictionCompletionReceipt),
        ("offline", OfflineEvaluationArtifact),
    ),
)
@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_from_dict_rejects_unknown_and_missing_top_level_fields(
    name: str,
    artifact_type: Any,
    mutation: str,
) -> None:
    payload = _artifact_set()[name].to_dict()
    if mutation == "unknown":
        payload["undeclared"] = True
        match = "unknown fields"
    else:
        payload.pop(next(iter(payload)))
        match = "missing fields"

    with pytest.raises(LabelSeparationError, match=match):
        artifact_type.from_dict(payload)


@pytest.mark.parametrize(
    ("name", "artifact_type", "mutate"),
    (
        (
            "execution",
            OnlineExecutionArtifact,
            lambda payload: payload["documents"][0].__setitem__("undeclared", 1),
        ),
        (
            "sealed_labels",
            SealedLabelArtifact,
            lambda payload: payload["labels"][0]["evidence_bundles"][0][
                "locations"
            ][0].__setitem__("undeclared", 1),
        ),
        (
            "predictions",
            PredictionArtifact,
            lambda payload: payload["predictions"][0].__setitem__("undeclared", 1),
        ),
        (
            "offline",
            OfflineEvaluationArtifact,
            lambda payload: payload["trials"][0]["labels"].__setitem__(
                "undeclared", 1
            ),
        ),
    ),
)
def test_from_dict_rejects_unknown_nested_fields(
    name: str,
    artifact_type: Any,
    mutate: Any,
) -> None:
    payload = copy.deepcopy(_artifact_set()[name].to_dict())
    mutate(payload)

    with pytest.raises(LabelSeparationError, match="unknown fields"):
        artifact_type.from_dict(payload)


@pytest.mark.parametrize(
    "encoded_mutation",
    (
        lambda canonical: canonical,
        lambda canonical: canonical + b"\n\n",
        lambda canonical: canonical + b"\r\n",
        lambda canonical: json.dumps(
            json.loads(canonical),
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n",
    ),
)
def test_loader_requires_canonical_bytes_and_one_newline(
    tmp_path: Path,
    encoded_mutation: Any,
) -> None:
    execution = _artifact_set()["execution"]
    target = tmp_path / "execution.json"
    target.write_bytes(encoded_mutation(execution.canonical_bytes()))

    with pytest.raises(LabelSeparationError, match="exactly one trailing newline"):
        separation.load_online_execution_artifact(target)


@pytest.mark.parametrize(
    ("before", "after", "match"),
    (
        (b'"document_id":0', b'"document_id":NaN', "non-finite"),
        (b'"document_id":0', b'"document_id":1e999', "non-finite"),
        (
            b'"document_id":0',
            b'"document_id":0,"document_id":0',
            "duplicate key",
        ),
    ),
)
def test_loader_rejects_nonfinite_numbers_and_nested_duplicate_keys(
    tmp_path: Path,
    before: bytes,
    after: bytes,
    match: str,
) -> None:
    execution = _artifact_set()["execution"]
    encoded = execution.canonical_bytes().replace(before, after, 1) + b"\n"
    target = tmp_path / "execution.json"
    target.write_bytes(encoded)

    with pytest.raises(LabelSeparationError, match=match):
        separation.load_online_execution_artifact(target)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_loader_rejects_symbolic_and_hard_links(
    tmp_path: Path,
    link_kind: str,
) -> None:
    execution = _artifact_set()["execution"]
    source = tmp_path / "source.json"
    source.write_bytes(execution.canonical_bytes() + b"\n")
    linked = tmp_path / "linked.json"
    if link_kind == "symlink":
        linked.symlink_to(source)
    else:
        os.link(source, linked)

    with pytest.raises(LabelSeparationError, match="symlink|hard-linked"):
        separation.load_online_execution_artifact(linked)


def test_writer_does_not_follow_a_preexisting_symbolic_link(tmp_path: Path) -> None:
    execution = _artifact_set()["execution"]
    destination = tmp_path / "destination.json"
    destination.write_text("sentinel", encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(destination)

    with pytest.raises(LabelSeparationError, match="already exists"):
        separation.write_online_execution_artifact(execution, linked)
    assert destination.read_text(encoding="utf-8") == "sentinel"


def test_secure_reader_toctou_rejection_crosses_the_public_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_changed_file(*args: Any, **kwargs: Any) -> bytes:
        raise ArtifactIntegrityError("artifact changed during read")

    monkeypatch.setattr(separation, "read_secure_regular_file", reject_changed_file)

    with pytest.raises(LabelSeparationError, match="changed during read"):
        separation.load_online_execution_artifact(tmp_path / "execution.json")


def test_from_dict_requires_json_arrays_not_python_tuple_shortcuts() -> None:
    execution = _artifact_set()["execution"]
    payload = execution.to_dict()
    payload["documents"] = tuple(payload["documents"])

    with pytest.raises(LabelSeparationError, match="JSON array"):
        OnlineExecutionArtifact.from_dict(payload)
