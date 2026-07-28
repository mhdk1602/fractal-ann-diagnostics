from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_confirmatory_analysis import _bound_input

import fractal_ann_diagnostics.confirmatory_input_operator as operator
from fractal_ann_diagnostics.confirmatory_analysis import ConfirmatoryAnalysisConfig
from fractal_ann_diagnostics.confirmatory_modeling import GeometryGainThresholds
from fractal_ann_diagnostics.sealed_online_execution import (
    OnlineOutputPin,
    SealedOnlineResultReceipt,
)
from fractal_ann_diagnostics.study import (
    EVIDENCE_CORPORA,
    FIXED_CORPORA,
    REGISTERED_ACTION_SET,
)


def _digest(value: bytes | str) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _analysis_config() -> ConfirmatoryAnalysisConfig:
    return ConfirmatoryAnalysisConfig(
        fixed_corpora=FIXED_CORPORA,
        evidence_corpora=EVIDENCE_CORPORA,
        action_set=REGISTERED_ACTION_SET,
        static_comparator_action="hnsw-high",
        low_geometry=(("lid", 1.0), ("instability", 0.1)),
        high_geometry=(("lid", 9.0), ("instability", 0.9)),
        geometry_gain_thresholds=GeometryGainThresholds(
            log_loss_reduction=0.001,
            brier_score_reduction=0.001,
            auprc_gain=0.001,
        ),
        selected_families_per_corpus=25,
        nested_rows_per_family=3,
        bootstrap_replicates=10_000,
        bootstrap_seed=20260713,
    )


def _inputs():
    return _bound_input(_analysis_config())


def _config(tmp_path: Path) -> operator.ConfirmatoryInputOperatorConfig:
    namespace = tmp_path / "suite"
    artifact_root = tmp_path / "artifacts"
    namespace.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    evidence = []
    for corpus_id in FIXED_CORPORA:
        paths = {
            name: tmp_path / f"{corpus_id}-{name}.json"
            for name in ("completion", "anchor-record", "anchor-receipt", "decryption")
        }
        evidence.append(
            operator.CorpusEvidenceLocation(
                corpus_id=corpus_id,
                prediction_completion_receipt_uri=paths["completion"].as_uri(),
                prediction_completion_anchor_record_uri=paths["anchor-record"].as_uri(),
                prediction_completion_anchor_receipt_uri=paths["anchor-receipt"].as_uri(),
                timelock_decryption_receipt_uri=paths["decryption"].as_uri(),
            )
        )
    return operator.ConfirmatoryInputOperatorConfig(
        suite_namespace_uri=namespace.as_uri(),
        manifest_uri=(tmp_path / "manifest.json").as_uri(),
        sealed_run_receipt_uri=(tmp_path / "run.json").as_uri(),
        artifact_verification_receipt_uri=(tmp_path / "verification.json").as_uri(),
        artifact_root_uri=artifact_root.as_uri(),
        corpus_evidence=tuple(reversed(evidence)),
    )


def _members(tmp_path: Path) -> tuple[operator.ConfirmatoryInputMember, ...]:
    required = (
        "action-panel",
        "action-panel-admission",
        "prediction-completion-anchor-receipt",
        "prediction-completion-anchor-record",
        "prediction-completion-receipt",
        "predictions",
        "released-sealed-labels",
        "sealed-online-result",
        "timelock-decryption-receipt",
    )
    rows = []
    for corpus_id in FIXED_CORPORA:
        for role in required:
            encoded = f'{{"corpus":"{corpus_id}","role":"{role}"}}\n'.encode()
            path = tmp_path / f"{corpus_id}-{role}.json"
            path.write_bytes(encoded)
            rows.append(
                operator._file_member(
                    path,
                    role=role,
                    corpus_id=corpus_id,
                    semantic_sha256=_digest(encoded[:-1]),
                )
            )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                (row.corpus_id or "").encode(),
                row.role.encode(),
                row.uri.encode(),
            ),
        )
    )


class _Token:
    def __init__(self) -> None:
        self.state = SimpleNamespace(
            suite_attempt_id=_digest("attempt"),
            record_sha256=_digest("labels-released-state"),
        )
        self.descriptor_sha256 = _digest("descriptor")
        self.current_checks = 0

    def assert_current(self) -> None:
        self.current_checks += 1


def test_config_is_closed_canonical_and_orders_fixed_corpora(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = tmp_path / "operator-config.json"
    path.write_bytes(config.canonical_bytes() + b"\n")

    loaded = operator.load_confirmatory_input_operator_config(path)
    assert loaded == config
    assert tuple(row.corpus_id for row in loaded.corpus_evidence) == FIXED_CORPORA

    payload = config.to_dict()
    payload["estimand"] = "caller-selected"
    with pytest.raises(operator.ConfirmatoryInputOperatorError, match="unexpected"):
        operator.ConfirmatoryInputOperatorConfig.from_dict(payload)


def test_config_rejects_missing_corpus_and_reused_evidence_uri(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(operator.ConfirmatoryInputOperatorError, match="each fixed corpus"):
        replace(config, corpus_evidence=config.corpus_evidence[:-1])

    first, second, *rest = config.corpus_evidence
    reused = replace(
        second,
        prediction_completion_receipt_uri=first.prediction_completion_receipt_uri,
    )
    with pytest.raises(operator.ConfirmatoryInputOperatorError, match="reuse"):
        replace(config, corpus_evidence=(first, reused, *rest))


def test_file_and_semantic_label_identities_cannot_be_substituted(tmp_path: Path) -> None:
    members = _members(tmp_path)
    label = next(row for row in members if row.role == "released-sealed-labels")
    assert label.semantic_sha256 != label.file_sha256
    operator._verify_member_files(members)

    substituted = tuple(
        replace(row, file_sha256=row.semantic_sha256) if row == label else row for row in members
    )
    with pytest.raises(operator.ConfirmatoryInputOperatorError, match="changed"):
        operator._verify_member_files(substituted)


def test_materialization_is_exclusive_and_detects_member_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    members = _members(tmp_path)
    token = _Token()
    artifact_path = tmp_path / f"{inputs.manifest_sha256}.confirmatory-input.json"
    receipt_path = tmp_path / f"{inputs.manifest_sha256}.confirmatory-input-receipt.json"
    monkeypatch.setattr(operator, "_assemble_input", lambda *args, **kwargs: (inputs, members))
    monkeypatch.setattr(operator, "confirmatory_input_path", lambda value: artifact_path)
    monkeypatch.setattr(operator, "confirmatory_input_receipt_path", lambda value: receipt_path)

    created = operator.materialize_confirmatory_input(SimpleNamespace(), token)  # type: ignore[arg-type]
    assert created.artifact_path.read_bytes() == inputs.canonical_bytes() + b"\n"
    assert created.receipt.artifact_sha256 == inputs.artifact_sha256
    assert token.current_checks == 1

    with pytest.raises(operator.ConfirmatoryInputOperatorError, match="cannot persist"):
        operator.materialize_confirmatory_input(SimpleNamespace(), token)  # type: ignore[arg-type]

    changed = Path(members[0].uri.removeprefix("file://"))
    changed.write_bytes(b"changed\n")
    with pytest.raises(operator.ConfirmatoryInputOperatorError, match="changed"):
        operator.load_materialized_confirmatory_input(
            SimpleNamespace(),
            token,  # type: ignore[arg-type]
        )


def test_receipt_is_reserved_if_artifact_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    members = _members(tmp_path)
    token = _Token()
    artifact_path = tmp_path / f"{inputs.manifest_sha256}.confirmatory-input.json"
    receipt_path = tmp_path / f"{inputs.manifest_sha256}.confirmatory-input-receipt.json"
    monkeypatch.setattr(operator, "_assemble_input", lambda *args, **kwargs: (inputs, members))
    monkeypatch.setattr(operator, "confirmatory_input_path", lambda value: artifact_path)
    monkeypatch.setattr(operator, "confirmatory_input_receipt_path", lambda value: receipt_path)
    real_write = operator.write_exclusive_receipt_bytes
    calls = 0

    def fail_second(encoded: bytes, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise operator.ArtifactIntegrityError("injected artifact failure")
        real_write(encoded, target)

    monkeypatch.setattr(operator, "write_exclusive_receipt_bytes", fail_second)
    with pytest.raises(operator.ConfirmatoryInputOperatorError, match="cannot persist"):
        operator.materialize_confirmatory_input(SimpleNamespace(), token)  # type: ignore[arg-type]
    assert receipt_path.exists()
    assert not artifact_path.exists()


def test_online_directory_rejects_extra_file_after_attested_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "online"
    root.mkdir(mode=0o700)
    manifest_digest = _digest("manifest")
    corpus_id = FIXED_CORPORA[0]
    attempt_path = root / f"{manifest_digest}.sealed-online-attempt.json"
    result_path = root / f"{manifest_digest}.sealed-online-result-receipt.json"
    attempt_bytes = b'{"attempt":true}\n'
    result_bytes = b'{"result":true}\n'
    attempt_path.write_bytes(attempt_bytes)
    result_path.write_bytes(result_bytes)
    attempt_sha = _digest("attempt-semantic")
    execution_sha = _digest("execution")
    output_roles = (
        "action-panel",
        "action-panel-admission",
        "audit-chain",
        "cache-preparation",
        "execution-order",
        "predictions",
    )
    pins = []
    for role in output_roles:
        encoded = f"{role}\n".encode()
        path = root / f"{manifest_digest}.{role}.json"
        path.write_bytes(encoded)
        pins.append(
            OnlineOutputPin(
                role=role,
                filename=path.name,
                byte_count=len(encoded),
                file_sha256=_digest(encoded),
                semantic_sha256=_digest(f"{role}-semantic"),
            )
        )
    result = SealedOnlineResultReceipt(
        manifest_sha256=manifest_digest,
        run_receipt_sha256=_digest("run"),
        execution_artifact_sha256=execution_sha,
        attempt_receipt_sha256=attempt_sha,
        audit_head_sha256=next(row.semantic_sha256 for row in pins if row.role == "audit-chain"),
        audit_record_count=1,
        outputs=tuple(sorted(pins, key=lambda row: row.role.encode())),
    )
    runtime_path = root / operator.RUNTIME_ATTESTATION_RECEIPT_FILENAME
    marker_path = root / operator.RUNTIME_INVOCATION_MARKER_FILENAME
    command_path = root / operator.PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME
    runtime_path.write_bytes(b"runtime\n")
    marker_path.write_bytes(b"marker\n")
    command_path.write_bytes(b"command\n")
    runtime_sha = _digest("runtime-semantic")
    command_sha = _digest("command-semantic")
    prediction_pin = next(row for row in pins if row.role == "predictions")
    panel_pin = next(row for row in pins if row.role == "action-panel")
    admission_pin = next(row for row in pins if row.role == "action-panel-admission")
    closure_values = {
        "corpus_id": corpus_id,
        "output_uri": root.as_uri(),
        "execution_artifact_sha256": execution_sha,
        "attempt_receipt_sha256": attempt_sha,
        "attempt_file_sha256": _digest(attempt_bytes),
        "result_receipt_sha256": result.receipt_sha256,
        "result_file_sha256": _digest(result_bytes),
        "runtime_attestation_receipt_sha256": runtime_sha,
        "runtime_attestation_receipt_file_sha256": _digest(runtime_path.read_bytes()),
        "runtime_invocation_marker_sha256": _digest(marker_path.read_bytes()),
        "runtime_invocation_marker_file_sha256": _digest(marker_path.read_bytes()),
        "production_command_attempt_sha256": command_sha,
        "production_command_attempt_file_sha256": _digest(command_path.read_bytes()),
        "prediction_artifact_sha256": prediction_pin.semantic_sha256,
        "prediction_file_sha256": prediction_pin.file_sha256,
        "action_panel_artifact_sha256": panel_pin.semantic_sha256,
        "action_panel_file_sha256": panel_pin.file_sha256,
        "action_panel_admission_receipt_sha256": admission_pin.semantic_sha256,
        "action_panel_admission_file_sha256": admission_pin.file_sha256,
        "audit_head_sha256": next(row.semantic_sha256 for row in pins if row.role == "audit-chain"),
        "audit_file_sha256": next(row.file_sha256 for row in pins if row.role == "audit-chain"),
        "cache_preparation_receipt_sha256": next(
            row.semantic_sha256 for row in pins if row.role == "cache-preparation"
        ),
        "cache_preparation_file_sha256": next(
            row.file_sha256 for row in pins if row.role == "cache-preparation"
        ),
        "execution_order_receipt_sha256": next(
            row.semantic_sha256 for row in pins if row.role == "execution-order"
        ),
        "execution_order_file_sha256": next(
            row.file_sha256 for row in pins if row.role == "execution-order"
        ),
    }
    closure = SimpleNamespace(**closure_values)
    monkeypatch.setattr(
        operator,
        "load_sealed_online_attempt_receipt",
        lambda path: SimpleNamespace(receipt_sha256=attempt_sha),
    )
    monkeypatch.setattr(operator, "load_sealed_online_result_receipt", lambda path: result)
    monkeypatch.setattr(operator, "verify_sealed_online_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        operator,
        "load_runtime_attestation_receipt",
        lambda path: SimpleNamespace(receipt_sha256=runtime_sha),
    )
    monkeypatch.setattr(
        operator,
        "load_production_corpus_command_attempt",
        lambda path: SimpleNamespace(
            receipt_sha256=command_sha,
            manifest_sha256=manifest_digest,
        ),
    )
    monkeypatch.setattr(
        operator,
        "load_prediction_artifact",
        lambda path: SimpleNamespace(artifact_sha256=prediction_pin.semantic_sha256),
    )
    monkeypatch.setattr(
        operator,
        "load_action_panel_artifact",
        lambda path: SimpleNamespace(artifact_sha256=panel_pin.semantic_sha256),
    )
    monkeypatch.setattr(
        operator,
        "load_action_panel_admission_receipt",
        lambda path: SimpleNamespace(receipt_sha256=admission_pin.semantic_sha256),
    )

    operator._assert_online_directory(closure, manifest_digest=manifest_digest)
    (root / "unattested-extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(operator.ConfirmatoryInputOperatorError, match="membership changed"):
        operator._assert_online_directory(closure, manifest_digest=manifest_digest)


def test_github_gate_requests_exact_labels_released_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    token = _Token()
    observed = {}

    class FakeVerified:
        pass

    verified = FakeVerified()
    monkeypatch.setattr(operator, "VerifiedSuiteLabelsReleased", FakeVerified)
    monkeypatch.setattr(
        operator,
        "GitHubSuiteEvidenceVerifier",
        lambda namespace: observed.setdefault("verifier_namespace", namespace),
    )

    def verify(namespace, *, verifier, expected_state):
        observed.update(
            namespace=namespace,
            verifier=verifier,
            expected_state=expected_state,
        )
        return verified

    monkeypatch.setattr(operator, "verify_suite_state", verify)
    assert operator._github_verified_labels(config) is verified
    assert observed["expected_state"] == "LABELS_RELEASED"
    assert observed["namespace"] == Path(config.suite_namespace_uri.removeprefix("file://"))
    del token


def test_one_shot_host_operator_is_fail_closed() -> None:
    with pytest.raises(
        operator.ConfirmatoryInputOperatorError,
        match="C1-pinned offline container",
    ):
        operator.run_materialized_confirmatory_analysis_once(
            SimpleNamespace(),
            _Token(),  # type: ignore[arg-type]
        )
