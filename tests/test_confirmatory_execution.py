from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.confirmatory_execution as execution
from fractal_ann_diagnostics.confirmatory_analysis import ConfirmatoryAnalysisError

MANIFEST_SHA256 = "a" * 64
RUN_RECEIPT_SHA256 = "b" * 64
INPUT_SHA256 = "c" * 64
MODEL_SUITE_SHA256 = "d" * 64
RUNNER_IDENTITY = "confirmatory-runner@example.test"


class _FakeInput:
    def __init__(self, results_store: str) -> None:
        self.frozen_manifest = {"sealed_execution": {"results_store": results_store}}
        self.manifest_sha256 = MANIFEST_SHA256
        self.run_receipt_sha256 = RUN_RECEIPT_SHA256
        self.artifact_sha256 = INPUT_SHA256
        self.run_receipt = SimpleNamespace(runner_identity=RUNNER_IDENTITY)
        self.admission_checks = 0

    def assert_model_suite_admitted(self, suite: object) -> None:
        self.admission_checks += 1
        if getattr(suite, "suite_digest", None) != MODEL_SUITE_SHA256:
            raise ConfirmatoryAnalysisError("model suite is not admitted")


class _FakeSuite:
    suite_digest = MODEL_SUITE_SHA256


class _FakeResult:
    manifest_sha256 = MANIFEST_SHA256
    run_receipt_sha256 = RUN_RECEIPT_SHA256
    confirmatory_input_artifact_sha256 = INPUT_SHA256
    model_suite_sha256 = MODEL_SUITE_SHA256

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "confirmatory_input_artifact_sha256": (self.confirmatory_input_artifact_sha256),
                "manifest_sha256": self.manifest_sha256,
                "h1": {"passed": False},
                "h2": {"passed": False},
                "h3": {"passed": False},
                "model_suite_sha256": self.model_suite_sha256,
                "primary_claim_passed": False,
                "run_receipt_sha256": self.run_receipt_sha256,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _install_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execution, "ConfirmatoryInputArtifact", _FakeInput)
    monkeypatch.setattr(execution, "FrozenModelSuite", _FakeSuite)
    monkeypatch.setattr(execution, "ConfirmatoryResultArtifact", _FakeResult)
    monkeypatch.setattr(
        execution,
        "_typed_confirmatory_result",
        lambda payload: _FakeResult(),
    )
    monkeypatch.setattr(
        execution,
        "_require_suite_labels_released",
        lambda *args, **kwargs: None,
    )


def _inputs(tmp_path: Path) -> _FakeInput:
    results = tmp_path / "sealed-results"
    results.mkdir(mode=0o700)
    return _FakeInput(results.as_uri())


def test_analysis_gate_rejects_bare_in_memory_object_before_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "ConfirmatoryInputArtifact", _FakeInput)
    monkeypatch.setattr(execution, "FrozenModelSuite", _FakeSuite)
    monkeypatch.setattr(execution, "ConfirmatoryResultArtifact", _FakeResult)
    inputs = _inputs(tmp_path)
    with pytest.raises(ConfirmatoryAnalysisError, match="canonical files"):
        execution.run_confirmatory_analysis_once(
            inputs,
            suite=_FakeSuite(),
            verified_labels_released=object(),
        )
    assert not execution.confirmatory_attempt_path(inputs).exists()


def test_attempt_is_durable_before_computation_and_result_is_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _inputs(tmp_path)
    suite = _FakeSuite()
    result = _FakeResult()
    observations: list[execution.ConfirmatoryAnalysisAttemptReceipt] = []

    def compute(observed_inputs: object, *, suite: object) -> _FakeResult:
        assert observed_inputs is inputs
        assert suite is not None
        attempt_target = execution.confirmatory_attempt_path(inputs)
        assert attempt_target.is_file()
        assert not execution.confirmatory_result_path(inputs).exists()
        observations.append(execution.load_confirmatory_analysis_attempt_receipt(attempt_target))
        return result

    monkeypatch.setattr(execution, "run_confirmatory_analysis", compute)

    assert execution.run_confirmatory_analysis_once(inputs, suite=suite) is result

    assert len(observations) == 1
    attempt = observations[0]
    assert attempt.manifest_sha256 == MANIFEST_SHA256
    assert attempt.run_receipt_sha256 == RUN_RECEIPT_SHA256
    assert attempt.confirmatory_input_artifact_sha256 == INPUT_SHA256
    assert attempt.model_suite_sha256 == MODEL_SUITE_SHA256
    assert attempt.runner_identity == RUNNER_IDENTITY
    assert attempt.result_uri == execution.confirmatory_result_path(inputs).as_uri()

    result_receipt = execution.load_confirmatory_analysis_result_receipt(
        execution.confirmatory_result_receipt_path(inputs)
    )
    assert result_receipt.attempt_receipt_sha256 == attempt.receipt_sha256
    assert result_receipt.result_artifact_sha256 == result.artifact_sha256
    assert (
        execution.load_confirmatory_result_artifact_bytes(
            execution.confirmatory_result_path(inputs),
            result_receipt_path=execution.confirmatory_result_receipt_path(inputs),
            attempt_receipt_path=execution.confirmatory_attempt_path(inputs),
        )
        == result.canonical_bytes()
    )

    assert execution.confirmatory_attempt_path(inputs).read_bytes() == (
        attempt.canonical_bytes() + b"\n"
    )
    assert execution.confirmatory_result_receipt_path(inputs).read_bytes() == (
        result_receipt.canonical_bytes() + b"\n"
    )


def test_preexisting_attempt_blocks_before_analysis_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _inputs(tmp_path)
    suite = _FakeSuite()
    attempt = execution._attempt_receipt(inputs, suite=suite)
    execution.confirmatory_attempt_path(inputs).write_bytes(attempt.canonical_bytes() + b"\n")
    compute_calls = 0

    def compute(*args: object, **kwargs: object) -> _FakeResult:
        nonlocal compute_calls
        compute_calls += 1
        return _FakeResult()

    monkeypatch.setattr(execution, "run_confirmatory_analysis", compute)

    with pytest.raises(ConfirmatoryAnalysisError, match="attempt was not admitted"):
        execution.run_confirmatory_analysis_once(inputs, suite=suite)

    assert compute_calls == 0


def test_failed_analysis_retains_attempt_and_permanently_blocks_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _inputs(tmp_path)
    suite = _FakeSuite()
    compute_calls = 0

    class AnalysisCrashed(RuntimeError):
        pass

    def compute(*args: object, **kwargs: object) -> _FakeResult:
        nonlocal compute_calls
        compute_calls += 1
        raise AnalysisCrashed("simulated runner crash")

    monkeypatch.setattr(execution, "run_confirmatory_analysis", compute)

    with pytest.raises(AnalysisCrashed, match="simulated runner crash"):
        execution.run_confirmatory_analysis_once(inputs, suite=suite)

    attempt_target = execution.confirmatory_attempt_path(inputs)
    assert attempt_target.is_file()
    assert not execution.confirmatory_result_path(inputs).exists()
    assert not execution.confirmatory_result_receipt_path(inputs).exists()

    with pytest.raises(ConfirmatoryAnalysisError, match="attempt was not admitted"):
        execution.run_confirmatory_analysis_once(inputs, suite=suite)
    assert compute_calls == 1


def test_concurrent_caller_cannot_compute_after_first_attempt_is_admitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _inputs(tmp_path)
    suite = _FakeSuite()
    entered = threading.Event()
    release = threading.Event()
    compute_calls = 0
    thread_errors: list[BaseException] = []

    def compute(*args: object, **kwargs: object) -> _FakeResult:
        nonlocal compute_calls
        compute_calls += 1
        entered.set()
        if not release.wait(timeout=5):
            raise AssertionError("test did not release the admitted analysis")
        return _FakeResult()

    def first_caller() -> None:
        try:
            execution.run_confirmatory_analysis_once(inputs, suite=suite)
        except BaseException as exc:  # pragma: no cover - reported by the main thread
            thread_errors.append(exc)

    monkeypatch.setattr(execution, "run_confirmatory_analysis", compute)
    thread = threading.Thread(target=first_caller)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(ConfirmatoryAnalysisError, match="attempt was not admitted"):
            execution.run_confirmatory_analysis_once(inputs, suite=suite)
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert thread_errors == []
    assert compute_calls == 1


@pytest.mark.parametrize("scheme", ["s3", "gs"])
def test_remote_result_store_is_explicitly_unsupported(
    scheme: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _FakeInput(f"{scheme}://immutable-bucket/study-results")

    with pytest.raises(ConfirmatoryAnalysisError, match="unsupported remote store"):
        execution.confirmatory_attempt_path(inputs)


@pytest.mark.parametrize(
    "uri",
    [
        "file://localhost/tmp/results",
        "file:///tmp/results?version=1",
        "file:///tmp/results#fragment",
        "file:///tmp/../results",
        "file:///tmp/%2E%2E/results",
        "file:///tmp/results%00poison",
        "file:/tmp/results",
        "file:///tmp//results",
        "file:///tmp/results/",
    ],
)
def test_noncanonical_or_hostile_file_store_is_rejected(
    uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _FakeInput(uri)

    with pytest.raises(ConfirmatoryAnalysisError):
        execution.confirmatory_result_path(inputs)


def test_attempt_loader_rejects_noncanonical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _inputs(tmp_path)
    attempt = execution._attempt_receipt(inputs, suite=_FakeSuite())
    target = execution.confirmatory_attempt_path(inputs)
    target.write_text(
        json.dumps(attempt.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfirmatoryAnalysisError, match="bytes are not canonical"):
        execution.load_confirmatory_analysis_attempt_receipt(target)


def test_attempt_loader_rejects_receipt_outside_manifest_derived_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _inputs(tmp_path)
    attempt = execution._attempt_receipt(inputs, suite=_FakeSuite())
    target = execution.confirmatory_attempt_path(inputs).with_name("relocated.json")
    target.write_bytes(attempt.canonical_bytes() + b"\n")

    with pytest.raises(ConfirmatoryAnalysisError, match="manifest-derived"):
        execution.load_confirmatory_analysis_attempt_receipt(target)


def test_result_loader_rejects_bytes_not_bound_by_detached_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        execution,
        "run_confirmatory_analysis",
        lambda *args, **kwargs: _FakeResult(),
    )
    execution.run_confirmatory_analysis_once(inputs, suite=_FakeSuite())
    result_target = execution.confirmatory_result_path(inputs)
    result_target.write_bytes(b'{"tampered":true}\n')

    with pytest.raises(ConfirmatoryAnalysisError, match="does not match"):
        execution.load_confirmatory_result_artifact_bytes(
            result_target,
            result_receipt_path=execution.confirmatory_result_receipt_path(inputs),
            attempt_receipt_path=execution.confirmatory_attempt_path(inputs),
        )


def test_result_loader_rejects_self_consistent_result_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        execution,
        "run_confirmatory_analysis",
        lambda *args, **kwargs: _FakeResult(),
    )
    execution.run_confirmatory_analysis_once(inputs, suite=_FakeSuite())
    attempt_target = execution.confirmatory_attempt_path(inputs)
    attempt = execution.load_confirmatory_analysis_attempt_receipt(attempt_target)
    result_target = execution.confirmatory_result_path(inputs)
    substituted = json.loads(result_target.read_text(encoding="utf-8"))
    substituted["manifest_sha256"] = "9" * 64
    substituted_bytes = json.dumps(
        substituted,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    result_target.write_bytes(substituted_bytes + b"\n")
    substituted_receipt = execution.ConfirmatoryAnalysisResultReceipt(
        manifest_sha256=attempt.manifest_sha256,
        attempt_receipt_sha256=attempt.receipt_sha256,
        result_artifact_sha256=hashlib.sha256(substituted_bytes).hexdigest(),
        result_uri=result_target.as_uri(),
    )
    result_receipt_target = execution.confirmatory_result_receipt_path(inputs)
    result_receipt_target.write_bytes(substituted_receipt.canonical_bytes() + b"\n")

    with pytest.raises(
        ConfirmatoryAnalysisError,
        match="result manifest_sha256 does not match the admitted attempt",
    ):
        execution.load_confirmatory_result_artifact_bytes(
            result_target,
            result_receipt_path=result_receipt_target,
            attempt_receipt_path=attempt_target,
        )


def test_result_loader_rejects_fake_attempt_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        execution,
        "run_confirmatory_analysis",
        lambda *args, **kwargs: _FakeResult(),
    )
    execution.run_confirmatory_analysis_once(inputs, suite=_FakeSuite())
    result_target = execution.confirmatory_result_path(inputs)
    admitted_receipt = execution.load_confirmatory_analysis_result_receipt(
        execution.confirmatory_result_receipt_path(inputs)
    )
    substituted_receipt = execution.ConfirmatoryAnalysisResultReceipt(
        manifest_sha256=admitted_receipt.manifest_sha256,
        attempt_receipt_sha256="9" * 64,
        result_artifact_sha256=admitted_receipt.result_artifact_sha256,
        result_uri=admitted_receipt.result_uri,
    )
    result_receipt_target = execution.confirmatory_result_receipt_path(inputs)
    result_receipt_target.write_bytes(substituted_receipt.canonical_bytes() + b"\n")

    with pytest.raises(
        ConfirmatoryAnalysisError,
        match="attempt_receipt_sha256 does not match the admitted attempt",
    ):
        execution.load_confirmatory_result_artifact_bytes(
            result_target,
            result_receipt_path=result_receipt_target,
            attempt_receipt_path=execution.confirmatory_attempt_path(inputs),
        )


def test_result_loader_ignores_forged_in_memory_receipt_for_nested_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_types(monkeypatch)
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        execution,
        "run_confirmatory_analysis",
        lambda *args, **kwargs: _FakeResult(),
    )
    execution.run_confirmatory_analysis_once(inputs, suite=_FakeSuite())
    result_target = execution.confirmatory_result_path(inputs)
    result_receipt_target = execution.confirmatory_result_receipt_path(inputs)
    durable_receipt = execution.load_confirmatory_analysis_result_receipt(result_receipt_target)
    substituted = json.loads(result_target.read_text(encoding="utf-8"))
    original_bindings = {
        name: substituted[name]
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "confirmatory_input_artifact_sha256",
            "model_suite_sha256",
        )
    }
    substituted["h1"]["passed"] = True
    substituted["h2"]["passed"] = True
    substituted["h3"]["passed"] = True
    assert all(substituted[name] == value for name, value in original_bindings.items())
    substituted_bytes = json.dumps(
        substituted,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    result_target.write_bytes(substituted_bytes + b"\n")
    forged_receipt = execution.ConfirmatoryAnalysisResultReceipt(
        manifest_sha256=durable_receipt.manifest_sha256,
        attempt_receipt_sha256=durable_receipt.attempt_receipt_sha256,
        result_artifact_sha256=hashlib.sha256(substituted_bytes).hexdigest(),
        result_uri=durable_receipt.result_uri,
    )
    assert forged_receipt.result_artifact_sha256 != (durable_receipt.result_artifact_sha256)

    with pytest.raises(
        ConfirmatoryAnalysisError,
        match="does not match its detached receipt",
    ):
        execution.load_confirmatory_result_artifact_bytes(
            result_target,
            result_receipt_path=result_receipt_target,
            attempt_receipt_path=execution.confirmatory_attempt_path(inputs),
        )
