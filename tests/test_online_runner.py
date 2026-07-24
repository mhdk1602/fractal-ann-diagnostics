from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import fractal_ann_diagnostics.controller as controller_module
from fractal_ann_diagnostics.artifact_integrity import (
    ArtifactVerificationReceipt,
    VerifiedArtifact,
)
from fractal_ann_diagnostics.audit import (
    VerifiedProvenanceRegistry,
    verify_audit_chain,
)
from fractal_ann_diagnostics.confirmatory_modeling import REGISTERED_FEATURE_SCHEMA
from fractal_ann_diagnostics.controller import (
    ControllerConfig,
    GovernedRetriever,
    RuleController,
)
from fractal_ann_diagnostics.corpora import (
    CorpusDocument,
    EvidenceQuery,
    NormalizedCorpus,
)
from fractal_ann_diagnostics.label_separation import (
    OnlineDocument,
    OnlineExecutionArtifact,
    OnlineTrial,
)
from fractal_ann_diagnostics.online_runner import (
    CACHE_PREPARATION_RECEIPT_SCHEMA,
    EXECUTION_ORDER_RECEIPT_SCHEMA,
    REGISTERED_ACTION_SET,
    FrozenFeatureContext,
    OnlineRunnerError,
    OnlineTrialRuntime,
    load_cache_preparation_receipt,
    load_execution_order_receipt,
    loads_cache_preparation_receipt,
    loads_execution_order_receipt,
    portable_balanced_action_orders,
    run_online_action_matrix,
    write_cache_preparation_receipt,
    write_execution_order_receipt,
)
from fractal_ann_diagnostics.policy import AuthorizationPolicy
from fractal_ann_diagnostics.study import SealedRunReceipt

_MANIFEST_SHA256 = "a" * 64
_PARTITION_SHA256 = sha256(b"query-partition-audit").hexdigest()
_PSEUDONYM_KEY = b"online-runner-pseudonym-key-material-32-bytes"
_COMPONENTS = (
    "application",
    "controller",
    "corpus",
    "embedding",
    "index",
    "policy",
)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _corpus(*, trials: int = 2) -> NormalizedCorpus:
    documents = tuple(
        CorpusDocument(
            document_id=document_id,
            external_id=f"document-{document_id}",
            title=f"Document {document_id}",
            text=f"deterministic text {document_id}",
            source_uri=f"fixture://document/{document_id}",
            content_hash=f"sha256:{_digest(f'document-content-{document_id}')}",
        )
        for document_id in range(12)
    )
    queries = tuple(
        EvidenceQuery(
            query_id=f"query-{index}",
            query_family=f"family-{index}",
            text=f"find document family {index}",
            corpus="scifact",
            stage="sealed",
            answer=None,
            gold_evidence=None,
        )
        for index in range(trials)
    )
    return NormalizedCorpus(
        name="scifact",
        stage="sealed",
        documents=documents,
        queries=queries,
    )


def _execution(corpus: NormalizedCorpus) -> OnlineExecutionArtifact:
    return OnlineExecutionArtifact(
        key_id="online-fixture-key",
        corpus=corpus.name,
        stage=corpus.stage,
        documents=tuple(
            OnlineDocument(
                document_id=document.document_id,
                external_id=document.external_id,
                title=document.title,
                text=document.text,
                source_uri=document.source_uri,
                content_hash=document.content_hash,
            )
            for document in corpus.documents
        ),
        trials=tuple(
            OnlineTrial(
                trial_key=_digest(f"trial:{query.query_id}"),
                family_key=_digest(f"family:{query.query_family}"),
                text=query.text,
                corpus=corpus.name,
                stage=corpus.stage,
            )
            for query in corpus.queries
        ),
    )


def _verified_artifact(component: str) -> VerifiedArtifact:
    digest = _digest(component)
    return VerifiedArtifact(
        artifact_id=f"{component}-artifact",
        relative_path=f"artifacts/{component}.bin",
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


def _registry(corpus: NormalizedCorpus) -> VerifiedProvenanceRegistry:
    receipt = ArtifactVerificationReceipt(
        manifest_sha256=_MANIFEST_SHA256,
        artifacts=tuple(_verified_artifact(component) for component in _COMPONENTS),
    )
    return VerifiedProvenanceRegistry(
        corpus=corpus,
        verification_receipt=receipt,
        component_artifact_ids={component: f"{component}-artifact" for component in _COMPONENTS},
    )


def _run_receipt() -> SealedRunReceipt:
    return SealedRunReceipt(
        manifest_sha256=_MANIFEST_SHA256,
        protocol_version="0.3.0",
        started_at_utc="2026-07-14T12:00:00+00:00",
        runner_identity="online-test-runner",
        code_commit="c" * 40,
        runner_image=f"ghcr.io/example/runner@sha256:{'d' * 64}",
        protocol_registration_receipt_uri="file:///controlled/protocol-receipt.json",
        protocol_registration_receipt_sha256="e" * 64,
        protocol_registration_record_uri="file:///controlled/protocol-record.json",
        verification_receipt_uri="file:///controlled/verification-receipt.json",
        verification_receipt_sha256="f" * 64,
        receipt_uri="file:///controlled/run-receipt.json",
    )


def _retriever(
    corpus: NormalizedCorpus,
    registry: VerifiedProvenanceRegistry,
    *,
    policy: object | None = None,
) -> GovernedRetriever:
    vectors = np.random.default_rng(71).normal(size=(len(corpus.documents), 4)).astype("float32")
    selected_policy = policy or AuthorizationPolicy(
        roles=("analyst",),
        visibility=np.ones((1, len(corpus.documents)), dtype=bool),
        version="policy-registered",
        document_universe_sha256=registry.document_universe_sha256,
    )
    controller = RuleController(
        ControllerConfig(
            low_ef=8,
            high_ef=16,
            probe_k=8,
            exact_scan_threshold=0,
            high_effort_threshold=0.98,
            exact_threshold=1.0,
        )
    )
    return GovernedRetriever(
        vectors,
        selected_policy,  # type: ignore[arg-type]
        role="analyst",
        expected_document_universe_sha256=registry.document_universe_sha256,
        controller=controller,
        hnsw_seed=19,
    )


def _runtimes(
    execution: OnlineExecutionArtifact,
    dimension: int = 4,
) -> dict[str, OnlineTrialRuntime]:
    context = FrozenFeatureContext(
        version_lag=2.0,
        backend="hnswlib-0.8.0",
        drift_family="centroid-shift",
        policy_complexity=4.0,
    )
    return {
        trial.trial_key: OnlineTrialRuntime(
            active_query_vector=np.random.default_rng(index + 3).normal(size=dimension),
            current_truth_query_vector=np.random.default_rng(index + 103).normal(size=dimension),
            feature_context=context,
            environment={"tenant": "research", "trial_slot": index},
        )
        for index, trial in enumerate(execution.trials)
    }


def _fixed_time(_trial: str, _action: str, sequence: int) -> str:
    return f"2026-07-14T12:00:{sequence:02d}+00:00"


class _ShardedExecutionFixture:
    """Control-only execution view whose corpus rows are absent by construction."""

    def __init__(
        self,
        execution: OnlineExecutionArtifact,
        *,
        document_universe_sha256: str,
    ) -> None:
        self.corpus = execution.corpus
        self.stage = execution.stage
        self.trials = execution.trials
        self.document_count = len(execution.documents)
        self.document_universe_sha256 = document_universe_sha256
        self._canonical = json.dumps(
            {
                "corpus": self.corpus,
                "document_count": self.document_count,
                "document_universe_sha256": document_universe_sha256,
                "stage": self.stage,
                "trial_keys": sorted(trial.trial_key for trial in self.trials),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @property
    def artifact_sha256(self) -> str:
        return sha256(self._canonical).hexdigest()

    def canonical_bytes(self) -> bytes:
        return self._canonical


def _run(
    *,
    trials: int = 2,
    retriever: GovernedRetriever | None = None,
):
    corpus = _corpus(trials=trials)
    execution = _execution(corpus)
    registry = _registry(corpus)
    active_retriever = retriever or _retriever(corpus, registry)
    output = run_online_action_matrix(
        execution=execution,
        run_receipt=_run_receipt(),
        retriever=active_retriever,
        provenance_registry=registry,
        trial_runtimes=_runtimes(execution),
        permutation_seed=20260714,
        expected_policy_version="policy-registered",
        query_partition_audit_sha256=_PARTITION_SHA256,
        pseudonym_key=_PSEUDONYM_KEY,
        pseudonym_key_id="online-test-pseudonym-key",
        k=3,
        occurred_at_factory=_fixed_time,
    )
    return output, execution, active_retriever


def test_runner_accepts_a_sharded_execution_without_materializing_documents() -> None:
    corpus = _corpus(trials=2)
    inline = _execution(corpus)
    registry = _registry(corpus)
    execution = _ShardedExecutionFixture(
        inline,
        document_universe_sha256=registry.document_universe_sha256,
    )
    retriever = _retriever(corpus, registry)

    output = run_online_action_matrix(
        execution=execution,
        run_receipt=_run_receipt(),
        retriever=retriever,
        provenance_registry=registry,
        trial_runtimes=_runtimes(inline),
        permutation_seed=20260714,
        expected_policy_version="policy-registered",
        query_partition_audit_sha256=_PARTITION_SHA256,
        pseudonym_key=_PSEUDONYM_KEY,
        pseudonym_key_id="online-test-pseudonym-key",
        k=3,
        occurred_at_factory=_fixed_time,
    )

    assert output.admitted_panel.panel.document_count == len(corpus.documents)
    assert output.admitted_panel.panel.execution_artifact_sha256 == execution.artifact_sha256


def test_portable_balanced_action_orders_have_a_fixed_cross_runtime_vector() -> None:
    trial_families = (
        ("a" * 64, "1" * 64),
        ("b" * 64, "1" * 64),
        ("c" * 64, "1" * 64),
        ("d" * 64, "2" * 64),
        ("e" * 64, "2" * 64),
    )
    schedule = portable_balanced_action_orders(
        permutation_seed=20260714,
        execution_artifact_sha256="f" * 64,
        trial_families=trial_families,
    )
    assert dict(schedule) == {
        "a" * 64: ("hnsw-high", "exact-authorized", "hnsw-low", "abstain"),
        "b" * 64: ("hnsw-low", "abstain", "hnsw-high", "exact-authorized"),
        "c" * 64: ("exact-authorized", "hnsw-low", "abstain", "hnsw-high"),
        "d" * 64: ("abstain", "hnsw-high", "exact-authorized", "hnsw-low"),
        "e" * 64: ("hnsw-high", "exact-authorized", "hnsw-low", "abstain"),
    }
    assert set(schedule) == {trial for trial, _ in trial_families}
    assert all(set(order) == set(REGISTERED_ACTION_SET) for order in schedule.values())
    for family in {family for _, family in trial_families}:
        family_trials = [trial for trial, observed in trial_families if observed == family]
        for action in REGISTERED_ACTION_SET:
            counts = [
                sum(schedule[trial][position] == action for trial in family_trials)
                for position in range(4)
            ]
            assert max(counts) - min(counts) <= 1


def test_runner_emits_complete_matrix_order_receipt_and_audit_chain() -> None:
    output, execution, retriever = _run()
    panel = output.admitted_panel.panel

    assert len(panel.rows) == len(execution.trials) * len(REGISTERED_ACTION_SET)
    assert panel.action_set == REGISTERED_ACTION_SET
    assert output.execution_order_receipt.schema_version == (EXECUTION_ORDER_RECEIPT_SCHEMA)
    assert output.cache_preparation_receipt.schema_version == (CACHE_PREPARATION_RECEIPT_SCHEMA)
    assert output.execution_order_receipt.cache_preparation_receipt_sha256 == (
        output.cache_preparation_receipt.receipt_sha256
    )
    assert all(
        record.index_refresh is not None and not record.index_refresh.rebuilt
        for record in output.audit_records
    )
    assert {tuple(row.actions) for row in output.execution_order_receipt.rows}
    assert all(
        set(row.actions) == set(REGISTERED_ACTION_SET)
        for row in output.execution_order_receipt.rows
    )
    assert all(
        next(
            panel_row
            for panel_row in panel.rows
            if panel_row.trial_key == trial.trial_key and panel_row.action == "exact-authorized"
        ).execution_state
        == "completed"
        for trial in execution.trials
    )
    assert all(
        sum(row.controller_selected for row in panel.rows if row.trial_key == trial.trial_key) == 1
        for trial in execution.trials
    )

    verification = verify_audit_chain(
        output.audit_records,
        expected_head_sha256=output.audit_records[-1].record_sha256,
        expected_length=len(output.audit_records),
    )
    assert verification.valid
    assert output.anchoring_digests["action_panel_artifact_sha256"] == (panel.artifact_sha256)
    assert output.anchoring_digests["execution_order_receipt_sha256"] == (
        output.execution_order_receipt.receipt_sha256
    )
    assert isinstance(retriever.controller, RuleController)
    assert isinstance(retriever.policy, AuthorizationPolicy)


def test_runner_routes_active_query_to_hnsw_and_current_query_to_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus(trials=1)
    execution = _execution(corpus)
    registry = _registry(corpus)
    retriever = _retriever(corpus, registry)
    active_query = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    current_query = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    context = FrozenFeatureContext(
        version_lag=2.0,
        backend="hnswlib-0.8.0",
        drift_family="revision-pair",
        policy_complexity=4.0,
    )
    runtime = OnlineTrialRuntime(
        active_query_vector=active_query,
        current_truth_query_vector=current_query,
        feature_context=context,
    )
    observed_probe: list[np.ndarray] = []
    observed_high: list[np.ndarray] = []
    observed_exact: list[np.ndarray] = []
    original_probe = controller_module.authorized_hnsw_probe
    original_high = controller_module.authorized_hnsw_search
    original_exact = controller_module.authorized_exact_search

    def capture_probe(index, query, *args, **kwargs):
        observed_probe.append(np.asarray(query).copy())
        return original_probe(index, query, *args, **kwargs)

    def capture_high(index, query, *args, **kwargs):
        observed_high.append(np.asarray(query).copy())
        return original_high(index, query, *args, **kwargs)

    def capture_exact(index, query, *args, **kwargs):
        observed_exact.append(np.asarray(query).copy())
        return original_exact(index, query, *args, **kwargs)

    monkeypatch.setattr(controller_module, "authorized_hnsw_probe", capture_probe)
    monkeypatch.setattr(controller_module, "authorized_hnsw_search", capture_high)
    monkeypatch.setattr(controller_module, "authorized_exact_search", capture_exact)

    output = run_online_action_matrix(
        execution=execution,
        run_receipt=_run_receipt(),
        retriever=retriever,
        provenance_registry=registry,
        trial_runtimes={execution.trials[0].trial_key: runtime},
        permutation_seed=20260714,
        expected_policy_version="policy-registered",
        query_partition_audit_sha256=_PARTITION_SHA256,
        pseudonym_key=_PSEUDONYM_KEY,
        pseudonym_key_id="online-test-pseudonym-key",
        k=3,
        occurred_at_factory=_fixed_time,
    )

    assert observed_probe
    assert observed_high
    assert len(observed_exact) == 1
    assert all(np.array_equal(query, active_query) for query in observed_probe)
    assert all(np.array_equal(query, active_query) for query in observed_high)
    assert np.array_equal(observed_exact[0], current_query)
    order = output.execution_order_receipt.rows[0]
    assert order.active_query_vector_sha256 == runtime.active_query_vector_sha256
    assert order.current_truth_query_vector_sha256 == runtime.current_truth_query_vector_sha256
    assert order.active_query_vector_sha256 != order.current_truth_query_vector_sha256


def test_trial_runtime_owns_two_immutable_query_epochs() -> None:
    active = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    current = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    context = FrozenFeatureContext(
        version_lag=2.0,
        backend="hnswlib-0.8.0",
        drift_family="revision-pair",
        policy_complexity=4.0,
    )
    runtime = OnlineTrialRuntime(
        active_query_vector=active,
        current_truth_query_vector=current,
        feature_context=context,
    )
    active[:] = -1.0
    current[:] = -1.0

    assert runtime.active_query_vector.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert runtime.current_truth_query_vector.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert not runtime.active_query_vector.flags.writeable
    assert not runtime.current_truth_query_vector.flags.writeable
    assert not np.shares_memory(
        runtime.active_query_vector,
        runtime.current_truth_query_vector,
    )
    assert runtime.active_query_vector_sha256 != runtime.current_truth_query_vector_sha256


@pytest.mark.parametrize("epoch", ["active", "current"])
def test_runner_rejects_query_epoch_mutability_drift(
    epoch: str,
) -> None:
    corpus = _corpus(trials=1)
    execution = _execution(corpus)
    registry = _registry(corpus)
    retriever = _retriever(corpus, registry)
    runtimes = _runtimes(execution)
    runtime = runtimes[execution.trials[0].trial_key]
    vector = (
        runtime.active_query_vector if epoch == "active" else runtime.current_truth_query_vector
    )
    vector.setflags(write=True)
    vector[0] += np.float32(1.0)
    vector.setflags(write=False)

    with pytest.raises(OnlineRunnerError, match="mutability drift"):
        run_online_action_matrix(
            execution=execution,
            run_receipt=_run_receipt(),
            retriever=retriever,
            provenance_registry=registry,
            trial_runtimes=runtimes,
            permutation_seed=20260714,
            expected_policy_version="policy-registered",
            query_partition_audit_sha256=_PARTITION_SHA256,
            pseudonym_key=_PSEUDONYM_KEY,
            pseudonym_key_id="online-test-pseudonym-key",
            k=3,
        )


@pytest.mark.parametrize(
    ("active", "current", "match"),
    [
        (np.ones(4), np.ones(3), "same width"),
        (np.asarray([np.nan, 0.0]), np.ones(2), "active_query_vector"),
        (np.ones(2), np.asarray([np.inf, 0.0]), "current_truth_query_vector"),
    ],
)
def test_trial_runtime_rejects_invalid_query_epochs(
    active: np.ndarray,
    current: np.ndarray,
    match: str,
) -> None:
    context = FrozenFeatureContext(
        version_lag=2.0,
        backend="hnswlib-0.8.0",
        drift_family="revision-pair",
        policy_complexity=4.0,
    )

    with pytest.raises(OnlineRunnerError, match=match):
        OnlineTrialRuntime(
            active_query_vector=active,
            current_truth_query_vector=current,
            feature_context=context,
        )


def test_hnsw_low_row_uses_exact_registered_feature_order() -> None:
    output, execution, _ = _run(trials=1)
    row = next(row for row in output.admitted_panel.panel.rows if row.action == "hnsw-low")
    assert row.feature_values is not None
    assert len(row.feature_values) == len(REGISTERED_FEATURE_SCHEMA.input_features)
    values = dict(zip(REGISTERED_FEATURE_SCHEMA.input_features, row.feature_values, strict=True))
    assert tuple(values) == REGISTERED_FEATURE_SCHEMA.input_features
    assert values["corpus_size"] == len(execution.documents)
    assert values["authorized_universe_size"] == len(execution.documents)
    assert values["embedding_dimension"] == 4.0
    assert values["corpus_stratum"] == "scifact"
    assert values["backend"] == "hnswlib-0.8.0"
    assert values["policy_churn"] == 0.0
    runtime = _runtimes(execution)[execution.trials[0].trial_key]
    expected_drift = 1.0 - float(
        np.dot(runtime.active_query_vector, runtime.current_truth_query_vector)
        / (
            np.linalg.norm(runtime.active_query_vector)
            * np.linalg.norm(runtime.current_truth_query_vector)
        )
    )
    assert values["drift_severity"] == pytest.approx(expected_drift)
    assert np.isnan(values["probe_work"])
    assert all(
        candidate.feature_values is None
        for candidate in output.admitted_panel.panel.rows
        if candidate.action != "hnsw-low"
    )


def test_order_receipt_is_closed_canonical_and_exclusively_writable(
    tmp_path: Path,
) -> None:
    output, _, _ = _run(trials=1)
    receipt = output.execution_order_receipt
    assert loads_execution_order_receipt(receipt.canonical_bytes()) == receipt

    changed = receipt.to_dict()
    changed["unexpected"] = True
    with pytest.raises(OnlineRunnerError, match="fields differ"):
        loads_execution_order_receipt(json.dumps(changed, sort_keys=True, separators=(",", ":")))
    with pytest.raises(OnlineRunnerError, match="not canonical"):
        loads_execution_order_receipt(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))

    parent = tmp_path / "receipts"
    parent.mkdir(mode=0o700)
    target = parent / "execution-order.json"
    write_execution_order_receipt(receipt, target)
    assert load_execution_order_receipt(target) == receipt
    linked = parent / "linked-execution-order.json"
    linked.symlink_to(target.name)
    with pytest.raises(Exception, match="symlink"):
        load_execution_order_receipt(linked)
    with pytest.raises(Exception, match="already exists"):
        write_execution_order_receipt(receipt, target)


def test_cache_preparation_receipt_is_closed_canonical_and_exclusively_writable(
    tmp_path: Path,
) -> None:
    output, _, _ = _run(trials=2)
    receipt = output.cache_preparation_receipt
    assert loads_cache_preparation_receipt(receipt.canonical_bytes()) == receipt

    changed = receipt.to_dict()
    changed["unexpected"] = True
    with pytest.raises(OnlineRunnerError, match="fields differ"):
        loads_cache_preparation_receipt(json.dumps(changed, sort_keys=True, separators=(",", ":")))
    with pytest.raises(OnlineRunnerError, match="not canonical"):
        loads_cache_preparation_receipt(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))

    parent = tmp_path / "receipts"
    parent.mkdir(mode=0o700)
    target = parent / "cache-preparation.json"
    write_cache_preparation_receipt(receipt, target)
    assert load_cache_preparation_receipt(target) == receipt
    linked = parent / "linked-cache-preparation.json"
    linked.symlink_to(target.name)
    with pytest.raises(Exception, match="symlink"):
        load_cache_preparation_receipt(linked)
    with pytest.raises(Exception, match="already exists"):
        write_cache_preparation_receipt(receipt, target)


def test_nonexact_timeout_is_retained_as_registered_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = controller_module.authorized_hnsw_search

    def timeout_high(*args, **kwargs):
        if kwargs.get("strategy") == "hnsw-high":
            raise TimeoutError("bounded backend timeout")
        return original(*args, **kwargs)

    monkeypatch.setattr(controller_module, "authorized_hnsw_search", timeout_high)
    output, _, _ = _run(trials=1)
    failed = next(row for row in output.admitted_panel.panel.rows if row.action == "hnsw-high")
    assert failed.execution_state == "failed"
    assert failed.failure_state == "backend-timeout"
    assert failed.audit_record_sha256 is None
    assert failed.returned_document_ids == ()
    assert {item.failure_code for item in output.failed_executions} == {"backend-timeout"}
    exact = next(
        row for row in output.admitted_panel.panel.rows if row.action == "exact-authorized"
    )
    assert exact.execution_state == "completed"


def test_exact_timeout_aborts_without_returning_a_partial_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout_exact(*_args, **_kwargs):
        raise TimeoutError("exact backend timeout")

    monkeypatch.setattr(controller_module, "authorized_exact_search", timeout_exact)
    corpus = _corpus(trials=1)
    execution = _execution(corpus)
    registry = _registry(corpus)
    retriever = _retriever(corpus, registry)
    original_controller = retriever.controller
    original_policy = retriever.policy

    with pytest.raises(OnlineRunnerError, match="exact-authorized must complete"):
        run_online_action_matrix(
            execution=execution,
            run_receipt=_run_receipt(),
            retriever=retriever,
            provenance_registry=registry,
            trial_runtimes=_runtimes(execution),
            permutation_seed=20260714,
            expected_policy_version="policy-registered",
            query_partition_audit_sha256=_PARTITION_SHA256,
            pseudonym_key=_PSEUDONYM_KEY,
            pseudonym_key_id="online-test-pseudonym-key",
            k=3,
            occurred_at_factory=_fixed_time,
        )
    assert retriever.controller is original_controller
    assert retriever.policy is original_policy


class _DriftingPolicy:
    def __init__(self, registry: VerifiedProvenanceRegistry) -> None:
        self.calls = 0
        self.full = AuthorizationPolicy(
            roles=("analyst",),
            visibility=np.ones((1, registry.document_count), dtype=bool),
            version="policy-registered",
            document_universe_sha256=registry.document_universe_sha256,
        )
        reduced = np.ones((1, registry.document_count), dtype=bool)
        reduced[0, -1] = False
        self.reduced = AuthorizationPolicy(
            roles=("analyst",),
            visibility=reduced,
            version="policy-registered",
            document_universe_sha256=registry.document_universe_sha256,
        )

    @property
    def n_documents(self) -> int:
        return self.full.n_documents

    @property
    def document_universe_sha256(self) -> str:
        return self.full.document_universe_sha256

    def decide(self, *args, **kwargs):
        self.calls += 1
        policy = self.full if self.calls == 1 else self.reduced
        return policy.decide(*args, **kwargs)


class _EnvironmentSwapPolicy:
    """Return admitted masks at prewarm, then swap their environment binding."""

    def __init__(self, registry: VerifiedProvenanceRegistry) -> None:
        full = np.ones((1, registry.document_count), dtype=bool)
        reduced = full.copy()
        reduced[0, -1] = False
        self.policies = (
            AuthorizationPolicy(
                roles=("analyst",),
                visibility=full,
                version="policy-registered",
                document_universe_sha256=registry.document_universe_sha256,
            ),
            AuthorizationPolicy(
                roles=("analyst",),
                visibility=reduced,
                version="policy-registered",
                document_universe_sha256=registry.document_universe_sha256,
            ),
        )
        self.calls: dict[int, int] = {}

    @property
    def n_documents(self) -> int:
        return self.policies[0].n_documents

    @property
    def document_universe_sha256(self) -> str:
        return self.policies[0].document_universe_sha256

    def decide(self, *args, **kwargs):
        slot = int(kwargs["environment"]["trial_slot"])
        count = self.calls.get(slot, 0)
        self.calls[slot] = count + 1
        selected = slot if count == 0 else 1 - slot
        return self.policies[selected].decide(*args, **kwargs)


def test_policy_mask_drift_aborts_the_matrix() -> None:
    corpus = _corpus(trials=1)
    execution = _execution(corpus)
    registry = _registry(corpus)
    retriever = _retriever(corpus, registry, policy=_DriftingPolicy(registry))

    with pytest.raises(OnlineRunnerError, match="universe|abstained|changed"):
        run_online_action_matrix(
            execution=execution,
            run_receipt=_run_receipt(),
            retriever=retriever,
            provenance_registry=registry,
            trial_runtimes=_runtimes(execution),
            permutation_seed=20260714,
            expected_policy_version="policy-registered",
            query_partition_audit_sha256=_PARTITION_SHA256,
            pseudonym_key=_PSEUDONYM_KEY,
            pseudonym_key_id="online-test-pseudonym-key",
            k=3,
        )


def test_prepared_mask_cannot_be_rebound_to_another_environment() -> None:
    corpus = _corpus(trials=2)
    execution = _execution(corpus)
    registry = _registry(corpus)
    retriever = _retriever(
        corpus,
        registry,
        policy=_EnvironmentSwapPolicy(registry),
    )

    with pytest.raises(OnlineRunnerError, match="changed"):
        run_online_action_matrix(
            execution=execution,
            run_receipt=_run_receipt(),
            retriever=retriever,
            provenance_registry=registry,
            trial_runtimes=_runtimes(execution),
            permutation_seed=20260714,
            expected_policy_version="policy-registered",
            query_partition_audit_sha256=_PARTITION_SHA256,
            pseudonym_key=_PSEUDONYM_KEY,
            pseudonym_key_id="online-test-pseudonym-key",
            k=3,
        )
    assert len(retriever._authorized_index_cache) == 2


@pytest.mark.parametrize("mutation", ("missing", "extra", "wrong-dimension"))
def test_trial_runtime_binding_rejects_inexact_or_wrong_queries(mutation: str) -> None:
    corpus = _corpus(trials=1)
    execution = _execution(corpus)
    registry = _registry(corpus)
    retriever = _retriever(corpus, registry)
    runtimes = _runtimes(execution)
    if mutation == "missing":
        runtimes.clear()
    elif mutation == "extra":
        runtimes["f" * 64] = next(iter(runtimes.values()))
    else:
        trial_key = execution.trials[0].trial_key
        runtimes[trial_key] = OnlineTrialRuntime(
            active_query_vector=np.ones(3),
            current_truth_query_vector=np.ones(3),
            feature_context=next(iter(runtimes.values())).feature_context,
        )

    with pytest.raises(OnlineRunnerError, match="exact execution trial set|width"):
        run_online_action_matrix(
            execution=execution,
            run_receipt=_run_receipt(),
            retriever=retriever,
            provenance_registry=registry,
            trial_runtimes=runtimes,
            permutation_seed=20260714,
            expected_policy_version="policy-registered",
            query_partition_audit_sha256=_PARTITION_SHA256,
            pseudonym_key=_PSEUDONYM_KEY,
            pseudonym_key_id="online-test-pseudonym-key",
            k=3,
        )


def test_runner_source_has_no_custody_data_import_or_path() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "fractal_ann_diagnostics" / "online_runner.py"
    ).read_text(encoding="utf-8")
    assert "label_separation" not in source
    assert "SealedLabel" not in source
    assert "sealed-label" not in source
    assert "custody/" not in source
