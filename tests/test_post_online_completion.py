from __future__ import annotations

import hashlib
import json
import os
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib import error as urllib_error

import pytest

import fractal_ann_diagnostics.post_online_completion as completion_module
from fractal_ann_diagnostics.execution_claim import (
    ExecutionBeaconContract,
    VerifiedPhaseClaimCapability,
)
from fractal_ann_diagnostics.label_separation import (
    ActionPanelBinding,
    PredictionCompletionReceipt,
)
from fractal_ann_diagnostics.post_online_completion import (
    POST_ONLINE_COMPLETION_AGGREGATE_FILENAME,
    DrandRoundPublicationGuard,
    PostOnlineCompletionAggregateReceipt,
    PostOnlineCompletionError,
    VerifiedPostOnlineCompletionAnchors,
    VerifiedPostOnlineCompletionAuthority,
    ZenodoAnonymousCompletionAnchorReader,
    ZenodoCompletionAnchorPublisher,
    ZenodoDeposition,
    ZenodoPublishedRecord,
    ZenodoRemoteFile,
    execute_post_online_completion,
    load_post_online_completion_aggregate_receipt,
    revalidate_post_online_completion_anchors,
    revalidate_post_online_completion_authority,
    verify_post_online_completion_directory,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA
from fractal_ann_diagnostics.suite_attempt import (
    OnlineSuiteClosure,
    PhaseClaimBindings,
    VerifiedProviderPredecessor,
)

CORPORA = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))
TOKEN = b"test-token-value-1234567890"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class FakePredecessor:
    namespace: Path
    fail_on_revalidation: int | None = None

    def __post_init__(self) -> None:
        self.revalidations = 0
        self.ledger_commit = "a" * 40
        self.evidences = tuple(
            SimpleNamespace(
                descriptor_sha256=digest(f"descriptor-{index}"),
                bundle_sha256=digest(f"bundle-{index}"),
            )
            for index in range(3)
        )

    def assert_current(self) -> None:
        self.revalidations += 1
        if self.revalidations == self.fail_on_revalidation:
            raise PostOnlineCompletionError("provider state became stale")


class FakePanel:
    def __init__(self, corpus: str, binding: ActionPanelBinding) -> None:
        self.corpus = corpus
        self._binding = binding

    def completion_binding(self) -> ActionPanelBinding:
        return self._binding


class FakeGuard:
    def __init__(self, *, fail_on: int | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on

    def assert_not_public(self, beacon: object) -> None:
        del beacon
        self.calls += 1
        if self.calls == self.fail_on:
            raise PostOnlineCompletionError(
                "registered label-release drand round is already public"
            )


class FakePublisher:
    def __init__(
        self,
        *,
        altered_readback: bool = False,
        extra_remote_file: bool = False,
        duplicate_remote_file: bool = False,
        publish_raises: bool = False,
        publish_becomes_public: bool = True,
        created_at: datetime | None = None,
        public_updated_at: datetime | None = None,
        mutate_public_snapshot: bool = False,
    ) -> None:
        self.altered_readback = altered_readback
        self.extra_remote_file = extra_remote_file
        self.duplicate_remote_file = duplicate_remote_file
        self.publish_raises = publish_raises
        self.publish_becomes_public = publish_becomes_public
        self.created_at = created_at or datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.public_updated_at = public_updated_at or (self.created_at + timedelta(minutes=1))
        self.mutate_public_snapshot = mutate_public_snapshot
        self.create_calls = 0
        self.publish_calls = 0
        self.public_record_calls = 0
        self.metadata_calls = 0
        self.uploads: dict[str, bytes] = {}
        self.public = False

    def create_deposition(self) -> ZenodoDeposition:
        self.create_calls += 1
        return ZenodoDeposition(
            record_id=123456,
            created_at_utc=self.created_at.isoformat(),
            self_uri="https://zenodo.org/api/deposit/depositions/123456",
            bucket_uri="https://zenodo.org/api/files/01234567-89ab-cdef-0123-456789abcdef",
            publish_uri=("https://zenodo.org/api/deposit/depositions/123456/actions/publish"),
        )

    def set_metadata(
        self,
        deposition: ZenodoDeposition,
        metadata: object,
    ) -> None:
        assert deposition.record_id == 123456
        assert isinstance(metadata, dict)
        self.metadata_calls += 1

    def upload(
        self,
        deposition: ZenodoDeposition,
        filename: str,
        payload: bytes,
    ) -> None:
        assert deposition.record_id == 123456
        if filename in self.uploads:
            raise AssertionError("operator uploaded a duplicate filename")
        self.uploads[filename] = payload

    def _inventory(self) -> tuple[ZenodoRemoteFile, ...]:
        rows = [
            ZenodoRemoteFile(
                filename=name,
                byte_count=len(payload),
                checksum=("sha256:" + hashlib.sha256(payload).hexdigest()),
            )
            for name, payload in self.uploads.items()
        ]
        if self.extra_remote_file:
            rows.append(ZenodoRemoteFile("unregistered.json", 1))
        if self.duplicate_remote_file and rows:
            rows.append(rows[0])
        return tuple(rows)

    def draft_files(
        self,
        deposition: ZenodoDeposition,
    ) -> tuple[ZenodoRemoteFile, ...]:
        assert deposition.record_id == 123456
        return self._inventory()

    def publish_once(self, deposition: ZenodoDeposition) -> None:
        assert deposition.record_id == 123456
        self.publish_calls += 1
        self.public = self.publish_becomes_public
        if self.publish_raises:
            raise PostOnlineCompletionError("connection ended after publication request")

    def public_record(
        self,
        deposition: ZenodoDeposition,
    ) -> ZenodoPublishedRecord | None:
        assert deposition.record_id == 123456
        self.public_record_calls += 1
        if not self.public:
            return None
        updated_at = self.public_updated_at
        if self.mutate_public_snapshot and self.public_record_calls > 1:
            updated_at += timedelta(seconds=1)
        return ZenodoPublishedRecord(
            record_id=deposition.record_id,
            created_at_utc=deposition.created_at_utc,
            updated_at_utc=updated_at.isoformat(),
            files=self._inventory(),
        )

    def anonymous_read(self, uri: str, *, max_bytes: int) -> bytes:
        assert "access_token" not in uri
        assert max_bytes >= 1
        filename = uri.split("/files/", 1)[1].removesuffix("/content")
        encoded = self.uploads[filename]
        return encoded + b"x" if self.altered_readback else encoded


class FakeAnonymousReader:
    def __init__(
        self,
        aggregate: PostOnlineCompletionAggregateReceipt,
        uploads: dict[str, bytes],
        *,
        updated_at_utc: str | None = None,
        altered_readback: bool = False,
        extra_file: bool = False,
        omit_checksum: bool = False,
        mutate_public_snapshot: bool = False,
    ) -> None:
        self.aggregate = aggregate
        self.uploads = dict(uploads)
        self.updated_at_utc = (
            aggregate.zenodo_public_record_updated_at_utc
            if updated_at_utc is None
            else updated_at_utc
        )
        self.altered_readback = altered_readback
        self.extra_file = extra_file
        self.omit_checksum = omit_checksum
        self.mutate_public_snapshot = mutate_public_snapshot
        self.public_calls = 0
        self.byte_uris: list[str] = []

    def public_record(
        self,
        *,
        record_id: int,
        expected_created_at_utc: str,
    ) -> ZenodoPublishedRecord:
        self.public_calls += 1
        assert record_id == self.aggregate.zenodo_record_id
        assert expected_created_at_utc == self.aggregate.zenodo_deposition_created_at_utc
        rows = [
            ZenodoRemoteFile(
                filename=filename,
                byte_count=len(encoded),
                checksum=(
                    None if self.omit_checksum else "sha256:" + hashlib.sha256(encoded).hexdigest()
                ),
            )
            for filename, encoded in self.uploads.items()
        ]
        if self.extra_file:
            rows.append(
                ZenodoRemoteFile(
                    "unexpected.json",
                    1,
                    checksum="sha256:" + hashlib.sha256(b"x").hexdigest(),
                )
            )
        updated_at_utc = self.updated_at_utc
        if self.mutate_public_snapshot and self.public_calls > 1:
            updated_at_utc = (
                datetime.fromisoformat(updated_at_utc) + timedelta(seconds=1)
            ).isoformat()
        return ZenodoPublishedRecord(
            record_id=record_id,
            created_at_utc=expected_created_at_utc,
            updated_at_utc=updated_at_utc,
            files=tuple(rows),
        )

    def anonymous_read(self, uri: str, *, max_bytes: int) -> bytes:
        assert "access_token" not in uri
        assert max_bytes >= 1
        self.byte_uris.append(uri)
        filename = uri.split("/files/", 1)[1].removesuffix("/content")
        encoded = self.uploads[filename]
        return encoded + b"x" if self.altered_readback else encoded


class FakeRecoveryReader:
    def __init__(self, publisher: FakePublisher) -> None:
        self.publisher = publisher
        self.public_calls = 0
        self.byte_uris: list[str] = []

    def public_record(
        self,
        *,
        record_id: int,
        expected_created_at_utc: str,
    ) -> ZenodoPublishedRecord | None:
        self.public_calls += 1
        assert record_id == 123456
        assert expected_created_at_utc == self.publisher.created_at.isoformat()
        if not self.publisher.public:
            return None
        return ZenodoPublishedRecord(
            record_id=record_id,
            created_at_utc=expected_created_at_utc,
            updated_at_utc=self.publisher.public_updated_at.isoformat(),
            files=self.publisher._inventory(),
        )

    def anonymous_read(self, uri: str, *, max_bytes: int) -> bytes:
        self.byte_uris.append(uri)
        return self.publisher.anonymous_read(uri, max_bytes=max_bytes)


def completion_receipt(corpus: str) -> PredictionCompletionReceipt:
    binding = ActionPanelBinding(
        manifest_sha256=digest("manifest"),
        run_receipt_sha256=digest("run"),
        execution_artifact_sha256=digest(f"execution-{corpus}"),
        corpus=corpus,
        stage="sealed",
        action_panel_artifact_sha256=digest(f"panel-{corpus}"),
    )
    return PredictionCompletionReceipt(
        manifest_sha256=binding.manifest_sha256,
        run_receipt_sha256=binding.run_receipt_sha256,
        execution_artifact_sha256=binding.execution_artifact_sha256,
        prediction_artifact_sha256=digest(f"prediction-{corpus}"),
        online_execution_result_receipt_sha256=digest(f"result-{corpus}"),
        action_panel_binding=binding,
        prediction_count=7,
        corpus=corpus,
        stage="sealed",
        external_anchor_identity="zenodo-record:123456",
        external_anchor_uri=(
            "https://zenodo.org/api/records/123456/files/"
            f"{corpus}-prediction-completion-anchor.json/content"
        ),
        anchored_at_utc="2026-01-01T00:00:00+00:00",
    )


def fake_suite(tmp_path: Path, predecessor: FakePredecessor) -> SimpleNamespace:
    publication = datetime(2026, 1, 2, tzinfo=timezone.utc)
    beacon = SimpleNamespace(
        label_release_round=99,
        label_release_publication_time=publication,
        contract_sha256=digest("label-release-beacon"),
    )
    sources = []
    for corpus in CORPORA:
        receipt = completion_receipt(corpus)
        sources.append(
            SimpleNamespace(
                corpus_id=corpus,
                predictions=SimpleNamespace(corpus=corpus),
                action_panel=FakePanel(corpus, receipt.action_panel_binding),
                online_result=SimpleNamespace(corpus=corpus),
                execution=SimpleNamespace(corpus=corpus),
            )
        )
    online_payload = SimpleNamespace(
        run_output_aggregate=SimpleNamespace(aggregate_sha256=digest("online-output-aggregate"))
    )
    online = SimpleNamespace(
        sequence=1,
        record_sha256=digest("online-state"),
        payload=online_payload,
    )
    claim = SimpleNamespace(
        suite_attempt_id=digest("suite"),
        run_receipt_sha256=digest("run"),
        record_sha256=digest("claim-state"),
    )
    return SimpleNamespace(
        predecessor=predecessor,
        online_record=online,
        claim_record=claim,
        manifest_digest=digest("manifest"),
        sealed_run=object(),
        beacon=beacon,
        completion_root=tmp_path / "completion",
        sources=tuple(sources),
    )


@pytest.fixture
def patched_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakePredecessor, SimpleNamespace]:
    predecessor = FakePredecessor(tmp_path)
    suite = fake_suite(tmp_path, predecessor)
    monkeypatch.setattr(
        completion_module,
        "_admit_post_online_suite",
        lambda authority: suite if authority is predecessor else None,
    )

    def create_receipt(
        predictions: object,
        *,
        external_anchor_uri: str,
        anchored_at_utc: str,
        **kwargs: object,
    ) -> PredictionCompletionReceipt:
        del kwargs
        corpus = predictions.corpus
        value = completion_receipt(corpus)
        assert external_anchor_uri == value.external_anchor_uri
        assert anchored_at_utc == value.anchored_at_utc
        return value

    monkeypatch.setattr(
        completion_module,
        "create_prediction_completion_receipt",
        create_receipt,
    )
    return predecessor, suite


_AUTHORITY_AGGREGATE_FIELDS = (
    "suite_attempt_id",
    "manifest_sha256",
    "run_receipt_sha256",
    "online_complete_state_sha256",
    "online_output_aggregate_sha256",
    "online_attestation_descriptor_sha256",
    "online_attestation_bundle_sha256",
    "label_release_claim_state_sha256",
    "label_release_claim_ledger_commit",
    "label_release_round",
    "label_release_beacon_contract_sha256",
    "label_release_publication_time_utc",
)


def authority_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    predecessor_fail_on: int | None = None,
) -> SimpleNamespace:
    namespace = tmp_path / "canonical-suite"
    namespace.mkdir()
    completion_root = namespace / "completion"
    provider_identity = object()
    release_beacon = ExecutionBeaconContract(
        drand_network="https://api.drand.sh",
        chain_hash=digest("authority-chain"),
        chain_scheme_id="bls-unchained-g1-rfc9380",
        chain_public_key="aa",
        chain_genesis_unix_seconds=1,
        chain_period_seconds=3,
        execution_round=10,
        label_release_round=20,
        minimum_label_release_safety_rounds=10,
        verification_identity=digest("authority-verification"),
    )
    phase_contract = SimpleNamespace(
        phase="label-release",
        label_release_beacon=release_beacon,
    )
    phase_freshness = {"calls": 0}

    def revalidate_phase() -> None:
        phase_freshness["calls"] += 1

    phase_claim = object.__new__(VerifiedPhaseClaimCapability)
    object.__setattr__(phase_claim, "contract", phase_contract)
    object.__setattr__(phase_claim, "provider_identity", provider_identity)
    object.__setattr__(phase_claim, "phase_claim_state_sha256", digest("authority-claim"))
    object.__setattr__(phase_claim, "phase_claim_ledger_commit", "a" * 40)
    object.__setattr__(phase_claim, "_fresh_revalidator", revalidate_phase)
    object.__setattr__(
        phase_claim,
        "_minted_monotonic_ns",
        completion_module.time.monotonic_ns(),
    )

    online_payload = object.__new__(OnlineSuiteClosure)
    object.__setattr__(
        online_payload,
        "run_output_aggregate",
        SimpleNamespace(aggregate_sha256=digest("authority-output-aggregate")),
    )
    online = SimpleNamespace(
        sequence=0,
        record_sha256=digest("authority-online"),
        payload=online_payload,
    )
    claim_payload = object.__new__(PhaseClaimBindings)
    object.__setattr__(claim_payload, "phase_claim", phase_contract)
    object.__setattr__(claim_payload, "provider_identity", provider_identity)
    claim = SimpleNamespace(
        suite_attempt_id=digest("authority-suite"),
        manifest_sha256=digest("authority-manifest"),
        run_receipt_sha256=digest("authority-run"),
        record_sha256=phase_claim.phase_claim_state_sha256,
        namespace_uri=namespace.as_uri(),
        payload=claim_payload,
    )
    predecessor_freshness = {"calls": 0}

    def revalidate_predecessor() -> None:
        predecessor_freshness["calls"] += 1
        if predecessor_freshness["calls"] == predecessor_fail_on:
            raise RuntimeError("provider state changed")

    online_evidence = SimpleNamespace(
        descriptor_sha256=digest("authority-online-descriptor"),
        bundle_sha256=digest("authority-online-bundle"),
        transition_id="b" * 40,
    )
    claim_evidence = SimpleNamespace(transition_id=phase_claim.phase_claim_ledger_commit)
    predecessor = object.__new__(VerifiedProviderPredecessor)
    object.__setattr__(predecessor, "records", (online, claim))
    object.__setattr__(predecessor, "evidences", (online_evidence, claim_evidence))
    object.__setattr__(predecessor, "_fresh_revalidator", revalidate_predecessor)
    admitted = SimpleNamespace(
        claim_record=claim,
        online_record=online,
        completion_root=completion_root,
    )
    monkeypatch.setattr(
        completion_module,
        "_admit_post_online_suite",
        lambda candidate: admitted if candidate is predecessor else None,
    )

    aggregate = object.__new__(PostOnlineCompletionAggregateReceipt)
    aggregate_values = {
        "suite_attempt_id": claim.suite_attempt_id,
        "manifest_sha256": claim.manifest_sha256,
        "run_receipt_sha256": claim.run_receipt_sha256,
        "online_complete_state_sha256": online.record_sha256,
        "online_output_aggregate_sha256": (online_payload.run_output_aggregate.aggregate_sha256),
        "online_attestation_descriptor_sha256": online_evidence.descriptor_sha256,
        "online_attestation_bundle_sha256": online_evidence.bundle_sha256,
        "label_release_claim_state_sha256": claim.record_sha256,
        "label_release_claim_ledger_commit": predecessor.ledger_commit,
        "label_release_round": release_beacon.label_release_round,
        "label_release_beacon_contract_sha256": release_beacon.contract_sha256,
        "label_release_publication_time_utc": (
            release_beacon.label_release_publication_time.isoformat()
        ),
    }
    for name, value in aggregate_values.items():
        object.__setattr__(aggregate, name, value)
    verified = object.__new__(VerifiedPostOnlineCompletionAnchors)
    object.__setattr__(verified, "completion_root", completion_root)
    object.__setattr__(verified, "aggregate", aggregate)
    anonymous_calls: list[tuple[Path, object | None]] = []

    def revalidate_anchors(
        root: str | Path,
        *,
        reader: object | None = None,
    ) -> VerifiedPostOnlineCompletionAnchors:
        anonymous_calls.append((Path(root), reader))
        return verified

    monkeypatch.setattr(
        completion_module,
        "revalidate_post_online_completion_anchors",
        revalidate_anchors,
    )
    return SimpleNamespace(
        aggregate=aggregate,
        anonymous_calls=anonymous_calls,
        completion_root=completion_root,
        phase_claim=phase_claim,
        phase_freshness=phase_freshness,
        predecessor=predecessor,
        predecessor_freshness=predecessor_freshness,
        verified=verified,
    )


def test_happy_path_writes_exact_closed_receipt_set(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, suite = patched_suite
    publisher = FakePublisher()
    guard = FakeGuard()

    aggregate = execute_post_online_completion(
        predecessor,  # type: ignore[arg-type]
        publisher=publisher,
        round_guard=guard,
    )

    assert publisher.create_calls == 1
    assert publisher.publish_calls == 1
    assert publisher.metadata_calls == 1
    assert len(publisher.uploads) == 5
    assert len(tuple(suite.completion_root.iterdir())) == 16
    assert (suite.completion_root / POST_ONLINE_COMPLETION_AGGREGATE_FILENAME).read_bytes() == (
        aggregate.canonical_file_bytes()
    )
    assert (
        load_post_online_completion_aggregate_receipt(
            suite.completion_root / POST_ONLINE_COMPLETION_AGGREGATE_FILENAME
        )
        == aggregate
    )
    assert verify_post_online_completion_directory(suite.completion_root) == aggregate
    assert aggregate.online_output_aggregate_sha256 == digest("online-output-aggregate")
    assert aggregate.online_attestation_descriptor_sha256 == digest("descriptor-1")
    assert aggregate.online_attestation_bundle_sha256 == digest("bundle-1")
    assert aggregate.label_release_beacon_contract_sha256 == digest("label-release-beacon")
    assert (
        aggregate.zenodo_deposition_created_at_utc
        < aggregate.zenodo_public_record_updated_at_utc
        < aggregate.label_release_publication_time_utc
    )
    assert guard.calls >= 5


def test_anonymous_revalidator_returns_unforgeable_typed_capability(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, suite = patched_suite
    publisher = FakePublisher()
    aggregate = execute_post_online_completion(
        predecessor,  # type: ignore[arg-type]
        publisher=publisher,
        round_guard=FakeGuard(),
    )
    reader = FakeAnonymousReader(aggregate, publisher.uploads)

    verified = revalidate_post_online_completion_anchors(
        suite.completion_root,
        reader=reader,
    )

    assert isinstance(verified, VerifiedPostOnlineCompletionAnchors)
    assert verified.aggregate == aggregate
    assert tuple(row.corpus for row in verified.records) == CORPORA
    assert verified.anchor_for(CORPORA[0]).record.corpus == CORPORA[0]
    assert reader.public_calls == 2
    assert len(reader.byte_uris) == 5
    with pytest.raises(PostOnlineCompletionError, match="only come from"):
        VerifiedPostOnlineCompletionAnchors(
            completion_root=suite.completion_root,
            aggregate=aggregate,
            anchors=verified.anchors,
            _capability=object(),
        )


def test_authority_revalidator_binds_public_anchors_to_live_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = authority_case(tmp_path, monkeypatch)
    reader = object()

    observed = revalidate_post_online_completion_authority(
        case.predecessor,
        case.phase_claim,
        reader=reader,
    )

    assert isinstance(observed, VerifiedPostOnlineCompletionAuthority)
    assert observed.completion is case.verified
    assert observed.provider_namespace == case.predecessor.namespace
    assert case.anonymous_calls == [(case.completion_root, reader)]
    assert case.predecessor_freshness["calls"] == 2
    assert case.phase_freshness["calls"] == 2

    with pytest.raises(PostOnlineCompletionError, match="only come from"):
        VerifiedPostOnlineCompletionAuthority(
            completion=case.verified,
            provider_namespace=case.predecessor.namespace,
            phase_claim_state_sha256=case.phase_claim.phase_claim_state_sha256,
            phase_claim_ledger_commit=case.phase_claim.phase_claim_ledger_commit,
            _capability=object(),
        )


@pytest.mark.parametrize("field", _AUTHORITY_AGGREGATE_FIELDS)
def test_authority_revalidator_rejects_each_aggregate_lineage_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    case = authority_case(tmp_path, monkeypatch)
    replacement: object
    if field == "label_release_round":
        replacement = case.aggregate.label_release_round + 1
    elif field == "label_release_claim_ledger_commit":
        replacement = "c" * 40
    elif field == "label_release_publication_time_utc":
        replacement = "2026-01-01T00:00:00+00:00"
    else:
        replacement = digest(f"changed-{field}")
    object.__setattr__(case.aggregate, field, replacement)

    with pytest.raises(
        PostOnlineCompletionError,
        match="aggregate differs from provider authority",
    ):
        revalidate_post_online_completion_authority(
            case.predecessor,
            case.phase_claim,
        )

    assert case.predecessor_freshness["calls"] == 1
    assert case.phase_freshness["calls"] == 1
    assert case.anonymous_calls == [(case.completion_root, None)]


def test_authority_revalidator_rejects_wrong_completion_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = authority_case(tmp_path, monkeypatch)
    object.__setattr__(
        case.verified,
        "completion_root",
        tmp_path / "different-suite" / "completion",
    )

    with pytest.raises(
        PostOnlineCompletionError,
        match="completion root differs from provider authority",
    ):
        revalidate_post_online_completion_authority(
            case.predecessor,
            case.phase_claim,
        )

    assert case.predecessor_freshness["calls"] == 1
    assert case.phase_freshness["calls"] == 1


def test_authority_revalidator_rejects_second_predecessor_freshness_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = authority_case(
        tmp_path,
        monkeypatch,
        predecessor_fail_on=2,
    )

    with pytest.raises(
        PostOnlineCompletionError,
        match="provider predecessor failed fresh revalidation",
    ):
        revalidate_post_online_completion_authority(
            case.predecessor,
            case.phase_claim,
        )

    assert case.anonymous_calls == [(case.completion_root, None)]
    assert case.predecessor_freshness["calls"] == 2
    assert case.phase_freshness["calls"] == 1


@pytest.mark.parametrize(
    ("reader_options", "message"),
    [
        (
            {
                "updated_at_utc": "2026-01-01T00:02:00+00:00",
            },
            "updated timestamp differs",
        ),
        ({"altered_readback": True}, "public anchor bytes differ"),
        ({"extra_file": True}, "exact five records"),
        ({"omit_checksum": True}, "lacks a checksum"),
    ],
)
def test_anonymous_revalidator_rejects_live_remote_drift(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
    reader_options: dict[str, object],
    message: str,
) -> None:
    predecessor, suite = patched_suite
    publisher = FakePublisher()
    aggregate = execute_post_online_completion(
        predecessor,  # type: ignore[arg-type]
        publisher=publisher,
        round_guard=FakeGuard(),
    )
    reader = FakeAnonymousReader(
        aggregate,
        publisher.uploads,
        **reader_options,
    )

    with pytest.raises(PostOnlineCompletionError, match=message):
        revalidate_post_online_completion_anchors(
            suite.completion_root,
            reader=reader,
        )


def test_anonymous_revalidator_rejects_inventory_mutation_during_readback(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, suite = patched_suite
    publisher = FakePublisher()
    aggregate = execute_post_online_completion(
        predecessor,  # type: ignore[arg-type]
        publisher=publisher,
        round_guard=FakeGuard(),
    )
    reader = FakeAnonymousReader(
        aggregate,
        publisher.uploads,
        mutate_public_snapshot=True,
    )

    with pytest.raises(PostOnlineCompletionError, match="changed during anchor readback"):
        revalidate_post_online_completion_anchors(
            suite.completion_root,
            reader=reader,
        )

    assert reader.public_calls == 2
    assert len(reader.byte_uris) == 5


def test_existing_completion_directory_is_terminal_before_remote_mutation(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, suite = patched_suite
    suite.completion_root.mkdir(mode=0o700)
    publisher = FakePublisher()

    with pytest.raises(PostOnlineCompletionError, match="not an exact recoverable"):
        execute_post_online_completion(
            predecessor,  # type: ignore[arg-type]
            publisher=publisher,
            round_guard=FakeGuard(),
        )

    assert publisher.create_calls == 0


def test_published_interruption_recovers_without_second_remote_mutation(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor, suite = patched_suite
    published = FakePublisher()
    real_writer = completion_module.write_prediction_completion_anchor_receipt
    receipt_writes = 0

    def interrupt_receipt_write(*args: object, **kwargs: object) -> None:
        nonlocal receipt_writes
        receipt_writes += 1
        if receipt_writes == 3:
            raise RuntimeError("simulated process interruption")
        real_writer(*args, **kwargs)

    monkeypatch.setattr(
        completion_module,
        "write_prediction_completion_anchor_receipt",
        interrupt_receipt_write,
    )
    with pytest.raises(PostOnlineCompletionError, match="cannot persist verified"):
        execute_post_online_completion(
            predecessor,  # type: ignore[arg-type]
            publisher=published,
            round_guard=FakeGuard(),
        )

    assert published.create_calls == 1
    assert published.publish_calls == 1
    assert len(tuple(suite.completion_root.iterdir())) == 12

    monkeypatch.setattr(
        completion_module,
        "write_prediction_completion_anchor_receipt",
        real_writer,
    )
    unused_publisher = FakePublisher()
    reader = FakeRecoveryReader(published)
    aggregate = execute_post_online_completion(
        predecessor,  # type: ignore[arg-type]
        publisher=unused_publisher,
        recovery_reader=reader,
        round_guard=FakeGuard(),
    )

    assert unused_publisher.create_calls == 0
    assert unused_publisher.publish_calls == 0
    assert published.publish_calls == 1
    assert reader.public_calls == 2
    assert len(reader.byte_uris) == 5
    assert verify_post_online_completion_directory(suite.completion_root) == aggregate


def test_completed_restart_returns_local_closure_without_remote_mutation(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, _suite = patched_suite
    published = FakePublisher()
    expected = execute_post_online_completion(
        predecessor,  # type: ignore[arg-type]
        publisher=published,
        round_guard=FakeGuard(),
    )
    unused_publisher = FakePublisher()
    public_round = FakeGuard(fail_on=1)

    observed = execute_post_online_completion(
        predecessor,  # type: ignore[arg-type]
        publisher=unused_publisher,
        round_guard=public_round,
    )

    assert observed == expected
    assert public_round.calls == 0
    assert unused_publisher.create_calls == 0
    assert unused_publisher.publish_calls == 0


@pytest.mark.parametrize(
    ("publisher", "message"),
    [
        (FakePublisher(altered_readback=True), "readback changed bytes"),
        (FakePublisher(extra_remote_file=True), "exact five records"),
        (FakePublisher(duplicate_remote_file=True), "exact five records"),
    ],
)
def test_remote_byte_or_inventory_change_is_terminal(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
    publisher: FakePublisher,
    message: str,
) -> None:
    predecessor, suite = patched_suite

    with pytest.raises(PostOnlineCompletionError, match=message):
        execute_post_online_completion(
            predecessor,  # type: ignore[arg-type]
            publisher=publisher,
            round_guard=FakeGuard(),
        )

    assert suite.completion_root.exists()
    assert not tuple(suite.completion_root.glob("*-anchor-receipt.json"))


def test_public_inventory_mutation_during_readback_is_terminal(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, suite = patched_suite
    publisher = FakePublisher(mutate_public_snapshot=True)

    with pytest.raises(PostOnlineCompletionError, match="changed during anonymous readback"):
        execute_post_online_completion(
            predecessor,  # type: ignore[arg-type]
            publisher=publisher,
            round_guard=FakeGuard(),
        )

    assert publisher.publish_calls == 1
    assert publisher.public_record_calls == 2
    assert not tuple(suite.completion_root.glob("*-anchor-receipt.json"))


def test_ambiguous_publish_is_reconciled_without_second_post(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, _suite = patched_suite
    publisher = FakePublisher(publish_raises=True, publish_becomes_public=True)

    execute_post_online_completion(
        predecessor,  # type: ignore[arg-type]
        publisher=publisher,
        round_guard=FakeGuard(),
    )

    assert publisher.publish_calls == 1


def test_unresolved_publish_ambiguity_is_terminal(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, suite = patched_suite
    publisher = FakePublisher(publish_raises=True, publish_becomes_public=False)

    with pytest.raises(PostOnlineCompletionError, match="outcome is ambiguous"):
        execute_post_online_completion(
            predecessor,  # type: ignore[arg-type]
            publisher=publisher,
            round_guard=FakeGuard(),
        )

    assert publisher.publish_calls == 1
    assert suite.completion_root.exists()


def test_public_round_and_server_time_are_fail_closed(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, suite = patched_suite
    publisher = FakePublisher()

    with pytest.raises(PostOnlineCompletionError, match="already public"):
        execute_post_online_completion(
            predecessor,  # type: ignore[arg-type]
            publisher=publisher,
            round_guard=FakeGuard(fail_on=1),
        )
    assert not suite.completion_root.exists()
    assert publisher.create_calls == 0

    late_predecessor = FakePredecessor(suite.completion_root.parent / "late")
    late_predecessor.namespace.mkdir(mode=0o700)
    late_suite = fake_suite(late_predecessor.namespace, late_predecessor)
    completion_module._admit_post_online_suite = lambda _authority: late_suite
    late_publisher = FakePublisher(created_at=late_suite.beacon.label_release_publication_time)
    with pytest.raises(PostOnlineCompletionError, match="at or after"):
        execute_post_online_completion(
            late_predecessor,  # type: ignore[arg-type]
            publisher=late_publisher,
            round_guard=FakeGuard(),
        )
    assert late_publisher.metadata_calls == 0


def test_public_record_updated_at_or_after_release_is_terminal(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, suite = patched_suite
    publisher = FakePublisher(public_updated_at=suite.beacon.label_release_publication_time)

    with pytest.raises(PostOnlineCompletionError, match="updated at or after"):
        execute_post_online_completion(
            predecessor,  # type: ignore[arg-type]
            publisher=publisher,
            round_guard=FakeGuard(),
        )

    assert publisher.publish_calls == 1
    assert suite.completion_root.exists()
    assert not tuple(suite.completion_root.glob("*-anchor-receipt.json"))


def test_stale_provider_state_stops_before_publication(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, suite = patched_suite
    predecessor.fail_on_revalidation = 4
    publisher = FakePublisher()

    with pytest.raises(PostOnlineCompletionError, match="stale"):
        execute_post_online_completion(
            predecessor,  # type: ignore[arg-type]
            publisher=publisher,
            round_guard=FakeGuard(),
        )

    assert publisher.publish_calls == 0
    assert suite.completion_root.exists()

    unused_publisher = FakePublisher()
    with pytest.raises(PostOnlineCompletionError, match="no exact anonymous public record"):
        execute_post_online_completion(
            predecessor,  # type: ignore[arg-type]
            publisher=unused_publisher,
            recovery_reader=FakeRecoveryReader(publisher),
            round_guard=FakeGuard(),
        )
    assert unused_publisher.create_calls == 0
    assert unused_publisher.publish_calls == 0


def test_round_becoming_public_mid_attempt_is_terminal(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
) -> None:
    predecessor, suite = patched_suite
    publisher = FakePublisher()

    with pytest.raises(PostOnlineCompletionError, match="already public"):
        execute_post_online_completion(
            predecessor,  # type: ignore[arg-type]
            publisher=publisher,
            round_guard=FakeGuard(fail_on=4),
        )

    assert publisher.publish_calls == 0
    assert suite.completion_root.exists()


def test_cross_corpus_completion_is_rejected(
    patched_suite: tuple[FakePredecessor, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor, _suite = patched_suite

    def crossed(predictions: object, **kwargs: object) -> PredictionCompletionReceipt:
        del predictions, kwargs
        return completion_receipt(CORPORA[-1])

    monkeypatch.setattr(
        completion_module,
        "create_prediction_completion_receipt",
        crossed,
    )
    with pytest.raises(PostOnlineCompletionError, match="crossed corpus"):
        execute_post_online_completion(
            predecessor,  # type: ignore[arg-type]
            publisher=FakePublisher(),
            round_guard=FakeGuard(),
        )


class FakeResponse:
    def __init__(
        self,
        uri: str,
        body: bytes,
        *,
        status: int = 200,
        response_uri: str | None = None,
    ) -> None:
        self._uri = uri if response_uri is None else response_uri
        self._body = body
        self._status = status
        self.headers: dict[str, str] = {
            "Content-Length": str(len(body)),
        }

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def getcode(self) -> int:
        return self._status

    def geturl(self) -> str:
        return self._uri

    def read(self, count: int) -> bytes:
        return self._body[:count]


class RecordingOpener:
    def __init__(self, responses: list[FakeResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests: list[object] = []
        self.failure: Exception | None = None

    def open(self, request: object, *, timeout: float) -> FakeResponse:
        assert timeout > 0
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return self.responses.pop(0)


def publisher_for_test(opener: RecordingOpener) -> ZenodoCompletionAnchorPublisher:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, TOKEN + b"\n")
    finally:
        os.close(write_fd)
    try:
        return ZenodoCompletionAnchorPublisher(token_fd=read_fd, opener=opener)
    finally:
        os.close(read_fd)


def create_deposition_body() -> bytes:
    return json.dumps(
        {
            "created": "2026-01-01T00:00:00Z",
            "id": 123456,
            "links": {
                "bucket": ("https://zenodo.org/api/files/01234567-89ab-cdef-0123-456789abcdef"),
                "publish": ("https://zenodo.org/api/deposit/depositions/123456/actions/publish"),
                "self": "https://zenodo.org/api/deposit/depositions/123456",
            },
            "state": "unsubmitted",
            "submitted": False,
        },
        sort_keys=True,
    ).encode()


def test_token_fd_auth_is_redacted_and_anonymous_read_has_no_auth() -> None:
    create_uri = "https://zenodo.org/api/deposit/depositions"
    content_uri = (
        "https://zenodo.org/api/records/123456/files/"
        "scifact-prediction-completion-anchor.json/content"
    )
    opener = RecordingOpener(
        [
            FakeResponse(create_uri, create_deposition_body(), status=201),
            FakeResponse(content_uri, b"record\n"),
        ]
    )
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, TOKEN + b"\n")
    finally:
        os.close(write_fd)
    publisher = ZenodoCompletionAnchorPublisher.from_token_fd(read_fd, opener=opener)
    os.close(read_fd)
    token_buffer = publisher._token

    deposition = publisher.create_deposition()
    assert deposition.created_at_utc == "2026-01-01T00:00:00+00:00"
    publisher.anonymous_read(content_uri, max_bytes=1024)

    first_headers = opener.requests[0].headers
    second_headers = opener.requests[1].headers
    assert first_headers["Authorization"] == f"Bearer {TOKEN.decode()}"
    assert "Authorization" not in second_headers
    assert TOKEN.decode() not in repr(publisher)
    assert TOKEN.decode() not in opener.requests[0].full_url
    publisher.close()
    assert set(token_buffer) == {0}


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("token-value-must-not-escape"),
        ssl.SSLError("token-value-must-not-escape"),
        urllib_error.URLError("token-value-must-not-escape"),
    ],
)
def test_tls_timeout_and_url_errors_are_sanitized(failure: Exception) -> None:
    opener = RecordingOpener()
    opener.failure = failure
    publisher = publisher_for_test(opener)

    with pytest.raises(PostOnlineCompletionError) as caught:
        publisher.create_deposition()

    assert str(caught.value) == "Zenodo HTTPS request failed"
    assert TOKEN.decode() not in str(caught.value)
    publisher.close()


def test_changed_response_url_is_refused() -> None:
    uri = "https://zenodo.org/api/deposit/depositions"
    opener = RecordingOpener(
        [
            FakeResponse(
                uri,
                create_deposition_body(),
                status=201,
                response_uri="https://example.test/redirected",
            )
        ]
    )
    publisher = publisher_for_test(opener)

    with pytest.raises(PostOnlineCompletionError, match="response URL changed"):
        publisher.create_deposition()

    publisher.close()


def test_public_inventory_requires_fixed_direct_content_uri() -> None:
    deposition = ZenodoDeposition(
        record_id=123456,
        created_at_utc="2026-01-01T00:00:00+00:00",
        self_uri="https://zenodo.org/api/deposit/depositions/123456",
        bucket_uri="https://zenodo.org/api/files/01234567-89ab-cdef-0123-456789abcdef",
        publish_uri=("https://zenodo.org/api/deposit/depositions/123456/actions/publish"),
    )
    public_uri = "https://zenodo.org/api/records/123456"
    body = json.dumps(
        {
            "created": "2026-01-01T00:00:00Z",
            "files": [
                {
                    "checksum": "md5:" + ("a" * 32),
                    "key": "scifact-prediction-completion-anchor.json",
                    "links": {
                        "self": ("https://zenodo.org/api/records/123456/files/other.json/content")
                    },
                    "size": 5,
                }
            ],
            "id": 123456,
            "updated": "2026-01-01T00:01:00Z",
        }
    ).encode()
    publisher = publisher_for_test(RecordingOpener([FakeResponse(public_uri, body)]))

    with pytest.raises(PostOnlineCompletionError, match="changes its content URI"):
        publisher.public_record(deposition)

    publisher.close()


def test_anonymous_reader_uses_exact_urls_without_authorization() -> None:
    filename = "scifact-prediction-completion-anchor.json"
    public_uri = "https://zenodo.org/api/records/123456"
    content_uri = f"https://zenodo.org/api/records/123456/files/{filename}/content"
    record_body = json.dumps(
        {
            "created": "2026-01-01T00:00:00Z",
            "files": [
                {
                    "checksum": "md5:"
                    + hashlib.md5(
                        b"record\n",
                        usedforsecurity=False,
                    ).hexdigest(),
                    "key": filename,
                    "links": {"self": content_uri},
                    "size": len(b"record\n"),
                }
            ],
            "id": 123456,
            "updated": "2026-01-01T00:01:00Z",
        }
    ).encode()
    opener = RecordingOpener(
        [
            FakeResponse(public_uri, record_body),
            FakeResponse(content_uri, b"record\n"),
        ]
    )
    reader = ZenodoAnonymousCompletionAnchorReader(opener=opener)

    record = reader.public_record(
        record_id=123456,
        expected_created_at_utc="2026-01-01T00:00:00+00:00",
    )
    assert record is not None
    assert record.updated_at_utc == "2026-01-01T00:01:00+00:00"
    assert reader.anonymous_read(content_uri, max_bytes=1024) == b"record\n"
    assert [request.full_url for request in opener.requests] == [
        public_uri,
        content_uri,
    ]
    assert all("Authorization" not in request.headers for request in opener.requests)


def test_anonymous_reader_refuses_redirect_status() -> None:
    public_uri = "https://zenodo.org/api/records/123456"
    opener = RecordingOpener()
    opener.failure = urllib_error.HTTPError(
        public_uri,
        302,
        "redirect",
        {},
        None,
    )
    reader = ZenodoAnonymousCompletionAnchorReader(opener=opener)

    with pytest.raises(PostOnlineCompletionError, match="redirect was refused"):
        reader.public_record(
            record_id=123456,
            expected_created_at_utc="2026-01-01T00:00:00+00:00",
        )


def test_https_opener_requires_hostname_and_certificate_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        check_hostname=False,
        verify_mode=ssl.CERT_NONE,
    )
    captured: list[object] = []
    monkeypatch.setattr(
        completion_module.ssl,
        "create_default_context",
        lambda *, purpose: context,
    )
    monkeypatch.setattr(
        completion_module.urllib_request,
        "build_opener",
        lambda *handlers: captured.extend(handlers) or object(),
    )

    completion_module._verified_opener()

    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert captured


def test_http_redirect_status_is_refused() -> None:
    uri = "https://zenodo.org/api/deposit/depositions"
    opener = RecordingOpener()
    opener.failure = urllib_error.HTTPError(uri, 302, "redirect", {}, None)
    publisher = publisher_for_test(opener)

    with pytest.raises(PostOnlineCompletionError, match="redirect was refused"):
        publisher.create_deposition()

    publisher.close()


def beacon() -> ExecutionBeaconContract:
    return ExecutionBeaconContract(
        drand_network="https://api.drand.sh",
        chain_hash=digest("chain"),
        chain_scheme_id="bls-unchained-g1-rfc9380",
        chain_public_key="aa",
        chain_genesis_unix_seconds=1,
        chain_period_seconds=3,
        execution_round=10,
        label_release_round=20,
        minimum_label_release_safety_rounds=10,
        verification_identity=digest("verification"),
    )


def test_drand_guard_accepts_only_404_as_not_public() -> None:
    contract = beacon()
    uri = f"https://api.drand.sh/{contract.chain_hash}/public/{contract.label_release_round}"

    class NotFoundOpener:
        def open(self, request: object, *, timeout: float) -> None:
            del request, timeout
            raise urllib_error.HTTPError(uri, 404, "missing", {}, None)

    DrandRoundPublicationGuard(opener=NotFoundOpener()).assert_not_public(contract)

    public = RecordingOpener(
        [
            FakeResponse(
                uri,
                json.dumps({"round": contract.label_release_round}).encode(),
            )
        ]
    )
    with pytest.raises(PostOnlineCompletionError, match="already public"):
        DrandRoundPublicationGuard(opener=public).assert_not_public(contract)


def test_drand_redirect_timeout_and_wrong_round_fail_closed() -> None:
    contract = beacon()
    uri = f"https://api.drand.sh/{contract.chain_hash}/public/{contract.label_release_round}"
    redirect = RecordingOpener(
        [
            FakeResponse(
                uri,
                b"{}",
                response_uri="https://example.test/round",
            )
        ]
    )
    with pytest.raises(PostOnlineCompletionError, match="URL changed"):
        DrandRoundPublicationGuard(opener=redirect).assert_not_public(contract)

    timeout = RecordingOpener()
    timeout.failure = TimeoutError()
    with pytest.raises(PostOnlineCompletionError, match="HTTPS request failed"):
        DrandRoundPublicationGuard(opener=timeout).assert_not_public(contract)

    wrong = RecordingOpener([FakeResponse(uri, json.dumps({"round": 19}).encode())])
    with pytest.raises(PostOnlineCompletionError, match="changed the requested round"):
        DrandRoundPublicationGuard(opener=wrong).assert_not_public(contract)
