from __future__ import annotations

import ast
import inspect
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import fractal_ann_diagnostics.sealed_orchestrator as orchestrator
from fractal_ann_diagnostics.artifact_integrity import (
    ArtifactVerificationReceipt,
    VerifiedArtifact,
)
from fractal_ann_diagnostics.audit import VerifiedProvenanceRegistry
from fractal_ann_diagnostics.controller import GovernedRetriever
from fractal_ann_diagnostics.corpora import (
    CorpusDocument,
    EvidenceQuery,
    NormalizedCorpus,
)
from fractal_ann_diagnostics.custody import OnlineCustodyAdmissionReceipt
from fractal_ann_diagnostics.label_separation import (
    OnlineDocument,
    OnlineExecutionArtifact,
    OnlineTrial,
)
from fractal_ann_diagnostics.online_runner import run_online_action_matrix
from fractal_ann_diagnostics.policy import AuthorizationPolicy
from fractal_ann_diagnostics.sealed_orchestrator import (
    RequiredArtifactIdBindings,
    SealedOrchestratorError,
    derive_required_artifact_id_bindings,
    load_required_artifact_id_bindings,
    run_admitted_online_matrix,
    write_required_artifact_id_bindings,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA, SealedRunReceipt, manifest_sha256

_MANIFEST_SHA256 = "a" * 64
_RUNNER_IDENTITY = "admitted-online-runner"
_COMPONENTS = (
    "application",
    "controller",
    "corpus",
    "embedding",
    "index",
    "policy",
)
_EXECUTION_ID = "online-execution-artifact"
_RUNNER_ID = "runner-binary-artifact"
_SOURCE_ID = "source-code-artifact"


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _corpus() -> NormalizedCorpus:
    documents = tuple(
        CorpusDocument(
            document_id=index,
            external_id=f"document-{index}",
            title=f"Document {index}",
            text=f"fixed content {index}",
            source_uri=f"fixture://document/{index}",
            content_hash=f"sha256:{_digest(f'content:{index}')}",
        )
        for index in range(2)
    )
    return NormalizedCorpus(
        name="scifact",
        stage="sealed",
        documents=documents,
        queries=(
            EvidenceQuery(
                query_id="query-0",
                query_family="family-0",
                text="find the fixed evidence",
                corpus="scifact",
                stage="sealed",
                answer=None,
                gold_evidence=None,
            ),
        ),
    )


def _execution(corpus: NormalizedCorpus, *, key_id: str = "online-key") -> OnlineExecutionArtifact:
    return OnlineExecutionArtifact(
        key_id=key_id,
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
        trials=(
            OnlineTrial(
                trial_key=_digest("trial:query-0"),
                family_key=_digest("family:family-0"),
                text="find the fixed evidence",
                corpus=corpus.name,
                stage=corpus.stage,
            ),
        ),
    )


def _verified_artifact(
    artifact_id: str,
    digest: str,
    *,
    exact: bool = True,
) -> VerifiedArtifact:
    if exact:
        return VerifiedArtifact(
            artifact_id=artifact_id,
            relative_path=f"objects/{artifact_id}.bin",
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
    return VerifiedArtifact(
        artifact_id=artifact_id,
        relative_path=f"objects/{artifact_id}",
        kind="directory",
        exact=False,
        expected_sha256=digest,
        verified_sha256=digest,
        file_count=1,
        directory_count=0,
        byte_count=1,
        observed_file_count=2,
        observed_directory_count=0,
        observed_byte_count=2,
    )


def _frozen_manifest_receipt() -> tuple[dict[str, object], ArtifactVerificationReceipt]:
    artifacts: list[dict[str, object]] = []

    def add(role: str, *, corpus_id: str | None = None) -> None:
        artifact_id = f"{corpus_id}-{role}" if corpus_id is not None else role
        row: dict[str, object] = {
            "id": artifact_id,
            "role": role,
            "sha256": _digest(artifact_id),
        }
        if corpus_id is not None:
            row["corpus_id"] = corpus_id
        if role == "online-execution":
            row["revision"] = f"sha256:{_digest(artifact_id + ':logical-plan')}"
        artifacts.append(row)

    for role in (
        "exact-authorized-oracle",
        "frozen-controller",
        "online-staging-package",
        "opa-pdp",
        "opa-runtime-binary",
        "primary-embedding",
        "query-partition-audit",
        "source-code",
        "strict-authorized-hnsw",
    ):
        add(role)
    for corpus_id in FIXED_CORPORA:
        for role in (
            "authorized-index-store",
            "corpus-normalizer",
            "embedding-store",
            "online-execution",
            "policy-workload",
            "sealed-label-ciphertext",
            "sealed-labels",
            "timelock-encryption-receipt",
            "trial-runtime-package",
        ):
            add(role, corpus_id=corpus_id)
    manifest: dict[str, object] = {
        "artifacts": artifacts,
        "status": "frozen",
    }
    receipt = ArtifactVerificationReceipt(
        manifest_sha256=manifest_sha256(manifest),
        artifacts=tuple(
            _verified_artifact(str(row["id"]), str(row["sha256"]))
            for row in manifest["artifacts"]  # type: ignore[union-attr]
        ),
    )
    return manifest, receipt


@pytest.fixture
def manifest_binding_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, object],
    ArtifactVerificationReceipt,
    list[tuple[object, bool]],
]:
    calls: list[tuple[object, bool]] = []

    def validate_fixture(payload: object, *, require_frozen: bool = False) -> None:
        calls.append((payload, require_frozen))
        assert isinstance(payload, dict)
        assert payload.get("status") == "frozen"
        assert require_frozen is True

    monkeypatch.setattr(orchestrator, "validate_study_manifest", validate_fixture)
    manifest, receipt = _frozen_manifest_receipt()
    return manifest, receipt, calls


def _manifest_artifact_id(
    manifest: dict[str, object],
    role: str,
    *,
    corpus_id: str | None = None,
) -> str:
    matches = [
        row
        for row in manifest["artifacts"]  # type: ignore[union-attr]
        if row["role"] == role and (corpus_id is None or row.get("corpus_id") == corpus_id)
    ]
    assert len(matches) == 1
    return str(matches[0]["id"])


@dataclass(frozen=True)
class _Harness:
    admission: OnlineCustodyAdmissionReceipt
    bindings: RequiredArtifactIdBindings
    execution: OnlineExecutionArtifact
    run_receipt: SealedRunReceipt
    retriever: GovernedRetriever
    registry: VerifiedProvenanceRegistry


def _harness(
    *,
    nonexact_id: str | None = None,
    execution_outer_sha256: str | None = None,
) -> _Harness:
    corpus = _corpus()
    execution = _execution(corpus)
    component_ids = {component: f"{component}-artifact" for component in _COMPONENTS}
    digests = {
        **{artifact_id: _digest(component) for component, artifact_id in component_ids.items()},
        _EXECUTION_ID: execution_outer_sha256 or execution.artifact_sha256,
        _RUNNER_ID: _digest("runner-binary"),
        _SOURCE_ID: _digest("source-code"),
    }
    receipt = ArtifactVerificationReceipt(
        manifest_sha256=_MANIFEST_SHA256,
        artifacts=tuple(
            _verified_artifact(
                artifact_id,
                digest,
                exact=artifact_id != nonexact_id,
            )
            for artifact_id, digest in digests.items()
        ),
    )
    registry = VerifiedProvenanceRegistry(
        corpus=corpus,
        verification_receipt=receipt,
        component_artifact_ids=component_ids,
    )
    policy = AuthorizationPolicy(
        roles=("analyst",),
        visibility=np.ones((1, len(corpus.documents)), dtype=bool),
        version="registered-policy",
        document_universe_sha256=registry.document_universe_sha256,
    )
    retriever = GovernedRetriever(
        np.eye(len(corpus.documents), dtype=np.float32),
        policy,
        "analyst",
        expected_document_universe_sha256=registry.document_universe_sha256,
    )
    run_receipt = SealedRunReceipt(
        manifest_sha256=_MANIFEST_SHA256,
        protocol_version="0.3.0",
        started_at_utc="2026-07-14T12:00:00+00:00",
        runner_identity=_RUNNER_IDENTITY,
        code_commit="c" * 40,
        runner_image=f"ghcr.io/example/runner@sha256:{'d' * 64}",
        protocol_registration_receipt_uri="file:///controlled/protocol-receipt.json",
        protocol_registration_receipt_sha256="e" * 64,
        protocol_registration_record_uri="file:///controlled/protocol-record.json",
        verification_receipt_uri="file:///controlled/artifact-verification.json",
        verification_receipt_sha256=receipt.receipt_sha256,
        receipt_uri="file:///controlled/sealed-run.json",
    )
    admitted_ids = tuple(sorted(digests, key=lambda value: value.encode("utf-8")))
    admission = OnlineCustodyAdmissionReceipt(
        manifest_sha256=_MANIFEST_SHA256,
        run_receipt_sha256=run_receipt.binding_sha256,
        artifact_verification_receipt_sha256=receipt.receipt_sha256,
        custody_seal_receipt_sha256="1" * 64,
        online_artifact_verification_receipt_sha256="2" * 64,
        runner_identity=_RUNNER_IDENTITY,
        verified_artifact_ids=admitted_ids,
    )
    provenance_bindings = tuple(
        sorted(component_ids.items(), key=lambda item: item[0].encode("utf-8"))
    )
    bindings = RequiredArtifactIdBindings(
        verification_receipt=receipt,
        execution_artifact_id=_EXECUTION_ID,
        execution_revision_sha256=execution.artifact_sha256,
        runner_artifact_ids=(_RUNNER_ID,),
        source_artifact_ids=(_SOURCE_ID,),
        retriever_artifact_ids=tuple(
            sorted(component_ids.values(), key=lambda value: value.encode("utf-8"))
        ),
        provenance_component_artifact_ids=provenance_bindings,
    )
    return _Harness(
        admission=admission,
        bindings=bindings,
        execution=execution,
        run_receipt=run_receipt,
        retriever=retriever,
        registry=registry,
    )


def _invoke(harness: _Harness) -> object:
    return run_admitted_online_matrix(
        admission_receipt=harness.admission,
        required_artifacts=harness.bindings,
        execution=harness.execution,
        run_receipt=harness.run_receipt,
        retriever=harness.retriever,
        provenance_registry=harness.registry,
        trial_runtimes={},
        permutation_seed=71,
        expected_policy_version="registered-policy",
        query_partition_audit_sha256="3" * 64,
        pseudonym_key=b"pseudonym-key-material-is-at-least-32-bytes",
        pseudonym_key_id="pseudonym-key",
        k=1,
        policy_action="retrieve",
        partition_label="primary",
        occurred_at_factory=None,
    )


def test_valid_admission_invokes_existing_runner_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness()
    sentinel = object()
    calls: list[dict[str, object]] = []

    def runner_spy(**kwargs: object) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(orchestrator, "run_online_action_matrix", runner_spy)

    assert _invoke(harness) is sentinel
    assert len(calls) == 1
    assert calls[0]["execution"] is harness.execution
    assert calls[0]["run_receipt"] is harness.run_receipt
    assert calls[0]["retriever"] is harness.retriever
    assert calls[0]["provenance_registry"] is harness.registry


def test_admission_separates_execution_tree_pin_from_logical_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer_tree_sha256 = _digest("sealed-execution-package-tree")
    harness = _harness(execution_outer_sha256=outer_tree_sha256)
    execution_row = next(
        row
        for row in harness.bindings.verification_receipt.artifacts
        if row.artifact_id == harness.bindings.execution_artifact_id
    )
    assert execution_row.verified_sha256 == outer_tree_sha256
    assert execution_row.verified_sha256 != harness.bindings.execution_revision_sha256
    assert harness.bindings.execution_revision_sha256 == harness.execution.artifact_sha256

    sentinel = object()
    calls = 0

    def runner_spy(**_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return sentinel

    monkeypatch.setattr(orchestrator, "run_online_action_matrix", runner_spy)
    assert _invoke(harness) is sentinel
    assert calls == 1


@pytest.mark.parametrize(
    "case",
    (
        "manifest",
        "run-receipt",
        "runner-identity",
        "verification-receipt",
        "unknown-admitted-id",
        "execution-id",
        "runner-id",
        "source-id",
        "execution-digest",
        "provenance-components",
        "nonexact-required",
    ),
)
def test_every_binding_mismatch_rejects_before_runner_call(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(nonexact_id=_RUNNER_ID if case == "nonexact-required" else None)

    if case == "manifest":
        harness = replace(
            harness,
            admission=replace(harness.admission, manifest_sha256="b" * 64),
        )
    elif case == "run-receipt":
        harness = replace(
            harness,
            admission=replace(harness.admission, run_receipt_sha256="b" * 64),
        )
    elif case == "runner-identity":
        harness = replace(
            harness,
            admission=replace(harness.admission, runner_identity="another-runner"),
        )
    elif case == "verification-receipt":
        harness = replace(
            harness,
            admission=replace(
                harness.admission,
                artifact_verification_receipt_sha256="b" * 64,
            ),
        )
    elif case == "unknown-admitted-id":
        identifiers = tuple(
            sorted(
                (*harness.admission.verified_artifact_ids, "unknown-artifact"),
                key=lambda value: value.encode("utf-8"),
            )
        )
        harness = replace(
            harness,
            admission=replace(harness.admission, verified_artifact_ids=identifiers),
        )
    elif case in {"execution-id", "runner-id", "source-id"}:
        removed = {
            "execution-id": _EXECUTION_ID,
            "runner-id": _RUNNER_ID,
            "source-id": _SOURCE_ID,
        }[case]
        harness = replace(
            harness,
            admission=replace(
                harness.admission,
                verified_artifact_ids=tuple(
                    artifact_id
                    for artifact_id in harness.admission.verified_artifact_ids
                    if artifact_id != removed
                ),
            ),
        )
    elif case == "execution-digest":
        harness = replace(harness, execution=_execution(_corpus(), key_id="changed-key"))
    elif case == "provenance-components":
        reduced = harness.bindings.provenance_component_artifact_ids[1:]
        harness = replace(
            harness,
            bindings=replace(
                harness.bindings,
                retriever_artifact_ids=tuple(
                    sorted(
                        (artifact_id for _, artifact_id in reduced),
                        key=lambda value: value.encode("utf-8"),
                    )
                ),
                provenance_component_artifact_ids=reduced,
            ),
        )

    calls = 0

    def runner_spy(**_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(orchestrator, "run_online_action_matrix", runner_spy)
    with pytest.raises(SealedOrchestratorError):
        _invoke(harness)
    assert calls == 0


def test_binding_object_rejects_implicit_or_incomplete_dependency_sets() -> None:
    harness = _harness()
    with pytest.raises(SealedOrchestratorError, match="must equal"):
        replace(
            harness.bindings,
            retriever_artifact_ids=harness.bindings.retriever_artifact_ids[:-1],
        )
    with pytest.raises(SealedOrchestratorError, match="non-empty tuple"):
        replace(harness.bindings, runner_artifact_ids=())


def test_wrapper_signature_preserves_every_existing_runner_input() -> None:
    wrapper_parameters = tuple(inspect.signature(run_admitted_online_matrix).parameters.values())
    runner_parameters = tuple(inspect.signature(run_online_action_matrix).parameters.values())
    assert tuple(parameter.name for parameter in wrapper_parameters[:2]) == (
        "admission_receipt",
        "required_artifacts",
    )
    assert wrapper_parameters[2:] == runner_parameters


def test_orchestrator_source_has_no_forbidden_execution_capability() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fractal_ann_diagnostics"
        / "sealed_orchestrator.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden_modules = {
        "aiohttp",
        "boto3",
        "confirmatory_analysis",
        "httpx",
        "label_separation",
        "requests",
        "socket",
        "urllib",
    }
    assert not any(
        any(
            module == forbidden or module.endswith(f".{forbidden}")
            for forbidden in forbidden_modules
        )
        for module in imported_modules
    )
    forbidden_fragments = (
        "SealedLabel",
        "confirmatory_analysis",
        "decrypt",
        "label_separation",
        "plaintext",
        "post_run",
        "qrels",
    )
    assert all(fragment not in source for fragment in forbidden_fragments)

    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_admitted_online_matrix"
    )
    parameter_names = {
        argument.arg
        for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
    }
    assert not {
        name
        for name in parameter_names
        if any(term in name for term in ("decrypt", "outcome", "plaintext", "qrel", "score"))
    }
    assert {name for name in parameter_names if "label" in name} == {"partition_label"}


def test_required_artifact_bindings_round_trip_as_one_closed_control(
    tmp_path: Path,
) -> None:
    bindings = _harness().bindings
    target = (tmp_path / "required-artifact-bindings.json").resolve()
    write_required_artifact_id_bindings(bindings, target)

    assert load_required_artifact_id_bindings(target) == bindings
    assert sha256(target.read_bytes()).hexdigest() == bindings.file_sha256
    assert target.read_bytes() == bindings.canonical_file_bytes()


def test_required_artifact_bindings_loader_rejects_unknown_duplicate_and_links(
    tmp_path: Path,
) -> None:
    bindings = _harness().bindings
    payload = bindings.to_dict()
    payload["alternate_execution_id"] = "injected"
    unknown = (tmp_path / "unknown.json").resolve()
    unknown.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    with pytest.raises(SealedOrchestratorError, match="unknown"):
        load_required_artifact_id_bindings(unknown)

    duplicate = (tmp_path / "duplicate.json").resolve()
    duplicate.write_bytes(
        bindings.canonical_file_bytes().replace(
            b'{"execution_artifact_id":',
            b'{"execution_artifact_id":"injected","execution_artifact_id":',
            1,
        )
    )
    with pytest.raises(SealedOrchestratorError, match="duplicate key"):
        load_required_artifact_id_bindings(duplicate)

    source = (tmp_path / "source.json").resolve()
    source.write_bytes(bindings.canonical_file_bytes())
    symbolic = (tmp_path / "symbolic.json").resolve()
    symbolic.symlink_to(source)
    with pytest.raises(SealedOrchestratorError, match="safely"):
        load_required_artifact_id_bindings(symbolic)

    hard = (tmp_path / "hard.json").resolve()
    hard.hardlink_to(source)
    with pytest.raises(SealedOrchestratorError, match="hard-linked"):
        load_required_artifact_id_bindings(hard)


def test_manifest_constructor_derives_the_registered_corpus_closure(
    manifest_binding_fixture: tuple[
        dict[str, object], ArtifactVerificationReceipt, list[tuple[object, bool]]
    ],
) -> None:
    manifest, receipt, calls = manifest_binding_fixture

    bindings = derive_required_artifact_id_bindings(
        manifest,
        receipt,
        corpus_id="scifact",
    )

    assert bindings.execution_artifact_id == _manifest_artifact_id(
        manifest,
        "online-execution",
        corpus_id="scifact",
    )
    assert bindings.runner_artifact_ids == tuple(
        sorted(
            (
                _manifest_artifact_id(manifest, role)
                for role in (
                    "exact-authorized-oracle",
                    "frozen-controller",
                    "opa-pdp",
                    "opa-runtime-binary",
                    "source-code",
                    "strict-authorized-hnsw",
                )
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    assert bindings.source_artifact_ids == tuple(
        sorted(
            (
                _manifest_artifact_id(
                    manifest,
                    role,
                    corpus_id="scifact" if corpus_bound else None,
                )
                for role, corpus_bound in (
                    ("authorized-index-store", True),
                    ("embedding-store", True),
                    ("online-staging-package", False),
                    ("policy-workload", True),
                    ("query-partition-audit", False),
                    ("trial-runtime-package", True),
                )
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    expected_components = {
        "application": _manifest_artifact_id(manifest, "source-code"),
        "controller": _manifest_artifact_id(manifest, "frozen-controller"),
        "corpus": _manifest_artifact_id(
            manifest,
            "corpus-normalizer",
            corpus_id="scifact",
        ),
        "embedding": _manifest_artifact_id(manifest, "primary-embedding"),
        "index": _manifest_artifact_id(manifest, "strict-authorized-hnsw"),
        "policy": _manifest_artifact_id(manifest, "opa-pdp"),
    }
    assert dict(bindings.provenance_component_artifact_ids) == expected_components
    assert bindings.retriever_artifact_ids == tuple(
        sorted(expected_components.values(), key=lambda value: value.encode("utf-8"))
    )

    withheld_ids = {
        str(row["id"])
        for row in manifest["artifacts"]  # type: ignore[union-attr]
        if row["role"]
        in {
            "sealed-labels",
            "sealed-label-ciphertext",
            "timelock-encryption-receipt",
        }
    }
    assert bindings.required_artifact_ids.isdisjoint(withheld_ids)
    assert calls == [(manifest, True)]
    execution_outer = next(
        row.verified_sha256
        for row in receipt.artifacts
        if row.artifact_id == bindings.execution_artifact_id
    )
    assert bindings.execution_revision_sha256 == _digest(
        bindings.execution_artifact_id + ":logical-plan"
    )
    assert bindings.execution_revision_sha256 != execution_outer


def test_manifest_constructor_has_no_caller_id_override_surface(
    manifest_binding_fixture: tuple[
        dict[str, object], ArtifactVerificationReceipt, list[tuple[object, bool]]
    ],
) -> None:
    manifest, receipt, _ = manifest_binding_fixture
    parameters = inspect.signature(derive_required_artifact_id_bindings).parameters
    assert tuple(parameters) == ("frozen_manifest", "verification_receipt", "corpus_id")
    with pytest.raises(TypeError, match="unexpected keyword"):
        derive_required_artifact_id_bindings(
            manifest,
            receipt,
            corpus_id="scifact",
            runner_artifact_ids=("scifact-sealed-labels",),  # type: ignore[call-arg]
        )


@pytest.mark.parametrize("case", ("omission", "extra", "component-swap"))
def test_manifest_constructor_rejects_receipt_closure_attacks(
    case: str,
    manifest_binding_fixture: tuple[
        dict[str, object], ArtifactVerificationReceipt, list[tuple[object, bool]]
    ],
) -> None:
    manifest, receipt, _ = manifest_binding_fixture
    rows = list(receipt.artifacts)
    if case == "omission":
        omitted = _manifest_artifact_id(manifest, "source-code")
        rows = [row for row in rows if row.artifact_id != omitted]
    elif case == "extra":
        rows.append(_verified_artifact("injected-artifact", _digest("injected")))
    else:
        first_id = _manifest_artifact_id(manifest, "source-code")
        second_id = _manifest_artifact_id(manifest, "frozen-controller")
        first = next(row for row in rows if row.artifact_id == first_id)
        second = next(row for row in rows if row.artifact_id == second_id)
        rows = [
            replace(
                row,
                expected_sha256=(
                    second.expected_sha256
                    if row.artifact_id == first_id
                    else first.expected_sha256
                    if row.artifact_id == second_id
                    else row.expected_sha256
                ),
                verified_sha256=(
                    second.verified_sha256
                    if row.artifact_id == first_id
                    else first.verified_sha256
                    if row.artifact_id == second_id
                    else row.verified_sha256
                ),
            )
            for row in rows
        ]
    attacked = ArtifactVerificationReceipt(
        manifest_sha256=receipt.manifest_sha256,
        artifacts=tuple(rows),
    )

    with pytest.raises(SealedOrchestratorError):
        derive_required_artifact_id_bindings(
            manifest,
            attacked,
            corpus_id="scifact",
        )


@pytest.mark.parametrize("case", ("duplicate-role", "wrong-corpus", "missing-role"))
def test_manifest_constructor_rejects_manifest_role_attacks(
    case: str,
    manifest_binding_fixture: tuple[
        dict[str, object], ArtifactVerificationReceipt, list[tuple[object, bool]]
    ],
) -> None:
    manifest, _, _ = manifest_binding_fixture
    attacked = deepcopy(manifest)
    artifacts = attacked["artifacts"]
    assert isinstance(artifacts, list)
    if case == "duplicate-role":
        source = next(row for row in artifacts if row["role"] == "source-code")
        duplicate = dict(source)
        duplicate["id"] = "alternate-source-code"
        duplicate["uri"] = "https://example.test/alternate-source-code"
        duplicate["sha256"] = _digest("alternate-source-code")
        artifacts.append(duplicate)
    elif case == "wrong-corpus":
        normalizer = next(
            row
            for row in artifacts
            if row["role"] == "corpus-normalizer" and row["corpus_id"] == "scifact"
        )
        normalizer["corpus_id"] = "hotpotqa-fullwiki"
    else:
        source = next(row for row in artifacts if row["role"] == "source-code")
        artifacts.remove(source)

    attacked_receipt = ArtifactVerificationReceipt(
        manifest_sha256=manifest_sha256(attacked),
        artifacts=tuple(
            _verified_artifact(str(row["id"]), str(row["sha256"])) for row in artifacts
        ),
    )
    with pytest.raises(SealedOrchestratorError):
        derive_required_artifact_id_bindings(
            attacked,
            attacked_receipt,
            corpus_id="scifact",
        )


def test_manifest_constructor_rejects_cross_corpus_request(
    manifest_binding_fixture: tuple[
        dict[str, object], ArtifactVerificationReceipt, list[tuple[object, bool]]
    ],
) -> None:
    manifest, receipt, _ = manifest_binding_fixture
    with pytest.raises(SealedOrchestratorError, match="fixed confirmatory suite"):
        derive_required_artifact_id_bindings(
            manifest,
            receipt,
            corpus_id="another-corpus",
        )
