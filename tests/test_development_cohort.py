from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fractal_ann_diagnostics.development_cohort as cohort
from fractal_ann_diagnostics.development_cohort import (
    CALIBRATION_FAMILY_COUNT,
    DEVELOPMENT_COHORT_SELECTION_SCHEMA,
    FIT_FAMILY_COUNT,
    DevelopmentCohortError,
    DevelopmentCohortMaterializationReceipt,
    DevelopmentEmbeddingBinding,
    DevelopmentExecutionPlan,
    DevelopmentExecutionTrial,
    load_development_cohort_selection,
    load_development_execution_plan,
    materialize_development_cohort,
    select_development_cohort,
    verify_materialized_development_cohort,
)
from fractal_ann_diagnostics.joint_power_design import EVIDENCE_CORPORA, FIXED_CORPORA
from fractal_ann_diagnostics.policy_intervention import (
    PolicyInterventionConfig,
    compile_policy_intervention,
)
from fractal_ann_diagnostics.scalable_custody import SourceArtifactPin
from fractal_ann_diagnostics.study_data import (
    ASSIGNMENT_ALGORITHM,
    ASSIGNMENT_SCHEMA,
    INVENTORY_SCHEMA,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(_canonical(row) + b"\n" for row in rows)
    path.write_bytes(encoded)
    return encoded


def _pin(
    root: Path,
    relative_path: str,
    rows: list[dict[str, Any]],
    *,
    dataset: str | None,
    stage: str | None,
    role: str,
) -> SourceArtifactPin:
    encoded = _write_jsonl(root / relative_path, rows)
    return SourceArtifactPin(
        path=relative_path,
        sha256=_digest(encoded),
        byte_count=len(encoded),
        record_count=len(rows),
        dataset=dataset,
        stage=stage,
        role=role,
        visibility="online",
    )


@dataclass
class _Fixture:
    root: Path
    inventory_sha256: str
    audit_sha256: str
    audit: SimpleNamespace
    query_sources: tuple[SourceArtifactPin, ...]
    label_sources: tuple[SourceArtifactPin, ...]


def _stage(tmp_path: Path) -> _Fixture:
    root = tmp_path / "stage"
    root.mkdir()
    assignments: list[dict[str, Any]] = []
    query_sources: list[SourceArtifactPin] = []
    label_sources: list[SourceArtifactPin] = []
    query_counts: list[SimpleNamespace] = []
    for corpus in FIXED_CORPORA:
        for source_stage, family_count in (
            ("fit", FIT_FAMILY_COUNT + 2),
            ("calibration", CALIBRATION_FAMILY_COUNT + 2),
        ):
            queries: list[dict[str, Any]] = []
            qrels: list[dict[str, Any]] = []
            evidence: list[dict[str, Any]] = []
            for component_index in range(family_count):
                component = _digest(f"component:{corpus}:{source_stage}:{component_index:04d}")
                suffixes = ("a", "b") if component_index == 0 else ("a",)
                for suffix in suffixes:
                    query_id = f"q-{component_index:04d}-{suffix}"
                    text = f"{corpus} {source_stage} question {component_index} {suffix}"
                    queries.append({"id": query_id, "text": text})
                    assignments.append(
                        {
                            "assignment_key_sha256": _digest(
                                f"assignment:{corpus}:{source_stage}:{query_id}"
                            ),
                            "dataset": corpus,
                            "domain": None,
                            "partition_component_sha256": component,
                            "query_id": query_id,
                            "query_text_sha256": _digest(text),
                            "schema_version": ASSIGNMENT_SCHEMA,
                            "source_split": f"fixture-{source_stage}",
                            "stage": source_stage,
                        }
                    )
                    qrels.append(
                        {
                            "document_id": "document-0000",
                            "query_id": query_id,
                            "relevance": 1,
                        }
                    )
                    if corpus in EVIDENCE_CORPORA:
                        evidence.append(
                            {
                                "answer": None,
                                "evidence_bundles": [
                                    {
                                        "bundle_id": f"bundle-{query_id}",
                                        "locations": [
                                            {
                                                "document_id": "document-0000",
                                                "locator": "line-1",
                                            }
                                        ],
                                    }
                                ],
                                "label_metadata": [],
                                "query_id": query_id,
                            }
                        )
            queries.sort(key=lambda row: str(row["id"]).encode())
            qrels.sort(key=lambda row: str(row["query_id"]).encode())
            evidence.sort(key=lambda row: str(row["query_id"]).encode())
            query_pin = _pin(
                root,
                f"datasets/{corpus}/{source_stage}/queries.jsonl",
                queries,
                dataset=corpus,
                stage=source_stage,
                role="queries",
            )
            query_sources.append(query_pin)
            query_counts.append(
                SimpleNamespace(
                    dataset=corpus,
                    stage=source_stage,
                    query_count=len(queries),
                )
            )
            label_sources.append(
                _pin(
                    root,
                    f"datasets/{corpus}/{source_stage}/qrels.jsonl",
                    qrels,
                    dataset=corpus,
                    stage=source_stage,
                    role="qrels",
                )
            )
            if evidence:
                label_sources.append(
                    _pin(
                        root,
                        f"datasets/{corpus}/{source_stage}/evidence-bundles.jsonl",
                        evidence,
                        dataset=corpus,
                        stage=source_stage,
                        role="evidence-bundles",
                    )
                )
    assignments.sort(
        key=lambda row: (
            str(row["dataset"]).encode(),
            str(row["stage"]).encode(),
            str(row["query_id"]).encode(),
        )
    )
    assignment_pin = _pin(
        root,
        "assignments.jsonl",
        assignments,
        dataset=None,
        stage=None,
        role="assignments",
    )
    artifacts = tuple(
        sorted(
            (assignment_pin, *query_sources, *label_sources),
            key=lambda row: row.path.encode(),
        )
    )
    inventory = {
        "artifacts": [row.to_dict() for row in artifacts],
        "assignment_algorithm": {
            "component_edges": [
                "normalized-query-text-equality",
                "registered-near-duplicate-token-rule",
                "shared-positive-document-content",
                "shared-positive-relevance-document",
            ],
            "cross_source_split_policy": "exclude-entire-component-v1",
            "fit_calibration_component_ratio": "4:1",
            "name": ASSIGNMENT_ALGORITHM,
            "three_way_component_ratio": "3:1:1",
        },
        "assignment_seed_sha256": _digest("assignment-seed"),
        "bright_document_identity": {},
        "bright_domains": [],
        "config_sha256": _digest("staging-config"),
        "counts": {corpus: {"documents": 257} for corpus in FIXED_CORPORA},
        "hotpotqa_fullwiki_scope": {},
        "withhold_sealed_labels_from_online_process": True,
        "schema_version": INVENTORY_SCHEMA,
        "sources": [],
    }
    inventory_bytes = _canonical(inventory) + b"\n"
    (root / "inventory.json").write_bytes(inventory_bytes)
    inventory_sha256 = _digest(inventory_bytes)
    (root / "inventory.sha256").write_text(
        f"{inventory_sha256}  inventory.json\n",
        encoding="ascii",
    )
    audit_sha256 = _digest("typed-partition-audit")
    audit = SimpleNamespace(
        artifact_sha256=audit_sha256,
        assignment_artifact_sha256=assignment_pin.sha256,
        component_membership_sha256=_digest("component-membership"),
        source_artifact_set_sha256=_digest("audit-source-artifact-set"),
        source_artifacts=artifacts,
        query_counts=tuple(query_counts),
    )
    return _Fixture(
        root=root,
        inventory_sha256=inventory_sha256,
        audit_sha256=audit_sha256,
        audit=audit,
        query_sources=tuple(query_sources),
        label_sources=tuple(label_sources),
    )


def _patch_audit(monkeypatch: pytest.MonkeyPatch, fixture: _Fixture) -> None:
    def load(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return fixture.audit

    monkeypatch.setattr(cohort, "load_scalable_partition_audit", load)


def _select(
    tmp_path: Path,
    fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Any]:
    _patch_audit(monkeypatch, fixture)
    output = tmp_path / "selection-receipt.json"
    receipt = select_development_cohort(
        fixture.root,
        output,
        staged_inventory_sha256=fixture.inventory_sha256,
        partition_audit_path=tmp_path / "partition-audit.json",
        partition_audit_sha256=fixture.audit_sha256,
    )
    return output, receipt


def test_selection_is_fixed_deterministic_and_never_opens_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _stage(tmp_path)
    _patch_audit(monkeypatch, fixture)
    original = cohort._iter_canonical_jsonl
    opened_roles: list[str] = []

    def sentinel(root_descriptor: int, source: SourceArtifactPin):
        opened_roles.append(source.role)
        if source.role in {"qrels", "evidence-bundles"}:
            raise AssertionError("selection opened a label-bearing source")
        yield from original(root_descriptor, source)

    monkeypatch.setattr(cohort, "_iter_canonical_jsonl", sentinel)
    output = tmp_path / "selection-receipt.json"
    receipt = select_development_cohort(
        fixture.root,
        output,
        staged_inventory_sha256=fixture.inventory_sha256,
        partition_audit_path=tmp_path / "partition-audit.json",
        partition_audit_sha256=fixture.audit_sha256,
    )

    assert receipt.schema_version == DEVELOPMENT_COHORT_SELECTION_SCHEMA
    assert set(opened_roles) == {"assignments", "queries"}
    assert sum(row.requested_family_count for row in receipt.selections) == (
        len(FIXED_CORPORA) * (FIT_FAMILY_COUNT + CALIBRATION_FAMILY_COUNT)
    )
    assert all(
        len(row.selected_families) == row.requested_family_count
        and row.available_component_count == row.requested_family_count + 2
        for row in receipt.selections
    )
    restored = load_development_cohort_selection(
        output,
        expected_artifact_sha256=receipt.artifact_sha256,
        expected_inventory_sha256=fixture.inventory_sha256,
    )
    assert restored == receipt
    second = select_development_cohort(
        fixture.root,
        tmp_path / "second-selection-receipt.json",
        staged_inventory_sha256=fixture.inventory_sha256,
        partition_audit_path=tmp_path / "partition-audit.json",
        partition_audit_sha256=fixture.audit_sha256,
    )
    assert second.canonical_file_bytes() == receipt.canonical_file_bytes()
    with pytest.raises(DevelopmentCohortError, match="already exists"):
        select_development_cohort(
            fixture.root,
            output,
            staged_inventory_sha256=fixture.inventory_sha256,
            partition_audit_path=tmp_path / "partition-audit.json",
            partition_audit_sha256=fixture.audit_sha256,
        )


def test_selection_loader_rejects_open_schema_pin_substitution_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _stage(tmp_path)
    selection_path, receipt = _select(tmp_path, fixture, monkeypatch)

    payload = receipt.to_dict()
    payload["unexpected"] = True
    open_schema = tmp_path / "open-schema.json"
    open_schema.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(DevelopmentCohortError, match="fields differ"):
        load_development_cohort_selection(open_schema)

    payload = receipt.to_dict()
    payload["query_artifacts"][0]["dataset"] = "substituted-corpus"
    substitution = tmp_path / "substitution.json"
    substitution.write_bytes(_canonical(payload) + b"\n")
    with pytest.raises(DevelopmentCohortError, match="query artifact set"):
        load_development_cohort_selection(substitution)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(receipt.canonical_file_bytes().replace(b"{", b"{ ", 1))
    with pytest.raises(DevelopmentCohortError, match="not canonical"):
        load_development_cohort_selection(noncanonical)

    linked = tmp_path / "linked-selection.json"
    os.symlink(selection_path, linked)
    with pytest.raises(ValueError):
        load_development_cohort_selection(linked)

    real_output_parent = tmp_path / "real-output-parent"
    real_output_parent.mkdir()
    linked_output_parent = tmp_path / "linked-output-parent"
    os.symlink(real_output_parent, linked_output_parent)
    with pytest.raises(DevelopmentCohortError, match="without symbolic links"):
        select_development_cohort(
            fixture.root,
            linked_output_parent / "selection.json",
            staged_inventory_sha256=fixture.inventory_sha256,
            partition_audit_path=tmp_path / "partition-audit.json",
            partition_audit_sha256=fixture.audit_sha256,
        )

    fit = receipt.selection("scifact", "development-fit")
    calibration = receipt.selection("scifact", "development-calibration")
    fit_component = fit.selected_families[0].component_sha256
    original = calibration.selected_families[0]
    overlapping = replace(
        original,
        component_sha256=fit_component,
        family_rank_sha256=cohort.family_selection_rank(
            corpus="scifact",
            stage="development-calibration",
            selection_seed_sha256=calibration.selection_seed_sha256,
            component_sha256=fit_component,
        ),
        representative_rank_sha256=cohort.representative_selection_rank(
            corpus="scifact",
            stage="development-calibration",
            selection_seed_sha256=calibration.selection_seed_sha256,
            component_sha256=fit_component,
            query_id_sha256=original.query_id_sha256,
        ),
    )
    calibration_families = tuple(
        sorted(
            (overlapping, *calibration.selected_families[1:]),
            key=lambda row: (row.family_rank_sha256, row.component_sha256),
        )
    )
    overlapping_calibration = replace(
        calibration,
        selected_families=calibration_families,
    )
    with pytest.raises(DevelopmentCohortError, match="fit and calibration.*overlap"):
        replace(
            receipt,
            selections=tuple(
                overlapping_calibration if row == calibration else row for row in receipt.selections
            ),
        )


def test_query_denominator_drift_is_rejected_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _stage(tmp_path)
    first = fixture.audit.query_counts[0]
    fixture.audit.query_counts = (
        SimpleNamespace(
            dataset=first.dataset,
            stage=first.stage,
            query_count=first.query_count - 1,
        ),
        *fixture.audit.query_counts[1:],
    )
    _patch_audit(monkeypatch, fixture)

    with pytest.raises(DevelopmentCohortError, match="query count"):
        select_development_cohort(
            fixture.root,
            tmp_path / "selection-receipt.json",
            staged_inventory_sha256=fixture.inventory_sha256,
            partition_audit_path=tmp_path / "partition-audit.json",
            partition_audit_sha256=fixture.audit_sha256,
        )


def test_materialization_reproduction_failure_never_reaches_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _stage(tmp_path)
    selection_path, receipt = _select(tmp_path, fixture, monkeypatch)
    query_path = fixture.root / fixture.query_sources[0].path
    query_path.write_bytes(query_path.read_bytes().replace(b"question", b"tampered", 1))
    label_opened = False

    def label_sentinel(*_args: object, **_kwargs: object) -> tuple[dict[str, object], ...]:
        nonlocal label_opened
        label_opened = True
        raise AssertionError("labels opened before selection reproduction")

    monkeypatch.setattr(cohort, "_label_sources", label_sentinel)
    monkeypatch.setattr(cohort, "_materialize_qrels", label_sentinel)
    with pytest.raises(DevelopmentCohortError):
        materialize_development_cohort(
            fixture.root,
            selection_path,
            tmp_path / "materialized",
            selection_receipt_sha256=receipt.artifact_sha256,
            partition_audit_path=tmp_path / "partition-audit.json",
            embedding_bindings=(),
        )
    assert label_opened is False


def _fake_embeddings(
    selection: Any,
    bindings: tuple[DevelopmentEmbeddingBinding, ...],
) -> tuple[
    tuple[DevelopmentEmbeddingBinding, ...],
    dict[tuple[str, str], tuple[SimpleNamespace, dict[str, int]]],
]:
    verified: dict[tuple[str, str], tuple[SimpleNamespace, dict[str, int]]] = {}
    for binding in bindings:
        stratum = selection.selection(binding.corpus, binding.development_stage)
        document_order = _digest(f"documents:{binding.corpus}")
        query_order = _digest(f"queries:{binding.development_stage}:{binding.corpus}")
        embedding = SimpleNamespace(
            document_count=257,
            receipt_sha256=binding.receipt_sha256,
            row_orders={
                "documents": SimpleNamespace(row_order_sha256=document_order),
                "queries": SimpleNamespace(row_order_sha256=query_order),
            },
        )
        positions = {
            query_id: position for position, query_id in enumerate(stratum.selected_query_ids)
        }
        verified[(binding.development_stage, binding.corpus)] = (
            embedding,
            positions,
        )
    return bindings, verified


def test_materialization_emits_closed_label_files_and_compiler_ready_plans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    fixture = _stage(tmp_path)
    selection_path, selection = _select(tmp_path, fixture, monkeypatch)
    bindings = tuple(
        DevelopmentEmbeddingBinding(
            corpus=corpus,
            development_stage=stage,
            root=(tmp_path / "embedding" / stage / corpus).resolve(),
            receipt_sha256=_digest(f"embedding:{stage}:{corpus}"),
        )
        for stage in ("development-fit", "development-calibration")
        for corpus in FIXED_CORPORA
    )
    monkeypatch.setattr(
        cohort,
        "_verify_embedding_bindings",
        lambda supplied, selected, **_kwargs: _fake_embeddings(
            selected,
            tuple(supplied),
        ),
    )
    output = tmp_path / "materialized"
    receipt = materialize_development_cohort(
        fixture.root,
        selection_path,
        output,
        selection_receipt_sha256=selection.artifact_sha256,
        partition_audit_path=tmp_path / "partition-audit.json",
        embedding_bindings=bindings,
    )

    assert receipt.selection_receipt_sha256 == selection.artifact_sha256
    assert len(receipt.artifacts) == 37
    assert not tuple(output.rglob("*.npy"))
    assert (
        verify_materialized_development_cohort(
            output,
            expected_receipt_sha256=receipt.artifact_sha256,
            verify_label_payloads=False,
        )
        == receipt
    )
    original_read = cohort.read_secure_regular_file

    def reject_label_read(path: str | Path, **kwargs: object) -> bytes:
        if Path(path).name in {"qrels.jsonl", "evidence-bundles.jsonl"}:
            raise AssertionError("online closure verification opened development labels")
        return original_read(path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cohort, "read_secure_regular_file", reject_label_read)
    assert (
        verify_materialized_development_cohort(
            output,
            expected_receipt_sha256=receipt.artifact_sha256,
            verify_label_payloads=False,
        )
        == receipt
    )
    monkeypatch.setattr(cohort, "read_secure_regular_file", original_read)

    extra = output / "unexpected.txt"
    extra.write_text("unexpected", encoding="utf-8")
    with pytest.raises(DevelopmentCohortError, match="membership differs"):
        verify_materialized_development_cohort(output)
    extra.unlink()

    linked = output / "unexpected-link"
    os.symlink(output / "selection-receipt.json", linked)
    with pytest.raises(DevelopmentCohortError, match="symbolic link"):
        verify_materialized_development_cohort(output)
    linked.unlink()

    package_parent_link = tmp_path / "package-parent-link"
    os.symlink(output.parent, package_parent_link)
    with pytest.raises(DevelopmentCohortError, match="symlink"):
        verify_materialized_development_cohort(package_parent_link / output.name)

    payload = receipt.to_dict()
    payload["artifacts"] = payload["artifacts"][1:]
    with pytest.raises(DevelopmentCohortError, match="protocol-complete"):
        DevelopmentCohortMaterializationReceipt.from_dict(payload)

    assert (
        cohort.main(
            [
                "verify-selection",
                "--receipt",
                str(selection_path),
                "--expected-sha256",
                selection.artifact_sha256,
                "--expected-inventory-sha256",
                fixture.inventory_sha256,
            ]
        )
        == 0
    )
    selection_result = json.loads(capfd.readouterr().out)
    assert selection_result["command"] == "verify-selection"
    assert selection_result["artifact_sha256"] == selection.artifact_sha256

    assert (
        cohort.main(
            [
                "verify-materialization",
                "--root",
                str(output),
                "--expected-sha256",
                receipt.artifact_sha256,
            ]
        )
        == 0
    )
    materialization_result = json.loads(capfd.readouterr().out)
    assert materialization_result["command"] == "verify-materialization"
    assert materialization_result["artifact_sha256"] == receipt.artifact_sha256
    plan_path = output / "development-fit" / "scifact" / "execution-plan.json"
    plan = load_development_execution_plan(plan_path)
    assert len(plan.trials) == FIT_FAMILY_COUNT * 3
    assert len({row.family_key for row in plan.trials}) == FIT_FAMILY_COUNT
    assert {row.nested_index for row in plan.trials} == {0, 1, 2}
    compiled = compile_policy_intervention(
        plan,
        PolicyInterventionConfig(
            seed_sha256=_digest("policy-seed"),
            baseline_seed_sha256=_digest("baseline-policy-seed"),
            policy_bundle_revision=f"sha256:{_digest('policy-bundle')}",
            baseline_policy_revision=f"sha256:{_digest('baseline-policy-bundle')}",
            subject_ids=("development-reader",),
            assignment_repetitions=1,
            grouped_execution_order=("high", "low", "medium"),
        ),
    )
    assert compiled.schedule.execution_artifact_sha256 == plan.artifact_sha256

    qrel_path = output / "development-fit" / "scifact" / "qrels.jsonl"
    receipt_path = output / "materialization-receipt.json"
    original_qrels = qrel_path.read_bytes()
    original_receipt = receipt_path.read_bytes()
    first_qrel = original_qrels.splitlines(keepends=True)[0]
    tampered_qrels = first_qrel + original_qrels
    qrel_path.write_bytes(tampered_qrels)
    tampered_payload = receipt.to_dict()
    qrel_binding = next(
        row
        for row in tampered_payload["artifacts"]
        if row["path"] == "development-fit/scifact/qrels.jsonl"
    )
    qrel_binding["byte_count"] = len(tampered_qrels)
    qrel_binding["record_count"] += 1
    qrel_binding["sha256"] = _digest(tampered_qrels)
    tampered_receipt = DevelopmentCohortMaterializationReceipt.from_dict(tampered_payload)
    receipt_path.write_bytes(tampered_receipt.canonical_file_bytes())
    with pytest.raises(DevelopmentCohortError, match="repeated or not canonically ordered"):
        verify_materialized_development_cohort(
            output,
            expected_receipt_sha256=tampered_receipt.artifact_sha256,
            verify_label_payloads=True,
        )
    qrel_path.write_bytes(original_qrels)
    receipt_path.write_bytes(original_receipt)


def test_execution_plan_rejects_nested_denominator_drift() -> None:
    trials: list[DevelopmentExecutionTrial] = []
    for family_index in range(FIT_FAMILY_COUNT):
        family = _digest(f"family-{family_index}")
        query_id = f"query-{family_index}"
        nested_count = 2 if family_index == FIT_FAMILY_COUNT - 1 else 3
        for nested_index in range(nested_count):
            trials.append(
                DevelopmentExecutionTrial(
                    family_key=family,
                    trial_key=cohort._hash_parts(
                        cohort.DEVELOPMENT_TRIAL_DOMAIN,
                        "scifact",
                        "development-fit",
                        family,
                        cohort.nested_trial_source_value(query_id, nested_index),
                    ),
                    query_id=query_id,
                    query_row=family_index,
                    nested_index=nested_index,
                )
            )
    with pytest.raises(DevelopmentCohortError, match="trial counts"):
        DevelopmentExecutionPlan(
            corpus="scifact",
            stage="development-fit",
            document_count=257,
            document_universe_sha256=_digest("documents"),
            document_row_order_sha256=_digest("documents"),
            query_row_order_sha256=_digest("queries"),
            embedding_receipt_sha256=_digest("embedding"),
            selection_receipt_sha256=_digest("selection"),
            selected_family_count=FIT_FAMILY_COUNT,
            trials=tuple(trials),
        )


def test_execution_plan_recomputes_trial_keys_and_requires_distinct_query_rows() -> None:
    trials: list[DevelopmentExecutionTrial] = []
    for family_index in range(CALIBRATION_FAMILY_COUNT):
        family = _digest(f"calibration-family-{family_index}")
        query_id = f"calibration-query-{family_index}"
        for nested_index in range(3):
            trials.append(
                DevelopmentExecutionTrial(
                    family_key=family,
                    trial_key=cohort._hash_parts(
                        cohort.DEVELOPMENT_TRIAL_DOMAIN,
                        "scifact",
                        "development-calibration",
                        family,
                        cohort.nested_trial_source_value(query_id, nested_index),
                    ),
                    query_id=query_id,
                    query_row=family_index,
                    nested_index=nested_index,
                )
            )
    plan = DevelopmentExecutionPlan(
        corpus="scifact",
        stage="development-calibration",
        document_count=257,
        document_universe_sha256=_digest("documents"),
        document_row_order_sha256=_digest("documents"),
        query_row_order_sha256=_digest("queries"),
        embedding_receipt_sha256=_digest("embedding"),
        selection_receipt_sha256=_digest("selection"),
        selected_family_count=CALIBRATION_FAMILY_COUNT,
        trials=tuple(sorted(trials, key=lambda row: (row.family_key, row.nested_index))),
    )

    tampered = replace(plan.trials[0], trial_key=_digest("tampered-trial-key"))
    with pytest.raises(DevelopmentCohortError, match="trial key does not reproduce"):
        replace(plan, trials=(tampered, *plan.trials[1:]))

    repeated_row_family = plan.trials[3].family_key
    repeated_row_trials = tuple(
        replace(row, query_row=plan.trials[0].query_row)
        if row.family_key == repeated_row_family
        else row
        for row in plan.trials
    )
    with pytest.raises(DevelopmentCohortError, match="distinct query IDs and embedding rows"):
        replace(plan, trials=repeated_row_trials)
