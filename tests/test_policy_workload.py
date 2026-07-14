from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError

import pytest

from fractal_ann_diagnostics.policy_workload import (
    POLICY_WORKLOAD_SCHEMA,
    ExplicitTrialSchedule,
    FrozenJsonObject,
    PolicyDataArtifactBinding,
    PolicyWorkloadError,
    RequiredPolicyTrial,
    SeededTrialSchedule,
    load_policy_workload,
    loads_policy_workload,
    parse_policy_workload,
    validate_policy_workload,
    validate_policy_workload_suite,
)


def _payload(*, corpus_id: str = "scifact", suffix: str = "") -> dict[str, object]:
    return {
        "artifact_id": f"policy-workload-{corpus_id}-v1{suffix}",
        "artifact_kind": "policy-data",
        "corpus_id": corpus_id,
        "document_universe_sha256": "a" * 64,
        "documents": [
            {
                "attributes": {
                    "classification": "public",
                    "owners": ["research"],
                },
                "document_id": "d-1",
            },
            {
                "attributes": {"classification": "restricted"},
                "document_id": "d-2",
            },
        ],
        "environments": [
            {
                "attributes": {
                    "network_zone": "research",
                    "request": {"hour_utc": 14},
                },
                "environment_id": "env-1",
            }
        ],
        "execution_schedule": {
            "kind": "explicit",
            "trial_order": ["trial-1", "trial-2"],
        },
        "mutation_schedule": [{"before_trial_id": "trial-2", "mutation_ids": ["mutation-1"]}],
        "mutations": [
            {
                "attributes": {"classification": "public"},
                "mutation_id": "mutation-1",
                "operation": "set-attributes",
                "target_id": "d-2",
                "target_kind": "document",
            }
        ],
        "schema_version": POLICY_WORKLOAD_SCHEMA,
        "subjects": [
            {
                "attributes": {
                    "clearance": "internal",
                    "groups": ["research", "staff"],
                },
                "subject_id": "subject-1",
            }
        ],
        "trials": [
            {
                "environment_id": "env-1",
                "query_id": "query-1",
                "subject_id": "subject-1",
                "trial_id": "trial-1",
            },
            {
                "environment_id": "env-1",
                "query_id": "query-2",
                "subject_id": "subject-1",
                "trial_id": "trial-2",
            },
        ],
    }


def _expectations(workload):
    return {
        "expected_corpus_id": "scifact",
        "expected_document_universe_sha256": "a" * 64,
        "expected_document_ids": ("d-1", "d-2"),
        "expected_query_ids": ("query-1", "query-2"),
        "required_trials": (
            RequiredPolicyTrial("trial-1", "query-1"),
            RequiredPolicyTrial("trial-2", "query-2"),
        ),
        "expected_artifact_binding": PolicyDataArtifactBinding(
            artifact_id="policy-workload-scifact-v1",
            canonical_sha256=workload.canonical_sha256,
        ),
    }


def test_parse_round_trip_is_canonical_and_deeply_immutable(tmp_path) -> None:
    payload = _payload()
    workload = parse_policy_workload(payload)

    assert workload.to_dict() == payload
    assert loads_policy_workload(workload.canonical_bytes()) == workload
    assert len(workload.canonical_sha256) == 64
    assert workload.ordered_trial_ids == ("trial-1", "trial-2")
    assert isinstance(workload.subjects[0].attributes, FrozenJsonObject)
    assert workload.subjects[0].attributes["groups"] == ("research", "staff")
    with pytest.raises(TypeError):
        workload.subjects[0].attributes["new"] = "forbidden"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        workload.corpus_id = "changed"  # type: ignore[misc]

    path = tmp_path / "workload.json"
    path.write_bytes(workload.canonical_bytes())
    assert load_policy_workload(path) == workload


def test_canonical_digest_is_independent_of_object_key_order() -> None:
    first = _payload()
    second = json.loads(json.dumps(first))
    second["subjects"][0]["attributes"] = {  # type: ignore[index]
        "groups": ["research", "staff"],
        "clearance": "internal",
    }
    assert (
        parse_policy_workload(first).canonical_sha256
        == parse_policy_workload(second).canonical_sha256
    )


def test_seeded_schedule_has_a_reproducible_total_order() -> None:
    payload = _payload()
    payload["execution_schedule"] = {
        "algorithm": "sha256-seeded-permutation-v1",
        "kind": "seeded",
        "seed": 20260713,
    }
    workload = parse_policy_workload(payload)
    assert isinstance(workload.execution_schedule, SeededTrialSchedule)
    assert workload.ordered_trial_ids == workload.execution_schedule.ordered_trial_ids(
        ("trial-2", "trial-1")
    )
    assert set(workload.ordered_trial_ids) == {"trial-1", "trial-2"}


def test_empty_mutation_schedule_is_valid_when_there_are_no_mutations() -> None:
    payload = _payload()
    payload["mutations"] = []
    payload["mutation_schedule"] = []
    workload = parse_policy_workload(payload)
    assert workload.mutations == ()
    assert workload.mutation_schedule == ()


@pytest.mark.parametrize(
    "path,value",
    [
        (("subjects", 0, "attributes", "label"), True),
        (("subjects", 0, "attributes", "modelLabels"), [1]),
        (("environments", 0, "attributes", "nested", "gold_documents"), ["d-1"]),
        (("documents", 0, "attributes", "relevance_score"), 1.0),
        (("documents", 0, "attributes", "answerText"), "x"),
        (("documents", 0, "attributes", "evidence_bundle"), ["d-1"]),
        (("documents", 0, "attributes", "correct_choice"), "d-1"),
        (("documents", 0, "attributes", "targetId"), "d-1"),
        (("mutations", 0, "attributes", "expected-outcome"), "pass"),
        (("mutations", 0, "attributes", "ground_truth"), {"x": 1}),
    ],
)
def test_label_or_outcome_fields_are_rejected_at_any_depth(path, value) -> None:
    payload = _payload()
    cursor = payload
    for part in path[:-1]:
        if isinstance(part, int):
            cursor = cursor[part]  # type: ignore[index,assignment]
        else:
            cursor = cursor.setdefault(part, {})  # type: ignore[assignment,union-attr]
    cursor[path[-1]] = value  # type: ignore[index]
    with pytest.raises(PolicyWorkloadError, match="forbidden label or outcome"):
        parse_policy_workload(payload)


def test_label_words_in_values_are_not_mistaken_for_fields() -> None:
    payload = _payload()
    payload["subjects"][0]["attributes"]["team"] = "gold answer review"  # type: ignore[index]
    assert parse_policy_workload(payload).subjects[0].attributes["team"] == ("gold answer review")


def test_loader_rejects_symlinked_and_hard_linked_workloads(tmp_path) -> None:
    workload = parse_policy_workload(_payload())
    source = tmp_path / "source.json"
    source.write_bytes(workload.canonical_bytes())
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(source)
    with pytest.raises(PolicyWorkloadError, match="symlink"):
        load_policy_workload(symlink)

    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(source)
    with pytest.raises(PolicyWorkloadError, match="hard-linked"):
        load_policy_workload(hardlink)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.update({"unknown": 1}), "schema mismatch"),
        (
            lambda p: p["trials"][0].update({"repeat": 1}),  # type: ignore[index]
            "schema mismatch",
        ),
        (lambda p: p.update({"trials": tuple(p["trials"])}), "JSON array"),
        (
            lambda p: p.update(
                {
                    "execution_schedule": {
                        "kind": "explicit",
                        "seed": 1,
                        "trial_order": ["trial-1", "trial-2"],
                    }
                }
            ),
            "schema mismatch",
        ),
        (
            lambda p: p.update(
                {
                    "execution_schedule": {
                        "algorithm": "sha256-seeded-permutation-v1",
                        "kind": "seeded",
                        "seed": True,
                    }
                }
            ),
            "JSON integer",
        ),
        (
            lambda p: p["subjects"][0].update({"attributes": ("not", "json")}),  # type: ignore[index]
            "JSON object",
        ),
        (lambda p: p.update({1: "not-a-field"}), "field names must be JSON strings"),
        (
            lambda p: p["subjects"][0]["attributes"].update({1: "not-a-field"}),  # type: ignore[index]
            "field names must be JSON strings",
        ),
    ],
)
def test_schema_and_json_container_types_are_closed(mutation, match) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(PolicyWorkloadError, match=match):
        parse_policy_workload(payload)


def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected() -> None:
    canonical = json.dumps(_payload(), separators=(",", ":"))
    duplicate = canonical.replace(
        '"artifact_id":"policy-workload-scifact-v1",',
        '"artifact_id":"one","artifact_id":"two",',
        1,
    )
    with pytest.raises(PolicyWorkloadError, match="duplicate key 'artifact_id'"):
        loads_policy_workload(duplicate)

    nonfinite = canonical.replace('"hour_utc":14', '"hour_utc":NaN')
    with pytest.raises(PolicyWorkloadError, match="non-finite"):
        loads_policy_workload(nonfinite)


def test_loader_rejects_semantically_equal_noncanonical_bytes() -> None:
    workload = parse_policy_workload(_payload())
    pretty = json.dumps(workload.to_dict(), indent=2, ensure_ascii=False)
    with pytest.raises(PolicyWorkloadError, match="bytes are not canonical"):
        loads_policy_workload(pretty)


@pytest.mark.parametrize(
    "value,match",
    [
        (2**53, "portable JSON range"),
        (-0.0, "negative zero"),
        (object(), "non-JSON value"),
    ],
)
def test_attribute_scalar_domain_is_portable(value, match) -> None:
    payload = _payload()
    payload["subjects"][0]["attributes"]["risk"] = value  # type: ignore[index]
    with pytest.raises(PolicyWorkloadError, match=match):
        parse_policy_workload(payload)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda p: p["subjects"].append(copy.deepcopy(p["subjects"][0])),  # type: ignore[union-attr,index]
            "duplicate stable IDs",
        ),
        (
            lambda p: p["documents"].reverse(),  # type: ignore[union-attr]
            "sorted by document_id",
        ),
        (
            lambda p: p["trials"][0].update({"subject_id": "missing"}),  # type: ignore[index]
            "unknown subject",
        ),
        (
            lambda p: p["trials"][0].update({"environment_id": "missing"}),  # type: ignore[index]
            "unknown environment",
        ),
        (
            lambda p: p["mutations"][0].update({"target_id": "missing"}),  # type: ignore[index]
            "unknown document",
        ),
        (
            lambda p: p["execution_schedule"].update(  # type: ignore[union-attr]
                {"trial_order": ["trial-1"]}
            ),
            "every declared trial",
        ),
        (
            lambda p: p.update({"mutation_schedule": []}),
            "does not exactly cover",
        ),
        (
            lambda p: p["mutation_schedule"][0].update(  # type: ignore[index]
                {"mutation_ids": ["mutation-1", "mutation-1"]}
            ),
            "unique within an entry",
        ),
    ],
)
def test_identity_references_and_schedule_coverage_fail_closed(mutation, match) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(PolicyWorkloadError, match=match):
        parse_policy_workload(payload)


def test_mutation_schedule_must_follow_execution_order() -> None:
    payload = _payload()
    payload["mutations"] = [
        {
            "attributes": {"clearance": "restricted"},
            "mutation_id": "mutation-0",
            "operation": "set-attributes",
            "target_id": "subject-1",
            "target_kind": "subject",
        },
        payload["mutations"][0],  # type: ignore[index]
    ]
    payload["mutation_schedule"] = [
        {"before_trial_id": "trial-2", "mutation_ids": ["mutation-1"]},
        {"before_trial_id": "trial-1", "mutation_ids": ["mutation-0"]},
    ]
    with pytest.raises(PolicyWorkloadError, match="execution order"):
        parse_policy_workload(payload)


def test_single_workload_validation_binds_artifact_corpus_queries_and_trials() -> None:
    workload = parse_policy_workload(_payload())
    assert validate_policy_workload(workload, **_expectations(workload)) is workload


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("expected_corpus_id", "bright", "corpus mismatch"),
        (
            "expected_document_universe_sha256",
            "b" * 64,
            "universe digest",
        ),
        ("expected_query_ids", ("query-1",), "query set mismatch"),
        ("expected_document_ids", ("d-1",), "document attribute coverage"),
        (
            "required_trials",
            (
                RequiredPolicyTrial("trial-1", "query-2"),
                RequiredPolicyTrial("trial-2", "query-1"),
            ),
            "pairing mismatch",
        ),
    ],
)
def test_single_workload_validation_rejects_sealed_identity_changes(field, value, match) -> None:
    workload = parse_policy_workload(_payload())
    expectations = _expectations(workload)
    expectations[field] = value
    with pytest.raises(PolicyWorkloadError, match=match):
        validate_policy_workload(workload, **expectations)


def test_single_workload_validation_rejects_artifact_id_or_digest_changes() -> None:
    workload = parse_policy_workload(_payload())
    expectations = _expectations(workload)
    expectations["expected_artifact_binding"] = PolicyDataArtifactBinding(
        "other-policy-data", workload.canonical_sha256
    )
    with pytest.raises(PolicyWorkloadError, match="artifact ID"):
        validate_policy_workload(workload, **expectations)

    expectations["expected_artifact_binding"] = PolicyDataArtifactBinding(
        workload.artifact_id, "b" * 64
    )
    with pytest.raises(PolicyWorkloadError, match="artifact digest"):
        validate_policy_workload(workload, **expectations)


def test_required_trials_are_exact_not_a_minimum_count() -> None:
    workload = parse_policy_workload(_payload())
    expectations = _expectations(workload)
    expectations["required_trials"] = (RequiredPolicyTrial("trial-1", "query-1"),)
    with pytest.raises(PolicyWorkloadError, match="pairing mismatch"):
        validate_policy_workload(workload, **expectations)


def test_suite_validation_requires_exact_corpus_and_expectation_coverage() -> None:
    scifact = parse_policy_workload(_payload())
    bright_payload = _payload(corpus_id="bright")
    bright_payload["document_universe_sha256"] = "b" * 64
    bright = parse_policy_workload(bright_payload)
    workloads = (scifact, bright)
    corpora = ("scifact", "bright")
    universes = {"scifact": "a" * 64, "bright": "b" * 64}
    queries = {corpus: ("query-1", "query-2") for corpus in corpora}
    documents = {corpus: ("d-1", "d-2") for corpus in corpora}
    trials = {
        corpus: (
            RequiredPolicyTrial("trial-1", "query-1"),
            RequiredPolicyTrial("trial-2", "query-2"),
        )
        for corpus in corpora
    }
    bindings = {
        workload.corpus_id: PolicyDataArtifactBinding(
            workload.artifact_id, workload.canonical_sha256
        )
        for workload in workloads
    }

    validated = validate_policy_workload_suite(
        workloads,
        expected_corpus_ids=corpora,
        expected_document_universe_sha256=universes,
        expected_document_ids=documents,
        expected_query_ids=queries,
        required_trials=trials,
        expected_artifact_bindings=bindings,
    )
    assert tuple(workload.corpus_id for workload in validated) == (
        "bright",
        "scifact",
    )

    incomplete_universes = {"scifact": "a" * 64}
    with pytest.raises(PolicyWorkloadError, match="corpus coverage mismatch"):
        validate_policy_workload_suite(
            workloads,
            expected_corpus_ids=corpora,
            expected_document_universe_sha256=incomplete_universes,
            expected_document_ids=documents,
            expected_query_ids=queries,
            required_trials=trials,
            expected_artifact_bindings=bindings,
        )


def test_suite_validation_rejects_missing_or_duplicate_corpus_workloads() -> None:
    workload = parse_policy_workload(_payload())
    kwargs = {
        "expected_corpus_ids": ("scifact",),
        "expected_document_universe_sha256": {"scifact": "a" * 64},
        "expected_document_ids": {"scifact": ("d-1", "d-2")},
        "expected_query_ids": {"scifact": ("query-1", "query-2")},
        "required_trials": {
            "scifact": (
                RequiredPolicyTrial("trial-1", "query-1"),
                RequiredPolicyTrial("trial-2", "query-2"),
            )
        },
        "expected_artifact_bindings": {
            "scifact": PolicyDataArtifactBinding(workload.artifact_id, workload.canonical_sha256)
        },
    }
    with pytest.raises(PolicyWorkloadError, match="duplicate corpus"):
        validate_policy_workload_suite((workload, workload), **kwargs)
    with pytest.raises(PolicyWorkloadError, match="workload corpus coverage"):
        validate_policy_workload_suite((), **kwargs)


def test_schedule_dataclasses_reject_ambiguous_direct_construction() -> None:
    with pytest.raises(PolicyWorkloadError, match="unique trial IDs"):
        ExplicitTrialSchedule(("trial-1", "trial-1"))
    with pytest.raises(PolicyWorkloadError, match="non-negative"):
        SeededTrialSchedule(seed=True)
