from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

import fractal_ann_diagnostics.policy_intervention as intervention
from fractal_ann_diagnostics.compiled_policy import CompiledPolicyMaskStore
from fractal_ann_diagnostics.policy_intervention import (
    CATALOG_FILENAME,
    CONFIG_FILENAME,
    OPA_DATA_FILENAME,
    RECEIPT_FILENAME,
    SCHEDULE_FILENAME,
    PolicyInterventionConfig,
    PolicyInterventionError,
    compile_policy_intervention,
    derive_policy_transition_evidence,
    load_policy_intervention_receipt,
    verify_policy_intervention_package,
    write_policy_intervention_package,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _TrialSpy:
    def __init__(self, trial_key: str, family_key: str) -> None:
        self._trial_key = trial_key
        self._family_key = family_key
        self.accesses: list[str] = []

    @property
    def trial_key(self) -> str:
        self.accesses.append("trial_key")
        return self._trial_key

    @property
    def family_key(self) -> str:
        self.accesses.append("family_key")
        return self._family_key

    def __getattr__(self, name: str) -> object:
        self.accesses.append(name)
        raise AssertionError(f"compiler accessed forbidden trial source {name!r}")


class _ExecutionSpy:
    def __init__(
        self,
        *,
        document_count: int = 257,
        universe: str | None = None,
        artifact_sha256: str | None = None,
        trial_keys: tuple[str, ...] | None = None,
    ) -> None:
        self._corpus = "fullwiki"
        self._stage = "sealed"
        self._document_count = document_count
        self._universe = universe or _digest(f"universe-{document_count}")
        self._artifact_sha256 = artifact_sha256 or _digest(
            f"execution-{document_count}-{self._universe}"
        )
        keys = trial_keys or tuple(_digest(f"trial-{index}") for index in range(3))
        self._trials = tuple(
            _TrialSpy(key, _digest(f"family-{position // 3}")) for position, key in enumerate(keys)
        )
        self.accesses: list[str] = []

    @property
    def corpus(self) -> str:
        self.accesses.append("corpus")
        return self._corpus

    @property
    def stage(self) -> str:
        self.accesses.append("stage")
        return self._stage

    @property
    def document_count(self) -> int:
        self.accesses.append("document_count")
        return self._document_count

    @property
    def document_universe_sha256(self) -> str:
        self.accesses.append("document_universe_sha256")
        return self._universe

    @property
    def trials(self) -> tuple[_TrialSpy, ...]:
        self.accesses.append("trials")
        return self._trials

    @property
    def artifact_sha256(self) -> str:
        self.accesses.append("artifact_sha256")
        return self._artifact_sha256

    def __getattr__(self, name: str) -> object:
        self.accesses.append(name)
        raise AssertionError(f"compiler accessed forbidden execution source {name!r}")


def _config(
    *,
    seed: str | None = None,
    baseline_seed: str | None = None,
    revision: str | None = None,
    baseline_revision: str | None = None,
) -> PolicyInterventionConfig:
    return PolicyInterventionConfig(
        seed_sha256=seed or _digest("intervention-seed"),
        baseline_seed_sha256=baseline_seed or _digest("baseline-intervention-seed"),
        policy_bundle_revision=revision or f"sha256:{_digest('opa-bundle')}",
        baseline_policy_revision=(baseline_revision or f"sha256:{_digest('baseline-opa-bundle')}"),
        subject_ids=("reader-a",),
        assignment_repetitions=1,
        grouped_execution_order=("high", "low", "medium"),
    )


def test_compilation_is_deterministic_and_reads_only_admitted_fields() -> None:
    first_source = _ExecutionSpy()
    second_source = _ExecutionSpy()
    first = compile_policy_intervention(first_source, _config())
    second = compile_policy_intervention(second_source, _config())

    assert first.payloads() == second.payloads()
    assert first.receipt.artifact_sha256 == second.receipt.artifact_sha256
    assert set(first_source.accesses) == {
        "artifact_sha256",
        "corpus",
        "document_count",
        "document_universe_sha256",
        "stage",
        "trials",
    }
    assert all(set(row.accesses) == {"family_key", "trial_key"} for row in first_source._trials)
    assert all(
        0 < row.descriptor.authorized_count < first_source.document_count for row in first.masks
    )
    assert [row.allow_rate for row in first.masks] == [0.25, 0.5, 0.75]
    low, medium, high = (row.encoded for row in first.masks)
    assert all((left & ~right) == 0 for left, right in zip(low, medium))
    assert all((left & ~right) == 0 for left, right in zip(medium, high))

    baseline_low, baseline_medium, baseline_high = (row.encoded for row in first.baseline_masks)
    assert all((left & ~right) == 0 for left, right in zip(baseline_low, baseline_medium))
    assert all((left & ~right) == 0 for left, right in zip(baseline_medium, baseline_high))
    assert all(row.policy_churn > 0.0 for row in first.schedule.rows)
    assert {row.policy_state for row in first.receipt.transitions} == {
        "low",
        "medium",
        "high",
    }


def test_transition_derivation_rejects_forged_churn_mask_and_revision(
    tmp_path: Path,
) -> None:
    source = _ExecutionSpy()
    config = _config()
    target = (tmp_path / "compiled-policy").resolve()
    write_policy_intervention_package(source, config, target)
    compiled = compile_policy_intervention(source, config)
    row = compiled.schedule.rows[0]
    current = CompiledPolicyMaskStore(target / CATALOG_FILENAME).mask(
        row.mask_id,
        expected_sha256=row.mask_sha256,
        expected_authorized_count=row.authorized_count,
    )
    evidence = derive_policy_transition_evidence(
        target,
        row,
        document_count=source.document_count,
        current_mask=current,
    )
    assert evidence.policy_churn == row.policy_churn

    with pytest.raises(PolicyInterventionError, match="policy churn differs"):
        derive_policy_transition_evidence(
            target,
            replace(row, policy_churn=min(1.0, row.policy_churn + 0.01)),
            document_count=source.document_count,
            current_mask=current,
        )
    with pytest.raises(PolicyInterventionError, match="baseline mask differs"):
        derive_policy_transition_evidence(
            target,
            replace(row, baseline_mask_sha256="0" * 64),
            document_count=source.document_count,
            current_mask=current,
        )
    forged_current = current.copy()
    forged_current[0] = not forged_current[0]
    with pytest.raises(PolicyInterventionError, match="current policy mask differs"):
        derive_policy_transition_evidence(
            target,
            row,
            document_count=source.document_count,
            current_mask=forged_current,
        )
    with pytest.raises(PolicyInterventionError, match="binding differs"):
        replace(
            compiled.schedule,
            rows=(
                replace(
                    row,
                    baseline_policy_revision=f"sha256:{_digest('forged-baseline')}",
                ),
                *compiled.schedule.rows[1:],
            ),
        )


def test_baseline_seed_and_revision_are_distinct_frozen_inputs() -> None:
    with pytest.raises(PolicyInterventionError, match="baseline_seed_sha256 must differ"):
        _config(
            baseline_seed=_digest("intervention-seed"),
        )
    with pytest.raises(PolicyInterventionError, match="baseline_policy_revision must differ"):
        _config(
            baseline_revision=f"sha256:{_digest('opa-bundle')}",
        )


def test_schedule_has_exact_coverage_and_explicit_mask_grouping() -> None:
    source = _ExecutionSpy(
        trial_keys=tuple(_digest(f"nested-trial-{position}") for position in range(6))
    )
    config = _config()
    compiled = compile_policy_intervention(source, config)
    rows = compiled.schedule.rows

    assert len(rows) == 6
    assert [row.schedule_order for row in rows] == list(range(len(rows)))
    assert [row.group_order for row in rows] == sorted(row.group_order for row in rows)
    assert {row.policy_state for row in rows if row.group_order == 0} == {"high"}
    assert {row.policy_state for row in rows if row.group_order == 1} == {"low"}
    assert {row.policy_state for row in rows if row.group_order == 2} == {"medium"}
    assert all(
        len({row.mask_id for row in rows if row.group_order == group}) == 1 for group in range(3)
    )
    trial_counts = Counter(row.trial_key for row in rows)
    assert set(trial_counts.values()) == {1}
    assignments = {(row.trial_key, row.repetition, row.subject, row.policy_state) for row in rows}
    assert len(assignments) == len(rows)
    family_states: dict[str, set[str]] = {}
    for row in rows:
        family_states.setdefault(row.family_key, set()).add(row.policy_state)
    assert set(map(frozenset, family_states.values())) == {frozenset({"low", "medium", "high"})}
    action_keys = {
        (row.trial_key, action) for row in rows for action in ("hnsw-low", "hnsw-high", "exact")
    }
    assert len(action_keys) == len(rows) * 3
    assert all(
        row.environment_sha256 and dict(row.environment)["assignment_repetition"] == row.repetition
        for row in rows
    )
    assert "action" not in compiled.schedule.to_dict()


def _contains_array(value: object) -> bool:
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        return any(_contains_array(nested) for nested in value.values())
    return False


def test_opa_data_matches_data_fractal_contract_and_stays_constant_size() -> None:
    small = compile_policy_intervention(_ExecutionSpy(document_count=257), _config())
    large = compile_policy_intervention(_ExecutionSpy(document_count=4093), _config())
    payload = small.opa_data.to_dict()

    assert set(payload) == {
        "assignments",
        "document_count",
        "document_universe_sha256",
        "mask_catalog_sha256",
        "policy_revision",
    }
    assert not _contains_array(payload)
    assert b'"document_ids"' not in small.opa_data.canonical_bytes()
    assert len(small.opa_data.canonical_bytes()) < 5000
    assert abs(len(small.opa_data.canonical_bytes()) - len(large.opa_data.canonical_bytes())) < 128
    assignments = payload["assignments"]
    assert isinstance(assignments, dict)
    assert set(assignments) == {"reader-a"}
    for states in assignments.values():
        assert isinstance(states, dict)
        assert set(states) == {"low", "medium", "high"}
        assert all(
            set(value) == {"authorized_count", "mask_id", "mask_sha256"}
            for value in states.values()
        )


def test_finalized_package_rejects_altered_seed_universe_and_overwrite(
    tmp_path: Path,
) -> None:
    source = _ExecutionSpy()
    config = _config()
    target = (tmp_path / "compiled-policy").resolve()
    result = write_policy_intervention_package(source, config, target)

    assert result.root == target
    assert verify_policy_intervention_package(target, source, config) == result
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert stat.S_IMODE((target / CONFIG_FILENAME).stat().st_mode) == 0o600
    receipt = load_policy_intervention_receipt(target / RECEIPT_FILENAME)
    assert receipt.execution_artifact_sha256 == source.artifact_sha256
    assert receipt.document_universe_sha256 == source.document_universe_sha256
    assert receipt.config_sha256 == config.config_sha256
    assert receipt.policy_bundle_revision == config.policy_bundle_revision
    compiled = compile_policy_intervention(source, config)
    assert {row.path for row in receipt.artifacts} == {
        CONFIG_FILENAME,
        CATALOG_FILENAME,
        OPA_DATA_FILENAME,
        SCHEDULE_FILENAME,
        *(row.descriptor.path for row in compiled.masks),
        *(row.descriptor.path for row in compiled.baseline_masks),
    }

    with pytest.raises(PolicyInterventionError, match="already exists"):
        write_policy_intervention_package(source, config, target)

    altered_seed = replace(config, seed_sha256=_digest("altered-seed"))
    with pytest.raises(PolicyInterventionError, match="differs"):
        verify_policy_intervention_package(target, source, altered_seed)
    altered_baseline_seed = replace(
        config,
        baseline_seed_sha256=_digest("altered-baseline-seed"),
    )
    with pytest.raises(PolicyInterventionError, match="differs"):
        verify_policy_intervention_package(target, source, altered_baseline_seed)
    altered_baseline_revision = replace(
        config,
        baseline_policy_revision=f"sha256:{_digest('altered-baseline-bundle')}",
    )
    with pytest.raises(PolicyInterventionError, match="differs"):
        verify_policy_intervention_package(target, source, altered_baseline_revision)

    altered_universe = _ExecutionSpy(
        universe=_digest("altered-universe"),
        artifact_sha256=source.artifact_sha256,
    )
    with pytest.raises(PolicyInterventionError, match="differs"):
        verify_policy_intervention_package(target, altered_universe, config)


@pytest.mark.parametrize("link_kind", ["hardlink", "symlink"])
def test_package_verifier_rejects_output_links(
    tmp_path: Path,
    link_kind: str,
) -> None:
    source = _ExecutionSpy()
    config = _config()
    target = (tmp_path / "compiled-policy").resolve()
    write_policy_intervention_package(source, config, target)
    mask = compile_policy_intervention(source, config).masks[0]
    mask_path = target.joinpath(*mask.descriptor.path.split("/"))
    if link_kind == "hardlink":
        os.link(mask_path, target / "masks" / "mask-alias.bin")
    else:
        replacement = target / "masks" / "replacement.bin"
        replacement.write_bytes(mask_path.read_bytes())
        mask_path.unlink()
        mask_path.symlink_to(replacement)

    with pytest.raises(PolicyInterventionError, match="verify intervention package tree"):
        verify_policy_intervention_package(target, source, config)


def test_duplicate_trials_degenerate_masks_and_movable_revision_fail() -> None:
    duplicate = _digest("duplicate-trial")
    with pytest.raises(PolicyInterventionError, match="duplicate trial"):
        compile_policy_intervention(
            _ExecutionSpy(trial_keys=(duplicate, duplicate)),
            _config(),
        )

    with pytest.raises(PolicyInterventionError, match="exactly three nested trials"):
        compile_policy_intervention(
            _ExecutionSpy(trial_keys=(_digest("trial-a"), _digest("trial-b"))),
            _config(),
        )

    with pytest.raises(PolicyInterventionError, match="degenerate"):
        compile_policy_intervention(
            _ExecutionSpy(document_count=1),
            _config(),
        )

    with pytest.raises(PolicyInterventionError, match="immutable"):
        _config(revision="opa-bundle:latest")

    with pytest.raises(PolicyInterventionError, match="exactly three schedule blocks"):
        PolicyInterventionConfig(
            seed_sha256=_digest("seed"),
            baseline_seed_sha256=_digest("baseline-seed"),
            policy_bundle_revision=f"sha256:{_digest('bundle')}",
            baseline_policy_revision=f"sha256:{_digest('baseline-bundle')}",
            subject_ids=("reader-a", "reader-b"),
        )


def test_schedule_rejects_an_altered_trial_state_mapping() -> None:
    compiled = compile_policy_intervention(_ExecutionSpy(), _config())
    rows = list(compiled.schedule.rows)
    first_key = rows[0].trial_key
    second_key = rows[1].trial_key
    rows[0] = replace(rows[0], trial_key=second_key)
    rows[1] = replace(rows[1], trial_key=first_key)
    changed_rows = tuple(rows)
    with pytest.raises(PolicyInterventionError, match="frozen ranking"):
        replace(
            compiled.schedule,
            rows=changed_rows,
            assignment_map_sha256=intervention._trial_assignment_map_sha256(changed_rows),
        )


class _SubstitutedExecution(_ExecutionSpy):
    def __init__(self) -> None:
        super().__init__()
        self._artifact_reads = 0

    @property
    def artifact_sha256(self) -> str:
        self.accesses.append("artifact_sha256")
        self._artifact_reads += 1
        if self._artifact_reads == 1:
            return self._artifact_sha256
        return _digest("substituted-execution")


def test_execution_source_substitution_is_rejected() -> None:
    with pytest.raises(PolicyInterventionError, match="source changed"):
        compile_policy_intervention(_SubstitutedExecution(), _config())


def test_atomic_finalization_does_not_publish_failed_self_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "compiled-policy").resolve()

    def reject(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise PolicyInterventionError("injected self-verification failure")

    monkeypatch.setattr(intervention, "_verify_expected_package", reject)
    with pytest.raises(PolicyInterventionError, match="injected"):
        write_policy_intervention_package(
            _ExecutionSpy(),
            _config(),
            target,
        )

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_forbidden_output_path_and_config_field_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(PolicyInterventionError, match="forbidden"):
        write_policy_intervention_package(
            _ExecutionSpy(),
            _config(),
            (tmp_path / "sealed-labels").resolve(),
        )

    payload = _config().to_dict()
    payload["qrels"] = "forbidden"
    with pytest.raises(PolicyInterventionError, match="unknown"):
        PolicyInterventionConfig.from_dict(payload)


def test_config_and_schedule_loaders_reject_noncanonical_bytes() -> None:
    config = _config()
    noncanonical = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    with pytest.raises(PolicyInterventionError, match="canonical JSON"):
        intervention.loads_policy_intervention_config(noncanonical)

    compiled = compile_policy_intervention(_ExecutionSpy(), config)
    schedule = compiled.schedule.to_dict()
    schedule["rows"][0]["response_outcome"] = True  # type: ignore[index]
    encoded = json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(PolicyInterventionError, match="unknown"):
        intervention.loads_canonical_trial_schedule(encoded)
