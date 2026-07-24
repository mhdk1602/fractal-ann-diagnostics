from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.post_embedding_development as operator
from fractal_ann_diagnostics.policy_intervention import PolicyInterventionError
from fractal_ann_diagnostics.post_embedding_development import (
    ANALYSIS_DIRECTORY,
    EXECUTION_DIRECTORY,
    FREEZE_DIRECTORY,
    INDEX_DIRECTORY,
    JOINT_POWER_DIRECTORY,
    JOINT_POWER_INVOCATION_FILENAME,
    JOINT_POWER_SELECTION_AUDIT_FILENAME,
    MATERIALIZATION_DIRECTORY,
    OPERATOR_CONFIG_FILENAME,
    POLICY_DIRECTORY,
    RECEIPT_FILENAME,
    SELECTION_FILENAME,
    PostEmbeddingArtifactPin,
    PostEmbeddingDevelopmentConfig,
    PostEmbeddingDevelopmentError,
    PostEmbeddingDevelopmentReceipt,
    PostEmbeddingStratumReceipt,
    load_post_embedding_development_config,
    load_post_embedding_development_receipt,
    post_embedding_development_status,
    verify_post_embedding_development,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _config(tmp_path: Path) -> PostEmbeddingDevelopmentConfig:
    return PostEmbeddingDevelopmentConfig(
        production_embedding_config_path=(tmp_path / "embedding-config.json").resolve(),
        production_embedding_config_sha256=_digest("embedding-config"),
        full_staged_root=(tmp_path / "study-data").resolve(),
        full_staged_inventory_sha256=_digest("inventory"),
        partition_audit_path=(tmp_path / "partition-audit.json").resolve(),
        partition_audit_file_sha256=_digest("audit"),
        design_seed_sha256=_digest("design"),
        output_root=(tmp_path / "development-operator").resolve(),
    )


def _strata() -> tuple[PostEmbeddingStratumReceipt, ...]:
    return tuple(
        PostEmbeddingStratumReceipt(
            corpus=corpus,
            development_stage=stage,
            source_stage=operator.SOURCE_STAGE[stage],
            embedding_receipt_sha256=_digest(f"embedding:{corpus}"),
            policy_config_sha256=_digest(f"policy-config:{stage}:{corpus}"),
            policy_intervention_receipt_sha256=_digest(f"policy:{stage}:{corpus}"),
            authorized_index_config_sha256=_digest("index-config"),
            authorized_index_receipt_sha256=_digest(f"index:{stage}:{corpus}"),
        )
        for stage, corpus in operator._stratum_keys()
    )


def _artifact_pins() -> tuple[PostEmbeddingArtifactPin, ...]:
    return tuple(
        PostEmbeddingArtifactPin(
            stage_id=stage_id,
            path=path,
            kind=kind,
            sha256=_digest(path),
            file_count=1,
            directory_count=0,
            byte_count=17,
        )
        for path, (stage_id, kind) in operator._artifact_contract().items()
    )


def _receipt() -> PostEmbeddingDevelopmentReceipt:
    return PostEmbeddingDevelopmentReceipt(
        config_sha256=_digest("config"),
        full_staged_inventory_sha256=_digest("inventory"),
        partition_audit_file_sha256=_digest("audit"),
        partition_audit_sha256=_digest("audit"),
        embedding_suite_receipt_sha256=_digest("suite"),
        embedding_bindings_sha256=_digest("bindings"),
        selection_receipt_sha256=_digest("selection"),
        development_materialization_receipt_sha256=_digest("materialization"),
        design_seed_sha256=_digest("design"),
        index_config_sha256=_digest("index-config"),
        execution_config_sha256=_digest("execution-config"),
        execution_receipt_sha256=_digest("execution"),
        freeze_config_sha256=_digest("freeze-config"),
        freeze_receipt_sha256=_digest("freeze-receipt"),
        freeze_tree_sha256=_digest("freeze-tree"),
        joint_power_invocation_sha256=_digest("invocation"),
        joint_power_config_sha256=_digest("power-config"),
        joint_power_report_sha256=_digest("power-report"),
        joint_power_report_tree_sha256=_digest("power-tree"),
        selected_families_per_corpus=75,
        development_family_count=1_375,
        paired_trial_count=4_125,
        paired_action_row_count=16_500,
        strata=_strata(),
        artifacts=_artifact_pins(),
    )


def _write_operator_top_level(config: PostEmbeddingDevelopmentConfig) -> None:
    config.output_root.mkdir(mode=0o700)
    for name in operator._KNOWN_TOP_LEVEL:
        path = config.output_root / name
        if name in {
            ANALYSIS_DIRECTORY,
            EXECUTION_DIRECTORY,
            FREEZE_DIRECTORY,
            INDEX_DIRECTORY,
            MATERIALIZATION_DIRECTORY,
            POLICY_DIRECTORY,
        }:
            path.mkdir(mode=0o700)
        else:
            path.write_bytes(b"x\n")
    (config.output_root / OPERATOR_CONFIG_FILENAME).write_bytes(config.canonical_file_bytes())
    (config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY).mkdir(mode=0o700)


def test_config_round_trip_is_closed_and_pinned(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = (tmp_path / "operator-config.json").resolve()
    path.write_bytes(config.canonical_file_bytes())

    assert (
        load_post_embedding_development_config(
            path,
            expected_sha256=config.file_sha256,
        )
        == config
    )
    with pytest.raises(PostEmbeddingDevelopmentError, match="caller pin"):
        load_post_embedding_development_config(path, expected_sha256=_digest("wrong"))
    path.write_bytes(config.canonical_file_bytes().replace(b"\n", b" \n"))
    with pytest.raises(PostEmbeddingDevelopmentError, match="caller pin|canonical"):
        load_post_embedding_development_config(path, expected_sha256=config.file_sha256)


@pytest.mark.parametrize("token", ["sealed", "labels", "outcomes", "results", "custody"])
def test_config_rejects_non_development_path_tokens(tmp_path: Path, token: str) -> None:
    config = _config(tmp_path)
    with pytest.raises(PostEmbeddingDevelopmentError, match="forbidden"):
        replace(config, output_root=(tmp_path / token / "operator").resolve())


def test_config_rejects_path_overlap_and_alias(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(PostEmbeddingDevelopmentError, match="overlap"):
        replace(config, output_root=config.full_staged_root / "operator")
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(PostEmbeddingDevelopmentError, match="alias or symbolic"):
        replace(config, output_root=alias / "operator")


def test_receipt_requires_exact_ten_strata_and_cardinality() -> None:
    receipt = _receipt()

    assert receipt.artifact_sha256 == hashlib.sha256(receipt.canonical_file_bytes()).hexdigest()
    with pytest.raises(PostEmbeddingDevelopmentError, match="fixed ten"):
        replace(receipt, strata=receipt.strata[:-1])
    with pytest.raises(PostEmbeddingDevelopmentError, match="cardinality"):
        replace(receipt, paired_action_row_count=16_499)
    with pytest.raises(PostEmbeddingDevelopmentError, match="artifact contract"):
        replace(receipt, artifacts=receipt.artifacts[:-1])


def test_terminal_receipt_round_trip_uses_factory_field_names(tmp_path: Path) -> None:
    receipt = _receipt()
    path = (tmp_path / RECEIPT_FILENAME).resolve()
    path.write_bytes(receipt.canonical_file_bytes())

    observed = load_post_embedding_development_receipt(
        path,
        expected_sha256=receipt.artifact_sha256,
    )

    assert observed == receipt
    assert observed.development_materialization_receipt_sha256 == _digest("materialization")
    assert observed.design_seed_sha256 == _digest("design")
    assert observed.index_config_sha256 == _digest("index-config")
    assert observed.joint_power_report_tree_sha256 == _digest("power-tree")


def test_upstream_admission_rejects_audit_and_embedding_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.full_staged_root.mkdir(mode=0o700)
    (config.full_staged_root / "inventory.json").write_bytes(b"inventory")
    config.partition_audit_path.write_bytes(b"audit")
    embedding_config = SimpleNamespace(
        online_inventory_sha256=config.full_staged_inventory_sha256,
        output_root=(tmp_path / "embeddings").resolve(),
    )
    suite = SimpleNamespace(
        production_config_sha256=config.production_embedding_config_sha256,
        online_inventory_sha256=config.full_staged_inventory_sha256,
    )
    audit = SimpleNamespace(artifact_sha256=_digest("other-audit"))

    def fake_digest(path: Path, *, label: str) -> str:
        del label
        if path == config.partition_audit_path:
            return config.partition_audit_file_sha256
        return config.full_staged_inventory_sha256

    monkeypatch.setattr(operator, "digest_regular_file", fake_digest)
    monkeypatch.setattr(
        operator,
        "load_production_embedding_config",
        lambda *a, **k: embedding_config,
    )
    monkeypatch.setattr(operator, "verify_production_embedding_suite", lambda value: suite)
    monkeypatch.setattr(operator, "load_scalable_partition_audit", lambda *a, **k: audit)
    with pytest.raises(PostEmbeddingDevelopmentError, match="not one cohort"):
        operator._admit_upstream(config)

    audit.artifact_sha256 = config.partition_audit_file_sha256
    suite.production_config_sha256 = _digest("other-config")
    with pytest.raises(PostEmbeddingDevelopmentError, match="not one cohort"):
        operator._admit_upstream(config)


def test_embedding_bindings_reuse_one_corpus_receipt_for_fit_and_calibration(
    tmp_path: Path,
) -> None:
    rows = tuple(
        SimpleNamespace(corpus_id=corpus, embedding_receipt_sha256=_digest(corpus))
        for corpus in operator.FIXED_CORPORA
    )
    upstream = SimpleNamespace(
        embedding_config=SimpleNamespace(output_root=(tmp_path / "embeddings").resolve()),
        embedding_suite=SimpleNamespace(corpora=rows),
    )

    bindings = operator._expected_embedding_bindings(upstream)
    for corpus in operator.FIXED_CORPORA:
        observed = [row for row in bindings if row.corpus == corpus]
        assert len(observed) == 2
        assert len({row.root for row in observed}) == 1
        assert len({row.receipt_sha256 for row in observed}) == 1


def test_policy_admission_rejects_hand_authored_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    materialization = config.output_root / MATERIALIZATION_DIRECTORY
    materialization.mkdir(mode=0o700)
    policy_parent = config.output_root / POLICY_DIRECTORY
    for stage, corpus in operator._stratum_keys():
        (policy_parent / stage / corpus).mkdir(mode=0o700, parents=True)
    bindings = tuple(
        SimpleNamespace(
            development_stage=stage,
            corpus=corpus,
            root=(tmp_path / "embedding" / corpus).resolve(),
            receipt_sha256=_digest(corpus),
        )
        for stage, corpus in operator._stratum_keys()
    )
    monkeypatch.setattr(
        operator,
        "verify_materialized_development_cohort",
        lambda *a, **k: SimpleNamespace(embedding_bindings=bindings),
    )
    monkeypatch.setattr(operator, "load_development_execution_plan", lambda path: object())
    monkeypatch.setattr(
        operator,
        "derive_production_policy_config",
        lambda *a: SimpleNamespace(config_sha256=_digest("derived-policy")),
    )

    def reject_policy(*args: object, **kwargs: object) -> None:
        raise PolicyInterventionError("policy bytes differ from deterministic compilation")

    monkeypatch.setattr(operator, "verify_policy_intervention_package", reject_policy)
    with pytest.raises(PolicyInterventionError, match="deterministic compilation"):
        operator._ensure_policy_and_indexes(
            config,
            _digest("materialization"),
            allow_writes=False,
        )


def test_package_prefix_accepts_only_complete_canonical_boundaries(tmp_path: Path) -> None:
    parent = (tmp_path / "packages").resolve()
    first = operator._stratum_keys()[:3]
    for stage, corpus in first:
        (parent / stage / corpus).mkdir(mode=0o700, parents=True)
    operator._assert_package_prefix(parent, label="packages")

    skipped_parent = (tmp_path / "skipped").resolve()
    stage, corpus = operator._stratum_keys()[1]
    (skipped_parent / stage / corpus).mkdir(mode=0o700, parents=True)
    with pytest.raises(PostEmbeddingDevelopmentError, match="canonical prefix"):
        operator._assert_package_prefix(skipped_parent, label="packages")

    (parent / "unexpected" / "corpus").mkdir(mode=0o700, parents=True)
    with pytest.raises(PostEmbeddingDevelopmentError, match="canonical prefix"):
        operator._assert_package_prefix(parent, label="packages")


def test_known_tree_rejects_extra_symlink_and_special_members(tmp_path: Path) -> None:
    root = (tmp_path / "operator").resolve()
    root.mkdir(mode=0o700)
    (root / OPERATOR_CONFIG_FILENAME).write_bytes(b"{}\n")
    (root / "extra").write_bytes(b"x")
    with pytest.raises(PostEmbeddingDevelopmentError, match="unexpected"):
        operator._assert_known_tree(root)
    (root / "extra").unlink()
    (root / SELECTION_FILENAME).symlink_to(root / OPERATOR_CONFIG_FILENAME)
    with pytest.raises(PostEmbeddingDevelopmentError, match="symlink"):
        operator._assert_known_tree(root)


def test_public_verifier_is_read_only_and_passes_verify_only_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _write_operator_top_level(config)
    before = operator.digest_directory_tree(config.output_root).sha256
    expected = object()
    calls: list[tuple[str, bool]] = []
    execution_receipt = SimpleNamespace(artifact_sha256=_digest("execution"))
    execution_config = SimpleNamespace(config_sha256=_digest("execution-config"))
    report = SimpleNamespace(sha256=_digest("report"), selected_families_per_corpus=75)
    exact_replays: list[bool] = []

    monkeypatch.setattr(operator, "_admit_upstream", lambda value: object())

    def cohort(*args: object, allow_writes: bool, **kwargs: object):
        calls.append(("cohort", allow_writes))
        return _digest("selection"), _digest("bindings"), _digest("materialization")

    def packages(*args: object, allow_writes: bool, **kwargs: object):
        calls.append(("packages", allow_writes))
        return _strata(), _digest("index-config")

    def execution(*args: object, allow_writes: bool, **kwargs: object):
        calls.append(("execution", allow_writes))
        return execution_config, execution_receipt

    def freeze(*args: object, allow_writes: bool, **kwargs: object):
        calls.append(("freeze", allow_writes))
        return _digest("freeze-config"), _digest("freeze-receipt"), _digest("freeze-tree")

    monkeypatch.setattr(operator, "_ensure_selection_and_materialization", cohort)
    monkeypatch.setattr(operator, "_ensure_policy_and_indexes", packages)
    monkeypatch.setattr(operator, "_ensure_execution", execution)
    monkeypatch.setattr(operator, "_ensure_freeze", freeze)

    def verify_power(*args: object, **kwargs: object):
        exact_replays.append(kwargs.get("reproduce_exact", True))
        return (
            SimpleNamespace(sha256=_digest("power-config")),
            (),
            report,
            _digest("tree"),
        )

    monkeypatch.setattr(operator, "_verify_joint_power_bundle", verify_power)
    monkeypatch.setattr(operator, "_build_receipt", lambda *a, **k: expected)
    monkeypatch.setattr(
        operator,
        "load_post_embedding_development_receipt",
        lambda *a, **k: expected,
    )

    assert verify_post_embedding_development(config.output_root) is expected
    assert calls == [
        ("cohort", False),
        ("packages", False),
        ("execution", False),
        ("freeze", False),
    ]
    assert exact_replays == [True]
    assert operator.digest_directory_tree(config.output_root).sha256 == before


def test_public_verifier_detects_changed_package_after_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _write_operator_top_level(config)
    expected = object()
    observed = object()
    execution_receipt = SimpleNamespace(artifact_sha256=_digest("execution"))
    monkeypatch.setattr(operator, "_admit_upstream", lambda value: object())
    monkeypatch.setattr(
        operator,
        "_ensure_selection_and_materialization",
        lambda *a, **k: (_digest("selection"), _digest("bindings"), _digest("materialization")),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_policy_and_indexes",
        lambda *a, **k: (_strata(), _digest("index")),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_execution",
        lambda *a, **k: (SimpleNamespace(), execution_receipt),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_freeze",
        lambda *a, **k: (_digest("a"), _digest("b"), _digest("c")),
    )
    monkeypatch.setattr(
        operator,
        "_verify_joint_power_bundle",
        lambda *a, **k: (SimpleNamespace(), (), SimpleNamespace(), _digest("tree")),
    )
    monkeypatch.setattr(operator, "_build_receipt", lambda *a, **k: expected)
    monkeypatch.setattr(
        operator,
        "load_post_embedding_development_receipt",
        lambda *a, **k: observed,
    )
    with pytest.raises(PostEmbeddingDevelopmentError, match="does not reproduce"):
        verify_post_embedding_development(config.output_root)


def test_joint_power_source_rejects_test_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = (tmp_path / "development-freeze").resolve()
    freeze.mkdir(mode=0o700)
    for name in (
        "joint-power-config.json",
        "joint-power-expected-panel.json",
        "joint-power-conservative-panel.json",
    ):
        (freeze / name).write_bytes(b"x\n")
    power = SimpleNamespace(
        test_mode=True,
        n_simulations=40,
        bound_calibration_simulations=40,
        effect_scenarios=(),
    )
    monkeypatch.setattr(operator, "load_joint_power_config", lambda encoded: power)
    monkeypatch.setattr(
        operator,
        "load_development_panel",
        lambda encoded: SimpleNamespace(scenario_id="x", sha256=_digest("panel")),
    )
    with pytest.raises(PostEmbeddingDevelopmentError, match="production mode"):
        operator._joint_power_source(freeze)


def test_joint_bundle_rejects_nonfreeze_report_and_recomputation_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "operator").resolve()
    freeze = root / FREEZE_DIRECTORY
    bundle = root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY
    panel_dir = bundle / "panels"
    panel_dir.mkdir(mode=0o700, parents=True)
    freeze.mkdir(mode=0o700)
    panel = SimpleNamespace(scenario_id="expected", sha256=_digest("panel"), encoded=b"panel\n")
    power = SimpleNamespace(sha256=_digest("power"), encoded=b"config\n")
    report = SimpleNamespace(
        config_sha256=power.sha256,
        panel_sha256s=((panel.scenario_id, panel.sha256),),
        test_mode=False,
        freeze_ready=False,
        selected_families_per_corpus=None,
        encoded=b"report\n",
    )
    selection_audit = SimpleNamespace(encoded=b"audit\n")
    (bundle / "config.json").write_bytes(power.encoded)
    (bundle / "report.json").write_bytes(report.encoded)
    (bundle / JOINT_POWER_SELECTION_AUDIT_FILENAME).write_bytes(selection_audit.encoded)
    (panel_dir / f"{panel.sha256}.json").write_bytes(panel.encoded)
    invocation = root / JOINT_POWER_INVOCATION_FILENAME
    freeze_tree_sha = _digest("freeze-tree")
    invocation.write_bytes(operator._invocation_payload(freeze_tree_sha, power, (panel,)))
    monkeypatch.setattr(operator, "_joint_power_source", lambda value: (power, (panel,)))
    monkeypatch.setattr(operator, "canonical_joint_power_config_bytes", lambda value: value.encoded)
    monkeypatch.setattr(operator, "canonical_development_panel_bytes", lambda value: value.encoded)
    monkeypatch.setattr(operator, "load_development_panel", lambda encoded: panel)
    monkeypatch.setattr(operator, "load_joint_power_report", lambda encoded: report)
    monkeypatch.setattr(
        operator,
        "load_joint_power_selection_audit",
        lambda encoded: selection_audit,
    )
    with pytest.raises(PostEmbeddingDevelopmentError, match="not freeze-ready"):
        operator._verify_joint_power_bundle(
            bundle,
            freeze_tree_sha256=freeze_tree_sha,
            invocation_path=invocation,
        )

    report.freeze_ready = True
    report.selected_families_per_corpus = 75
    monkeypatch.setattr(
        operator,
        "run_joint_power_design",
        lambda *a, **kw: SimpleNamespace(encoded=b"x"),
    )
    monkeypatch.setattr(
        operator,
        "verify_joint_power_selection_audit",
        lambda *a: selection_audit,
    )
    monkeypatch.setattr(operator, "canonical_joint_power_report_bytes", lambda value: value.encoded)
    with pytest.raises(PostEmbeddingDevelopmentError, match="does not reproduce"):
        operator._verify_joint_power_bundle(
            bundle,
            freeze_tree_sha256=freeze_tree_sha,
            invocation_path=invocation,
        )


def test_exclusive_publication_rejects_preexisting_target(tmp_path: Path) -> None:
    work = (tmp_path / "work").resolve()
    target = (tmp_path / "target").resolve()
    work.mkdir(mode=0o700)
    target.mkdir(mode=0o700)
    with pytest.raises(PostEmbeddingDevelopmentError, match="already exists"):
        operator._exclusive_publish_directory(work, target)
    assert work.is_dir()
    assert target.is_dir()


def test_run_rejects_existing_root_and_resume_uses_write_enabled_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    (config.output_root / OPERATOR_CONFIG_FILENAME).write_bytes(config.canonical_file_bytes())
    with pytest.raises(PostEmbeddingDevelopmentError, match="already exists"):
        operator.run_post_embedding_development(config)

    calls: list[tuple[str, bool]] = []
    final_receipt = _receipt()
    execution_receipt = SimpleNamespace(artifact_sha256=_digest("execution"))
    execution_config = SimpleNamespace(config_sha256=_digest("execution-config"))
    power = SimpleNamespace(sha256=_digest("power"))
    report = SimpleNamespace(sha256=_digest("report"), selected_families_per_corpus=75)
    exact_replays: list[bool] = []
    monkeypatch.setattr(operator, "_admit_upstream", lambda value: object())

    def stage(name: str, result: object):
        def execute(*args: object, allow_writes: bool, **kwargs: object):
            calls.append((name, allow_writes))
            return result

        return execute

    monkeypatch.setattr(
        operator,
        "_ensure_selection_and_materialization",
        stage("cohort", (_digest("selection"), _digest("bindings"), _digest("materialization"))),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_policy_and_indexes",
        stage("packages", (_strata(), _digest("index"))),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_execution",
        stage("execution", (execution_config, execution_receipt)),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_freeze",
        stage("freeze", (_digest("a"), _digest("b"), _digest("c"))),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_joint_power",
        lambda *a: (power, (), report, _digest("t")),
    )
    (config.output_root / JOINT_POWER_INVOCATION_FILENAME).write_bytes(b"invocation\n")

    def verify_power(*args: object, **kwargs: object):
        exact_replays.append(kwargs.get("reproduce_exact", True))
        return power, (), report, _digest("t")

    monkeypatch.setattr(operator, "_verify_joint_power_bundle", verify_power)
    monkeypatch.setattr(operator, "_build_receipt", lambda *a, **k: final_receipt)
    monkeypatch.setattr(
        operator,
        "_verify_post_embedding_development_config",
        lambda *a, **k: final_receipt,
    )
    assert operator.resume_post_embedding_development(config) is final_receipt
    assert calls == [
        ("cohort", True),
        ("packages", True),
        ("execution", True),
        ("freeze", True),
    ]
    assert exact_replays == [True]
    assert (config.output_root / RECEIPT_FILENAME).read_bytes() == (
        final_receipt.canonical_file_bytes()
    )


def test_new_run_executes_one_generation_and_defers_exact_replay_to_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    final_receipt = _receipt()
    exact_calls: list[str] = []
    power = SimpleNamespace(sha256=_digest("power"))
    report = SimpleNamespace(sha256=_digest("report"), selected_families_per_corpus=75)
    execution_config = SimpleNamespace(config_sha256=_digest("execution-config"))
    execution_receipt = SimpleNamespace(artifact_sha256=_digest("execution"))
    monkeypatch.setattr(operator, "_admit_upstream", lambda value: object())
    monkeypatch.setattr(
        operator,
        "_ensure_selection_and_materialization",
        lambda *a, **k: (_digest("selection"), _digest("bindings"), _digest("materialization")),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_policy_and_indexes",
        lambda *a, **k: ((), _digest("index")),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_execution",
        lambda *a, **k: (execution_config, execution_receipt),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_freeze",
        lambda *a, **k: (_digest("freeze-config"), _digest("freeze-receipt"), _digest("freeze")),
    )

    def generate(*args: object):
        exact_calls.append("generation")
        bundle = config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY
        bundle.mkdir(mode=0o700, parents=True)
        (bundle / "placeholder").write_bytes(b"published\n")
        (config.output_root / JOINT_POWER_INVOCATION_FILENAME).write_bytes(b"invocation\n")
        return power, (), report, operator.digest_directory_tree(bundle).sha256

    def unexpected_replay(*args: object, **kwargs: object):
        raise AssertionError("fresh operator execution must defer exact replay to freeze")

    def terminal(*args: object, **kwargs: object):
        assert isinstance(
            kwargs.get("fresh_joint_power"),
            operator._FreshJointPowerVerification,
        )
        return final_receipt

    monkeypatch.setattr(operator, "_ensure_joint_power", generate)
    monkeypatch.setattr(operator, "_verify_joint_power_bundle", unexpected_replay)
    monkeypatch.setattr(operator, "_build_receipt", lambda *a, **k: final_receipt)
    monkeypatch.setattr(operator, "_verify_post_embedding_development_config", terminal)

    assert operator.run_post_embedding_development(config) is final_receipt
    assert exact_calls == ["generation"]


def test_fresh_verification_token_rejects_tree_or_invocation_mutation(
    tmp_path: Path,
) -> None:
    bundle = (tmp_path / "joint-power-design").resolve()
    bundle.mkdir(mode=0o700)
    artifact = bundle / "artifact.json"
    artifact.write_bytes(b"first\n")
    power = SimpleNamespace(sha256=_digest("power"))
    panel = SimpleNamespace(scenario_id="expected", sha256=_digest("panel"))
    report = SimpleNamespace(sha256=_digest("report"))
    freeze_tree = _digest("freeze")
    invocation = (tmp_path / "joint-power-invocation.json").resolve()
    invocation_bytes = operator._invocation_payload(freeze_tree, power, (panel,))
    invocation.write_bytes(invocation_bytes)
    verification = operator._FreshJointPowerVerification(
        bundle_root=bundle,
        freeze_tree_sha256=freeze_tree,
        invocation_bytes=invocation_bytes,
        power_config=power,
        panels=(panel,),
        report=report,
        tree_sha256=operator.digest_directory_tree(bundle).sha256,
    )
    assert operator._reuse_fresh_joint_power_verification(
        verification,
        bundle=bundle,
        freeze_tree_sha256=freeze_tree,
        invocation_path=invocation,
    ) == (power, (panel,), report, verification.tree_sha256)

    artifact.write_bytes(b"second\n")
    with pytest.raises(PostEmbeddingDevelopmentError, match="bundle changed"):
        operator._reuse_fresh_joint_power_verification(
            verification,
            bundle=bundle,
            freeze_tree_sha256=freeze_tree,
            invocation_path=invocation,
        )

    artifact.write_bytes(b"first\n")
    invocation.write_bytes(b"mutated\n")
    with pytest.raises(PostEmbeddingDevelopmentError, match="invocation changed"):
        operator._reuse_fresh_joint_power_verification(
            verification,
            bundle=bundle,
            freeze_tree_sha256=freeze_tree,
            invocation_path=invocation,
        )


def test_status_marks_interrupted_single_invocation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert post_embedding_development_status(config)["output_exists"] is False
    config.output_root.mkdir(mode=0o700)
    (config.output_root / OPERATOR_CONFIG_FILENAME).write_bytes(config.canonical_file_bytes())
    (config.output_root / JOINT_POWER_INVOCATION_FILENAME).write_bytes(b"{}\n")

    status = post_embedding_development_status(config)

    assert status["joint_power_interrupted"] is True
    assert status["completed"] is False


def test_single_invocation_marker_blocks_retry_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    marker = config.output_root / JOINT_POWER_INVOCATION_FILENAME
    marker.write_bytes(b"{}\n")
    monkeypatch.setattr(
        operator,
        "_joint_power_source",
        lambda value: (SimpleNamespace(), ()),
    )
    with pytest.raises(PostEmbeddingDevelopmentError, match="retry is forbidden"):
        operator._ensure_joint_power(config, _digest("freeze-tree"))


def test_existing_joint_bundle_defers_exact_replay_to_terminal_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    bundle = config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY
    bundle.mkdir(mode=0o700, parents=True)
    invocation = config.output_root / JOINT_POWER_INVOCATION_FILENAME
    invocation.write_bytes(b"invocation\n")
    power = SimpleNamespace()
    observed: list[bool] = []
    monkeypatch.setattr(operator, "_joint_power_source", lambda value: (power, ()))

    def inspect(*args: object, **kwargs: object):
        observed.append(kwargs["reproduce_exact"])
        return power, (), SimpleNamespace(), _digest("tree")

    monkeypatch.setattr(operator, "_verify_joint_power_bundle", inspect)
    operator._ensure_joint_power(config, _digest("freeze-tree"))
    assert observed == [False]


def test_changed_stage_file_changes_its_bound_tree_digest(tmp_path: Path) -> None:
    root = (tmp_path / "operator").resolve()
    stage = root / MATERIALIZATION_DIRECTORY
    stage.mkdir(mode=0o700, parents=True)
    payload = stage / "receipt.json"
    payload.write_bytes(b"first\n")
    first = operator._pin_artifact(root, MATERIALIZATION_DIRECTORY)
    payload.write_bytes(b"second\n")
    second = operator._pin_artifact(root, MATERIALIZATION_DIRECTORY)

    assert first.sha256 != second.sha256


def test_cli_help_exposes_all_operator_boundaries(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        operator._parser().parse_args(["--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "write-config" in output
    assert "run" in output
    assert "resume" in output
    assert "verify" in output
    assert "status" in output
