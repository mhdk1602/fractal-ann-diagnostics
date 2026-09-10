from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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


def _ensured_joint_power(
    config: PostEmbeddingDevelopmentConfig,
    *,
    freeze_tree_sha256: str,
    power: object,
    panels: tuple[object, ...],
    report: object,
    tree_sha256: str,
    exact_replay_performed: bool = False,
) -> operator._EnsuredJointPower:
    invocation = config.output_root / JOINT_POWER_INVOCATION_FILENAME
    return operator._EnsuredJointPower(
        verification=operator._FreshJointPowerVerification(
            bundle_root=config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY,
            freeze_tree_sha256=freeze_tree_sha256,
            invocation_bytes=invocation.read_bytes(),
            power_config=power,
            panels=panels,
            report=report,
            tree_sha256=tree_sha256,
        ),
        exact_replay_performed=exact_replay_performed,
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
    monkeypatch.setattr(
        operator,
        "admit_frozen_production_embedding_suite",
        lambda value: suite,
    )
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


def test_frozen_verifier_replays_package_without_full_staged_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _write_operator_top_level(config)
    embedding_config = Mock(spec=operator.ProductionEmbeddingConfig)
    embedding_config.file_sha256 = config.production_embedding_config_sha256
    embedding_config.online_inventory_sha256 = config.full_staged_inventory_sha256
    embedding_suite = Mock(spec=operator.ProductionEmbeddingSuiteReceipt)
    embedding_suite.production_config_sha256 = config.production_embedding_config_sha256
    embedding_suite.online_inventory_sha256 = config.full_staged_inventory_sha256
    partition_audit = Mock(spec=operator.ScalableQueryPartitionAuditReceipt)
    partition_audit.artifact_sha256 = config.partition_audit_file_sha256
    partition_audit.staged_inventory_sha256 = config.full_staged_inventory_sha256
    terminal = object()
    observed: list[operator._AdmittedUpstream] = []
    execution_receipt = SimpleNamespace(artifact_sha256=_digest("execution"))
    execution_config = SimpleNamespace(config_sha256=_digest("execution-config"))
    report = SimpleNamespace(sha256=_digest("report"), selected_families_per_corpus=75)

    monkeypatch.setattr(operator, "_require_read_only_filesystem", lambda *_a, **_k: None)
    monkeypatch.setattr(
        operator,
        "_admit_upstream",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("frozen admission reopened the full staged tree")
        ),
    )

    def cohort(
        value: PostEmbeddingDevelopmentConfig,
        admitted_upstream: operator._AdmittedUpstream,
        *,
        allow_writes: bool,
    ) -> tuple[str, str, str]:
        assert value == config
        assert allow_writes is False
        observed.append(admitted_upstream)
        return _digest("selection"), _digest("bindings"), _digest("materialization")

    monkeypatch.setattr(operator, "_ensure_selection_and_materialization", cohort)
    monkeypatch.setattr(
        operator,
        "_ensure_policy_and_indexes",
        lambda *_a, **_k: (_strata(), _digest("index")),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_execution",
        lambda *_a, **_k: (execution_config, execution_receipt),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_freeze",
        lambda *_a, **_k: (_digest("a"), _digest("b"), _digest("c")),
    )
    monkeypatch.setattr(
        operator,
        "_verify_joint_power_bundle",
        lambda *_a, **_k: (
            SimpleNamespace(sha256=_digest("power-config")),
            (),
            report,
            _digest("power-tree"),
        ),
    )
    monkeypatch.setattr(operator, "_build_receipt", lambda *_a, **_k: terminal)
    monkeypatch.setattr(
        operator,
        "load_post_embedding_development_receipt",
        lambda *_a, **_k: terminal,
    )

    assert (
        operator.admit_frozen_post_embedding_development(
            config.output_root,
            expected_receipt_sha256=_digest("terminal"),
            production_embedding_config_path=config.production_embedding_config_path,
            embedding_config=embedding_config,
            embedding_suite=embedding_suite,
            partition_audit_path=config.partition_audit_path,
            partition_audit=partition_audit,
        )
        is terminal
    )
    assert observed == [
        operator._AdmittedUpstream(
            embedding_config=embedding_config,
            embedding_suite=embedding_suite,
            partition_audit=partition_audit,
        )
    ]
    assert not config.full_staged_root.exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "embedding-path",
        "embedding-config-sha256",
        "inventory-sha256",
        "audit-path",
        "audit-file-sha256",
        "suite-config-sha256",
        "suite-inventory-sha256",
        "audit-inventory-sha256",
    ),
)
def test_frozen_verifier_rejects_every_upstream_join_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config = _config(tmp_path)
    _write_operator_top_level(config)
    embedding_config = Mock(spec=operator.ProductionEmbeddingConfig)
    embedding_config.file_sha256 = config.production_embedding_config_sha256
    embedding_config.online_inventory_sha256 = config.full_staged_inventory_sha256
    embedding_suite = Mock(spec=operator.ProductionEmbeddingSuiteReceipt)
    embedding_suite.production_config_sha256 = config.production_embedding_config_sha256
    embedding_suite.online_inventory_sha256 = config.full_staged_inventory_sha256
    partition_audit = Mock(spec=operator.ScalableQueryPartitionAuditReceipt)
    partition_audit.artifact_sha256 = config.partition_audit_file_sha256
    partition_audit.staged_inventory_sha256 = config.full_staged_inventory_sha256
    embedded_config = config
    embedding_path = config.production_embedding_config_path
    audit_path = config.partition_audit_path
    wrong = _digest(f"wrong:{mutation}")
    if mutation == "embedding-path":
        embedded_config = replace(
            config,
            production_embedding_config_path=(tmp_path / "alternate-embedding.json").resolve(),
        )
    elif mutation == "embedding-config-sha256":
        embedded_config = replace(config, production_embedding_config_sha256=wrong)
    elif mutation == "inventory-sha256":
        embedded_config = replace(config, full_staged_inventory_sha256=wrong)
    elif mutation == "audit-path":
        embedded_config = replace(
            config,
            partition_audit_path=(tmp_path / "alternate-audit.json").resolve(),
        )
    elif mutation == "audit-file-sha256":
        embedded_config = replace(config, partition_audit_file_sha256=wrong)
    elif mutation == "suite-config-sha256":
        embedding_suite.production_config_sha256 = wrong
    elif mutation == "suite-inventory-sha256":
        embedding_suite.online_inventory_sha256 = wrong
    else:
        partition_audit.staged_inventory_sha256 = wrong
    (config.output_root / OPERATOR_CONFIG_FILENAME).write_bytes(
        embedded_config.canonical_file_bytes()
    )
    verifier_calls: list[object] = []
    monkeypatch.setattr(operator, "_require_read_only_filesystem", lambda *_a, **_k: None)
    monkeypatch.setattr(
        operator,
        "_verify_post_embedding_development_config",
        lambda *_a, **_k: verifier_calls.append(object()),
    )

    with pytest.raises(PostEmbeddingDevelopmentError, match="upstream differs"):
        operator.admit_frozen_post_embedding_development(
            config.output_root,
            expected_receipt_sha256=_digest("terminal"),
            production_embedding_config_path=embedding_path,
            embedding_config=embedding_config,
            embedding_suite=embedding_suite,
            partition_audit_path=audit_path,
            partition_audit=partition_audit,
        )
    assert verifier_calls == []


def test_frozen_verifier_rejects_writable_package_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    reads: list[Path] = []
    monkeypatch.setattr(
        operator,
        "_require_read_only_filesystem",
        lambda *_a, **_k: (_ for _ in ()).throw(
            PostEmbeddingDevelopmentError("frozen operator root must be mounted read-only")
        ),
    )
    monkeypatch.setattr(
        operator,
        "_read_control",
        lambda path, **_kwargs: reads.append(path) or b"",
    )

    with pytest.raises(PostEmbeddingDevelopmentError, match="mounted read-only"):
        operator.admit_frozen_post_embedding_development(
            config.output_root,
            expected_receipt_sha256=_digest("terminal"),
            production_embedding_config_path=config.production_embedding_config_path,
            embedding_config=Mock(spec=operator.ProductionEmbeddingConfig),
            embedding_suite=Mock(spec=operator.ProductionEmbeddingSuiteReceipt),
            partition_audit_path=config.partition_audit_path,
            partition_audit=Mock(spec=operator.ScalableQueryPartitionAuditReceipt),
        )
    assert reads == []


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
        lambda *a: _ensured_joint_power(
            config,
            freeze_tree_sha256=_digest("c"),
            power=power,
            panels=(),
            report=report,
            tree_sha256=_digest("t"),
            exact_replay_performed=True,
        ),
    )
    (config.output_root / JOINT_POWER_INVOCATION_FILENAME).write_bytes(b"invocation\n")

    def unexpected_replay(*args: object, **kwargs: object):
        raise AssertionError("resume must consume the exact verification returned by ensure")

    monkeypatch.setattr(operator, "_verify_joint_power_bundle", unexpected_replay)
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
    assert (config.output_root / RECEIPT_FILENAME).read_bytes() == (
        final_receipt.canonical_file_bytes()
    )


def test_resume_removes_validated_interrupted_freeze_staging_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    (config.output_root / OPERATOR_CONFIG_FILENAME).write_bytes(config.canonical_file_bytes())
    interrupted = config.output_root / f".{FREEZE_DIRECTORY}.staging-interrupted"
    interrupted.mkdir(mode=0o700)
    (interrupted / "partial.json").write_bytes(b"partial\n")
    final_receipt = _receipt()
    execution_receipt = SimpleNamespace(artifact_sha256=_digest("execution"))
    execution_config = SimpleNamespace(config_sha256=_digest("execution-config"))
    power = SimpleNamespace(sha256=_digest("power"))
    report = SimpleNamespace(sha256=_digest("report"), selected_families_per_corpus=75)
    monkeypatch.setattr(operator, "_admit_upstream", lambda value: object())
    monkeypatch.setattr(
        operator,
        "_ensure_selection_and_materialization",
        lambda *args, **kwargs: (
            _digest("selection"),
            _digest("bindings"),
            _digest("materialization"),
        ),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_policy_and_indexes",
        lambda *args, **kwargs: ((), _digest("index")),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_execution",
        lambda *args, **kwargs: (execution_config, execution_receipt),
    )
    monkeypatch.setattr(
        operator,
        "_ensure_freeze",
        lambda *args, **kwargs: (
            _digest("freeze-config"),
            _digest("freeze-receipt"),
            _digest("freeze"),
        ),
    )

    def ensure_power(*args: object):
        assert not interrupted.exists()
        bundle = config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY
        bundle.mkdir(mode=0o700, parents=True)
        (bundle / "placeholder").write_bytes(b"published\n")
        (config.output_root / JOINT_POWER_INVOCATION_FILENAME).write_bytes(b"invocation\n")
        return _ensured_joint_power(
            config,
            freeze_tree_sha256=_digest("freeze"),
            power=power,
            panels=(),
            report=report,
            tree_sha256=operator.digest_directory_tree(bundle).sha256,
        )

    monkeypatch.setattr(operator, "_ensure_joint_power", ensure_power)
    monkeypatch.setattr(operator, "_build_receipt", lambda *args, **kwargs: final_receipt)
    monkeypatch.setattr(
        operator,
        "_verify_post_embedding_development_config",
        lambda *args, **kwargs: final_receipt,
    )

    assert operator.resume_post_embedding_development(config) is final_receipt
    assert not interrupted.exists()


def test_operator_lock_rejects_a_concurrent_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    (config.output_root / OPERATOR_CONFIG_FILENAME).write_bytes(config.canonical_file_bytes())
    descriptor = operator._acquire_operator_lock(config.output_root)
    try:
        with pytest.raises(
            PostEmbeddingDevelopmentError,
            match="another post-embedding development process is active",
        ):
            operator.resume_post_embedding_development(config)
    finally:
        operator._release_operator_lock(descriptor)


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
        return _ensured_joint_power(
            config,
            freeze_tree_sha256=_digest("freeze"),
            power=power,
            panels=(),
            report=report,
            tree_sha256=operator.digest_directory_tree(bundle).sha256,
        )

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


def test_status_admits_but_does_not_delete_interrupted_freeze_staging(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    (config.output_root / OPERATOR_CONFIG_FILENAME).write_bytes(config.canonical_file_bytes())
    interrupted = config.output_root / f".{FREEZE_DIRECTORY}.staging-interrupted"
    interrupted.mkdir(mode=0o700)
    (interrupted / "partial.json").write_bytes(b"partial\n")

    status = post_embedding_development_status(config)

    assert status["completed"] is False
    assert interrupted.name in status["present"]
    assert interrupted.is_dir()


def _mock_resumable_joint_power(
    config: PostEmbeddingDevelopmentConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    feasible: bool = True,
):
    power = SimpleNamespace(sha256=_digest("power"), encoded=b"power\n")
    panel = SimpleNamespace(
        scenario_id="expected",
        sha256=_digest("panel"),
        encoded=b"panel\n",
    )
    selected = 75 if feasible else None
    audit = SimpleNamespace(
        config_sha256=power.sha256,
        panel_sha256s=((panel.scenario_id, panel.sha256),),
        selection_basis_sha256=_digest("selection-basis"),
        exact_bootstrap_replicates=10_000,
        coverage_rule="closed-certificate",
        selected_families_per_corpus=selected,
        selection_satisfied=feasible,
        test_mode=False,
        sha256=_digest("selection-audit"),
        encoded=b"audit\n",
    )
    report = SimpleNamespace(
        config_sha256=power.sha256,
        panel_sha256s=((panel.scenario_id, panel.sha256),),
        selection_audit_sha256=audit.sha256,
        selection_audit_basis_sha256=audit.selection_basis_sha256,
        selection_audit_exact_bootstrap_replicates=audit.exact_bootstrap_replicates,
        selection_audit_coverage_rule=audit.coverage_rule,
        test_mode=False,
        freeze_ready=feasible,
        selected_families_per_corpus=selected,
        selection_satisfied=feasible,
        encoded=b"report\n",
    )
    calls = {"audit": 0, "design": 0, "exact": 0}
    monkeypatch.setattr(operator, "_joint_power_source", lambda value: (power, (panel,)))

    def run_audit(*args: object):
        calls["audit"] += 1
        return audit

    def run_design(*args: object, **kwargs: object):
        calls["design"] += 1
        return report

    def load_audit(encoded: bytes):
        if encoded != audit.encoded:
            raise operator.JointPowerDesignError("invalid audit bytes")
        return audit

    def load_report(encoded: bytes):
        if encoded != report.encoded:
            raise operator.JointPowerDesignError("invalid report bytes")
        return report

    def verify(*args: object, **kwargs: object):
        calls["exact"] += 1
        return power, (panel,), report, operator.digest_directory_tree(args[0]).sha256

    monkeypatch.setattr(operator, "run_joint_power_selection_audit", run_audit)
    monkeypatch.setattr(operator, "run_joint_power_design", run_design)
    monkeypatch.setattr(operator, "load_joint_power_selection_audit", load_audit)
    monkeypatch.setattr(operator, "load_joint_power_report", load_report)
    monkeypatch.setattr(
        operator,
        "canonical_joint_power_config_bytes",
        lambda value: value.encoded,
    )
    monkeypatch.setattr(
        operator,
        "canonical_development_panel_bytes",
        lambda value: value.encoded,
    )
    monkeypatch.setattr(
        operator,
        "canonical_joint_power_selection_audit_bytes",
        lambda value: value.encoded,
    )
    monkeypatch.setattr(
        operator,
        "canonical_joint_power_report_bytes",
        lambda value: value.encoded,
    )
    monkeypatch.setattr(
        operator,
        "_verify_joint_power_bundle",
        verify,
    )
    return power, panel, audit, report, calls


def _seed_joint_power_pending_boundary(
    pending: Path,
    *,
    boundary: str,
    power: object,
    panel: object,
    audit: object,
    report: object,
) -> None:
    pending.mkdir(mode=0o700, parents=True)
    audit_path = pending / JOINT_POWER_SELECTION_AUDIT_FILENAME
    report_path = pending / "report.json"
    config_path = pending / "config.json"
    panel_path = pending / "panels" / f"{panel.sha256}.json"
    if boundary == "empty-pending":
        return
    if boundary == "audit-next":
        operator._checkpoint_next_path(audit_path).write_bytes(audit.encoded)
        return
    audit_path.write_bytes(audit.encoded)
    if boundary == "audit-final":
        return
    if boundary == "report-next":
        operator._checkpoint_next_path(report_path).write_bytes(report.encoded)
        return
    report_path.write_bytes(report.encoded)
    if boundary == "report-final":
        return
    if boundary == "config-next":
        operator._checkpoint_next_path(config_path).write_bytes(power.encoded)
        return
    config_path.write_bytes(power.encoded)
    if boundary == "config-final":
        return
    panel_path.parent.mkdir(mode=0o700)
    if boundary == "panels-directory":
        return
    if boundary == "panel-next":
        operator._checkpoint_next_path(panel_path).write_bytes(panel.encoded)
        return
    if boundary == "complete-pending":
        panel_path.write_bytes(panel.encoded)
        return
    raise AssertionError(f"unrecognized boundary: {boundary}")


def test_single_invocation_marker_resumes_the_same_deterministic_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, _audit, report, calls = _mock_resumable_joint_power(config, monkeypatch)
    freeze_tree = _digest("freeze-tree")
    marker = config.output_root / JOINT_POWER_INVOCATION_FILENAME
    marker_bytes = operator._invocation_payload(freeze_tree, power, (panel,))
    marker.write_bytes(marker_bytes)

    observed = operator._ensure_joint_power(config, freeze_tree)

    assert observed.verification.power_config is power
    assert observed.verification.panels == (panel,)
    assert observed.verification.report is report
    assert observed.exact_replay_performed is False
    assert calls == {"audit": 1, "design": 1, "exact": 0}
    assert marker.read_bytes() == marker_bytes
    assert (
        config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY / "report.json"
    ).read_bytes() == report.encoded


@pytest.mark.parametrize(
    "boundary",
    (
        "empty-pending",
        "audit-next",
        "audit-final",
        "report-next",
        "report-final",
        "config-next",
        "config-final",
        "panels-directory",
        "panel-next",
    ),
)
def test_joint_power_resume_recovers_every_checkpoint_kill_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, audit, report, calls = _mock_resumable_joint_power(config, monkeypatch)
    freeze_tree = _digest("freeze-tree")
    marker = config.output_root / JOINT_POWER_INVOCATION_FILENAME
    marker.write_bytes(operator._invocation_payload(freeze_tree, power, (panel,)))
    pending = config.output_root / ANALYSIS_DIRECTORY / operator.JOINT_POWER_PENDING_DIRECTORY
    _seed_joint_power_pending_boundary(
        pending,
        boundary=boundary,
        power=power,
        panel=panel,
        audit=audit,
        report=report,
    )

    ensured = operator._ensure_joint_power(config, freeze_tree)

    bundle = config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY
    assert bundle.is_dir()
    assert not pending.exists()
    assert set(operator.digest_directory_tree(bundle).entries) == {
        "config.json",
        "report.json",
        JOINT_POWER_SELECTION_AUDIT_FILENAME,
        "panels",
        f"panels/{panel.sha256}.json",
    }
    generated_audit = boundary == "empty-pending"
    generated_report = boundary in {"empty-pending", "audit-next", "audit-final"}
    assert calls == {
        "audit": int(generated_audit),
        "design": int(generated_report),
        "exact": int(not generated_audit),
    }
    assert ensured.exact_replay_performed is not generated_audit


def test_joint_power_resume_promotes_a_complete_invocation_next_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, _audit, _report, calls = _mock_resumable_joint_power(
        config,
        monkeypatch,
    )
    freeze_tree = _digest("freeze-tree")
    invocation_bytes = operator._invocation_payload(freeze_tree, power, (panel,))
    analysis = config.output_root / ANALYSIS_DIRECTORY
    analysis.mkdir(mode=0o700)
    invocation_next = analysis / operator._JOINT_POWER_INVOCATION_NEXT_FILENAME
    invocation_next.write_bytes(invocation_bytes)

    ensured = operator._ensure_joint_power(config, freeze_tree)

    assert (config.output_root / JOINT_POWER_INVOCATION_FILENAME).read_bytes() == invocation_bytes
    assert not invocation_next.exists()
    assert calls == {"audit": 1, "design": 1, "exact": 0}
    assert ensured.exact_replay_performed is False


@pytest.mark.parametrize(
    "boundary",
    ("invocation-next", "audit-next", "report-next", "config-next", "panel-next"),
)
def test_joint_power_resume_discards_only_torn_uncommitted_next_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, audit, report, calls = _mock_resumable_joint_power(config, monkeypatch)
    freeze_tree = _digest("freeze-tree")
    invocation_bytes = operator._invocation_payload(freeze_tree, power, (panel,))
    analysis = config.output_root / ANALYSIS_DIRECTORY
    analysis.mkdir(mode=0o700)
    marker = config.output_root / JOINT_POWER_INVOCATION_FILENAME
    if boundary == "invocation-next":
        (analysis / operator._JOINT_POWER_INVOCATION_NEXT_FILENAME).write_bytes(b"torn")
    else:
        marker.write_bytes(invocation_bytes)
        prior = {
            "audit-next": "empty-pending",
            "report-next": "audit-final",
            "config-next": "report-final",
            "panel-next": "panels-directory",
        }[boundary]
        pending = analysis / operator.JOINT_POWER_PENDING_DIRECTORY
        _seed_joint_power_pending_boundary(
            pending,
            boundary=prior,
            power=power,
            panel=panel,
            audit=audit,
            report=report,
        )
        target = {
            "audit-next": pending / JOINT_POWER_SELECTION_AUDIT_FILENAME,
            "report-next": pending / "report.json",
            "config-next": pending / "config.json",
            "panel-next": pending / "panels" / f"{panel.sha256}.json",
        }[boundary]
        operator._checkpoint_next_path(target).write_bytes(b"torn")

    ensured = operator._ensure_joint_power(config, freeze_tree)

    bundle = config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY
    assert bundle.is_dir()
    assert marker.read_bytes() == invocation_bytes
    audit_was_regenerated = boundary in {"invocation-next", "audit-next"}
    report_was_regenerated = boundary in {
        "invocation-next",
        "audit-next",
        "report-next",
    }
    assert calls == {
        "audit": int(audit_was_regenerated),
        "design": int(report_was_regenerated),
        "exact": int(not audit_was_regenerated),
    }
    assert ensured.exact_replay_performed is not audit_was_regenerated


def test_no_feasible_joint_power_is_checkpointed_and_resumes_without_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, _audit, _report, calls = _mock_resumable_joint_power(
        config,
        monkeypatch,
        feasible=False,
    )
    freeze_tree = _digest("freeze-tree")
    invocation_bytes = operator._invocation_payload(freeze_tree, power, (panel,))
    (config.output_root / JOINT_POWER_INVOCATION_FILENAME).write_bytes(invocation_bytes)

    for _ in range(2):
        with pytest.raises(
            PostEmbeddingDevelopmentError,
            match="no feasible registered candidate.*sealed execution remains unauthorized",
        ):
            operator._ensure_joint_power(config, freeze_tree)

    pending = config.output_root / ANALYSIS_DIRECTORY / operator.JOINT_POWER_PENDING_DIRECTORY
    checkpoint = pending / operator.JOINT_POWER_NO_FEASIBLE_FILENAME
    decoded = operator._decode(checkpoint.read_bytes(), label="no-feasible checkpoint")
    assert decoded == {
        "disposition": "no-feasible-registered-candidate",
        "freeze_tree_sha256": freeze_tree,
        "joint_power_config_sha256": power.sha256,
        "joint_power_report_sha256": _digest("report\n"),
        "panel_sha256s": {panel.scenario_id: panel.sha256},
        "sealed_execution_authorized": False,
        "schema_version": operator.JOINT_POWER_NO_FEASIBLE_SCHEMA,
        "selection_audit_sha256": _digest("audit\n"),
        "selected_families_per_corpus": None,
        "single_invocation_sha256": hashlib.sha256(invocation_bytes).hexdigest(),
    }
    assert calls == {"audit": 1, "design": 1, "exact": 0}
    assert set(operator.digest_directory_tree(pending).entries) == {
        JOINT_POWER_SELECTION_AUDIT_FILENAME,
        "report.json",
        operator.JOINT_POWER_NO_FEASIBLE_FILENAME,
    }
    assert not (config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY).exists()
    assert not (config.output_root / RECEIPT_FILENAME).exists()


@pytest.mark.parametrize("encoded", (b"complete", b"torn"))
def test_no_feasible_terminal_checkpoint_recovers_its_atomic_next_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoded: bytes,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, audit, report, calls = _mock_resumable_joint_power(
        config,
        monkeypatch,
        feasible=False,
    )
    freeze_tree = _digest("freeze-tree")
    invocation_bytes = operator._invocation_payload(freeze_tree, power, (panel,))
    (config.output_root / JOINT_POWER_INVOCATION_FILENAME).write_bytes(invocation_bytes)
    pending = config.output_root / ANALYSIS_DIRECTORY / operator.JOINT_POWER_PENDING_DIRECTORY
    _seed_joint_power_pending_boundary(
        pending,
        boundary="report-final",
        power=power,
        panel=panel,
        audit=audit,
        report=report,
    )
    terminal = pending / operator.JOINT_POWER_NO_FEASIBLE_FILENAME
    expected = operator._joint_power_no_feasible_payload(
        freeze_tree_sha256=freeze_tree,
        invocation_bytes=invocation_bytes,
        power_config=power,
        panels=(panel,),
        selection_audit=audit,
        report=report,
    )
    operator._checkpoint_next_path(terminal).write_bytes(
        expected if encoded == b"complete" else encoded
    )

    with pytest.raises(PostEmbeddingDevelopmentError, match="no feasible registered candidate"):
        operator._ensure_joint_power(config, freeze_tree)

    assert terminal.read_bytes() == expected
    assert not operator._checkpoint_next_path(terminal).exists()
    assert calls == {"audit": 0, "design": 0, "exact": 0}


def test_mismatched_no_feasible_terminal_checkpoint_is_immutable_and_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, audit, report, calls = _mock_resumable_joint_power(
        config,
        monkeypatch,
        feasible=False,
    )
    freeze_tree = _digest("freeze-tree")
    (config.output_root / JOINT_POWER_INVOCATION_FILENAME).write_bytes(
        operator._invocation_payload(freeze_tree, power, (panel,))
    )
    pending = config.output_root / ANALYSIS_DIRECTORY / operator.JOINT_POWER_PENDING_DIRECTORY
    _seed_joint_power_pending_boundary(
        pending,
        boundary="report-final",
        power=power,
        panel=panel,
        audit=audit,
        report=report,
    )
    terminal = pending / operator.JOINT_POWER_NO_FEASIBLE_FILENAME
    terminal.write_bytes(b"{}\n")

    with pytest.raises(PostEmbeddingDevelopmentError, match="differs from its invocation"):
        operator._ensure_joint_power(config, freeze_tree)

    assert terminal.read_bytes() == b"{}\n"
    assert calls == {"audit": 0, "design": 0, "exact": 0}


def test_complete_joint_power_pending_bundle_is_reused_and_published_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, audit, report, calls = _mock_resumable_joint_power(config, monkeypatch)
    freeze_tree = _digest("freeze-tree")
    marker = config.output_root / JOINT_POWER_INVOCATION_FILENAME
    marker.write_bytes(operator._invocation_payload(freeze_tree, power, (panel,)))
    pending = config.output_root / ANALYSIS_DIRECTORY / operator.JOINT_POWER_PENDING_DIRECTORY
    _seed_joint_power_pending_boundary(
        pending,
        boundary="complete-pending",
        power=power,
        panel=panel,
        audit=audit,
        report=report,
    )

    first = operator._ensure_joint_power(config, freeze_tree)
    published = config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY
    first_tree = operator.digest_directory_tree(published).sha256
    second = operator._ensure_joint_power(config, freeze_tree)

    assert calls == {"audit": 0, "design": 0, "exact": 2}
    assert first.exact_replay_performed is True
    assert second.exact_replay_performed is True
    assert operator.digest_directory_tree(published).sha256 == first_tree
    assert not pending.exists()


def test_failed_exact_replay_leaves_complete_pending_unpublished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, audit, report, _calls = _mock_resumable_joint_power(
        config,
        monkeypatch,
    )
    freeze_tree = _digest("freeze-tree")
    (config.output_root / JOINT_POWER_INVOCATION_FILENAME).write_bytes(
        operator._invocation_payload(freeze_tree, power, (panel,))
    )
    pending = config.output_root / ANALYSIS_DIRECTORY / operator.JOINT_POWER_PENDING_DIRECTORY
    _seed_joint_power_pending_boundary(
        pending,
        boundary="complete-pending",
        power=power,
        panel=panel,
        audit=audit,
        report=report,
    )

    def reject_exact(*args: object, **kwargs: object):
        assert args[0] == pending
        assert kwargs["reproduce_exact"] is True
        raise PostEmbeddingDevelopmentError("exact checkpoint mismatch")

    monkeypatch.setattr(operator, "_verify_joint_power_bundle", reject_exact)
    with pytest.raises(PostEmbeddingDevelopmentError, match="exact checkpoint mismatch"):
        operator._ensure_joint_power(config, freeze_tree)

    assert pending.is_dir()
    assert not (config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY).exists()


def test_ambiguous_invocation_next_is_rejected_before_legacy_work_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, _audit, _report, calls = _mock_resumable_joint_power(
        config,
        monkeypatch,
    )
    freeze_tree = _digest("freeze-tree")
    invocation_bytes = operator._invocation_payload(freeze_tree, power, (panel,))
    (config.output_root / JOINT_POWER_INVOCATION_FILENAME).write_bytes(invocation_bytes)
    analysis = config.output_root / ANALYSIS_DIRECTORY
    analysis.mkdir(mode=0o700)
    (analysis / operator._JOINT_POWER_INVOCATION_NEXT_FILENAME).write_bytes(invocation_bytes)
    interrupted = analysis / f".{JOINT_POWER_DIRECTORY}.work-interrupted"
    interrupted.mkdir(mode=0o700)
    partial = interrupted / "partial.json"
    partial.write_bytes(b"partial\n")

    with pytest.raises(PostEmbeddingDevelopmentError, match="both final and next"):
        operator._ensure_joint_power(config, freeze_tree)

    assert partial.read_bytes() == b"partial\n"
    assert calls == {"audit": 0, "design": 0, "exact": 0}


def test_joint_power_resume_rejects_unexpected_pending_members_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, _audit, _report, calls = _mock_resumable_joint_power(
        config,
        monkeypatch,
    )
    freeze_tree = _digest("freeze-tree")
    (config.output_root / JOINT_POWER_INVOCATION_FILENAME).write_bytes(
        operator._invocation_payload(freeze_tree, power, (panel,))
    )
    pending = config.output_root / ANALYSIS_DIRECTORY / operator.JOINT_POWER_PENDING_DIRECTORY
    pending.mkdir(mode=0o700, parents=True)
    unexpected = pending / "unexpected.json"
    unexpected.write_bytes(b"private\n")

    with pytest.raises(PostEmbeddingDevelopmentError, match="unexpected members"):
        operator._ensure_joint_power(config, freeze_tree)

    assert unexpected.read_bytes() == b"private\n"
    assert calls == {"audit": 0, "design": 0, "exact": 0}


def test_joint_power_resume_rejects_a_changed_invocation_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    _mock_resumable_joint_power(config, monkeypatch)
    marker = config.output_root / JOINT_POWER_INVOCATION_FILENAME
    marker.write_bytes(b"{}\n")
    interrupted = (
        config.output_root / ANALYSIS_DIRECTORY / f".{JOINT_POWER_DIRECTORY}.work-interrupted"
    )
    interrupted.mkdir(mode=0o700, parents=True)
    (interrupted / "partial.json").write_bytes(b"partial\n")
    with pytest.raises(PostEmbeddingDevelopmentError, match="marker differs"):
        operator._ensure_joint_power(config, _digest("freeze-tree"))
    assert interrupted.is_dir()


def test_joint_power_resume_removes_only_bound_interrupted_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir(mode=0o700)
    power, panel, _audit, _report, _calls = _mock_resumable_joint_power(
        config,
        monkeypatch,
    )
    freeze_tree = _digest("freeze-tree")
    marker = config.output_root / JOINT_POWER_INVOCATION_FILENAME
    marker.write_bytes(operator._invocation_payload(freeze_tree, power, (panel,)))
    interrupted = (
        config.output_root / ANALYSIS_DIRECTORY / f".{JOINT_POWER_DIRECTORY}.work-interrupted"
    )
    interrupted.mkdir(mode=0o700, parents=True)
    (interrupted / "partial.json").write_bytes(b"partial\n")

    operator._ensure_joint_power(config, freeze_tree)

    assert not interrupted.exists()
    assert (config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY).is_dir()


def test_existing_joint_bundle_performs_one_exact_replay_for_the_persisted_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    bundle = config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY
    bundle.mkdir(mode=0o700, parents=True)
    invocation = config.output_root / JOINT_POWER_INVOCATION_FILENAME
    invocation.write_bytes(b"invocation\n")
    power = SimpleNamespace(sha256=_digest("power"))
    observed: list[bool] = []
    monkeypatch.setattr(operator, "_joint_power_source", lambda value: (power, ()))

    def inspect(*args: object, **kwargs: object):
        observed.append(kwargs["reproduce_exact"])
        return power, (), SimpleNamespace(), _digest("tree")

    monkeypatch.setattr(operator, "_verify_joint_power_bundle", inspect)
    ensured = operator._ensure_joint_power(config, _digest("freeze-tree"))
    assert observed == [True]
    assert ensured.exact_replay_performed is True


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
