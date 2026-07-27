from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import signal
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_study import _candidate_rehearsal_manifest

from fractal_ann_diagnostics.study import C0_COMMIT_SENTINEL
from operators import candidate_study_manifest_composer as composer


def _digest(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _request(tmp_path: Path) -> composer.CompositionRequest:
    input_root = tmp_path / "inputs"
    bindings: list[tuple[str, composer.ExactInput]] = []
    for name in composer._INPUT_NAMES:
        filename = composer._EXPECTED_DIRECT_NAMES.get(
            name,
            "candidate-image-closure.json",
        )
        bindings.append(
            (
                name,
                composer.ExactInput(
                    path=input_root / filename,
                    sha256=hashlib.sha256(name.encode("ascii")).hexdigest(),
                ),
            )
        )
    return composer.CompositionRequest(inputs=tuple(bindings))


def _capturable_request(tmp_path: Path) -> tuple[composer.CompositionRequest, Path]:
    input_root = tmp_path / "inputs"
    bindings: list[tuple[str, composer.ExactInput]] = []
    for name in composer._INPUT_NAMES:
        filename = composer._EXPECTED_DIRECT_NAMES.get(
            name,
            "candidate-image-closure.json",
        )
        path = input_root / filename
        encoded = composer._canonical_bytes({"role": name})
        _write_private(path, encoded)
        bindings.append(
            (
                name,
                composer.ExactInput(
                    path=path,
                    sha256=_digest(encoded),
                ),
            )
        )
    request = composer.CompositionRequest(inputs=tuple(bindings))
    request_path = tmp_path / "composition-request.json"
    _write_private(request_path, request.canonical_file_bytes())
    return request, request_path


def _write_private(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(encoded)
    path.chmod(0o600)


def _publication_stubs(
    monkeypatch: pytest.MonkeyPatch,
    request: composer.CompositionRequest,
) -> tuple[dict[str, object], dict[str, object]]:
    source: dict[str, object] = {"artifacts": []}
    receipt: dict[str, object] = {"schema_version": "test-composition-receipt"}
    captured = SimpleNamespace(
        request=request,
        capture_set_sha256="c" * 64,
        assert_current=lambda: None,
        close=lambda: None,
    )
    monkeypatch.setattr(
        composer,
        "_capture_input_set",
        lambda path, digest: captured,
    )
    monkeypatch.setattr(composer, "_load_request", lambda path, digest: request)
    monkeypatch.setattr(
        composer,
        "_derive_candidate_source",
        lambda value, captured=None: source,
    )
    monkeypatch.setattr(
        composer,
        "_composition_receipt",
        lambda request_path, value, candidate, captured=None: receipt,
    )
    return source, receipt


def _design_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[composer._Authorities, dict[str, object]]:
    candidate = _candidate_rehearsal_manifest()
    sealed = candidate["sealed_execution"]
    analysis = candidate["analysis"]
    assert isinstance(sealed, dict)
    assert isinstance(analysis, dict)
    power = analysis["power"]
    assert isinstance(power, dict)
    sealed["provider_phase_plans"] = "tbd"
    scenario_ids = (
        "expected-development-effect",
        "conservative-registered-attenuation",
    )
    power["effect_scenarios"] = sorted(scenario_ids)

    thresholds = {
        "auprc_gain": 0.0,
        "brier_score_reduction": 0.0,
        "log_loss_reduction": 0.0,
    }
    effect_scenarios = tuple(
        SimpleNamespace(
            panel_sha256=hashlib.sha256(name.encode("ascii")).hexdigest(),
            scenario_id=name,
            selection_required=True,
        )
        for name in scenario_ids
    )
    power_config = SimpleNamespace(
        alpha=analysis["alpha"],
        candidate_families_per_corpus=tuple(power["candidate_families_per_corpus"]),
        dependence_source=SimpleNamespace(
            artifact_uri=f"urn:sha256:{'9' * 64}",
        ),
        effect_scenarios=effect_scenarios,
        evidence_corpora=tuple(analysis["evidence_corpora"]),
        evidence_sufficiency_noninferiority_margin=analysis[
            "evidence_sufficiency_noninferiority_margin"
        ],
        fixed_corpora=tuple(analysis["fixed_corpora"]),
        geometry_gain_thresholds=SimpleNamespace(to_dict=lambda: thresholds),
        maximum_denied_emissions=analysis["maximum_entitlement_violations"],
        maximum_p95_latency_ratio=analysis["maximum_p95_latency_ratio"],
        minimum_corpora_with_geometry_gain=analysis["minimum_corpora_with_geometry_gain"],
        minimum_latency_reduction=analysis["minimum_cost_reduction"],
        n_simulations=power["simulation_count"],
        nested_rows_per_family=analysis["nested_rows_per_family"],
        retrieval_target_noninferiority_margin=analysis["retrieval_target_noninferiority_margin"],
        simulation_seed=power["simulation_seed"],
        target_power=analysis["power_target"],
    )
    estimates = tuple(
        SimpleNamespace(
            families_per_corpus=power["selected_families_per_corpus"],
            joint_probability=SimpleNamespace(lower_probability_bound=0.91),
            scenario_id=name,
            selection_required=True,
        )
        for name in scenario_ids
    )
    power_report = SimpleNamespace(
        estimates=estimates,
        freeze_ready=True,
        selected_families_per_corpus=power["selected_families_per_corpus"],
        selection_cell_alpha=power["selection_cell_alpha"],
        selection_family_size=power["selection_family_size"],
        selection_familywise_confidence=power["selection_familywise_confidence"],
        selection_multiplicity_method=power["selection_multiplicity_method"],
        selection_satisfied=True,
    )
    production_config = SimpleNamespace(
        approval_environment=sealed["approval_environment"],
        candidate_image_source_commit="a" * 40,
        file_sha256="3" * 64,
        runner_identity=sealed["runner_identity"],
        scientific_production_reference=sealed["runner_image"],
    )
    blueprint = SimpleNamespace(
        candidate_image_source_commit="a" * 40,
        file_sha256="5" * 64,
        runner_image=sealed["runner_image"],
        semantic_sha256="4" * 64,
    )
    workloads = copy.deepcopy(candidate["production_workloads"])
    hardware = copy.deepcopy(sealed["hardware"])
    assert isinstance(workloads, list)
    assert isinstance(hardware, dict)
    authorities = composer._Authorities(
        template={},
        artifact_inventory=object(),
        post_embedding=object(),
        power_config=power_config,
        power_report=power_report,
        geometry_profiles={
            "geometry_gain_thresholds": thresholds,
            "high_geometry": copy.deepcopy(analysis["high_geometry"]),
            "low_geometry": copy.deepcopy(analysis["low_geometry"]),
        },
        static_comparator="hnsw-high",
        production_config=production_config,
        production_config_write_receipt=object(),
        production_blueprint=blueprint,
        workloads=workloads,
        hardware=hardware,
        candidate_image_closure=SimpleNamespace(github_sha="a" * 40),
        deployment=composer.CandidateDeploymentFragment(
            custodian="custodian@example.test",
            receipt_uri_template=("file:///controlled/receipts/{manifest_sha256}.json"),
            results_store="s3://immutable-results/candidate",
        ),
    )
    monkeypatch.setattr(
        composer,
        "apply_candidate_artifact_inventory",
        lambda template, inventory: copy.deepcopy(candidate),
    )
    return authorities, candidate


def test_request_surface_is_exact_and_excludes_outcome_payloads(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert tuple(name for name, _ in request.inputs) == composer._INPUT_NAMES
    assert "artifact_inventory_receipt" in composer._INPUT_NAMES
    forbidden = ("action_panel", "label_payload", "sealed_label", "confirmatory_result")
    assert not any(token in name for name in composer._INPUT_NAMES for token in forbidden)

    missing = tuple(item for item in request.inputs if item[0] != "joint_power_report")
    with pytest.raises(composer.CandidateSourceComposerError, match="roles differ"):
        composer.CompositionRequest(inputs=missing)
    with pytest.raises(composer.CandidateSourceComposerError, match="roles differ"):
        composer.CompositionRequest(inputs=request.inputs + (request.inputs[0],))
    aliased = dict(request.inputs)
    aliased["template"] = aliased["static_comparator"]
    with pytest.raises(composer.CandidateSourceComposerError, match="paths must be distinct"):
        composer.CompositionRequest(inputs=tuple(aliased.items()))


def test_request_file_requires_exact_digest_and_canonical_bytes(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request_path = tmp_path / "composition-request.json"
    _write_private(request_path, request.canonical_file_bytes())
    assert composer._load_request(request_path, request.file_sha256) == request

    pretty = json.dumps(request.to_dict(), indent=2).encode("utf-8") + b"\n"
    _write_private(tmp_path / "noncanonical.json", pretty)
    with pytest.raises(composer.CandidateSourceComposerError, match="canonical"):
        composer._load_request(tmp_path / "noncanonical.json", _digest(pretty))
    with pytest.raises(composer.CandidateSourceComposerError, match="request file digest"):
        composer._load_request(request_path, "0" * 64)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("custodian", " tbd "),
        ("results_store", "file:relative"),
        ("results_store", "s3:/missing-bucket"),
        ("results_store", "s3://user:secret@bucket/prefix"),
        (
            "receipt_uri_template",
            "file:///controlled/receipts/manifest.json",
        ),
        (
            "receipt_uri_template",
            "s3://bucket/{manifest_sha256}.json",
        ),
    ),
)
def test_deployment_fragment_rejects_movable_or_credentialed_values(
    field: str,
    replacement: str,
) -> None:
    value = {
        "custodian": "custodian@example.test",
        "receipt_uri_template": "file:///receipts/{manifest_sha256}.json",
        "results_store": "s3://immutable-results/candidate",
        "schema_version": composer.DEPLOYMENT_FRAGMENT_SCHEMA,
    }
    value[field] = replacement
    with pytest.raises(composer.CandidateSourceComposerError):
        composer.CandidateDeploymentFragment.from_dict(value)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(composer.CandidateSourceComposerError, match="repeats"):
        composer._decode_json(b'{"a":1,"a":2}\n', label="duplicate")
    with pytest.raises(composer.CandidateSourceComposerError, match="non-finite"):
        composer._decode_json(b'{"a":NaN}\n', label="nonfinite")


def test_owned_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "producer.json"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(composer.CandidateSourceComposerError, match="regular file"):
        composer._read_owned_file(fifo, label="fifo")


def test_owned_reader_rejects_a_linked_parent_component(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    encoded = b'{"value":"pinned"}\n'
    _write_private(real / "producer.json", encoded)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(composer.CandidateSourceComposerError, match="cannot open"):
        composer._read_owned_file(alias / "producer.json", label="linked producer")


def test_directory_identity_readback_reopens_every_ancestor(tmp_path: Path) -> None:
    original = tmp_path / "original"
    parent = original / "parent"
    parent.mkdir(mode=0o700, parents=True)
    descriptor = composer._open_directory_chain(parent, label="test parent")
    moved = tmp_path / "moved"
    try:
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
        with pytest.raises(composer.CandidateSourceComposerError, match="cannot open"):
            composer._assert_path_names_directory(
                parent,
                descriptor,
                label="test parent",
            )
    finally:
        os.close(descriptor)


def test_directory_descriptor_proves_removal_not_relocation(tmp_path: Path) -> None:
    original = tmp_path / "staging"
    relocated = tmp_path / "relocated"
    original.mkdir(mode=0o700)
    descriptor = os.open(
        original,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        original.rename(relocated)
        assert not composer._directory_descriptor_is_unlinked(descriptor)
        relocated.rmdir()
        assert composer._directory_descriptor_is_unlinked(descriptor)
    finally:
        os.close(descriptor)


def test_inventory_receipt_is_validated_from_captured_bytes_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_candidate_manifest_assembler import _inventory

    template, inventory = _inventory()
    bindings = dict(_request(tmp_path).inputs)
    bindings["artifact_inventory"] = composer.ExactInput(
        path=bindings["artifact_inventory"].path,
        sha256=inventory.file_sha256,
    )
    request = composer.CompositionRequest(inputs=tuple(bindings.items()))
    receipt = {
        "artifact_count": 79,
        "artifact_root": "/controlled/artifacts",
        "inventory_file_sha256": inventory.file_sha256,
        "repository_root": "/controlled/repository",
        "schema_version": composer.CANDIDATE_ARTIFACT_PIN_RECEIPT_SCHEMA,
        "template_sha256": inventory.template_sha256,
    }

    def unexpected_path_read(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("captured inventory validation reopened a path")

    monkeypatch.setattr(composer, "_read_owned_file", unexpected_path_read)
    assert (
        composer._load_captured_artifact_inventory(
            request=request,
            inventory_value=inventory.to_dict(),
            inventory_bytes=inventory.canonical_file_bytes,
            receipt_value=receipt,
            template_value=template,
        )
        == inventory
    )
    changed = dict(receipt)
    changed["inventory_file_sha256"] = "f" * 64
    with pytest.raises(composer.CandidateSourceComposerError, match="captured-byte closure"):
        composer._load_captured_artifact_inventory(
            request=request,
            inventory_value=inventory.to_dict(),
            inventory_bytes=inventory.canonical_file_bytes,
            receipt_value=changed,
            template_value=template,
        )


def test_workload_fragment_rejects_digest_and_commit_overrides() -> None:
    candidate = _candidate_rehearsal_manifest()
    sealed = candidate["sealed_execution"]
    workloads = candidate["production_workloads"]
    power = candidate["analysis"]["power"]  # type: ignore[index]
    assert isinstance(sealed, dict)
    assert isinstance(workloads, list)
    assert isinstance(power, dict)
    config = SimpleNamespace(
        runner_identity=sealed["runner_identity"],
        scientific_production_reference=sealed["runner_image"],
    )
    observed = composer._typed_workload_fragment(
        workloads,
        selected_families_per_corpus=power["selected_families_per_corpus"],
        config=config,
    )
    assert observed == workloads

    changed = copy.deepcopy(workloads)
    changed[0]["canonical_file_sha256"] = "0" * 64
    with pytest.raises(composer.CandidateSourceComposerError, match="override"):
        composer._typed_workload_fragment(
            changed,
            selected_families_per_corpus=power["selected_families_per_corpus"],
            config=config,
        )
    changed = copy.deepcopy(workloads)
    changed[0]["spec"]["code_commit"] = "b" * 40
    with pytest.raises(composer.CandidateSourceComposerError, match="override"):
        composer._typed_workload_fragment(
            changed,
            selected_families_per_corpus=power["selected_families_per_corpus"],
            config=config,
        )


def test_freeze_design_provenance_binds_metadata_without_opening_panels() -> None:
    calibration = "1" * 64
    conservative = "2" * 64
    expected = "3" * 64
    pins = {
        "development-calibration-outcomes.json": calibration,
        "joint-power-conservative-panel.json": conservative,
        "joint-power-expected-panel.json": expected,
    }
    thresholds = {
        "auprc_gain": 0.0,
        "brier_score_reduction": 0.0,
        "log_loss_reduction": 0.0,
    }
    config = SimpleNamespace(
        dependence_source=SimpleNamespace(
            artifact_sha256=calibration,
            artifact_uri=f"urn:sha256:{calibration}",
            partition="development-calibration",
        ),
        effect_scenarios=(
            SimpleNamespace(
                panel_sha256=expected,
                scenario_id="expected-development-effect",
            ),
            SimpleNamespace(
                panel_sha256=conservative,
                scenario_id="conservative-registered-attenuation",
            ),
        ),
        geometry_gain_thresholds=SimpleNamespace(to_dict=lambda: thresholds),
    )
    report = SimpleNamespace(
        panel_sha256s=(
            ("conservative-registered-attenuation", conservative),
            ("expected-development-effect", expected),
        )
    )
    geometry = {"geometry_gain_thresholds": thresholds}
    composer._assert_freeze_design_provenance(
        freeze_pins=pins,
        geometry_profiles=geometry,
        config=config,
        report=report,
    )

    changed = copy.deepcopy(config)
    changed.effect_scenarios[0].scenario_id = "substituted-effect"
    with pytest.raises(composer.CandidateSourceComposerError, match="scenario pins"):
        composer._assert_freeze_design_provenance(
            freeze_pins=pins,
            geometry_profiles=geometry,
            config=changed,
            report=report,
        )
    changed = copy.deepcopy(config)
    changed.dependence_source.artifact_sha256 = "4" * 64
    with pytest.raises(composer.CandidateSourceComposerError, match="dependence source"):
        composer._assert_freeze_design_provenance(
            freeze_pins=pins,
            geometry_profiles=geometry,
            config=changed,
            report=report,
        )
    changed_report = copy.deepcopy(report)
    changed_report.panel_sha256s = (
        ("conservative-registered-attenuation", "5" * 64),
        ("expected-development-effect", expected),
    )
    with pytest.raises(composer.CandidateSourceComposerError, match="scenario pins"):
        composer._assert_freeze_design_provenance(
            freeze_pins=pins,
            geometry_profiles=geometry,
            config=config,
            report=changed_report,
        )
    with pytest.raises(composer.CandidateSourceComposerError, match="thresholds"):
        composer._assert_freeze_design_provenance(
            freeze_pins=pins,
            geometry_profiles={"geometry_gain_thresholds": {**thresholds, "auprc_gain": 0.1}},
            config=config,
            report=report,
        )


def test_composition_derives_the_exact_pre_provider_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorities, _ = _design_authorities(monkeypatch)
    source = composer._compose_candidate_source(authorities)
    assert source["freeze_blockers"] == list(composer._FREEZE_BLOCKERS)
    assert composer._collect_exact_values(
        source,
        composer._PLACEHOLDER_VALUES,
    ) == set(composer._ALLOWED_UNRESOLVED)
    assert len(composer._collect_exact_values(source, frozenset({C0_COMMIT_SENTINEL}))) == 7
    assert source["analysis"]["power"]["effect_scenarios"] == [  # type: ignore[index]
        "conservative-registered-attenuation",
        "expected-development-effect",
    ]


def test_tracked_scenario_ids_cannot_be_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorities, candidate = _design_authorities(monkeypatch)
    power = candidate["analysis"]["power"]  # type: ignore[index]
    assert isinstance(power, dict)
    power["effect_scenarios"] = ["alternative-a", "alternative-b"]
    with pytest.raises(composer.CandidateSourceComposerError, match="effect_scenarios"):
        composer._compose_candidate_source(authorities)


def test_publication_is_exclusive_private_and_umask_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    source, receipt = _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    old_umask = os.umask(0o477)
    try:
        assert (
            composer.compose_from_request(
                request_path=tmp_path / "request.json",
                request_sha256="f" * 64,
                output_directory=destination,
            )
            == receipt
        )
    finally:
        os.umask(old_umask)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert {path.name for path in destination.iterdir()} == {
        composer.SOURCE_FILENAME,
        composer.COMPOSITION_RECEIPT_FILENAME,
    }
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in destination.iterdir())
    assert composer._read_published_package(destination) == (source, receipt)
    with pytest.raises(composer.CandidateSourceComposerError, match="already exists"):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )


def test_interrupted_publication_leaves_no_output_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    original = composer._write_private_at
    calls = 0

    def fail_second(parent_descriptor: int, name: str, encoded: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write interruption")
        original(parent_descriptor, name, encoded)

    monkeypatch.setattr(composer, "_write_private_at", fail_second)
    destination = tmp_path / "candidate-source-package"
    with pytest.raises(OSError, match="injected"):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-source-package.staging-*"))


def test_first_named_stat_failure_is_indeterminate_without_captured_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    original_stat = os.stat
    injected = False

    def fail_first_staging_stat(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal injected
        if (
            not injected
            and isinstance(path, str)
            and path.startswith(".candidate-source-package.staging-")
            and kwargs.get("dir_fd") is not None
        ):
            injected = True
            raise OSError("injected first named stat failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fail_first_staging_stat)
    destination = tmp_path / "candidate-source-package"
    with pytest.raises(
        composer.CandidateSourcePublicationIndeterminateError,
        match="staging cleanup cannot be proved",
    ):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert injected
    assert not destination.exists()
    assert len(list(tmp_path.glob(".candidate-source-package.staging-*"))) == 1


def test_first_stat_name_swap_preserves_both_unproven_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    original_stat = os.stat
    alternate_name = ".operator-created-staging-relocated"
    swapped_name: str | None = None

    def swap_then_fail_first_staging_stat(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal swapped_name
        directory_fd = kwargs.get("dir_fd")
        if (
            swapped_name is None
            and isinstance(path, str)
            and path.startswith(".candidate-source-package.staging-")
            and isinstance(directory_fd, int)
        ):
            swapped_name = path
            os.rename(
                path,
                alternate_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.mkdir(path, 0o700, dir_fd=directory_fd)
            raise OSError("injected first-stat staging-name ambiguity")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", swap_then_fail_first_staging_stat)
    destination = tmp_path / "candidate-source-package"
    with pytest.raises(
        composer.CandidateSourcePublicationIndeterminateError,
        match="staging cleanup cannot be proved",
    ):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )

    assert swapped_name is not None
    assert not destination.exists()
    assert (tmp_path / alternate_name).is_dir()
    assert (tmp_path / swapped_name).is_dir()
    assert (tmp_path / alternate_name).stat().st_ino != (tmp_path / swapped_name).stat().st_ino


def test_staging_is_cleaned_when_setup_fails_before_descriptor_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    original_chmod = os.chmod
    injected = False

    def fail_staging_chmod(path: object, *args: object, **kwargs: object) -> None:
        nonlocal injected
        if (
            not injected
            and isinstance(path, str)
            and path.startswith(".candidate-source-package.staging-")
            and kwargs.get("dir_fd") is not None
        ):
            injected = True
            raise OSError("injected pre-binding chmod failure")
        original_chmod(path, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", fail_staging_chmod)
    destination = tmp_path / "candidate-source-package"
    with pytest.raises(OSError, match="pre-binding"):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert injected
    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-source-package.staging-*"))


def test_preopen_staging_swap_does_not_delete_foreign_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    original_chmod = os.chmod
    alternate_name = ".captured-created-staging"
    swapped_name: str | None = None
    sentinel = b'{"foreign":true}\n'

    def swap_before_open(path: object, *args: object, **kwargs: object) -> None:
        nonlocal swapped_name
        directory_fd = kwargs.get("dir_fd")
        if (
            swapped_name is None
            and isinstance(path, str)
            and path.startswith(".candidate-source-package.staging-")
            and isinstance(directory_fd, int)
        ):
            swapped_name = path
            os.rename(
                path,
                alternate_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.mkdir(path, 0o700, dir_fd=directory_fd)
            foreign = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=directory_fd,
            )
            try:
                composer._write_private_at(
                    foreign,
                    composer.SOURCE_FILENAME,
                    sentinel,
                )
            finally:
                os.close(foreign)
        original_chmod(path, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", swap_before_open)
    destination = tmp_path / "candidate-source-package"
    with pytest.raises(composer.CandidateSourceComposerError, match="substituted"):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert swapped_name is not None
    assert not destination.exists()
    assert not (tmp_path / alternate_name).exists()
    assert (tmp_path / swapped_name / composer.SOURCE_FILENAME).read_bytes() == sentinel


def test_captured_input_reproduction_occurs_immediately_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    source, _ = _publication_stubs(monkeypatch, request)
    derivation_calls = 0

    def changing_derivation(
        value: composer.CompositionRequest,
        captured: object = None,
    ) -> dict[str, object]:
        del captured
        nonlocal derivation_calls
        derivation_calls += 1
        if derivation_calls == 1:
            return source
        return {"artifacts": [{"changed": True}]}

    monkeypatch.setattr(composer, "_derive_candidate_source", changing_derivation)
    destination = tmp_path / "candidate-source-package"
    with pytest.raises(
        composer.CandidateSourceComposerError,
        match="captured producer evidence changed",
    ):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert derivation_calls == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-source-package.staging-*"))


def test_prerename_parent_control_drift_fails_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    source, _ = _publication_stubs(monkeypatch, request)
    derivation_calls = 0

    def drift_parent_during_revalidation(
        value: composer.CompositionRequest,
        captured: object = None,
    ) -> dict[str, object]:
        del captured
        nonlocal derivation_calls
        derivation_calls += 1
        if derivation_calls == 2:
            tmp_path.chmod(0o777)
        return source

    monkeypatch.setattr(
        composer,
        "_derive_candidate_source",
        drift_parent_during_revalidation,
    )
    destination = tmp_path / "candidate-source-package"
    try:
        with pytest.raises(
            composer.CandidateSourceComposerError,
            match="output parent identity, ownership, or mode changed",
        ):
            composer.compose_from_request(
                request_path=tmp_path / "request.json",
                request_sha256="f" * 64,
                output_directory=destination,
            )
    finally:
        tmp_path.chmod(0o700)

    assert derivation_calls == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-source-package.staging-*"))


def test_verification_requires_exact_output_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    composer.compose_from_request(
        request_path=tmp_path / "request.json",
        request_sha256="f" * 64,
        output_directory=destination,
    )

    source_path = destination / composer.SOURCE_FILENAME
    source_path.chmod(0o400)
    with pytest.raises(composer.CandidateSourceComposerError, match="exact regular file"):
        composer._read_published_package(destination)
    source_path.chmod(0o600)
    destination.chmod(0o750)
    with pytest.raises(composer.CandidateSourceComposerError, match="private directory"):
        composer._read_published_package(destination)


@pytest.mark.parametrize("operation", ("compose", "verify"))
def test_joint_package_scan_detects_receipt_mutation_after_its_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    source: dict[str, object] = {"artifacts": []}
    captured = SimpleNamespace(
        request=request,
        capture_set_sha256="c" * 64,
        assert_current=lambda: None,
        close=lambda: None,
    )
    monkeypatch.setattr(
        composer,
        "_capture_input_set",
        lambda path, digest: captured,
    )
    monkeypatch.setattr(
        composer,
        "_derive_candidate_source",
        lambda value, captured=None: source,
    )
    destination = tmp_path / "candidate-source-package"
    if operation == "verify":
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )

    original_read = composer._RetainedFileRead.read
    receipt_was_read = False
    mutated = False

    def mutate_between_member_reads(self: composer._RetainedFileRead) -> bytes:
        nonlocal mutated, receipt_was_read
        encoded = original_read(self)
        if self.name == composer.COMPOSITION_RECEIPT_FILENAME:
            receipt_was_read = True
        elif self.name == composer.SOURCE_FILENAME and receipt_was_read and not mutated:
            descriptor = os.open(
                composer.COMPOSITION_RECEIPT_FILENAME,
                os.O_WRONLY | os.O_TRUNC,
                dir_fd=self.parent_descriptor,
            )
            try:
                os.write(descriptor, b'{"tampered":true}\n')
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            mutated = True
        return encoded

    monkeypatch.setattr(
        composer._RetainedFileRead,
        "read",
        mutate_between_member_reads,
    )
    with pytest.raises(
        composer.CandidateSourceComposerError, match="changed during its exact read"
    ):
        if operation == "compose":
            composer.compose_from_request(
                request_path=tmp_path / "request.json",
                request_sha256="f" * 64,
                output_directory=destination,
            )
        else:
            composer.verify_composed_package(destination)
    assert mutated
    if operation == "compose":
        assert not destination.exists()
        assert not list(tmp_path.glob(".candidate-source-package.*"))


def test_cleanup_quarantines_identity_before_preserving_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    parent = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    composer._acquire_exclusive_lease(parent, label="test cleanup parent")
    preferred = ".candidate-source-package.staging-test"
    relocated = ".candidate-source-package.staging-relocated"
    os.mkdir(preferred, 0o700, dir_fd=parent)
    stage = os.open(
        preferred,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        dir_fd=parent,
    )
    metadata = os.fstat(stage)
    expected_inode = (metadata.st_dev, metadata.st_ino)
    original_stat = os.stat
    injected = False

    def replace_after_cleanup_stat(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal injected
        observed = original_stat(path, *args, **kwargs)
        if (
            not injected
            and path == preferred
            and kwargs.get("dir_fd") == parent
            and kwargs.get("follow_symlinks") is False
        ):
            injected = True
            os.rename(
                preferred,
                relocated,
                src_dir_fd=parent,
                dst_dir_fd=parent,
            )
            os.mkdir(preferred, 0o700, dir_fd=parent)
            foreign = os.open(
                preferred,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=parent,
            )
            try:
                composer._write_private_at(
                    foreign,
                    composer.SOURCE_FILENAME,
                    b'{"foreign":true}\n',
                )
            finally:
                os.close(foreign)
        return observed

    monkeypatch.setattr(os, "stat", replace_after_cleanup_stat)
    try:
        assert composer._remove_staging_directory(
            parent,
            stage,
            expected_inode,
            preferred_name=preferred,
        )
        assert injected
        assert not (tmp_path / relocated).exists()
        assert (
            tmp_path / preferred / composer.SOURCE_FILENAME
        ).read_bytes() == b'{"foreign":true}\n'
        assert not list(tmp_path.glob("*.quarantine-*"))
    finally:
        os.close(stage)
        os.close(parent)


def test_final_destination_boundary_blocks_cooperating_same_uid_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    original = composer._assert_named_directory_matches_descriptor
    blocked = False

    def attempt_cooperating_replacement(*args: object, **kwargs: object) -> None:
        nonlocal blocked
        if kwargs.get("label") == "composition output directory final readback":
            contender = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                blocked = True
            finally:
                os.close(contender)
        original(*args, **kwargs)

    monkeypatch.setattr(
        composer,
        "_assert_named_directory_matches_descriptor",
        attempt_cooperating_replacement,
    )
    composer.compose_from_request(
        request_path=tmp_path / "request.json",
        request_sha256="f" * 64,
        output_directory=destination,
    )
    assert blocked
    assert destination.is_dir()


def test_captured_input_mutation_before_publish_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request, request_path = _capturable_request(tmp_path)
    source: dict[str, object] = {"artifacts": []}
    calls = 0

    def mutate_after_second_derivation(
        value: composer.CompositionRequest,
        captured: composer._CapturedInputSet | None = None,
    ) -> dict[str, object]:
        nonlocal calls
        assert value == request
        assert captured is not None
        calls += 1
        if calls == 2:
            binding = request.binding("artifact_inventory")
            binding.path.write_bytes(b'{"mutated":true}\n')
        return source

    monkeypatch.setattr(
        composer,
        "_derive_candidate_source",
        mutate_after_second_derivation,
    )
    destination = tmp_path / "candidate-source-package"
    with pytest.raises(
        composer.CandidateSourceComposerError, match="changed during its exact read"
    ):
        composer.compose_from_request(
            request_path=request_path,
            request_sha256=request.file_sha256,
            output_directory=destination,
        )
    assert calls == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-source-package.*"))


@pytest.mark.parametrize("signum", (signal.SIGTERM, signal.SIGHUP))
@pytest.mark.parametrize("phase", ("before-rename", "after-rename"))
def test_term_and_hup_follow_clean_interruption_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signum: int,
    phase: str,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    original = composer._rename_noreplace

    def interrupt(
        parent: int,
        source: str,
        target: str,
        *,
        expected_inode: tuple[int, int],
    ) -> None:
        if phase == "before-rename":
            os.kill(os.getpid(), signum)
        original(
            parent,
            source,
            target,
            expected_inode=expected_inode,
        )
        os.kill(os.getpid(), signum)

    monkeypatch.setattr(composer, "_rename_noreplace", interrupt)
    with pytest.raises(composer.CandidateSourceInterruptedError) as raised:
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert raised.value.signum == signum
    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-source-package.*"))


def test_system_exit_during_final_stage_descriptor_close_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    original_boundary = composer._assert_named_directory_matches_descriptor
    original_close = os.close
    retained_stage: int | None = None
    boundary_calls = 0
    injected = False

    def record_stage(*args: object, **kwargs: object) -> None:
        nonlocal boundary_calls, retained_stage
        retained_stage = args[2]
        boundary_calls += 1
        original_boundary(*args, **kwargs)

    def interrupt_close(descriptor: int) -> None:
        nonlocal injected
        if (
            not injected
            and retained_stage is not None
            and descriptor == retained_stage
            and boundary_calls == 2
        ):
            injected = True
            raise SystemExit("injected final descriptor close")
        original_close(descriptor)

    monkeypatch.setattr(
        composer,
        "_assert_named_directory_matches_descriptor",
        record_stage,
    )
    monkeypatch.setattr(os, "close", interrupt_close)
    with pytest.raises(composer.CandidateSourceComposerError, match="was rolled back"):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert injected
    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-source-package.*"))


def test_postrename_parent_control_drift_enters_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    original_rename = composer._rename_noreplace

    def drift_parent_after_rename(
        parent: int,
        source: str,
        target: str,
        *,
        expected_inode: tuple[int, int],
    ) -> None:
        original_rename(
            parent,
            source,
            target,
            expected_inode=expected_inode,
        )
        os.fchmod(parent, 0o777)

    monkeypatch.setattr(composer, "_rename_noreplace", drift_parent_after_rename)
    try:
        with pytest.raises(
            composer.CandidateSourcePublicationIndeterminateError,
            match="clean rollback cannot be proved",
        ):
            composer.compose_from_request(
                request_path=tmp_path / "request.json",
                request_sha256="f" * 64,
                output_directory=destination,
            )
    finally:
        tmp_path.chmod(0o700)

    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-source-package.*"))


def test_parent_control_drift_during_postrename_readback_never_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    original_read = composer._read_package_at
    read_calls = 0

    def drift_parent_after_readback(
        descriptor: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        nonlocal read_calls
        observed = original_read(descriptor)
        read_calls += 1
        if read_calls == 2:
            tmp_path.chmod(0o777)
        return observed

    monkeypatch.setattr(composer, "_read_package_at", drift_parent_after_readback)
    try:
        with pytest.raises(
            composer.CandidateSourcePublicationIndeterminateError,
            match="clean rollback cannot be proved",
        ):
            composer.compose_from_request(
                request_path=tmp_path / "request.json",
                request_sha256="f" * 64,
                output_directory=destination,
            )
    finally:
        tmp_path.chmod(0o700)

    assert read_calls == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-source-package.*"))


def test_post_rename_fsync_failure_rolls_back_when_proven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    original_fsync = os.fsync
    injected = False

    def fail_first_parent_fsync(descriptor: int) -> None:
        nonlocal injected
        metadata = os.fstat(descriptor)
        if not injected and (metadata.st_dev, metadata.st_ino) == parent_identity:
            injected = True
            raise OSError("injected parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_parent_fsync)
    with pytest.raises(composer.CandidateSourceComposerError, match="was rolled back"):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert injected
    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-source-package.*"))


def test_post_rename_readback_failure_rolls_back_when_inode_is_proven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    original_rename = composer._rename_noreplace

    def alter_member_mode_after_rename(
        parent: int,
        source: str,
        target: str,
        *,
        expected_inode: tuple[int, int],
    ) -> None:
        original_rename(
            parent,
            source,
            target,
            expected_inode=expected_inode,
        )
        published = os.open(
            target,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        try:
            os.chmod(
                composer.SOURCE_FILENAME,
                0o400,
                dir_fd=published,
                follow_symlinks=False,
            )
        finally:
            os.close(published)

    monkeypatch.setattr(composer, "_rename_noreplace", alter_member_mode_after_rename)
    with pytest.raises(composer.CandidateSourceComposerError, match="was rolled back"):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".candidate-source-package.*"))


def test_unprovable_post_rename_durability_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    original_fsync = os.fsync

    def fail_parent_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == parent_identity:
            raise OSError("injected persistent parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_parent_fsync)
    with pytest.raises(
        composer.CandidateSourcePublicationIndeterminateError,
        match="clean rollback cannot be proved",
    ):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert not destination.exists()


def test_post_rename_parent_relocation_never_returns_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled = tmp_path / "controlled"
    controlled.mkdir(mode=0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = controlled / "candidate-source-package"
    moved = tmp_path / "controlled-moved"
    original_rename = composer._rename_noreplace

    def relocate_parent_after_rename(*args: object, **kwargs: object) -> None:
        original_rename(*args, **kwargs)
        controlled.rename(moved)
        controlled.mkdir(mode=0o700)

    monkeypatch.setattr(composer, "_rename_noreplace", relocate_parent_after_rename)
    with pytest.raises(
        composer.CandidateSourcePublicationIndeterminateError,
        match="clean rollback cannot be proved",
    ):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert not destination.exists()
    assert not (moved / destination.name).exists()


def test_source_name_swap_is_restored_without_stranding_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    original_raw_rename = composer._raw_rename_noreplace
    swapped_name: str | None = None
    alternate_name = ".captured-original-staging"

    def swap_at_syscall(parent: int, source: str, target: str) -> None:
        nonlocal swapped_name
        if swapped_name is None and target == destination.name:
            swapped_name = source
            os.rename(
                source,
                alternate_name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
            )
            os.mkdir(source, 0o700, dir_fd=parent)
        original_raw_rename(parent, source, target)

    monkeypatch.setattr(composer, "_raw_rename_noreplace", swap_at_syscall)
    with pytest.raises(composer.CandidateSourceComposerError, match="foreign entry was restored"):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert swapped_name is not None
    assert not destination.exists()
    assert not (tmp_path / alternate_name).exists()
    assert (tmp_path / swapped_name).is_dir()


def test_destination_swap_after_rename_is_reported_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    _publication_stubs(monkeypatch, request)
    destination = tmp_path / "candidate-source-package"
    relocated = tmp_path / ".relocated-published-package"
    original_rename = composer._rename_noreplace

    def swap_destination_after_rename(*args: object, **kwargs: object) -> None:
        original_rename(*args, **kwargs)
        destination.rename(relocated)
        destination.mkdir(mode=0o700)

    monkeypatch.setattr(composer, "_rename_noreplace", swap_destination_after_rename)
    with pytest.raises(
        composer.CandidateSourcePublicationIndeterminateError,
        match="clean rollback cannot be proved",
    ):
        composer.compose_from_request(
            request_path=tmp_path / "request.json",
            request_sha256="f" * 64,
            output_directory=destination,
        )
    assert destination.is_dir()
    assert relocated.is_dir()


def test_exclusive_rename_rejects_staging_name_substitution(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    parent = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    source = ".source"
    alternate = ".source-original"
    try:
        os.mkdir(source, 0o700, dir_fd=parent)
        metadata = os.stat(source, dir_fd=parent, follow_symlinks=False)
        expected_inode = (metadata.st_dev, metadata.st_ino)
        os.rename(source, alternate, src_dir_fd=parent, dst_dir_fd=parent)
        os.mkdir(source, 0o700, dir_fd=parent)
        with pytest.raises(composer.CandidateSourceComposerError, match="substituted"):
            composer._rename_noreplace(
                parent,
                source,
                "destination",
                expected_inode=expected_inode,
            )
        with pytest.raises(FileNotFoundError):
            os.stat("destination", dir_fd=parent, follow_symlinks=False)
    finally:
        os.close(parent)


def test_verifier_rejects_package_path_relocation_during_authority_reproduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    request = _request(tmp_path)
    source: dict[str, object] = {"artifacts": []}
    captured = SimpleNamespace(
        request=request,
        request_path=tmp_path / "request.json",
        request_bytes=request.canonical_file_bytes(),
        encoded_inputs={},
        capture_set_sha256="c" * 64,
        assert_current=lambda: None,
        close=lambda: None,
    )
    monkeypatch.setattr(
        composer,
        "_capture_input_set",
        lambda path, digest: captured,
    )
    monkeypatch.setattr(
        composer,
        "_derive_candidate_source",
        lambda value, captured=None: source,
    )
    destination = tmp_path / "candidate-source-package"
    relocated = tmp_path / ".relocated-candidate-source-package"
    composer.compose_from_request(
        request_path=tmp_path / "request.json",
        request_sha256="f" * 64,
        output_directory=destination,
    )

    def relocate_during_reproduction(
        value: composer.CompositionRequest,
        captured: object = None,
    ) -> dict[str, object]:
        del captured
        destination.rename(relocated)
        return source

    monkeypatch.setattr(
        composer,
        "_derive_candidate_source",
        relocate_during_reproduction,
    )
    with pytest.raises(
        composer.CandidateSourceComposerError,
        match="cannot open composition package identity readback",
    ):
        composer.verify_composed_package(destination)

    assert not destination.exists()
    assert relocated.is_dir()


def test_cli_rejects_abbreviations_and_manifest_overrides() -> None:
    parser = composer._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "compose",
                "--req",
                "/tmp/request.json",
                "--request-sha256",
                "a" * 64,
                "--output-directory",
                "/tmp/output",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "compose",
                "--request",
                "/tmp/request.json",
                "--request-sha256",
                "a" * 64,
                "--output-directory",
                "/tmp/output",
                "--custodian",
                "operator@example.test",
            ]
        )


def test_cli_formats_imported_domain_failures_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        composer,
        "compose_from_request",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("typed producer rejected")),
    )
    assert (
        composer.main(
            [
                "compose",
                "--request",
                "/tmp/request.json",
                "--request-sha256",
                "a" * 64,
                "--output-directory",
                "/tmp/output",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ("candidate source composition failed: typed producer rejected\n")


def test_cli_distinguishes_indeterminate_publication(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        composer,
        "compose_from_request",
        lambda **kwargs: (_ for _ in ()).throw(
            composer.CandidateSourcePublicationIndeterminateError("cannot prove rollback")
        ),
    )
    assert (
        composer.main(
            [
                "compose",
                "--request",
                "/tmp/request.json",
                "--request-sha256",
                "a" * 64,
                "--output-directory",
                "/tmp/output",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ("candidate source publication indeterminate: cannot prove rollback\n")


def test_cli_reports_clean_signal_interruption_with_shell_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        composer,
        "compose_from_request",
        lambda **kwargs: (_ for _ in ()).throw(
            composer.CandidateSourceInterruptedError(signal.SIGTERM)
        ),
    )
    assert (
        composer.main(
            [
                "compose",
                "--request",
                "/tmp/request.json",
                "--request-sha256",
                "a" * 64,
                "--output-directory",
                "/tmp/output",
            ]
        )
        == 128 + signal.SIGTERM
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ("candidate source composition interrupted: received signal SIGTERM\n")
