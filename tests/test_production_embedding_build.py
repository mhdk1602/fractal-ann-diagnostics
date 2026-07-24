from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fractal_ann_diagnostics.production_embedding_build as production
from fractal_ann_diagnostics.embedding_store import (
    EmbeddingStoreReceipt,
    RowOrderDescriptor,
    VectorDescriptor,
)
from fractal_ann_diagnostics.joint_power_design import FIXED_CORPORA
from fractal_ann_diagnostics.production_embedding_build import (
    PRODUCTION_EMBEDDING_BUILDER_KIND,
    PRODUCTION_EMBEDDING_BUILDER_PLATFORM,
    PRODUCTION_EMBEDDING_DEVICE,
    PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY,
    ImportedModuleOrigin,
    InstalledDistribution,
    ProductionEmbeddingBuilderReceipt,
    ProductionEmbeddingBuildError,
    ProductionEmbeddingConfig,
    ProductionEmbeddingProbeReceipt,
    PythonImportRoot,
    aggregate_production_embedding_shards,
    build_production_embedding_shard,
    build_production_embedding_suite,
    load_production_embedding_builder_receipt,
    load_production_embedding_config,
    production_embedding_status,
    verify_production_embedding_builder_runtime,
    verify_production_embedding_suite,
    write_production_embedding_builder_receipt,
    write_production_embedding_config,
)
from fractal_ann_diagnostics.qwen_revision_encoder import (
    QWEN_CURRENT_REVISION,
    QWEN_CURRENT_TREE_SHA256,
    QWEN_DOCUMENT_PROMPT,
    QWEN_OUTPUT_DIMENSION,
    QWEN_QUERY_PROMPT,
    QWEN_STALE_REVISION,
    QWEN_STALE_TREE_SHA256,
    QwenPairedRevisionEmbeddingAdapter,
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
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _builder_receipt(
    tmp_path: Path,
    *,
    current_model: Path,
    stale_model: Path,
) -> tuple[ProductionEmbeddingBuilderReceipt, Path]:
    repository = (tmp_path / "repository").resolve()
    repository.mkdir()
    uv_lock = repository / "uv.lock"
    uv_lock.write_bytes(b"version = 1\n")
    python = (tmp_path / "python3.12").resolve()
    python.write_bytes(b"fixed-python")
    prefix = (tmp_path / "builder-venv").resolve()
    prefix.mkdir()
    base_prefix = (tmp_path / "python-base").resolve()
    base_stdlib = base_prefix / "lib" / "python3.12"
    base_stdlib.mkdir(parents=True)
    site_packages = prefix / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    source_root = repository / "src" / "fractal_ann_diagnostics"
    source_root.mkdir(parents=True)
    (source_root / "__init__.py").write_bytes(b'__version__ = "0.3.0"\n')
    distributions = tuple(
        InstalledDistribution(name, version)
        for name, version in (
            ("fractal-ann-diagnostics", "0.3.0"),
            ("numpy", "2.5.1"),
            ("tokenizers", "0.22.2"),
            ("torch", "2.13.0"),
            ("transformers", "5.13.1"),
        )
    )
    current = production.QwenRevisionEncoderConfig.for_arm(
        "current", batch_size=64, device="mps", deterministic_seed=20260714
    )
    stale = production.QwenRevisionEncoderConfig.for_arm(
        "stale", batch_size=64, device="mps", deterministic_seed=20260714
    )
    probe = ProductionEmbeddingProbeReceipt(
        texts_sha256=_digest(_canonical({"texts": list(production._PROBE_TEXTS)})),
        row_count=len(production._PROBE_TEXTS),
        current_encoder_config_sha256=current.sha256,
        stale_encoder_config_sha256=stale.sha256,
        current_vectors_sha256=_digest("current probe vectors"),
        stale_vectors_sha256=_digest("stale probe vectors"),
        output_dimension=256,
        repeat_exact=True,
        first_elapsed_monotonic_ns=10,
        second_elapsed_monotonic_ns=9,
    )
    receipt = ProductionEmbeddingBuilderReceipt(
        repository_root=repository,
        source_commit="a" * 40,
        builder_source_sha256=_digest("builder source"),
        builder_source_file_count=19,
        uv_lock_path=uv_lock,
        uv_lock_sha256=_digest(uv_lock.read_bytes()),
        git_executable=Path("/usr/bin/git"),
        git_executable_sha256=_digest("system git"),
        python_executable=python,
        python_executable_sha256=_digest(python.read_bytes()),
        python_prefix=prefix,
        python_prefix_configuration_sha256=_digest("pyvenv.cfg"),
        python_base_prefix=base_prefix,
        python_version="3.12.13",
        python_safe_path=True,
        python_dont_write_bytecode=True,
        python_user_site_enabled=False,
        python_sys_path=(source_root.parent, base_stdlib, site_packages),
        python_import_roots=(
            PythonImportRoot(
                path=base_stdlib,
                kind="directory",
                sha256=_digest("base stdlib"),
                file_count=10,
                directory_count=2,
                byte_count=100,
            ),
        ),
        site_packages_root=site_packages,
        site_packages_tree_sha256=_digest("site packages"),
        site_packages_file_count=100,
        site_packages_directory_count=20,
        site_packages_byte_count=1_000,
        imported_module_origins=tuple(
            sorted(
                (
                    ImportedModuleOrigin("fractal_ann_diagnostics", source_root / "__init__.py"),
                    ImportedModuleOrigin("numpy", site_packages / "numpy" / "__init__.py"),
                    ImportedModuleOrigin(
                        "tokenizers", site_packages / "tokenizers" / "__init__.py"
                    ),
                    ImportedModuleOrigin("torch", site_packages / "torch" / "__init__.py"),
                    ImportedModuleOrigin(
                        "transformers", site_packages / "transformers" / "__init__.py"
                    ),
                )
            )
        ),
        process_environment=production._current_builder_environment(),
        installed_distributions=distributions,
        installed_distributions_sha256=_digest(
            _canonical([row.to_dict() for row in distributions])
        ),
        project_version="0.3.0",
        torch_version="2.13.0",
        transformers_version="5.13.1",
        numpy_version="2.5.1",
        tokenizers_version="0.22.2",
        macos_version="26.3.1",
        macos_build="25D771280a",
        platform=PRODUCTION_EMBEDDING_BUILDER_PLATFORM,
        machine="arm64",
        model="Mac16,6",
        chip="Apple M4 Max",
        logical_cores=14,
        memory_bytes=36 * 1024**3,
        mps_built=True,
        mps_available=True,
        builder_kind=PRODUCTION_EMBEDDING_BUILDER_KIND,
        device=PRODUCTION_EMBEDDING_DEVICE,
        batch_size=64,
        deterministic_seed=20260714,
        current_model_root=current_model,
        stale_model_root=stale_model,
        current_model_tree_sha256=QWEN_CURRENT_TREE_SHA256,
        stale_model_tree_sha256=QWEN_STALE_TREE_SHA256,
        current_encoder_config_sha256=current.sha256,
        stale_encoder_config_sha256=stale.sha256,
        probe=probe,
    )
    os.chmod(current_model, 0o555)
    os.chmod(stale_model, 0o555)
    path = (tmp_path / "builder-receipt.json").resolve()
    path.write_bytes(receipt.canonical_file_bytes())
    return receipt, path


def _artifact(
    path: str,
    payload: bytes,
    *,
    corpus_id: str,
    role: str,
    stage: str | None,
) -> dict[str, object]:
    return {
        "byte_count": len(payload),
        "dataset": corpus_id,
        "path": path,
        "record_count": payload.count(b"\n"),
        "role": role,
        "sha256": _digest(payload),
        "stage": stage,
        "visibility": "online",
    }


def _projection(tmp_path: Path) -> tuple[Path, str, str]:
    root = (tmp_path / "online").resolve()
    artifacts: list[dict[str, object]] = []
    for corpus in FIXED_CORPORA:
        document_path = f"datasets/{corpus}/corpus/corpus.jsonl"
        document_bytes = (
            _canonical({"id": f"{corpus}-d", "text": "document", "title": "title"}) + b"\n"
        )
        target = root / document_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(document_bytes)
        artifacts.append(
            _artifact(
                document_path,
                document_bytes,
                corpus_id=corpus,
                role="corpus",
                stage=None,
            )
        )
        for stage in ("fit", "calibration", "sealed"):
            query_path = f"datasets/{corpus}/{stage}/online/queries.jsonl"
            query_bytes = _canonical({"id": f"{corpus}-{stage}-q", "text": "query"}) + b"\n"
            target = root / query_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(query_bytes)
            artifacts.append(
                _artifact(
                    query_path,
                    query_bytes,
                    corpus_id=corpus,
                    role="queries",
                    stage=stage,
                )
            )
    artifacts.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    inventory_bytes = _canonical({"artifacts": artifacts}) + b"\n"
    (root / "inventory.json").write_bytes(inventory_bytes)
    return root, _digest(inventory_bytes), _digest("projected-artifact-set")


def _write_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ProductionEmbeddingConfig, Path]:
    staging, inventory_sha256, projected_sha256 = _projection(tmp_path)
    current_model = (tmp_path / "qwen-current").resolve()
    stale_model = (tmp_path / "qwen-stale").resolve()
    current_model.mkdir()
    stale_model.mkdir()
    builder, builder_path = _builder_receipt(
        tmp_path,
        current_model=current_model,
        stale_model=stale_model,
    )
    output_root = (tmp_path / "embeddings").resolve()
    config_path = tmp_path / "production-embeddings.json"

    def verify_projection(
        root: Path,
        *,
        expected_inventory_sha256: str,
    ) -> SimpleNamespace:
        assert root == staging
        assert expected_inventory_sha256 == inventory_sha256
        return SimpleNamespace(
            inventory_sha256=inventory_sha256,
            projected_artifact_set_sha256=projected_sha256,
        )

    monkeypatch.setattr(production, "verify_online_staging_projection", verify_projection)
    monkeypatch.setattr(production, "verify_qwen_revision_tree", lambda *_args: None)
    monkeypatch.setattr(
        production,
        "verify_production_embedding_builder_runtime",
        lambda *_args: None,
    )
    config = write_production_embedding_config(
        online_staging_root=staging,
        expected_inventory_sha256=inventory_sha256,
        builder_receipt_path=builder_path,
        expected_builder_receipt_sha256=builder.file_sha256,
        current_model_root=current_model,
        stale_model_root=stale_model,
        output_root=output_root,
        batch_size=64,
        device="mps",
        deterministic_seed=20260714,
        output_dtype="float32",
        destination=config_path,
    )
    return config, config_path


def test_config_derives_all_source_paths_and_is_hash_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, path = _write_config(tmp_path, monkeypatch)

    assert tuple(row.corpus_id for row in config.corpora) == FIXED_CORPORA
    assert all(len(row.document_paths) == 1 for row in config.corpora)
    assert all(len(row.query_paths) == 3 for row in config.corpora)
    assert load_production_embedding_config(path, expected_sha256=config.file_sha256) == config
    with pytest.raises(ProductionEmbeddingBuildError, match="caller pin"):
        load_production_embedding_config(path, expected_sha256=_digest("substituted"))

    value = json.loads(path.read_text())
    value["document_paths"] = ["caller/supplied.jsonl"]
    substituted = tmp_path / "substituted.json"
    encoded = _canonical(value) + b"\n"
    substituted.write_bytes(encoded)
    with pytest.raises(ProductionEmbeddingBuildError, match="fields differ"):
        load_production_embedding_config(substituted, expected_sha256=_digest(encoded))


def test_builder_receipt_is_canonical_hash_pinned_and_runtime_revalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = (tmp_path / "current").resolve()
    stale = (tmp_path / "stale").resolve()
    current.mkdir()
    stale.mkdir()
    receipt, path = _builder_receipt(tmp_path, current_model=current, stale_model=stale)
    assert (
        load_production_embedding_builder_receipt(
            path,
            expected_sha256=receipt.file_sha256,
        )
        == receipt
    )
    with pytest.raises(ProductionEmbeddingBuildError, match="caller pin"):
        load_production_embedding_builder_receipt(
            path,
            expected_sha256=_digest("substituted builder receipt"),
        )

    observed = {key: getattr(receipt, key) for key in production._BUILDER_OBSERVATION_FIELDS}
    monkeypatch.setattr(
        production,
        "_observe_mps_builder_environment",
        lambda **_kwargs: observed,
    )
    monkeypatch.setattr(production, "verify_qwen_revision_tree", lambda *_args: None)
    monkeypatch.setattr(
        production,
        "_execute_fixed_mps_probe",
        lambda **_kwargs: receipt.probe,
    )
    verify_production_embedding_builder_runtime(receipt)

    drifted = {**observed, "macos_build": "25D-drifted"}
    monkeypatch.setattr(
        production,
        "_observe_mps_builder_environment",
        lambda **_kwargs: drifted,
    )
    with pytest.raises(ProductionEmbeddingBuildError, match="macos_build"):
        verify_production_embedding_builder_runtime(receipt)

    swapped = {**observed, "python_executable": (tmp_path / "path-swapped-python").resolve()}
    monkeypatch.setattr(
        production,
        "_observe_mps_builder_environment",
        lambda **_kwargs: swapped,
    )
    with pytest.raises(ProductionEmbeddingBuildError, match="python_executable"):
        verify_production_embedding_builder_runtime(receipt)

    monkeypatch.setattr(
        production,
        "_observe_mps_builder_environment",
        lambda **_kwargs: observed,
    )
    monkeypatch.setattr(
        production,
        "_execute_fixed_mps_probe",
        lambda **_kwargs: replace(
            receipt.probe,
            current_vectors_sha256=_digest("drifted MPS kernel vectors"),
        ),
    )
    with pytest.raises(ProductionEmbeddingBuildError, match="fixed-probe vectors"):
        verify_production_embedding_builder_runtime(receipt)


def test_builder_receipt_writer_observes_identity_around_fixed_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = (tmp_path / "current").resolve()
    stale = (tmp_path / "stale").resolve()
    current.mkdir()
    stale.mkdir()
    expected, _path = _builder_receipt(tmp_path, current_model=current, stale_model=stale)
    observed = {key: getattr(expected, key) for key in production._BUILDER_OBSERVATION_FIELDS}
    observations: list[dict[str, object]] = []

    def observe(**_kwargs: object) -> dict[str, object]:
        observations.append(observed)
        return observed

    monkeypatch.setattr(production, "_observe_mps_builder_environment", observe)
    monkeypatch.setattr(
        production,
        "_execute_fixed_mps_probe",
        lambda **_kwargs: expected.probe,
    )
    monkeypatch.setattr(production, "verify_qwen_revision_tree", lambda *_args: None)
    destination = (tmp_path / "control" / "builder.json").resolve()
    destination.parent.mkdir()
    receipt = write_production_embedding_builder_receipt(
        repository_root=expected.repository_root,
        expected_source_commit=expected.source_commit,
        uv_lock_path=expected.uv_lock_path,
        current_model_root=current,
        stale_model_root=stale,
        batch_size=64,
        deterministic_seed=20260714,
        destination=destination,
    )
    assert receipt == expected
    assert destination.read_bytes() == receipt.canonical_file_bytes()
    assert len(observations) == 2


def test_builder_source_identity_rehashes_a_clean_exact_git_tree(tmp_path: Path) -> None:
    repository = (tmp_path / "source").resolve()
    package = repository / "src" / "fractal_ann_diagnostics"
    package.mkdir(parents=True)
    (repository / "pyproject.toml").write_bytes(b"[project]\nname = 'probe'\n")
    (repository / "uv.lock").write_bytes(b"version = 1\n")
    (repository / ".gitignore").write_bytes(b"src/sitecustomize.py\n")
    (package / "__init__.py").write_bytes(b'IDENTITY = "fixed"\n')
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.name", "mhdk1602"),
        check=True,
    )
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "config",
            "user.email",
            "mhdk1602@users.noreply.github.com",
        ),
        check=True,
    )
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-q", "-m", "fixed builder source"),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for path in (repository, repository / "src", package):
        os.chmod(path, 0o555)
    for path in (
        repository / "pyproject.toml",
        repository / "uv.lock",
        repository / ".gitignore",
        package / "__init__.py",
    ):
        os.chmod(path, 0o444)
    source_sha256, file_count = production._git_source_identity(
        repository,
        expected_commit=commit,
    )
    assert len(source_sha256) == 64
    assert file_count == 3

    os.chmod(repository / "src", 0o755)
    ignored_shadow = repository / "src" / "sitecustomize.py"
    ignored_shadow.write_bytes(b"raise RuntimeError('shadowed')\n")
    os.chmod(ignored_shadow, 0o444)
    os.chmod(repository / "src", 0o555)
    with pytest.raises(ProductionEmbeddingBuildError, match="ignored or untracked"):
        production._git_source_identity(repository, expected_commit=commit)
    os.chmod(repository / "src", 0o755)
    ignored_shadow.unlink()
    os.chmod(repository / "src", 0o555)

    os.chmod(package / "__init__.py", 0o644)
    (package / "__init__.py").write_bytes(b'IDENTITY = "drifted"\n')
    os.chmod(package / "__init__.py", 0o444)
    with pytest.raises(ProductionEmbeddingBuildError, match="checkout must be clean"):
        production._git_source_identity(repository, expected_commit=commit)


def test_site_packages_inventory_rehashes_code_metadata_pth_and_native_bytes(
    tmp_path: Path,
) -> None:
    site_packages = (tmp_path / "venv" / "lib" / "python3.12" / "site-packages").resolve()
    dist_info = site_packages / "probe-1.0.dist-info"
    package = site_packages / "probe"
    dist_info.mkdir(parents=True)
    package.mkdir()
    pth = site_packages / "_editable_probe.pth"
    native = package / "kernel.cpython-312-darwin.so"
    for path, payload in (
        (pth, b"/pinned/source\n"),
        (dist_info / "METADATA", b"Name: probe\nVersion: 1.0\n"),
        (package / "__init__.py", b"VALUE = 1\n"),
        (native, b"fixed-native-bytes"),
    ):
        path.write_bytes(payload)
        os.chmod(path, 0o444)
    for path in (package, dist_info, site_packages):
        os.chmod(path, 0o555)

    first = production._site_packages_tree_identity(site_packages)
    os.chmod(pth, 0o644)
    pth.write_bytes(b"/substituted/source\n")
    os.chmod(pth, 0o444)
    second = production._site_packages_tree_identity(site_packages)
    assert second["site_packages_tree_sha256"] != first["site_packages_tree_sha256"]

    os.chmod(package, 0o755)
    cache = package / "probe.pyc"
    cache.write_bytes(b"unregistered cache")
    os.chmod(cache, 0o444)
    os.chmod(package, 0o555)
    with pytest.raises(ProductionEmbeddingBuildError, match="bytecode cache"):
        production._site_packages_tree_identity(site_packages)


def test_python_import_root_records_only_the_interpreter_absent_zip_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = (tmp_path / "python-base").resolve()
    (base / "lib").mkdir(parents=True)
    monkeypatch.setattr(production.sys, "version_info", SimpleNamespace(major=3, minor=12))
    expected = base / "lib" / "python312.zip"
    absent = production._python_import_root_identity(expected, python_base_prefix=base)
    assert absent.kind == "absent"
    with pytest.raises(ProductionEmbeddingBuildError, match="only the interpreter-derived"):
        production._python_import_root_identity(
            base / "lib" / "caller-controlled.zip",
            python_base_prefix=base,
        )

    stdlib = base / "lib" / "python3.12"
    native_root = stdlib / "lib-dynload"
    native_root.mkdir(parents=True)
    module = stdlib / "hashlib.py"
    native = native_root / "_hashlib.cpython-312-darwin.so"
    module.write_bytes(b"PINNED = 1\n")
    native.write_bytes(b"pinned native extension")
    for path in (module, native):
        os.chmod(path, 0o444)
    for path in (native_root, stdlib):
        os.chmod(path, 0o555)
    first = production._python_import_root_identity(stdlib, python_base_prefix=base)
    os.chmod(module, 0o644)
    module.write_bytes(b"PINNED = 2\n")
    os.chmod(module, 0o444)
    second = production._python_import_root_identity(stdlib, python_base_prefix=base)
    assert first.sha256 != second.sha256


def test_cli_requires_builder_receipt_and_pin_for_config() -> None:
    parser = production._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "write-config",
                "--online-staging-root",
                "/online",
                "--expected-inventory-sha256",
                "a" * 64,
                "--current-model-root",
                "/current",
                "--stale-model-root",
                "/stale",
                "--output-root",
                "/output",
                "--batch-size",
                "64",
                "--device",
                "mps",
                "--seed",
                "20260714",
                "--output",
                "/control/config.json",
            ]
        )

    arguments = parser.parse_args(
        [
            "write-config",
            "--online-staging-root",
            "/online",
            "--expected-inventory-sha256",
            "a" * 64,
            "--builder-receipt",
            "/control/builder.json",
            "--builder-receipt-sha256",
            "b" * 64,
            "--current-model-root",
            "/current",
            "--stale-model-root",
            "/stale",
            "--output-root",
            "/output",
            "--batch-size",
            "64",
            "--device",
            "mps",
            "--seed",
            "20260714",
            "--output",
            "/control/config.json",
        ]
    )
    assert arguments.builder_receipt == Path("/control/builder.json")
    assert arguments.builder_receipt_sha256 == "b" * 64


def test_config_rejects_path_overlap_and_incomplete_query_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    value = config.to_dict()
    value["output_root"] = str(config.online_staging_root / "derived")
    with pytest.raises(ProductionEmbeddingBuildError, match="cannot overlap"):
        ProductionEmbeddingConfig.from_dict(value)

    artifacts = list(production._load_admitted_inventory(config.online_staging_root))
    artifacts = [
        row
        for row in artifacts
        if not (
            row["dataset"] == "scifact" and row["role"] == "queries" and row["stage"] == "sealed"
        )
    ]
    with pytest.raises(ProductionEmbeddingBuildError, match="do not cover"):
        production._derive_corpus_sources(artifacts)


def test_config_rejects_builder_receipt_substitution_and_movable_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    candidate = config.to_dict()
    candidate["builder_receipt_sha256"] = _digest("substituted receipt")
    with pytest.raises(ProductionEmbeddingBuildError, match="receipt digest differs"):
        ProductionEmbeddingConfig.from_dict(candidate)

    candidate = config.to_dict()
    receipt = dict(candidate["builder_receipt"])
    receipt["macos_build"] = "25D-substituted"
    candidate["builder_receipt"] = receipt
    with pytest.raises(ProductionEmbeddingBuildError, match="receipt digest differs"):
        ProductionEmbeddingConfig.from_dict(candidate)

    candidate = config.to_dict()
    candidate["builder_image"] = "ghcr.io/caller/movable:latest"
    with pytest.raises(ProductionEmbeddingBuildError, match="fields differ"):
        ProductionEmbeddingConfig.from_dict(candidate)

    candidate = config.to_dict()
    receipt = dict(candidate["builder_receipt"])
    environment = dict(receipt["process_environment"])
    environment["OMP_NUM_THREADS"] = "8"
    receipt["process_environment"] = environment
    candidate["builder_receipt"] = receipt
    with pytest.raises(ProductionEmbeddingBuildError, match="fixed minimal environment"):
        ProductionEmbeddingConfig.from_dict(candidate)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("platform", "linux/arm64", "Darwin arm64"),
        ("machine", "x86_64", "Darwin arm64"),
        ("mps_available", False, "expose MPS"),
        ("mps_built", False, "include MPS"),
    ),
)
def test_config_rejects_non_mps_builder_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    candidate = config.to_dict()
    receipt = dict(candidate["builder_receipt"])
    receipt[field] = value
    candidate["builder_receipt"] = receipt
    with pytest.raises(ProductionEmbeddingBuildError, match=message):
        ProductionEmbeddingConfig.from_dict(candidate)


@pytest.mark.parametrize("device", ("cpu", "cuda", "cuda:0"))
def test_config_rejects_non_mps_builder_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    candidate = config.to_dict()
    current = dict(candidate["current_encoder_config"])
    stale = dict(candidate["stale_encoder_config"])
    current["device"] = device
    stale["device"] = device
    candidate["current_encoder_config"] = current
    candidate["stale_encoder_config"] = stale
    with pytest.raises(ProductionEmbeddingBuildError, match="device 'mps'"):
        ProductionEmbeddingConfig.from_dict(candidate)


@pytest.mark.parametrize("missing", ("builder_receipt", "builder_receipt_sha256"))
def test_config_rejects_missing_builder_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    candidate = config.to_dict()
    del candidate[missing]
    with pytest.raises(ProductionEmbeddingBuildError, match="fields differ"):
        ProductionEmbeddingConfig.from_dict(candidate)


def test_status_distinguishes_pending_resume_and_impossible_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    assert production_embedding_status(config)["status"] == "pending"

    config.output_root.mkdir()
    (config.output_root / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY).mkdir()
    partial, checkpoint = production._partial_paths(config.output_root, "scifact")
    partial.mkdir()
    with pytest.raises(ProductionEmbeddingBuildError, match="must appear together"):
        production_embedding_status(config)
    checkpoint.write_bytes(b"{}\n")
    status = production_embedding_status(config)
    assert status["status"] == "resumable"
    assert status["corpora"][0] == {"corpus_id": "scifact", "status": "resumable"}


def _fake_receipt(
    *,
    selection: Any,
    config: Any,
    current_model: Any,
    old_model: Any,
    paired_encoder: Any,
) -> EmbeddingStoreReceipt:
    resolved = production.embedding_store_module._load_sources(selection)
    document_order = _digest(f"{selection.document_paths}:documents")
    query_order = _digest(f"{selection.query_paths}:queries")
    row_orders = {
        "documents": RowOrderDescriptor(
            relative_path="document-rows.jsonl",
            row_count=resolved.document_count,
            byte_count=1,
            row_order_sha256=document_order,
            file_sha256=document_order,
        ),
        "queries": RowOrderDescriptor(
            relative_path="query-rows.jsonl",
            row_count=resolved.query_count,
            byte_count=1,
            row_order_sha256=query_order,
            file_sha256=query_order,
        ),
    }

    def vector(name: str, *, current: bool, documents: bool) -> VectorDescriptor:
        model = current_model if current else old_model
        prompt = QWEN_DOCUMENT_PROMPT if documents else QWEN_QUERY_PROMPT
        rows = resolved.document_count if documents else resolved.query_count
        return VectorDescriptor(
            relative_path=f"{name.replace('_', '-')}.npy",
            dtype="float32",
            shape=(rows, QWEN_OUTPUT_DIMENSION),
            row_order_sha256=document_order if documents else query_order,
            byte_count=rows * QWEN_OUTPUT_DIMENSION * 4,
            file_sha256=_digest(name),
            model_tree_sha256=model.tree_sha256,
            model_revision=model.revision,
            prompt_sha256=_digest(prompt),
        )

    vectors = {
        "current_documents": vector("current_documents", current=True, documents=True),
        "current_queries": vector("current_queries", current=True, documents=False),
        "old_documents": vector("old_documents", current=False, documents=True),
        "old_queries": vector("old_queries", current=False, documents=False),
    }
    assert current_model.revision == QWEN_CURRENT_REVISION
    assert current_model.tree_sha256 == QWEN_CURRENT_TREE_SHA256
    assert old_model.revision == QWEN_STALE_REVISION
    assert old_model.tree_sha256 == QWEN_STALE_TREE_SHA256
    return EmbeddingStoreReceipt(
        staged_inventory_sha256=selection.inventory_sha256,
        source_inventory_sha256=resolved.source_inventory_sha256,
        config_sha256=config.sha256,
        document_count=resolved.document_count,
        query_count=resolved.query_count,
        current_model=current_model.binding(encoder_id=paired_encoder.current_implementation_id),
        old_model=old_model.binding(encoder_id=paired_encoder.old_implementation_id),
        row_orders=row_orders,
        vectors=vectors,
    )


def test_five_corpus_build_resumes_and_suite_verification_rejects_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    config.output_root.mkdir()
    (config.output_root / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY).mkdir()
    partial, checkpoint = production._partial_paths(config.output_root, "scifact")
    partial.mkdir()
    checkpoint.write_bytes(b"{}\n")
    receipts: dict[str, EmbeddingStoreReceipt] = {}

    def fake_build(
        selection: Any,
        output_root: Path,
        **kwargs: Any,
    ) -> EmbeddingStoreReceipt:
        partial_root, checkpoint_path = production._partial_paths(
            output_root.parent, output_root.name
        )
        if partial_root.exists():
            partial_root.rmdir()
            checkpoint_path.unlink()
        output_root.mkdir()
        (output_root / "store-payload.bin").write_bytes(output_root.name.encode())
        receipt = _fake_receipt(selection=selection, **kwargs)
        receipts[output_root.name] = receipt
        return receipt

    def fake_verify(root: Path) -> EmbeddingStoreReceipt:
        return receipts[root.name]

    monkeypatch.setattr(production, "build_embedding_store", fake_build)
    monkeypatch.setattr(production, "verify_embedding_store", fake_verify)
    adapter = QwenPairedRevisionEmbeddingAdapter(
        config.current_encoder_config,
        config.stale_encoder_config,
    )
    suite = build_production_embedding_suite(config, paired_encoder=adapter)
    assert tuple(row.corpus_id for row in suite.corpora) == FIXED_CORPORA
    assert production_embedding_status(config)["status"] == "complete"
    assert verify_production_embedding_suite(config) == suite

    evidence_path = config.output_root / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY / "scifact.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["status"] == "resumed"
    evidence["embedding_tree_sha256"] = _digest("substituted-tree")
    evidence_path.write_bytes(_canonical(evidence) + b"\n")
    with pytest.raises(ProductionEmbeddingBuildError, match="differs from the final store"):
        verify_production_embedding_suite(config)


def _install_fake_store_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, EmbeddingStoreReceipt]:
    receipts: dict[str, EmbeddingStoreReceipt] = {}

    def fake_build(
        selection: Any,
        output_root: Path,
        **kwargs: Any,
    ) -> EmbeddingStoreReceipt:
        partial_root, checkpoint_path = production._partial_paths(
            output_root.parent, output_root.name
        )
        if partial_root.exists():
            partial_root.rmdir()
            checkpoint_path.unlink()
        output_root.mkdir()
        (output_root / "store-payload.bin").write_bytes(output_root.name.encode())
        receipt = _fake_receipt(selection=selection, **kwargs)
        receipts[output_root.name] = receipt
        return receipt

    def fake_verify(root: Path) -> EmbeddingStoreReceipt:
        return receipts[root.name]

    monkeypatch.setattr(production, "build_embedding_store", fake_build)
    monkeypatch.setattr(production, "verify_embedding_store", fake_verify)
    return receipts


def test_reverse_order_shards_aggregate_to_exact_monolithic_derivation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    receipts = _install_fake_store_builder(monkeypatch)
    monkeypatch.setattr(production, "_utc_now", lambda: "2026-07-16T12:00:00Z")
    monkeypatch.setattr(production.time, "monotonic_ns", lambda: 100)
    monkeypatch.setattr(production, "_peak_rss_bytes", lambda: 4096)
    adapter = QwenPairedRevisionEmbeddingAdapter(
        config.current_encoder_config,
        config.stale_encoder_config,
    )

    shard_receipts = [
        build_production_embedding_shard(config, corpus, paired_encoder=adapter)
        for corpus in reversed(FIXED_CORPORA)
    ]
    assert tuple(item.corpus_id for item in shard_receipts) == tuple(reversed(FIXED_CORPORA))
    sharded = aggregate_production_embedding_shards(config)
    assert tuple(row.corpus_id for row in sharded.corpora) == FIXED_CORPORA
    sharded_bytes = sharded.canonical_file_bytes()
    sharded_store_bindings = tuple(
        (row.embedding_receipt_sha256, row.embedding_tree_sha256) for row in sharded.corpora
    )

    shutil.rmtree(config.output_root)
    receipts.clear()
    monolithic = build_production_embedding_suite(config, paired_encoder=adapter)
    assert monolithic.canonical_file_bytes() == sharded_bytes
    assert (
        tuple(
            (row.embedding_receipt_sha256, row.embedding_tree_sha256) for row in monolithic.corpora
        )
        == sharded_store_bindings
    )
    assert verify_production_embedding_suite(config) == monolithic


def test_every_production_entrypoint_readmits_builder_and_model_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    _install_fake_store_builder(monkeypatch)
    runtime_receipts: list[ProductionEmbeddingBuilderReceipt] = []
    model_receipts: list[ProductionEmbeddingBuilderReceipt] = []

    def verify_runtime(receipt: ProductionEmbeddingBuilderReceipt) -> None:
        runtime_receipts.append(receipt)

    def verify_models(observed: ProductionEmbeddingConfig) -> None:
        assert observed.current_model_root == observed.builder_receipt.current_model_root
        assert observed.stale_model_root == observed.builder_receipt.stale_model_root
        assert (
            observed.current_encoder_config.sha256
            == observed.builder_receipt.current_encoder_config_sha256
        )
        assert (
            observed.stale_encoder_config.sha256
            == observed.builder_receipt.stale_encoder_config_sha256
        )
        model_receipts.append(observed.builder_receipt)

    monkeypatch.setattr(production, "verify_production_embedding_builder_runtime", verify_runtime)
    monkeypatch.setattr(production, "_verify_model_roots", verify_models)
    adapter = QwenPairedRevisionEmbeddingAdapter(
        config.current_encoder_config,
        config.stale_encoder_config,
    )
    for corpus_id in FIXED_CORPORA:
        build_production_embedding_shard(config, corpus_id, paired_encoder=adapter)
    aggregate_production_embedding_shards(config)
    verify_production_embedding_suite(config)
    shutil.rmtree(config.output_root)
    build_production_embedding_suite(config, paired_encoder=adapter)

    assert runtime_receipts == [config.builder_receipt] * 8
    assert model_receipts == [config.builder_receipt] * 8


def test_shard_rechecks_runtime_inside_lock_before_forward_or_output_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    writes: list[Path] = []
    inside_lock = False

    @contextmanager
    def mutating_lock(_output_root: Path, _corpus_id: str) -> Any:
        nonlocal inside_lock
        inside_lock = True
        try:
            yield
        finally:
            inside_lock = False

    def reject_mutated_runtime(_receipt: ProductionEmbeddingBuilderReceipt) -> None:
        assert inside_lock
        raise ProductionEmbeddingBuildError("injected runtime mutation")

    def forbidden_build(_selection: Any, output_root: Path, **_kwargs: Any) -> Any:
        writes.append(output_root)
        raise AssertionError("model forward was reached after runtime mutation")

    monkeypatch.setattr(production, "_corpus_worker_lock", mutating_lock)
    monkeypatch.setattr(
        production,
        "verify_production_embedding_builder_runtime",
        reject_mutated_runtime,
    )
    monkeypatch.setattr(production, "build_embedding_store", forbidden_build)
    with pytest.raises(ProductionEmbeddingBuildError, match="injected runtime mutation"):
        build_production_embedding_shard(config, FIXED_CORPORA[0])
    assert writes == []
    assert not config.output_root.exists()


def test_shard_aggregation_rejects_missing_duplicate_extra_and_unfinished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    _install_fake_store_builder(monkeypatch)
    adapter = QwenPairedRevisionEmbeddingAdapter(
        config.current_encoder_config,
        config.stale_encoder_config,
    )
    first = build_production_embedding_shard(
        config,
        FIXED_CORPORA[0],
        paired_encoder=adapter,
    )
    with pytest.raises(ProductionEmbeddingBuildError, match="missing_stores"):
        aggregate_production_embedding_shards(config)

    row = config.corpora[0]
    receipt, _expected, tree_sha256 = production._verify_one_store(
        config,
        row,
        adapter=adapter,
    )
    completed = (row, receipt, tree_sha256, first)
    with pytest.raises(ProductionEmbeddingBuildError, match="repeat registered corpus"):
        production._ordered_completed_rows(config, [completed] * len(FIXED_CORPORA))
    with pytest.raises(ProductionEmbeddingBuildError, match="completed shard set differs"):
        production._ordered_completed_rows(config, [completed])

    (config.output_root / "unregistered-worker-output").write_bytes(b"forbidden")
    with pytest.raises(ProductionEmbeddingBuildError, match="unexpected"):
        aggregate_production_embedding_shards(config)
    (config.output_root / "unregistered-worker-output").unlink()

    partial, _checkpoint = production._partial_paths(config.output_root, FIXED_CORPORA[1])
    partial.mkdir()
    with pytest.raises(ProductionEmbeddingBuildError, match="must appear together"):
        aggregate_production_embedding_shards(config)


def test_distinct_shard_worker_ignores_only_peer_transient_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    _install_fake_store_builder(monkeypatch)
    config.output_root.mkdir()
    (config.output_root / PRODUCTION_EMBEDDING_EVIDENCE_DIRECTORY).mkdir()
    peer_partial, _peer_checkpoint = production._partial_paths(config.output_root, FIXED_CORPORA[0])
    peer_partial.mkdir()
    evidence = build_production_embedding_shard(config, FIXED_CORPORA[1])
    assert evidence.corpus_id == FIXED_CORPORA[1]
    with pytest.raises(ProductionEmbeddingBuildError, match="must appear together"):
        aggregate_production_embedding_shards(config)


def test_shard_selector_and_encoder_are_closed_to_registered_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    _install_fake_store_builder(monkeypatch)
    with pytest.raises(ProductionEmbeddingBuildError, match="one of FIXED_CORPORA"):
        build_production_embedding_shard(config, "caller-corpus")

    wrong_current = production.QwenRevisionEncoderConfig(
        **{
            **config.current_encoder_config.to_dict(),
            "deterministic_seed": config.current_encoder_config.deterministic_seed + 1,
        }
    )
    wrong_stale = production.QwenRevisionEncoderConfig(
        **{
            **config.stale_encoder_config.to_dict(),
            "deterministic_seed": config.stale_encoder_config.deterministic_seed + 1,
        }
    )
    wrong_adapter = QwenPairedRevisionEmbeddingAdapter(wrong_current, wrong_stale)
    with pytest.raises(ProductionEmbeddingBuildError, match="config-derived Qwen arms"):
        build_production_embedding_shard(
            config,
            FIXED_CORPORA[0],
            paired_encoder=wrong_adapter,
        )


def test_corpus_worker_lock_rejects_duplicate_and_hostile_inode(tmp_path: Path) -> None:
    output_root = (tmp_path / "embedding-stores").resolve()
    corpus_id = FIXED_CORPORA[0]
    with production._corpus_worker_lock(output_root, corpus_id):
        with pytest.raises(ProductionEmbeddingBuildError, match="active embedding worker"):
            with production._corpus_worker_lock(output_root, corpus_id):
                raise AssertionError("duplicate lock unexpectedly admitted")
        with production._corpus_worker_lock(output_root, FIXED_CORPORA[1]):
            pass
    with production._corpus_worker_lock(output_root, corpus_id):
        pass

    lock_root = production._worker_lock_root(output_root)
    lock_path = lock_root / f"{corpus_id}.lock"
    os.chmod(lock_root, 0o770)
    with pytest.raises(ProductionEmbeddingBuildError, match="mode-0700"):
        with production._corpus_worker_lock(output_root, corpus_id):
            raise AssertionError("group-writable lock root unexpectedly admitted")
    os.chmod(lock_root, 0o700)

    original_parent_mode = stat.S_IMODE(output_root.parent.stat().st_mode)
    os.chmod(output_root.parent, original_parent_mode | 0o020)
    with pytest.raises(ProductionEmbeddingBuildError, match="output parent"):
        with production._corpus_worker_lock(output_root, corpus_id):
            raise AssertionError("group-writable output parent unexpectedly admitted")
    os.chmod(output_root.parent, original_parent_mode)

    lock_path.unlink()
    hostile = tmp_path / "hostile-lock"
    hostile.write_bytes(b"")
    os.chmod(hostile, 0o600)
    os.link(hostile, lock_path)
    with pytest.raises(ProductionEmbeddingBuildError, match="private empty inode"):
        with production._corpus_worker_lock(output_root, corpus_id):
            raise AssertionError("hard-linked lock unexpectedly admitted")
    lock_path.unlink()
    hostile.unlink()

    lock_path.write_bytes(b"hostile lock content")
    os.chmod(lock_path, 0o600)
    with pytest.raises(ProductionEmbeddingBuildError, match="private empty inode"):
        with production._corpus_worker_lock(output_root, corpus_id):
            raise AssertionError("nonempty lock unexpectedly admitted")
    lock_path.unlink()

    with pytest.raises(ProductionEmbeddingBuildError, match="identity changed"):
        with production._corpus_worker_lock(output_root, corpus_id):
            lock_path.unlink()
            lock_path.write_bytes(b"")
            os.chmod(lock_path, 0o600)
    lock_path.unlink()

    with pytest.raises(ProductionEmbeddingBuildError, match="identity changed"):
        with production._corpus_worker_lock(output_root, corpus_id):
            lock_path.write_bytes(b"mutated while locked")


def test_shard_aggregate_and_verify_cannot_cross_a_live_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _path = _write_config(tmp_path, monkeypatch)
    _install_fake_store_builder(monkeypatch)
    adapter = QwenPairedRevisionEmbeddingAdapter(
        config.current_encoder_config,
        config.stale_encoder_config,
    )
    with production._corpus_worker_lock(config.output_root, FIXED_CORPORA[0]):
        with pytest.raises(ProductionEmbeddingBuildError, match="active embedding worker"):
            build_production_embedding_shard(
                config,
                FIXED_CORPORA[0],
                paired_encoder=adapter,
            )

    for corpus_id in FIXED_CORPORA:
        build_production_embedding_shard(config, corpus_id, paired_encoder=adapter)
    with production._corpus_worker_lock(config.output_root, FIXED_CORPORA[-1]):
        with pytest.raises(ProductionEmbeddingBuildError, match="active embedding worker"):
            aggregate_production_embedding_shards(config)
    aggregate_production_embedding_shards(config)
    with production._corpus_worker_lock(config.output_root, FIXED_CORPORA[2]):
        with pytest.raises(ProductionEmbeddingBuildError, match="active embedding worker"):
            verify_production_embedding_suite(config)
