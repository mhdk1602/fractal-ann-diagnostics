from __future__ import annotations

import json
from pathlib import Path

import pytest

from fractal_ann_diagnostics.freeze_package import (
    FREEZE_READINESS_SCHEMA,
    FreezePackageError,
    artifact_map_payload,
    compile_freeze_package,
    layout_from_manifest,
    main,
    validate_freeze_artifact_map,
)
from fractal_ann_diagnostics.study import load_study_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "research" / "study-manifest.json"


def _manifest() -> dict[str, object]:
    return load_study_manifest(MANIFEST_PATH)


def _write_map(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_layout_is_derived_from_every_manifest_artifact(tmp_path: Path) -> None:
    manifest = _manifest()
    layout = layout_from_manifest(manifest, REPOSITORY_ROOT)

    expected_ids = {str(row["id"]) for row in manifest["artifacts"]}  # type: ignore[index]
    assert len(layout) == len(expected_ids)
    assert len(layout) == 79
    assert {row.artifact_id for row in layout} == expected_ids
    assert len({row.relative_path for row in layout}) == len(layout)

    paths = {row.artifact_id: row.relative_path for row in layout}
    assert paths["scifact-sealed-label-ciphertext"].endswith(".tlock")
    assert paths["custody-seal-receipt"] == "custody/custody-seal-receipt.json"
    assert paths["tlock-release-provenance"] == "custody/tlock-release-provenance.json"
    assert paths["drand-timelock-tool"] == "custody/bin/tle"
    assert paths["registered-power-analysis-report"] == "analysis/joint-power-design"
    assert paths["complete-staged-study-data"] == "study-data/custody-complete"
    assert paths["label-free-online-staging-projection"] == "study-data/online-projection"
    assert paths["development-freeze-package"] == "development/freeze-package"
    assert paths["scifact-runtime-attestation-plan-template"] == (
        "runtime/scifact/runtime-attestation-plan.template.json"
    )
    assert paths["miracl-transfer-runtime-attestation-plan-template"] == (
        "runtime/miracl-transfer/runtime-attestation-plan.template.json"
    )
    assert paths["suite-attestation-descriptor"] == ("suite/suite-attestation-descriptor.json")
    assert paths["open-policy-agent-runtime-binary"] == "runtime/opa"
    assert (
        next(row.kind for row in layout if row.artifact_id == "registered-power-analysis-report")
        == "directory"
    )

    map_path = tmp_path / "artifact-map.json"
    _write_map(map_path, artifact_map_payload(layout))
    assert (
        validate_freeze_artifact_map(
            manifest,
            REPOSITORY_ROOT,
            map_path,
        )
        == layout
    )


def test_compile_without_copy_reports_generatable_code_and_missing_data(
    tmp_path: Path,
) -> None:
    manifest_before = MANIFEST_PATH.read_bytes()
    package = tmp_path / "freeze-package"

    report = compile_freeze_package(
        MANIFEST_PATH,
        REPOSITORY_ROOT,
        package,
        copy_code=False,
    )

    assert report["schema_version"] == FREEZE_READINESS_SCHEMA
    assert report["artifact_count"] == 79
    assert report["state_counts"]["generatable"] >= 7
    assert report["state_counts"]["missing"] > 0
    assert report["ready_for_freeze_review"] is False
    assert report["sealed_run_authorized"] is False
    assert MANIFEST_PATH.read_bytes() == manifest_before

    rows = {row["artifact_id"]: row for row in report["artifacts"]}
    assert rows["scifact-normalizer"]["state"] == "generatable"
    assert rows["scifact-sealed-inputs"]["state"] == "missing"
    assert not (package / "artifacts" / "normalizers" / "scifact" / "corpora.py").exists()
    assert (package / "artifact-map.json").is_file()
    assert (package / "freeze-readiness.json").is_file()


def test_compile_copies_code_to_separate_artifact_files_deterministically(
    tmp_path: Path,
) -> None:
    package = tmp_path / "freeze-package"
    first = compile_freeze_package(MANIFEST_PATH, REPOSITORY_ROOT, package)
    map_bytes = (package / "artifact-map.json").read_bytes()
    report_bytes = (package / "freeze-readiness.json").read_bytes()

    source = REPOSITORY_ROOT / "src" / "fractal_ann_diagnostics" / "corpora.py"
    copies = [
        package / "artifacts" / "normalizers" / corpus / "corpora.py"
        for corpus in (
            "scifact",
            "hotpotqa-fullwiki",
            "t2-ragbench",
            "bright",
            "miracl-transfer",
        )
    ]
    assert all(copy.read_bytes() == source.read_bytes() for copy in copies)
    assert len({copy.stat().st_ino for copy in copies}) == len(copies)

    rows = {row["artifact_id"]: row for row in first["artifacts"]}
    assert rows["scifact-normalizer"]["state"] == "present"
    assert rows["exact-authorized-numpy"]["state"] == "present"
    assert rows["geometry-informed-controller"]["state"] == "present"
    assert rows["custody-seal-builder"]["state"] == "present"
    assert (package / "artifacts" / "custody" / "builder.py").read_bytes() == (
        REPOSITORY_ROOT / "src" / "fractal_ann_diagnostics" / "custody.py"
    ).read_bytes()

    second = compile_freeze_package(MANIFEST_PATH, REPOSITORY_ROOT, package)
    assert second == first
    assert (package / "artifact-map.json").read_bytes() == map_bytes
    assert (package / "freeze-readiness.json").read_bytes() == report_bytes


@pytest.mark.parametrize("mutation", ("missing", "extra", "reassigned"))
def test_validate_map_rejects_any_coverage_or_assignment_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    manifest = _manifest()
    layout = layout_from_manifest(manifest, REPOSITORY_ROOT)
    payload = artifact_map_payload(layout)
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, list)
    if mutation == "missing":
        artifacts.pop()
    elif mutation == "extra":
        artifacts.append(
            {
                "artifact_id": "unregistered-artifact",
                "kind": "file",
                "relative_path": "unregistered.bin",
            }
        )
    else:
        artifacts[0]["relative_path"] = "wrong/location.json"
    map_path = tmp_path / "artifact-map.json"
    _write_map(map_path, payload)

    with pytest.raises(FreezePackageError):
        validate_freeze_artifact_map(manifest, REPOSITORY_ROOT, map_path)


def test_compile_refuses_to_write_a_package_inside_the_repository() -> None:
    forbidden = REPOSITORY_ROOT / ".freeze-package-test-output"
    assert not forbidden.exists()
    with pytest.raises(FreezePackageError, match="outside the source repository"):
        compile_freeze_package(MANIFEST_PATH, REPOSITORY_ROOT, forbidden)
    assert not forbidden.exists()


def test_compile_refuses_an_internal_package_symlink(tmp_path: Path) -> None:
    package = tmp_path / "freeze-package"
    outside = tmp_path / "outside"
    package.mkdir()
    outside.mkdir()
    (package / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(FreezePackageError, match="cannot be a symlink"):
        compile_freeze_package(MANIFEST_PATH, REPOSITORY_ROOT, package)
    assert not list(outside.iterdir())


def test_code_copy_drift_needs_explicit_refresh(tmp_path: Path) -> None:
    package = tmp_path / "freeze-package"
    compile_freeze_package(MANIFEST_PATH, REPOSITORY_ROOT, package)
    target = package / "artifacts" / "normalizers" / "scifact" / "corpora.py"
    target.write_bytes(b"changed")

    with pytest.raises(FreezePackageError, match="--refresh-code"):
        compile_freeze_package(MANIFEST_PATH, REPOSITORY_ROOT, package)

    compile_freeze_package(
        MANIFEST_PATH,
        REPOSITORY_ROOT,
        package,
        refresh_code=True,
    )
    source = REPOSITORY_ROOT / "src" / "fractal_ann_diagnostics" / "corpora.py"
    assert target.read_bytes() == source.read_bytes()


def test_standalone_module_cli_compiles_and_validates_map(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package = tmp_path / "freeze-package"
    assert (
        main(
            [
                "compile",
                "--repository-root",
                str(REPOSITORY_ROOT),
                "--manifest",
                str(MANIFEST_PATH),
                "--package-root",
                str(package),
                "--no-copy-code",
            ]
        )
        == 0
    )
    assert "mapped 79 manifest artifacts" in capsys.readouterr().out

    assert (
        main(
            [
                "validate-map",
                "--repository-root",
                str(REPOSITORY_ROOT),
                "--manifest",
                str(MANIFEST_PATH),
                "--artifact-map",
                str(package / "artifact-map.json"),
            ]
        )
        == 0
    )
    assert "valid exact artifact-map coverage: 79 artifacts" in capsys.readouterr().out
