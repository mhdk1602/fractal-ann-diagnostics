from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from fractal_ann_diagnostics.runner_image_security import (
    RUNNER_SECURITY_SCHEMA,
    RunnerSecurityError,
    adjudicate_runner_security,
    main,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _finding(*, severity: str = "MEDIUM", vulnerability_id: str = "CVE-2026-1234"):
    return {
        "FixedVersion": "1.2.4",
        "InstalledVersion": "1.2.3",
        "PkgID": "example@1.2.3",
        "PkgName": "example",
        "Severity": severity,
        "Status": "fixed",
        "VulnerabilityID": vulnerability_id,
    }


def _unknown_x_crypto(version: str) -> dict[str, object]:
    return {
        "FixedVersion": None,
        "InstalledVersion": version,
        "PkgID": f"golang.org/x/crypto@{version}",
        "PkgName": "golang.org/x/crypto",
        "Severity": "UNKNOWN",
        "Status": "affected",
        "VulnerabilityID": "GO-2026-5932",
    }


def _trivy(
    *,
    artifact_type: str,
    findings: list[dict[str, object]],
    result_type: str = "python-pkg",
):
    return {
        "ArtifactName": "image@sha256:" + "a" * 64,
        "ArtifactType": artifact_type,
        "Results": [
            {
                "Class": "lang-pkgs",
                "Target": "Python",
                "Type": result_type,
                "Vulnerabilities": findings,
            }
        ],
        "SchemaVersion": 2,
    }


def _cyclonedx():
    return {
        "bomFormat": "CycloneDX",
        "components": [
            {
                "bom-ref": "pkg:pypi/example@1.2.3",
                "name": "example",
                "type": "library",
                "version": "1.2.3",
            }
        ],
        "metadata": {
            "component": {
                "bom-ref": "pkg:oci/example@sha256:" + "a" * 64,
                "name": "example",
                "type": "container",
            }
        },
        "serialNumber": "urn:uuid:12345678-1234-1234-1234-123456789abc",
        "specVersion": "1.7",
        "version": 1,
    }


def _write_evidence(
    tmp_path: Path,
    *,
    direct_findings: list[dict[str, object]] | None = None,
    sbom_findings: list[dict[str, object]] | None = None,
    direct_document: object | None = None,
    sbom_document: object | None = None,
    cyclonedx_document: object | None = None,
    result_type: str = "python-pkg",
) -> tuple[Path, Path, Path, Path]:
    findings = [_finding()]
    direct = tmp_path / "direct.json"
    sbom = tmp_path / "sbom-scan.json"
    cyclonedx = tmp_path / "sbom.cdx.json"
    output = tmp_path / "adjudication.json"
    direct.write_bytes(
        _canonical(
            direct_document
            if direct_document is not None
            else _trivy(
                artifact_type="container_image",
                findings=findings if direct_findings is None else direct_findings,
                result_type=result_type,
            )
        )
    )
    sbom.write_bytes(
        _canonical(
            sbom_document
            if sbom_document is not None
            else _trivy(
                artifact_type="cyclonedx",
                findings=findings if sbom_findings is None else sbom_findings,
                result_type=result_type,
            )
        )
    )
    cyclonedx.write_bytes(
        _canonical(_cyclonedx() if cyclonedx_document is None else cyclonedx_document)
    )
    return direct, sbom, cyclonedx, output


def _adjudicate(paths: tuple[Path, Path, Path, Path]):
    direct, sbom, cyclonedx, output = paths
    return adjudicate_runner_security(
        platform="linux/arm64",
        image_role="scientific",
        direct_trivy_path=direct,
        sbom_trivy_path=sbom,
        cyclonedx_path=cyclonedx,
        output_path=output,
    )


def test_equal_raw_scans_with_no_serious_findings_emit_closed_receipt(
    tmp_path: Path,
) -> None:
    paths = _write_evidence(tmp_path)
    receipt = _adjudicate(paths)
    output = paths[-1]

    assert receipt["schema_version"] == RUNNER_SECURITY_SCHEMA
    assert receipt["direct_sbom_parity"] is True
    assert receipt["finding_count"] == 1
    assert receipt["image_role"] == "scientific"
    assert receipt["raw_high_critical_count"] == 0
    assert receipt["severity_counts"] == {
        "CRITICAL": 0,
        "HIGH": 0,
        "LOW": 0,
        "MEDIUM": 1,
        "UNKNOWN": 0,
    }
    assert receipt["vex_documents"] == []
    assert receipt["vex_required"] is False
    assert output.read_bytes() == _canonical(receipt) + b"\n"
    assert stat.S_IMODE(output.stat().st_mode) == 0o444


def test_cli_writes_the_same_receipt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    direct, sbom, cyclonedx, output = _write_evidence(tmp_path)

    status = main(
        [
            "--platform",
            "linux/arm64",
            "--image-role",
            "scientific",
            "--direct-trivy",
            str(direct),
            "--sbom-trivy",
            str(sbom),
            "--cyclonedx",
            str(cyclonedx),
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == RUNNER_SECURITY_SCHEMA
    assert json.loads(output.read_text())["platform"] == "linux/arm64"


def test_direct_and_sbom_rescan_drift_is_rejected(tmp_path: Path) -> None:
    paths = _write_evidence(
        tmp_path,
        sbom_findings=[_finding(vulnerability_id="CVE-2026-5678")],
    )

    with pytest.raises(RunnerSecurityError, match="vulnerability sets differ"):
        _adjudicate(paths)
    assert not paths[-1].exists()


@pytest.mark.parametrize("severity", ["HIGH", "CRITICAL"])
def test_any_raw_serious_finding_is_rejected(tmp_path: Path, severity: str) -> None:
    findings = [_finding(severity=severity)]
    paths = _write_evidence(
        tmp_path,
        direct_findings=findings,
        sbom_findings=findings,
    )

    with pytest.raises(RunnerSecurityError, match="HIGH or CRITICAL"):
        _adjudicate(paths)
    assert not paths[-1].exists()


def test_duplicate_normalized_finding_is_rejected(tmp_path: Path) -> None:
    finding = _finding()
    paths = _write_evidence(tmp_path, direct_findings=[finding, dict(finding)])

    with pytest.raises(RunnerSecurityError, match="repeats one normalized"):
        _adjudicate(paths)


def test_unknown_severity_token_is_rejected(tmp_path: Path) -> None:
    paths = _write_evidence(
        tmp_path,
        direct_findings=[_finding(severity="IMPORTANT")],
    )

    with pytest.raises(RunnerSecurityError, match="outside the closed set"):
        _adjudicate(paths)


@pytest.mark.parametrize(
    "direct_document",
    [
        {},
        {"ArtifactName": "x", "ArtifactType": "container_image", "SchemaVersion": 2},
        {
            "ArtifactName": "x",
            "ArtifactType": "container_image",
            "Results": [],
            "SchemaVersion": 2,
        },
        {
            "ArtifactName": "x",
            "ArtifactType": "filesystem",
            "Results": [{}],
            "SchemaVersion": 2,
        },
    ],
)
def test_malformed_direct_scan_is_rejected(tmp_path: Path, direct_document: object) -> None:
    paths = _write_evidence(tmp_path, direct_document=direct_document)

    with pytest.raises(RunnerSecurityError):
        _adjudicate(paths)


def test_malformed_cyclonedx_is_rejected(tmp_path: Path) -> None:
    document = _cyclonedx()
    document["components"] = []
    paths = _write_evidence(tmp_path, cyclonedx_document=document)

    with pytest.raises(RunnerSecurityError, match="components"):
        _adjudicate(paths)


def test_hardlinked_raw_evidence_is_rejected(tmp_path: Path) -> None:
    direct, sbom, cyclonedx, output = _write_evidence(tmp_path)
    second_link = tmp_path / "direct-second-link.json"
    second_link.hardlink_to(direct)

    with pytest.raises(RunnerSecurityError, match="singly linked"):
        _adjudicate((direct, sbom, cyclonedx, output))


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    paths[-1].write_text("custodied\n", encoding="utf-8")

    with pytest.raises(RunnerSecurityError, match="cannot create"):
        _adjudicate(paths)
    assert paths[-1].read_text() == "custodied\n"


def test_timelock_release_admits_only_the_two_measured_unknown_go_findings(
    tmp_path: Path,
) -> None:
    unknowns = [_unknown_x_crypto("v0.53.0"), _unknown_x_crypto("v0.54.0")]
    direct, sbom, cyclonedx, output = _write_evidence(
        tmp_path,
        direct_findings=unknowns,
        sbom_findings=unknowns,
        result_type="gobinary",
    )

    receipt = adjudicate_runner_security(
        platform="linux/arm64",
        image_role="timelock-release",
        direct_trivy_path=direct,
        sbom_trivy_path=sbom,
        cyclonedx_path=cyclonedx,
        output_path=output,
    )

    assert receipt["finding_count"] == 2
    assert receipt["severity_counts"]["UNKNOWN"] == 2
    assert [row["installed_version"] for row in receipt["findings"]] == [
        "v0.53.0",
        "v0.54.0",
    ]
    assert {row["vulnerability_id"] for row in receipt["findings"]} == {"GO-2026-5932"}
    assert {row["fixed_version"] for row in receipt["findings"]} == {""}
    assert receipt["vex_documents"] == []
    assert receipt["vex_required"] is False


def test_timelock_release_rejects_an_extra_nonserious_finding(tmp_path: Path) -> None:
    findings = [
        _unknown_x_crypto("v0.53.0"),
        _unknown_x_crypto("v0.54.0"),
        _finding(severity="LOW"),
    ]
    direct, sbom, cyclonedx, output = _write_evidence(
        tmp_path,
        direct_findings=findings,
        sbom_findings=findings,
        result_type="gobinary",
    )

    with pytest.raises(RunnerSecurityError, match="exact admitted UNKNOWN"):
        adjudicate_runner_security(
            platform="linux/arm64",
            image_role="timelock-release",
            direct_trivy_path=direct,
            sbom_trivy_path=sbom,
            cyclonedx_path=cyclonedx,
            output_path=output,
        )
    assert not output.exists()
