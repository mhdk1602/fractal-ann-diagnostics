"""Fail-closed adjudication for retained runner-image scan evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNNER_SECURITY_SCHEMA = "fractal-runner-security-adjudication-v1"

_SEVERITIES = ("UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL")
_SERIOUS = frozenset({"HIGH", "CRITICAL"})
_PLATFORMS = frozenset({"linux/amd64", "linux/arm64"})
_IMAGE_ROLES = frozenset({"scientific", "timelock-release"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VULNERABILITY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}$")
_MAX_EVIDENCE_BYTES = 2 * 1024 * 1024 * 1024


class RunnerSecurityError(ValueError):
    """One retained scan is malformed, divergent, or policy-failing."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read_regular_file(path: Path, *, label: str) -> bytes:
    try:
        path.resolve(strict=True)
        observed = path.lstat()
    except OSError as error:
        raise RunnerSecurityError(f"cannot open {label}: {error}") from error
    if path.is_symlink():
        raise RunnerSecurityError(f"{label} must not be a symbolic link")
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise RunnerSecurityError(f"{label} must be one singly linked regular file")
    if observed.st_size <= 0 or observed.st_size > _MAX_EVIDENCE_BYTES:
        raise RunnerSecurityError(f"{label} has an invalid byte count")
    payload = path.read_bytes()
    if len(payload) != observed.st_size:
        raise RunnerSecurityError(f"{label} changed while it was read")
    return payload


def _object_pairs(label: str):
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RunnerSecurityError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    return reject_duplicates


def _json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise RunnerSecurityError(f"{label} contains non-finite number {value!r}")

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_object_pairs(label),
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerSecurityError(f"{label} is not one valid UTF-8 JSON document") from error
    if not isinstance(value, Mapping):
        raise RunnerSecurityError(f"{label} must be one JSON object")
    return value


def _string(value: object, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise RunnerSecurityError(f"{label} must be a string")
    if value != value.strip() or any(ord(character) < 0x20 for character in value):
        raise RunnerSecurityError(f"{label} is not canonical text")
    return value


def _array(value: object, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RunnerSecurityError(f"{label} must be an array")
    return value


@dataclass(frozen=True, order=True)
class Finding:
    result_class: str
    result_type: str
    vulnerability_id: str
    package_id: str
    package_name: str
    installed_version: str
    fixed_version: str
    severity: str
    status: str

    def as_dict(self) -> dict[str, str]:
        return {
            "fixed_version": self.fixed_version,
            "installed_version": self.installed_version,
            "package_id": self.package_id,
            "package_name": self.package_name,
            "result_class": self.result_class,
            "result_type": self.result_type,
            "severity": self.severity,
            "status": self.status,
            "vulnerability_id": self.vulnerability_id,
        }


@dataclass(frozen=True)
class TrivyProjection:
    artifact_name: str
    artifact_type: str
    findings: tuple[Finding, ...]
    result_count: int


def _parse_trivy(payload: bytes, *, label: str, expected_artifact_type: str) -> TrivyProjection:
    document = _json_object(payload, label=label)
    if document.get("SchemaVersion") != 2:
        raise RunnerSecurityError(f"{label} SchemaVersion must equal 2")
    artifact_name = _string(document.get("ArtifactName"), label=f"{label} ArtifactName")
    artifact_type = _string(document.get("ArtifactType"), label=f"{label} ArtifactType")
    if artifact_type != expected_artifact_type:
        raise RunnerSecurityError(f"{label} ArtifactType must equal {expected_artifact_type!r}")
    results = _array(document.get("Results"), label=f"{label} Results")
    if not results:
        raise RunnerSecurityError(f"{label} Results cannot be empty")
    findings: list[Finding] = []
    seen: set[Finding] = set()
    for result_position, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise RunnerSecurityError(f"{label} result {result_position} must be an object")
        result_class = _string(result.get("Class"), label=f"{label} result {result_position} Class")
        result_type = _string(result.get("Type"), label=f"{label} result {result_position} Type")
        vulnerabilities = result.get("Vulnerabilities")
        if vulnerabilities is None:
            vulnerabilities = []
        for vulnerability_position, vulnerability in enumerate(
            _array(
                vulnerabilities,
                label=f"{label} result {result_position} Vulnerabilities",
            )
        ):
            item_label = f"{label} result {result_position} vulnerability {vulnerability_position}"
            if not isinstance(vulnerability, Mapping):
                raise RunnerSecurityError(f"{item_label} must be an object")
            vulnerability_id = _string(
                vulnerability.get("VulnerabilityID"),
                label=f"{item_label} VulnerabilityID",
            )
            if _VULNERABILITY_ID.fullmatch(vulnerability_id) is None:
                raise RunnerSecurityError(f"{item_label} VulnerabilityID is invalid")
            severity = _string(vulnerability.get("Severity"), label=f"{item_label} Severity")
            if severity not in _SEVERITIES:
                raise RunnerSecurityError(f"{item_label} Severity is outside the closed set")
            package_name = _string(vulnerability.get("PkgName"), label=f"{item_label} PkgName")
            installed_version = _string(
                vulnerability.get("InstalledVersion"),
                label=f"{item_label} InstalledVersion",
            )
            raw_package_id = vulnerability.get("PkgID")
            if raw_package_id is not None:
                _string(raw_package_id, label=f"{item_label} PkgID")
            fixed_version_value = vulnerability.get("FixedVersion")
            if fixed_version_value is None:
                fixed_version_value = ""
            finding = Finding(
                result_class=result_class,
                result_type=result_type,
                vulnerability_id=vulnerability_id,
                package_id=f"{package_name}@{installed_version}",
                package_name=package_name,
                installed_version=installed_version,
                fixed_version=_string(
                    fixed_version_value,
                    label=f"{item_label} FixedVersion",
                    allow_empty=True,
                ),
                severity=severity,
                status=_string(
                    vulnerability.get("Status", "unknown"),
                    label=f"{item_label} Status",
                ),
            )
            if finding in seen:
                raise RunnerSecurityError(f"{label} repeats one normalized vulnerability")
            seen.add(finding)
            findings.append(finding)
    return TrivyProjection(
        artifact_name=artifact_name,
        artifact_type=artifact_type,
        findings=tuple(sorted(findings)),
        result_count=len(results),
    )


def _parse_cyclonedx(payload: bytes) -> dict[str, object]:
    document = _json_object(payload, label="CycloneDX SBOM")
    if document.get("bomFormat") != "CycloneDX":
        raise RunnerSecurityError("CycloneDX SBOM has another bomFormat")
    spec_version = _string(document.get("specVersion"), label="CycloneDX SBOM specVersion")
    if re.fullmatch(r"1\.[4-9]", spec_version) is None:
        raise RunnerSecurityError("CycloneDX SBOM specVersion is outside the admitted range")
    if document.get("version") != 1:
        raise RunnerSecurityError("CycloneDX SBOM version must equal 1")
    serial_number = _string(document.get("serialNumber"), label="CycloneDX SBOM serialNumber")
    if re.fullmatch(r"urn:uuid:[0-9a-f-]{36}", serial_number) is None:
        raise RunnerSecurityError("CycloneDX SBOM serialNumber is not a lowercase UUID URN")
    components = _array(document.get("components"), label="CycloneDX SBOM components")
    if not components or not all(isinstance(component, Mapping) for component in components):
        raise RunnerSecurityError("CycloneDX SBOM components are empty or malformed")
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("component"), Mapping):
        raise RunnerSecurityError("CycloneDX SBOM lacks its root component")
    return {
        "component_count": len(components),
        "serial_number": serial_number,
        "spec_version": spec_version,
    }


def adjudicate_runner_security(
    *,
    platform: str,
    image_role: str,
    direct_trivy_path: Path,
    sbom_trivy_path: Path,
    cyclonedx_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Validate raw scans and write one immutable canonical adjudication receipt."""
    if platform not in _PLATFORMS:
        raise RunnerSecurityError("platform must be linux/amd64 or linux/arm64")
    if image_role not in _IMAGE_ROLES:
        raise RunnerSecurityError("image_role is outside the closed role set")
    if image_role == "timelock-release" and platform != "linux/arm64":
        raise RunnerSecurityError("timelock-release is restricted to linux/arm64")
    direct_bytes = _read_regular_file(direct_trivy_path, label="direct Trivy JSON")
    sbom_scan_bytes = _read_regular_file(sbom_trivy_path, label="SBOM Trivy JSON")
    cyclonedx_bytes = _read_regular_file(cyclonedx_path, label="CycloneDX SBOM")
    direct = _parse_trivy(
        direct_bytes,
        label="direct Trivy JSON",
        expected_artifact_type="container_image",
    )
    sbom_scan = _parse_trivy(
        sbom_scan_bytes,
        label="SBOM Trivy JSON",
        expected_artifact_type="cyclonedx",
    )
    cyclonedx = _parse_cyclonedx(cyclonedx_bytes)
    direct_set = set(direct.findings)
    sbom_set = set(sbom_scan.findings)
    direct_only = sorted(direct_set - sbom_set)
    sbom_only = sorted(sbom_set - direct_set)
    if direct_only or sbom_only:
        raise RunnerSecurityError("direct and CycloneDX-rescan vulnerability sets differ")
    serious = [finding for finding in direct.findings if finding.severity in _SERIOUS]
    if serious:
        raise RunnerSecurityError("raw scan contains a HIGH or CRITICAL vulnerability")
    if image_role == "timelock-release":
        expected_unknown = Finding(
            result_class="lang-pkgs",
            result_type="gobinary",
            vulnerability_id="GO-2026-5932",
            package_id="golang.org/x/crypto@v0.54.0",
            package_name="golang.org/x/crypto",
            installed_version="v0.54.0",
            fixed_version="",
            severity="UNKNOWN",
            status="affected",
        )
        if direct.findings != (expected_unknown,):
            raise RunnerSecurityError(
                "timelock-release scan differs from the sole admitted UNKNOWN finding"
            )
    counts = Counter(finding.severity for finding in direct.findings)
    receipt: dict[str, object] = {
        "cyclonedx": cyclonedx,
        "cyclonedx_sha256": hashlib.sha256(cyclonedx_bytes).hexdigest(),
        "direct_artifact_name": direct.artifact_name,
        "direct_artifact_type": direct.artifact_type,
        "direct_result_count": direct.result_count,
        "direct_sbom_parity": True,
        "direct_sha256": hashlib.sha256(direct_bytes).hexdigest(),
        "finding_count": len(direct.findings),
        "findings": [finding.as_dict() for finding in direct.findings],
        "image_role": image_role,
        "platform": platform,
        "policy": "zero-raw-high-critical-and-direct-sbom-parity",
        "raw_high_critical_count": 0,
        "sbom_artifact_name": sbom_scan.artifact_name,
        "sbom_artifact_type": sbom_scan.artifact_type,
        "sbom_result_count": sbom_scan.result_count,
        "sbom_scan_sha256": hashlib.sha256(sbom_scan_bytes).hexdigest(),
        "schema_version": RUNNER_SECURITY_SCHEMA,
        "severity_counts": {severity: counts[severity] for severity in _SEVERITIES},
        "vex_documents": [],
        "vex_required": False,
    }
    payload = _canonical_json(receipt) + b"\n"
    try:
        descriptor = os.open(
            output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as error:
        raise RunnerSecurityError(f"cannot create adjudication receipt: {error}") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        output_path.chmod(0o444)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    if not _SHA256.fullmatch(hashlib.sha256(payload).hexdigest()):
        raise AssertionError("unreachable receipt digest invariant")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=sorted(_PLATFORMS))
    parser.add_argument("--image-role", required=True, choices=sorted(_IMAGE_ROLES))
    parser.add_argument("--direct-trivy", required=True, type=Path)
    parser.add_argument("--sbom-trivy", required=True, type=Path)
    parser.add_argument("--cyclonedx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = adjudicate_runner_security(
        platform=arguments.platform,
        image_role=arguments.image_role,
        direct_trivy_path=arguments.direct_trivy,
        sbom_trivy_path=arguments.sbom_trivy,
        cyclonedx_path=arguments.cyclonedx,
        output_path=arguments.output,
    )
    print(_canonical_json(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
