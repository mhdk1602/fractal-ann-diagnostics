"""Deterministic, non-consuming compiler for a confirmatory freeze package.

This module prepares storage paths and copies locally available code artifacts.
It never changes the study manifest, supplies missing scientific artifacts,
registers a protocol, creates a run receipt, or opens a sealed run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from .artifact_integrity import (
    LOCAL_ARTIFACT_MAP_SCHEMA,
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    load_local_artifact_map,
    read_secure_regular_file,
)
from .artifact_stage_bundles import (
    ArtifactStageBundleError,
    verify_index_stage_bundle,
    verify_policy_stage_bundle,
)
from .embedding_store import EmbeddingStoreError, verify_embedding_store
from .joint_power_design import (
    JointPowerDesignError,
    JointPowerDesignReport,
    canonical_joint_power_report_bytes,
    load_development_panel,
    load_joint_power_config,
    load_joint_power_report,
    load_joint_power_selection_audit,
    run_joint_power_design,
    verify_joint_power_selection_audit,
)
from .opa_runtime_binary import (
    OpaRuntimeBinaryError,
    load_runtime_attestation_plan_template,
    verify_opa_runtime_binary,
)
from .scalable_execution import (
    ONLINE_EXECUTION_PLAN_FILENAME,
    ScalableExecutionError,
    load_sharded_online_execution_plan,
    verify_online_execution_package,
)
from .scalable_partition_audit import (
    ScalablePartitionAuditError,
    load_scalable_partition_audit,
)
from .study import load_study_manifest, manifest_sha256, validate_study_manifest

FREEZE_READINESS_SCHEMA = "fractal-freeze-readiness-v1"
ArtifactKind = Literal["file", "directory"]
ArtifactState = Literal["missing", "generatable", "present"]

_PLACEHOLDERS = {"", "tbd", "todo", "latest", "main", "master", "unassigned"}
_SAFE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_SENTINEL_SHA256 = "0" * 64

_CORPUS_LAYOUTS: dict[str, tuple[str, ArtifactKind]] = {
    "sealed-inputs": ("corpora/{corpus_id}/sealed-inputs.json", "file"),
    "sealed-labels": ("custody/labels/{corpus_id}.json", "file"),
    "sealed-label-ciphertext": (
        "custody/ciphertext/{corpus_id}-sealed-labels.tlock",
        "file",
    ),
    "timelock-encryption-receipt": (
        "custody/receipts/{corpus_id}-timelock-encryption.json",
        "file",
    ),
    "online-execution": ("custody/online/{corpus_id}", "directory"),
    "corpus-normalizer": ("normalizers/{corpus_id}/corpora.py", "file"),
    "policy-workload": ("policy-workloads/{corpus_id}", "directory"),
    "embedding-store": ("embedding-stores/{corpus_id}", "directory"),
    "authorized-index-store": ("authorized-index-stores/{corpus_id}", "directory"),
    "trial-runtime-package": ("trial-runtime/{corpus_id}", "directory"),
    "runtime-attestation-plan-template": (
        "runtime/{corpus_id}/runtime-attestation-plan.template.json",
        "file",
    ),
}

_SINGLETON_LAYOUTS: dict[str, tuple[str, ArtifactKind]] = {
    "study-data-package": ("study-data/custody-complete", "directory"),
    "online-staging-package": ("study-data/online-projection", "directory"),
    "development-freeze-package": ("development/freeze-package", "directory"),
    "development-fit-data": ("development/fit.json", "file"),
    "development-calibration-data": ("development/calibration.json", "file"),
    "query-partition-audit": ("development/query-partition-audit.json", "file"),
    "exact-authorized-oracle": ("backends/exact/retrieval.py", "file"),
    "strict-authorized-hnsw": ("backends/hnsw/hnswlib-runtime.whl", "file"),
    "opa-pdp": ("policy/opa-bundle", "directory"),
    "opa-runtime-binary": ("runtime/opa", "file"),
    "frozen-controller": ("controller/controller.py", "file"),
    "static-comparator": ("controller/static-comparator.json", "file"),
    "h1-predictive-model": ("models/h1-full-model.json", "file"),
    "h2-model-suite": ("models/h2-suite.json", "file"),
    "power-analysis-report": ("analysis/joint-power-design", "directory"),
    "analysis-runner": ("analysis/runner", "directory"),
    "source-code": ("source/fractal-ann-diagnostics-C0.tar", "file"),
    "custody-seal-receipt": ("custody/custody-seal-receipt.json", "file"),
    "tlock-release-provenance": ("custody/tlock-release-provenance.json", "file"),
    "timelock-tool": ("custody/bin/tle", "file"),
    "custody-builder": ("custody/builder.py", "file"),
    "suite-attestation-descriptor": (
        "suite/suite-attestation-descriptor.json",
        "file",
    ),
}

_COPYABLE_CODE_ROLES = {
    "corpus-normalizer",
    "exact-authorized-oracle",
    "frozen-controller",
    "custody-builder",
}

_POWER_CONFIG_FILENAME = "config.json"
_POWER_REPORT_FILENAME = "report.json"
_POWER_SELECTION_AUDIT_FILENAME = "selection-audit.json"
_POWER_PANEL_DIRECTORY = "panels"
_POWER_MAX_FILE_BYTES = 256 * 1024 * 1024

_C0_C1_SEQUENCE = (
    {
        "step": 1,
        "phase": "C0",
        "action": (
            "Finalize executable code, converters, schemas, conformance tests, and all "
            "development-only generation recipes."
        ),
    },
    {
        "step": 2,
        "phase": "C0",
        "action": (
            "Build the OCI image, code archive, runtime binaries, and development-derived "
            "artifacts from C0 without embedding the study manifest."
        ),
    },
    {
        "step": 3,
        "phase": "C1",
        "action": (
            "Create a manifest-only freeze commit that pins C0, every admitted artifact digest, "
            "the OCI digest, hardware, runner identity, custody records, and empty blockers."
        ),
    },
    {
        "step": 4,
        "phase": "registration",
        "action": (
            "Deposit the exact canonical C1 manifest in the independent registry and verify the "
            "immutable HTTPS record before any sealed execution."
        ),
    },
    {
        "step": 5,
        "phase": "sealed-run",
        "action": (
            "Mount C1 and admitted artifacts read-only, verify them again, and only then invoke "
            "the separately controlled one-shot run opener."
        ),
    },
)

_CIRCULARITY_CONTROLS = (
    {
        "risk": "source-commit-self-reference",
        "control": "The source artifact and runner name C0; the frozen manifest is created in C1.",
    },
    {
        "risk": "image-manifest-self-reference",
        "control": "Build the C0 image without the manifest, then mount the C1 manifest read-only.",
    },
    {
        "risk": "label-manifest-fixed-point",
        "control": (
            "Custody artifacts bind their direct inputs; the outer C1 manifest binds their exact "
            "bytes. They do not embed the C1 digest."
        ),
    },
    {
        "risk": "receipt-preexistence",
        "control": (
            "Registration, completion, and run receipts are post-freeze controls, not artifacts "
            "whose bytes are embedded in the manifest they attest."
        ),
    },
)


class FreezePackageError(ValueError):
    """Raised when a freeze-package plan or materialization is unsafe."""


@dataclass(frozen=True)
class FreezeArtifactLayout:
    """One manifest-derived controlled storage assignment."""

    artifact_id: str
    role: str
    relative_path: str
    kind: ArtifactKind
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.role:
            raise FreezePackageError("artifact layout needs a non-empty ID and role")
        if self.kind not in {"file", "directory"}:
            raise FreezePackageError("artifact layout kind must be file or directory")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or str(path) != self.relative_path:
            raise FreezePackageError("artifact layout paths must be canonical and relative")
        if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise FreezePackageError("artifact layout paths cannot traverse parents")
        if self.source_path is not None:
            source = PurePosixPath(self.source_path)
            if source.is_absolute() or str(source) != self.source_path:
                raise FreezePackageError("code source paths must be canonical and relative")


@dataclass(frozen=True)
class JointPowerBundleVerification:
    """Typed result of recomputing one closed development power bundle."""

    tree_sha256: str
    config_sha256: str
    report_sha256: str
    selection_audit_sha256: str
    scenario_count: int
    selected_families_per_corpus: int
    selected_joint_power_lower_bound: float


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _is_placeholder(value: object) -> bool:
    return not isinstance(value, str) or value.strip().casefold() in _PLACEHOLDERS


def _safe_component(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise FreezePackageError(f"{field} cannot be represented as a controlled path component")
    return value


def _local_code_source(artifact: Mapping[str, Any], repository_root: Path) -> str | None:
    role = str(artifact["role"])
    if role not in _COPYABLE_CODE_ROLES:
        return None
    uri = artifact.get("uri")
    if _is_placeholder(uri):
        return None
    assert isinstance(uri, str)
    path_text = uri.split("#", 1)[0]
    if not path_text or _URI_SCHEME.match(path_text) or "\\" in path_text:
        return None
    relative = PurePosixPath(path_text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    candidate = repository_root.joinpath(*relative.parts)
    if not candidate.is_file() or candidate.is_symlink():
        return None
    try:
        candidate.resolve(strict=True).relative_to(repository_root)
    except ValueError:
        return None
    return str(relative)


def _layout_path(artifact: Mapping[str, Any]) -> tuple[str, ArtifactKind]:
    role = str(artifact["role"])
    if role in _CORPUS_LAYOUTS:
        corpus_id = _safe_component(artifact.get("corpus_id"), field="corpus_id")
        template, kind = _CORPUS_LAYOUTS[role]
        return template.format(corpus_id=corpus_id), kind
    if role in {"primary-embedding", "stale-embedding"}:
        artifact_id = _safe_component(artifact["id"], field="artifact id")
        return f"embedding/{artifact_id}", "directory"
    try:
        return _SINGLETON_LAYOUTS[role]
    except KeyError as exc:
        raise FreezePackageError(f"artifact role {role!r} has no controlled-layout rule") from exc


def layout_from_manifest(
    manifest: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[FreezeArtifactLayout, ...]:
    """Derive one exact layout row per artifact in a valid draft or frozen manifest."""

    validate_study_manifest(manifest)
    repository = Path(repository_root).expanduser().resolve(strict=True)
    values = manifest["artifacts"]
    assert isinstance(values, Sequence)
    rows: list[FreezeArtifactLayout] = []
    for artifact in values:
        assert isinstance(artifact, Mapping)
        relative_path, kind = _layout_path(artifact)
        rows.append(
            FreezeArtifactLayout(
                artifact_id=str(artifact["id"]),
                role=str(artifact["role"]),
                relative_path=relative_path,
                kind=kind,
                source_path=_local_code_source(artifact, repository),
            )
        )
    identifiers = [row.artifact_id for row in rows]
    paths = [row.relative_path for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise FreezePackageError("manifest-derived layout contains duplicate artifact IDs")
    if len(paths) != len(set(paths)):
        raise FreezePackageError("manifest-derived layout contains duplicate artifact paths")
    return tuple(rows)


def artifact_map_payload(layout: Sequence[FreezeArtifactLayout]) -> dict[str, Any]:
    """Return the verifier-compatible map without inventing any digest."""

    return {
        "artifacts": [
            {
                "artifact_id": row.artifact_id,
                "kind": row.kind,
                "relative_path": row.relative_path,
            }
            for row in layout
        ],
        "schema_version": LOCAL_ARTIFACT_MAP_SCHEMA,
    }


def validate_freeze_artifact_map(
    manifest: Mapping[str, Any],
    repository_root: str | Path,
    artifact_map_path: str | Path,
) -> tuple[FreezeArtifactLayout, ...]:
    """Validate exact map coverage without requiring pins or opening a run."""

    layout = layout_from_manifest(manifest, repository_root)
    expected_sha256_by_id = {row.artifact_id: _SENTINEL_SHA256 for row in layout}
    try:
        specs = load_local_artifact_map(
            Path(artifact_map_path).expanduser().resolve(strict=True),
            expected_sha256_by_id=expected_sha256_by_id,
        )
    except ArtifactIntegrityError as exc:
        raise FreezePackageError(str(exc)) from exc
    expected = {row.artifact_id: row for row in layout}
    for spec in specs:
        row = expected[spec.artifact_id]
        if spec.relative_path != row.relative_path or spec.kind != row.kind:
            raise FreezePackageError(
                f"artifact map assignment for {spec.artifact_id!r} differs from the "
                "manifest-derived controlled layout"
            )
    return layout


def _write_atomic(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    _ensure_directory(path.parent)
    if path.is_symlink():
        raise FreezePackageError(f"refusing to replace symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _ensure_directory(path: Path) -> None:
    """Create one package directory without following an internal symlink."""

    if path.is_symlink():
        raise FreezePackageError(f"package directory cannot be a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise FreezePackageError(f"package directory path is not a directory: {path}")
        return
    _ensure_directory(path.parent)
    path.mkdir(mode=0o700)


def _copy_code_artifact(
    source: Path,
    target: Path,
    *,
    refresh: bool,
) -> None:
    source_digest = digest_regular_file(source.resolve(strict=True), label="code source")
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise FreezePackageError(f"code artifact target is not a regular file: {target}")
        target_digest = digest_regular_file(target.resolve(strict=True), label="code artifact")
        if target_digest == source_digest:
            return
        if not refresh:
            raise FreezePackageError(
                f"code artifact differs from its source: {target}; rerun with --refresh-code"
            )
    _write_atomic(target, source.read_bytes())
    copied_digest = digest_regular_file(target.resolve(strict=True), label="copied code artifact")
    if copied_digest != source_digest:
        raise FreezePackageError(f"copied code artifact digest mismatch: {target}")


def _read_power_bundle_file(path: Path, *, label: str) -> bytes:
    try:
        return read_secure_regular_file(
            path,
            max_bytes=_POWER_MAX_FILE_BYTES,
            label=label,
        )
    except ArtifactIntegrityError as exc:
        raise FreezePackageError(f"cannot read {label}: {exc}") from exc


def _selected_joint_power_lower_bound(
    report: JointPowerDesignReport,
    *,
    required_scenarios: frozenset[str],
) -> float:
    selected = report.selected_families_per_corpus
    if selected is None:
        raise FreezePackageError("power bundle has no selected family count")
    bounds = tuple(
        estimate.joint_probability.lower_probability_bound
        for estimate in report.estimates
        if estimate.scenario_id in required_scenarios and estimate.families_per_corpus == selected
    )
    if len(bounds) != len(required_scenarios):
        raise FreezePackageError(
            "power report does not contain one selected estimate per required scenario"
        )
    return min(bounds)


def _require_power_manifest_value(
    field: str,
    observed: object,
    expected: object,
) -> None:
    if observed != expected:
        raise FreezePackageError(f"manifest {field} differs from the recomputed power bundle")


def _cross_check_power_manifest(
    manifest: Mapping[str, Any],
    *,
    config: Any,
    report: JointPowerDesignReport,
    selection_audit: Any,
    selected_lower_bound: float,
) -> None:
    analysis = manifest.get("analysis")
    if not isinstance(analysis, Mapping):
        raise FreezePackageError("manifest analysis must be an object")
    power = analysis.get("power")
    if not isinstance(power, Mapping):
        raise FreezePackageError("manifest analysis.power must be an object")

    scenarios = tuple(item.scenario_id for item in config.effect_scenarios)
    _require_power_manifest_value(
        "analysis.power_target",
        analysis.get("power_target"),
        config.target_power,
    )
    _require_power_manifest_value(
        "analysis.power.candidate_families_per_corpus",
        tuple(power.get("candidate_families_per_corpus", ())),
        config.candidate_families_per_corpus,
    )
    _require_power_manifest_value(
        "analysis.power.simulation_seed",
        power.get("simulation_seed"),
        config.simulation_seed,
    )
    _require_power_manifest_value(
        "analysis.power.simulation_count",
        power.get("simulation_count"),
        config.n_simulations,
    )
    _require_power_manifest_value(
        "analysis.power.bound_calibration_simulations",
        power.get("simulation_count"),
        config.bound_calibration_simulations,
    )
    _require_power_manifest_value(
        "analysis.power.effect_scenarios",
        tuple(power.get("effect_scenarios", ())),
        scenarios,
    )
    _require_power_manifest_value(
        "analysis.power.dependence_source",
        power.get("dependence_source"),
        config.dependence_source.artifact_uri,
    )
    _require_power_manifest_value(
        "analysis.power.selected_families_per_corpus",
        power.get("selected_families_per_corpus"),
        report.selected_families_per_corpus,
    )
    _require_power_manifest_value(
        "analysis.power.selected_joint_power_lower_bound",
        power.get("selected_joint_power_lower_bound"),
        selected_lower_bound,
    )
    _require_power_manifest_value(
        "analysis.power.registered_endpoints",
        tuple(power.get("registered_endpoints", ())),
        config.endpoint_order,
    )
    _require_power_manifest_value(
        "analysis.power.selection_multiplicity_method",
        power.get("selection_multiplicity_method"),
        config.selection_multiplicity_method,
    )
    _require_power_manifest_value(
        "analysis.power.selection_familywise_confidence",
        power.get("selection_familywise_confidence"),
        config.monte_carlo_confidence,
    )
    _require_power_manifest_value(
        "analysis.power.selection_family_size",
        power.get("selection_family_size"),
        config.selection_family_size,
    )
    _require_power_manifest_value(
        "analysis.power.selection_cell_alpha",
        power.get("selection_cell_alpha"),
        config.selection_cell_alpha,
    )
    qualifying_thresholds = {row.required_successes for row in selection_audit.certificates}
    blocking_thresholds = {row.required_failures for row in selection_audit.certificates}
    if len(qualifying_thresholds) != 1 or len(blocking_thresholds) != 1:
        raise FreezePackageError("selection audit contains inconsistent probability thresholds")
    _require_power_manifest_value(
        "analysis.power.selection_exact_qualifying_passes",
        power.get("selection_exact_qualifying_passes"),
        next(iter(qualifying_thresholds)),
    )
    _require_power_manifest_value(
        "analysis.power.selection_exact_blocking_failures",
        power.get("selection_exact_blocking_failures"),
        next(iter(blocking_thresholds)),
    )

    scalar_bindings = (
        ("alpha", config.alpha),
        ("nested_rows_per_family", config.nested_rows_per_family),
        (
            "minimum_corpora_with_geometry_gain",
            config.minimum_corpora_with_geometry_gain,
        ),
        ("minimum_cost_reduction", config.minimum_latency_reduction),
        (
            "retrieval_target_noninferiority_margin",
            config.retrieval_target_noninferiority_margin,
        ),
        (
            "evidence_sufficiency_noninferiority_margin",
            config.evidence_sufficiency_noninferiority_margin,
        ),
        ("maximum_p95_latency_ratio", config.maximum_p95_latency_ratio),
        ("maximum_entitlement_violations", config.maximum_denied_emissions),
    )
    for field, expected in scalar_bindings:
        _require_power_manifest_value(
            f"analysis.{field}",
            analysis.get(field),
            expected,
        )
    _require_power_manifest_value(
        "analysis.geometry_gain_thresholds",
        analysis.get("geometry_gain_thresholds"),
        config.geometry_gain_thresholds.to_dict(),
    )
    _require_power_manifest_value(
        "analysis.fixed_corpora",
        tuple(analysis.get("fixed_corpora", ())),
        config.fixed_corpora,
    )
    _require_power_manifest_value(
        "analysis.evidence_corpora",
        tuple(analysis.get("evidence_corpora", ())),
        config.evidence_corpora,
    )


def verify_joint_power_bundle(
    bundle_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    expected_tree_sha256: str | None = None,
) -> JointPowerBundleVerification:
    """Typed-load, rerun, and manifest-bind one joint-power artifact tree."""

    bundle = Path(bundle_root).expanduser()
    if not bundle.is_absolute():
        raise FreezePackageError("power bundle path must be absolute")
    if bundle.is_symlink():
        raise FreezePackageError("power bundle root cannot be a symlink")
    try:
        tree = digest_directory_tree(bundle)
    except ArtifactIntegrityError as exc:
        raise FreezePackageError(f"cannot verify power bundle tree: {exc}") from exc
    if expected_tree_sha256 is not None and tree.sha256 != expected_tree_sha256:
        raise FreezePackageError("power bundle tree digest differs")

    config_bytes = _read_power_bundle_file(
        bundle / _POWER_CONFIG_FILENAME,
        label="joint-power config",
    )
    report_bytes = _read_power_bundle_file(
        bundle / _POWER_REPORT_FILENAME,
        label="joint-power report",
    )
    audit_bytes = _read_power_bundle_file(
        bundle / _POWER_SELECTION_AUDIT_FILENAME,
        label="joint-power selection audit",
    )
    try:
        config = load_joint_power_config(config_bytes)
        report = load_joint_power_report(report_bytes)
        selection_audit = load_joint_power_selection_audit(audit_bytes)
    except JointPowerDesignError as exc:
        raise FreezePackageError(f"invalid typed power bundle: {exc}") from exc
    if config.test_mode or report.test_mode:
        raise FreezePackageError("power bundle cannot use test mode")
    if not report.freeze_ready:
        raise FreezePackageError("power bundle report is not freeze ready")

    panel_entries = tuple(
        f"{_POWER_PANEL_DIRECTORY}/{scenario.panel_sha256}.json"
        for scenario in config.effect_scenarios
    )
    expected_entries = tuple(
        sorted(
            (
                _POWER_CONFIG_FILENAME,
                _POWER_REPORT_FILENAME,
                _POWER_SELECTION_AUDIT_FILENAME,
                _POWER_PANEL_DIRECTORY,
                *panel_entries,
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if tree.entries != expected_entries:
        raise FreezePackageError(
            "power bundle file set differs from the registered audit, config, report, and panels"
        )

    panels = []
    for scenario, relative_path in zip(config.effect_scenarios, panel_entries, strict=True):
        panel_bytes = _read_power_bundle_file(
            bundle.joinpath(*PurePosixPath(relative_path).parts),
            label=f"joint-power panel {scenario.scenario_id!r}",
        )
        try:
            panel = load_development_panel(panel_bytes)
        except JointPowerDesignError as exc:
            raise FreezePackageError(
                f"invalid typed power panel {scenario.scenario_id!r}: {exc}"
            ) from exc
        if panel.scenario_id != scenario.scenario_id or panel.sha256 != scenario.panel_sha256:
            raise FreezePackageError(
                f"power panel {scenario.scenario_id!r} differs from its config pin"
            )
        panels.append(panel)

    try:
        recomputed_audit = verify_joint_power_selection_audit(
            config,
            tuple(panels),
            selection_audit,
        )
        recomputed = run_joint_power_design(
            config,
            tuple(panels),
            selection_audit=recomputed_audit,
        )
    except JointPowerDesignError as exc:
        raise FreezePackageError(f"power bundle recomputation failed: {exc}") from exc
    if canonical_joint_power_report_bytes(recomputed) != report_bytes:
        raise FreezePackageError(
            "power report bytes differ from a fresh run over the pinned config and panels"
        )
    if not recomputed.freeze_ready or recomputed.test_mode:
        raise FreezePackageError("recomputed power report is not freeze admissible")

    required_scenarios = frozenset(
        scenario.scenario_id for scenario in config.effect_scenarios if scenario.selection_required
    )
    selected_lower_bound = _selected_joint_power_lower_bound(
        recomputed,
        required_scenarios=required_scenarios,
    )
    _cross_check_power_manifest(
        manifest,
        config=config,
        report=recomputed,
        selection_audit=recomputed_audit,
        selected_lower_bound=selected_lower_bound,
    )
    try:
        final_tree = digest_directory_tree(bundle)
    except ArtifactIntegrityError as exc:
        raise FreezePackageError(f"cannot reverify power bundle tree: {exc}") from exc
    if final_tree != tree:
        raise FreezePackageError("power bundle changed during typed verification")
    return JointPowerBundleVerification(
        tree_sha256=tree.sha256,
        config_sha256=config.sha256,
        report_sha256=recomputed.sha256,
        selection_audit_sha256=recomputed_audit.sha256,
        scenario_count=len(panels),
        selected_families_per_corpus=recomputed.selected_families_per_corpus,
        selected_joint_power_lower_bound=selected_lower_bound,
    )


def _inspect_target(
    row: FreezeArtifactLayout,
    artifact_root: Path,
    repository_root: Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = artifact_root.joinpath(*PurePosixPath(row.relative_path).parts)
    source_sha256: str | None = None
    if row.source_path is not None:
        source = repository_root.joinpath(*PurePosixPath(row.source_path).parts)
        source_sha256 = digest_regular_file(source.resolve(strict=True), label="code source")

    if target.is_symlink():
        raise FreezePackageError(f"artifact target cannot be a symlink: {target}")
    if not target.exists():
        state: ArtifactState = "generatable" if source_sha256 is not None else "missing"
        return {
            "artifact_id": row.artifact_id,
            "byte_count": None,
            "directory_count": None,
            "file_count": None,
            "kind": row.kind,
            "relative_path": row.relative_path,
            "revision": None,
            "role": row.role,
            "sha256": None,
            "source_path": row.source_path,
            "source_sha256": source_sha256,
            "state": state,
        }
    if row.kind == "file":
        if not target.is_file():
            raise FreezePackageError(f"artifact target must be a file: {target}")
        digest = digest_regular_file(target.resolve(strict=True), label=row.artifact_id)
        revision: str | None = None
        if row.role == "query-partition-audit":
            try:
                audit = load_scalable_partition_audit(
                    target.resolve(strict=True),
                    expected_artifact_sha256=digest,
                )
            except ScalablePartitionAuditError as exc:
                raise FreezePackageError(
                    f"invalid typed query-partition audit {row.artifact_id!r}: {exc}"
                ) from exc
            revision = f"sha256:{audit.artifact_sha256}"
        if row.role == "runtime-attestation-plan-template":
            try:
                load_runtime_attestation_plan_template(target.resolve(strict=True))
            except OpaRuntimeBinaryError as exc:
                raise FreezePackageError(
                    f"invalid runtime plan template {row.artifact_id!r}: {exc}"
                ) from exc
            revision = f"sha256:{digest}"
        if row.role == "opa-runtime-binary":
            revision = f"sha256:{digest}"
        return {
            "artifact_id": row.artifact_id,
            "byte_count": target.stat().st_size,
            "directory_count": 0,
            "file_count": 1,
            "kind": row.kind,
            "relative_path": row.relative_path,
            "revision": revision,
            "role": row.role,
            "sha256": digest,
            "source_path": row.source_path,
            "source_sha256": source_sha256,
            "state": "present",
        }
    if not target.is_dir():
        raise FreezePackageError(f"artifact target must be a directory: {target}")
    try:
        tree = digest_directory_tree(target.resolve(strict=True))
    except ArtifactIntegrityError as exc:
        raise FreezePackageError(
            f"cannot verify directory artifact {row.artifact_id!r}: {exc}"
        ) from exc
    state = "present" if tree.file_count > 0 else "missing"
    revision: str | None = None
    if row.role == "online-execution" and state == "present":
        try:
            plan = load_sharded_online_execution_plan(target / ONLINE_EXECUTION_PLAN_FILENAME)
            package = verify_online_execution_package(
                target.resolve(strict=True),
                expected_tree_sha256=tree.sha256,
                expected_plan_revision=f"sha256:{plan.artifact_sha256}",
            )
        except ScalableExecutionError as exc:
            raise FreezePackageError(
                f"invalid online-execution package {row.artifact_id!r}: {exc}"
            ) from exc
        revision = package.revision
    if row.role == "policy-workload" and state == "present":
        corpus_id = PurePosixPath(row.relative_path).name
        try:
            receipt = verify_policy_stage_bundle(
                target.resolve(strict=True),
                expected_corpus_id=corpus_id,
            )
        except ArtifactStageBundleError as exc:
            raise FreezePackageError(
                f"invalid typed policy stage bundle {row.artifact_id!r}: {exc}"
            ) from exc
        revision = f"sha256:{receipt.receipt_sha256}"
    if row.role == "embedding-store" and state == "present":
        try:
            receipt = verify_embedding_store(target.resolve(strict=True))
        except EmbeddingStoreError as exc:
            raise FreezePackageError(
                f"invalid typed embedding store {row.artifact_id!r}: {exc}"
            ) from exc
        revision = f"sha256:{receipt.receipt_sha256}"
    if row.role == "authorized-index-store" and state == "present":
        corpus_id = PurePosixPath(row.relative_path).name
        try:
            receipt = verify_index_stage_bundle(
                target.resolve(strict=True),
                embedding_store_root=(artifact_root / "embedding-stores" / corpus_id),
                policy_bundle_root=(artifact_root / "policy-workloads" / corpus_id),
                expected_corpus_id=corpus_id,
            )
        except ArtifactStageBundleError as exc:
            raise FreezePackageError(
                f"invalid typed authorized-index stage bundle {row.artifact_id!r}: {exc}"
            ) from exc
        revision = f"sha256:{receipt.receipt_sha256}"
    if row.role == "power-analysis-report" and state == "present":
        if manifest is None:
            raise FreezePackageError("power-analysis bundle inspection requires the study manifest")
        verification = verify_joint_power_bundle(
            target.resolve(strict=True),
            manifest,
            expected_tree_sha256=tree.sha256,
        )
        revision = f"sha256:{verification.tree_sha256}"
    try:
        final_tree = digest_directory_tree(target.resolve(strict=True))
    except ArtifactIntegrityError as exc:
        raise FreezePackageError(
            f"cannot reverify directory artifact {row.artifact_id!r}: {exc}"
        ) from exc
    if final_tree != tree:
        raise FreezePackageError(
            f"directory artifact {row.artifact_id!r} changed during inspection"
        )
    return {
        "artifact_id": row.artifact_id,
        "byte_count": tree.byte_count,
        "directory_count": tree.directory_count,
        "file_count": tree.file_count,
        "kind": row.kind,
        "relative_path": row.relative_path,
        "revision": revision,
        "role": row.role,
        "sha256": tree.sha256,
        "source_path": row.source_path,
        "source_sha256": source_sha256,
        "state": state,
    }


def _pin_fields(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    missing: list[str] = []
    for field in ("uri", "revision", "sha256", "license"):
        if _is_placeholder(artifact.get(field)):
            missing.append(field)
    sha256 = artifact.get("sha256")
    if "sha256" not in missing and (
        not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
    ):
        missing.append("sha256")
    return tuple(missing)


def _readiness_report(
    manifest: Mapping[str, Any],
    layout: Sequence[FreezeArtifactLayout],
    artifact_root: Path,
    repository_root: Path,
    artifact_map_bytes: bytes,
) -> dict[str, Any]:
    manifest_artifacts = {
        str(value["id"]): value for value in manifest["artifacts"] if isinstance(value, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for layout_row in layout:
        row = _inspect_target(
            layout_row,
            artifact_root,
            repository_root,
            manifest,
        )
        pin_missing_fields = _pin_fields(manifest_artifacts[layout_row.artifact_id])
        manifest_pin = manifest_artifacts[layout_row.artifact_id].get("sha256")
        manifest_revision = manifest_artifacts[layout_row.artifact_id].get("revision")
        row["manifest_pin_missing_fields"] = list(pin_missing_fields)
        if pin_missing_fields or row["sha256"] is None:
            row["manifest_pin_matches_observed"] = None
        else:
            row["manifest_pin_matches_observed"] = manifest_pin == row["sha256"]
        if row["revision"] is None or _is_placeholder(manifest_revision):
            row["manifest_revision_matches_observed"] = None
        else:
            row["manifest_revision_matches_observed"] = manifest_revision == row["revision"]
        rows.append(row)

    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_role.setdefault(str(row["role"]), []).append(row)
    opa_binding: Mapping[str, Any] | None = None
    opa_rows = rows_by_role.get("opa-runtime-binary", [])
    plan_rows = rows_by_role.get("runtime-attestation-plan-template", [])
    if (
        len(opa_rows) == 1
        and len(plan_rows) == 5
        and opa_rows[0]["state"] == "present"
        and all(row["state"] == "present" for row in plan_rows)
    ):
        plan_paths = {
            PurePosixPath(str(row["relative_path"])).parts[-2]: artifact_root.joinpath(
                *PurePosixPath(str(row["relative_path"])).parts
            )
            for row in plan_rows
        }
        sealed = manifest.get("sealed_execution")
        if not isinstance(sealed, Mapping):
            raise FreezePackageError("manifest sealed_execution must be an object")
        image = sealed.get("runner_image")
        if _is_placeholder(image):
            first_plan = load_runtime_attestation_plan_template(next(iter(plan_paths.values())))
            image = first_plan.oci_image_digest
        assert isinstance(image, str)
        try:
            verification = verify_opa_runtime_binary(
                artifact_root / "runtime" / "opa",
                image=image,
                plan_paths=plan_paths,
            )
        except OpaRuntimeBinaryError as exc:
            raise FreezePackageError(f"OPA runtime binary binding is invalid: {exc}") from exc
        code_commit = sealed.get("code_commit")
        if not _is_placeholder(code_commit) and verification.code_commit != code_commit:
            raise FreezePackageError("OPA runtime plans name another C0 code commit")
        observed_plan_sha256 = {
            PurePosixPath(str(row["relative_path"])).parts[-2]: row["sha256"] for row in plan_rows
        }
        if dict(verification.plan_template_sha256_by_corpus) != observed_plan_sha256:
            raise FreezePackageError("OPA runtime verification plan digests changed during review")
        opa_binding = verification.to_dict()

    counts = {
        state: sum(row["state"] == state for row in rows)
        for state in ("missing", "generatable", "present")
    }
    incomplete_pins = sum(bool(row["manifest_pin_missing_fields"]) for row in rows)
    pin_mismatches = sum(row["manifest_pin_matches_observed"] is False for row in rows)
    revision_mismatches = sum(row["manifest_revision_matches_observed"] is False for row in rows)
    blockers = list(manifest.get("freeze_blockers", []))
    artifact_bytes_complete = counts["missing"] == 0 and counts["generatable"] == 0
    pins_complete = incomplete_pins == 0 and pin_mismatches == 0 and revision_mismatches == 0
    ready_for_freeze_review = artifact_bytes_complete and pins_complete and not blockers
    return {
        "artifact_count": len(rows),
        "artifact_map_sha256": hashlib.sha256(artifact_map_bytes).hexdigest(),
        "artifacts": rows,
        "c0_c1_sequence": list(_C0_C1_SEQUENCE),
        "circularity_controls": list(_CIRCULARITY_CONTROLS),
        "freeze_blocker_count": len(blockers),
        "freeze_blockers": blockers,
        "manifest_pin_incomplete_count": incomplete_pins,
        "manifest_pin_mismatch_count": pin_mismatches,
        "manifest_revision_mismatch_count": revision_mismatches,
        "manifest_sha256": manifest_sha256(manifest),
        "manifest_status": manifest["status"],
        "online_runner_access_contract": (
            "Exact map coverage is an inventory control, not an access grant. The online runner "
            "must open only its separately admitted whitelist and must never open sealed inputs, "
            "plaintext labels, or custody secrets."
        ),
        "opa_runtime_binding": opa_binding,
        "protocol_version": manifest["protocol_version"],
        "ready_for_freeze_review": ready_for_freeze_review,
        "schema_version": FREEZE_READINESS_SCHEMA,
        "sealed_run_authorized": False,
        "state_counts": counts,
        "warning": (
            "Typed roles are structurally inspected; other roles retain presence-and-hash "
            "semantics. This compiler does not freeze the manifest, register the protocol, "
            "release labels, or consume the run."
        ),
    }


def compile_freeze_package(
    manifest_path: str | Path,
    repository_root: str | Path,
    package_root: str | Path,
    *,
    copy_code: bool = True,
    refresh_code: bool = False,
) -> dict[str, Any]:
    """Write the controlled map and readiness report, optionally copying code."""

    repository = Path(repository_root).expanduser().resolve(strict=True)
    manifest_file = Path(manifest_path).expanduser()
    if not manifest_file.is_absolute():
        manifest_file = repository / manifest_file
    manifest = load_study_manifest(manifest_file.resolve(strict=True))
    validate_study_manifest(manifest)
    package = Path(package_root).expanduser().resolve(strict=False)
    try:
        package.relative_to(repository)
    except ValueError:
        pass
    else:
        raise FreezePackageError("package root must be outside the source repository")
    if package.exists() and package.is_symlink():
        raise FreezePackageError("package root cannot be a symlink")
    _ensure_directory(package)
    artifact_root = package / "artifacts"
    _ensure_directory(artifact_root)

    layout = layout_from_manifest(manifest, repository)
    for row in layout:
        target = artifact_root.joinpath(*PurePosixPath(row.relative_path).parts)
        _ensure_directory(target.parent)
        if copy_code and row.source_path is not None:
            source = repository.joinpath(*PurePosixPath(row.source_path).parts)
            _copy_code_artifact(source, target, refresh=refresh_code)

    map_payload = artifact_map_payload(layout)
    map_bytes = _canonical_json_bytes(map_payload)
    map_path = package / "artifact-map.json"
    _write_atomic(map_path, map_bytes)
    validate_freeze_artifact_map(manifest, repository, map_path)

    report = _readiness_report(
        manifest,
        layout,
        artifact_root,
        repository,
        map_bytes,
    )
    _write_atomic(package / "freeze-readiness.json", _canonical_json_bytes(report))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.freeze_package",
        description="prepare or validate a non-consuming confirmatory freeze package",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser(
        "compile",
        help="write the controlled layout, code copies, artifact map, and readiness report",
    )
    compile_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/study-manifest.json"),
    )
    compile_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    compile_parser.add_argument("--package-root", type=Path, required=True)
    compile_parser.add_argument(
        "--no-copy-code",
        action="store_true",
        help="report locally generatable code artifacts without copying them",
    )
    compile_parser.add_argument(
        "--refresh-code",
        action="store_true",
        help="replace a controlled code copy when it differs from its repository source",
    )

    validate_parser = subparsers.add_parser(
        "validate-map",
        help="validate exact draft-manifest artifact coverage without requiring frozen pins",
    )
    validate_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/study-manifest.json"),
    )
    validate_parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    validate_parser.add_argument("--artifact-map", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository_root.expanduser().resolve(strict=True)
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repository / manifest_path
    if args.command == "compile":
        report = compile_freeze_package(
            manifest_path,
            repository,
            args.package_root,
            copy_code=not args.no_copy_code,
            refresh_code=args.refresh_code,
        )
        print(f"mapped {report['artifact_count']} manifest artifacts")
        print(
            "states: "
            + ", ".join(
                f"{name}={report['state_counts'][name]}"
                for name in ("present", "generatable", "missing")
            )
        )
        print(f"ready for freeze review: {str(report['ready_for_freeze_review']).lower()}")
        print(f"package: {args.package_root.expanduser().resolve(strict=False)}")
        return 0
    if args.command == "validate-map":
        manifest = load_study_manifest(manifest_path.resolve(strict=True))
        layout = validate_freeze_artifact_map(
            manifest,
            repository,
            args.artifact_map,
        )
        print(f"valid exact artifact-map coverage: {len(layout)} artifacts")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
