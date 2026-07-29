"""Hosted, label-payload-excluded prerequisite admission for provider phase claims.

The hosted claim job may materialize portable evidence, but it cannot confer
authority by serializing that evidence.  This module composes the existing C1,
runner, and protected-ledger verifiers and returns one in-memory capability.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .artifact_integrity import (
    digest_directory_tree,
    read_secure_control_file,
    write_exclusive_receipt_bytes,
)
from .execution_claim import (
    C1_REGISTRATION_PACKAGE_FILE_COUNT,
    AnonymousZenodoAdmission,
    GitHubReadApi,
    ProviderPhase,
    ProviderPhasePlan,
    ProviderRunnerBootstrapReceipt,
    ProviderRunnerReadinessReceipt,
    load_provider_phase_plans,
    materialize_provider_phase_plan,
    provider_phase_plan_templates_sha256,
    verify_provider_runner_ready,
)
from .github_state_attestation import (
    COMMON_CONTROL_LIMITATION,
    REGISTRY_ATTESTATION_RECEIPT_SCHEMA,
    REPOSITORY,
    ZENODO_RECORD_ID,
    ZENODO_REGISTRY_URI,
    ZENODO_RESERVED_DOI,
    C1AttestationVerifier,
    GitHubApi,
    LedgerSnapshot,
    load_ledger_snapshot,
)
from .provider_state_transport import (
    MaterializedProviderPredecessor,
    ProviderStateArtifactReadApi,
    materialize_provider_predecessor,
)
from .study import (
    ProtocolRegistrationReceipt,
    ProtocolRegistryRecord,
    VerifiedC1ProtocolRegistration,
    load_study_manifest,
)
from .suite_attempt import (
    SuiteAttestationDescriptor,
    SuiteOpenBindings,
)
from .suite_attempt import (
    suite_attempt_id as derive_suite_attempt_id,
)
from .zenodo_publication import (
    PACKAGE_FILE_NAMES,
    ZENODO_PUBLIC_API_URI,
    ValidatedRegistrationPackage,
    _public_file_uri,
    _verify_production_protocol_registration,
    _verify_public_payload,
    _ZenodoHttpsTransport,
    validate_registration_package,
)
from .zenodo_publication import (
    _inventory as _zenodo_inventory,
)

if TYPE_CHECKING:
    from .provider_workflow_orchestration import ProviderWorkflowContext

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_PACKAGE_BYTES = 128 * 1024 * 1024
_READBACK_SCHEMA = "fractal-zenodo-anonymous-readback-v2"
_REGISTRY_ATTESTATION_FIELDS = frozenset(
    {
        "c1_commit",
        "control_boundary",
        "manifest_rekor_entry_id",
        "manifest_rekor_integrated_at_utc",
        "registry_record_bundle_sha256",
        "registry_record_rekor_entry_id",
        "registry_record_rekor_integrated_at_utc",
        "registry_record_sha256",
        "registry_record_verification_sha256",
        "schema_version",
    }
)
_CAPABILITY = object()
_PREDECESSOR = {
    "online": ("OPENED", 0),
    "label-release": ("ONLINE_COMPLETE", 2),
    "analysis": ("LABELS_RELEASED", 4),
}


class HostedPrerequisiteError(ValueError):
    """Hosted prerequisite evidence is absent, mutable, or cross-bound incorrectly."""


class ZenodoPublicReadApi(Protocol):
    def get_json(self, url: str, *, authenticated: bool) -> Mapping[str, Any]: ...

    def get_bytes(self, url: str, *, authenticated: bool) -> bytes: ...


PhasePlanLoader = Callable[..., Mapping[ProviderPhase, ProviderPhasePlan]]
ManifestLoader = Callable[[str | Path], Mapping[str, Any]]
PhasePlanMaterializer = Callable[[ProviderPhasePlan, str | Path], Path]
RunnerReadinessVerifier = Callable[..., ProviderRunnerReadinessReceipt]
PredecessorMaterializer = Callable[..., MaterializedProviderPredecessor]
SnapshotLoader = Callable[..., LedgerSnapshot]


@dataclass(frozen=True)
class HostedPrerequisiteServices:
    """Pure orchestration seams; production uses the fixed module functions below."""

    phase_plan_loader: PhasePlanLoader = field(repr=False)
    manifest_loader: ManifestLoader = field(repr=False)
    phase_plan_materializer: PhasePlanMaterializer = field(repr=False)
    runner_readiness_verifier: RunnerReadinessVerifier = field(repr=False)
    predecessor_materializer: PredecessorMaterializer = field(repr=False)
    snapshot_loader: SnapshotLoader = field(repr=False)
    plan_templates_hasher: Callable[[Mapping[str, Any]], str] = field(repr=False)

    def __post_init__(self) -> None:
        if not all(
            callable(value)
            for value in (
                self.phase_plan_loader,
                self.manifest_loader,
                self.phase_plan_materializer,
                self.runner_readiness_verifier,
                self.predecessor_materializer,
                self.snapshot_loader,
                self.plan_templates_hasher,
            )
        ):
            raise HostedPrerequisiteError("hosted prerequisite services must be callable")


PRODUCTION_SERVICES = HostedPrerequisiteServices(
    phase_plan_loader=load_provider_phase_plans,
    manifest_loader=load_study_manifest,
    phase_plan_materializer=materialize_provider_phase_plan,
    runner_readiness_verifier=verify_provider_runner_ready,
    predecessor_materializer=materialize_provider_predecessor,
    snapshot_loader=load_ledger_snapshot,
    plan_templates_hasher=provider_phase_plan_templates_sha256,
)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HostedPrerequisiteError("prerequisite evidence is not canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise HostedPrerequisiteError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _commit(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA1.fullmatch(value) is None:
        raise HostedPrerequisiteError(f"{label} must be a lowercase Git SHA-1")
    return value


def _timestamp(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise HostedPrerequisiteError(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HostedPrerequisiteError(f"{label} is not an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise HostedPrerequisiteError(f"{label} must carry a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat()


def _controlled_new_root(path: Path) -> Path:
    if not path.is_absolute() or path.exists():
        raise HostedPrerequisiteError("prerequisite output root must be new and absolute")
    try:
        parent = path.parent.resolve(strict=True)
        metadata = path.parent.lstat()
    except OSError as exc:
        raise HostedPrerequisiteError("prerequisite output parent is unavailable") from exc
    if (
        parent != path.parent
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise HostedPrerequisiteError("prerequisite output parent is not controlled")
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise HostedPrerequisiteError("cannot create prerequisite output root") from exc
    return path


def _write_once(path: Path, encoded: bytes, *, label: str) -> None:
    try:
        write_exclusive_receipt_bytes(encoded, path)
        observed = read_secure_control_file(path, label=label)
    except Exception as exc:
        raise HostedPrerequisiteError(f"cannot materialize {label} exactly once") from exc
    if observed != encoded:
        raise HostedPrerequisiteError(f"{label} failed exact readback")


def _package_inventory(package: ValidatedRegistrationPackage) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": item.name,
            "sha256": item.sha256,
            "size_bytes": item.size,
        }
        for item in package.files
    )


def _package_fingerprint(package: ValidatedRegistrationPackage) -> str:
    return _sha256(
        _canonical_bytes(
            {
                "c0_commit": package.c0_commit,
                "c1_commit": package.c1_commit,
                "files": _package_inventory(package),
                "manifest_sha256": package.manifest_sha256,
                "registry_record_sha256": package.registry_record_sha256,
            }
        )
    )


def _closed_json_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise HostedPrerequisiteError(f"{label} repeats field {name!r}")
            result[name] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise HostedPrerequisiteError(f"{label} contains non-finite number {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HostedPrerequisiteError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping) or any(type(name) is not str for name in value):
        raise HostedPrerequisiteError(f"{label} must be a JSON object")
    return value


def _read_registry_rekor_times(
    package: ValidatedRegistrationPackage,
    registration: VerifiedC1ProtocolRegistration,
) -> tuple[str, str, str]:
    name = "registry-attestation-validation.json"
    try:
        encoded = read_secure_control_file(
            package.root / name,
            label="C1 registry attestation receipt",
        )
    except Exception as exc:
        raise HostedPrerequisiteError("cannot read the C1 registry attestation receipt") from exc
    if _sha256(encoded) != package.inventory[name].sha256:
        raise HostedPrerequisiteError("C1 registry attestation receipt changed")
    value = _closed_json_object(encoded, label="C1 registry attestation receipt")
    if set(value) != _REGISTRY_ATTESTATION_FIELDS or encoded != _canonical_bytes(value) + b"\n":
        raise HostedPrerequisiteError(
            "C1 registry attestation receipt is not the exact canonical schema"
        )
    if (
        value.get("schema_version") != REGISTRY_ATTESTATION_RECEIPT_SCHEMA
        or value.get("control_boundary") != COMMON_CONTROL_LIMITATION
        or value.get("c1_commit") != package.c1_commit
        or value.get("registry_record_sha256") != package.registry_record_sha256
    ):
        raise HostedPrerequisiteError("C1 registry attestation receipt identity differs")
    for field_name in (
        "manifest_rekor_entry_id",
        "registry_record_rekor_entry_id",
    ):
        field_value = value.get(field_name)
        if type(field_value) is not str or not field_value:
            raise HostedPrerequisiteError(
                f"C1 registry attestation receipt {field_name} is malformed"
            )
    for field_name in (
        "registry_record_bundle_sha256",
        "registry_record_verification_sha256",
    ):
        _digest(value.get(field_name), label=field_name)
    manifest_time = _timestamp(
        value.get("manifest_rekor_integrated_at_utc"),
        label="manifest Rekor integrated time",
    )
    registry_time = _timestamp(
        value.get("registry_record_rekor_integrated_at_utc"),
        label="registry-record Rekor integrated time",
    )
    registered_at = _timestamp(
        registration.record.registered_at_utc,
        label="protocol registration time",
    )
    if manifest_time != registered_at or datetime.fromisoformat(
        registry_time
    ) < datetime.fromisoformat(manifest_time):
        raise HostedPrerequisiteError(
            "C1 Rekor integration times differ from the protocol registration"
        )
    return manifest_time, registry_time, _sha256(encoded)


def _public_payload_identity(payload: object) -> tuple[Mapping[str, Any], str]:
    if not isinstance(payload, Mapping):
        raise HostedPrerequisiteError("Zenodo public readback is not a JSON object")
    if (
        payload.get("id") != ZENODO_RECORD_ID
        or type(payload.get("id")) is not int
        or payload.get("doi") != ZENODO_RESERVED_DOI
        or payload.get("submitted") is not True
        or payload.get("state") != "done"
        or payload.get("status") != "published"
    ):
        raise HostedPrerequisiteError("Zenodo public readback differs from record 21361837")
    published_at = _timestamp(payload.get("created"), label="Zenodo publication time")
    try:
        rows = _zenodo_inventory(
            payload.get("files"),
            label="Zenodo public files",
            require_public_content_links=True,
        )
    except Exception as exc:
        raise HostedPrerequisiteError("Zenodo public file inventory is malformed") from exc
    if set(rows) != set(PACKAGE_FILE_NAMES) or len(rows) != C1_REGISTRATION_PACKAGE_FILE_COUNT:
        raise HostedPrerequisiteError(
            "Zenodo public inventory is not the exact "
            f"{C1_REGISTRATION_PACKAGE_FILE_COUNT}-file package"
        )
    return payload, published_at


def materialize_anonymous_c1_package(
    destination: Path,
    *,
    transport: ZenodoPublicReadApi,
) -> tuple[ValidatedRegistrationPackage, Mapping[str, Any], str]:
    """Fetch the fixed public package anonymously, then validate it twice."""

    if not destination.is_absolute() or destination.exists():
        raise HostedPrerequisiteError("C1 package destination must be a new absolute path")
    try:
        first_payload, published_at = _public_payload_identity(
            transport.get_json(ZENODO_PUBLIC_API_URI, authenticated=False)
        )
    except HostedPrerequisiteError:
        raise
    except Exception as exc:
        raise HostedPrerequisiteError("cannot read the anonymous Zenodo record") from exc
    staging = Path(tempfile.mkdtemp(prefix=".c1-package-", dir=destination.parent))
    os.chmod(staging, 0o700)
    total = 0
    try:
        for name in PACKAGE_FILE_NAMES:
            try:
                encoded = transport.get_bytes(_public_file_uri(name), authenticated=False)
            except Exception as exc:
                raise HostedPrerequisiteError(
                    f"cannot fetch anonymous Zenodo file {name!r}"
                ) from exc
            if not isinstance(encoded, bytes) or len(encoded) > _MAX_FILE_BYTES:
                raise HostedPrerequisiteError(f"Zenodo file {name!r} exceeds its bound")
            total += len(encoded)
            if total > _MAX_PACKAGE_BYTES:
                raise HostedPrerequisiteError("Zenodo package exceeds its total byte bound")
            _write_once(staging / name, encoded, label=f"Zenodo file {name}")
        staged = validate_registration_package(staging)
        _verify_public_payload(staged, first_payload)
        try:
            final_payload, final_published_at = _public_payload_identity(
                transport.get_json(ZENODO_PUBLIC_API_URI, authenticated=False)
            )
        except Exception as exc:
            raise HostedPrerequisiteError("cannot reread the anonymous Zenodo record") from exc
        _verify_public_payload(staged, final_payload)
        if final_payload != first_payload or final_published_at != published_at:
            raise HostedPrerequisiteError("Zenodo public record changed during materialization")
        os.replace(staging, destination)
        admitted = validate_registration_package(destination)
        if _package_fingerprint(admitted) != _package_fingerprint(staged):
            raise HostedPrerequisiteError("C1 package changed during final materialization")
        return admitted, final_payload, published_at
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise


@dataclass(frozen=True)
class AnonymousZenodoReadbackReceipt:
    record_id: int
    doi: str
    record_uri: str
    published_at_utc: str
    verified_at_utc: str
    file_count: int
    package_tree_sha256: str
    package_aggregate_sha256: str
    public_payload_sha256: str
    schema_version: str = _READBACK_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.record_id != ZENODO_RECORD_ID
            or self.doi != ZENODO_RESERVED_DOI
            or self.record_uri != ZENODO_REGISTRY_URI
            or self.file_count != C1_REGISTRATION_PACKAGE_FILE_COUNT
        ):
            raise HostedPrerequisiteError("anonymous Zenodo receipt identity differs")
        published = _timestamp(self.published_at_utc, label="Zenodo published_at_utc")
        verified = _timestamp(self.verified_at_utc, label="Zenodo verified_at_utc")
        if datetime.fromisoformat(verified) < datetime.fromisoformat(published):
            raise HostedPrerequisiteError("Zenodo verification predates publication")
        for name in (
            "package_tree_sha256",
            "package_aggregate_sha256",
            "public_payload_sha256",
        ):
            _digest(getattr(self, name), label=name)
        if self.schema_version != _READBACK_SCHEMA:
            raise HostedPrerequisiteError("anonymous Zenodo receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())


def _materialize_registration_receipt(
    package: ValidatedRegistrationPackage,
    target: Path,
) -> ProtocolRegistrationReceipt:
    try:
        payload = json.loads(package.registry_record_bytes.decode("utf-8", errors="strict"))
        record = ProtocolRegistryRecord.from_dict(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HostedPrerequisiteError("C1 registry record cannot create a local receipt") from exc
    if (
        record.canonical_bytes() + b"\n" != package.registry_record_bytes
        or record.manifest_sha256 != package.manifest_sha256
        or record.record_sha256 != package.registry_record_sha256
    ):
        raise HostedPrerequisiteError("C1 registry record differs from the admitted package")
    receipt = ProtocolRegistrationReceipt(
        manifest_sha256=record.manifest_sha256,
        protocol_version=record.protocol_version,
        registered_at_utc=record.registered_at_utc,
        registry_identity=record.registry_identity,
        registry_uri=record.registry_uri,
        registry_record_sha256=record.record_sha256,
    )
    _write_once(
        target,
        receipt.canonical_bytes() + b"\n",
        label="protocol registration receipt",
    )
    return receipt


def _load_hosted_plan(path: Path, expected: ProviderPhasePlan) -> ProviderPhasePlan:
    try:
        encoded = read_secure_control_file(path, label="hosted provider plan")
        value = json.loads(encoded.decode("utf-8", errors="strict"))
        observed = ProviderPhasePlan.from_dict(value)
    except Exception as exc:
        raise HostedPrerequisiteError("hosted provider plan is malformed") from exc
    if encoded != expected.canonical_file_bytes() or observed != expected:
        raise HostedPrerequisiteError("hosted provider plan differs from C1")
    return observed


def _load_hosted_bootstrap(
    path: Path,
    expected: ProviderRunnerBootstrapReceipt,
) -> ProviderRunnerBootstrapReceipt:
    try:
        encoded = read_secure_control_file(path, label="hosted runner bootstrap")
        value = json.loads(encoded.decode("utf-8", errors="strict"))
        observed = ProviderRunnerBootstrapReceipt.from_dict(value)
    except Exception as exc:
        raise HostedPrerequisiteError("hosted runner bootstrap is malformed") from exc
    if encoded != expected.canonical_file_bytes() or observed != expected:
        raise HostedPrerequisiteError("hosted runner bootstrap differs from C1")
    return observed


def _artifact_inventory_sha256(value: MaterializedProviderPredecessor) -> str:
    rows = [
        {
            "artifact_id": row.artifact_id,
            "inventory_name": row.inventory_name,
            "inventory_sha256": row.inventory_sha256,
        }
        for row in value.receipt.artifacts
    ]
    return _sha256(_canonical_bytes(rows))


def _assert_snapshot(
    snapshot: LedgerSnapshot,
    *,
    phase: ProviderPhase,
    suite: str,
    package: ValidatedRegistrationPackage,
    registration: VerifiedC1ProtocolRegistration,
    predecessor: MaterializedProviderPredecessor,
) -> str:
    expected_state, expected_sequence = _PREDECESSOR[phase]
    tip = snapshot.tip
    genesis = snapshot.transitions[0].state
    opening = genesis.payload
    descriptor_rows = [
        control for control in snapshot.controls if control.role == "attestation-descriptor"
    ]
    try:
        descriptor_payload = json.loads(descriptor_rows[0].encoded.decode("utf-8", errors="strict"))
        descriptor = SuiteAttestationDescriptor.from_dict(descriptor_payload)
    except (IndexError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HostedPrerequisiteError(
            "protected ledger attestation descriptor is malformed"
        ) from exc
    if (
        len(descriptor_rows) != 1
        or descriptor_rows[0].encoded != descriptor.canonical_bytes() + b"\n"
        or descriptor.descriptor_sha256 != getattr(opening, "attestation_descriptor_sha256", None)
        or descriptor.expected_signer_digest != package.c0_commit
    ):
        raise HostedPrerequisiteError("protected ledger attestation descriptor differs from C0")
    if (
        not isinstance(opening, SuiteOpenBindings)
        or opening.code_commit != package.c0_commit
        or opening.protocol_registration_receipt_sha256 != registration.receipt.receipt_sha256
        or opening.protocol_registration_receipt_file_sha256
        != _sha256(registration.receipt.canonical_bytes() + b"\n")
        or opening.protocol_registry_record_sha256 != package.registry_record_sha256
        or opening.registered_at_utc != registration.record.registered_at_utc
    ):
        raise HostedPrerequisiteError("OPENED ledger genesis differs from C1 registration")
    if (
        snapshot.repository != REPOSITORY
        or tip.state.suite_attempt_id != suite
        or tip.state.manifest_sha256 != package.manifest_sha256
        or tip.state.state != expected_state
        or tip.state.sequence != expected_sequence
        or tip.commit_oid != predecessor.predecessor.ledger_commit
        or tip.state.record_sha256 != predecessor.predecessor.state.record_sha256
        or len(snapshot.transitions) != expected_sequence + 1
    ):
        raise HostedPrerequisiteError("protected ledger snapshot differs from the predecessor")
    if _sha256(snapshot.control_inventory_bytes) != predecessor.receipt.control_inventory_sha256:
        raise HostedPrerequisiteError("protected ledger controls differ from predecessor evidence")
    return _commit(tip.tree_oid, label="predecessor ledger tree")


@dataclass(frozen=True)
class HostedProductionPrerequisites:
    """Non-serializable hosted admission; persisted derivatives remain evidence only."""

    context: object
    phase: ProviderPhase
    suite_attempt_id: str
    workflow_context_sha256: str
    package: ValidatedRegistrationPackage
    package_inventory_sha256: str
    registration: VerifiedC1ProtocolRegistration
    manifest_rekor_integrated_at_utc: str
    registry_record_rekor_integrated_at_utc: str
    zenodo_admission: AnonymousZenodoAdmission
    zenodo_readback_receipt_path: Path
    plan: ProviderPhasePlan
    plan_materialization_path: Path
    plan_templates_sha256: str
    bootstrap: ProviderRunnerBootstrapReceipt
    bootstrap_materialization_path: Path
    runner_readiness: ProviderRunnerReadinessReceipt
    predecessor: MaterializedProviderPredecessor
    predecessor_ledger_tree: str
    predecessor_artifact_inventory_sha256: str
    _fresh_revalidator: Callable[[], None] = field(repr=False, compare=False)
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _CAPABILITY:
            raise HostedPrerequisiteError("hosted prerequisites were not minted by the factory")
        _digest(self.suite_attempt_id, label="suite_attempt_id")
        _digest(self.workflow_context_sha256, label="workflow_context_sha256")
        _digest(self.package_inventory_sha256, label="package_inventory_sha256")
        _digest(self.plan_templates_sha256, label="plan_templates_sha256")
        _commit(self.predecessor_ledger_tree, label="predecessor_ledger_tree")
        _digest(
            self.predecessor_artifact_inventory_sha256,
            label="predecessor_artifact_inventory_sha256",
        )
        manifest_time = _timestamp(
            self.manifest_rekor_integrated_at_utc,
            label="manifest_rekor_integrated_at_utc",
        )
        registry_time = _timestamp(
            self.registry_record_rekor_integrated_at_utc,
            label="registry_record_rekor_integrated_at_utc",
        )
        if (
            self.phase not in _PREDECESSOR
            or self.plan.phase != self.phase
            or self.plan.suite_attempt_id != self.suite_attempt_id
            or self.package.manifest_sha256 != self.plan.manifest_sha256
            or self.registration.manifest_sha256 != self.plan.manifest_sha256
            or self.bootstrap != self.plan.runner_bootstrap_receipt
            or self.runner_readiness.provider_plan_sha256 != self.plan.plan_sha256
            or self.predecessor.predecessor.state.suite_attempt_id != self.suite_attempt_id
            or manifest_time
            != _timestamp(
                self.registration.record.registered_at_utc,
                label="protocol registration time",
            )
            or datetime.fromisoformat(registry_time) < datetime.fromisoformat(manifest_time)
        ):
            raise HostedPrerequisiteError("hosted prerequisite capabilities are cross-bound poorly")
        if not callable(self._fresh_revalidator):
            raise HostedPrerequisiteError("hosted prerequisites lack a fresh revalidator")
        self.assert_current()

    def assert_current(self) -> None:
        try:
            result = self._fresh_revalidator()
        except HostedPrerequisiteError:
            raise
        except Exception as exc:
            raise HostedPrerequisiteError(
                f"hosted prerequisite revalidation failed: {exc}"
            ) from exc
        if result is not None:
            raise HostedPrerequisiteError("hosted prerequisite revalidator returned data")

    def prerequisite_fields(self) -> dict[str, object]:
        state = self.predecessor.predecessor.state
        receipt = self.predecessor.receipt
        return {
            "phase": self.phase,
            "suite_attempt_id": self.suite_attempt_id,
            "manifest_sha256": self.package.manifest_sha256,
            "c1_commit": self.package.c1_commit,
            "c1_package_root": str(self.package.root),
            "c1_package_inventory_sha256": self.package_inventory_sha256,
            "c1_package_file_count": len(self.package.files),
            "zenodo_admission_sha256": self.zenodo_admission.receipt_sha256,
            "provider_plan_sha256": self.plan.plan_sha256,
            "provider_plan_file_sha256": self.plan.file_sha256,
            "provider_plan_materialization_path": str(self.plan_materialization_path),
            "provider_plan_templates_sha256": self.plan_templates_sha256,
            "runner_bootstrap_receipt_path": self.plan.runner_bootstrap_receipt_path,
            "runner_bootstrap_receipt_file_sha256": self.bootstrap.file_sha256,
            "runner_readiness_receipt_sha256": self.runner_readiness.receipt_sha256,
            "predecessor_state": state.state,
            "predecessor_sequence": state.sequence,
            "predecessor_state_record_sha256": state.record_sha256,
            "predecessor_ledger_commit": self.predecessor.predecessor.ledger_commit,
            "predecessor_ledger_tree": self.predecessor_ledger_tree,
            "predecessor_control_inventory_sha256": receipt.control_inventory_sha256,
            "predecessor_artifact_receipt_sha256": receipt.receipt_sha256,
            "predecessor_artifact_inventory_sha256": (self.predecessor_artifact_inventory_sha256),
            "predecessor_artifact_materialized_root": receipt.materialized_root,
            "workflow_context_sha256": self.workflow_context_sha256,
            "phase_evidence_root": self.plan.phase_evidence_root(self.suite_attempt_id),
        }

    def execution_output_fields(self) -> dict[str, str]:
        host = self.plan.host_tools
        return {
            "docker_client_build": host.docker_client_build,
            "docker_client_version": host.docker_client_version,
            "docker_file_sha256": host.docker_executable_sha256,
            "docker_path": host.docker_executable,
            "docker_resolved_path": host.docker_resolved_executable,
            "gh_file_sha256": host.gh_executable_sha256,
            "gh_path": host.gh_executable,
            "gh_version": host.gh_version,
            "host_controlled_root": host.controlled_root,
            "host_python_file_sha256": host.python_executable_sha256,
            "host_python_import_root": host.python_import_root,
            "host_python_import_tree_sha256": host.python_import_tree_sha256,
            "host_python_launcher_sha256": host.python_launcher_sha256,
            "host_python_package_content_sha256": host.python_package_content_sha256,
            "host_python_package_source_commit": host.python_package_source_commit,
            "host_python_package_source_tree": host.python_package_source_tree,
            "host_python_package_tree_sha256": host.python_package_tree_sha256,
            "host_python_path": host.python_executable,
            "host_python_venv_root": host.venv_root,
            "host_python_venv_symlink_inventory_sha256": (host.venv_symlink_inventory_sha256),
            "host_python_venv_tree_sha256": host.venv_tree_sha256,
            "oci_index_digest": self.plan.oci_index_digest,
            "oci_platform_manifest_digest": self.plan.oci_platform_manifest_digest,
            "phase_evidence_root": self.plan.phase_evidence_root(self.suite_attempt_id),
            "provider_plan_file_sha256": self.plan.file_sha256,
            "provider_plan_materialization_path": str(self.plan_materialization_path),
            "provider_plan_path": self.plan.provider_plan_path,
            "runner_bootstrap_receipt_file_sha256": self.bootstrap.file_sha256,
            "runner_bootstrap_receipt_path": self.plan.runner_bootstrap_receipt_path,
            "runner_listener_file_sha256": host.runner_listener_sha256,
            "runner_listener_path": host.runner_listener_executable,
            "runtime_image": self.plan.runtime_image,
            "runtime_image_role": self.plan.runtime_image_role,
            "runtime_index_role": self.plan.runtime_index_role,
            "runtime_platform": self.plan.runtime_platform,
            "tle_binary_sha256": self.plan.tle_binary_sha256 or "",
        }


def build_hosted_production_prerequisites(
    context: ProviderWorkflowContext,
    phase: ProviderPhase,
    suite_attempt_id: str,
    output_root: Path,
    *,
    verified_at_utc: str,
    runner_api: GitHubReadApi,
    ledger_api: GitHubApi,
    artifact_api: ProviderStateArtifactReadApi,
    zenodo_transport: ZenodoPublicReadApi | None = None,
    c1_attestation_verifier: C1AttestationVerifier | None = None,
    services: HostedPrerequisiteServices = PRODUCTION_SERVICES,
) -> HostedProductionPrerequisites:
    """Build and freshly reverify the hosted claim-job prerequisite capability."""

    from .provider_workflow_orchestration import ProviderWorkflowContext

    if (
        not isinstance(context, ProviderWorkflowContext)
        or context.job != "claim"
        or context.phase != phase
    ):
        raise HostedPrerequisiteError("factory requires the admitted hosted claim context")
    if phase not in _PREDECESSOR:
        raise HostedPrerequisiteError("provider phase is not admitted")
    suite = _digest(suite_attempt_id, label="suite_attempt_id")
    verified_at = _timestamp(verified_at_utc, label="verified_at_utc")
    root = _controlled_new_root(Path(output_root))
    owned_transport = zenodo_transport is None
    transport: ZenodoPublicReadApi = (
        _ZenodoHttpsTransport(None) if zenodo_transport is None else zenodo_transport
    )
    try:
        package, public_payload, published_at = materialize_anonymous_c1_package(
            root / "c1-package",
            transport=transport,
        )
        if suite != derive_suite_attempt_id(package.manifest_sha256):
            raise HostedPrerequisiteError("suite-attempt ID differs from the C1 manifest")
        receipt_path = root / "protocol-registration-receipt.json"
        _materialize_registration_receipt(package, receipt_path)
        registration = _verify_production_protocol_registration(
            package.root,
            registration_record_path=package.root / "protocol-registry-record.json",
            registration_receipt_path=receipt_path,
            verifier=c1_attestation_verifier,
            transport=None if owned_transport else transport,
        )
        (
            manifest_rekor_integrated_at,
            registry_record_rekor_integrated_at,
            registry_attestation_file_sha256,
        ) = _read_registry_rekor_times(package, registration)
        manifest = services.manifest_loader(package.root / "study-manifest.json")
        plans = services.phase_plan_loader(
            package.root / "study-manifest.json",
            c1_commit=package.c1_commit,
        )
        if not isinstance(plans, Mapping) or set(plans) != set(_PREDECESSOR):
            raise HostedPrerequisiteError("C1 does not resolve exactly three provider plans")
        plan = plans[phase]
        if (
            not isinstance(plan, ProviderPhasePlan)
            or plan.manifest_sha256 != package.manifest_sha256
            or plan.c1_commit != package.c1_commit
            or plan.workflow_sha != package.c0_commit
            or plan.phase != context.phase
            or plan.workflow_path != context.workflow_path
            or plan.workflow_ref != context.workflow_ref
            or plan.workflow_sha != context.workflow_sha
            or plan.suite_attempt_id != suite
        ):
            raise HostedPrerequisiteError("C1 provider plan differs from package or workflow")
        templates_digest = _digest(
            services.plan_templates_hasher(manifest),
            label="provider plan templates SHA-256",
        )
        plan_path = services.phase_plan_materializer(plan, root / "provider-plan")
        _load_hosted_plan(plan_path, plan)
        bootstrap_path = root / "runner-bootstrap-receipt.json"
        _write_once(
            bootstrap_path,
            plan.runner_bootstrap_receipt.canonical_file_bytes(),
            label="hosted runner bootstrap",
        )
        bootstrap = _load_hosted_bootstrap(bootstrap_path, plan.runner_bootstrap_receipt)
        # Reconstructing the typed contract repeats every official archive,
        # executable, platform, and probe pin in PhaseHostToolContract.__post_init__.
        if type(plan.host_tools).from_dict(plan.host_tools.to_dict()) != plan.host_tools:
            raise HostedPrerequisiteError("C1 host-tool closure is not self-consistent")
        readiness = services.runner_readiness_verifier(
            plan=plan,
            api=runner_api,
            verified_at_utc=verified_at,
        )
        state_parent = root / "provider-state"
        state_parent.mkdir(mode=0o700)
        predecessor = services.predecessor_materializer(
            phase,
            suite,
            state_parent,
            ledger_api=ledger_api,
            artifact_api=artifact_api,
        )
        snapshot = services.snapshot_loader(
            repository=REPOSITORY,
            suite_attempt_id=suite,
            api=ledger_api,
        )
        ledger_tree = _assert_snapshot(
            snapshot,
            phase=phase,
            suite=suite,
            package=package,
            registration=registration,
            predecessor=predecessor,
        )
        package_inventory = _package_inventory(package)
        package_inventory_digest = _sha256(_canonical_bytes(package_inventory))
        package_tree = digest_directory_tree(package.root)
        if (
            package_tree.file_count != C1_REGISTRATION_PACKAGE_FILE_COUNT
            or package_tree.directory_count != 0
            or package_tree.observed_file_count != C1_REGISTRATION_PACKAGE_FILE_COUNT
            or package_tree.observed_directory_count != 0
            or package_tree.entries
            != tuple(sorted(PACKAGE_FILE_NAMES, key=lambda value: value.encode("utf-8")))
        ):
            raise HostedPrerequisiteError(
                "materialized C1 package tree is not exactly "
                f"{C1_REGISTRATION_PACKAGE_FILE_COUNT} files"
            )
        readback = AnonymousZenodoReadbackReceipt(
            record_id=ZENODO_RECORD_ID,
            doi=ZENODO_RESERVED_DOI,
            record_uri=ZENODO_REGISTRY_URI,
            published_at_utc=published_at,
            verified_at_utc=verified_at,
            file_count=C1_REGISTRATION_PACKAGE_FILE_COUNT,
            package_tree_sha256=package_tree.sha256,
            package_aggregate_sha256=package_inventory_digest,
            public_payload_sha256=_sha256(_canonical_bytes(public_payload)),
        )
        readback_path = root / "zenodo-anonymous-readback.json"
        _write_once(
            readback_path,
            readback.canonical_file_bytes(),
            label="anonymous Zenodo readback receipt",
        )
        admission = AnonymousZenodoAdmission(
            record_id=ZENODO_RECORD_ID,
            doi=ZENODO_RESERVED_DOI,
            record_uri=ZENODO_REGISTRY_URI,
            published_at_utc=published_at,
            file_count=C1_REGISTRATION_PACKAGE_FILE_COUNT,
            package_tree_sha256=package_tree.sha256,
            package_aggregate_sha256=package_inventory_digest,
            receipt_file_sha256=readback.file_sha256,
            verified_at_utc=verified_at,
        )
        context_digest = context.identity_sha256
        package_fingerprint = _package_fingerprint(package)
        artifact_inventory_digest = _artifact_inventory_sha256(predecessor)

        def fresh_revalidator() -> None:
            if context.identity_sha256 != context_digest:
                raise HostedPrerequisiteError("hosted workflow context changed")
            refreshed = validate_registration_package(package.root)
            if _package_fingerprint(refreshed) != package_fingerprint:
                raise HostedPrerequisiteError("C1 package changed after admission")
            registration.assert_current()
            current_manifest_time, current_registry_time, current_registry_file = (
                _read_registry_rekor_times(package, registration)
            )
            if (
                current_manifest_time != manifest_rekor_integrated_at
                or current_registry_time != registry_record_rekor_integrated_at
                or current_registry_file != registry_attestation_file_sha256
            ):
                raise HostedPrerequisiteError("C1 Rekor integration evidence changed")
            current_manifest = services.manifest_loader(package.root / "study-manifest.json")
            current_plans = services.phase_plan_loader(
                package.root / "study-manifest.json",
                c1_commit=package.c1_commit,
            )
            if (
                current_plans.get(phase) != plan
                or services.plan_templates_hasher(current_manifest) != templates_digest
                or _load_hosted_plan(plan_path, plan) != plan
                or _load_hosted_bootstrap(bootstrap_path, bootstrap) != bootstrap
                or type(plan.host_tools).from_dict(plan.host_tools.to_dict()) != plan.host_tools
            ):
                raise HostedPrerequisiteError("C1 plan or host closure changed")
            rerun_readiness = services.runner_readiness_verifier(
                plan=plan,
                api=runner_api,
                verified_at_utc=verified_at,
            )
            if rerun_readiness != readiness:
                raise HostedPrerequisiteError("live idle runner changed during admission")
            predecessor.predecessor.assert_current()
            current_snapshot = services.snapshot_loader(
                repository=REPOSITORY,
                suite_attempt_id=suite,
                api=ledger_api,
            )
            if (
                _assert_snapshot(
                    current_snapshot,
                    phase=phase,
                    suite=suite,
                    package=package,
                    registration=registration,
                    predecessor=predecessor,
                )
                != ledger_tree
                or _artifact_inventory_sha256(predecessor) != artifact_inventory_digest
            ):
                raise HostedPrerequisiteError("provider predecessor changed during admission")
            if (
                read_secure_control_file(
                    readback_path,
                    label="anonymous Zenodo readback receipt",
                )
                != readback.canonical_file_bytes()
            ):
                raise HostedPrerequisiteError("anonymous Zenodo readback receipt changed")

        return HostedProductionPrerequisites(
            context=context,
            phase=phase,
            suite_attempt_id=suite,
            workflow_context_sha256=context_digest,
            package=package,
            package_inventory_sha256=package_inventory_digest,
            registration=registration,
            manifest_rekor_integrated_at_utc=manifest_rekor_integrated_at,
            registry_record_rekor_integrated_at_utc=(registry_record_rekor_integrated_at),
            zenodo_admission=admission,
            zenodo_readback_receipt_path=readback_path,
            plan=plan,
            plan_materialization_path=plan_path,
            plan_templates_sha256=templates_digest,
            bootstrap=bootstrap,
            bootstrap_materialization_path=bootstrap_path,
            runner_readiness=readiness,
            predecessor=predecessor,
            predecessor_ledger_tree=ledger_tree,
            predecessor_artifact_inventory_sha256=artifact_inventory_digest,
            _fresh_revalidator=fresh_revalidator,
            _capability=_CAPABILITY,
        )
    except Exception:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        if owned_transport and isinstance(transport, _ZenodoHttpsTransport):
            transport.close()
