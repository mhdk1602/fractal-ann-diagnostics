"""Admission-gated sequencing for the existing online action runner."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .artifact_integrity import (
    ArtifactIntegrityError,
    ArtifactVerificationReceipt,
    VerifiedArtifact,
    read_secure_control_file,
    write_exclusive_receipt_bytes,
)
from .audit import AdmittedProvenanceRegistry
from .controller import GovernedRetriever
from .custody import OnlineCustodyAdmissionReceipt
from .online_runner import (
    OnlineRunArtifacts,
    OnlineTrialRuntime,
    run_online_action_matrix,
)
from .study import (
    FIXED_CORPORA,
    SealedRunReceipt,
    StudyManifestError,
    manifest_sha256,
    validate_study_manifest,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_ARTIFACT_BINDINGS_SCHEMA = "fractal-required-artifact-bindings-v2"
_REQUIRED_ARTIFACT_BINDING_FIELDS = frozenset(
    {
        "execution_artifact_id",
        "execution_revision_sha256",
        "provenance_component_artifact_ids",
        "retriever_artifact_ids",
        "runner_artifact_ids",
        "schema_version",
        "source_artifact_ids",
        "verification_receipt",
    }
)
_PROVENANCE_ARTIFACT_BINDING_FIELDS = frozenset({"artifact_id", "component"})

# These tables are the sole translation from registered manifest roles to the
# high-level dependency names recorded by the online audit chain.  Corpus-local
# runtime data stays in the source closure; executable implementations stay in
# the runner closure.  Callers cannot replace any ID selected here.
_PROVENANCE_COMPONENT_ROLES = (
    ("application", "source-code", False),
    ("controller", "frozen-controller", False),
    ("corpus", "corpus-normalizer", True),
    ("embedding", "primary-embedding", False),
    ("index", "strict-authorized-hnsw", False),
    ("policy", "opa-pdp", False),
)
_RUNNER_ARTIFACT_ROLES = (
    ("exact-authorized-oracle", False),
    ("frozen-controller", False),
    ("opa-pdp", False),
    ("opa-runtime-binary", False),
    ("source-code", False),
    ("strict-authorized-hnsw", False),
)
_SOURCE_ARTIFACT_ROLES = (
    ("authorized-index-store", True),
    ("embedding-store", True),
    ("online-staging-package", False),
    ("policy-workload", True),
    ("query-partition-audit", False),
    ("trial-runtime-package", True),
)
_EXECUTION_ARTIFACT_ROLE = "online-execution"
_WITHHELD_ARTIFACT_ROLES = frozenset(
    {
        "sealed-label-ciphertext",
        "sealed-labels",
        "timelock-encryption-receipt",
    }
)


class SealedOrchestratorError(RuntimeError):
    """Raised when the online call is not bound to its admission evidence."""


def _require_identifier(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise SealedOrchestratorError(f"{name} must be a canonical non-empty identifier")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SealedOrchestratorError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_id_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise SealedOrchestratorError(f"{name} must be a non-empty tuple")
    identifiers = tuple(
        _require_identifier(f"{name}[{position}]", item) for position, item in enumerate(value)
    )
    ordered = tuple(sorted(identifiers, key=lambda item: item.encode("utf-8")))
    if identifiers != ordered or len(identifiers) != len(set(identifiers)):
        raise SealedOrchestratorError(f"{name} must be unique and bytewise sorted")
    return identifiers


def _require_component_bindings(
    value: object,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple) or not value:
        raise SealedOrchestratorError("provenance_component_artifact_ids must be a non-empty tuple")
    pairs: list[tuple[str, str]] = []
    for position, item in enumerate(value):
        if not isinstance(item, tuple) or len(item) != 2:
            raise SealedOrchestratorError(
                f"provenance_component_artifact_ids[{position}] must be one pair"
            )
        component = _require_identifier(
            f"provenance_component_artifact_ids[{position}].component",
            item[0],
        )
        artifact_id = _require_identifier(
            f"provenance_component_artifact_ids[{position}].artifact_id",
            item[1],
        )
        pairs.append((component, artifact_id))
    bindings = tuple(pairs)
    ordered = tuple(
        sorted(
            bindings,
            key=lambda item: (item[0].encode("utf-8"), item[1].encode("utf-8")),
        )
    )
    components = [component for component, _ in bindings]
    artifact_ids = [artifact_id for _, artifact_id in bindings]
    if bindings != ordered:
        raise SealedOrchestratorError("provenance_component_artifact_ids must be bytewise sorted")
    if len(components) != len(set(components)) or len(artifact_ids) != len(set(artifact_ids)):
        raise SealedOrchestratorError(
            "provenance component names and artifact IDs must each be unique"
        )
    return bindings


@dataclass(frozen=True)
class RequiredArtifactIdBindings:
    """Explicit IDs and verification evidence for every online dependency."""

    verification_receipt: ArtifactVerificationReceipt
    execution_artifact_id: str
    execution_revision_sha256: str
    runner_artifact_ids: tuple[str, ...]
    source_artifact_ids: tuple[str, ...]
    retriever_artifact_ids: tuple[str, ...]
    provenance_component_artifact_ids: tuple[tuple[str, str], ...]
    schema_version: str = REQUIRED_ARTIFACT_BINDINGS_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.verification_receipt, ArtifactVerificationReceipt):
            raise SealedOrchestratorError(
                "verification_receipt must be an ArtifactVerificationReceipt"
            )
        if self.schema_version != REQUIRED_ARTIFACT_BINDINGS_SCHEMA:
            raise SealedOrchestratorError(
                f"schema_version must equal {REQUIRED_ARTIFACT_BINDINGS_SCHEMA!r}"
            )
        object.__setattr__(
            self,
            "execution_artifact_id",
            _require_identifier("execution_artifact_id", self.execution_artifact_id),
        )
        object.__setattr__(
            self,
            "execution_revision_sha256",
            _require_sha256(
                "execution_revision_sha256",
                self.execution_revision_sha256,
            ),
        )
        for name in (
            "runner_artifact_ids",
            "source_artifact_ids",
            "retriever_artifact_ids",
        ):
            object.__setattr__(self, name, _require_id_tuple(name, getattr(self, name)))
        object.__setattr__(
            self,
            "provenance_component_artifact_ids",
            _require_component_bindings(self.provenance_component_artifact_ids),
        )
        provenance_ids = tuple(
            sorted(
                (artifact_id for _, artifact_id in self.provenance_component_artifact_ids),
                key=lambda item: item.encode("utf-8"),
            )
        )
        if self.retriever_artifact_ids != provenance_ids:
            raise SealedOrchestratorError(
                "retriever_artifact_ids must equal the provenance component artifact IDs"
            )

    @property
    def required_artifact_ids(self) -> frozenset[str]:
        """Return the caller-declared dependency closure."""

        return frozenset(
            (
                self.execution_artifact_id,
                *self.runner_artifact_ids,
                *self.source_artifact_ids,
                *self.retriever_artifact_ids,
                *(artifact_id for _, artifact_id in self.provenance_component_artifact_ids),
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return the closed canonical control representation."""

        return {
            "execution_artifact_id": self.execution_artifact_id,
            "execution_revision_sha256": self.execution_revision_sha256,
            "provenance_component_artifact_ids": [
                {"artifact_id": artifact_id, "component": component}
                for component, artifact_id in self.provenance_component_artifact_ids
            ],
            "retriever_artifact_ids": list(self.retriever_artifact_ids),
            "runner_artifact_ids": list(self.runner_artifact_ids),
            "schema_version": self.schema_version,
            "source_artifact_ids": list(self.source_artifact_ids),
            "verification_receipt": self.verification_receipt.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> RequiredArtifactIdBindings:
        row = _closed_binding_mapping(
            value,
            fields=_REQUIRED_ARTIFACT_BINDING_FIELDS,
            label="required artifact bindings",
        )
        provenance = row["provenance_component_artifact_ids"]
        if isinstance(provenance, (str, bytes)) or not isinstance(provenance, Sequence):
            raise SealedOrchestratorError("provenance_component_artifact_ids must be an array")
        pairs: list[tuple[str, str]] = []
        for position, item in enumerate(provenance):
            binding = _closed_binding_mapping(
                item,
                fields=_PROVENANCE_ARTIFACT_BINDING_FIELDS,
                label=f"provenance_component_artifact_ids[{position}]",
            )
            pairs.append((binding["component"], binding["artifact_id"]))
        return cls(
            verification_receipt=ArtifactVerificationReceipt.from_dict(row["verification_receipt"]),
            execution_artifact_id=row["execution_artifact_id"],
            execution_revision_sha256=row["execution_revision_sha256"],
            runner_artifact_ids=_binding_array("runner_artifact_ids", row["runner_artifact_ids"]),
            source_artifact_ids=_binding_array("source_artifact_ids", row["source_artifact_ids"]),
            retriever_artifact_ids=_binding_array(
                "retriever_artifact_ids", row["retriever_artifact_ids"]
            ),
            provenance_component_artifact_ids=tuple(pairs),
            schema_version=row["schema_version"],
        )


def _validated_manifest_artifacts(
    frozen_manifest: Mapping[str, Any],
    verification_receipt: ArtifactVerificationReceipt,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(frozen_manifest, Mapping):
        raise SealedOrchestratorError("frozen_manifest must be a mapping")
    if not isinstance(verification_receipt, ArtifactVerificationReceipt):
        raise SealedOrchestratorError("verification_receipt must be an ArtifactVerificationReceipt")
    try:
        validate_study_manifest(frozen_manifest, require_frozen=True)
    except StudyManifestError as exc:
        raise SealedOrchestratorError(f"invalid frozen study manifest: {exc}") from exc

    digest = manifest_sha256(frozen_manifest)
    if verification_receipt.manifest_sha256 != digest:
        raise SealedOrchestratorError(
            "artifact verification receipt belongs to another frozen manifest"
        )
    artifact_values = frozen_manifest.get("artifacts")
    if not isinstance(artifact_values, Sequence) or isinstance(artifact_values, (str, bytes)):
        raise SealedOrchestratorError("validated manifest artifacts are malformed")
    artifacts = tuple(artifact_values)
    if not all(isinstance(row, Mapping) for row in artifacts):
        raise SealedOrchestratorError("validated manifest artifact rows are malformed")

    manifest_by_id = {str(row["id"]): row for row in artifacts}
    receipt_by_id = _artifact_rows(verification_receipt)
    manifest_ids = set(manifest_by_id)
    receipt_ids = set(receipt_by_id)
    if manifest_ids != receipt_ids:
        missing = sorted(manifest_ids - receipt_ids)
        extra = sorted(receipt_ids - manifest_ids)
        raise SealedOrchestratorError(
            "artifact verification must cover the exact frozen manifest; "
            f"missing={missing}, extra={extra}"
        )
    for artifact_id, manifest_row in manifest_by_id.items():
        receipt_row = receipt_by_id[artifact_id]
        expected = manifest_row["sha256"]
        if (
            not receipt_row.exact
            or receipt_row.expected_sha256 != expected
            or receipt_row.verified_sha256 != expected
        ):
            raise SealedOrchestratorError(
                f"artifact verification is not exact for manifest artifact {artifact_id!r}"
            )
    return artifacts


def _manifest_role_artifact(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    role: str,
    corpus_id: str,
    corpus_bound: bool,
) -> Mapping[str, Any]:
    role_rows = tuple(row for row in artifacts if row.get("role") == role)
    if corpus_bound:
        matches = tuple(row for row in role_rows if row.get("corpus_id") == corpus_id)
        if len(matches) != 1:
            raise SealedOrchestratorError(
                f"manifest role {role!r} must identify exactly one artifact for {corpus_id!r}"
            )
        if any(row.get("corpus_id") not in FIXED_CORPORA for row in role_rows):
            raise SealedOrchestratorError(
                f"manifest role {role!r} contains an unregistered corpus binding"
            )
    else:
        matches = role_rows
        if len(matches) != 1 or "corpus_id" in matches[0]:
            raise SealedOrchestratorError(
                f"manifest role {role!r} must identify exactly one suite artifact"
            )
    _require_identifier(f"manifest role {role!r} artifact ID", matches[0]["id"])
    return matches[0]


def _manifest_role_artifact_id(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    role: str,
    corpus_id: str,
    corpus_bound: bool,
) -> str:
    row = _manifest_role_artifact(
        artifacts,
        role=role,
        corpus_id=corpus_id,
        corpus_bound=corpus_bound,
    )
    return str(row["id"])


def derive_required_artifact_id_bindings(
    frozen_manifest: Mapping[str, Any],
    verification_receipt: ArtifactVerificationReceipt,
    *,
    corpus_id: str,
) -> RequiredArtifactIdBindings:
    """Derive one corpus's closed online dependency IDs from C1 evidence.

    The frozen manifest role table is authoritative.  This API accepts no ID,
    component, runner, or source override from its caller.
    """

    corpus = _require_identifier("corpus_id", corpus_id)
    if corpus not in FIXED_CORPORA:
        raise SealedOrchestratorError("corpus_id is not in the fixed confirmatory suite")
    artifacts = _validated_manifest_artifacts(frozen_manifest, verification_receipt)

    execution_row = _manifest_role_artifact(
        artifacts,
        role=_EXECUTION_ARTIFACT_ROLE,
        corpus_id=corpus,
        corpus_bound=True,
    )
    execution_artifact_id = str(execution_row["id"])
    revision = execution_row.get("revision")
    if (
        not isinstance(revision, str)
        or not revision.startswith("sha256:")
        or _SHA256.fullmatch(revision[7:]) is None
    ):
        raise SealedOrchestratorError(
            "online-execution revision must encode its logical plan SHA-256"
        )
    execution_revision_sha256 = revision[7:]
    runner_artifact_ids = tuple(
        sorted(
            (
                _manifest_role_artifact_id(
                    artifacts,
                    role=role,
                    corpus_id=corpus,
                    corpus_bound=corpus_bound,
                )
                for role, corpus_bound in _RUNNER_ARTIFACT_ROLES
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    source_artifact_ids = tuple(
        sorted(
            (
                _manifest_role_artifact_id(
                    artifacts,
                    role=role,
                    corpus_id=corpus,
                    corpus_bound=corpus_bound,
                )
                for role, corpus_bound in _SOURCE_ARTIFACT_ROLES
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    component_bindings = tuple(
        sorted(
            (
                (
                    component,
                    _manifest_role_artifact_id(
                        artifacts,
                        role=role,
                        corpus_id=corpus,
                        corpus_bound=corpus_bound,
                    ),
                )
                for component, role, corpus_bound in _PROVENANCE_COMPONENT_ROLES
            ),
            key=lambda item: (item[0].encode("utf-8"), item[1].encode("utf-8")),
        )
    )
    retriever_artifact_ids = tuple(
        sorted(
            (artifact_id for _, artifact_id in component_bindings),
            key=lambda value: value.encode("utf-8"),
        )
    )
    selected_ids = {
        execution_artifact_id,
        *runner_artifact_ids,
        *source_artifact_ids,
        *retriever_artifact_ids,
    }
    selected_roles = {str(row["role"]) for row in artifacts if str(row["id"]) in selected_ids}
    forbidden = selected_roles & _WITHHELD_ARTIFACT_ROLES
    if forbidden:
        raise SealedOrchestratorError(
            f"online dependency closure includes withheld artifact roles {sorted(forbidden)}"
        )
    return RequiredArtifactIdBindings(
        verification_receipt=verification_receipt,
        execution_artifact_id=execution_artifact_id,
        execution_revision_sha256=execution_revision_sha256,
        runner_artifact_ids=runner_artifact_ids,
        source_artifact_ids=source_artifact_ids,
        retriever_artifact_ids=retriever_artifact_ids,
        provenance_component_artifact_ids=component_bindings,
    )


def _closed_binding_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SealedOrchestratorError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        missing = sorted(fields - observed)
        unknown = sorted(observed - fields)
        raise SealedOrchestratorError(
            f"{label} fields differ; missing={missing}, unknown={unknown}"
        )
    return value


def _binding_array(name: str, value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SealedOrchestratorError(f"{name} must be an array")
    return tuple(value)


def _decode_required_artifact_bindings(encoded: bytes) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SealedOrchestratorError(
                    f"required artifact bindings contain duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise SealedOrchestratorError(
            f"required artifact bindings contain non-finite number {value!r}"
        )

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedOrchestratorError(
            "required artifact bindings must be one UTF-8 JSON object"
        ) from exc
    if not isinstance(value, Mapping):
        raise SealedOrchestratorError("required artifact bindings must contain one object")
    return value


def load_required_artifact_id_bindings(
    path: str | Path,
) -> RequiredArtifactIdBindings:
    """Load exact bindings without following links or accepting hard links."""

    try:
        encoded = read_secure_control_file(path, label="required artifact bindings")
    except ArtifactIntegrityError as exc:
        raise SealedOrchestratorError(
            f"cannot read required artifact bindings safely: {exc}"
        ) from exc
    bindings = RequiredArtifactIdBindings.from_dict(_decode_required_artifact_bindings(encoded))
    if encoded != bindings.canonical_file_bytes():
        raise SealedOrchestratorError("required artifact bindings bytes are not canonical")
    return bindings


def write_required_artifact_id_bindings(
    bindings: RequiredArtifactIdBindings,
    target: str | Path,
) -> None:
    """Publish exact bindings once through a no-follow filesystem operation."""

    if not isinstance(bindings, RequiredArtifactIdBindings):
        raise SealedOrchestratorError("bindings must be RequiredArtifactIdBindings")
    try:
        write_exclusive_receipt_bytes(bindings.canonical_file_bytes(), target)
    except ArtifactIntegrityError as exc:
        raise SealedOrchestratorError(
            f"cannot publish required artifact bindings safely: {exc}"
        ) from exc


def _artifact_rows(
    receipt: ArtifactVerificationReceipt,
) -> dict[str, VerifiedArtifact]:
    return {artifact.artifact_id: artifact for artifact in receipt.artifacts}


def _execution_sha256(execution: object) -> str:
    try:
        digest = getattr(execution, "artifact_sha256")
    except Exception as exc:
        raise SealedOrchestratorError("execution must expose a stable artifact_sha256") from exc
    return _require_sha256("execution.artifact_sha256", digest)


def _verify_admission(
    *,
    admission_receipt: OnlineCustodyAdmissionReceipt,
    required_artifacts: RequiredArtifactIdBindings,
    execution: object,
    run_receipt: SealedRunReceipt,
    retriever: GovernedRetriever,
    provenance_registry: AdmittedProvenanceRegistry,
) -> None:
    if not isinstance(admission_receipt, OnlineCustodyAdmissionReceipt):
        raise SealedOrchestratorError("admission_receipt must be an OnlineCustodyAdmissionReceipt")
    if not isinstance(required_artifacts, RequiredArtifactIdBindings):
        raise SealedOrchestratorError("required_artifacts must be RequiredArtifactIdBindings")
    if not isinstance(run_receipt, SealedRunReceipt):
        raise SealedOrchestratorError("run_receipt must be a SealedRunReceipt")
    if not isinstance(retriever, GovernedRetriever):
        raise SealedOrchestratorError("retriever must be a GovernedRetriever")
    if not isinstance(provenance_registry, AdmittedProvenanceRegistry):
        raise SealedOrchestratorError(
            "provenance_registry lacks the admitted digest-only interface"
        )

    verification = required_artifacts.verification_receipt
    manifest_digests = {
        admission_receipt.manifest_sha256,
        run_receipt.manifest_sha256,
        verification.manifest_sha256,
    }
    if len(manifest_digests) != 1:
        raise SealedOrchestratorError(
            "admission, run, and artifact verification bind different manifests"
        )

    run_receipt_sha256 = run_receipt.binding_sha256
    if admission_receipt.run_receipt_sha256 != run_receipt_sha256:
        raise SealedOrchestratorError("admission binds a different sealed run receipt")
    if admission_receipt.runner_identity != run_receipt.runner_identity:
        raise SealedOrchestratorError("admission and sealed run use different runner identities")

    verification_sha256 = verification.receipt_sha256
    exposed_verification_digests = {
        admission_receipt.artifact_verification_receipt_sha256,
        run_receipt.verification_receipt_sha256,
        provenance_registry.verification_receipt_sha256,
        verification_sha256,
    }
    if len(exposed_verification_digests) != 1:
        raise SealedOrchestratorError(
            "artifact verification binding differs across admitted inputs"
        )

    rows = _artifact_rows(verification)
    admitted_ids = frozenset(admission_receipt.verified_artifact_ids)
    missing_admitted_rows = admitted_ids - rows.keys()
    if missing_admitted_rows:
        raise SealedOrchestratorError(
            "admission names IDs absent from artifact verification: "
            f"{sorted(missing_admitted_rows)}"
        )
    missing_required = required_artifacts.required_artifact_ids - admitted_ids
    if missing_required:
        raise SealedOrchestratorError(
            f"required artifact IDs were not admitted: {sorted(missing_required)}"
        )
    nonexact_required = sorted(
        artifact_id
        for artifact_id in required_artifacts.required_artifact_ids
        if not rows[artifact_id].exact
    )
    if nonexact_required:
        raise SealedOrchestratorError(
            f"required artifacts were not verified exactly: {nonexact_required}"
        )

    if _execution_sha256(execution) != required_artifacts.execution_revision_sha256:
        raise SealedOrchestratorError(
            "execution logical digest differs from its manifest-derived revision"
        )

    component_ids = dict(required_artifacts.provenance_component_artifact_ids)
    component_revisions = dict(provenance_registry.component_revisions)
    if component_ids.keys() != component_revisions.keys():
        raise SealedOrchestratorError(
            "provenance component bindings do not cover the exact registry components"
        )
    for component, artifact_id in component_ids.items():
        if rows[artifact_id].verified_sha256 != component_revisions[component]:
            raise SealedOrchestratorError(
                f"provenance component {component!r} binds a different artifact digest"
            )


def run_admitted_online_matrix(
    *,
    admission_receipt: OnlineCustodyAdmissionReceipt,
    required_artifacts: RequiredArtifactIdBindings,
    execution: object,
    run_receipt: SealedRunReceipt,
    retriever: GovernedRetriever,
    provenance_registry: AdmittedProvenanceRegistry,
    trial_runtimes: Mapping[str, OnlineTrialRuntime],
    permutation_seed: int,
    expected_policy_version: str,
    query_partition_audit_sha256: str,
    pseudonym_key: bytes,
    pseudonym_key_id: str,
    k: int = 10,
    policy_action: str = "retrieve",
    partition_label: Literal["primary", "reserve"] = "primary",
    occurred_at_factory: Callable[[str, str, int], str | None] | None = None,
) -> OnlineRunArtifacts:
    """Verify the admission chain, then invoke the existing runner once."""

    _verify_admission(
        admission_receipt=admission_receipt,
        required_artifacts=required_artifacts,
        execution=execution,
        run_receipt=run_receipt,
        retriever=retriever,
        provenance_registry=provenance_registry,
    )
    return run_online_action_matrix(
        execution=execution,
        run_receipt=run_receipt,
        retriever=retriever,
        provenance_registry=provenance_registry,
        trial_runtimes=trial_runtimes,
        permutation_seed=permutation_seed,
        expected_policy_version=expected_policy_version,
        query_partition_audit_sha256=query_partition_audit_sha256,
        pseudonym_key=pseudonym_key,
        pseudonym_key_id=pseudonym_key_id,
        k=k,
        policy_action=policy_action,
        partition_label=partition_label,
        occurred_at_factory=occurred_at_factory,
    )
