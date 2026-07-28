"""Closed records shared by the provider phase workflow jobs.

Serialized workflow receipts are evidence, never authority.  Each CLI process
must re-open the exact files and provider APIs named by a receipt before it can
mint one of the in-memory capabilities used by the suite state machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from .execution_claim import (
    C1_REGISTRATION_PACKAGE_FILE_COUNT,
    ExecutionClaimContract,
    GitHubReadApi,
    PhaseClaimContract,
    PhaseCorpusBinding,
    ProviderExecutionIdentity,
    ProviderPhasePlan,
    derive_phase_runner_label,
)
from .study import (
    FIXED_CORPORA,
    PROVIDER_PHASE_JOB_NAMES,
    PROVIDER_PHASE_WORKFLOWS,
    VerifiedC1ProtocolRegistration,
    load_study_manifest,
    manifest_sha256,
    validate_study_manifest,
)
from .suite_attempt import (
    LabelCorpusClosure,
    OnlineSuiteClosure,
    RunClaimBindings,
    VerifiedProviderPredecessor,
)
from .suite_attempt import (
    suite_attempt_id as derive_suite_attempt_id,
)

ProviderPhase = Literal["online", "label-release", "analysis"]
ProviderWorkflowJob = Literal["claim", "execute", "complete", "fail"]
PreparationMode = Literal["completion", "failure"]

REPOSITORY = "mhdk1602/fractal-ann-diagnostics"
REPOSITORY_ID = 1_239_189_910
REPOSITORY_NODE_ID = "R_kgDOSdyJlg"
OWNER_LOGIN = "mhdk1602"
OWNER_ID = 9_646_005
OWNER_NODE_ID = "MDQ6VXNlcjk2NDYwMDU="
C0_REF = "refs/tags/confirmatory-apparatus-c0"
C0_TAG = "confirmatory-apparatus-c0"

WORKFLOW_CONTEXT_SCHEMA = "fractal-provider-workflow-context-v1"
PREREQUISITE_RECEIPT_SCHEMA = "fractal-provider-prerequisite-receipt-v2"
CLAIM_RECEIPT_SCHEMA = "fractal-provider-claim-receipt-v1"
PREPARATION_RECEIPT_SCHEMA = "fractal-provider-transition-preparation-v1"

_PHASES = frozenset(PROVIDER_PHASE_WORKFLOWS)
_JOBS = frozenset({"claim", "execute", "complete", "fail"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RUNNER_LABEL = re.compile(r"^fractal-ann-confirmatory-[a-z0-9][a-z0-9-]{15,95}$")
_CAPABILITY = object()

_TRANSITION_PREDICATE_SCHEMA = "fractal-provider-transition-predicate-v1"
_FAILURE_INCIDENT_SCHEMA = "fractal-provider-failure-incident-v1"
_MAX_EVIDENCE_FILES = 16_384
_MAX_EVIDENCE_FILE_BYTES = 64 * 1024**3
_MAX_EVIDENCE_TOTAL_BYTES = 256 * 1024**3
_TRANSITION_PREDICATE_TYPES: Mapping[tuple[PreparationMode, ProviderPhase], str] = {
    (
        "completion",
        "online",
    ): "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/online-output-aggregate/v1",
    (
        "completion",
        "label-release",
    ): "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/label-release/v1",
    (
        "completion",
        "analysis",
    ): "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/confirmatory-analysis/v1",
    (
        "failure",
        "online",
    ): "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/provider-failure/v1",
    (
        "failure",
        "label-release",
    ): "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/provider-failure/v1",
    (
        "failure",
        "analysis",
    ): "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/provider-failure/v1",
}

_PREDECESSOR: Mapping[ProviderPhase, tuple[str, int]] = {
    "online": ("OPENED", 0),
    "label-release": ("ONLINE_COMPLETE", 2),
    "analysis": ("LABELS_RELEASED", 4),
}
_CLAIMED: Mapping[ProviderPhase, tuple[str, int]] = {
    "online": ("RUN_CLAIMED", 1),
    "label-release": ("LABEL_RELEASE_CLAIMED", 3),
    "analysis": ("ANALYSIS_CLAIMED", 5),
}
_COMPLETED: Mapping[ProviderPhase, tuple[str, int]] = {
    "online": ("ONLINE_COMPLETE", 2),
    "label-release": ("LABELS_RELEASED", 4),
    "analysis": ("ANALYSIS_COMPLETE", 6),
}
_PHASE_DRIVER_IDS: Mapping[ProviderPhase, str] = {
    "online": "sealed-online-corpus-v1",
    "label-release": "timelock-label-release-v1",
    "analysis": "confirmatory-analysis-v1",
}


class ProviderWorkflowOrchestrationError(ValueError):
    """A workflow context or persisted provider receipt is malformed."""


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
        raise ProviderWorkflowOrchestrationError(
            "provider workflow evidence is not canonical JSON"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderWorkflowOrchestrationError(f"{name} must be one canonical non-empty string")
    return value


def _digest(name: str, value: object) -> str:
    text = _text(name, value)
    if _SHA256.fullmatch(text) is None:
        raise ProviderWorkflowOrchestrationError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _commit(name: str, value: object) -> str:
    text = _text(name, value)
    if _GIT_COMMIT.fullmatch(text) is None:
        raise ProviderWorkflowOrchestrationError(f"{name} must be a lowercase Git SHA-1 commit")
    return text


def _positive(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ProviderWorkflowOrchestrationError(f"{name} must be a positive integer")
    return value


def _nonnegative(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ProviderWorkflowOrchestrationError(f"{name} must be a non-negative integer")
    return value


def _phase(value: object) -> ProviderPhase:
    if value not in _PHASES:
        raise ProviderWorkflowOrchestrationError("provider phase is not registered")
    return value  # type: ignore[return-value]


def _absolute_path(name: str, value: object) -> Path:
    text = _text(name, value)
    path = Path(text)
    if (
        not path.is_absolute()
        or path.anchor != "/"
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or str(path) != text
    ):
        raise ProviderWorkflowOrchestrationError(
            f"{name} must be one canonical absolute POSIX path"
        )
    return path


def _relative_path(name: str, value: object) -> str:
    text = _text(name, value)
    if text != unicodedata.normalize("NFC", text) or "\\" in text or "\x00" in text:
        raise ProviderWorkflowOrchestrationError(f"{name} is not a canonical relative path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise ProviderWorkflowOrchestrationError(f"{name} is not a safe relative path")
    return text


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProviderWorkflowOrchestrationError(f"{label} must be a JSON object with string keys")
    observed = set(value)
    if observed != fields:
        raise ProviderWorkflowOrchestrationError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _strict_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ProviderWorkflowOrchestrationError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ProviderWorkflowOrchestrationError(f"{label} contains non-finite number {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderWorkflowOrchestrationError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ProviderWorkflowOrchestrationError(f"{label} must be a JSON object")
    return value


def _api_object(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProviderWorkflowOrchestrationError(f"{label} must be one GitHub API object")
    return value


def _api_array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProviderWorkflowOrchestrationError(f"{label} must be one GitHub API array")
    return value


def _secure_receipt_bytes(path: Path, *, label: str, max_bytes: int = 8 * 1024**2) -> bytes:
    if not path.is_absolute():
        raise ProviderWorkflowOrchestrationError(f"{label} path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProviderWorkflowOrchestrationError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise ProviderWorkflowOrchestrationError(
                f"{label} must be one bounded singly linked regular file"
            )
        chunks: list[bytes] = []
        observed = 0
        while observed <= max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ProviderWorkflowOrchestrationError(f"cannot read {label}") from exc
    finally:
        os.close(descriptor)
    if observed > max_bytes or (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ProviderWorkflowOrchestrationError(f"{label} changed while read")
    return b"".join(chunks)


def _write_exclusive(path: Path, encoded: bytes, *, label: str) -> None:
    if not path.is_absolute():
        raise ProviderWorkflowOrchestrationError(f"{label} path must be absolute")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        parent = path.parent.lstat()
    except OSError as exc:
        raise ProviderWorkflowOrchestrationError(f"cannot inspect {label} parent") from exc
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or stat.S_IMODE(parent.st_mode) & 0o022
        or (hasattr(os, "geteuid") and parent.st_uid != os.geteuid())
    ):
        raise ProviderWorkflowOrchestrationError(f"{label} parent is not controlled")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise ProviderWorkflowOrchestrationError(f"cannot create {label} once") from exc


@dataclass(frozen=True)
class ProviderWorkflowContext:
    """Non-serializable admission of the exact fixed GitHub job context."""

    phase: ProviderPhase
    job: ProviderWorkflowJob
    repository: str
    repository_id: int
    repository_owner: str
    repository_owner_id: int
    actor: str
    actor_id: int
    triggering_actor: str
    workflow_path: str
    workflow_ref: str
    workflow_sha: str
    github_ref: str
    github_ref_name: str
    github_ref_type: str
    github_ref_protected: bool
    github_sha: str
    run_id: int
    run_attempt: int
    event_name: str
    runner_environment: str
    runner_os: str
    runner_arch: str
    schema_version: str = WORKFLOW_CONTEXT_SCHEMA
    _capability: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _CAPABILITY:
            raise ProviderWorkflowOrchestrationError(
                "workflow context can only be minted from the live environment"
            )
        phase = _phase(self.phase)
        if self.job not in _JOBS:
            raise ProviderWorkflowOrchestrationError("GitHub job is not a provider phase job")
        workflow_path = PROVIDER_PHASE_WORKFLOWS[phase]
        expected_ref = f"{REPOSITORY}/{workflow_path}@{C0_REF}"
        if (
            self.repository != REPOSITORY
            or self.repository_id != REPOSITORY_ID
            or self.repository_owner != OWNER_LOGIN
            or self.repository_owner_id != OWNER_ID
            or self.actor != OWNER_LOGIN
            or self.actor_id != OWNER_ID
            or self.triggering_actor != OWNER_LOGIN
            or self.workflow_path != workflow_path
            or self.workflow_ref != expected_ref
            or self.github_ref != C0_REF
            or self.github_ref_name != C0_TAG
            or self.github_ref_type != "tag"
            or self.github_ref_protected is not True
            or self.event_name != "workflow_dispatch"
            or self.run_attempt != 1
        ):
            raise ProviderWorkflowOrchestrationError(
                "GitHub workflow context differs from the fixed C0 production identity"
            )
        _positive("run_id", self.run_id)
        _commit("workflow_sha", self.workflow_sha)
        _commit("github_sha", self.github_sha)
        if self.workflow_sha != self.github_sha:
            raise ProviderWorkflowOrchestrationError(
                "workflow source and dispatch source commits differ"
            )
        expected_runner = (
            ("self-hosted", "macOS", "ARM64")
            if self.job == "execute"
            else ("github-hosted", "Linux", "X64")
        )
        if (self.runner_environment, self.runner_os, self.runner_arch) != expected_runner:
            raise ProviderWorkflowOrchestrationError(
                "provider job runs on another runner class or architecture"
            )
        if self.schema_version != WORKFLOW_CONTEXT_SCHEMA:
            raise ProviderWorkflowOrchestrationError("workflow context schema differs")

    @classmethod
    def from_environment(
        cls,
        phase: ProviderPhase,
        environ: Mapping[str, str] | None = None,
    ) -> ProviderWorkflowContext:
        source = os.environ if environ is None else environ
        if not isinstance(source, Mapping):
            raise ProviderWorkflowOrchestrationError("environment must be a string mapping")

        def required(name: str) -> str:
            value = source.get(name)
            if type(value) is not str or not value:
                raise ProviderWorkflowOrchestrationError(f"GitHub environment omits {name}")
            return value

        if required("GITHUB_ACTIONS") != "true" or required("CI") != "true":
            raise ProviderWorkflowOrchestrationError(
                "provider workflow context must run inside GitHub Actions"
            )
        if (
            required("GITHUB_SERVER_URL") != "https://github.com"
            or required("GITHUB_API_URL") != "https://api.github.com"
            or required("GITHUB_GRAPHQL_URL") != "https://api.github.com/graphql"
        ):
            raise ProviderWorkflowOrchestrationError("GitHub service origins differ")
        try:
            context = cls(
                phase=phase,
                job=required("GITHUB_JOB"),  # type: ignore[arg-type]
                repository=required("GITHUB_REPOSITORY"),
                repository_id=int(required("GITHUB_REPOSITORY_ID")),
                repository_owner=required("GITHUB_REPOSITORY_OWNER"),
                repository_owner_id=int(required("GITHUB_REPOSITORY_OWNER_ID")),
                actor=required("GITHUB_ACTOR"),
                actor_id=int(required("GITHUB_ACTOR_ID")),
                triggering_actor=required("GITHUB_TRIGGERING_ACTOR"),
                workflow_path=PROVIDER_PHASE_WORKFLOWS[_phase(phase)],
                workflow_ref=required("GITHUB_WORKFLOW_REF"),
                workflow_sha=required("GITHUB_WORKFLOW_SHA"),
                github_ref=required("GITHUB_REF"),
                github_ref_name=required("GITHUB_REF_NAME"),
                github_ref_type=required("GITHUB_REF_TYPE"),
                github_ref_protected=required("GITHUB_REF_PROTECTED") == "true",
                github_sha=required("GITHUB_SHA"),
                run_id=int(required("GITHUB_RUN_ID")),
                run_attempt=int(required("GITHUB_RUN_ATTEMPT")),
                event_name=required("GITHUB_EVENT_NAME"),
                runner_environment=required("RUNNER_ENVIRONMENT"),
                runner_os=required("RUNNER_OS"),
                runner_arch=required("RUNNER_ARCH"),
                _capability=_CAPABILITY,
            )
        except ProviderWorkflowOrchestrationError:
            raise
        except ValueError as exc:
            raise ProviderWorkflowOrchestrationError(
                "numeric GitHub workflow identity is malformed"
            ) from exc
        return context

    def identity_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__ if name != "_capability"
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.identity_dict()))


def verify_provider_execution_identity(
    *,
    context: ProviderWorkflowContext,
    plan: ProviderPhasePlan,
    api: GitHubReadApi,
) -> ProviderExecutionIdentity:
    """Bind the current hosted claim job to the C1-selected self-hosted runner."""

    if not isinstance(context, ProviderWorkflowContext) or context.job != "claim":
        raise ProviderWorkflowOrchestrationError(
            "provider identity requires the admitted hosted claim context"
        )
    if not isinstance(plan, ProviderPhasePlan):
        raise ProviderWorkflowOrchestrationError(
            "provider identity requires one typed C1 provider plan"
        )
    runner_label = derive_phase_runner_label(plan.claim_nonce, plan.phase)
    expected_plan = {
        "phase": context.phase,
        "repository": context.repository,
        "workflow_path": context.workflow_path,
        "workflow_ref": context.workflow_ref,
        "workflow_sha": context.workflow_sha,
        "run_head_branch": context.github_ref_name,
        "runner_label": runner_label,
    }
    observed_plan = {
        "phase": plan.phase,
        "repository": plan.repository,
        "workflow_path": plan.workflow_path,
        "workflow_ref": plan.workflow_ref,
        "workflow_sha": plan.workflow_sha,
        "run_head_branch": plan.run_head_branch,
        "runner_label": derive_phase_runner_label(plan.claim_nonce, plan.phase),
    }
    if observed_plan != expected_plan:
        raise ProviderWorkflowOrchestrationError(
            "live claim context and C1 provider plan identify another execution"
        )

    try:
        run = _api_object(
            api.get(
                f"repos/{REPOSITORY}/actions/runs/{context.run_id}/attempts/{context.run_attempt}"
            ),
            label="GitHub provider workflow run",
        )
    except ProviderWorkflowOrchestrationError:
        raise
    except Exception as exc:
        raise ProviderWorkflowOrchestrationError(
            "cannot read the live provider workflow run"
        ) from exc
    repository = _api_object(run.get("repository"), label="GitHub provider workflow repository")
    head_repository = _api_object(
        run.get("head_repository"), label="GitHub provider workflow head repository"
    )
    repository_owner = _api_object(
        repository.get("owner"), label="GitHub provider repository owner"
    )
    head_owner = _api_object(
        head_repository.get("owner"), label="GitHub provider head-repository owner"
    )
    actor = _api_object(run.get("actor"), label="GitHub provider workflow actor")
    triggering_actor = _api_object(
        run.get("triggering_actor"), label="GitHub provider triggering actor"
    )
    run_exact: Mapping[str, object] = {
        "id": context.run_id,
        "run_attempt": 1,
        "event": "workflow_dispatch",
        "status": "in_progress",
        "conclusion": None,
        "head_sha": context.github_sha,
        "head_branch": context.github_ref_name,
        "path": context.workflow_path,
    }
    for name, expected in run_exact.items():
        if run.get(name) != expected or type(run.get(name)) is not type(expected):
            raise ProviderWorkflowOrchestrationError(
                f"live GitHub provider run {name} differs from C0"
            )
    repository_exact: Mapping[str, object] = {
        "full_name": REPOSITORY,
        "id": REPOSITORY_ID,
        "node_id": REPOSITORY_NODE_ID,
    }
    owner_exact: Mapping[str, object] = {
        "login": OWNER_LOGIN,
        "id": OWNER_ID,
        "node_id": OWNER_NODE_ID,
    }
    for row, label in (
        (repository, "repository"),
        (head_repository, "head repository"),
    ):
        for name, expected in repository_exact.items():
            if row.get(name) != expected or type(row.get(name)) is not type(expected):
                raise ProviderWorkflowOrchestrationError(
                    f"live GitHub provider {label} {name} differs"
                )
    for row, label in (
        (repository_owner, "repository owner"),
        (head_owner, "head-repository owner"),
        (actor, "actor"),
        (triggering_actor, "triggering actor"),
    ):
        for name, expected in owner_exact.items():
            if row.get(name) != expected or type(row.get(name)) is not type(expected):
                raise ProviderWorkflowOrchestrationError(
                    f"live GitHub provider {label} {name} differs"
                )

    try:
        jobs_response = _api_object(
            api.get(
                f"repos/{REPOSITORY}/actions/runs/{context.run_id}/attempts/"
                f"{context.run_attempt}/jobs?per_page=100"
            ),
            label="GitHub provider workflow jobs",
        )
    except ProviderWorkflowOrchestrationError:
        raise
    except Exception as exc:
        raise ProviderWorkflowOrchestrationError(
            "cannot read the live provider workflow jobs"
        ) from exc
    jobs = _api_array(jobs_response.get("jobs"), label="GitHub provider workflow jobs")
    if (
        type(jobs_response.get("total_count")) is not int
        or jobs_response["total_count"] != len(jobs)
        or len(jobs) > 100
        or any(not isinstance(row, Mapping) for row in jobs)
    ):
        raise ProviderWorkflowOrchestrationError(
            "live GitHub provider job inventory is incomplete or malformed"
        )
    matches = [
        _api_object(row, label="GitHub provider claim job")
        for row in jobs
        if isinstance(row, Mapping) and row.get("name") == plan.claim_job_name
    ]
    if len(matches) != 1:
        raise ProviderWorkflowOrchestrationError(
            "live GitHub provider claim job is not a singleton"
        )
    claim_job = matches[0]
    claim_labels = _api_array(claim_job.get("labels"), label="GitHub hosted claim-job labels")
    if not all(type(label) is str and label for label in claim_labels):
        raise ProviderWorkflowOrchestrationError("GitHub hosted claim-job labels are malformed")
    if "ubuntu-24.04" not in claim_labels or "self-hosted" in claim_labels:
        raise ProviderWorkflowOrchestrationError(
            "GitHub claim job did not run on the fixed hosted image"
        )
    claim_exact: Mapping[str, object] = {
        "name": plan.claim_job_name,
        "run_id": context.run_id,
        "run_attempt": 1,
        "status": "in_progress",
        "conclusion": None,
    }
    for name, expected in claim_exact.items():
        if claim_job.get(name) != expected or type(claim_job.get(name)) is not type(expected):
            raise ProviderWorkflowOrchestrationError(
                f"live GitHub claim job {name} differs from C0"
            )
    claim_job_id = _positive("claim_job_id", claim_job.get("id"))
    identity = ProviderExecutionIdentity(
        repository=plan.repository,
        workflow_path=plan.workflow_path,
        workflow_ref=plan.workflow_ref,
        workflow_sha=plan.workflow_sha,
        run_head_branch=plan.run_head_branch,
        run_id=context.run_id,
        run_attempt=context.run_attempt,
        claim_job_id=claim_job_id,
        claim_job_name=plan.claim_job_name,
        execute_job_name=plan.execute_job_name,
        runner_id=plan.runner_id,
        runner_name=plan.runner_name,
        runner_group_id=plan.runner_group_id,
        runner_label=runner_label,
        runner_version=plan.runner_version,
        runner_archive_sha256=plan.runner_archive_sha256,
        provider_operating_system=plan.provider_operating_system,
        provider_architecture=plan.provider_architecture,
        host_tool_contract_sha256=plan.host_tools.contract_sha256,
        runtime_probe_receipt_sha256=plan.runtime_probe_receipt_sha256,
        self_hosted=True,
    )
    # Defeat a run/ref race immediately before the identity escapes this function.
    try:
        final_run = _api_object(
            api.get(
                f"repos/{REPOSITORY}/actions/runs/{context.run_id}/attempts/{context.run_attempt}"
            ),
            label="final GitHub provider workflow run",
        )
    except ProviderWorkflowOrchestrationError:
        raise
    except Exception as exc:
        raise ProviderWorkflowOrchestrationError(
            "cannot reread the live provider workflow run"
        ) from exc
    if any(final_run.get(name) != expected for name, expected in run_exact.items()):
        raise ProviderWorkflowOrchestrationError(
            "live GitHub provider run changed during identity verification"
        )
    return identity


PostOnlinePhase = Literal["label-release", "analysis"]


@dataclass(frozen=True)
class DerivedPhaseClaim:
    """C1- and predecessor-derived post-online contract plus its fixed local paths."""

    contract: PhaseClaimContract
    input_paths: tuple[tuple[str, str], ...]
    supporting_input_paths: tuple[tuple[str, str], ...]
    output_paths: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contract, PhaseClaimContract):
            raise ProviderWorkflowOrchestrationError(
                "derived phase claim requires one typed contract"
            )
        ordered = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))
        for name in ("input_paths", "supporting_input_paths", "output_paths"):
            rows = tuple(getattr(self, name))
            if tuple(row[0] for row in rows) != ordered or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or type(row[0]) is not str
                or type(row[1]) is not str
                for row in rows
            ):
                raise ProviderWorkflowOrchestrationError(
                    f"derived phase {name} does not cover the fixed corpus order"
                )
            for corpus_id, value in rows:
                del corpus_id
                _absolute_path(f"derived phase {name}", value)
            object.__setattr__(self, name, rows)


def _canonical_file_uri(value: object, *, label: str) -> tuple[str, Path]:
    uri = _text(label, value)
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not parsed.path
    ):
        raise ProviderWorkflowOrchestrationError(f"{label} must be one local file URI")
    try:
        decoded = unquote(parsed.path, errors="strict")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProviderWorkflowOrchestrationError(f"{label} has invalid URI encoding") from exc
    path = _absolute_path(label, decoded)
    if path.as_uri() != uri:
        raise ProviderWorkflowOrchestrationError(f"{label} is not a canonical file URI")
    return uri, path


def _corpus_artifacts(
    manifest: Mapping[str, Any],
    *,
    role: str,
) -> Mapping[str, tuple[str, Path, str]]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list):
        raise ProviderWorkflowOrchestrationError("frozen C1 manifest lacks artifacts")
    rows: dict[str, tuple[str, Path, str]] = {}
    for item in raw:
        if not isinstance(item, Mapping) or item.get("role") != role:
            continue
        corpus_id = item.get("corpus_id")
        if corpus_id not in FIXED_CORPORA or type(corpus_id) is not str:
            raise ProviderWorkflowOrchestrationError(
                f"frozen C1 {role} artifact names another corpus"
            )
        if corpus_id in rows:
            raise ProviderWorkflowOrchestrationError(f"frozen C1 repeats {role} for {corpus_id}")
        uri, path = _canonical_file_uri(item.get("uri"), label=f"{corpus_id} {role} URI")
        digest = _digest(f"{corpus_id} {role} SHA-256", item.get("sha256"))
        rows[corpus_id] = (uri, path, digest)
    if set(rows) != set(FIXED_CORPORA):
        raise ProviderWorkflowOrchestrationError(
            f"frozen C1 {role} artifacts do not cover the fixed corpora"
        )
    return rows


def _root_online_claim(predecessor: VerifiedProviderPredecessor) -> ExecutionClaimContract:
    matches = [
        record
        for record in predecessor.records
        if record.state == "RUN_CLAIMED" and isinstance(record.payload, RunClaimBindings)
    ]
    if len(matches) != 1:
        raise ProviderWorkflowOrchestrationError(
            "provider predecessor lacks one root RUN_CLAIMED contract"
        )
    contract = matches[0].payload.execution_claim
    if not isinstance(contract, ExecutionClaimContract):
        raise ProviderWorkflowOrchestrationError("root online claim contract is untyped")
    return contract


def derive_post_online_phase_claim(
    *,
    phase: PostOnlinePhase,
    registration: VerifiedC1ProtocolRegistration,
    predecessor: VerifiedProviderPredecessor,
    plan: ProviderPhasePlan,
) -> DerivedPhaseClaim:
    """Derive label-release or analysis solely from C1 and the protected predecessor."""

    if phase not in {"label-release", "analysis"}:
        raise ProviderWorkflowOrchestrationError("post-online phase is not registered")
    if not isinstance(registration, VerifiedC1ProtocolRegistration):
        raise ProviderWorkflowOrchestrationError(
            "post-online phase derivation requires verified C1"
        )
    if not isinstance(predecessor, VerifiedProviderPredecessor):
        raise ProviderWorkflowOrchestrationError(
            "post-online phase derivation requires provider predecessor authority"
        )
    if not isinstance(plan, ProviderPhasePlan) or plan.phase != phase:
        raise ProviderWorkflowOrchestrationError(
            "post-online phase derivation received another C1 plan"
        )
    registration.assert_current()
    predecessor.assert_current()
    manifest_path = registration.package_root / "study-manifest.json"
    try:
        manifest = load_study_manifest(manifest_path)
        validate_study_manifest(manifest, require_frozen=True)
    except ValueError as exc:
        raise ProviderWorkflowOrchestrationError(
            "verified C1 study manifest cannot derive the provider phase"
        ) from exc
    if (
        manifest_sha256(manifest) != registration.manifest_sha256
        or plan.manifest_sha256 != registration.manifest_sha256
        or plan.c1_commit != registration.c1_commit
        or predecessor.state.manifest_sha256 != registration.manifest_sha256
        or predecessor.state.suite_attempt_id
        != derive_suite_attempt_id(registration.manifest_sha256)
    ):
        raise ProviderWorkflowOrchestrationError(
            "C1 plan and provider predecessor name different manifests"
        )
    expected_predecessor = "ONLINE_COMPLETE" if phase == "label-release" else "LABELS_RELEASED"
    if predecessor.state.state != expected_predecessor:
        raise ProviderWorkflowOrchestrationError(
            "provider predecessor differs from the requested post-online phase"
        )

    root = _root_online_claim(predecessor)
    expected_plan_uri = (
        root.label_release_provider_plan_uri
        if phase == "label-release"
        else root.analysis_provider_plan_uri
    )
    expected_plan_sha256 = (
        root.label_release_provider_plan_sha256
        if phase == "label-release"
        else root.analysis_provider_plan_sha256
    )
    expected_oci = (
        (root.release_oci_index_digest, root.release_oci_platform_manifest_digest)
        if phase == "label-release"
        else (root.oci_index_digest, root.analysis_oci_platform_manifest_digest)
    )
    if (
        root.manifest_sha256 != registration.manifest_sha256
        or root.c1_commit != registration.c1_commit
        or Path(plan.provider_plan_path).as_uri() != expected_plan_uri
        or plan.plan_sha256 != expected_plan_sha256
        or (plan.oci_index_digest, plan.oci_platform_manifest_digest) != expected_oci
        or plan.workflow_sha != root.claim_workflow_sha
        or plan.host_tools.contract_sha256 != root.host_tools.contract_sha256
    ):
        raise ProviderWorkflowOrchestrationError(
            "post-online C1 plan changes the root execution claim lineage"
        )

    ordered = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))
    phase_root = _absolute_path(
        "post-online phase evidence root",
        plan.phase_evidence_root(predecessor.state.suite_attempt_id),
    )
    input_paths: dict[str, Path] = {}
    supporting_paths: dict[str, Path] = {}
    output_paths: dict[str, Path] = {}
    rows: list[PhaseCorpusBinding] = []
    if phase == "label-release":
        ciphertexts = _corpus_artifacts(manifest, role="sealed-label-ciphertext")
        receipts = _corpus_artifacts(manifest, role="timelock-encryption-receipt")
        for corpus_id in ordered:
            input_uri, input_path, input_digest = ciphertexts[corpus_id]
            support_uri, support_path, support_digest = receipts[corpus_id]
            output_path = phase_root / corpus_id / "released-labels.json"
            rows.append(
                PhaseCorpusBinding(
                    corpus_id=corpus_id,
                    input_uri=input_uri,
                    input_sha256=input_digest,
                    supporting_input_uri=support_uri,
                    supporting_input_sha256=support_digest,
                    output_uri=output_path.as_uri(),
                )
            )
            input_paths[corpus_id] = input_path
            supporting_paths[corpus_id] = support_path
            output_paths[corpus_id] = output_path
    else:
        labels = predecessor.state.payload
        if (
            not isinstance(labels, tuple)
            or len(labels) != len(FIXED_CORPORA)
            or not all(isinstance(row, LabelCorpusClosure) for row in labels)
        ):
            raise ProviderWorkflowOrchestrationError(
                "analysis predecessor lacks five typed label closures"
            )
        online_matches = [
            record
            for record in predecessor.records
            if record.state == "ONLINE_COMPLETE" and isinstance(record.payload, OnlineSuiteClosure)
        ]
        if len(online_matches) != 1:
            raise ProviderWorkflowOrchestrationError(
                "analysis predecessor lacks one online completion closure"
            )
        by_label = {row.corpus_id: row for row in labels}
        by_online = {row.corpus_id: row for row in online_matches[0].payload.corpora}
        if set(by_label) != set(FIXED_CORPORA) or set(by_online) != set(FIXED_CORPORA):
            raise ProviderWorkflowOrchestrationError(
                "analysis predecessor closure does not cover the fixed corpora"
            )
        analysis_output = phase_root / "analysis"
        for corpus_id in ordered:
            label = by_label[corpus_id]
            online = by_online[corpus_id]
            input_uri, input_path = _canonical_file_uri(
                label.plaintext_uri, label=f"{corpus_id} released-label URI"
            )
            support_uri, support_path = _canonical_file_uri(
                online.output_uri, label=f"{corpus_id} online-output URI"
            )
            rows.append(
                PhaseCorpusBinding(
                    corpus_id=corpus_id,
                    input_uri=input_uri,
                    input_sha256=label.plaintext_sha256,
                    supporting_input_uri=support_uri,
                    supporting_input_sha256=online.sealed_launch_output_tree_sha256,
                    output_uri=analysis_output.as_uri(),
                )
            )
            input_paths[corpus_id] = input_path
            supporting_paths[corpus_id] = support_path
            output_paths[corpus_id] = analysis_output

    phase_input_sha256 = _sha256(
        _canonical_bytes(
            {
                "corpora": [
                    {
                        "corpus_id": row.corpus_id,
                        "input_sha256": row.input_sha256,
                        "input_uri": row.input_uri,
                        "supporting_input_sha256": row.supporting_input_sha256,
                        "supporting_input_uri": row.supporting_input_uri,
                    }
                    for row in rows
                ],
                "manifest_sha256": registration.manifest_sha256,
                "phase": phase,
                "predecessor_state_sha256": predecessor.state.record_sha256,
            }
        )
    )
    phase_output_sha256 = _sha256(
        _canonical_bytes(
            {
                "corpora": [
                    {"corpus_id": row.corpus_id, "output_uri": row.output_uri} for row in rows
                ],
                "manifest_sha256": registration.manifest_sha256,
                "phase": phase,
            }
        )
    )
    contract = PhaseClaimContract(
        phase=phase,
        repository=plan.repository,
        claim_workflow_path=plan.workflow_path,
        claim_workflow_ref=plan.workflow_ref,
        claim_workflow_sha=plan.workflow_sha,
        run_head_branch=plan.run_head_branch,
        claim_job_name=plan.claim_job_name,
        execute_job_name=plan.execute_job_name,
        claim_nonce=plan.claim_nonce,
        unique_runner_label=derive_phase_runner_label(plan.claim_nonce, phase),
        runner_id=plan.runner_id,
        runner_name=plan.runner_name,
        runner_group_id=plan.runner_group_id,
        runner_version=plan.runner_version,
        runner_archive_sha256=plan.runner_archive_sha256,
        provider_operating_system=plan.provider_operating_system,
        provider_architecture=plan.provider_architecture,
        host_tool_contract_sha256=plan.host_tools.contract_sha256,
        runtime_probe_receipt_sha256=plan.runtime_probe_receipt_sha256,
        c1_commit=registration.c1_commit,
        manifest_sha256=registration.manifest_sha256,
        c1_provider_plan_uri=Path(plan.provider_plan_path).as_uri(),
        c1_provider_plan_sha256=plan.plan_sha256,
        run_receipt_sha256=predecessor.state.run_receipt_sha256,
        oci_index_digest=plan.oci_index_digest,
        oci_platform_manifest_digest=plan.oci_platform_manifest_digest,
        tle_binary_sha256=(plan.tle_binary_sha256 if phase == "label-release" else None),
        online_execution_claim_contract_sha256=root.contract_sha256,
        predecessor_state_sha256=predecessor.state.record_sha256,
        predecessor_ledger_commit=predecessor.ledger_commit,
        corpora=tuple(rows),
        phase_input_aggregate_sha256=phase_input_sha256,
        phase_output_identity=phase_output_sha256,
        maximum_runtime_seconds=plan.maximum_runtime_seconds,
        label_release_beacon=(root.beacon if phase == "label-release" else None),
    )
    predecessor.assert_current()
    return DerivedPhaseClaim(
        contract=contract,
        input_paths=tuple((corpus_id, str(input_paths[corpus_id])) for corpus_id in ordered),
        supporting_input_paths=tuple(
            (corpus_id, str(supporting_paths[corpus_id])) for corpus_id in ordered
        ),
        output_paths=tuple((corpus_id, str(output_paths[corpus_id])) for corpus_id in ordered),
    )


@dataclass(frozen=True)
class EvidenceInventoryRow:
    role: str
    relative_path: str
    file_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        _text("evidence role", self.role)
        object.__setattr__(
            self,
            "relative_path",
            _relative_path("evidence relative_path", self.relative_path),
        )
        _digest("evidence file_sha256", self.file_sha256)
        _nonnegative("evidence byte_count", self.byte_count)

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> EvidenceInventoryRow:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="evidence inventory row",
            )
        )


def _inventory(
    rows: Sequence[EvidenceInventoryRow],
    *,
    label: str,
) -> tuple[EvidenceInventoryRow, ...]:
    values = tuple(rows)
    if not values or not all(isinstance(row, EvidenceInventoryRow) for row in values):
        raise ProviderWorkflowOrchestrationError(f"{label} lacks typed evidence rows")
    paths = [row.relative_path for row in values]
    roles = [row.role for row in values]
    aliases = [unicodedata.normalize("NFC", path).casefold() for path in paths]
    if (
        paths != sorted(paths, key=lambda path: path.encode("utf-8"))
        or len(paths) != len(set(paths))
        or len(roles) != len(set(roles))
        or len(aliases) != len(set(aliases))
    ):
        raise ProviderWorkflowOrchestrationError(
            f"{label} paths and roles must be unique UTF-8-byte sorted values"
        )
    return values


def inventory_sha256(rows: Sequence[EvidenceInventoryRow]) -> str:
    values = _inventory(rows, label="evidence inventory")
    return _sha256(_canonical_bytes([row.to_dict() for row in values]))


class _CanonicalReceipt:
    schema_version: str

    def to_dict(self) -> dict[str, object]:  # pragma: no cover - implemented by records
        raise NotImplementedError

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())


@dataclass(frozen=True)
class ProviderPrerequisiteReceipt(_CanonicalReceipt):
    phase: ProviderPhase
    suite_attempt_id: str
    manifest_sha256: str
    c1_commit: str
    c1_package_root: str
    c1_package_inventory_sha256: str
    c1_package_file_count: int
    zenodo_admission_sha256: str
    provider_plan_sha256: str
    provider_plan_file_sha256: str
    provider_plan_materialization_path: str
    provider_plan_templates_sha256: str
    runner_bootstrap_receipt_path: str
    runner_bootstrap_receipt_file_sha256: str
    runner_readiness_receipt_sha256: str
    predecessor_state: str
    predecessor_sequence: int
    predecessor_state_record_sha256: str
    predecessor_ledger_commit: str
    predecessor_ledger_tree: str
    predecessor_control_inventory_sha256: str
    predecessor_artifact_receipt_sha256: str
    predecessor_artifact_inventory_sha256: str
    predecessor_artifact_materialized_root: str
    workflow_context_sha256: str
    phase_evidence_root: str
    schema_version: str = PREREQUISITE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        phase = _phase(self.phase)
        for name in (
            "suite_attempt_id",
            "manifest_sha256",
            "c1_package_inventory_sha256",
            "zenodo_admission_sha256",
            "provider_plan_sha256",
            "provider_plan_file_sha256",
            "provider_plan_templates_sha256",
            "runner_bootstrap_receipt_file_sha256",
            "runner_readiness_receipt_sha256",
            "predecessor_state_record_sha256",
            "predecessor_control_inventory_sha256",
            "predecessor_artifact_receipt_sha256",
            "predecessor_artifact_inventory_sha256",
            "workflow_context_sha256",
        ):
            _digest(name, getattr(self, name))
        _commit("c1_commit", self.c1_commit)
        _commit("predecessor_ledger_commit", self.predecessor_ledger_commit)
        _commit("predecessor_ledger_tree", self.predecessor_ledger_tree)
        for name in (
            "c1_package_root",
            "provider_plan_materialization_path",
            "runner_bootstrap_receipt_path",
            "predecessor_artifact_materialized_root",
            "phase_evidence_root",
        ):
            _absolute_path(name, getattr(self, name))
        if self.suite_attempt_id != derive_suite_attempt_id(self.manifest_sha256):
            raise ProviderWorkflowOrchestrationError(
                "prerequisite suite attempt ID is not manifest-derived"
            )
        if self.c1_package_file_count != C1_REGISTRATION_PACKAGE_FILE_COUNT:
            raise ProviderWorkflowOrchestrationError(
                "prerequisite C1 package must contain exactly "
                f"{C1_REGISTRATION_PACKAGE_FILE_COUNT} files"
            )
        expected_state, expected_sequence = _PREDECESSOR[phase]
        if (
            self.predecessor_state != expected_state
            or self.predecessor_sequence != expected_sequence
        ):
            raise ProviderWorkflowOrchestrationError(
                "prerequisite predecessor differs from the phase state machine"
            )
        if self.schema_version != PREREQUISITE_RECEIPT_SCHEMA:
            raise ProviderWorkflowOrchestrationError("prerequisite receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> ProviderPrerequisiteReceipt:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="provider prerequisite receipt",
            )
        )


@dataclass(frozen=True)
class ProviderClaimReceipt(_CanonicalReceipt):
    phase: ProviderPhase
    suite_attempt_id: str
    manifest_sha256: str
    run_id: int
    workflow_context_sha256: str
    prerequisite_receipt_path: str
    prerequisite_receipt_file_sha256: str
    provider_plan_sha256: str
    provider_identity_sha256: str
    predecessor_state: str
    predecessor_sequence: int
    predecessor_state_record_sha256: str
    predecessor_ledger_commit: str
    target_state: str
    target_sequence: int
    target_state_record_sha256: str
    target_ledger_commit: str
    claim_contract_sha256: str
    publication_receipt_path: str
    publication_receipt_file_sha256: str
    claim_subject_path: str
    claim_subject_sha256: str
    claim_predicate_path: str
    claim_predicate_sha256: str
    runner_label: str
    suite_namespace: str
    expected_execute_job_name: str
    expected_claim_artifact_name: str
    schema_version: str = CLAIM_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        phase = _phase(self.phase)
        for name in (
            "suite_attempt_id",
            "manifest_sha256",
            "workflow_context_sha256",
            "prerequisite_receipt_file_sha256",
            "provider_plan_sha256",
            "provider_identity_sha256",
            "predecessor_state_record_sha256",
            "target_state_record_sha256",
            "claim_contract_sha256",
            "publication_receipt_file_sha256",
            "claim_subject_sha256",
            "claim_predicate_sha256",
        ):
            _digest(name, getattr(self, name))
        _positive("run_id", self.run_id)
        _commit("predecessor_ledger_commit", self.predecessor_ledger_commit)
        _commit("target_ledger_commit", self.target_ledger_commit)
        for name in (
            "prerequisite_receipt_path",
            "publication_receipt_path",
            "claim_subject_path",
            "claim_predicate_path",
            "suite_namespace",
        ):
            _absolute_path(name, getattr(self, name))
        if self.suite_attempt_id != derive_suite_attempt_id(self.manifest_sha256):
            raise ProviderWorkflowOrchestrationError(
                "claim suite attempt ID is not manifest-derived"
            )
        predecessor_state, predecessor_sequence = _PREDECESSOR[phase]
        target_state, target_sequence = _CLAIMED[phase]
        if (self.predecessor_state, self.predecessor_sequence) != (
            predecessor_state,
            predecessor_sequence,
        ) or (self.target_state, self.target_sequence) != (target_state, target_sequence):
            raise ProviderWorkflowOrchestrationError(
                "claim state pair differs from the phase state machine"
            )
        if _RUNNER_LABEL.fullmatch(self.runner_label) is None:
            raise ProviderWorkflowOrchestrationError("claim runner label is invalid")
        if self.expected_execute_job_name != PROVIDER_PHASE_JOB_NAMES[phase][1]:
            raise ProviderWorkflowOrchestrationError(
                "claim execute-job name differs from the fixed phase workflow"
            )
        namespace = _absolute_path("suite_namespace", self.suite_namespace)
        if namespace.name != f"suite-attempt-{self.suite_attempt_id}":
            raise ProviderWorkflowOrchestrationError("claim namespace name is not manifest-derived")
        expected_artifact = f"confirmatory-{phase}-claim-{self.suite_attempt_id}-{self.run_id}"
        if self.expected_claim_artifact_name != expected_artifact:
            raise ProviderWorkflowOrchestrationError(
                "claim artifact name is not derived from phase, suite, and run"
            )
        if self.schema_version != CLAIM_RECEIPT_SCHEMA:
            raise ProviderWorkflowOrchestrationError("provider claim receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> ProviderClaimReceipt:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="provider claim receipt",
            )
        )


@dataclass(frozen=True)
class ProviderTransitionPreparationReceipt(_CanonicalReceipt):
    mode: PreparationMode
    phase: ProviderPhase
    suite_attempt_id: str
    manifest_sha256: str
    workflow_context_sha256: str
    claim_receipt_path: str
    claim_receipt_file_sha256: str
    claim_state_record_sha256: str
    claim_ledger_commit: str
    provider_identity_sha256: str
    execute_job_id: int
    live_execute_job_receipt_path: str
    live_execute_job_receipt_file_sha256: str
    evidence_root: str
    evidence_inventory: tuple[EvidenceInventoryRow, ...]
    evidence_inventory_sha256: str
    target_state: str
    target_sequence: int
    target_state_record_sha256: str
    prepared_subject_path: str
    prepared_subject_sha256: str
    predicate_path: str
    predicate_sha256: str
    phase_closure_sha256: str
    failed_execute_job_receipt_sha256: str | None
    incident_inventory_sha256: str | None
    schema_version: str = PREPARATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        phase = _phase(self.phase)
        if self.mode not in {"completion", "failure"}:
            raise ProviderWorkflowOrchestrationError("preparation mode is not registered")
        for name in (
            "suite_attempt_id",
            "manifest_sha256",
            "workflow_context_sha256",
            "claim_receipt_file_sha256",
            "claim_state_record_sha256",
            "provider_identity_sha256",
            "live_execute_job_receipt_file_sha256",
            "evidence_inventory_sha256",
            "target_state_record_sha256",
            "prepared_subject_sha256",
            "predicate_sha256",
            "phase_closure_sha256",
        ):
            _digest(name, getattr(self, name))
        _commit("claim_ledger_commit", self.claim_ledger_commit)
        _positive("execute_job_id", self.execute_job_id)
        for name in (
            "claim_receipt_path",
            "live_execute_job_receipt_path",
            "evidence_root",
            "prepared_subject_path",
            "predicate_path",
        ):
            _absolute_path(name, getattr(self, name))
        if self.suite_attempt_id != derive_suite_attempt_id(self.manifest_sha256):
            raise ProviderWorkflowOrchestrationError(
                "preparation suite attempt ID is not manifest-derived"
            )
        rows = _inventory(self.evidence_inventory, label="preparation evidence inventory")
        if inventory_sha256(rows) != self.evidence_inventory_sha256:
            raise ProviderWorkflowOrchestrationError(
                "preparation evidence inventory digest differs"
            )
        object.__setattr__(self, "evidence_inventory", rows)
        root = _absolute_path("evidence_root", self.evidence_root)
        for row in rows:
            candidate = root.joinpath(*PurePosixPath(row.relative_path).parts)
            if candidate == root or root not in candidate.parents:
                raise ProviderWorkflowOrchestrationError(
                    "preparation evidence row escapes its root"
                )
        claim_state, claim_sequence = _CLAIMED[phase]
        del claim_state
        expected_state, expected_sequence = (
            ("FAILED", claim_sequence + 1) if self.mode == "failure" else _COMPLETED[phase]
        )
        if (self.target_state, self.target_sequence) != (
            expected_state,
            expected_sequence,
        ):
            raise ProviderWorkflowOrchestrationError(
                "prepared target differs from the phase state machine"
            )
        if self.mode == "completion":
            if (
                self.failed_execute_job_receipt_sha256 is not None
                or self.incident_inventory_sha256 is not None
            ):
                raise ProviderWorkflowOrchestrationError(
                    "completion preparation carries failure-only evidence"
                )
        else:
            if (
                self.failed_execute_job_receipt_sha256 is None
                or self.incident_inventory_sha256 is None
            ):
                raise ProviderWorkflowOrchestrationError(
                    "failure preparation requires failed-job and incident evidence"
                )
            _digest(
                "failed_execute_job_receipt_sha256",
                self.failed_execute_job_receipt_sha256,
            )
            _digest("incident_inventory_sha256", self.incident_inventory_sha256)
        if self.schema_version != PREPARATION_RECEIPT_SCHEMA:
            raise ProviderWorkflowOrchestrationError("preparation receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "evidence_inventory"
            },
            "evidence_inventory": [row.to_dict() for row in self.evidence_inventory],
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderTransitionPreparationReceipt:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="provider transition preparation receipt",
        )
        inventory = row["evidence_inventory"]
        if not isinstance(inventory, list):
            raise ProviderWorkflowOrchestrationError(
                "preparation evidence inventory must be an array"
            )
        return cls(
            **{key: item for key, item in row.items() if key != "evidence_inventory"},
            evidence_inventory=tuple(EvidenceInventoryRow.from_dict(item) for item in inventory),
        )


Receipt = ProviderPrerequisiteReceipt | ProviderClaimReceipt | ProviderTransitionPreparationReceipt


def _load_receipt(path: str | Path, receipt_type: type[Receipt], *, label: str) -> Receipt:
    candidate = _absolute_path(f"{label} path", str(path))
    encoded = _secure_receipt_bytes(candidate, label=label)
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ProviderWorkflowOrchestrationError(f"{label} must end with exactly one newline")
    value = _strict_object(encoded[:-1], label=label)
    receipt = receipt_type.from_dict(value)
    if encoded != receipt.canonical_file_bytes():
        raise ProviderWorkflowOrchestrationError(f"{label} bytes are not canonical")
    return receipt


def load_provider_prerequisite_receipt(
    path: str | Path,
) -> ProviderPrerequisiteReceipt:
    return _load_receipt(
        path,
        ProviderPrerequisiteReceipt,
        label="provider prerequisite receipt",
    )  # type: ignore[return-value]


def load_provider_claim_receipt(path: str | Path) -> ProviderClaimReceipt:
    return _load_receipt(
        path,
        ProviderClaimReceipt,
        label="provider claim receipt",
    )  # type: ignore[return-value]


def load_provider_transition_preparation_receipt(
    path: str | Path,
) -> ProviderTransitionPreparationReceipt:
    return _load_receipt(
        path,
        ProviderTransitionPreparationReceipt,
        label="provider transition preparation receipt",
    )  # type: ignore[return-value]


def write_provider_receipt(receipt: Receipt, path: str | Path) -> Path:
    if not isinstance(
        receipt,
        (
            ProviderPrerequisiteReceipt,
            ProviderClaimReceipt,
            ProviderTransitionPreparationReceipt,
        ),
    ):
        raise ProviderWorkflowOrchestrationError("provider receipt must use a closed typed schema")
    target = _absolute_path("provider receipt output", str(path))
    _write_exclusive(
        target,
        receipt.canonical_file_bytes(),
        label="provider workflow receipt",
    )
    if (
        _sha256(_secure_receipt_bytes(target, label="provider workflow receipt"))
        != receipt.file_sha256
    ):
        raise ProviderWorkflowOrchestrationError("provider workflow receipt failed exact readback")
    return target


def _create_command_output_dir(path: Path) -> Path:
    """Create one private command root below an already controlled parent."""

    if not path.is_absolute() or path.exists():
        raise ProviderWorkflowOrchestrationError(
            "provider command output directory must be new and absolute"
        )
    try:
        parent = path.parent.resolve(strict=True)
        metadata = path.parent.lstat()
    except OSError as exc:
        raise ProviderWorkflowOrchestrationError(
            "provider command output parent is unavailable"
        ) from exc
    if (
        parent != path.parent
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise ProviderWorkflowOrchestrationError("provider command output parent is not controlled")
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise ProviderWorkflowOrchestrationError(
            "cannot create provider command output directory"
        ) from exc
    return path


def _assert_prerequisite_evidence_matches_fresh_authority(
    persisted: ProviderPrerequisiteReceipt,
    fresh_fields: Mapping[str, object],
) -> None:
    """Cross-check stable evidence fields without treating the receipt as authority."""

    stable = (
        "phase",
        "suite_attempt_id",
        "manifest_sha256",
        "c1_commit",
        "c1_package_inventory_sha256",
        "c1_package_file_count",
        "provider_plan_sha256",
        "provider_plan_file_sha256",
        "provider_plan_templates_sha256",
        "runner_bootstrap_receipt_path",
        "runner_bootstrap_receipt_file_sha256",
        "predecessor_state",
        "predecessor_sequence",
        "predecessor_state_record_sha256",
        "predecessor_ledger_commit",
        "predecessor_ledger_tree",
        "predecessor_control_inventory_sha256",
        "predecessor_artifact_inventory_sha256",
        "workflow_context_sha256",
        "phase_evidence_root",
    )
    if any(getattr(persisted, name) != fresh_fields.get(name) for name in stable):
        raise ProviderWorkflowOrchestrationError(
            "persisted prerequisite evidence differs from fresh authority"
        )


def _production_verify_prerequisites_command(
    *,
    phase: ProviderPhase,
    suite_attempt_id: str,
    output_dir: Path,
    claim_receipt_path: Path | None,
    activate_and_execute: bool,
) -> Mapping[str, str]:
    context = ProviderWorkflowContext.from_environment(phase)
    if activate_and_execute:
        if claim_receipt_path is None:
            raise ProviderWorkflowOrchestrationError(
                "provider activation requires the downloaded claim receipt"
            )
        github_token = os.environ.get("GH_TOKEN")
        if (
            type(github_token) is not str
            or not github_token
            or github_token != github_token.strip()
        ):
            raise ProviderWorkflowOrchestrationError(
                "provider activation requires the ephemeral GitHub token"
            )
        artifact_id_text = os.environ.get("CLAIM_ARTIFACT_ID")
        artifact_digest = os.environ.get("CLAIM_ARTIFACT_DIGEST")
        inventory_sha256 = os.environ.get("CLAIM_PACKAGE_INVENTORY_SHA256")
        try:
            artifact_id = int(artifact_id_text or "")
        except ValueError as exc:
            raise ProviderWorkflowOrchestrationError(
                "provider activation claim artifact ID is malformed"
            ) from exc
        if artifact_id <= 0:
            raise ProviderWorkflowOrchestrationError(
                "provider activation claim artifact ID is malformed"
            )
        if (
            type(artifact_digest) is not str
            or not artifact_digest.startswith("sha256:")
            or _SHA256.fullmatch(artifact_digest.removeprefix("sha256:")) is None
        ):
            raise ProviderWorkflowOrchestrationError(
                "provider activation claim artifact digest is malformed"
            )
        if type(inventory_sha256) is not str or _SHA256.fullmatch(inventory_sha256) is None:
            raise ProviderWorkflowOrchestrationError(
                "provider activation claim inventory digest is malformed"
            )
        completion_anchor_token_fd: int | None = None
        raw_completion_anchor_token_fd = os.environ.get("COMPLETION_ANCHOR_TOKEN_FD")
        if phase == "label-release":
            try:
                completion_anchor_token_fd = int(raw_completion_anchor_token_fd or "")
            except ValueError as exc:
                raise ProviderWorkflowOrchestrationError(
                    "label-release activation requires a Zenodo token file descriptor"
                ) from exc
            if completion_anchor_token_fd < 0:
                raise ProviderWorkflowOrchestrationError(
                    "label-release activation requires a Zenodo token file descriptor"
                )
        elif raw_completion_anchor_token_fd is not None:
            raise ProviderWorkflowOrchestrationError(
                "only label-release activation accepts a Zenodo token file descriptor"
            )
        try:
            from .github_artifact_transport import UrllibGitHubArtifactReadApi
            from .github_state_attestation import GhApiClient
            from .provider_activation_factory import (
                ProviderActivationError,
                activate_and_execute_provider_phase,
            )

            api = GhApiClient()
            activated = activate_and_execute_provider_phase(
                context=context,
                phase=phase,
                suite_attempt_id=suite_attempt_id,
                artifact_id=artifact_id,
                artifact_digest=artifact_digest,
                expected_inventory_sha256=inventory_sha256,
                claim_receipt_destination=claim_receipt_path,
                output_dir=output_dir,
                github_api=api,
                artifact_api=UrllibGitHubArtifactReadApi(github_token),
                completion_anchor_token_fd=completion_anchor_token_fd,
            )
            return activated.output_fields()
        except ProviderWorkflowOrchestrationError:
            raise
        except ProviderActivationError as exc:
            raise ProviderWorkflowOrchestrationError(f"provider activation failed: {exc}") from exc
        except Exception as exc:
            raise ProviderWorkflowOrchestrationError(
                "provider activation could not establish authority"
            ) from exc
    if claim_receipt_path is not None:
        raise ProviderWorkflowOrchestrationError(
            "hosted prerequisite verification rejects a claim receipt"
        )
    github_token = os.environ.get("GH_TOKEN")
    if type(github_token) is not str or not github_token or github_token != github_token.strip():
        raise ProviderWorkflowOrchestrationError(
            "hosted prerequisite verification requires the ephemeral GitHub token"
        )
    try:
        from .github_artifact_transport import UrllibGitHubArtifactReadApi
        from .github_state_attestation import GhApiClient
        from .provider_prerequisite_factory import (
            HostedPrerequisiteError,
            build_hosted_production_prerequisites,
        )

        api = GhApiClient()
        admitted = build_hosted_production_prerequisites(
            context,
            phase,
            suite_attempt_id,
            output_dir,
            verified_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            runner_api=api,
            ledger_api=api,
            artifact_api=UrllibGitHubArtifactReadApi(github_token),
        )
        admitted.assert_current()
        receipt = ProviderPrerequisiteReceipt(**admitted.prerequisite_fields())
        receipt_path = output_dir / "provider-prerequisite-receipt.json"
        write_provider_receipt(receipt, receipt_path)
        admitted.assert_current()
    except ProviderWorkflowOrchestrationError:
        raise
    except HostedPrerequisiteError as exc:
        raise ProviderWorkflowOrchestrationError(
            f"hosted prerequisite verification failed: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderWorkflowOrchestrationError(
            "hosted prerequisite verification could not establish authority"
        ) from exc
    outputs = {
        **admitted.execution_output_fields(),
        "prerequisite_receipt_path": str(receipt_path),
        "prerequisite_receipt_sha256": receipt.file_sha256,
    }
    from .execution_claim import PREREQUISITE_OUTPUT_KEYS

    if set(outputs) != set(PREREQUISITE_OUTPUT_KEYS) or any(
        type(key) is not str or type(value) is not str for key, value in outputs.items()
    ):
        raise ProviderWorkflowOrchestrationError(
            "hosted prerequisite output interface differs from C0"
        )
    return outputs


def _production_claim_command(
    *,
    phase: ProviderPhase,
    suite_attempt_id: str,
    prerequisite_receipt_path: Path,
    output_dir: Path,
) -> Mapping[str, str]:
    context = ProviderWorkflowContext.from_environment(phase)
    persisted = load_provider_prerequisite_receipt(prerequisite_receipt_path)
    if (
        persisted.phase != phase
        or persisted.suite_attempt_id != suite_attempt_id
        or persisted.workflow_context_sha256 != context.identity_sha256
    ):
        raise ProviderWorkflowOrchestrationError(
            "provider claim received prerequisite evidence from another invocation"
        )
    github_token = os.environ.get("GH_TOKEN")
    if type(github_token) is not str or not github_token or github_token != github_token.strip():
        raise ProviderWorkflowOrchestrationError(
            "provider claim requires the ephemeral GitHub token"
        )
    root = _create_command_output_dir(output_dir)
    try:
        from .github_artifact_transport import UrllibGitHubArtifactReadApi
        from .github_state_attestation import GhApiClient
        from .provider_claim_publication import (
            ProviderClaimPublicationError,
            derive_and_publish_provider_claim,
        )
        from .provider_prerequisite_factory import (
            HostedPrerequisiteError,
            build_hosted_production_prerequisites,
        )

        api = GhApiClient()
        admitted = build_hosted_production_prerequisites(
            context,
            phase,
            suite_attempt_id,
            root / "fresh-prerequisites",
            verified_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            runner_api=api,
            ledger_api=api,
            artifact_api=UrllibGitHubArtifactReadApi(github_token),
        )
        _assert_prerequisite_evidence_matches_fresh_authority(
            persisted,
            admitted.prerequisite_fields(),
        )
        admitted.assert_current()
        result = derive_and_publish_provider_claim(
            registration=admitted.registration,
            plan=admitted.plan,
            predecessor=admitted.predecessor.predecessor,
            zenodo_admission=admitted.zenodo_admission,
            context=context,
            c1_manifest_rekor_integrated_at_utc=(admitted.manifest_rekor_integrated_at_utc),
            c1_registry_rekor_integrated_at_utc=(admitted.registry_record_rekor_integrated_at_utc),
            output_dir=root,
            github_api=api,
        )
        receipt = ProviderClaimReceipt(
            phase=phase,
            suite_attempt_id=suite_attempt_id,
            manifest_sha256=admitted.registration.manifest_sha256,
            run_id=context.run_id,
            workflow_context_sha256=context.identity_sha256,
            prerequisite_receipt_path=str(prerequisite_receipt_path),
            prerequisite_receipt_file_sha256=persisted.file_sha256,
            provider_plan_sha256=admitted.plan.plan_sha256,
            provider_identity_sha256=result.provider_identity.identity_sha256,
            predecessor_state=admitted.predecessor.predecessor.state.state,
            predecessor_sequence=admitted.predecessor.predecessor.state.sequence,
            predecessor_state_record_sha256=(admitted.predecessor.predecessor.state.record_sha256),
            predecessor_ledger_commit=admitted.predecessor.predecessor.ledger_commit,
            target_state=result.state.state,
            target_sequence=result.state.sequence,
            target_state_record_sha256=result.state.record_sha256,
            target_ledger_commit=result.publication_receipt.commit_oid,
            claim_contract_sha256=result.contract.contract_sha256,
            publication_receipt_path=str(result.publication_receipt_path),
            publication_receipt_file_sha256=result.publication_receipt.receipt_sha256,
            claim_subject_path=str(result.subject_path),
            claim_subject_sha256=result.subject_sha256,
            claim_predicate_path=str(result.predicate_path),
            claim_predicate_sha256=result.predicate_sha256,
            runner_label=result.runner_label,
            suite_namespace=str(result.suite_namespace),
            expected_execute_job_name=admitted.plan.execute_job_name,
            expected_claim_artifact_name=(
                f"confirmatory-{phase}-claim-{suite_attempt_id}-{context.run_id}"
            ),
        )
        claim_receipt_path = root / "claim-receipt.json"
        write_provider_receipt(receipt, claim_receipt_path)
    except ProviderWorkflowOrchestrationError:
        raise
    except (HostedPrerequisiteError, ProviderClaimPublicationError) as exc:
        raise ProviderWorkflowOrchestrationError(
            f"provider claim publication failed: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderWorkflowOrchestrationError(
            "provider claim publication could not establish authority"
        ) from exc
    outputs = {
        "claim_ledger_commit": result.publication_receipt.commit_oid,
        "claim_predicate_path": str(result.predicate_path),
        "claim_predicate_sha256": result.predicate_sha256,
        "claim_receipt_path": str(claim_receipt_path),
        "claim_receipt_sha256": receipt.file_sha256,
        "claim_state_sha256": result.state.record_sha256,
        "claim_subject_path": str(result.subject_path),
        "claim_subject_sha256": result.subject_sha256,
        "expected_execute_job_name": admitted.plan.execute_job_name,
        "provider_identity_sha256": result.provider_identity.identity_sha256,
        "runner_label": result.runner_label,
        "suite_namespace": str(result.suite_namespace),
    }
    from .execution_claim import CLAIM_OUTPUT_KEYS

    if set(outputs) != set(CLAIM_OUTPUT_KEYS):
        raise ProviderWorkflowOrchestrationError("provider claim output interface differs from C0")
    return outputs


def _controlled_evidence_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ProviderWorkflowOrchestrationError(f"{label} must be absolute")
    try:
        metadata = path.stat(follow_symlinks=False)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProviderWorkflowOrchestrationError(f"cannot inspect {label}") from exc
    if (
        resolved != path
        or path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise ProviderWorkflowOrchestrationError(f"{label} is not a controlled directory")
    return path


def _evidence_member_name(name: str, *, label: str) -> str:
    if (
        not name
        or name in {".", ".."}
        or name != unicodedata.normalize("NFC", name)
        or "/" in name
        or "\\" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise ProviderWorkflowOrchestrationError(f"{label} has a non-canonical member name")
    return name


def _regular_evidence_files(
    root: Path,
    *,
    label: str,
) -> tuple[tuple[str, Path], ...]:
    base = _controlled_evidence_directory(root, label=label)
    rows: list[tuple[str, Path]] = []
    total_bytes = 0

    def visit(directory: Path, relative: PurePosixPath | None) -> None:
        nonlocal total_bytes
        try:
            entries = sorted(
                os.scandir(directory),
                key=lambda item: item.name.encode("utf-8"),
            )
        except (OSError, UnicodeEncodeError) as exc:
            raise ProviderWorkflowOrchestrationError(f"cannot enumerate {label}") from exc
        for entry in entries:
            name = _evidence_member_name(entry.name, label=label)
            member = PurePosixPath(name) if relative is None else relative / name
            path = directory / name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ProviderWorkflowOrchestrationError(f"cannot inspect {label} member") from exc
            if stat.S_ISDIR(metadata.st_mode) and not entry.is_symlink():
                visit(path, member)
                continue
            if (
                not stat.S_ISREG(metadata.st_mode)
                or entry.is_symlink()
                or metadata.st_nlink != 1
                or metadata.st_size < 0
                or metadata.st_size > _MAX_EVIDENCE_FILE_BYTES
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise ProviderWorkflowOrchestrationError(
                    f"{label} contains a non-regular or aliased member"
                )
            total_bytes += metadata.st_size
            if total_bytes > _MAX_EVIDENCE_TOTAL_BYTES:
                raise ProviderWorkflowOrchestrationError(f"{label} exceeds its byte bound")
            rows.append((member.as_posix(), path))
            if len(rows) > _MAX_EVIDENCE_FILES:
                raise ProviderWorkflowOrchestrationError(f"{label} exceeds its file bound")

    visit(base, None)
    return tuple(rows)


def _file_stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _copy_regular_file_snapshot(source: Path, target: Path, *, label: str) -> str:
    if not source.is_absolute() or not target.is_absolute():
        raise ProviderWorkflowOrchestrationError(f"{label} paths must be absolute")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _controlled_evidence_directory(target.parent, label=f"{label} target parent")
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        source_fd = os.open(source, source_flags)
    except OSError as exc:
        raise ProviderWorkflowOrchestrationError(f"cannot open {label}") from exc
    target_fd: int | None = None
    try:
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > _MAX_EVIDENCE_FILE_BYTES
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise ProviderWorkflowOrchestrationError(
                f"{label} must be one bounded singly linked regular file"
            )
        target_fd = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > _MAX_EVIDENCE_FILE_BYTES:
                raise ProviderWorkflowOrchestrationError(f"{label} changed beyond its bound")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                view = view[written:]
        os.fsync(target_fd)
        after = os.fstat(source_fd)
        target_metadata = os.fstat(target_fd)
        if (
            _file_stat_identity(before) != _file_stat_identity(after)
            or copied != before.st_size
            or not stat.S_ISREG(target_metadata.st_mode)
            or target_metadata.st_nlink != 1
            or target_metadata.st_size != copied
        ):
            raise ProviderWorkflowOrchestrationError(f"{label} changed while copied")
        return digest.hexdigest()
    except OSError as exc:
        raise ProviderWorkflowOrchestrationError(f"cannot copy {label}") from exc
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)


def _copy_evidence_tree(source: Path, target: Path, *, label: str) -> None:
    _controlled_evidence_directory(source, label=label)
    if target.exists():
        raise ProviderWorkflowOrchestrationError(f"{label} target already exists")
    target.mkdir(mode=0o700, parents=True)
    for relative, source_path in _regular_evidence_files(source, label=label):
        target_path = target.joinpath(*PurePosixPath(relative).parts)
        _copy_regular_file_snapshot(source_path, target_path, label=f"{label} member")


def _hash_regular_evidence(path: Path, *, label: str) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProviderWorkflowOrchestrationError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > _MAX_EVIDENCE_FILE_BYTES
        ):
            raise ProviderWorkflowOrchestrationError(f"{label} is not a bounded regular file")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > _MAX_EVIDENCE_FILE_BYTES:
                raise ProviderWorkflowOrchestrationError(f"{label} exceeds its bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ProviderWorkflowOrchestrationError(f"cannot hash {label}") from exc
    finally:
        os.close(descriptor)
    if _file_stat_identity(before) != _file_stat_identity(after) or observed != before.st_size:
        raise ProviderWorkflowOrchestrationError(f"{label} changed while hashed")
    return digest.hexdigest(), observed


def _portable_inventory(root: Path) -> tuple[EvidenceInventoryRow, ...]:
    rows = []
    for relative, path in _regular_evidence_files(root, label="portable transition evidence"):
        digest, byte_count = _hash_regular_evidence(
            path,
            label="portable transition evidence member",
        )
        rows.append(
            EvidenceInventoryRow(
                role=f"transition-evidence:{relative}",
                relative_path=relative,
                file_sha256=digest,
                byte_count=byte_count,
            )
        )
    return _inventory(rows, label="portable transition evidence inventory")


def _write_canonical_json_file(path: Path, value: object, *, label: str) -> tuple[str, int]:
    encoded = _canonical_bytes(value) + b"\n"
    _write_exclusive(path, encoded, label=label)
    observed = _secure_receipt_bytes(path, label=label)
    if observed != encoded:
        raise ProviderWorkflowOrchestrationError(f"{label} failed exact readback")
    return _sha256(encoded), len(encoded)


@dataclass
class _ProviderClaimAuthority:
    phase: ProviderPhase
    suite_attempt_id: str
    root: Path
    api: Any
    artifact_api: Any
    _counter: int = 0

    def snapshot(self) -> Any:
        from .github_state_attestation import load_ledger_snapshot

        return load_ledger_snapshot(
            repository=REPOSITORY,
            suite_attempt_id=self.suite_attempt_id,
            api=self.api,
        )

    def recover(self) -> Any:
        from .provider_state_transport import materialize_provider_claim

        self._counter += 1
        parent = self.root / f"claim-recovery-{self._counter:04d}"
        parent.mkdir(mode=0o700)
        return materialize_provider_claim(
            self.phase,
            self.suite_attempt_id,
            parent,
            ledger_api=self.api,
            artifact_api=self.artifact_api,
        )


def _open_provider_claim_authority(
    *,
    phase: ProviderPhase,
    suite_attempt_id: str,
    root: Path,
    github_token: str,
) -> _ProviderClaimAuthority:
    from .github_artifact_transport import UrllibGitHubArtifactReadApi
    from .github_state_attestation import GhApiClient

    authority_root: Path | None = None
    for index in range(1, 9):
        candidate = root / f"fresh-claim-authority-{index:02d}"
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        authority_root = candidate
        break
    if authority_root is None:
        raise ProviderWorkflowOrchestrationError(
            "provider command exhausted its bounded fresh-authority roots"
        )
    return _ProviderClaimAuthority(
        phase=phase,
        suite_attempt_id=suite_attempt_id,
        root=authority_root,
        api=GhApiClient(),
        artifact_api=UrllibGitHubArtifactReadApi(github_token),
    )


def _unique_named_file(root: Path, name: str, *, label: str) -> Path:
    _evidence_member_name(name, label=label)
    matches = tuple(
        sorted(
            (path for path in root.rglob(name) if path.is_file() and not path.is_symlink()),
            key=lambda path: str(path).encode("utf-8"),
        )
    )
    if len(matches) != 1:
        raise ProviderWorkflowOrchestrationError(f"{label} is not a singleton")
    _hash_regular_evidence(matches[0], label=label)
    return matches[0]


def _claim_artifact_root(materialized: Any) -> Path:
    receipt = materialized.receipt
    predecessor = materialized.predecessor
    if not receipt.artifacts:
        raise ProviderWorkflowOrchestrationError("fresh claim has no retained artifact authority")
    latest = receipt.artifacts[-1]
    if (
        latest.sequence != predecessor.state.sequence
        or latest.state != predecessor.state.state
        or latest.ledger_commit != predecessor.ledger_commit
    ):
        raise ProviderWorkflowOrchestrationError(
            "fresh claim artifact authority differs from the ledger tip"
        )
    root = (
        Path(receipt.materialized_root)
        / "artifacts"
        / f"{predecessor.state.sequence:03d}-{latest.artifact_id}"
    )
    return _controlled_evidence_directory(root, label="fresh claim artifact")


@dataclass(frozen=True)
class _RecoveredProviderClaim:
    materialized: Any
    predecessor: VerifiedProviderPredecessor
    contract: ExecutionClaimContract | PhaseClaimContract
    provider_identity: ProviderExecutionIdentity
    claim_receipt: ProviderClaimReceipt
    claim_receipt_path: Path
    provider_plan: ProviderPhasePlan
    provider_plan_path: Path
    manifest: Mapping[str, Any]
    manifest_path: Path


def _recover_provider_claim(
    authority: _ProviderClaimAuthority,
    *,
    context: ProviderWorkflowContext,
    caller_claim_receipt_path: Path | None,
) -> _RecoveredProviderClaim:
    from .suite_attempt import PhaseClaimBindings

    materialized = authority.recover()
    predecessor = materialized.predecessor
    payload = predecessor.state.payload
    if authority.phase == "online" and isinstance(payload, RunClaimBindings):
        contract: ExecutionClaimContract | PhaseClaimContract = payload.execution_claim
        provider_identity = payload.provider_identity
    elif authority.phase != "online" and isinstance(payload, PhaseClaimBindings):
        contract = payload.phase_claim
        provider_identity = payload.provider_identity
    else:
        raise ProviderWorkflowOrchestrationError(
            "fresh claim payload differs from the registered provider phase"
        )
    artifact_root = _claim_artifact_root(materialized)
    retained_receipt_path = _unique_named_file(
        artifact_root,
        "claim-receipt.json",
        label="retained claim receipt",
    )
    retained_receipt = load_provider_claim_receipt(retained_receipt_path)
    if caller_claim_receipt_path is not None:
        caller = load_provider_claim_receipt(caller_claim_receipt_path)
        caller_bytes = _secure_receipt_bytes(
            caller_claim_receipt_path,
            label="caller claim receipt",
        )
        retained_bytes = _secure_receipt_bytes(
            retained_receipt_path,
            label="retained claim receipt",
        )
        if caller != retained_receipt or caller_bytes != retained_bytes:
            raise ProviderWorkflowOrchestrationError(
                "caller claim receipt differs from fresh retained authority"
            )
        claim_receipt = caller
        claim_receipt_path = caller_claim_receipt_path
    else:
        claim_receipt = retained_receipt
        claim_receipt_path = retained_receipt_path
    state = predecessor.state
    _, state_namespace = _canonical_file_uri(
        state.namespace_uri,
        label="fresh provider state namespace URI",
    )
    expected_state, expected_sequence = _CLAIMED[authority.phase]
    if (
        state.state != expected_state
        or state.sequence != expected_sequence
        or state.suite_attempt_id != authority.suite_attempt_id
        or claim_receipt.phase != authority.phase
        or claim_receipt.suite_attempt_id != authority.suite_attempt_id
        or claim_receipt.manifest_sha256 != state.manifest_sha256
        or claim_receipt.run_id != context.run_id
        or claim_receipt.provider_identity_sha256 != provider_identity.identity_sha256
        or claim_receipt.target_state != state.state
        or claim_receipt.target_sequence != state.sequence
        or claim_receipt.target_state_record_sha256 != state.record_sha256
        or claim_receipt.target_ledger_commit != predecessor.ledger_commit
        or claim_receipt.claim_contract_sha256 != contract.contract_sha256
        or claim_receipt.suite_namespace != str(state_namespace)
        or claim_receipt.expected_execute_job_name != PROVIDER_PHASE_JOB_NAMES[authority.phase][1]
    ):
        raise ProviderWorkflowOrchestrationError(
            "claim receipt evidence differs from fresh provider authority"
        )
    manifest_path = _unique_named_file(
        artifact_root,
        "study-manifest.json",
        label="retained frozen study manifest",
    )
    manifest = load_study_manifest(manifest_path)
    validate_study_manifest(manifest, require_frozen=True)
    if manifest_sha256(manifest) != state.manifest_sha256:
        raise ProviderWorkflowOrchestrationError(
            "retained study manifest differs from the fresh provider claim"
        )
    provider_plan_path = _unique_named_file(
        artifact_root,
        "provider-plan.materialized.json",
        label="retained materialized provider plan",
    )
    provider_plan_bytes = _secure_receipt_bytes(
        provider_plan_path,
        label="retained materialized provider plan",
    )
    if not provider_plan_bytes.endswith(b"\n") or provider_plan_bytes.endswith(b"\n\n"):
        raise ProviderWorkflowOrchestrationError(
            "retained materialized provider plan needs one terminal newline"
        )
    try:
        provider_plan = ProviderPhasePlan.from_dict(
            _strict_object(
                provider_plan_bytes[:-1],
                label="retained materialized provider plan",
            )
        )
    except Exception as exc:
        raise ProviderWorkflowOrchestrationError(
            "retained materialized provider plan is malformed"
        ) from exc
    if (
        provider_plan.canonical_file_bytes() != provider_plan_bytes
        or provider_plan.phase != authority.phase
        or provider_plan.repository != context.repository
        or provider_plan.workflow_path != context.workflow_path
        or provider_plan.workflow_ref != context.workflow_ref
        or provider_plan.workflow_sha != context.workflow_sha
        or provider_plan.manifest_sha256 != state.manifest_sha256
        or provider_plan.plan_sha256 != claim_receipt.provider_plan_sha256
        or provider_plan.execute_job_name != claim_receipt.expected_execute_job_name
    ):
        raise ProviderWorkflowOrchestrationError(
            "retained provider plan differs from the fresh claim authority"
        )
    predecessor.assert_current()
    return _RecoveredProviderClaim(
        materialized=materialized,
        predecessor=predecessor,
        contract=contract,
        provider_identity=provider_identity,
        claim_receipt=claim_receipt,
        claim_receipt_path=claim_receipt_path,
        provider_plan=provider_plan,
        provider_plan_path=provider_plan_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )


def _analysis_results_store(recovered: _RecoveredProviderClaim) -> Path:
    contract = recovered.contract
    if not isinstance(contract, PhaseClaimContract) or contract.phase != "analysis":
        raise ProviderWorkflowOrchestrationError(
            "analysis output authority requires the typed analysis claim"
        )
    contract_paths = tuple(
        _canonical_file_uri(
            binding.output_uri,
            label=f"{binding.corpus_id} analysis output URI",
        )[1]
        for binding in contract.corpora
    )
    sealed = recovered.manifest.get("sealed_execution")
    if not isinstance(sealed, Mapping):
        raise ProviderWorkflowOrchestrationError("frozen manifest lacks sealed execution controls")
    manifest_path = _canonical_file_uri(
        sealed.get("results_store"),
        label="confirmatory results store URI",
    )[1]
    if len(set(contract_paths)) != 1 or contract_paths[0] != manifest_path:
        raise ProviderWorkflowOrchestrationError(
            "analysis claim output differs from the frozen results store"
        )
    return manifest_path


def _analysis_store_entries(manifest_digest: str) -> tuple[str, ...]:
    digest = _digest("analysis manifest digest", manifest_digest)
    return tuple(
        sorted(
            (
                f"{digest}.confirmatory-analysis-attempt.json",
                f"{digest}.confirmatory-input-receipt.json",
                f"{digest}.confirmatory-input.json",
                f"{digest}.confirmatory-result-receipt.json",
                f"{digest}.confirmatory-result.json",
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )


def _load_phase_execution_receipt(
    *,
    phase: ProviderPhase,
    suite_attempt_id: str,
    evidence_root: Path,
    recovered: _RecoveredProviderClaim,
) -> Any:
    from .artifact_integrity import digest_directory_tree
    from .provider_phase_runtime import (
        PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME,
        LabelReleaseOutputAuthority,
        ProviderDriverOutput,
        ProviderPhaseExecutionReceipt,
    )

    path = evidence_root / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME
    encoded = _secure_receipt_bytes(path, label="provider phase execution receipt")
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ProviderWorkflowOrchestrationError(
            "provider phase execution receipt needs one terminal newline"
        )
    row = _closed(
        _strict_object(encoded[:-1], label="provider phase execution receipt"),
        frozenset(ProviderPhaseExecutionReceipt.__dataclass_fields__),
        label="provider phase execution receipt",
    )
    raw_outputs = row["outputs"]
    if not isinstance(raw_outputs, list):
        raise ProviderWorkflowOrchestrationError(
            "provider phase execution outputs must be an array"
        )
    outputs = []
    for raw in raw_outputs:
        output_row = _closed(
            raw,
            frozenset(ProviderDriverOutput.__dataclass_fields__),
            label="provider driver output",
        )
        entries = output_row["output_entries"]
        if not isinstance(entries, list):
            raise ProviderWorkflowOrchestrationError(
                "provider driver output entries must be an array"
            )
        raw_label_authority = output_row["label_release_authority"]
        if raw_label_authority is not None and not isinstance(
            raw_label_authority,
            Mapping,
        ):
            raise ProviderWorkflowOrchestrationError(
                "label-release output authority must be an object or null"
            )
        outputs.append(
            ProviderDriverOutput(
                **{
                    key: value
                    for key, value in output_row.items()
                    if key not in {"label_release_authority", "output_entries"}
                },
                output_entries=tuple(entries),
                label_release_authority=(
                    None
                    if raw_label_authority is None
                    else LabelReleaseOutputAuthority.from_dict(raw_label_authority)
                ),
            )
        )
    receipt = ProviderPhaseExecutionReceipt(
        **{key: value for key, value in row.items() if key != "outputs"},
        outputs=tuple(outputs),
    )
    if receipt.canonical_file_bytes() != encoded:
        raise ProviderWorkflowOrchestrationError(
            "provider phase execution receipt bytes are not canonical"
        )
    expected_ids = (
        tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
        if phase != "analysis"
        else ("all-five",)
    )
    observed_ids = tuple(output.corpus_id for output in receipt.outputs)
    claim_receipt = recovered.claim_receipt
    if (
        receipt.phase != phase
        or receipt.suite_attempt_id != suite_attempt_id
        or receipt.claim_receipt_file_sha256 != claim_receipt.file_sha256
        or receipt.provider_plan_sha256 != claim_receipt.provider_plan_sha256
        or receipt.provider_plan_file_sha256 != recovered.provider_plan.file_sha256
        or observed_ids != expected_ids
        or any(output.driver_id != _PHASE_DRIVER_IDS[phase] for output in receipt.outputs)
    ):
        raise ProviderWorkflowOrchestrationError(
            "provider phase execution receipt differs from the fresh claim"
        )
    for output in receipt.outputs:
        expected_root = (
            evidence_root / output.corpus_id
            if phase != "analysis"
            else _analysis_results_store(recovered)
        )
        if Path(output.output_root) != expected_root:
            raise ProviderWorkflowOrchestrationError(
                "provider driver output root differs from its claim-bound authority"
            )
        try:
            if phase == "analysis":
                from .offline_analysis_contract import (
                    OfflineAnalysisContractError,
                    load_offline_analysis_execution_receipt,
                )

                expected_entries = _analysis_store_entries(
                    recovered.predecessor.state.manifest_sha256
                )
                if output.output_entries != expected_entries:
                    raise ProviderWorkflowOrchestrationError(
                        "analysis output inventory differs from the five registered files"
                    )
                observed = digest_directory_tree(expected_root)
                if (
                    observed.entries != expected_entries
                    or observed.file_count != 5
                    or observed.directory_count != 0
                ):
                    raise ProviderWorkflowOrchestrationError(
                        "analysis results store contains pre-existing or extraneous evidence"
                    )
                assert output.analysis_execution_receipt_uri is not None
                assert output.analysis_execution_receipt_sha256 is not None
                assert output.analysis_execution_receipt_file_sha256 is not None
                execution_path = _canonical_file_uri(
                    output.analysis_execution_receipt_uri,
                    label="offline analysis execution receipt URI",
                )[1]
                try:
                    offline_execution = load_offline_analysis_execution_receipt(
                        execution_path,
                        expected_receipt_sha256=(output.analysis_execution_receipt_sha256),
                        expected_file_sha256=(output.analysis_execution_receipt_file_sha256),
                    )
                except OfflineAnalysisContractError as exc:
                    raise ProviderWorkflowOrchestrationError(
                        f"offline analysis execution receipt is invalid: {exc}"
                    ) from exc
                contract = recovered.contract
                if (
                    not isinstance(contract, PhaseClaimContract)
                    or offline_execution.suite_attempt_id
                    != recovered.predecessor.state.suite_attempt_id
                    or offline_execution.manifest_sha256
                    != recovered.predecessor.state.manifest_sha256
                    or offline_execution.run_receipt_sha256
                    != recovered.predecessor.state.run_receipt_sha256
                    or offline_execution.provider_state_record_sha256
                    != recovered.predecessor.state.record_sha256
                    or offline_execution.provider_ledger_commit
                    != recovered.predecessor.ledger_commit
                    or offline_execution.phase_claim_contract_sha256 != contract.contract_sha256
                    or offline_execution.phase_claim_state_sha256
                    != recovered.predecessor.state.record_sha256
                    or offline_execution.phase_claim_ledger_commit
                    != recovered.predecessor.ledger_commit
                    or offline_execution.provider_identity_sha256
                    != recovered.provider_identity.identity_sha256
                    or offline_execution.c1_commit != contract.c1_commit
                    or Path(
                        _canonical_file_uri(
                            offline_execution.result_uri,
                            label="offline execution result URI",
                        )[1]
                    ).parent
                    != expected_root
                ):
                    raise ProviderWorkflowOrchestrationError(
                        "offline execution receipt differs from fresh analysis authority"
                    )
            elif phase == "label-release":
                contract = recovered.contract
                if not isinstance(contract, PhaseClaimContract):
                    raise ProviderWorkflowOrchestrationError(
                        "label output inventory lacks its typed claim"
                    )
                matches = [
                    binding for binding in contract.corpora if binding.corpus_id == output.corpus_id
                ]
                if len(matches) != 1:
                    raise ProviderWorkflowOrchestrationError(
                        "label output inventory names another claim corpus"
                    )
                plaintext_path = _canonical_file_uri(
                    matches[0].output_uri,
                    label=f"{output.corpus_id} released plaintext URI",
                )[1]
                expected_entries = tuple(
                    sorted(
                        (
                            "timelock-decryption-receipt.json",
                            plaintext_path.name,
                        ),
                        key=lambda value: value.encode("utf-8"),
                    )
                )
                observed = digest_directory_tree(expected_root)
                if (
                    plaintext_path.parent != expected_root
                    or output.output_entries != expected_entries
                    or observed.entries != expected_entries
                    or observed.file_count != 2
                    or observed.directory_count != 0
                ):
                    raise ProviderWorkflowOrchestrationError(
                        "label output inventory differs from its exact two-file closure"
                    )
            else:
                observed = digest_directory_tree(expected_root)
        except Exception as exc:
            if isinstance(exc, ProviderWorkflowOrchestrationError):
                raise
            raise ProviderWorkflowOrchestrationError(
                "cannot rehash provider driver output"
            ) from exc
        if (
            observed.sha256 != output.output_tree_sha256
            or observed.entries != output.output_entries
        ):
            raise ProviderWorkflowOrchestrationError(
                "provider driver output changed after execution"
            )
    return receipt


@dataclass(frozen=True)
class _CompletionCandidate:
    state: Any
    supporting_files: tuple[tuple[str, Path], ...]


def _completion_live_job(
    *,
    authority: _ProviderClaimAuthority,
    recovered: _RecoveredProviderClaim,
    verified_at_utc: str,
) -> Any:
    from .execution_claim import verify_live_execute_job

    return verify_live_execute_job(
        api=authority.api,
        contract=recovered.contract,
        provider_identity=recovered.provider_identity,
        verified_at_utc=verified_at_utc,
    )


def _load_online_launch_inventory(
    *,
    suite_attempt_id: str,
) -> tuple[Path, Mapping[tuple[str, str], Path]]:
    runner_temp = _absolute_path("RUNNER_TEMP", os.environ.get("RUNNER_TEMP"))
    path = runner_temp / "online-activation" / "launch-receipt-inventory.json"
    encoded = _secure_receipt_bytes(path, label="online launch evidence inventory")
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ProviderWorkflowOrchestrationError(
            "online launch evidence inventory needs one terminal newline"
        )
    row = _closed(
        _strict_object(encoded[:-1], label="online launch evidence inventory"),
        frozenset({"evidence", "phase", "schema_version", "suite_attempt_id"}),
        label="online launch evidence inventory",
    )
    raw_evidence = row["evidence"]
    if not isinstance(raw_evidence, list):
        raise ProviderWorkflowOrchestrationError(
            "online launch inventory evidence must be an array"
        )
    expected_corpora = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
    expected_roles = (
        "runtime_attestation_plan",
        "runtime_attestation_receipt",
        "sealed_launch_receipt",
    )
    expected_order = tuple(
        (corpus_id, role) for corpus_id in expected_corpora for role in expected_roles
    )
    observed: list[tuple[str, str]] = []
    result: dict[tuple[str, str], Path] = {}
    for item in raw_evidence:
        value = _closed(
            item,
            frozenset({"corpus_id", "file_sha256", "path", "role"}),
            label="online launch evidence row",
        )
        corpus_id = value["corpus_id"]
        role = value["role"]
        if corpus_id not in FIXED_CORPORA or role not in expected_roles:
            raise ProviderWorkflowOrchestrationError("online launch evidence row is not registered")
        member = _absolute_path("online launch evidence path", value["path"])
        digest, _ = _hash_regular_evidence(member, label="online launch evidence")
        if digest != _digest("online launch evidence digest", value["file_sha256"]):
            raise ProviderWorkflowOrchestrationError(
                "online launch evidence changed after inventory closure"
            )
        key = (corpus_id, role)
        observed.append(key)
        result[key] = member
    if (
        row["phase"] != "online"
        or row["schema_version"] != "fractal-provider-launch-inventory-v1"
        or row["suite_attempt_id"] != suite_attempt_id
        or tuple(observed) != expected_order
        or len(result) != len(expected_order)
        or encoded != _canonical_bytes(row) + b"\n"
    ):
        raise ProviderWorkflowOrchestrationError(
            "online launch evidence inventory differs from its closed interface"
        )
    return path, result


def _online_completion_candidate(
    *,
    authority: _ProviderClaimAuthority,
    recovered: _RecoveredProviderClaim,
    evidence_root: Path,
    live_execute_job_receipt: Any,
    verified_at_utc: str,
) -> _CompletionCandidate:
    from .drand_beacon import QuicknetExecutionBeaconVerifier
    from .production_controls import (
        PREFLIGHT_CONTRACT_FILENAME,
        RUNTIME_PLAN_TRANSITION_RECEIPT_FILENAME,
        verify_production_run_closure_authority,
    )
    from .production_corpus_run import (
        RUNTIME_ATTESTATION_PLAN_FILENAME,
        RUNTIME_ATTESTATION_RECEIPT_FILENAME,
    )
    from .sealed_container_launcher import (
        load_preflight_launch_contract,
        load_runtime_plan_transition,
        load_sealed_launch_contract,
    )
    from .suite_attempt import SuiteOpenBindings, admit_run_claim_beacon, complete_online_suite

    contract = recovered.contract
    if not isinstance(contract, ExecutionClaimContract):
        raise ProviderWorkflowOrchestrationError(
            "online completion requires the typed execution claim"
        )
    beacon_verifier = QuicknetExecutionBeaconVerifier()
    beacon_bytes = beacon_verifier.fetch(contract.beacon)
    capability = admit_run_claim_beacon(
        recovered.predecessor,
        beacon_bytes=beacon_bytes,
        beacon_verifier=beacon_verifier,
        live_execute_job_receipt=live_execute_job_receipt,
        verified_at_utc=verified_at_utc,
        fresh_state_revalidator=lambda: authority.recover().predecessor,
    )
    opened = recovered.predecessor.records[0].payload
    if not isinstance(opened, SuiteOpenBindings):
        raise ProviderWorkflowOrchestrationError("online claim lacks its typed OPENED controls")
    finalization_uri, finalization_receipt_path = _canonical_file_uri(
        opened.production_finalization_receipt_uri,
        label="production finalization receipt URI",
    )
    del finalization_uri
    finalization_request_path = finalization_receipt_path.with_name("finalization-request.json")
    launch_inventory_path, launch_inventory = _load_online_launch_inventory(
        suite_attempt_id=authority.suite_attempt_id
    )
    closures: dict[str, Any] = {}
    plan_paths: dict[str, Path] = {}
    runtime_receipt_paths: dict[str, Path] = {}
    launch_paths: dict[str, Path] = {}
    supporting: list[tuple[str, Path]] = [
        ("online-launch-inventory", launch_inventory_path),
        ("production-finalization-request", finalization_request_path),
        ("production-finalization-receipt", finalization_receipt_path),
    ]
    for binding in opened.runtime_attestation_plans:
        corpus_id = binding.corpus_id
        _, sealed_contract_path = _canonical_file_uri(
            binding.sealed_launch_contract_uri,
            label=f"{corpus_id} sealed launch contract URI",
        )
        sealed_contract = load_sealed_launch_contract(sealed_contract_path)
        if (
            sealed_contract.contract_sha256 != binding.sealed_launch_contract_sha256
            or sealed_contract.file_sha256 != binding.sealed_launch_contract_file_sha256
        ):
            raise ProviderWorkflowOrchestrationError(
                f"{corpus_id} sealed launch contract differs from OPENED"
            )
        control_root = _absolute_path(
            f"{corpus_id} sealed control root",
            sealed_contract.geometry.control_mount.source,
        )
        preflight_path = control_root / PREFLIGHT_CONTRACT_FILENAME
        transition_path = control_root / RUNTIME_PLAN_TRANSITION_RECEIPT_FILENAME
        preflight = load_preflight_launch_contract(preflight_path)
        transition = load_runtime_plan_transition(transition_path)
        closures[corpus_id] = verify_production_run_closure_authority(
            finalization_request_path=finalization_request_path,
            finalization_receipt_path=finalization_receipt_path,
            preflight=preflight,
            transition=transition,
        )
        plan_paths[corpus_id] = launch_inventory[(corpus_id, "runtime_attestation_plan")]
        runtime_receipt_paths[corpus_id] = launch_inventory[
            (corpus_id, "runtime_attestation_receipt")
        ]
        launch_paths[corpus_id] = launch_inventory[(corpus_id, "sealed_launch_receipt")]
        if (
            plan_paths[corpus_id] != control_root / RUNTIME_ATTESTATION_PLAN_FILENAME
            or runtime_receipt_paths[corpus_id]
            != evidence_root / corpus_id / RUNTIME_ATTESTATION_RECEIPT_FILENAME
            or launch_paths[corpus_id].name != "sealed-launch-receipt.json"
        ):
            raise ProviderWorkflowOrchestrationError(
                f"{corpus_id} launch inventory changes a state-machine evidence path"
            )
        supporting.extend(
            (
                (f"sealed-launch-contract:{corpus_id}", sealed_contract_path),
                (f"preflight-contract:{corpus_id}", preflight_path),
                (f"runtime-plan-transition:{corpus_id}", transition_path),
                (f"runtime-attestation-plan:{corpus_id}", plan_paths[corpus_id]),
                (f"sealed-launch-receipt:{corpus_id}", launch_paths[corpus_id]),
            )
        )
    candidate = complete_online_suite(
        recovered.predecessor,
        run_claim=capability,
        verified_production_closures=closures,
        runtime_attestation_plan_paths=plan_paths,
        runtime_attestation_receipt_paths=runtime_receipt_paths,
        sealed_launch_receipt_paths=launch_paths,
    )
    transfer_path = recovered.predecessor.namespace.parent / (
        f"{recovered.predecessor.namespace.name}.output-transfer.json"
    )
    supporting.append(("suite-output-transfer-receipt", transfer_path))
    return _CompletionCandidate(state=candidate, supporting_files=tuple(supporting))


def _label_completion_candidate(
    *,
    authority: _ProviderClaimAuthority,
    recovered: _RecoveredProviderClaim,
    evidence_root: Path,
    live_execute_job_receipt: Any,
    phase_execution_receipt: Any,
) -> _CompletionCandidate:
    from .drand_beacon import QuicknetExecutionBeaconVerifier
    from .execution_claim import verify_live_execute_job
    from .post_online_completion import (
        POST_ONLINE_COMPLETION_AGGREGATE_FILENAME,
        revalidate_post_online_completion_authority,
    )
    from .provider_phase_runtime import (
        LabelReleaseOutputAuthority,
        ProviderPhaseExecutionReceipt,
    )
    from .suite_attempt import admit_label_release_claim_beacon, complete_label_release

    contract = recovered.contract
    if not isinstance(contract, PhaseClaimContract) or contract.label_release_beacon is None:
        raise ProviderWorkflowOrchestrationError(
            "label completion requires the typed beacon-bound phase claim"
        )
    if not isinstance(phase_execution_receipt, ProviderPhaseExecutionReceipt):
        raise ProviderWorkflowOrchestrationError(
            "label completion lacks its typed phase execution receipt"
        )
    authorities = {
        output.corpus_id: output.label_release_authority
        for output in phase_execution_receipt.outputs
        if isinstance(output.label_release_authority, LabelReleaseOutputAuthority)
    }
    if set(authorities) != set(FIXED_CORPORA):
        raise ProviderWorkflowOrchestrationError(
            "label phase receipt lacks five action authorities"
        )
    verifier = QuicknetExecutionBeaconVerifier()
    last_observation = datetime.fromisoformat(live_execute_job_receipt.verified_at_utc)
    aggregate_file_sha256: str | None = None
    aggregate_path: Path | None = None

    def fresh_capability() -> Any:
        nonlocal aggregate_file_sha256, aggregate_path, last_observation
        observed_at = datetime.now(timezone.utc)
        while observed_at <= last_observation:
            observed_at = datetime.now(timezone.utc)
        admitted_at_utc = observed_at.isoformat()
        last_observation = observed_at
        fresh_recovered = authority.recover()
        fresh_live = verify_live_execute_job(
            api=authority.api,
            contract=fresh_recovered.contract,
            provider_identity=fresh_recovered.provider_identity,
            verified_at_utc=admitted_at_utc,
        )
        beacon_bytes = verifier.fetch(contract.label_release_beacon)
        capability = admit_label_release_claim_beacon(
            fresh_recovered.predecessor,
            beacon_bytes=beacon_bytes,
            beacon_verifier=verifier,
            live_execute_job_receipt=fresh_live,
            verified_at_utc=admitted_at_utc,
            fresh_state_revalidator=lambda: authority.recover().predecessor,
        )
        completion = revalidate_post_online_completion_authority(
            fresh_recovered.predecessor,
            capability,
        )
        if (
            aggregate_file_sha256 is not None
            and completion.aggregate.file_sha256 != aggregate_file_sha256
        ):
            raise ProviderWorkflowOrchestrationError(
                "post-online completion aggregate changed during label closure"
            )
        aggregate_file_sha256 = completion.aggregate.file_sha256
        aggregate_path = completion.completion_root / POST_ONLINE_COMPLETION_AGGREGATE_FILENAME
        return capability

    capability = fresh_capability()
    assert aggregate_file_sha256 is not None
    receipts: dict[str, Path] = {}
    plaintexts: dict[str, Path] = {}
    for binding in contract.corpora:
        receipts[binding.corpus_id] = (
            evidence_root / binding.corpus_id / "timelock-decryption-receipt.json"
        )
        _, plaintext = _canonical_file_uri(
            binding.output_uri,
            label=f"{binding.corpus_id} released label URI",
        )
        if plaintext.parent != evidence_root / binding.corpus_id:
            raise ProviderWorkflowOrchestrationError(
                "released label output differs from the phase evidence root"
            )
        plaintexts[binding.corpus_id] = plaintext
    candidate = complete_label_release(
        recovered.predecessor,
        phase_claim=capability,
        phase_claim_factory=fresh_capability,
        manifest=recovered.manifest,
        decryption_receipt_paths=receipts,
        plaintext_paths=plaintexts,
        post_online_completion_aggregate_file_sha256=aggregate_file_sha256,
        label_release_authorities=authorities,
    )
    assert aggregate_path is not None
    return _CompletionCandidate(
        state=candidate,
        supporting_files=(
            ("frozen-study-manifest", recovered.manifest_path),
            ("post-online-completion-aggregate", aggregate_path),
        ),
    )


def _analysis_completion_candidate(
    *,
    authority: _ProviderClaimAuthority,
    recovered: _RecoveredProviderClaim,
    live_execute_job_receipt: Any,
    phase_execution_receipt: Any,
) -> _CompletionCandidate:
    from .confirmatory_execution import load_confirmatory_analysis_attempt_receipt
    from .suite_attempt import admit_analysis_claim, complete_confirmatory_analysis

    contract = recovered.contract
    if not isinstance(contract, PhaseClaimContract):
        raise ProviderWorkflowOrchestrationError(
            "analysis completion requires the typed phase claim"
        )
    capability = admit_analysis_claim(
        recovered.predecessor,
        live_execute_job_receipt=live_execute_job_receipt,
        fresh_state_revalidator=lambda: authority.recover().predecessor,
    )
    results_store = _analysis_results_store(recovered)
    manifest_digest = recovered.predecessor.state.manifest_sha256
    attempt_path = results_store / f"{manifest_digest}.confirmatory-analysis-attempt.json"
    receipt_path = results_store / f"{manifest_digest}.confirmatory-result-receipt.json"
    result_path = results_store / f"{manifest_digest}.confirmatory-result.json"
    attempt = load_confirmatory_analysis_attempt_receipt(attempt_path)
    if (
        len(phase_execution_receipt.outputs) != 1
        or phase_execution_receipt.outputs[0].corpus_id != "all-five"
    ):
        raise ProviderWorkflowOrchestrationError(
            "analysis phase receipt lacks its sole execution output"
        )
    execution_output = phase_execution_receipt.outputs[0]
    if (
        execution_output.analysis_execution_receipt_uri is None
        or execution_output.analysis_execution_receipt_sha256 is None
        or execution_output.analysis_execution_receipt_file_sha256 is None
    ):
        raise ProviderWorkflowOrchestrationError(
            "analysis phase receipt discarded offline execution evidence"
        )
    execution_path = _canonical_file_uri(
        execution_output.analysis_execution_receipt_uri,
        label="offline analysis execution receipt URI",
    )[1]
    candidate = complete_confirmatory_analysis(
        recovered.predecessor,
        phase_claim=capability,
        confirmatory_input_artifact_sha256=attempt.confirmatory_input_artifact_sha256,
        execution_receipt_path=execution_path,
        execution_receipt_sha256=(execution_output.analysis_execution_receipt_sha256),
        execution_receipt_file_sha256=(execution_output.analysis_execution_receipt_file_sha256),
        attempt_receipt_path=attempt_path,
        result_receipt_path=receipt_path,
        final_result_path=result_path,
    )
    supporting: list[tuple[str, Path]] = [
        ("frozen-study-manifest", recovered.manifest_path),
        ("offline-analysis-execution-receipt", execution_path),
        ("confirmatory-analysis-attempt", attempt_path),
        ("confirmatory-analysis-result-receipt", receipt_path),
        ("confirmatory-analysis-result", result_path),
    ]
    for role, suffix in (
        ("confirmatory-input", ".confirmatory-input.json"),
        ("confirmatory-input-receipt", ".confirmatory-input-receipt.json"),
    ):
        path = results_store / f"{manifest_digest}{suffix}"
        if not path.exists():
            raise ProviderWorkflowOrchestrationError(f"{role} evidence is absent")
        supporting.append((role, path))
    return _CompletionCandidate(state=candidate, supporting_files=tuple(supporting))


def _derive_provider_completion_candidate(
    *,
    authority: _ProviderClaimAuthority,
    recovered: _RecoveredProviderClaim,
    evidence_root: Path,
    live_execute_job_receipt: Any,
    verified_at_utc: str,
) -> _CompletionCandidate:
    phase_execution_receipt = _load_phase_execution_receipt(
        phase=authority.phase,
        suite_attempt_id=authority.suite_attempt_id,
        evidence_root=evidence_root,
        recovered=recovered,
    )
    if authority.phase == "online":
        result = _online_completion_candidate(
            authority=authority,
            recovered=recovered,
            evidence_root=evidence_root,
            live_execute_job_receipt=live_execute_job_receipt,
            verified_at_utc=verified_at_utc,
        )
    elif authority.phase == "label-release":
        result = _label_completion_candidate(
            authority=authority,
            recovered=recovered,
            evidence_root=evidence_root,
            live_execute_job_receipt=live_execute_job_receipt,
            phase_execution_receipt=phase_execution_receipt,
        )
    else:
        result = _analysis_completion_candidate(
            authority=authority,
            recovered=recovered,
            live_execute_job_receipt=live_execute_job_receipt,
            phase_execution_receipt=phase_execution_receipt,
        )
    expected_state, expected_sequence = _COMPLETED[authority.phase]
    if (
        result.state.state != expected_state
        or result.state.sequence != expected_sequence
        or result.state.previous_state_record_sha256 != recovered.predecessor.state.record_sha256
    ):
        raise ProviderWorkflowOrchestrationError(
            "completion primitive produced another state-machine candidate"
        )
    recovered.predecessor.assert_current()
    return result


def _state_payload_sha256(state: Any) -> str:
    payload = state.payload
    value: object
    if isinstance(payload, tuple):
        value = [row.to_dict() for row in payload]
    else:
        value = payload.to_dict()
    return _sha256(_canonical_bytes(value))


def _copy_supporting_files(
    supporting_files: Sequence[tuple[str, Path]],
    *,
    portable_root: Path,
) -> None:
    roles: set[str] = set()
    sources: set[Path] = set()
    for index, (role, source) in enumerate(supporting_files):
        _text("supporting evidence role", role)
        if role in roles or source in sources:
            raise ProviderWorkflowOrchestrationError(
                "completion supporting evidence repeats a role or source"
            )
        roles.add(role)
        sources.add(source)
        slug = re.sub(r"[^a-z0-9]+", "-", role.lower()).strip("-")
        if not slug:
            raise ProviderWorkflowOrchestrationError(
                "completion supporting evidence role has no path identity"
            )
        suffix = source.suffix if source.suffix in {".json", ".jsonl", ".txt"} else ".bin"
        target = portable_root / "authority" / f"{index:03d}-{slug}{suffix}"
        _copy_regular_file_snapshot(source, target, label=f"supporting evidence {role}")


def _assert_copied_label_completion_evidence(candidate: Any, portable_root: Path) -> None:
    if candidate.state != "LABELS_RELEASED":
        return
    closures = tuple(candidate.payload)
    expected_corpora = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
    if (
        len(closures) != len(expected_corpora)
        or not all(isinstance(row, LabelCorpusClosure) for row in closures)
        or tuple(row.corpus_id for row in closures) != expected_corpora
    ):
        raise ProviderWorkflowOrchestrationError(
            "label completion candidate lacks its exact five closures"
        )
    for closure in closures:
        _, source_receipt = _canonical_file_uri(
            closure.decryption_receipt_uri,
            label=f"{closure.corpus_id} decryption receipt URI",
        )
        _, source_plaintext = _canonical_file_uri(
            closure.plaintext_uri,
            label=f"{closure.corpus_id} released plaintext URI",
        )
        if (
            source_receipt.parent.name != closure.corpus_id
            or source_plaintext.parent != source_receipt.parent
        ):
            raise ProviderWorkflowOrchestrationError(
                "label completion paths differ from the corpus evidence layout"
            )
        copied_root = portable_root / "phase" / closure.corpus_id
        copied_receipt = copied_root / source_receipt.name
        copied_plaintext = copied_root / source_plaintext.name
        receipt_sha256, receipt_size = _hash_regular_evidence(
            copied_receipt,
            label=f"copied {closure.corpus_id} decryption receipt",
        )
        plaintext_sha256, plaintext_size = _hash_regular_evidence(
            copied_plaintext,
            label=f"copied {closure.corpus_id} released plaintext",
        )
        if (
            receipt_sha256 != closure.decryption_receipt_file_sha256
            or receipt_size != closure.decryption_receipt_byte_count
            or plaintext_sha256 != closure.plaintext_sha256
            or plaintext_size != closure.plaintext_byte_count
        ):
            raise ProviderWorkflowOrchestrationError(
                f"copied {closure.corpus_id} label evidence differs from its closure"
            )


def _write_job_receipt(receipt: Any, path: Path, *, label: str) -> tuple[str, int]:
    if not hasattr(receipt, "to_dict") or not hasattr(receipt, "receipt_sha256"):
        raise ProviderWorkflowOrchestrationError(f"{label} must use a typed receipt")
    return _write_canonical_json_file(path, receipt.to_dict(), label=label)


def _transition_predicate(
    *,
    mode: PreparationMode,
    phase: ProviderPhase,
    context: ProviderWorkflowContext,
    recovered: _RecoveredProviderClaim,
    candidate: Any,
    execute_job_id: int,
    evidence_inventory_sha256_value: str,
    phase_closure_sha256: str,
    failed_execute_job_receipt_sha256: str | None,
    incident_inventory_sha256: str | None,
) -> dict[str, object]:
    return {
        "attestation_predicate_type": _TRANSITION_PREDICATE_TYPES[(mode, phase)],
        "candidate": {
            "manifest_sha256": candidate.manifest_sha256,
            "previous_state_record_sha256": candidate.previous_state_record_sha256,
            "state": candidate.state,
            "state_record_sha256": candidate.record_sha256,
            "state_sequence": candidate.sequence,
            "suite_attempt_id": candidate.suite_attempt_id,
        },
        "claim": {
            "execute_job_id": execute_job_id,
            "ledger_commit": recovered.predecessor.ledger_commit,
            "provider_identity_sha256": recovered.provider_identity.identity_sha256,
            "state_record_sha256": recovered.predecessor.state.record_sha256,
        },
        "evidence": {
            "failed_execute_job_receipt_sha256": failed_execute_job_receipt_sha256,
            "incident_inventory_sha256": incident_inventory_sha256,
            "inventory_sha256": evidence_inventory_sha256_value,
            "phase_closure_sha256": phase_closure_sha256,
        },
        "mode": mode,
        "phase": phase,
        "schema_version": _TRANSITION_PREDICATE_SCHEMA,
        "workflow_context_sha256": context.identity_sha256,
    }


def _write_transition_preparation(
    *,
    mode: PreparationMode,
    phase: ProviderPhase,
    context: ProviderWorkflowContext,
    recovered: _RecoveredProviderClaim,
    candidate: Any,
    execute_job_id: int,
    job_receipt_path: Path,
    job_receipt_file_sha256: str,
    portable_root: Path,
    output_root: Path,
    failed_execute_job_receipt_sha256: str | None,
    incident_inventory_sha256: str | None,
) -> Mapping[str, str]:
    inventory = _portable_inventory(portable_root)
    inventory_digest = inventory_sha256(inventory)
    phase_closure_digest = _state_payload_sha256(candidate)
    preparation_root = output_root / "preparation"
    preparation_root.mkdir(mode=0o700)
    subject_path = preparation_root / "prepared-subject.json"
    subject_bytes = candidate.canonical_bytes() + b"\n"
    _write_exclusive(subject_path, subject_bytes, label="prepared transition subject")
    if _secure_receipt_bytes(subject_path, label="prepared transition subject") != subject_bytes:
        raise ProviderWorkflowOrchestrationError(
            "prepared transition subject failed exact readback"
        )
    subject_sha256 = _sha256(subject_bytes)
    if subject_sha256 != candidate.record_sha256:
        raise ProviderWorkflowOrchestrationError(
            "prepared transition subject differs from the typed state digest"
        )
    predicate_value = _transition_predicate(
        mode=mode,
        phase=phase,
        context=context,
        recovered=recovered,
        candidate=candidate,
        execute_job_id=execute_job_id,
        evidence_inventory_sha256_value=inventory_digest,
        phase_closure_sha256=phase_closure_digest,
        failed_execute_job_receipt_sha256=failed_execute_job_receipt_sha256,
        incident_inventory_sha256=incident_inventory_sha256,
    )
    predicate_name = (
        "completion-predicate.json" if mode == "completion" else "failure-predicate.json"
    )
    predicate_path = preparation_root / predicate_name
    predicate_sha256, _ = _write_canonical_json_file(
        predicate_path,
        predicate_value,
        label=f"{mode} predicate",
    )
    receipt = ProviderTransitionPreparationReceipt(
        mode=mode,
        phase=phase,
        suite_attempt_id=recovered.predecessor.state.suite_attempt_id,
        manifest_sha256=recovered.predecessor.state.manifest_sha256,
        workflow_context_sha256=context.identity_sha256,
        claim_receipt_path=str(recovered.claim_receipt_path),
        claim_receipt_file_sha256=recovered.claim_receipt.file_sha256,
        claim_state_record_sha256=recovered.predecessor.state.record_sha256,
        claim_ledger_commit=recovered.predecessor.ledger_commit,
        provider_identity_sha256=recovered.provider_identity.identity_sha256,
        execute_job_id=execute_job_id,
        live_execute_job_receipt_path=str(job_receipt_path),
        live_execute_job_receipt_file_sha256=job_receipt_file_sha256,
        evidence_root=str(portable_root),
        evidence_inventory=inventory,
        evidence_inventory_sha256=inventory_digest,
        target_state=candidate.state,
        target_sequence=candidate.sequence,
        target_state_record_sha256=candidate.record_sha256,
        prepared_subject_path=str(subject_path),
        prepared_subject_sha256=subject_sha256,
        predicate_path=str(predicate_path),
        predicate_sha256=predicate_sha256,
        phase_closure_sha256=phase_closure_digest,
        failed_execute_job_receipt_sha256=failed_execute_job_receipt_sha256,
        incident_inventory_sha256=incident_inventory_sha256,
    )
    receipt_name = f"{mode}-preparation-receipt.json"
    receipt_path = preparation_root / receipt_name
    write_provider_receipt(receipt, receipt_path)
    outputs = {
        "preparation_receipt_path": str(receipt_path),
        "preparation_receipt_sha256": receipt.file_sha256,
        "prepared_subject_path": str(subject_path),
        "prepared_subject_sha256": subject_sha256,
        f"{mode}_predicate_path": str(predicate_path),
        f"{mode}_predicate_sha256": predicate_sha256,
    }
    if mode == "failure":
        outputs["no_claim_to_fail"] = "false"
    return outputs


def _prepare_completion_transition(
    *,
    phase: ProviderPhase,
    suite_attempt_id: str,
    context: ProviderWorkflowContext,
    authority: _ProviderClaimAuthority,
    claim_receipt_path: Path,
    evidence_root: Path,
    output_root: Path,
) -> Mapping[str, str]:
    if context.job != "execute":
        raise ProviderWorkflowOrchestrationError(
            "completion preparation requires the self-hosted execute job"
        )
    _controlled_evidence_directory(evidence_root, label="phase evidence root")
    recovered = _recover_provider_claim(
        authority,
        context=context,
        caller_claim_receipt_path=claim_receipt_path,
    )
    verified_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    live = _completion_live_job(
        authority=authority,
        recovered=recovered,
        verified_at_utc=verified_at_utc,
    )
    if live.execute_job_id <= 0:
        raise ProviderWorkflowOrchestrationError("live execute job lacks a positive identity")
    derived = _derive_provider_completion_candidate(
        authority=authority,
        recovered=recovered,
        evidence_root=evidence_root,
        live_execute_job_receipt=live,
        verified_at_utc=verified_at_utc,
    )
    portable_root = output_root / "portable-evidence"
    portable_root.mkdir(mode=0o700)
    _copy_evidence_tree(
        evidence_root,
        portable_root / "phase",
        label="phase completion evidence",
    )
    _copy_supporting_files(derived.supporting_files, portable_root=portable_root)
    _assert_copied_label_completion_evidence(derived.state, portable_root)
    claim_copy = portable_root / "control" / "claim-receipt.json"
    _copy_regular_file_snapshot(
        claim_receipt_path,
        claim_copy,
        label="caller claim receipt evidence",
    )
    live_path = portable_root / "control" / "live-execute-job-receipt.json"
    live_file_sha256, _ = _write_job_receipt(
        live,
        live_path,
        label="live execute-job receipt",
    )
    recovered.predecessor.assert_current()
    return _write_transition_preparation(
        mode="completion",
        phase=phase,
        context=context,
        recovered=recovered,
        candidate=derived.state,
        execute_job_id=live.execute_job_id,
        job_receipt_path=live_path,
        job_receipt_file_sha256=live_file_sha256,
        portable_root=portable_root,
        output_root=output_root,
        failed_execute_job_receipt_sha256=None,
        incident_inventory_sha256=None,
    )


def _failure_status(conclusion: str) -> tuple[int | None, int | None]:
    if conclusion == "cancelled":
        return None, 15
    if conclusion == "timed_out":
        return 124, None
    return 1, None


def _failure_incident_value(
    *,
    phase: ProviderPhase,
    context: ProviderWorkflowContext,
    recovered: _RecoveredProviderClaim,
    failed: Any,
    partial_inventory_sha256: str,
) -> dict[str, object]:
    return {
        "claim_ledger_commit": recovered.predecessor.ledger_commit,
        "claim_state_sha256": recovered.predecessor.state.record_sha256,
        "execute_job_conclusion": failed.conclusion,
        "execute_job_id": failed.execute_job_id,
        "failed_execute_job_receipt_sha256": failed.receipt_sha256,
        "partial_evidence_inventory_sha256": partial_inventory_sha256,
        "phase": phase,
        "provider_identity_sha256": recovered.provider_identity.identity_sha256,
        "schema_version": _FAILURE_INCIDENT_SCHEMA,
        "suite_attempt_id": recovered.predecessor.state.suite_attempt_id,
        "workflow_context_sha256": context.identity_sha256,
    }


def _prepare_failure_transition(
    *,
    phase: ProviderPhase,
    context: ProviderWorkflowContext,
    authority: _ProviderClaimAuthority,
    evidence_root: Path,
    output_root: Path,
) -> Mapping[str, str]:
    from .execution_claim import (
        PartialEvidenceBinding,
        ProviderPhaseFailure,
        verify_failed_execute_job,
    )
    from .suite_attempt import fail_provider_candidate

    if context.job != "fail":
        raise ProviderWorkflowOrchestrationError(
            "failure preparation requires the hosted recovery job"
        )
    snapshot = authority.snapshot()
    expected_state, expected_sequence = _CLAIMED[phase]
    if (
        snapshot.tip.state.state != expected_state
        or snapshot.tip.state.sequence != expected_sequence
    ):
        return {
            "failure_predicate_path": "",
            "failure_predicate_sha256": "",
            "no_claim_to_fail": "true",
            "preparation_receipt_path": "",
            "preparation_receipt_sha256": "",
            "prepared_subject_path": "",
            "prepared_subject_sha256": "",
        }
    _controlled_evidence_directory(evidence_root, label="provider recovery evidence root")
    recovered = _recover_provider_claim(
        authority,
        context=context,
        caller_claim_receipt_path=None,
    )
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    failed = verify_failed_execute_job(
        api=authority.api,
        contract=recovered.contract,
        provider_identity=recovered.provider_identity,
        verified_at_utc=now,
    )
    portable_root = output_root / "portable-evidence"
    portable_root.mkdir(mode=0o700)
    _copy_evidence_tree(
        evidence_root,
        portable_root / "partial",
        label="provider partial failure evidence",
    )
    failed_path = portable_root / "failed-execute-job-receipt.json"
    failed_file_sha256, _ = _write_job_receipt(
        failed,
        failed_path,
        label="failed execute-job receipt",
    )
    partial_inventory = _portable_inventory(portable_root)
    partial_inventory_digest = inventory_sha256(partial_inventory)
    partial_bindings = tuple(
        PartialEvidenceBinding(
            relative_path=row.relative_path,
            byte_count=row.byte_count,
            file_sha256=row.file_sha256,
        )
        for row in partial_inventory
    )
    incident_path = portable_root / "provider-failure-incident.json"
    incident_file_sha256, incident_byte_count = _write_canonical_json_file(
        incident_path,
        _failure_incident_value(
            phase=phase,
            context=context,
            recovered=recovered,
            failed=failed,
            partial_inventory_sha256=partial_inventory_digest,
        ),
        label="provider failure incident",
    )
    exit_status, termination_signal = _failure_status(failed.conclusion)
    failure = ProviderPhaseFailure(
        phase=phase,
        claim_state_sha256=recovered.predecessor.state.record_sha256,
        claim_ledger_commit=recovered.predecessor.ledger_commit,
        provider_identity_sha256=recovered.provider_identity.identity_sha256,
        failed_execute_job_receipt_sha256=failed.receipt_sha256,
        execute_job_id=failed.execute_job_id,
        phase_input_sha256=recovered.contract.contract_sha256,
        exit_status=exit_status,
        termination_signal=termination_signal,
        incident_uri=incident_path.as_uri(),
        incident_byte_count=incident_byte_count,
        incident_file_sha256=incident_file_sha256,
        partial_evidence=partial_bindings,
    )
    candidate = fail_provider_candidate(
        recovered.predecessor,
        provider_failure=failure,
        failed_execute_job_receipt=failed,
    )
    expected_sequence = recovered.predecessor.state.sequence + 1
    if (
        candidate.state != "FAILED"
        or candidate.sequence != expected_sequence
        or candidate.previous_state_record_sha256 != recovered.predecessor.state.record_sha256
    ):
        raise ProviderWorkflowOrchestrationError(
            "failure primitive produced another state-machine candidate"
        )
    recovered.predecessor.assert_current()
    return _write_transition_preparation(
        mode="failure",
        phase=phase,
        context=context,
        recovered=recovered,
        candidate=candidate,
        execute_job_id=failed.execute_job_id,
        job_receipt_path=failed_path,
        job_receipt_file_sha256=failed_file_sha256,
        portable_root=portable_root,
        output_root=output_root,
        failed_execute_job_receipt_sha256=failed.receipt_sha256,
        incident_inventory_sha256=partial_inventory_digest,
    )


def _relocated_portable_root(preparation_receipt_path: Path) -> Path:
    candidates = (
        preparation_receipt_path.parent / "portable-evidence",
        preparation_receipt_path.parent.parent / "portable-evidence",
    )
    observed = []
    for candidate in candidates:
        try:
            observed.append(
                _controlled_evidence_directory(
                    candidate,
                    label="relocated portable transition evidence",
                )
            )
        except ProviderWorkflowOrchestrationError:
            continue
    unique = tuple(dict.fromkeys(observed))
    if len(unique) != 1:
        raise ProviderWorkflowOrchestrationError(
            "preparation receipt does not identify one fixed portable evidence root"
        )
    return unique[0]


def _fixed_prepared_file(
    preparation_receipt_path: Path,
    *,
    name: str,
    label: str,
) -> Path:
    _evidence_member_name(name, label=label)
    path = preparation_receipt_path.parent / name
    _hash_regular_evidence(path, label=label)
    return path


def _inventory_map(root: Path, *, label: str) -> Mapping[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    for relative, path in _regular_evidence_files(root, label=label):
        result[relative] = _hash_regular_evidence(path, label=f"{label} member")
    return result


def _assert_failure_source_matches_portable(
    *,
    evidence_root: Path,
    portable_root: Path,
) -> None:
    source = _inventory_map(evidence_root, label="provider recovery evidence root")
    retained = _inventory_map(
        portable_root / "partial",
        label="retained provider recovery evidence",
    )
    if source != retained:
        raise ProviderWorkflowOrchestrationError(
            "failure recovery evidence differs from its prepared portable copy"
        )


def _expected_prepared_predicate(
    *,
    preparation: ProviderTransitionPreparationReceipt,
    candidate: Any,
) -> dict[str, object]:
    return {
        "attestation_predicate_type": _TRANSITION_PREDICATE_TYPES[
            (preparation.mode, preparation.phase)
        ],
        "candidate": {
            "manifest_sha256": candidate.manifest_sha256,
            "previous_state_record_sha256": candidate.previous_state_record_sha256,
            "state": candidate.state,
            "state_record_sha256": candidate.record_sha256,
            "state_sequence": candidate.sequence,
            "suite_attempt_id": candidate.suite_attempt_id,
        },
        "claim": {
            "execute_job_id": preparation.execute_job_id,
            "ledger_commit": preparation.claim_ledger_commit,
            "provider_identity_sha256": preparation.provider_identity_sha256,
            "state_record_sha256": preparation.claim_state_record_sha256,
        },
        "evidence": {
            "failed_execute_job_receipt_sha256": (preparation.failed_execute_job_receipt_sha256),
            "incident_inventory_sha256": preparation.incident_inventory_sha256,
            "inventory_sha256": preparation.evidence_inventory_sha256,
            "phase_closure_sha256": preparation.phase_closure_sha256,
        },
        "mode": preparation.mode,
        "phase": preparation.phase,
        "schema_version": _TRANSITION_PREDICATE_SCHEMA,
        "workflow_context_sha256": preparation.workflow_context_sha256,
    }


def _load_prepared_candidate(
    *,
    preparation: ProviderTransitionPreparationReceipt,
    subject_path: Path,
    predicate_path: Path,
    portable_root: Path,
) -> Any:
    from .suite_attempt import SuiteStateRecord

    subject_bytes = _secure_receipt_bytes(
        subject_path,
        label="prepared provider transition subject",
    )
    if not subject_bytes.endswith(b"\n") or subject_bytes.endswith(b"\n\n"):
        raise ProviderWorkflowOrchestrationError(
            "prepared provider transition subject needs one terminal newline"
        )
    try:
        candidate = SuiteStateRecord.from_dict(
            _strict_object(subject_bytes[:-1], label="prepared provider transition subject")
        )
    except Exception as exc:
        raise ProviderWorkflowOrchestrationError(
            "prepared provider transition subject is not a typed state"
        ) from exc
    if (
        candidate.canonical_bytes() + b"\n" != subject_bytes
        or candidate.record_sha256 != _sha256(subject_bytes)
        or preparation.prepared_subject_sha256 != candidate.record_sha256
        or preparation.target_state != candidate.state
        or preparation.target_sequence != candidate.sequence
        or preparation.target_state_record_sha256 != candidate.record_sha256
        or preparation.phase_closure_sha256 != _state_payload_sha256(candidate)
    ):
        raise ProviderWorkflowOrchestrationError(
            "prepared subject differs from its closed preparation receipt"
        )
    predicate_bytes = _secure_receipt_bytes(
        predicate_path,
        label="prepared provider transition predicate",
    )
    expected_predicate = _expected_prepared_predicate(
        preparation=preparation,
        candidate=candidate,
    )
    expected_predicate_bytes = _canonical_bytes(expected_predicate) + b"\n"
    if (
        predicate_bytes != expected_predicate_bytes
        or _sha256(predicate_bytes) != preparation.predicate_sha256
    ):
        raise ProviderWorkflowOrchestrationError(
            "prepared predicate differs from its closed preparation receipt"
        )
    inventory = _portable_inventory(portable_root)
    if (
        inventory != preparation.evidence_inventory
        or inventory_sha256(inventory) != preparation.evidence_inventory_sha256
    ):
        raise ProviderWorkflowOrchestrationError(
            "portable transition evidence differs from the prepared closed inventory"
        )
    return candidate


def _assert_preparation_matches_fresh_claim(
    *,
    preparation: ProviderTransitionPreparationReceipt,
    recovered: _RecoveredProviderClaim,
    candidate: Any,
) -> None:
    predecessor = recovered.predecessor
    if (
        preparation.suite_attempt_id != predecessor.state.suite_attempt_id
        or preparation.manifest_sha256 != predecessor.state.manifest_sha256
        or preparation.claim_receipt_file_sha256 != recovered.claim_receipt.file_sha256
        or preparation.claim_state_record_sha256 != predecessor.state.record_sha256
        or preparation.claim_ledger_commit != predecessor.ledger_commit
        or preparation.provider_identity_sha256 != recovered.provider_identity.identity_sha256
        or candidate.previous_state_record_sha256 != predecessor.state.record_sha256
        or candidate.suite_attempt_id != predecessor.state.suite_attempt_id
        or candidate.manifest_sha256 != predecessor.state.manifest_sha256
        or candidate.run_receipt_sha256 != predecessor.state.run_receipt_sha256
        or candidate.namespace_uri != predecessor.state.namespace_uri
    ):
        raise ProviderWorkflowOrchestrationError(
            "preparation evidence differs from freshly recovered claim authority"
        )


def _assert_failure_candidate_evidence(
    *,
    preparation: ProviderTransitionPreparationReceipt,
    candidate: Any,
    portable_root: Path,
) -> None:
    from .execution_claim import ProviderPhaseFailure

    failure = candidate.payload
    if not isinstance(failure, ProviderPhaseFailure):
        raise ProviderWorkflowOrchestrationError(
            "failure preparation subject lacks typed provider failure evidence"
        )
    failed_path = portable_root / "failed-execute-job-receipt.json"
    failed_bytes = _secure_receipt_bytes(
        failed_path,
        label="failed execute-job receipt evidence",
    )
    incident_path = portable_root / "provider-failure-incident.json"
    incident_digest, incident_size = _hash_regular_evidence(
        incident_path,
        label="provider failure incident evidence",
    )
    partial = tuple(
        row
        for row in preparation.evidence_inventory
        if row.relative_path != "provider-failure-incident.json"
    )
    partial_digest = inventory_sha256(partial)
    if (
        not failed_bytes.endswith(b"\n")
        or _sha256(failed_bytes[:-1]) != failure.failed_execute_job_receipt_sha256
        or failure.failed_execute_job_receipt_sha256
        != preparation.failed_execute_job_receipt_sha256
        or failure.execute_job_id != preparation.execute_job_id
        or failure.incident_file_sha256 != incident_digest
        or failure.incident_byte_count != incident_size
        or partial_digest != preparation.incident_inventory_sha256
        or tuple(
            (row.relative_path, row.file_sha256, row.byte_count) for row in failure.partial_evidence
        )
        != tuple((row.relative_path, row.file_sha256, row.byte_count) for row in partial)
    ):
        raise ProviderWorkflowOrchestrationError(
            "failure candidate differs from retained job or incident evidence"
        )


def _publish_provider_transition(
    *,
    mode: PreparationMode,
    phase: ProviderPhase,
    suite_attempt_id: str,
    context: ProviderWorkflowContext,
    authority: _ProviderClaimAuthority,
    claim_receipt_path: Path | None,
    evidence_root: Path,
    attestation_bundle_path: Path,
    preparation_receipt_path: Path,
    output_root: Path,
) -> Mapping[str, str]:
    from .provider_transition_publication import verify_and_publish_provider_transition

    expected_job = "complete" if mode == "completion" else "fail"
    if context.job != expected_job:
        raise ProviderWorkflowOrchestrationError(
            "transition publication runs in another hosted job"
        )
    preparation = load_provider_transition_preparation_receipt(preparation_receipt_path)
    if (
        preparation.mode != mode
        or preparation.phase != phase
        or preparation.suite_attempt_id != suite_attempt_id
    ):
        raise ProviderWorkflowOrchestrationError(
            "transition preparation receipt belongs to another invocation"
        )
    subject_path = _fixed_prepared_file(
        preparation_receipt_path,
        name="prepared-subject.json",
        label="prepared provider transition subject",
    )
    predicate_path = _fixed_prepared_file(
        preparation_receipt_path,
        name=("completion-predicate.json" if mode == "completion" else "failure-predicate.json"),
        label="prepared provider transition predicate",
    )
    portable_root = _relocated_portable_root(preparation_receipt_path)
    candidate = _load_prepared_candidate(
        preparation=preparation,
        subject_path=subject_path,
        predicate_path=predicate_path,
        portable_root=portable_root,
    )
    _assert_copied_label_completion_evidence(candidate, portable_root)
    if mode == "completion":
        if claim_receipt_path is None:
            raise ProviderWorkflowOrchestrationError(
                "completion publication requires the exact caller claim receipt"
            )
        if portable_root.parent != evidence_root:
            raise ProviderWorkflowOrchestrationError(
                "completion portable evidence is outside the downloaded artifact"
            )
    else:
        if claim_receipt_path is not None:
            raise ProviderWorkflowOrchestrationError(
                "failure publication rejects caller-supplied claim evidence"
            )
        _assert_failure_source_matches_portable(
            evidence_root=evidence_root,
            portable_root=portable_root,
        )
        _assert_failure_candidate_evidence(
            preparation=preparation,
            candidate=candidate,
            portable_root=portable_root,
        )
    recovered = _recover_provider_claim(
        authority,
        context=context,
        caller_claim_receipt_path=claim_receipt_path,
    )
    _assert_preparation_matches_fresh_claim(
        preparation=preparation,
        recovered=recovered,
        candidate=candidate,
    )
    recovered.predecessor.assert_current()
    result = verify_and_publish_provider_transition(
        context=context,
        phase=phase,
        mode=mode,
        suite_attempt_id=suite_attempt_id,
        predecessor=recovered.predecessor,
        target=candidate,
        subject_path=subject_path,
        predicate_path=predicate_path,
        bundle_path=attestation_bundle_path,
        output_dir=output_root,
        github_api=authority.api,
    )
    outputs = {
        "ledger_commit": result.publication_receipt.commit_oid,
        "publication_receipt_path": str(result.publication_receipt_path),
        "publication_receipt_sha256": result.publication_receipt.receipt_sha256,
        "state_record_sha256": result.state.record_sha256,
    }
    if mode == "failure":
        outputs["no_claim_to_fail"] = "false"
    return outputs


def _production_transition_command(
    *,
    mode: PreparationMode,
    phase: ProviderPhase,
    suite_attempt_id: str,
    prepare: bool,
    publish: bool,
    claim_receipt_path: Path | None,
    evidence_root: Path,
    attestation_bundle_path: Path | None,
    preparation_receipt_path: Path | None,
    output_dir: Path,
) -> Mapping[str, str]:
    if (prepare, publish) not in {(True, False), (False, True)}:
        raise ProviderWorkflowOrchestrationError(
            "transition command requires exactly one prepare or publish operation"
        )
    context = ProviderWorkflowContext.from_environment(phase)
    github_token = os.environ.get("GH_TOKEN")
    if type(github_token) is not str or not github_token or github_token != github_token.strip():
        raise ProviderWorkflowOrchestrationError(
            "provider transition requires the ephemeral GitHub token"
        )
    if prepare or mode == "completion":
        root = _create_command_output_dir(output_dir)
    else:
        root = _controlled_evidence_directory(
            output_dir,
            label="provider failure publication output root",
        )
    try:
        authority = _open_provider_claim_authority(
            phase=phase,
            suite_attempt_id=suite_attempt_id,
            root=root,
            github_token=github_token,
        )
        if prepare and mode == "completion":
            if claim_receipt_path is None:
                raise ProviderWorkflowOrchestrationError(
                    "completion preparation requires a claim receipt"
                )
            outputs = _prepare_completion_transition(
                phase=phase,
                suite_attempt_id=suite_attempt_id,
                context=context,
                authority=authority,
                claim_receipt_path=claim_receipt_path,
                evidence_root=evidence_root,
                output_root=root,
            )
        elif prepare:
            if claim_receipt_path is not None:
                raise ProviderWorkflowOrchestrationError(
                    "failure preparation rejects caller claim receipts"
                )
            outputs = _prepare_failure_transition(
                phase=phase,
                context=context,
                authority=authority,
                evidence_root=evidence_root,
                output_root=root,
            )
        else:
            if attestation_bundle_path is None or preparation_receipt_path is None:
                raise ProviderWorkflowOrchestrationError(
                    "transition publication lacks attestation or preparation evidence"
                )
            outputs = _publish_provider_transition(
                mode=mode,
                phase=phase,
                suite_attempt_id=suite_attempt_id,
                context=context,
                authority=authority,
                claim_receipt_path=claim_receipt_path,
                evidence_root=evidence_root,
                attestation_bundle_path=attestation_bundle_path,
                preparation_receipt_path=preparation_receipt_path,
                output_root=root,
            )
    except ProviderWorkflowOrchestrationError:
        raise
    except Exception as exc:
        operation = "preparation" if prepare else "publication"
        raise ProviderWorkflowOrchestrationError(
            f"provider transition {operation} could not establish authority"
        ) from exc
    from .execution_claim import (
        PREPARE_COMMON_OUTPUT_KEYS,
        PUBLISH_OUTPUT_KEYS,
    )

    expected = (
        PREPARE_COMMON_OUTPUT_KEYS
        | (
            {"completion_predicate_path", "completion_predicate_sha256"}
            if mode == "completion"
            else {
                "failure_predicate_path",
                "failure_predicate_sha256",
                "no_claim_to_fail",
            }
        )
        if prepare
        else PUBLISH_OUTPUT_KEYS | ({"no_claim_to_fail"} if mode == "failure" else set())
    )
    if set(outputs) != set(expected) or any(
        type(key) is not str or type(value) is not str for key, value in outputs.items()
    ):
        raise ProviderWorkflowOrchestrationError(
            "provider transition output interface differs from C0"
        )
    return outputs


def _command_path(name: str, value: str | Path) -> Path:
    return _absolute_path(name, str(value))


def execute_verify_prerequisites_command(
    *,
    phase: ProviderPhase,
    suite_attempt_id: str,
    output_dir: str | Path,
    claim_receipt_path: str | Path | None = None,
    activate_and_execute: bool = False,
) -> Mapping[str, str]:
    """Run the hosted prerequisite verifier or the self-hosted activation path."""

    admitted_phase = _phase(phase)
    suite = _digest("suite_attempt_id", suite_attempt_id)
    output = _command_path("output_dir", output_dir)
    claim_path = (
        None
        if claim_receipt_path is None
        else _command_path("claim_receipt_path", claim_receipt_path)
    )
    if activate_and_execute != (claim_path is not None):
        raise ProviderWorkflowOrchestrationError("activation and claim-receipt presence must agree")
    return _production_verify_prerequisites_command(
        phase=admitted_phase,
        suite_attempt_id=suite,
        output_dir=output,
        claim_receipt_path=claim_path,
        activate_and_execute=activate_and_execute,
    )


def execute_claim_command(
    *,
    phase: ProviderPhase,
    suite_attempt_id: str,
    prerequisite_receipt_path: str | Path,
    output_dir: str | Path,
) -> Mapping[str, str]:
    """Publish the sole provider claim from freshly reverified prerequisites."""

    return _production_claim_command(
        phase=_phase(phase),
        suite_attempt_id=_digest("suite_attempt_id", suite_attempt_id),
        prerequisite_receipt_path=_command_path(
            "prerequisite_receipt_path", prerequisite_receipt_path
        ),
        output_dir=_command_path("output_dir", output_dir),
    )


def _execute_transition_command(
    *,
    mode: PreparationMode,
    phase: ProviderPhase,
    suite_attempt_id: str,
    prepare: bool,
    publish: bool,
    claim_receipt_path: str | Path | None,
    evidence_root: str | Path,
    attestation_bundle_path: str | Path | None,
    preparation_receipt_path: str | Path | None,
    output_dir: str | Path,
) -> Mapping[str, str]:
    if type(prepare) is not bool or type(publish) is not bool or prepare == publish:
        raise ProviderWorkflowOrchestrationError(
            "exactly one transition mode, prepare or publish, is required"
        )
    claim_path = (
        None
        if claim_receipt_path is None
        else _command_path("claim_receipt_path", claim_receipt_path)
    )
    bundle_path = (
        None
        if attestation_bundle_path is None
        else _command_path("attestation_bundle_path", attestation_bundle_path)
    )
    preparation_path = (
        None
        if preparation_receipt_path is None
        else _command_path("preparation_receipt_path", preparation_receipt_path)
    )
    if publish != (bundle_path is not None and preparation_path is not None):
        raise ProviderWorkflowOrchestrationError(
            "publish requires both attestation and preparation receipts"
        )
    return _production_transition_command(
        mode=mode,
        phase=_phase(phase),
        suite_attempt_id=_digest("suite_attempt_id", suite_attempt_id),
        prepare=prepare,
        publish=publish,
        claim_receipt_path=claim_path,
        evidence_root=_command_path("evidence_root", evidence_root),
        attestation_bundle_path=bundle_path,
        preparation_receipt_path=preparation_path,
        output_dir=_command_path("output_dir", output_dir),
    )


def execute_complete_command(
    *,
    phase: ProviderPhase,
    suite_attempt_id: str,
    prepare: bool,
    publish: bool,
    claim_receipt_path: str | Path | None,
    evidence_root: str | Path,
    attestation_bundle_path: str | Path | None,
    preparation_receipt_path: str | Path | None,
    output_dir: str | Path,
) -> Mapping[str, str]:
    """Prepare or publish the registered successful phase transition."""

    if claim_receipt_path is None:
        raise ProviderWorkflowOrchestrationError("completion requires its exact claim receipt")
    return _execute_transition_command(
        mode="completion",
        phase=phase,
        suite_attempt_id=suite_attempt_id,
        prepare=prepare,
        publish=publish,
        claim_receipt_path=claim_receipt_path,
        evidence_root=evidence_root,
        attestation_bundle_path=attestation_bundle_path,
        preparation_receipt_path=preparation_receipt_path,
        output_dir=output_dir,
    )


def execute_fail_command(
    *,
    phase: ProviderPhase,
    suite_attempt_id: str,
    prepare: bool,
    publish: bool,
    claim_receipt_path: str | Path | None,
    evidence_root: str | Path,
    attestation_bundle_path: str | Path | None,
    preparation_receipt_path: str | Path | None,
    output_dir: str | Path,
) -> Mapping[str, str]:
    """Prepare or publish FAILED after recovering the live provider claim."""

    if claim_receipt_path is not None:
        raise ProviderWorkflowOrchestrationError(
            "failure must recover the live claim and rejects a caller receipt"
        )
    return _execute_transition_command(
        mode="failure",
        phase=phase,
        suite_attempt_id=suite_attempt_id,
        prepare=prepare,
        publish=publish,
        claim_receipt_path=None,
        evidence_root=evidence_root,
        attestation_bundle_path=attestation_bundle_path,
        preparation_receipt_path=preparation_receipt_path,
        output_dir=output_dir,
    )
