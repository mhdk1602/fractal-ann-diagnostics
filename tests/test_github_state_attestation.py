from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

import fractal_ann_diagnostics.github_state_attestation as github_attestation
from fractal_ann_diagnostics.c0_evidence_release import canonical_apparatus_evidence_bytes
from fractal_ann_diagnostics.c0_public_verification import (
    C0_PUBLIC_VERIFICATION_SCHEMA,
    C0PublicVerificationReceipt,
    GitTagRow,
)
from fractal_ann_diagnostics.c1_manifest_transition import (
    C1ManifestTransitionError,
    C1ManifestTransitionReceipt,
)
from fractal_ann_diagnostics.github_state_attestation import (
    C0_PUBLIC_VERIFICATION_PATH,
    C0_REF,
    C1_LOCK_PATH,
    C1_MANIFEST_PATH,
    C1_REF,
    C1_RESERVATION_PATH,
    C1_TRANSITION_RECEIPT_PATH,
    COMMON_CONTROL_LIMITATION,
    FREEZE_TAG_RULE_TYPES,
    FREEZE_TAG_RULESET_INCLUDES,
    FREEZE_TAG_RULESET_NAME,
    GIT_IDENTITY_EMAIL,
    GIT_IDENTITY_NAME,
    LEDGER_PATH_PREFIX,
    LEDGER_REF_PREFIX,
    LEDGER_RULE_TYPES,
    LEDGER_RULESET_INCLUDE,
    LEDGER_RULESET_NAME,
    OIDC_ISSUER,
    PREDICATE_TYPE,
    REGISTRATION_PREDICATE_TYPE,
    REGISTRATION_WORKFLOW_PATH,
    REGISTRY_RECORD_PREDICATE_TYPE,
    REGISTRY_RECORD_SUBJECT_PATH,
    REKOR_IDENTITY,
    REKOR_URI,
    REPOSITORY,
    STATE_SERVICE_IDENTITY,
    STATE_SERVICE_URI,
    WORKFLOW_PATH,
    ZENODO_DRAFT_URI,
    ZENODO_RECORD_ID,
    ZENODO_REGISTRY_IDENTITY,
    ZENODO_REGISTRY_URI,
    ZENODO_RESERVATION_CREATED_AT_UTC,
    ZENODO_RESERVED_DOI,
    GhApiClient,
    GhAttestationVerifier,
    GhC1AttestationVerifier,
    GitHubSuiteEvidenceVerifier,
    LedgerSnapshot,
    LedgerTransition,
    SigstoreObservation,
    c1_registration_predicate,
    emit_attestation_evidence,
    install_freeze_tag_ruleset,
    install_ledger_ruleset,
    ledger_predicate,
    load_ledger_snapshot,
    materialize_protocol_registry_record,
    parse_sigstore_bundle,
    prepare_c1_registration,
    publish_candidate_ledger_transition,
    publish_ledger_transition,
    registry_record_predicate,
    required_freeze_tag_ruleset,
    validate_workflow_transition,
    verify_c1_registry_record_attestation,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA, ProtocolRegistryRecord
from fractal_ann_diagnostics.suite_attempt import (
    CorpusDigest,
    CorpusNamespace,
    CorpusRuntimePlanBinding,
    SuiteAttemptError,
    SuiteAttestationDescriptor,
    SuiteAttestationEvidence,
    SuiteOpenBindings,
    SuiteStateRecord,
    suite_attempt_id,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _blob_oid(encoded: bytes) -> str:
    header = f"blob {len(encoded)}\0".encode("ascii")
    return hashlib.sha1(header + encoded, usedforsecurity=False).hexdigest()


def _canonical_file(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _public_gh_result(binding: dict[str, object], *, marker: str) -> dict[str, object]:
    return {
        "attestation": {"bundle": {"marker": marker}},
        "verificationResult": {
            "mediaType": "application/vnd.dev.sigstore.verificationresult+json;version=0.1",
            "statement": {
                "subject": [
                    {
                        "digest": {"sha256": binding["asset_sha256"]},
                        "name": binding["asset_name"],
                    },
                    {
                        "digest": {"sha256": binding["checksum_asset_sha256"]},
                        "name": binding["checksum_asset_name"],
                    },
                ]
            },
        },
    }


def _c0_public_verification_receipt(
    *,
    c0_commit: str,
    frozen_manifest_bytes: bytes,
    binding: dict[str, object] | None = None,
) -> C0PublicVerificationReceipt:
    from test_study import _c0_evidence_release_binding

    admitted_binding = copy.deepcopy(
        _c0_evidence_release_binding(c0_commit) if binding is None else binding
    )
    release_id = 90210
    release_api: dict[str, object] = {
        "assets": [
            {
                "browser_download_url": admitted_binding["asset_url"],
                "digest": f"sha256:{admitted_binding['asset_sha256']}",
                "id": 901,
                "name": admitted_binding["asset_name"],
                "size": admitted_binding["asset_size"],
                "state": "uploaded",
                "url": (
                    "https://api.github.com/repos/"
                    f"{admitted_binding['repository']}/releases/assets/901"
                ),
            },
            {
                "browser_download_url": admitted_binding["checksum_asset_url"],
                "digest": f"sha256:{admitted_binding['checksum_asset_sha256']}",
                "id": 902,
                "name": admitted_binding["checksum_asset_name"],
                "size": admitted_binding["checksum_asset_size"],
                "state": "uploaded",
                "url": (
                    "https://api.github.com/repos/"
                    f"{admitted_binding['repository']}/releases/assets/902"
                ),
            },
        ],
        "assets_url": (
            "https://api.github.com/repos/"
            f"{admitted_binding['repository']}/releases/{release_id}/assets"
        ),
        "draft": False,
        "html_url": admitted_binding["release_url"],
        "id": release_id,
        "immutable": True,
        "name": "Confirmatory apparatus C0 evidence",
        "prerelease": False,
        "published_at": "2026-07-18T17:00:00Z",
        "tag_name": admitted_binding["release_tag"],
        "target_commitish": c0_commit,
        "url": (
            f"https://api.github.com/repos/{admitted_binding['repository']}/releases/{release_id}"
        ),
    }
    gh_version_text = (
        "gh version 2.96.0 (2026-07-02)\nhttps://github.com/cli/cli/releases/tag/v2.96.0\n"
    )
    tag_text = f"{c0_commit}\trefs/tags/{admitted_binding['release_tag']}\n"
    release_verification = _public_gh_result(admitted_binding, marker="release")
    asset_verification = _public_gh_result(admitted_binding, marker="asset")
    release_verification_text = _canonical_file(release_verification).decode("ascii")
    asset_verification_text = _canonical_file(asset_verification).decode("ascii")
    return C0PublicVerificationReceipt(
        binding_source_kind="frozen-manifest",
        binding_source_file_sha256=hashlib.sha256(frozen_manifest_bytes).hexdigest(),
        binding_sha256=hashlib.sha256(_canonical_file(admitted_binding)).hexdigest(),
        c0_evidence_release_binding=admitted_binding,
        repository=str(admitted_binding["repository"]),
        release_tag=str(admitted_binding["release_tag"]),
        target_commit=c0_commit,
        release_id=release_id,
        gh_version="2.96.0",
        gh_version_file_sha256=hashlib.sha256(gh_version_text.encode("ascii")).hexdigest(),
        gh_version_text=gh_version_text,
        release_api=release_api,
        release_api_file_sha256=hashlib.sha256(_canonical_file(release_api)).hexdigest(),
        tag_rows=(
            GitTagRow(
                object_id=c0_commit,
                ref=f"refs/tags/{admitted_binding['release_tag']}",
            ),
        ),
        tag_ls_remote_file_sha256=hashlib.sha256(tag_text.encode("ascii")).hexdigest(),
        tag_kind="lightweight",
        release_verification=release_verification,
        release_verification_file_sha256=hashlib.sha256(
            release_verification_text.encode("ascii")
        ).hexdigest(),
        release_verification_text=release_verification_text,
        asset_verification=asset_verification,
        asset_verification_file_sha256=hashlib.sha256(
            asset_verification_text.encode("ascii")
        ).hexdigest(),
        asset_verification_text=asset_verification_text,
        archive_name=str(admitted_binding["asset_name"]),
        archive_sha256=str(admitted_binding["asset_sha256"]),
        archive_size=int(admitted_binding["asset_size"]),
        checksum_name=str(admitted_binding["checksum_asset_name"]),
        checksum_sha256=str(admitted_binding["checksum_asset_sha256"]),
        checksum_size=int(admitted_binding["checksum_asset_size"]),
        checksum_text=(f"{admitted_binding['asset_sha256']}  {admitted_binding['asset_name']}\n"),
    )


def _write_public_verification(
    path: Path,
    receipt: C0PublicVerificationReceipt,
) -> None:
    path.write_bytes(receipt.canonical_file_bytes())
    path.chmod(0o600)


def _descriptor(*, key_digest: str = (b"k" * 32).hex()) -> SuiteAttestationDescriptor:
    return SuiteAttestationDescriptor(
        expected_signer_identity=(f"https://github.com/{REPOSITORY}/{WORKFLOW_PATH}@{C0_REF}"),
        expected_oidc_issuer=OIDC_ISSUER,
        expected_repository=REPOSITORY,
        expected_workflow=WORKFLOW_PATH,
        expected_git_ref=C0_REF,
        expected_signer_digest="1" * 40,
        transparency_log_identity=REKOR_IDENTITY,
        transparency_log_uri=REKOR_URI,
        transparency_log_public_key_sha256=key_digest,
        timestamp_authority_identity=REKOR_IDENTITY,
        timestamp_authority_uri=REKOR_URI,
        timestamp_authority_public_key_sha256=key_digest,
        state_service_identity=STATE_SERVICE_IDENTITY,
        state_service_uri=STATE_SERVICE_URI,
        state_key_prefix=LEDGER_REF_PREFIX,
    )


def _open_state(tmp_path: Path) -> tuple[Path, SuiteAttestationDescriptor, SuiteStateRecord]:
    manifest_digest = _digest("manifest")
    attempt_id = suite_attempt_id(manifest_digest)
    namespace = tmp_path / f"suite-attempt-{attempt_id}"
    namespace.mkdir(mode=0o700)
    descriptor = _descriptor()
    (namespace / "attestation-descriptor.json").write_bytes(descriptor.canonical_bytes() + b"\n")
    finalization_path = namespace / "production-control-finalization-receipt.json"
    finalization_bytes = b'{"fixture":"production-finalization"}\n'
    finalization_path.write_bytes(finalization_bytes)
    ordered = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))
    contract_paths: dict[str, Path] = {}
    contract_digests: dict[str, str] = {}
    for corpus_id in ordered:
        path = namespace / "sealed-contracts" / corpus_id / "sealed-launch-contract.json"
        path.parent.mkdir(mode=0o700, parents=True)
        encoded = (
            json.dumps(
                {"corpus_id": corpus_id, "fixture": "sealed-launch-contract"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        path.write_bytes(encoded)
        contract_paths[corpus_id] = path
        contract_digests[corpus_id] = hashlib.sha256(encoded).hexdigest()
    payload = SuiteOpenBindings(
        protocol_registration_receipt_sha256=_digest("registration"),
        protocol_registration_receipt_file_sha256=_digest("registration-file"),
        protocol_registry_record_sha256=_digest("registry-record"),
        registered_at_utc="2026-07-14T12:00:00+00:00",
        run_receipt_file_sha256=_digest("run-file"),
        run_started_at_utc="2026-07-14T12:01:00+00:00",
        code_commit="1" * 40,
        runner_image=f"ghcr.io/example/runner@sha256:{'2' * 64}",
        attestation_descriptor_sha256=descriptor.descriptor_sha256,
        production_finalization_receipt_uri=finalization_path.as_uri(),
        production_finalization_receipt_file_sha256=hashlib.sha256(finalization_bytes).hexdigest(),
        production_finalization_request_sha256=_digest("production-finalization-request"),
        provisional_closure_tree_sha256=_digest("provisional-closure-tree"),
        instantiated_closure_tree_sha256=_digest("instantiated-closure-tree"),
        runtime_attestation_plans=tuple(
            CorpusRuntimePlanBinding(
                corpus_id=corpus_id,
                plan_sha256=_digest(f"runtime-plan:{corpus_id}"),
                file_sha256=_digest(f"runtime-plan-file:{corpus_id}"),
                production_run_closure_binding_receipt_sha256=_digest(
                    f"production-run-closure-binding:{corpus_id}"
                ),
                registered_plan_instantiation_receipt_sha256=_digest(
                    f"registered-plan-instantiation:{corpus_id}"
                ),
                registered_plan_instantiation_file_sha256=_digest(
                    f"registered-plan-instantiation-file:{corpus_id}"
                ),
                sealed_launch_contract_uri=contract_paths[corpus_id].as_uri(),
                sealed_launch_contract_sha256=_digest(f"sealed-launch-contract:{corpus_id}"),
                sealed_launch_contract_file_sha256=contract_digests[corpus_id],
            )
            for corpus_id in ordered
        ),
        execution_artifacts=tuple(
            CorpusDigest(corpus_id, _digest(f"execution:{corpus_id}")) for corpus_id in ordered
        ),
        staging_namespaces=tuple(
            CorpusNamespace(corpus_id, (namespace / "staging" / corpus_id).as_uri())
            for corpus_id in ordered
        ),
        output_namespaces=tuple(
            CorpusNamespace(corpus_id, (namespace / "online" / corpus_id).as_uri())
            for corpus_id in ordered
        ),
    )
    state = SuiteStateRecord(
        suite_attempt_id=attempt_id,
        manifest_sha256=manifest_digest,
        run_receipt_sha256=_digest("run"),
        namespace_uri=namespace.as_uri(),
        sequence=0,
        state="OPENED",
        previous_state_record_sha256=None,
        payload=payload,
    )
    return namespace, descriptor, state


class _Api:
    def __init__(
        self,
        state: SuiteStateRecord,
        *,
        tip: str = "a" * 40,
        force_pushes: bool = False,
        extra_path: str | None = None,
    ) -> None:
        self.state = state
        self.tip = tip
        self.tree = "b" * 40
        self.state_bytes = state.canonical_bytes() + b"\n"
        self.blob = _blob_oid(self.state_bytes)
        self.path = f"{LEDGER_PATH_PREFIX}/{state.suite_attempt_id}/000.state.json"
        namespace = Path(state.namespace_uri.removeprefix("file://"))
        self.controls = github_attestation._local_ledger_controls(namespace, state)
        self.control_inventory = github_attestation._ledger_control_inventory(
            state.suite_attempt_id,
            self.controls,
        )
        self.control_inventory_path = github_attestation._control_inventory_path(
            state.suite_attempt_id
        )
        self.control_blobs = {control.blob_oid: control.encoded for control in self.controls}
        self.control_blobs[_blob_oid(self.control_inventory)] = self.control_inventory
        self.force_pushes = force_pushes
        self.extra_path = extra_path

    def get(self, endpoint: str) -> object:
        branch = f"confirmatory-ledger/{self.state.suite_attempt_id}"
        if endpoint == f"repos/{REPOSITORY}/rulesets?includes_parents=false&per_page=100":
            return [{"id": 17, "name": LEDGER_RULESET_NAME}]
        if endpoint == f"repos/{REPOSITORY}/rulesets/17":
            return {
                "bypass_actors": [],
                "conditions": {"ref_name": {"exclude": [], "include": [LEDGER_RULESET_INCLUDE]}},
                "enforcement": "active",
                "id": 17,
                "name": LEDGER_RULESET_NAME,
                "rules": [{"type": rule_type} for rule_type in sorted(LEDGER_RULE_TYPES)],
                "target": "branch",
            }
        if endpoint == f"repos/{REPOSITORY}/git/ref/heads/{branch}":
            return {
                "object": {"sha": self.tip, "type": "commit"},
                "ref": f"{LEDGER_REF_PREFIX}/{self.state.suite_attempt_id}",
            }
        encoded_branch = branch.replace("/", "%2F")
        if endpoint == f"repos/{REPOSITORY}/branches/{encoded_branch}":
            return {
                "commit": {"sha": self.tip},
                "name": branch,
                "protected": True,
            }
        if endpoint == (f"repos/{REPOSITORY}/rules/branches/{encoded_branch}?per_page=100"):
            rule_types = ["deletion", "required_linear_history"]
            if not self.force_pushes:
                rule_types.append("non_fast_forward")
            return [{"ruleset_id": 17, "type": rule_type} for rule_type in rule_types]
        commit_prefix = f"repos/{REPOSITORY}/git/commits/"
        if endpoint.startswith(commit_prefix):
            commit_oid = endpoint.removeprefix(commit_prefix)
            return {
                "author": {
                    "date": "2026-07-14T12:00:00+00:00",
                    "email": GIT_IDENTITY_EMAIL,
                    "name": GIT_IDENTITY_NAME,
                },
                "committer": {
                    "date": "2026-07-14T12:00:00+00:00",
                    "email": GIT_IDENTITY_EMAIL,
                    "name": GIT_IDENTITY_NAME,
                },
                "message": (
                    f"confirmatory-state {self.state.suite_attempt_id} 000 "
                    f"OPENED {self.state.record_sha256}"
                ),
                "parents": [],
                "sha": commit_oid,
                "tree": {"sha": self.tree},
            }
        if endpoint == f"repos/{REPOSITORY}/git/trees/{self.tree}?recursive=1":
            entries = [
                {"mode": "040000", "path": LEDGER_PATH_PREFIX, "type": "tree"},
                {
                    "mode": "040000",
                    "path": f"{LEDGER_PATH_PREFIX}/{self.state.suite_attempt_id}",
                    "type": "tree",
                },
                {
                    "mode": "100644",
                    "path": self.path,
                    "sha": self.blob,
                    "type": "blob",
                },
                {
                    "mode": "040000",
                    "path": github_attestation.LEDGER_CONTROL_PREFIX,
                    "type": "tree",
                },
                {
                    "mode": "040000",
                    "path": (
                        f"{github_attestation.LEDGER_CONTROL_PREFIX}/{self.state.suite_attempt_id}"
                    ),
                    "type": "tree",
                },
                {
                    "mode": "040000",
                    "path": (
                        f"{github_attestation.LEDGER_CONTROL_PREFIX}/"
                        f"{self.state.suite_attempt_id}/sealed-launch-contracts"
                    ),
                    "type": "tree",
                },
                {
                    "mode": "100644",
                    "path": self.control_inventory_path,
                    "sha": _blob_oid(self.control_inventory),
                    "type": "blob",
                },
                *(
                    {
                        "mode": "100644",
                        "path": control.ledger_path,
                        "sha": control.blob_oid,
                        "type": "blob",
                    }
                    for control in self.controls
                ),
            ]
            if self.extra_path is not None:
                entries.append(
                    {
                        "mode": "100644",
                        "path": self.extra_path,
                        "sha": self.blob,
                        "type": "blob",
                    }
                )
            return {"tree": entries, "truncated": False}
        if endpoint == f"repos/{REPOSITORY}/git/blobs/{self.blob}":
            return {
                "content": base64.b64encode(self.state_bytes).decode("ascii"),
                "encoding": "base64",
                "size": len(self.state_bytes),
            }
        blob_prefix = f"repos/{REPOSITORY}/git/blobs/"
        if endpoint.startswith(blob_prefix):
            oid = endpoint.removeprefix(blob_prefix)
            encoded = self.control_blobs[oid]
            return {
                "content": base64.b64encode(encoded).decode("ascii"),
                "encoding": "base64",
                "size": len(encoded),
            }
        raise AssertionError(f"unexpected GitHub endpoint: {endpoint}")


class _RefRaceApi(_Api):
    def __init__(self, state: SuiteStateRecord) -> None:
        super().__init__(state)
        self.ref_reads = 0

    def get(self, endpoint: str) -> object:
        branch = f"confirmatory-ledger/{self.state.suite_attempt_id}"
        if endpoint == f"repos/{REPOSITORY}/git/ref/heads/{branch}":
            self.ref_reads += 1
            if self.ref_reads == 2:
                return {
                    "object": {"sha": "f" * 40, "type": "commit"},
                    "ref": f"{LEDGER_REF_PREFIX}/{self.state.suite_attempt_id}",
                }
        return super().get(endpoint)


class _WritableApi:
    def __init__(self) -> None:
        self.ruleset: dict[str, object] | None = None
        self.blobs: dict[str, bytes] = {}
        self.trees: dict[str, list[dict[str, object]]] = {}
        self.commits: dict[str, dict[str, object]] = {}
        self.tip: str | None = None
        self.force_values: list[bool] = []

    @staticmethod
    def _oid(value: object) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha1(encoded, usedforsecurity=False).hexdigest()

    def get(self, endpoint: str) -> object:
        if endpoint == f"repos/{REPOSITORY}/rulesets?includes_parents=false&per_page=100":
            if self.ruleset is None:
                return []
            return [{"id": self.ruleset["id"], "name": self.ruleset["name"]}]
        if endpoint == f"repos/{REPOSITORY}/rulesets/17":
            assert self.ruleset is not None
            return self.ruleset
        matching = f"repos/{REPOSITORY}/git/matching-refs/heads/confirmatory-ledger/"
        if endpoint.startswith(matching):
            if self.tip is None:
                return []
            attempt_id = endpoint.removeprefix(matching)
            return [
                {
                    "object": {"sha": self.tip, "type": "commit"},
                    "ref": f"{LEDGER_REF_PREFIX}/{attempt_id}",
                }
            ]
        ref_prefix = f"repos/{REPOSITORY}/git/ref/heads/confirmatory-ledger/"
        if endpoint.startswith(ref_prefix):
            assert self.tip is not None
            attempt_id = endpoint.removeprefix(ref_prefix)
            return {
                "object": {"sha": self.tip, "type": "commit"},
                "ref": f"{LEDGER_REF_PREFIX}/{attempt_id}",
            }
        branch_prefix = f"repos/{REPOSITORY}/branches/confirmatory-ledger%2F"
        if endpoint.startswith(branch_prefix):
            assert self.tip is not None
            attempt_id = endpoint.removeprefix(branch_prefix)
            return {
                "commit": {"sha": self.tip},
                "name": f"confirmatory-ledger/{attempt_id}",
                "protected": True,
            }
        rules_prefix = f"repos/{REPOSITORY}/rules/branches/confirmatory-ledger%2F"
        if endpoint.startswith(rules_prefix):
            return [
                {"ruleset_id": 17, "type": rule_type} for rule_type in sorted(LEDGER_RULE_TYPES)
            ]
        commit_prefix = f"repos/{REPOSITORY}/git/commits/"
        if endpoint.startswith(commit_prefix):
            return self.commits[endpoint.removeprefix(commit_prefix)]
        tree_prefix = f"repos/{REPOSITORY}/git/trees/"
        if endpoint.startswith(tree_prefix):
            tree_oid = endpoint.removeprefix(tree_prefix).removesuffix("?recursive=1")
            blobs = self.trees[tree_oid]
            attempt_id = str(blobs[-1]["path"]).split("/")[1]
            return {
                "tree": [
                    {"mode": "040000", "path": LEDGER_PATH_PREFIX, "type": "tree"},
                    {
                        "mode": "040000",
                        "path": f"{LEDGER_PATH_PREFIX}/{attempt_id}",
                        "type": "tree",
                    },
                    *blobs,
                ],
                "truncated": False,
            }
        blob_prefix = f"repos/{REPOSITORY}/git/blobs/"
        if endpoint.startswith(blob_prefix):
            oid = endpoint.removeprefix(blob_prefix)
            encoded = self.blobs[oid]
            return {
                "content": base64.b64encode(encoded).decode("ascii"),
                "encoding": "base64",
                "size": len(encoded),
            }
        raise AssertionError(f"unexpected GitHub endpoint: {endpoint}")

    def post(self, endpoint: str, payload: dict[str, object]) -> object:
        if endpoint == f"repos/{REPOSITORY}/rulesets":
            self.ruleset = {"id": 17, **payload}
            return self.ruleset
        if endpoint == f"repos/{REPOSITORY}/git/blobs":
            assert payload["encoding"] == "base64"
            encoded = base64.b64decode(str(payload["content"]), validate=True)
            oid = _blob_oid(encoded)
            self.blobs[oid] = encoded
            return {"sha": oid}
        if endpoint == f"repos/{REPOSITORY}/git/trees":
            entries: list[dict[str, object]] = []
            base_tree = payload.get("base_tree")
            if base_tree is not None:
                entries.extend(self.trees[str(base_tree)])
            entries.extend(payload["tree"])  # type: ignore[arg-type]
            oid = self._oid(entries)
            self.trees[oid] = entries
            return {"sha": oid}
        if endpoint == f"repos/{REPOSITORY}/git/commits":
            oid = self._oid(payload)
            author = dict(payload["author"])  # type: ignore[arg-type]
            committer = dict(payload["committer"])  # type: ignore[arg-type]
            self.commits[oid] = {
                "author": author,
                "committer": committer,
                "message": payload["message"],
                "parents": [{"sha": value} for value in payload["parents"]],
                "sha": oid,
                "tree": {"sha": payload["tree"]},
            }
            return {"sha": oid}
        if endpoint == f"repos/{REPOSITORY}/git/refs":
            assert self.tip is None
            self.tip = str(payload["sha"])
            return {
                "object": {"sha": self.tip, "type": "commit"},
                "ref": payload["ref"],
            }
        raise AssertionError(f"unexpected GitHub POST endpoint: {endpoint}")

    def patch(self, endpoint: str, payload: dict[str, object]) -> object:
        assert endpoint.startswith(f"repos/{REPOSITORY}/git/refs/heads/confirmatory-ledger/")
        self.force_values.append(bool(payload["force"]))
        assert payload["force"] is False
        self.tip = str(payload["sha"])
        attempt_id = endpoint.rsplit("/", 1)[-1]
        return {
            "object": {"sha": self.tip, "type": "commit"},
            "ref": f"{LEDGER_REF_PREFIX}/{attempt_id}",
        }


class _AuthorityRaceWritableApi(_WritableApi):
    def __init__(self) -> None:
        super().__init__()
        self.ref_reads = 0
        self.advance_at: int | None = None

    def get(self, endpoint: str) -> object:
        exact_ref = f"repos/{REPOSITORY}/git/ref/heads/confirmatory-ledger/"
        if endpoint.startswith(exact_ref):
            self.ref_reads += 1
            if self.advance_at == self.ref_reads:
                attempt_id = endpoint.removeprefix(exact_ref)
                return {
                    "object": {"sha": "f" * 40, "type": "commit"},
                    "ref": f"{LEDGER_REF_PREFIX}/{attempt_id}",
                }
        return super().get(endpoint)


class _FreezeTagRulesetApi:
    def __init__(self) -> None:
        self.ruleset: dict[str, object] | None = None
        self.post_count = 0

    def get(self, endpoint: str) -> object:
        if endpoint == f"repos/{REPOSITORY}/rulesets?includes_parents=false&per_page=100":
            if self.ruleset is None:
                return []
            return [{"id": self.ruleset["id"], "name": self.ruleset["name"]}]
        if endpoint == f"repos/{REPOSITORY}/rulesets/23":
            assert self.ruleset is not None
            return self.ruleset
        raise AssertionError(f"unexpected GitHub endpoint: {endpoint}")

    def post(self, endpoint: str, payload: dict[str, object]) -> object:
        assert endpoint == f"repos/{REPOSITORY}/rulesets"
        self.post_count += 1
        self.ruleset = {"id": 23, **payload}
        return self.ruleset

    def patch(self, endpoint: str, payload: dict[str, object]) -> object:
        raise AssertionError((endpoint, payload))


class _ApiWithoutBypassVisibility(_Api):
    def get(self, endpoint: str) -> object:
        value = super().get(endpoint)
        if endpoint == f"repos/{REPOSITORY}/rulesets/17":
            assert isinstance(value, dict)
            value = dict(value)
            value.pop("bypass_actors")
        return value


def _bundle(snapshot: LedgerSnapshot, transition: LedgerTransition) -> bytes:
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicate": ledger_predicate(snapshot, transition),
        "predicateType": PREDICATE_TYPE,
        "subject": [
            {
                "digest": {"sha256": transition.state.record_sha256},
                "name": transition.state_path,
            }
        ],
    }
    value = {
        "dsseEnvelope": {
            "payload": base64.b64encode(
                json.dumps(statement, sort_keys=True, separators=(",", ":")).encode()
            ).decode("ascii"),
            "payloadType": "application/vnd.in-toto+json",
            "signatures": [{"sig": "signature"}],
        },
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "verificationMaterial": {
            "tlogEntries": [
                {
                    "canonicalizedBody": base64.b64encode(b"rekor-body").decode(),
                    "inclusionPromise": {
                        "signedEntryTimestamp": base64.b64encode(b"rekor-set").decode()
                    },
                    "integratedTime": "1784030520",
                    "logId": {"keyId": base64.b64encode(b"k" * 32).decode()},
                    "logIndex": "42",
                }
            ]
        },
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _c1_bundle(
    *,
    predicate: object,
    predicate_type: str,
    subject_name: str,
    subject_digest: str,
    integrated_time: int,
    log_index: int,
) -> bytes:
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicate": predicate,
        "predicateType": predicate_type,
        "subject": [{"digest": {"sha256": subject_digest}, "name": subject_name}],
    }
    value = {
        "dsseEnvelope": {
            "payload": base64.b64encode(
                json.dumps(statement, sort_keys=True, separators=(",", ":")).encode()
            ).decode("ascii"),
            "payloadType": "application/vnd.in-toto+json",
            "signatures": [{"sig": "signature"}],
        },
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "verificationMaterial": {
            "tlogEntries": [
                {
                    "canonicalizedBody": base64.b64encode(b"rekor-body").decode(),
                    "inclusionPromise": {
                        "signedEntryTimestamp": base64.b64encode(
                            f"rekor-set-{log_index}".encode()
                        ).decode()
                    },
                    "integratedTime": integrated_time,
                    "logId": {"keyId": base64.b64encode(b"k" * 32).decode()},
                    "logIndex": log_index,
                }
            ]
        },
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _evidence(
    descriptor: SuiteAttestationDescriptor,
    snapshot: LedgerSnapshot,
    bundle: bytes,
) -> SuiteAttestationEvidence:
    transition = snapshot.tip
    observation = parse_sigstore_bundle(bundle)
    return SuiteAttestationEvidence(
        suite_attempt_id=transition.state.suite_attempt_id,
        state_sequence=transition.state.sequence,
        state_name=transition.state.state,
        state_record_sha256=transition.state.record_sha256,
        descriptor_sha256=descriptor.descriptor_sha256,
        bundle_sha256=hashlib.sha256(bundle).hexdigest(),
        bundle_byte_count=len(bundle),
        signer_identity=descriptor.expected_signer_identity,
        oidc_issuer=descriptor.expected_oidc_issuer,
        repository=descriptor.expected_repository,
        workflow=descriptor.expected_workflow,
        git_ref=descriptor.expected_git_ref,
        signer_digest=descriptor.expected_signer_digest,
        github_hosted_runner=True,
        transparency_log_identity=REKOR_IDENTITY,
        transparency_entry_id=observation.entry_id,
        transparency_log_index=observation.log_index,
        integrated_at_utc=observation.integrated_at_utc,
        timestamp_authority_identity=REKOR_IDENTITY,
        timestamp_token_sha256=observation.timestamp_token_sha256,
        signed_at_utc=observation.integrated_at_utc,
        state_service_identity=STATE_SERVICE_IDENTITY,
        state_key=snapshot.state_key,
        transition_id=transition.commit_oid,
        previous_transition_id=None,
    )


class _VerifiedCommand:
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, **kwargs: object) -> bytes:
        self.calls += 1
        assert Path(kwargs["state_path"]).read_bytes()
        assert Path(kwargs["bundle_path"]).read_bytes()
        return b'[{"verificationResult":{}}]'


class _VerifiedC1Command:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def verify(
        self,
        *,
        subject_path: Path,
        bundle_path: Path,
        c1_commit: str,
        predicate_type: str,
    ) -> bytes:
        assert subject_path.is_file()
        assert bundle_path.is_file()
        assert c1_commit == "1" * 40 or len(c1_commit) == 40
        self.calls.append((subject_path.name, predicate_type))
        return b'[{"verificationResult":{}}]\n'


def test_loads_exact_protected_manifest_ledger(tmp_path: Path) -> None:
    _, _, state = _open_state(tmp_path)
    snapshot = load_ledger_snapshot(
        repository=REPOSITORY,
        suite_attempt_id=state.suite_attempt_id,
        api=_Api(state),
    )
    assert snapshot.tip.state == state
    assert snapshot.tip.previous_commit_oid is None
    assert snapshot.state_key == f"{LEDGER_REF_PREFIX}/{state.suite_attempt_id}"
    assert COMMON_CONTROL_LIMITATION["independent_organizational_custody"] is False


def test_rejects_self_consistent_control_uri_substitution(tmp_path: Path) -> None:
    _, _, state = _open_state(tmp_path)
    api = _Api(state)
    api.controls = tuple(
        replace(control, materialization_uri=(tmp_path / "substituted.json").as_uri())
        if control.role == "production-finalization-receipt"
        else control
        for control in api.controls
    )
    api.control_inventory = github_attestation._ledger_control_inventory(
        state.suite_attempt_id,
        api.controls,
    )
    api.control_blobs[_blob_oid(api.control_inventory)] = api.control_inventory
    with pytest.raises(SuiteAttemptError, match="differs from OPENED"):
        load_ledger_snapshot(
            repository=REPOSITORY,
            suite_attempt_id=state.suite_attempt_id,
            api=api,
        )


def test_rejects_self_consistent_descriptor_substitution(tmp_path: Path) -> None:
    _, _, state = _open_state(tmp_path)
    api = _Api(state)
    substituted = _descriptor(key_digest=(b"z" * 32).hex()).canonical_bytes() + b"\n"
    api.controls = tuple(
        replace(
            control,
            encoded=substituted,
            file_sha256=hashlib.sha256(substituted).hexdigest(),
            byte_count=len(substituted),
            blob_oid=_blob_oid(substituted),
        )
        if control.role == "attestation-descriptor"
        else control
        for control in api.controls
    )
    api.control_inventory = github_attestation._ledger_control_inventory(
        state.suite_attempt_id,
        api.controls,
    )
    api.control_blobs = {control.blob_oid: control.encoded for control in api.controls}
    api.control_blobs[_blob_oid(api.control_inventory)] = api.control_inventory
    with pytest.raises(SuiteAttemptError, match="descriptor bytes differ from OPENED"):
        load_ledger_snapshot(
            repository=REPOSITORY,
            suite_attempt_id=state.suite_attempt_id,
            api=api,
        )


def test_rejects_non_manifest_derived_genesis_identity(tmp_path: Path) -> None:
    _, _, state = _open_state(tmp_path)
    substituted = replace(state, manifest_sha256=_digest("another-manifest"))
    with pytest.raises(SuiteAttemptError, match="manifest-derived OPENED"):
        load_ledger_snapshot(
            repository=REPOSITORY,
            suite_attempt_id=substituted.suite_attempt_id,
            api=_Api(substituted),
        )


def test_rejects_ref_change_during_snapshot_reconstruction(tmp_path: Path) -> None:
    _, _, state = _open_state(tmp_path)
    with pytest.raises(SuiteAttemptError, match="changed during reconstruction"):
        load_ledger_snapshot(
            repository=REPOSITORY,
            suite_attempt_id=state.suite_attempt_id,
            api=_RefRaceApi(state),
        )


def test_idempotent_publication_rechecks_tip_before_receipt(tmp_path: Path) -> None:
    namespace, _, state = _open_state(tmp_path)
    (namespace / "000.state.json").write_bytes(state.canonical_bytes() + b"\n")
    api = _AuthorityRaceWritableApi()
    install_ledger_ruleset(api=api)
    receipt_path = namespace / "000.ledger-publication.json"
    publish_ledger_transition(namespace=namespace, receipt_path=receipt_path, api=api)
    receipt_path.unlink()
    api.ref_reads = 0
    api.advance_at = 3
    with pytest.raises(SuiteAttemptError, match="changed before authority use"):
        publish_ledger_transition(namespace=namespace, receipt_path=receipt_path, api=api)
    assert not receipt_path.exists()


def test_installs_ruleset_and_publishes_exact_genesis_idempotently(tmp_path: Path) -> None:
    namespace, _, state = _open_state(tmp_path)
    (namespace / "000.state.json").write_bytes(state.canonical_bytes() + b"\n")
    api = _WritableApi()
    assert install_ledger_ruleset(api=api) == 17
    receipt_path = namespace / "000.ledger-publication.json"
    receipt, created = publish_ledger_transition(
        namespace=namespace,
        receipt_path=receipt_path,
        api=api,
    )
    assert created is True
    assert receipt.state_record_sha256 == state.record_sha256
    assert receipt.previous_commit_oid is None
    assert receipt.blob_oid == _blob_oid(state.canonical_bytes() + b"\n")
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["ruleset_id"] == 17

    repeated, created_again = publish_ledger_transition(
        namespace=namespace,
        receipt_path=receipt_path,
        api=api,
    )
    assert created_again is False
    assert repeated == receipt


def _hosted_candidate_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_WritableApi, LedgerSnapshot, SuiteStateRecord]:
    namespace, _, opened = _open_state(tmp_path)
    (namespace / "000.state.json").write_bytes(opened.canonical_bytes() + b"\n")
    api = _WritableApi()
    install_ledger_ruleset(api=api)
    publish_ledger_transition(
        namespace=namespace,
        receipt_path=namespace / "000.ledger-publication.json",
        api=api,
    )
    predecessor = load_ledger_snapshot(
        repository=REPOSITORY,
        suite_attempt_id=opened.suite_attempt_id,
        api=api,
    )
    target = copy.copy(opened)
    object.__setattr__(target, "sequence", 1)
    object.__setattr__(target, "previous_state_record_sha256", opened.record_sha256)
    monkeypatch.setattr(github_attestation, "_assert_state_transition", lambda *_args: None)

    return api, predecessor, target


def test_hosted_candidate_uses_exact_byte_cas_and_read_only_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, predecessor, target = _hosted_candidate_fixture(tmp_path, monkeypatch)
    calls = 0

    def snapshot(**_kwargs: object) -> LedgerSnapshot:
        nonlocal calls
        calls += 1
        if calls == 1:
            return predecessor
        assert api.tip is not None
        commit = api.commits[api.tip]
        transition = LedgerTransition(
            commit_oid=api.tip,
            previous_commit_oid=predecessor.tip.commit_oid,
            tree_oid=str(commit["tree"]["sha"]),  # type: ignore[index]
            state_path=f"{LEDGER_PATH_PREFIX}/{target.suite_attempt_id}/001.state.json",
            state_bytes=target.canonical_bytes() + b"\n",
            state=target,
        )
        return LedgerSnapshot(
            repository=REPOSITORY,
            state_key=f"{LEDGER_REF_PREFIX}/{target.suite_attempt_id}",
            protection=predecessor.protection,
            transitions=(*predecessor.transitions, transition),
        )

    monkeypatch.setattr(github_attestation, "load_ledger_snapshot", snapshot)
    receipt_path = (tmp_path / "candidate-publication.json").resolve()
    receipt, published = publish_candidate_ledger_transition(
        target=target,
        expected_predecessor_commit=predecessor.tip.commit_oid,
        receipt_path=receipt_path,
        api=api,
    )
    assert published is True
    assert receipt.previous_commit_oid == predecessor.tip.commit_oid
    assert api.force_values == [False]
    counts = (len(api.blobs), len(api.trees), len(api.commits), len(api.force_values))

    calls = 2
    repeated, republished = publish_candidate_ledger_transition(
        target=target,
        expected_predecessor_commit=predecessor.tip.commit_oid,
        receipt_path=receipt_path,
        api=api,
    )
    assert republished is False
    assert repeated == receipt
    assert (len(api.blobs), len(api.trees), len(api.commits), len(api.force_values)) == counts


def test_hosted_candidate_rejects_stale_writer_before_creating_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, predecessor, target = _hosted_candidate_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        github_attestation,
        "load_ledger_snapshot",
        lambda **_kwargs: predecessor,
    )
    counts = (len(api.blobs), len(api.trees), len(api.commits), len(api.force_values))
    with pytest.raises(SuiteAttemptError, match="exact provider CAS predecessor"):
        publish_candidate_ledger_transition(
            target=target,
            expected_predecessor_commit="f" * 40,
            receipt_path=(tmp_path / "must-not-exist.json").resolve(),
            api=api,
        )
    assert (len(api.blobs), len(api.trees), len(api.commits), len(api.force_values)) == counts


def test_rejects_ruleset_with_a_bypass_actor(tmp_path: Path) -> None:
    namespace, _, state = _open_state(tmp_path)
    (namespace / "000.state.json").write_bytes(state.canonical_bytes() + b"\n")
    api = _WritableApi()
    install_ledger_ruleset(api=api)
    assert api.ruleset is not None
    api.ruleset["bypass_actors"] = [
        {"actor_id": 1, "actor_type": "RepositoryRole", "bypass_mode": "always"}
    ]
    with pytest.raises(SuiteAttemptError, match="bypass actor"):
        publish_ledger_transition(
            namespace=namespace,
            receipt_path=namespace / "must-not-exist.json",
            api=api,
        )
    assert api.tip is None


def test_installs_exact_no_bypass_freeze_tag_ruleset_idempotently() -> None:
    api = _FreezeTagRulesetApi()
    assert install_freeze_tag_ruleset(api=api) == 23
    assert api.post_count == 1
    assert api.ruleset == {
        "bypass_actors": [],
        "conditions": {
            "ref_name": {
                "exclude": [],
                "include": list(FREEZE_TAG_RULESET_INCLUDES),
            }
        },
        "enforcement": "active",
        "id": 23,
        "name": FREEZE_TAG_RULESET_NAME,
        "rules": [{"type": value} for value in sorted(FREEZE_TAG_RULE_TYPES)],
        "target": "tag",
    }
    assert install_freeze_tag_ruleset(api=api) == 23
    assert required_freeze_tag_ruleset(api) == 23
    assert api.post_count == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row.update(
                {
                    "bypass_actors": [
                        {
                            "actor_id": 5,
                            "actor_type": "RepositoryRole",
                            "bypass_mode": "always",
                        }
                    ]
                }
            ),
            "bypass actor",
        ),
        (
            lambda row: row.update(
                {
                    "conditions": {
                        "ref_name": {
                            "exclude": [],
                            "include": [*FREEZE_TAG_RULESET_INCLUDES, "refs/tags/other"],
                        }
                    }
                }
            ),
            "only the exact C0 and C1 refs",
        ),
        (lambda row: row.update({"target": "branch"}), "identity or enforcement"),
        (
            lambda row: row.update(
                {"rules": [*row["rules"], {"type": "creation"}]}  # type: ignore[misc]
            ),
            "exactly the deletion and non-fast-forward rules",
        ),
    ],
)
def test_rejects_weakened_or_overbroad_freeze_tag_ruleset(
    mutation: object,
    message: str,
) -> None:
    api = _FreezeTagRulesetApi()
    install_freeze_tag_ruleset(api=api)
    assert api.ruleset is not None
    mutation(api.ruleset)  # type: ignore[operator]
    with pytest.raises(SuiteAttemptError, match=message):
        required_freeze_tag_ruleset(api)


def test_read_only_workflow_can_check_ruleset_while_admin_verifier_requires_bypass_field(
    tmp_path: Path,
) -> None:
    _, _, state = _open_state(tmp_path)
    api = _ApiWithoutBypassVisibility(state)
    with pytest.raises(SuiteAttemptError, match="bypass actors are not visible"):
        load_ledger_snapshot(
            repository=REPOSITORY,
            suite_attempt_id=state.suite_attempt_id,
            api=api,
        )
    snapshot = load_ledger_snapshot(
        repository=REPOSITORY,
        suite_attempt_id=state.suite_attempt_id,
        api=api,
        require_ruleset_bypass_visibility=False,
    )
    assert snapshot.tip.state == state


@pytest.mark.parametrize(
    ("api_kwargs", "message"),
    [
        ({"force_pushes": True}, "prevent deletion and non-fast-forward"),
        (
            {"extra_path": "suite-attempts/" + "f" * 64 + "/000.state.json"},
            "mixes suite-attempt",
        ),
    ],
)
def test_rejects_ledger_rewrite_surfaces(
    tmp_path: Path,
    api_kwargs: dict[str, object],
    message: str,
) -> None:
    _, _, state = _open_state(tmp_path)
    with pytest.raises(SuiteAttemptError, match=message):
        load_ledger_snapshot(
            repository=REPOSITORY,
            suite_attempt_id=state.suite_attempt_id,
            api=_Api(state, **api_kwargs),
        )


def test_production_verifier_returns_only_live_provider_claims(tmp_path: Path) -> None:
    namespace, descriptor, state = _open_state(tmp_path)
    api = _Api(state)
    snapshot = load_ledger_snapshot(
        repository=REPOSITORY,
        suite_attempt_id=state.suite_attempt_id,
        api=api,
    )
    bundle = _bundle(snapshot, snapshot.tip)
    (namespace / "000.state.json").write_bytes(state.canonical_bytes() + b"\n")
    command = _VerifiedCommand()
    claims = GitHubSuiteEvidenceVerifier(
        namespace,
        api=api,
        attestation_verifier=command,
    ).verify(
        bundle=bundle,
        evidence=_evidence(descriptor, snapshot, bundle),
        descriptor=descriptor,
        state_record_bytes=state.canonical_bytes() + b"\n",
    )
    assert claims.subject_sha256 == state.record_sha256
    assert claims.transition_id == "a" * 40
    assert claims.exclusive_transition is True
    assert claims.transparency_verified is True
    assert command.calls == 1


def test_rejects_stale_local_chain_before_signature_check(tmp_path: Path) -> None:
    namespace, descriptor, state = _open_state(tmp_path)
    api = _Api(state)
    snapshot = load_ledger_snapshot(
        repository=REPOSITORY,
        suite_attempt_id=state.suite_attempt_id,
        api=api,
    )
    bundle = _bundle(snapshot, snapshot.tip)
    command = _VerifiedCommand()
    with pytest.raises(SuiteAttemptError, match="stale or ahead"):
        GitHubSuiteEvidenceVerifier(
            namespace,
            api=api,
            attestation_verifier=command,
        ).verify(
            bundle=bundle,
            evidence=_evidence(descriptor, snapshot, bundle),
            descriptor=descriptor,
            state_record_bytes=state.canonical_bytes() + b"\n",
        )
    assert command.calls == 0


def test_rejects_subject_or_predicate_substitution(tmp_path: Path) -> None:
    _, _, state = _open_state(tmp_path)
    snapshot = load_ledger_snapshot(
        repository=REPOSITORY,
        suite_attempt_id=state.suite_attempt_id,
        api=_Api(state),
    )
    bundle = json.loads(_bundle(snapshot, snapshot.tip))
    statement = json.loads(base64.b64decode(bundle["dsseEnvelope"]["payload"]))
    statement["subject"][0]["name"] = "caller-selected.bin"
    bundle["dsseEnvelope"]["payload"] = base64.b64encode(
        json.dumps(statement, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    from fractal_ann_diagnostics.github_state_attestation import _verify_statement

    with pytest.raises(SuiteAttemptError, match="subject name"):
        _verify_statement(
            parse_sigstore_bundle(json.dumps(bundle).encode()),
            snapshot=snapshot,
            transition=snapshot.tip,
        )


def test_rejects_more_than_one_transparency_entry(tmp_path: Path) -> None:
    _, _, state = _open_state(tmp_path)
    snapshot = load_ledger_snapshot(
        repository=REPOSITORY,
        suite_attempt_id=state.suite_attempt_id,
        api=_Api(state),
    )
    bundle = json.loads(_bundle(snapshot, snapshot.tip))
    entries = bundle["verificationMaterial"]["tlogEntries"]
    entries.append(entries[0])
    with pytest.raises(SuiteAttemptError, match="exactly one Rekor"):
        parse_sigstore_bundle(json.dumps(bundle).encode())


def test_workflow_validator_rejects_non_tip_and_writes_closed_inputs(tmp_path: Path) -> None:
    _, _, state = _open_state(tmp_path)
    api = _Api(state)
    with pytest.raises(SuiteAttemptError, match="not the protected ledger tip"):
        validate_workflow_transition(
            ledger_commit="c" * 40,
            repository=REPOSITORY,
            github_ref=C0_REF,
            github_sha="1" * 40,
            workflow_ref=f"{REPOSITORY}/{WORKFLOW_PATH}@{C0_REF}",
            workflow_sha="1" * 40,
            output_dir=tmp_path / "rejected",
            api=api,
        )
    outputs = validate_workflow_transition(
        ledger_commit=api.tip,
        repository=REPOSITORY,
        github_ref=C0_REF,
        github_sha="1" * 40,
        workflow_ref=f"{REPOSITORY}/{WORKFLOW_PATH}@{C0_REF}",
        workflow_sha="1" * 40,
        output_dir=tmp_path / "accepted",
        api=api,
    )
    assert outputs["state_name"].endswith("/000.state.json")
    assert Path(outputs["state_path"]).read_bytes() == state.canonical_bytes() + b"\n"


def test_emit_evidence_binds_bundle_and_ledger_receipt(tmp_path: Path) -> None:
    _, descriptor, state = _open_state(tmp_path)
    api = _Api(state)
    outputs = validate_workflow_transition(
        ledger_commit=api.tip,
        repository=REPOSITORY,
        github_ref=C0_REF,
        github_sha="1" * 40,
        workflow_ref=f"{REPOSITORY}/{WORKFLOW_PATH}@{C0_REF}",
        workflow_sha="1" * 40,
        output_dir=tmp_path / "validated",
        api=api,
    )
    snapshot = load_ledger_snapshot(
        repository=REPOSITORY,
        suite_attempt_id=state.suite_attempt_id,
        api=api,
    )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(_bundle(snapshot, snapshot.tip))
    output = tmp_path / "000.attestation.json"
    evidence = emit_attestation_evidence(
        bundle_path=bundle_path,
        receipt_path=Path(outputs["receipt_path"]),
        output_path=output,
    )
    assert evidence.descriptor_sha256 == descriptor.descriptor_sha256
    assert evidence.transition_id == api.tip
    assert output.read_bytes() == evidence.canonical_bytes() + b"\n"
    receipt = json.loads(Path(outputs["receipt_path"]).read_text())
    assert receipt["control_boundary"] == COMMON_CONTROL_LIMITATION


def test_gh_verifier_enforces_every_identity_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor()
    state_path = tmp_path / "state.json"
    bundle_path = tmp_path / "bundle.json"
    state_path.write_bytes(b"state")
    bundle_path.write_bytes(b"bundle")
    observed: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed.extend(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b'[{"verificationResult":{}}]',
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    GhAttestationVerifier().verify(
        state_path=state_path,
        bundle_path=bundle_path,
        descriptor=descriptor,
    )
    for flag in (
        "--bundle",
        "--hostname",
        "--repo",
        "--cert-identity",
        "--cert-oidc-issuer",
        "--signer-digest",
        "--source-digest",
        "--source-ref",
        "--deny-self-hosted-runners",
        "--predicate-type",
    ):
        assert flag in observed
    assert observed[observed.index("--hostname") + 1] == "github.com"
    assert PREDICATE_TYPE in observed


def test_gh_api_client_pins_public_github_host(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"{}", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert GhApiClient().get("/repos/mhdk1602/fractal-ann-diagnostics") == {}
    assert observed[observed.index("--hostname") + 1] == "github.com"


def test_c1_gh_verifier_pins_registry_workflow_ref_and_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = tmp_path / REGISTRY_RECORD_SUBJECT_PATH
    bundle = tmp_path / "registry.bundle.json"
    subject.write_bytes(b"record")
    bundle.write_bytes(b"bundle")
    observed: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed.extend(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b'[{"verificationResult":{}}]',
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    GhC1AttestationVerifier().verify(
        subject_path=subject,
        bundle_path=bundle,
        c1_commit="1" * 40,
        predicate_type=REGISTRY_RECORD_PREDICATE_TYPE,
    )
    identity = (
        "https://github.com/mhdk1602/fractal-ann-diagnostics/.github/workflows/"
        "confirmatory-registration-attestation.yml@refs/tags/confirmatory-freeze-c1"
    )
    assert observed[observed.index("--cert-identity") + 1] == identity
    assert observed[observed.index("--hostname") + 1] == "github.com"
    assert observed[observed.index("--source-ref") + 1] == C1_REF
    assert observed[observed.index("--signer-digest") + 1] == "1" * 40
    assert observed[observed.index("--source-digest") + 1] == "1" * 40
    assert "--deny-self-hosted-runners" in observed
    assert observed[observed.index("--predicate-type") + 1] == REGISTRY_RECORD_PREDICATE_TYPE


def test_wrong_c0_ref_is_not_a_supported_policy() -> None:
    descriptor = replace(_descriptor(), expected_git_ref="refs/heads/master")
    from fractal_ann_diagnostics.github_state_attestation import _assert_descriptor

    with pytest.raises(SuiteAttemptError, match="expected_git_ref"):
        _assert_descriptor(descriptor)


def test_workflow_has_no_repository_write_permission() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github" / "workflows" / "confirmatory-state-attestation.yml"
    ).read_text(encoding="utf-8")
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "artifact-metadata: write" not in workflow
    assert "subject-digest: sha256:${{ steps.validate.outputs.state_digest }}" in workflow
    assert "predicate-path: ${{ steps.validate.outputs.predicate_path }}" in workflow
    assert "create-storage-record: false" in workflow
    assert "runs-on: ubuntu-24.04" in workflow
    assert "test \"$GITHUB_REPOSITORY\" = 'mhdk1602/fractal-ann-diagnostics'" in workflow
    assert "test \"$GITHUB_ACTOR\" = 'mhdk1602'" in workflow
    assert "test \"$GITHUB_TRIGGERING_ACTOR\" = 'mhdk1602'" in workflow
    assert "test \"$GITHUB_EVENT_NAME\" = 'workflow_dispatch'" in workflow
    assert "test \"$GITHUB_REF\" = 'refs/tags/confirmatory-apparatus-c0'" in workflow
    assert 'test "$GITHUB_SHA" = "$GITHUB_WORKFLOW_SHA"' in workflow


def test_c1_registration_predicate_closes_subject_and_control_boundary() -> None:
    predicate = c1_registration_predicate(
        c1_commit="1" * 40,
        c0_commit="0" * 40,
        tag_object_id="2" * 40,
        tag_object_type="tag",
        manifest_digest="3" * 64,
        manifest_file_digest="4" * 64,
        lock_file_digest="5" * 64,
        transition_receipt_file_digest="6" * 64,
        c0_public_verification_file_digest="b" * 64,
        c0_public_verification_binding_digest="c" * 64,
        candidate_manifest_digest="7" * 64,
        candidate_manifest_file_digest="8" * 64,
        candidate_assembly_receipt_file_digest="9" * 64,
        reservation_file_digest="a" * 64,
    )
    assert predicate["control_boundary"] == COMMON_CONTROL_LIMITATION
    assert predicate["control_boundary"]["independent_organizational_custody"] is False
    assert predicate["freeze"] == {
        "c0_commit": "0" * 40,
        "c0_ref": C0_REF,
        "c1_commit": "1" * 40,
        "c1_ref": C1_REF,
        "tag_object_id": "2" * 40,
        "tag_object_type": "tag",
    }
    assert predicate["manifest"]["path"] == C1_MANIFEST_PATH
    assert predicate["lock"]["path"] == C1_LOCK_PATH
    assert predicate["c0_public_verification"] == {
        "binding_sha256": "c" * 64,
        "file_sha256": "b" * 64,
        "path": C0_PUBLIC_VERIFICATION_PATH,
        "release_tag": "confirmatory-apparatus-c0",
        "schema_version": C0_PUBLIC_VERIFICATION_SCHEMA,
        "target_commit": "0" * 40,
    }
    assert predicate["manifest_transition"] == {
        "candidate_manifest_assembly_receipt_file_sha256": "9" * 64,
        "candidate_manifest_file_sha256": "8" * 64,
        "candidate_manifest_sha256": "7" * 64,
        "file_sha256": "6" * 64,
        "path": C1_TRANSITION_RECEIPT_PATH,
        "schema_version": "fractal-c1-manifest-transition-receipt-v2",
    }
    assert predicate["registry_reservation"] == {
        "deposition_id": ZENODO_RECORD_ID,
        "direct_registry_record_uri": ZENODO_REGISTRY_URI,
        "file_sha256": "a" * 64,
        "path": C1_RESERVATION_PATH,
        "registry_identity": ZENODO_REGISTRY_IDENTITY,
        "reserved_doi": ZENODO_RESERVED_DOI,
    }


@pytest.mark.parametrize(
    ("repository", "github_ref"),
    [
        ("attacker/repository", C1_REF),
        (REPOSITORY, "refs/tags/caller-selected"),
    ],
)
def test_c1_registration_rejects_another_repository_or_ref_before_git_access(
    tmp_path: Path,
    repository: str,
    github_ref: str,
) -> None:
    with pytest.raises(SuiteAttemptError, match="fixed repository at C1"):
        prepare_c1_registration(
            repository=repository,
            github_ref=github_ref,
            github_sha="1" * 40,
            workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
            workflow_sha="1" * 40,
            repository_root=tmp_path,
            c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
            output_dir=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_registration_workflow_has_zero_inputs_and_a_fixed_c1_subject() -> None:
    workflow = (
        Path(__file__).parents[1]
        / ".github"
        / "workflows"
        / "confirmatory-registration-attestation.yml"
    ).read_text(encoding="utf-8")
    dispatch_block = workflow.split("permissions:", maxsplit=1)[0]
    assert "workflow_dispatch:" in dispatch_block
    assert "inputs:" not in dispatch_block
    assert "github.event.inputs" not in workflow
    assert "${{ inputs." not in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "artifact-metadata: write" not in workflow
    assert "runs-on: ubuntu-24.04" in workflow
    assert "subject-path: research/study-manifest.json" in workflow
    assert "subject-path: protocol-registry-record.json" in workflow
    assert "subject-digest:" not in workflow
    assert "subject-name:" not in workflow
    assert workflow.count("create-storage-record: false") == 2
    assert C1_REF in workflow
    assert "test \"$GITHUB_REPOSITORY\" = 'mhdk1602/fractal-ann-diagnostics'" in workflow
    assert "test \"$GITHUB_ACTOR\" = 'mhdk1602'" in workflow
    assert "test \"$GITHUB_TRIGGERING_ACTOR\" = 'mhdk1602'" in workflow
    assert "test \"$GITHUB_EVENT_NAME\" = 'workflow_dispatch'" in workflow
    assert "test \"$GITHUB_REF\" = 'refs/tags/confirmatory-freeze-c1'" in workflow
    assert 'test "$GITHUB_SHA" = "$GITHUB_WORKFLOW_SHA"' in workflow
    assert REGISTRATION_PREDICATE_TYPE in workflow
    assert REGISTRY_RECORD_PREDICATE_TYPE in workflow
    assert '--repository-root "$GITHUB_WORKSPACE"' in workflow
    assert '--output-dir "${RUNNER_TEMP}/validated-c1-registration"' in workflow
    assert "materialize-c1-registry-record" in workflow
    assert "verify-c1-registry-record" in workflow
    assert "protocol-registry-record.sigstore.bundle.json" in workflow
    assert "registry-attestation-validation.json" in workflow
    assert "research/manifest-transition-receipt.json" in workflow
    assert '"$package/manifest-transition-receipt.json"' in workflow
    assert workflow.count("actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6") == 2
    for forbidden in ("--subject-path", "--subject-digest", "--predicate-path"):
        assert forbidden not in workflow
    for action in (
        "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
        "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990",
        "actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
    ):
        assert action in workflow


def _make_c1_repository(
    tmp_path: Path,
    *,
    extra_path: bool = False,
    forge_transition: bool = False,
    forge_transition_field: str | None = None,
    omit_transition: bool = False,
    registry_uri: str = ZENODO_REGISTRY_URI,
) -> tuple[Path, str]:
    root = tmp_path / "repository"
    root.mkdir()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "-b", "main")
    git("config", "user.name", "mhdk1602")
    git("config", "user.email", "mhdk1602@users.noreply.github.com")
    research = root / "research"
    research.mkdir()
    (research / "study-manifest.json").write_text('{"state":"draft"}\n', encoding="utf-8")
    reservation = {
        "created_at_utc": ZENODO_RESERVATION_CREATED_AT_UTC,
        "creator": "mhdk1602",
        "deposition_id": ZENODO_RECORD_ID,
        "direct_registry_record_uri": registry_uri,
        "draft_uri": ZENODO_DRAFT_URI,
        "protocol_version": "0.3.0",
        "reserved_doi": ZENODO_RESERVED_DOI,
        "schema_version": "fractal-zenodo-reservation-v1",
        "state": "unsubmitted",
        "submitted": False,
    }
    (research / "zenodo-reservation.json").write_text(
        json.dumps(reservation, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    git("add", "research/study-manifest.json", "research/zenodo-reservation.json")
    git("commit", "-m", "C0 apparatus")
    git("tag", "confirmatory-apparatus-c0")
    c0_commit = git("rev-parse", "HEAD")

    semantic_digest = _digest("frozen-manifest")
    frozen_bytes = b'{"state":"frozen"}\n'
    (research / "study-manifest.json").write_bytes(frozen_bytes)
    (research / "study-manifest.sha256").write_text(f"{semantic_digest}\n", encoding="ascii")
    _write_public_verification(
        tmp_path / C0_PUBLIC_VERIFICATION_PATH,
        _c0_public_verification_receipt(
            c0_commit=c0_commit,
            frozen_manifest_bytes=frozen_bytes,
        ),
    )
    if not omit_transition:
        candidate_package = (tmp_path / "candidate-package").resolve()
        transition = C1ManifestTransitionReceipt(
            c0_commit=c0_commit,
            candidate_manifest_package_uri=candidate_package.as_uri(),
            candidate_manifest_uri=(candidate_package / "candidate-study-manifest.json").as_uri(),
            candidate_manifest_sha256=_digest("candidate-manifest"),
            candidate_manifest_file_sha256=_digest("candidate-manifest-file"),
            candidate_manifest_assembly_receipt_uri=(
                candidate_package / "candidate-manifest-assembly-receipt.json"
            ).as_uri(),
            candidate_manifest_assembly_receipt_file_sha256=_digest("assembly-receipt"),
            candidate_manifest_assembly_receipt_schema=("fractal-candidate-manifest-assembly-v1"),
            c0_evidence_release_uri=(tmp_path / "c0-evidence-release.json").resolve().as_uri(),
            c0_evidence_release_sha256=_digest("c0-evidence"),
            c0_evidence_release_file_sha256=_digest("c0-evidence-file"),
            apparatus_evidence_sha256=_digest("apparatus-evidence"),
            provider_phase_plan_closure_sha256=_digest("provider-plans"),
            frozen_manifest_uri=(research / "study-manifest.json").resolve().as_uri(),
            frozen_manifest_sha256=semantic_digest,
            frozen_manifest_file_sha256=hashlib.sha256(frozen_bytes).hexdigest(),
            frozen_manifest_byte_count=len(frozen_bytes),
            frozen_manifest_mode="0600",
        )
        if forge_transition_field is not None:
            transition = replace(transition, **{forge_transition_field: "0" * 64})
        transition_bytes = transition.canonical_file_bytes()
        if forge_transition:
            transition_bytes = b'{"forged":true}\n'
        (root / C1_TRANSITION_RECEIPT_PATH).write_bytes(transition_bytes)
    if extra_path:
        (root / "caller-selected.txt").write_text("inadmissible\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "C1 frozen manifest")
    git("tag", "confirmatory-freeze-c1")
    return root, git("rev-parse", "HEAD")


def _accept_synthetic_transition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        github_attestation,
        "verify_c1_manifest_transition_receipt_bindings",
        lambda *_args, **_kwargs: None,
    )


def _synthetic_frozen_manifest(tmp_path: Path) -> dict[str, object]:
    receipt = json.loads((tmp_path / C0_PUBLIC_VERIFICATION_PATH).read_text(encoding="ascii"))
    return {
        "sealed_execution": {
            "c0_evidence_release": receipt["c0_evidence_release_binding"],
        },
        "state": "frozen",
    }


def _accept_synthetic_manifest(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest: dict[str, object],
) -> None:
    monkeypatch.setattr(
        github_attestation,
        "load_study_manifest",
        lambda _path: copy.deepcopy(manifest),
    )
    monkeypatch.setattr(
        github_attestation,
        "validate_study_manifest",
        lambda _manifest, *, require_frozen: None,
    )
    monkeypatch.setattr(
        github_attestation,
        "manifest_sha256",
        lambda _manifest: _digest("frozen-manifest"),
    )
    _accept_synthetic_transition(monkeypatch)


def _prepare_synthetic_c1(
    *,
    root: Path,
    c1_commit: str,
    tmp_path: Path,
    output_name: str,
) -> Mapping[str, str]:
    return prepare_c1_registration(
        repository=REPOSITORY,
        github_ref=C1_REF,
        github_sha=c1_commit,
        workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
        workflow_sha=c1_commit,
        repository_root=root,
        c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
        output_dir=tmp_path / output_name,
    )


def test_prepare_c1_registration_accepts_only_the_closed_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, c1_commit = _make_c1_repository(tmp_path)
    semantic_digest = _digest("frozen-manifest")
    monkeypatch.setattr(
        github_attestation,
        "load_study_manifest",
        lambda _path: _synthetic_frozen_manifest(tmp_path),
    )
    monkeypatch.setattr(
        github_attestation,
        "validate_study_manifest",
        lambda _manifest, *, require_frozen: None,
    )
    monkeypatch.setattr(
        github_attestation,
        "manifest_sha256",
        lambda _manifest: semantic_digest,
    )
    _accept_synthetic_transition(monkeypatch)
    outputs = prepare_c1_registration(
        repository=REPOSITORY,
        github_ref=C1_REF,
        github_sha=c1_commit,
        workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
        workflow_sha=c1_commit,
        repository_root=root,
        c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
        output_dir=tmp_path / "registration",
    )
    assert outputs["c1_commit"] == c1_commit
    assert outputs["manifest_digest"] == semantic_digest
    predicate = json.loads(Path(outputs["predicate_path"]).read_text(encoding="utf-8"))
    receipt = json.loads(Path(outputs["receipt_path"]).read_text(encoding="utf-8"))
    assert predicate["control_boundary"]["independent_organizational_custody"] is False
    assert predicate["c0_public_verification"] == {
        "binding_sha256": receipt["predicate"]["c0_public_verification"]["binding_sha256"],
        "file_sha256": outputs["c0_public_verification_file_digest"],
        "path": C0_PUBLIC_VERIFICATION_PATH,
        "release_tag": "confirmatory-apparatus-c0",
        "schema_version": C0_PUBLIC_VERIFICATION_SCHEMA,
        "target_commit": outputs["c0_commit"],
    }
    assert receipt["predicate"] == predicate


def test_prepare_c1_registration_requires_the_public_c0_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, c1_commit = _make_c1_repository(tmp_path)
    manifest = _synthetic_frozen_manifest(tmp_path)
    (tmp_path / C0_PUBLIC_VERIFICATION_PATH).unlink()
    _accept_synthetic_manifest(monkeypatch, manifest=manifest)

    with pytest.raises(SuiteAttemptError, match="receipt is invalid:.*missing"):
        _prepare_synthetic_c1(
            root=root,
            c1_commit=c1_commit,
            tmp_path=tmp_path,
            output_name="missing-receipt",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("forged", "receipt is invalid"),
        ("source-substitution", "must use the frozen manifest"),
        ("binding", "binding differs from the frozen manifest"),
        ("digest", "source digest differs from the frozen manifest file"),
        ("c0", "target commit differs from C0"),
        ("schema", "receipt is invalid:.*schema differs"),
    ],
)
def test_prepare_c1_registration_rejects_hostile_public_c0_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    root, c1_commit = _make_c1_repository(tmp_path)
    receipt_path = tmp_path / C0_PUBLIC_VERIFICATION_PATH
    manifest = _synthetic_frozen_manifest(tmp_path)
    receipt = C0PublicVerificationReceipt.from_dict(
        json.loads(receipt_path.read_text(encoding="ascii"))
    )
    c0_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"{C0_REF}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    frozen_manifest_bytes = (root / C1_MANIFEST_PATH).read_bytes()

    if mutation == "forged":
        receipt_path.write_bytes(b'{"forged":true}\n')
        receipt_path.chmod(0o600)
    elif mutation == "source-substitution":
        _write_public_verification(
            receipt_path,
            replace(
                receipt,
                binding_source_kind="c0-binding",
                binding_source_file_sha256=receipt.binding_sha256,
            ),
        )
    elif mutation == "binding":
        binding = copy.deepcopy(dict(receipt.c0_evidence_release_binding))
        apparatus = dict(binding["apparatus_evidence"])
        apparatus["candidate_image_run_id"] = int(apparatus["candidate_image_run_id"]) + 1
        binding["apparatus_evidence"] = apparatus
        binding["apparatus_evidence_sha256"] = hashlib.sha256(
            canonical_apparatus_evidence_bytes(apparatus)
        ).hexdigest()
        _write_public_verification(
            receipt_path,
            _c0_public_verification_receipt(
                c0_commit=c0_commit,
                frozen_manifest_bytes=frozen_manifest_bytes,
                binding=binding,
            ),
        )
    elif mutation == "digest":
        _write_public_verification(
            receipt_path,
            replace(receipt, binding_source_file_sha256="0" * 64),
        )
    elif mutation == "c0":
        _write_public_verification(
            receipt_path,
            _c0_public_verification_receipt(
                c0_commit="2" * 40,
                frozen_manifest_bytes=frozen_manifest_bytes,
            ),
        )
    else:
        row = receipt.to_dict()
        row["schema_version"] = "fractal-c0-public-verification-v0"
        receipt_path.write_bytes(_canonical_file(row))
        receipt_path.chmod(0o600)

    _accept_synthetic_manifest(monkeypatch, manifest=manifest)
    with pytest.raises(SuiteAttemptError, match=message):
        _prepare_synthetic_c1(
            root=root,
            c1_commit=c1_commit,
            tmp_path=tmp_path,
            output_name=f"hostile-{mutation}",
        )


def test_prepare_c1_registration_rejects_public_c0_receipt_tamper_after_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, c1_commit = _make_c1_repository(tmp_path)
    manifest = _synthetic_frozen_manifest(tmp_path)
    stable_loader = github_attestation.load_c0_public_verification_receipt

    def load_then_tamper(path: Path) -> C0PublicVerificationReceipt:
        receipt = stable_loader(path)
        path.write_bytes(receipt.canonical_file_bytes() + b" ")
        path.chmod(0o600)
        return receipt

    monkeypatch.setattr(
        github_attestation,
        "load_c0_public_verification_receipt",
        load_then_tamper,
    )
    _accept_synthetic_manifest(monkeypatch, manifest=manifest)

    with pytest.raises(SuiteAttemptError, match="changed during C1 admission"):
        _prepare_synthetic_c1(
            root=root,
            c1_commit=c1_commit,
            tmp_path=tmp_path,
            output_name="tampered-receipt",
        )


def test_prepare_c1_registration_rejects_an_extra_changed_path(tmp_path: Path) -> None:
    root, c1_commit = _make_c1_repository(tmp_path, extra_path=True)
    with pytest.raises(SuiteAttemptError, match="must change only"):
        prepare_c1_registration(
            repository=REPOSITORY,
            github_ref=C1_REF,
            github_sha=c1_commit,
            workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
            workflow_sha=c1_commit,
            repository_root=root,
            c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
            output_dir=tmp_path / "rejected-registration",
        )
    assert not (tmp_path / "rejected-registration").exists()


def test_prepare_c1_registration_rejects_missing_transition_receipt(tmp_path: Path) -> None:
    root, c1_commit = _make_c1_repository(tmp_path, omit_transition=True)
    with pytest.raises(SuiteAttemptError, match="must change only"):
        prepare_c1_registration(
            repository=REPOSITORY,
            github_ref=C1_REF,
            github_sha=c1_commit,
            workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
            workflow_sha=c1_commit,
            repository_root=root,
            c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
            output_dir=tmp_path / "must-not-exist",
        )


def test_prepare_c1_registration_rejects_forged_transition_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, c1_commit = _make_c1_repository(tmp_path, forge_transition=True)
    monkeypatch.setattr(
        github_attestation,
        "load_study_manifest",
        lambda _path: _synthetic_frozen_manifest(tmp_path),
    )
    monkeypatch.setattr(
        github_attestation,
        "validate_study_manifest",
        lambda _manifest, *, require_frozen: None,
    )
    monkeypatch.setattr(
        github_attestation,
        "manifest_sha256",
        lambda _manifest: _digest("frozen-manifest"),
    )
    with pytest.raises(SuiteAttemptError, match="transition receipt is invalid"):
        prepare_c1_registration(
            repository=REPOSITORY,
            github_ref=C1_REF,
            github_sha=c1_commit,
            workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
            workflow_sha=c1_commit,
            repository_root=root,
            c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
            output_dir=tmp_path / "must-not-exist",
        )


@pytest.mark.parametrize(
    "field",
    (
        "c0_evidence_release_file_sha256",
        "candidate_manifest_file_sha256",
        "candidate_manifest_assembly_receipt_file_sha256",
    ),
)
def test_prepare_c1_registration_rejects_each_ungrounded_transition_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    root, c1_commit = _make_c1_repository(tmp_path, forge_transition_field=field)
    monkeypatch.setattr(
        github_attestation,
        "load_study_manifest",
        lambda _path: _synthetic_frozen_manifest(tmp_path),
    )
    monkeypatch.setattr(
        github_attestation,
        "validate_study_manifest",
        lambda _manifest, *, require_frozen: None,
    )
    monkeypatch.setattr(
        github_attestation,
        "manifest_sha256",
        lambda _manifest: _digest("frozen-manifest"),
    )

    def verify_grounding(receipt: C1ManifestTransitionReceipt, **_arguments: object) -> None:
        expected = {
            "c0_evidence_release_file_sha256": _digest("c0-evidence-file"),
            "candidate_manifest_file_sha256": _digest("candidate-manifest-file"),
            "candidate_manifest_assembly_receipt_file_sha256": _digest("assembly-receipt"),
        }
        if any(getattr(receipt, name) != value for name, value in expected.items()):
            raise C1ManifestTransitionError("fixture transition grounding differs")

    monkeypatch.setattr(
        github_attestation,
        "verify_c1_manifest_transition_receipt_bindings",
        verify_grounding,
    )
    with pytest.raises(SuiteAttemptError, match="transition receipt is invalid"):
        prepare_c1_registration(
            repository=REPOSITORY,
            github_ref=C1_REF,
            github_sha=c1_commit,
            workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
            workflow_sha=c1_commit,
            repository_root=root,
            c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
            output_dir=tmp_path / f"must-not-exist-{field}",
        )


@pytest.mark.parametrize("mutation", ("tracked", "untracked"))
def test_prepare_c1_registration_rejects_worktree_bytes_outside_the_c1_tree(
    tmp_path: Path,
    mutation: str,
) -> None:
    root, c1_commit = _make_c1_repository(tmp_path)
    if mutation == "tracked":
        (root / C1_MANIFEST_PATH).write_text('{"state":"substituted"}\n', encoding="utf-8")
    else:
        (root / "untracked-shadow.py").write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(SuiteAttemptError, match="worktree changes"):
        prepare_c1_registration(
            repository=REPOSITORY,
            github_ref=C1_REF,
            github_sha=c1_commit,
            workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
            workflow_sha=c1_commit,
            repository_root=root,
            c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
            output_dir=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("mutation", ("identity", "coauthor"))
def test_prepare_c1_registration_requires_the_sole_mhdk1602_commit_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    root, _ = _make_c1_repository(tmp_path)

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    if mutation == "identity":
        git("config", "user.name", "Another Author")
        git("config", "user.email", "another@example.invalid")
        git("commit", "--amend", "--no-edit", "--reset-author")
        message = "fixed mhdk1602 identity"
    else:
        git(
            "commit",
            "--amend",
            "-m",
            "C1 frozen manifest\n\nCo-authored-by: mhdk1602 <mhdk1602@users.noreply.github.com>",
        )
        message = "co-author trailer"
    git("tag", "-f", "confirmatory-freeze-c1")
    c1_commit = git("rev-parse", "HEAD")
    with pytest.raises(SuiteAttemptError, match=message):
        prepare_c1_registration(
            repository=REPOSITORY,
            github_ref=C1_REF,
            github_sha=c1_commit,
            workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
            workflow_sha=c1_commit,
            repository_root=root,
            c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
            output_dir=tmp_path / "must-not-exist",
        )


def test_prepare_c1_registration_rejects_an_annotated_tag_from_another_identity(
    tmp_path: Path,
) -> None:
    root, c1_commit = _make_c1_repository(tmp_path)

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )

    git("tag", "-d", "confirmatory-freeze-c1")
    git("config", "user.name", "Another Tagger")
    git("config", "user.email", "another@example.invalid")
    git("tag", "-a", "confirmatory-freeze-c1", "-m", "C1 freeze")
    with pytest.raises(SuiteAttemptError, match="tagger identity"):
        prepare_c1_registration(
            repository=REPOSITORY,
            github_ref=C1_REF,
            github_sha=c1_commit,
            workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
            workflow_sha=c1_commit,
            repository_root=root,
            c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
            output_dir=tmp_path / "must-not-exist",
        )


def test_prepare_c1_registration_rejects_another_zenodo_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, c1_commit = _make_c1_repository(
        tmp_path,
        registry_uri=(
            "https://zenodo.org.attacker.invalid/api/records/21361837/files/"
            "protocol-registry-record.json/content"
        ),
    )
    semantic_digest = _digest("frozen-manifest")
    monkeypatch.setattr(
        github_attestation,
        "load_study_manifest",
        lambda _path: _synthetic_frozen_manifest(tmp_path),
    )
    monkeypatch.setattr(
        github_attestation,
        "validate_study_manifest",
        lambda _manifest, *, require_frozen: None,
    )
    monkeypatch.setattr(
        github_attestation,
        "manifest_sha256",
        lambda _manifest: semantic_digest,
    )
    _accept_synthetic_transition(monkeypatch)
    with pytest.raises(SuiteAttemptError, match="Zenodo reservation differs"):
        prepare_c1_registration(
            repository=REPOSITORY,
            github_ref=C1_REF,
            github_sha=c1_commit,
            workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
            workflow_sha=c1_commit,
            repository_root=root,
            c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
            output_dir=tmp_path / "rejected-host",
        )


def test_registry_record_uses_verified_manifest_rekor_time_and_has_its_own_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, c1_commit = _make_c1_repository(tmp_path)
    semantic_digest = _digest("frozen-manifest")
    monkeypatch.setattr(
        github_attestation,
        "load_study_manifest",
        lambda _path: _synthetic_frozen_manifest(tmp_path),
    )
    monkeypatch.setattr(
        github_attestation,
        "validate_study_manifest",
        lambda _manifest, *, require_frozen: None,
    )
    monkeypatch.setattr(
        github_attestation,
        "manifest_sha256",
        lambda _manifest: semantic_digest,
    )
    _accept_synthetic_transition(monkeypatch)
    working = tmp_path / "registration"
    prepared = prepare_c1_registration(
        repository=REPOSITORY,
        github_ref=C1_REF,
        github_sha=c1_commit,
        workflow_ref=f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
        workflow_sha=c1_commit,
        repository_root=root,
        c0_public_verification_path=tmp_path / C0_PUBLIC_VERIFICATION_PATH,
        output_dir=working,
    )
    manifest_predicate = json.loads(Path(prepared["predicate_path"]).read_text())
    manifest_bundle = working / "manifest.sigstore.bundle.json"
    manifest_bundle.write_bytes(
        _c1_bundle(
            predicate=manifest_predicate,
            predicate_type=REGISTRATION_PREDICATE_TYPE,
            subject_name=C1_MANIFEST_PATH,
            subject_digest=prepared["manifest_file_digest"],
            integrated_time=1_784_030_520,
            log_index=41,
        )
    )
    record_path = root / REGISTRY_RECORD_SUBJECT_PATH
    registry_predicate_path = working / "registry-record-predicate.json"
    verifier = _VerifiedC1Command()
    materialized = materialize_protocol_registry_record(
        manifest_path=root / C1_MANIFEST_PATH,
        lock_path=root / C1_LOCK_PATH,
        reservation_path=root / C1_RESERVATION_PATH,
        manifest_bundle_path=manifest_bundle,
        manifest_predicate_path=Path(prepared["predicate_path"]),
        record_output_path=record_path,
        registry_predicate_output_path=registry_predicate_path,
        receipt_output_path=working / "registry-materialization.json",
        verification_output_path=working / "manifest-gh-verification.json",
        verifier=verifier,
    )
    first_observation = parse_sigstore_bundle(manifest_bundle.read_bytes())
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["registered_at_utc"] == first_observation.integrated_at_utc
    assert record["registry_identity"] == ZENODO_REGISTRY_IDENTITY
    assert record["registry_uri"] == ZENODO_REGISTRY_URI
    assert (
        materialized["registry_record_digest"]
        == hashlib.sha256(record_path.read_bytes()).hexdigest()
    )

    registry_predicate = json.loads(registry_predicate_path.read_text(encoding="utf-8"))
    assert registry_predicate["control_boundary"] == COMMON_CONTROL_LIMITATION
    assert registry_predicate["manifest_attestation"]["rekor_entry_id"] == (
        first_observation.entry_id
    )
    registry_bundle = working / "registry.sigstore.bundle.json"
    registry_bundle.write_bytes(
        _c1_bundle(
            predicate=registry_predicate,
            predicate_type=REGISTRY_RECORD_PREDICATE_TYPE,
            subject_name=REGISTRY_RECORD_SUBJECT_PATH,
            subject_digest=materialized["registry_record_digest"],
            integrated_time=1_784_030_521,
            log_index=42,
        )
    )
    verified = verify_c1_registry_record_attestation(
        record_path=record_path,
        predicate_path=registry_predicate_path,
        bundle_path=registry_bundle,
        receipt_output_path=working / "registry-attestation-validation.json",
        verification_output_path=working / "registry-gh-verification.json",
        verifier=verifier,
    )
    assert verified["registry_record_digest"] == materialized["registry_record_digest"]
    assert verifier.calls == [
        ("study-manifest.json", REGISTRATION_PREDICATE_TYPE),
        (REGISTRY_RECORD_SUBJECT_PATH, REGISTRY_RECORD_PREDICATE_TYPE),
    ]
    final_receipt = json.loads(
        (working / "registry-attestation-validation.json").read_text(encoding="utf-8")
    )
    assert (
        final_receipt["manifest_rekor_entry_id"] != final_receipt["registry_record_rekor_entry_id"]
    )
    assert final_receipt["control_boundary"]["independent_organizational_custody"] is False


def test_registry_record_attestation_rejects_a_backdated_rekor_entry(tmp_path: Path) -> None:
    first_log_key = (b"k" * 32).hex()
    first = SigstoreObservation(
        statement={},
        log_key_sha256=first_log_key,
        log_index=41,
        entry_id=f"rekor:{first_log_key}:41",
        integrated_at_utc="2026-07-14T12:03:00+00:00",
        timestamp_token_sha256=_digest("first-timestamp"),
    )
    record = ProtocolRegistryRecord(
        manifest_sha256=_digest("manifest"),
        protocol_version="0.3.0",
        registered_at_utc=first.integrated_at_utc,
        registry_identity=ZENODO_REGISTRY_IDENTITY,
        registry_uri=ZENODO_REGISTRY_URI,
    )
    record_path = tmp_path / REGISTRY_RECORD_SUBJECT_PATH
    record_path.write_bytes(record.canonical_bytes() + b"\n")
    predicate = registry_record_predicate(
        c1_commit="1" * 40,
        c0_commit="0" * 40,
        manifest_digest=record.manifest_sha256,
        manifest_file_digest=_digest("manifest-file"),
        registry_record=record,
        manifest_bundle_digest=_digest("manifest-bundle"),
        manifest_observation=first,
    )
    predicate_path = tmp_path / "registry-predicate.json"
    predicate_path.write_text(
        json.dumps(predicate, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    bundle_path = tmp_path / "backdated.bundle.json"
    bundle_path.write_bytes(
        _c1_bundle(
            predicate=predicate,
            predicate_type=REGISTRY_RECORD_PREDICATE_TYPE,
            subject_name=REGISTRY_RECORD_SUBJECT_PATH,
            subject_digest=record.record_sha256,
            integrated_time=1_784_030_520,
            log_index=42,
        )
    )
    with pytest.raises(SuiteAttemptError, match="does not follow"):
        verify_c1_registry_record_attestation(
            record_path=record_path,
            predicate_path=predicate_path,
            bundle_path=bundle_path,
            receipt_output_path=tmp_path / "must-not-exist.json",
            verification_output_path=tmp_path / "must-not-exist-verification.json",
            verifier=_VerifiedC1Command(),
        )
