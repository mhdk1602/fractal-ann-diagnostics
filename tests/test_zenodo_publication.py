from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.zenodo_publication as publication
from fractal_ann_diagnostics.c1_manifest_transition import (
    C1ManifestTransitionError,
    C1ManifestTransitionReceipt,
)
from fractal_ann_diagnostics.github_state_attestation import (
    C1_MANIFEST_PATH,
    C1_REF,
    COMMON_CONTROL_LIMITATION,
    REGISTRATION_PREDICATE_TYPE,
    REGISTRATION_RECEIPT_SCHEMA,
    REGISTRATION_WORKFLOW_PATH,
    REGISTRY_ATTESTATION_RECEIPT_SCHEMA,
    REGISTRY_MATERIALIZATION_SCHEMA,
    REGISTRY_RECORD_PREDICATE_TYPE,
    REGISTRY_RECORD_SUBJECT_PATH,
    REPOSITORY,
    ZENODO_DRAFT_URI,
    ZENODO_RECORD_ID,
    ZENODO_REGISTRY_IDENTITY,
    ZENODO_REGISTRY_URI,
    ZENODO_RESERVATION_CREATED_AT_UTC,
    ZENODO_RESERVED_DOI,
    c1_registration_predicate,
    parse_sigstore_bundle,
    registry_record_predicate,
)
from fractal_ann_diagnostics.study import (
    ProtocolRegistrationReceipt,
    ProtocolRegistryRecord,
    StudyManifestError,
)


def _digest(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _bundle(
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
            "payload": base64.b64encode(_canonical(statement)).decode(),
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
    return _canonical(value)


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def _git_oid(kind: str, encoded: bytes) -> str:
    return hashlib.sha1(
        f"{kind} {len(encoded)}\0".encode() + encoded,
        usedforsecurity=False,
    ).hexdigest()


def _reservation() -> dict[str, object]:
    return {
        "created_at_utc": ZENODO_RESERVATION_CREATED_AT_UTC,
        "creator": "mhdk1602",
        "deposition_id": ZENODO_RECORD_ID,
        "direct_registry_record_uri": ZENODO_REGISTRY_URI,
        "draft_uri": ZENODO_DRAFT_URI,
        "protocol_version": "0.3.0",
        "reserved_doi": ZENODO_RESERVED_DOI,
        "schema_version": "fractal-zenodo-reservation-v1",
        "state": "unsubmitted",
        "submitted": False,
    }


def _make_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    root = tmp_path / "confirmatory-c1-registration"
    root.mkdir(mode=0o700, parents=True)
    semantic_digest = _digest("semantic frozen manifest")
    monkeypatch.setattr(publication, "validate_study_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(publication, "manifest_sha256", lambda _value: semantic_digest)

    def verify_transition_grounding(
        receipt: C1ManifestTransitionReceipt,
        **_arguments: object,
    ) -> None:
        expected = {
            "c0_evidence_release_file_sha256": _digest("c0-evidence-file"),
            "candidate_manifest_assembly_receipt_file_sha256": _digest(
                "candidate-assembly-receipt"
            ),
            "candidate_manifest_file_sha256": _digest("candidate-manifest-file"),
        }
        if any(getattr(receipt, name) != value for name, value in expected.items()):
            raise C1ManifestTransitionError("fixture transition grounding differs")

    monkeypatch.setattr(
        publication,
        "verify_c1_manifest_transition_receipt_bindings",
        verify_transition_grounding,
    )

    c0_commit = "0" * 40
    c0_binding = {
        "fixture": "closed-c0-release-binding",
        "target_commit": c0_commit,
    }
    manifest = (
        _canonical(
            {
                "sealed_execution": {"c0_evidence_release": c0_binding},
                "status": "frozen",
            }
        )
        + b"\n"
    )
    lock = f"{semantic_digest}\n".encode()
    reservation = _canonical(_reservation()) + b"\n"
    gh_version = (
        b"gh version 2.96.0 (2026-07-02)\nhttps://github.com/cli/cli/releases/tag/v2.96.0\n"
    )
    c0_public_bytes = (
        _canonical(
            {
                "fixture": "fresh-public-c0-verification",
                "schema_version": publication.C0_PUBLIC_VERIFICATION_SCHEMA,
            }
        )
        + b"\n"
    )
    c0_public = SimpleNamespace(
        binding_sha256=_digest(_canonical(c0_binding) + b"\n"),
        binding_source_file_sha256=_digest(manifest),
        binding_source_kind="frozen-manifest",
        c0_evidence_release_binding=c0_binding,
        canonical_file_bytes=lambda: c0_public_bytes,
        file_sha256=_digest(c0_public_bytes),
        gh_version_text=gh_version.decode(),
        release_tag="confirmatory-apparatus-c0",
        schema_version=publication.C0_PUBLIC_VERIFICATION_SCHEMA,
        target_commit=c0_commit,
    )

    def load_c0_public(path: Path) -> SimpleNamespace:
        if Path(path).read_bytes() != c0_public_bytes:
            raise publication.C0PublicVerificationError(
                "fixture C0 public-verification bytes differ"
            )
        return c0_public

    monkeypatch.setattr(
        publication,
        "load_c0_public_verification_receipt",
        load_c0_public,
    )
    commit = (
        f"tree {'1' * 40}\n"
        f"parent {c0_commit}\n"
        "author mhdk1602 <mhdk1602@users.noreply.github.com> 1784030400 +0000\n"
        "committer mhdk1602 <mhdk1602@users.noreply.github.com> 1784030400 +0000\n"
        "\nFreeze the confirmatory apparatus at C1.\n"
    ).encode()
    c1_commit = _git_oid("commit", commit)
    candidate_package = (tmp_path / "candidate-package").resolve()
    transition_receipt = C1ManifestTransitionReceipt(
        c0_commit=c0_commit,
        candidate_manifest_package_uri=candidate_package.as_uri(),
        candidate_manifest_uri=(candidate_package / "candidate-study-manifest.json").as_uri(),
        candidate_manifest_sha256=_digest("candidate-manifest"),
        candidate_manifest_file_sha256=_digest("candidate-manifest-file"),
        candidate_manifest_assembly_receipt_uri=(
            candidate_package / "candidate-manifest-assembly-receipt.json"
        ).as_uri(),
        candidate_manifest_assembly_receipt_file_sha256=_digest("candidate-assembly-receipt"),
        candidate_manifest_assembly_receipt_schema="fractal-candidate-manifest-assembly-v1",
        c0_evidence_release_uri=(tmp_path / "c0-evidence-release.json").resolve().as_uri(),
        c0_evidence_release_sha256=_digest("c0-evidence"),
        c0_evidence_release_file_sha256=_digest("c0-evidence-file"),
        apparatus_evidence_sha256=_digest("apparatus-evidence"),
        provider_phase_plan_closure_sha256=_digest("provider-plans"),
        frozen_manifest_uri=(tmp_path / "study-manifest.json").resolve().as_uri(),
        frozen_manifest_sha256=semantic_digest,
        frozen_manifest_file_sha256=_digest(manifest),
        frozen_manifest_byte_count=len(manifest),
        frozen_manifest_mode="0600",
    )
    transition_bytes = transition_receipt.canonical_file_bytes()
    predicate = c1_registration_predicate(
        c1_commit=c1_commit,
        c0_commit=c0_commit,
        tag_object_id=c1_commit,
        tag_object_type="commit",
        manifest_digest=semantic_digest,
        manifest_file_digest=_digest(manifest),
        lock_file_digest=_digest(lock),
        transition_receipt_file_digest=_digest(transition_bytes),
        c0_public_verification_file_digest=_digest(c0_public_bytes),
        c0_public_verification_binding_digest=c0_public.binding_sha256,
        candidate_manifest_digest=transition_receipt.candidate_manifest_sha256,
        candidate_manifest_file_digest=transition_receipt.candidate_manifest_file_sha256,
        candidate_assembly_receipt_file_digest=(
            transition_receipt.candidate_manifest_assembly_receipt_file_sha256
        ),
        reservation_file_digest=_digest(reservation),
    )
    first_bundle = _bundle(
        predicate=predicate,
        predicate_type=REGISTRATION_PREDICATE_TYPE,
        subject_name=C1_MANIFEST_PATH,
        subject_digest=_digest(manifest),
        integrated_time=1_784_030_520,
        log_index=41,
    )
    first = parse_sigstore_bundle(first_bundle)
    record = ProtocolRegistryRecord(
        manifest_sha256=semantic_digest,
        protocol_version="0.3.0",
        registered_at_utc=first.integrated_at_utc,
        registry_identity=ZENODO_REGISTRY_IDENTITY,
        registry_uri=ZENODO_REGISTRY_URI,
    )
    record_bytes = record.canonical_bytes() + b"\n"
    registry_predicate = registry_record_predicate(
        c1_commit=c1_commit,
        c0_commit=c0_commit,
        manifest_digest=semantic_digest,
        manifest_file_digest=_digest(manifest),
        registry_record=record,
        manifest_bundle_digest=_digest(first_bundle),
        manifest_observation=first,
    )
    second_bundle = _bundle(
        predicate=registry_predicate,
        predicate_type=REGISTRY_RECORD_PREDICATE_TYPE,
        subject_name=REGISTRY_RECORD_SUBJECT_PATH,
        subject_digest=record.record_sha256,
        integrated_time=1_784_030_521,
        log_index=42,
    )
    second = parse_sigstore_bundle(second_bundle)
    gh_verified = _canonical([{"verificationResult": {"verified": True}}])
    registration_receipt = {
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "predicate": predicate,
        "predicate_type": REGISTRATION_PREDICATE_TYPE,
        "repository": REPOSITORY,
        "schema_version": REGISTRATION_RECEIPT_SCHEMA,
        "workflow_ref": f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
        "workflow_sha": c1_commit,
    }
    materialization = {
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "manifest_attestation_verification_sha256": _digest(gh_verified),
        "predicate": registry_predicate,
        "predicate_type": REGISTRY_RECORD_PREDICATE_TYPE,
        "registry_record_sha256": record.record_sha256,
        "schema_version": REGISTRY_MATERIALIZATION_SCHEMA,
    }
    final_receipt = {
        "c1_commit": c1_commit,
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "manifest_rekor_entry_id": first.entry_id,
        "manifest_rekor_integrated_at_utc": first.integrated_at_utc,
        "registry_record_bundle_sha256": _digest(second_bundle),
        "registry_record_rekor_entry_id": second.entry_id,
        "registry_record_rekor_integrated_at_utc": second.integrated_at_utc,
        "registry_record_sha256": record.record_sha256,
        "registry_record_verification_sha256": _digest(gh_verified),
        "schema_version": REGISTRY_ATTESTATION_RECEIPT_SCHEMA,
    }

    files: dict[str, bytes] = {
        "c0-commit.txt": f"{c0_commit}\n".encode(),
        "c0-public-verification.json": c0_public_bytes,
        "c1-commit-object.txt": commit,
        "c1-commit.txt": f"{c1_commit}\n".encode(),
        "c1-tag-object-record.txt": commit,
        "c1-tag-object.txt": f"{c1_commit} commit\n".encode(),
        "gh-version.txt": gh_version,
        "manifest-gh-verification.json": gh_verified,
        "manifest-github-attestation-id.txt": b"1001\n",
        "manifest-github-attestation-url.txt": (
            f"https://github.com/{REPOSITORY}/attestations/1001\n".encode()
        ),
        "manifest-transition-receipt.json": transition_bytes,
        "protocol-registry-record.json": record_bytes,
        "protocol-registry-record.sigstore.bundle.json": second_bundle,
        "registration-predicate.json": _canonical(predicate) + b"\n",
        "registration-validation.json": _canonical(registration_receipt) + b"\n",
        "registry-attestation-validation.json": _canonical(final_receipt) + b"\n",
        "registry-gh-verification.json": gh_verified,
        "registry-materialization.json": _canonical(materialization) + b"\n",
        "registry-record-github-attestation-id.txt": b"1002\n",
        "registry-record-github-attestation-url.txt": (
            f"https://github.com/{REPOSITORY}/attestations/1002\n".encode()
        ),
        "registry-record-predicate.json": _canonical(registry_predicate) + b"\n",
        "study-manifest.json": manifest,
        "study-manifest.sha256": lock,
        "study-manifest.sigstore.bundle.json": first_bundle,
        "workflow-run.txt": (
            f"https://github.com/{REPOSITORY}/actions/runs/29053941472/attempts/1\n".encode()
        ),
        "zenodo-reservation.json": reservation,
    }
    assert set(files) == set(publication.PACKAGE_FILE_NAMES) - {"SHA256SUMS"}
    for name, encoded in files.items():
        (root / name).write_bytes(encoded)
    sums = b"".join(
        f"{_digest(files[name])}  ./{name}\n".encode()
        for name in sorted(files, key=lambda value: value.encode())
    )
    (root / "SHA256SUMS").write_bytes(sums)
    return root, semantic_digest


@pytest.fixture
def package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> publication.ValidatedRegistrationPackage:
    root, semantic_digest = _make_package(tmp_path, monkeypatch)
    result = publication.validate_registration_package(root)
    assert result.manifest_sha256 == semantic_digest
    return result


def test_closed_file_set_matches_registration_workflow() -> None:
    assert len(publication.PACKAGE_FILE_NAMES) == 27
    assert len(set(publication.PACKAGE_FILE_NAMES)) == 27
    workflow = Path(".github/workflows/confirmatory-registration-attestation.yml").read_text(
        encoding="utf-8"
    )
    for name in publication.PACKAGE_FILE_NAMES:
        assert name in workflow
    assert "fractal_ann_diagnostics.zenodo_publication" in workflow
    assert "preflight" in workflow


def test_fresh_preflight_verifies_both_retained_attestations(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    class Verifier:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        def verify(
            self,
            *,
            subject_path: Path,
            bundle_path: Path,
            c1_commit: str,
            predicate_type: str,
        ) -> bytes:
            assert c1_commit == package.c1_commit
            self.calls.append((subject_path.name, bundle_path.name, predicate_type))
            return _canonical([{"verificationResult": {"verified": True}}])

    verifier = Verifier()
    result = publication.verify_registration_package_attestations(
        package,
        verifier=verifier,
    )
    assert result["verified"] is True
    assert verifier.calls == [
        (
            "study-manifest.json",
            "study-manifest.sigstore.bundle.json",
            REGISTRATION_PREDICATE_TYPE,
        ),
        (
            "protocol-registry-record.json",
            "protocol-registry-record.sigstore.bundle.json",
            REGISTRY_RECORD_PREDICATE_TYPE,
        ),
    ]


def test_fresh_preflight_verifies_private_snapshots_and_detects_verifier_mutation(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    expected = package.inventory["study-manifest.json"].data

    class MutatingVerifier:
        def verify(
            self,
            *,
            subject_path: Path,
            bundle_path: Path,
            c1_commit: str,
            predicate_type: str,
        ) -> bytes:
            del bundle_path, c1_commit, predicate_type
            assert subject_path.parent != package.root
            if subject_path.name == "study-manifest.json":
                assert subject_path.read_bytes() == expected
                subject_path.write_bytes(b"substituted after provider admission")
            return _canonical([{"verificationResult": {"verified": True}}])

    with pytest.raises(publication.ZenodoPublicationError, match="snapshot changed"):
        publication.verify_registration_package_attestations(
            package,
            verifier=MutatingVerifier(),
        )


def test_fresh_preflight_rejects_live_package_substitution_after_offline_admission(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    expected = package.inventory["study-manifest.json"].data
    (package.root / "study-manifest.json").write_bytes(b'{"status":"substituted"}\n')

    class SnapshotVerifier:
        def verify(
            self,
            *,
            subject_path: Path,
            bundle_path: Path,
            c1_commit: str,
            predicate_type: str,
        ) -> bytes:
            del bundle_path, c1_commit, predicate_type
            if subject_path.name == "study-manifest.json":
                assert subject_path.read_bytes() == expected
            return _canonical([{"verificationResult": {"verified": True}}])

    with pytest.raises(publication.ZenodoPublicationError):
        publication.verify_registration_package_attestations(
            package,
            verifier=SnapshotVerifier(),
        )


def _rewrite_sums(root: Path) -> None:
    names = sorted(
        set(publication.PACKAGE_FILE_NAMES) - {"SHA256SUMS"},
        key=lambda value: value.encode(),
    )
    (root / "SHA256SUMS").write_bytes(
        b"".join(f"{_digest((root / name).read_bytes())}  ./{name}\n".encode() for name in names)
    )


def test_closed_package_rejects_missing_extra_checksum_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    (root / "workflow-run.txt").unlink()
    with pytest.raises(publication.ZenodoPublicationError, match="file set differs"):
        publication.validate_registration_package(root)

    (root / "workflow-run.txt").write_text(
        f"https://github.com/{REPOSITORY}/actions/runs/1/attempts/1\n"
    )
    (root / "surprise.txt").write_text("unregistered\n")
    with pytest.raises(publication.ZenodoPublicationError, match="extra"):
        publication.validate_registration_package(root)
    (root / "surprise.txt").unlink()
    with pytest.raises(publication.ZenodoPublicationError, match="SHA256SUMS"):
        publication.validate_registration_package(root)
    _rewrite_sums(root)
    (root / "workflow-run.txt").unlink()
    (root / "workflow-run.txt").symlink_to(root / "c1-commit.txt")
    with pytest.raises(publication.ZenodoPublicationError, match="without following links"):
        publication.validate_registration_package(root)

    root, _ = _make_package(tmp_path / "fifo", monkeypatch)
    (root / "workflow-run.txt").unlink()
    os.mkfifo(root / "workflow-run.txt", mode=0o600)
    with pytest.raises(publication.ZenodoPublicationError, match="bounded regular file"):
        publication.validate_registration_package(root)


def test_closed_package_rejects_root_directory_replacement_while_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    original_read = publication._read_package_file_at
    moved = root.with_name("admitted-package-moved")
    swapped = False

    def swap_root_after_first_read(root_descriptor: int, *, name: str) -> bytes:
        nonlocal swapped
        encoded = original_read(root_descriptor, name=name)
        if not swapped:
            root.rename(moved)
            root.mkdir(mode=0o700)
            swapped = True
        return encoded

    monkeypatch.setattr(publication, "_read_package_file_at", swap_root_after_first_read)
    with pytest.raises(publication.ZenodoPublicationError, match="root was replaced"):
        publication.validate_registration_package(root)


def test_package_rejects_changed_predicate_even_with_rewritten_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    predicate = json.loads((root / "registration-predicate.json").read_text())
    predicate["freeze"]["c1_ref"] = "refs/tags/other"
    _write_json(root / "registration-predicate.json", predicate)
    _rewrite_sums(root)
    with pytest.raises(publication.ZenodoPublicationError, match="registration predicate"):
        publication.validate_registration_package(root)


def test_package_rejects_orphaned_transition_receipt_even_with_rewritten_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    transition_path = root / "manifest-transition-receipt.json"
    transition = json.loads(transition_path.read_text())
    transition["candidate_manifest_sha256"] = "f" * 64
    _write_json(transition_path, transition)
    _rewrite_sums(root)
    with pytest.raises(publication.ZenodoPublicationError, match="signed C1 predicate"):
        publication.validate_registration_package(root)


def test_package_path_must_be_absolute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    with pytest.raises(publication.ZenodoPublicationError, match="must be absolute"):
        publication.validate_registration_package(Path(root.name))


def test_package_rejects_changed_git_commit_even_with_rewritten_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    (root / "c1-commit-object.txt").write_bytes(
        (root / "c1-commit-object.txt").read_bytes() + b"forged\n"
    )
    _rewrite_sums(root)
    with pytest.raises(publication.ZenodoPublicationError, match="do not hash"):
        publication.validate_registration_package(root)


def test_package_rejects_swapped_bundle_and_rewritten_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    bundle_path = root / "study-manifest.sigstore.bundle.json"
    bundle = json.loads(bundle_path.read_text())
    statement = json.loads(base64.b64decode(bundle["dsseEnvelope"]["payload"]))
    statement["subject"][0]["digest"]["sha256"] = "f" * 64
    bundle["dsseEnvelope"]["payload"] = base64.b64encode(_canonical(statement)).decode()
    bundle_path.write_bytes(_canonical(bundle))
    _rewrite_sums(root)
    with pytest.raises(publication.ZenodoPublicationError, match="attestation package"):
        publication.validate_registration_package(root)

    root, _ = _make_package(tmp_path / "second", monkeypatch)
    receipt_path = root / "registry-attestation-validation.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["registry_record_verification_sha256"] = "f" * 64
    _write_json(receipt_path, receipt)
    _rewrite_sums(root)
    with pytest.raises(publication.ZenodoPublicationError, match="receipt differs"):
        publication.validate_registration_package(root)


def test_package_requires_distinct_rekor_signed_timestamp_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    first = json.loads((root / "study-manifest.sigstore.bundle.json").read_text())
    second_path = root / "protocol-registry-record.sigstore.bundle.json"
    second = json.loads(second_path.read_text())
    second["verificationMaterial"]["tlogEntries"][0]["inclusionPromise"]["signedEntryTimestamp"] = (
        first["verificationMaterial"]["tlogEntries"][0]["inclusionPromise"]["signedEntryTimestamp"]
    )
    second_path.write_bytes(_canonical(second))
    _rewrite_sums(root)
    with pytest.raises(publication.ZenodoPublicationError, match="does not follow"):
        publication.validate_registration_package(root)


def test_package_requires_distinct_github_attestation_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    (root / "registry-record-github-attestation-id.txt").write_bytes(b"1001\n")
    (root / "registry-record-github-attestation-url.txt").write_bytes(
        f"https://github.com/{REPOSITORY}/attestations/1001\n".encode()
    )
    _rewrite_sums(root)
    with pytest.raises(publication.ZenodoPublicationError, match="distinct IDs"):
        publication.validate_registration_package(root)


@pytest.mark.parametrize(
    "field",
    (
        "c0_evidence_release_file_sha256",
        "candidate_manifest_file_sha256",
        "candidate_manifest_assembly_receipt_file_sha256",
    ),
)
def test_package_rejects_each_ungrounded_transition_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    transition_path = root / "manifest-transition-receipt.json"
    transition = json.loads(transition_path.read_bytes())
    transition[field] = "0" * 64
    _write_json(transition_path, transition)
    _rewrite_sums(root)

    with pytest.raises(publication.ZenodoPublicationError, match="transition receipt"):
        publication.validate_registration_package(root)


def test_package_rejects_changed_c0_public_receipt_even_with_rewritten_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    receipt_path = root / "c0-public-verification.json"
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    _rewrite_sums(root)

    with pytest.raises(publication.ZenodoPublicationError, match="C0 public verification"):
        publication.validate_registration_package(root)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("binding_source_file_sha256", "f" * 64),
        ("target_commit", "f" * 40),
        ("c0_evidence_release_binding", {"fixture": "substituted"}),
        ("gh_version_text", "gh version 2.96.0 (2026-07-02)\nsubstituted\n"),
    ),
)
def test_package_rejects_c0_public_cross_binding_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    root, _ = _make_package(tmp_path, monkeypatch)
    receipt_path = root / "c0-public-verification.json"
    original = publication.load_c0_public_verification_receipt(receipt_path)
    substituted = SimpleNamespace(**{**vars(original), field: replacement})
    monkeypatch.setattr(
        publication,
        "load_c0_public_verification_receipt",
        lambda _path: substituted,
    )

    with pytest.raises(
        publication.ZenodoPublicationError,
        match="retained C0 public verification",
    ):
        publication.validate_registration_package(root)


def _metadata(*, creator: str = publication.ZENODO_CREATOR) -> dict[str, object]:
    return {
        "access_right": publication.ZENODO_ACCESS_RIGHT,
        "creators": [
            {
                "name": creator,
                "orcid": publication.ZENODO_CREATOR_ORCID,
            }
        ],
        "description": publication.ZENODO_DESCRIPTION,
        "keywords": list(publication.ZENODO_KEYWORDS),
        "license": publication.ZENODO_LICENSE_ID,
        "notes": publication.ZENODO_NOTES,
        "publication_date": publication.ZENODO_PUBLICATION_DATE,
        "prereserve_doi": {
            "doi": ZENODO_RESERVED_DOI,
            "recid": ZENODO_RECORD_ID,
        },
        "publication_type": publication.ZENODO_PUBLICATION_TYPE,
        "title": publication.ZENODO_TITLE,
        "upload_type": publication.ZENODO_UPLOAD_TYPE,
    }


def _file_row(
    item: publication.RegistrationPackageFile,
    *,
    public: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "checksum": f"md5:{item.md5}",
        "key": item.name,
        "size": item.size,
    }
    if public:
        row["links"] = {"self": publication._public_file_uri(item.name)}
    return row


def _draft(
    package: publication.ValidatedRegistrationPackage,
    names: set[str],
) -> dict[str, object]:
    inventory = package.inventory
    return {
        "files": [_file_row(inventory[name]) for name in sorted(names)],
        "id": ZENODO_RECORD_ID,
        "links": {
            "bucket": "https://zenodo.org/api/files/12345678-1234-1234-1234-123456789abc",
            "publish": publication.ZENODO_PUBLISH_API_URI,
            "self": publication.ZENODO_DRAFT_API_URI,
        },
        "metadata": _metadata(),
        "record_id": ZENODO_RECORD_ID,
        "state": "unsubmitted",
        "submitted": False,
    }


def _public(package: publication.ValidatedRegistrationPackage) -> dict[str, object]:
    inventory = package.inventory
    metadata = _metadata()
    metadata.pop("prereserve_doi")
    metadata.pop("publication_type")
    metadata.pop("upload_type")
    metadata["resource_type"] = {
        "subtype": publication.ZENODO_PUBLICATION_TYPE,
        "title": "Other",
        "type": publication.ZENODO_UPLOAD_TYPE,
    }
    metadata["license"] = {"id": publication.ZENODO_LICENSE_ID}
    return {
        "doi": ZENODO_RESERVED_DOI,
        "files": [_file_row(inventory[name], public=True) for name in sorted(inventory)],
        "id": ZENODO_RECORD_ID,
        "metadata": metadata,
        "recid": str(ZENODO_RECORD_ID),
        "state": "done",
        "status": "published",
        "submitted": True,
    }


def test_fixed_metadata_omits_null_affiliation_and_binds_reservation_date() -> None:
    fixed = publication._fixed_protocol_metadata()
    creators = fixed["creators"]
    assert isinstance(creators, list)
    assert "affiliation" not in creators[0]
    assert fixed["publication_date"] == "2026-07-14"

    with_empty_affiliation = _metadata()
    creator = with_empty_affiliation["creators"][0]  # type: ignore[index]
    creator["affiliation"] = ""  # type: ignore[index]
    publication._verify_fixed_metadata(with_empty_affiliation, public=False)

    creator["affiliation"] = "another institution"  # type: ignore[index]
    with pytest.raises(publication.ZenodoPublicationError, match="affiliation differs"):
        publication._verify_fixed_metadata(with_empty_affiliation, public=False)

    wrong_date = _metadata()
    wrong_date["publication_date"] = "2026-07-18"
    with pytest.raises(publication.ZenodoPublicationError, match="publication date differs"):
        publication._verify_fixed_metadata(wrong_date, public=False)


class _FakeTransport:
    def __init__(
        self,
        package: publication.ValidatedRegistrationPackage,
        names: set[str],
        *,
        public_available: bool = True,
    ) -> None:
        self.package = package
        self.names = set(names)
        self.calls: list[tuple[str, str, bool | None]] = []
        self.public = _public(package)
        self.public_available = public_available

    def get_json(self, url: str, *, authenticated: bool) -> dict[str, object]:
        self.calls.append(("GET", url, authenticated))
        if url == publication.ZENODO_DRAFT_API_URI:
            return _draft(self.package, self.names)
        if url == publication.ZENODO_PUBLIC_API_URI:
            if not self.public_available:
                raise publication._ZenodoHttpStatusError(404)
            return self.public
        raise AssertionError(url)

    def get_bytes(self, url: str, *, authenticated: bool) -> bytes:
        self.calls.append(("GET-BYTES", url, authenticated))
        if authenticated:
            name = url.rsplit("/", 1)[-1]
            return self.package.inventory[name].data
        matches = [
            item for item in self.package.files if publication._public_file_uri(item.name) == url
        ]
        assert len(matches) == 1
        return matches[0].data

    def put_bytes(self, url: str, data: bytes) -> dict[str, object]:
        self.calls.append(("PUT", url, True))
        name = url.rsplit("/", 1)[-1]
        item = self.package.inventory[name]
        assert data == item.data
        self.names.add(name)
        return _file_row(item)

    def put_json(self, url: str, value: dict[str, object]) -> dict[str, object]:
        self.calls.append(("PUT-JSON", url, True))
        assert url == publication.ZENODO_DRAFT_API_URI
        assert value == {"metadata": publication._fixed_protocol_metadata()}
        return _draft(self.package, self.names)

    def post_json(self, url: str) -> dict[str, object]:
        self.calls.append(("POST", url, True))
        self.public_available = True
        return self.public


class _AcceptingVerifier:
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
        del bundle_path, c1_commit
        self.calls.append((subject_path.name, predicate_type))
        return _canonical([{"verificationResult": {"verified": True}}])


def _local_registration_evidence(
    tmp_path: Path,
    package: publication.ValidatedRegistrationPackage,
) -> tuple[Path, Path]:
    record_path = tmp_path / "local-protocol-registry-record.json"
    record_path.write_bytes(package.registry_record_bytes)
    record = ProtocolRegistryRecord.from_dict(json.loads(package.registry_record_bytes))
    receipt = ProtocolRegistrationReceipt(
        manifest_sha256=record.manifest_sha256,
        protocol_version=record.protocol_version,
        registered_at_utc=record.registered_at_utc,
        registry_identity=record.registry_identity,
        registry_uri=record.registry_uri,
        registry_record_sha256=record.record_sha256,
    )
    receipt_path = tmp_path / "local-protocol-registration-receipt.json"
    receipt_path.write_bytes(receipt.canonical_bytes() + b"\n")
    return record_path, receipt_path


def test_production_registration_mints_only_after_full_public_revalidation(
    tmp_path: Path,
    package: publication.ValidatedRegistrationPackage,
) -> None:
    record_path, receipt_path = _local_registration_evidence(tmp_path, package)
    verifier = _AcceptingVerifier()
    transport = _FakeTransport(package, set(package.inventory))

    verified = publication._verify_production_protocol_registration(
        package.root,
        registration_record_path=record_path,
        registration_receipt_path=receipt_path,
        verifier=verifier,
        transport=transport,
    )
    verified.assert_current()

    assert verified.record.registry_identity == ZENODO_REGISTRY_IDENTITY
    assert verified.record.registry_uri == ZENODO_REGISTRY_URI
    assert len(verified.package_file_sha256s) == 27
    assert len(verifier.calls) == 4
    assert sum(call[0] == "GET" for call in transport.calls) == 2
    assert sum(call[0] == "GET-BYTES" for call in transport.calls) == 54
    assert all(call[2] is False for call in transport.calls)


def test_public_production_verifier_exposes_no_trust_substitution_hooks() -> None:
    parameters = inspect.signature(publication.verify_production_protocol_registration).parameters
    assert "verifier" not in parameters
    assert "transport" not in parameters


def test_production_registration_rejects_incomplete_public_record(
    tmp_path: Path,
    package: publication.ValidatedRegistrationPackage,
) -> None:
    record_path, receipt_path = _local_registration_evidence(tmp_path, package)
    transport = _FakeTransport(package, set(package.inventory))
    transport.public["files"].pop()

    with pytest.raises(publication.ZenodoPublicationError, match="incomplete"):
        publication._verify_production_protocol_registration(
            package.root,
            registration_record_path=record_path,
            registration_receipt_path=receipt_path,
            verifier=_AcceptingVerifier(),
            transport=transport,
        )


def test_production_registration_rejects_failed_registry_attestation(
    tmp_path: Path,
    package: publication.ValidatedRegistrationPackage,
) -> None:
    record_path, receipt_path = _local_registration_evidence(tmp_path, package)

    class RejectSecond(_AcceptingVerifier):
        def verify(self, **arguments: object) -> bytes:
            encoded = super().verify(**arguments)  # type: ignore[arg-type]
            if len(self.calls) == 2:
                return b"[]"
            return encoded

    with pytest.raises(publication.ZenodoPublicationError, match="fresh C1 attestation"):
        publication._verify_production_protocol_registration(
            package.root,
            registration_record_path=record_path,
            registration_receipt_path=receipt_path,
            verifier=RejectSecond(),
            transport=_FakeTransport(package, set(package.inventory)),
        )


def test_production_registration_rechecks_public_record_at_run_admission(
    tmp_path: Path,
    package: publication.ValidatedRegistrationPackage,
) -> None:
    record_path, receipt_path = _local_registration_evidence(tmp_path, package)
    transport = _FakeTransport(package, set(package.inventory))
    verified = publication._verify_production_protocol_registration(
        package.root,
        registration_record_path=record_path,
        registration_receipt_path=receipt_path,
        verifier=_AcceptingVerifier(),
        transport=transport,
    )
    transport.public["files"].pop()

    with pytest.raises(StudyManifestError, match="fresh C1 registration") as error:
        verified.assert_current()
    assert isinstance(error.value.__cause__, publication.ZenodoPublicationError)
    assert "incomplete" in str(error.value.__cause__)


def test_production_registration_rejects_non_zenodo_local_record(
    tmp_path: Path,
    package: publication.ValidatedRegistrationPackage,
) -> None:
    record_path, receipt_path = _local_registration_evidence(tmp_path, package)
    substituted = ProtocolRegistryRecord(
        manifest_sha256=package.manifest_sha256,
        protocol_version="0.3.0",
        registered_at_utc="2026-07-13T12:00:00+00:00",
        registry_identity="osf-registration:substituted",
        registry_uri="https://osf.io/registries/substituted",
    )
    record_path.write_bytes(substituted.canonical_bytes() + b"\n")
    receipt = ProtocolRegistrationReceipt(
        manifest_sha256=substituted.manifest_sha256,
        protocol_version=substituted.protocol_version,
        registered_at_utc=substituted.registered_at_utc,
        registry_identity=substituted.registry_identity,
        registry_uri=substituted.registry_uri,
        registry_record_sha256=substituted.record_sha256,
    )
    receipt_path.write_bytes(receipt.canonical_bytes() + b"\n")

    with pytest.raises(publication.ZenodoPublicationError, match="differ from the verified"):
        publication._verify_production_protocol_registration(
            package.root,
            registration_record_path=record_path,
            registration_receipt_path=receipt_path,
            verifier=_AcceptingVerifier(),
            transport=_FakeTransport(package, set(package.inventory)),
        )


def test_stage_resumes_only_missing_file_then_rechecks_full_inventory(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    names = set(package.inventory)
    names.remove("SHA256SUMS")
    transport = _FakeTransport(package, names)
    result = publication._stage_package(package, transport)
    assert result["uploaded"] == ["SHA256SUMS"]
    assert result["ready_to_publish"] is True
    assert [call[0] for call in transport.calls[:4]] == [
        "PUT-JSON",
        "GET",
        "PUT",
        "GET",
    ]
    assert sum(call[0] == "GET-BYTES" and call[2] is True for call in transport.calls) == 27


def test_stage_is_read_only_when_the_exact_draft_is_already_complete(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    transport = _FakeTransport(package, set(package.inventory))
    result = publication._stage_package(package, transport)
    assert result["uploaded"] == []
    assert [call[0] for call in transport.calls[:3]] == ["PUT-JSON", "GET", "GET"]
    assert sum(call[0] == "GET-BYTES" and call[2] is True for call in transport.calls) == 27


def test_stage_metadata_put_is_idempotent_and_precedes_every_file_put(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    transport = _FakeTransport(package, set())
    first = publication._stage_package(package, transport)
    first_file_puts = sum(call[0] == "PUT" for call in transport.calls)
    second = publication._stage_package(package, transport)

    assert first["uploaded"] == sorted(
        publication.PACKAGE_FILE_NAMES,
        key=lambda value: value.encode("utf-8"),
    )
    assert second["uploaded"] == []
    assert first_file_puts == 27
    assert sum(call[0] == "PUT" for call in transport.calls) == first_file_puts
    metadata_positions = [
        position for position, call in enumerate(transport.calls) if call[0] == "PUT-JSON"
    ]
    file_positions = [position for position, call in enumerate(transport.calls) if call[0] == "PUT"]
    assert len(metadata_positions) == 2
    assert metadata_positions[0] < file_positions[0]


@pytest.mark.parametrize("defect", ("put-response", "authenticated-readback"))
def test_stage_rejects_metadata_drift_before_any_file_put(
    package: publication.ValidatedRegistrationPackage,
    defect: str,
) -> None:
    class MetadataDrift(_FakeTransport):
        def put_json(self, url: str, value: dict[str, object]) -> dict[str, object]:
            response = super().put_json(url, value)
            if defect == "put-response":
                response["metadata"]["software_record_injection"] = True
            return response

        def get_json(self, url: str, *, authenticated: bool) -> dict[str, object]:
            response = super().get_json(url, authenticated=authenticated)
            if defect == "authenticated-readback" and url == publication.ZENODO_DRAFT_API_URI:
                response["metadata"]["software_record_injection"] = True
            return response

    transport = MetadataDrift(package, set())
    with pytest.raises(publication.ZenodoPublicationError, match="exact protocol-record payload"):
        publication._stage_package(package, transport)
    assert not any(call[0] == "PUT" for call in transport.calls)


def test_stage_rejects_local_package_mutation_during_remote_admission(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    class MutatingTransport(_FakeTransport):
        def get_json(self, url: str, *, authenticated: bool) -> dict[str, object]:
            value = super().get_json(url, authenticated=authenticated)
            if sum(call[0] == "GET" for call in self.calls) == 2:
                (package.root / "workflow-run.txt").write_bytes(b"substituted\n")
            return value

    with pytest.raises(publication.ZenodoPublicationError):
        publication._stage_package(
            package,
            MutatingTransport(package, set(package.inventory)),
        )


@pytest.mark.parametrize(
    "mutation",
    ["extra", "changed", "creator", "contributor", "submitted"],
)
def test_stage_rejects_remote_drift_before_upload(
    package: publication.ValidatedRegistrationPackage,
    mutation: str,
) -> None:
    class Drift(_FakeTransport):
        def get_json(self, url: str, *, authenticated: bool) -> dict[str, object]:
            value = super().get_json(url, authenticated=authenticated)
            if mutation == "extra":
                value["files"].append({"checksum": "md5:" + "0" * 32, "key": "extra", "size": 1})
            elif mutation == "changed":
                value["files"][0]["checksum"] = "md5:" + "0" * 32
            elif mutation == "creator":
                value["metadata"] = _metadata(creator="another-user")
            elif mutation == "contributor":
                value["metadata"]["contributors"] = [{"name": "Another Author"}]
            else:
                value["submitted"] = True
            return value

    transport = Drift(package, set(package.inventory))
    with pytest.raises(publication.ZenodoPublicationError):
        publication._stage_package(package, transport)
    assert not any(call[0] == "PUT" for call in transport.calls)


def test_publish_requires_complete_draft_and_verifies_public_anonymously(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    partial = _FakeTransport(
        package,
        set(package.inventory) - {"SHA256SUMS"},
        public_available=False,
    )
    with pytest.raises(publication.ZenodoPublicationError, match="incomplete"):
        publication._publish_package(package, partial)
    assert not any(call[0] == "POST" for call in partial.calls)

    transport = _FakeTransport(package, set(package.inventory), public_available=False)
    result = publication._publish_package(package, transport)
    assert result["submitted"] is True
    assert ("POST", publication.ZENODO_PUBLISH_API_URI, True) in transport.calls
    assert ("GET", publication.ZENODO_PUBLIC_API_URI, False) in transport.calls
    assert ("GET-BYTES", ZENODO_REGISTRY_URI, False) in transport.calls
    assert sum(call[0] == "GET-BYTES" and call[2] is True for call in transport.calls) == len(
        package.files
    )
    assert sum(call[0] == "GET-BYTES" and call[2] is False for call in transport.calls) == len(
        package.files
    )


def test_publish_polls_only_bounded_integration_statuses(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    class Delayed(_FakeTransport):
        def __init__(self) -> None:
            super().__init__(package, set(package.inventory), public_available=False)
            self.public_attempts = 0

        def get_json(self, url: str, *, authenticated: bool) -> dict[str, object]:
            if url == publication.ZENODO_PUBLIC_API_URI:
                self.public_attempts += 1
                if self.public_attempts < 4:
                    raise publication._ZenodoHttpStatusError(404)
            return super().get_json(url, authenticated=authenticated)

    sleeps: list[float] = []
    transport = Delayed()
    result = publication._publish_package(package, transport, sleep=sleeps.append)
    assert result["byte_verified_file_count"] == len(package.files)
    assert transport.public_attempts == 4
    assert sleeps == [publication._PUBLICATION_POLL_SECONDS] * 2

    class Forbidden(Delayed):
        def get_json(self, url: str, *, authenticated: bool) -> dict[str, object]:
            if url == publication.ZENODO_PUBLIC_API_URI:
                raise publication._ZenodoHttpStatusError(403)
            return super().get_json(url, authenticated=authenticated)

    with pytest.raises(publication._ZenodoHttpStatusError):
        publication._publish_package(package, Forbidden(), sleep=sleeps.append)


def test_publish_is_read_only_when_exact_public_record_already_exists(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    transport = _FakeTransport(package, set(package.inventory))

    result = publication._publish_package(package, transport)

    assert result["byte_verified_file_count"] == len(package.files)
    assert not any(call[0] == "POST" for call in transport.calls)
    assert not any(call[1] == publication.ZENODO_DRAFT_API_URI for call in transport.calls)


@pytest.mark.parametrize("failure", ("mismatch", "forbidden"))
def test_publish_preflight_permits_post_only_after_exact_public_404(
    package: publication.ValidatedRegistrationPackage,
    failure: str,
) -> None:
    class PublicFailure(_FakeTransport):
        def get_json(self, url: str, *, authenticated: bool) -> dict[str, object]:
            if url == publication.ZENODO_PUBLIC_API_URI and failure == "forbidden":
                self.calls.append(("GET", url, authenticated))
                raise publication._ZenodoHttpStatusError(403)
            return super().get_json(url, authenticated=authenticated)

    transport = PublicFailure(package, set(package.inventory))
    if failure == "mismatch":
        transport.public["doi"] = "10.5281/zenodo.1"

    with pytest.raises(publication.ZenodoPublicationError):
        publication._publish_package(package, transport)

    assert not any(call[0] == "POST" for call in transport.calls)
    assert not any(call[1] == publication.ZENODO_DRAFT_API_URI for call in transport.calls)


def test_publish_recovers_lost_post_response_without_reposting(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    class LostPostResponse(_FakeTransport):
        def post_json(self, url: str) -> dict[str, object]:
            self.calls.append(("POST", url, True))
            self.public_available = True
            raise publication._ZenodoHttpStatusError(503)

    transport = LostPostResponse(
        package,
        set(package.inventory),
        public_available=False,
    )
    first = publication._publish_package(package, transport)
    second = publication._publish_package(package, transport)

    assert first == second
    assert sum(call[0] == "POST" for call in transport.calls) == 1


def test_publish_ambiguous_post_exhaustion_is_bounded_and_never_reposts(
    package: publication.ValidatedRegistrationPackage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LostPostWithoutPublication(_FakeTransport):
        def post_json(self, url: str) -> dict[str, object]:
            self.calls.append(("POST", url, True))
            raise publication._ZenodoHttpStatusError(503)

    monkeypatch.setattr(publication, "_PUBLICATION_POLL_ATTEMPTS", 3)
    transport = LostPostWithoutPublication(
        package,
        set(package.inventory),
        public_available=False,
    )
    sleeps: list[float] = []

    with pytest.raises(publication._ZenodoHttpStatusError, match="503"):
        publication._publish_package(package, transport, sleep=sleeps.append)

    assert sum(call[0] == "POST" for call in transport.calls) == 1
    assert sleeps == [publication._PUBLICATION_POLL_SECONDS] * 2


def test_synthetic_27_file_package_rehearses_preflight_stage_and_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, semantic_digest = _make_package(tmp_path, monkeypatch)
    package = publication.validate_registration_package(root)

    class Verifier:
        def verify(self, **_arguments: object) -> bytes:
            return _canonical([{"verificationResult": {"verified": True}}])

    preflight = publication.verify_registration_package_attestations(
        package,
        verifier=Verifier(),
    )
    transport = _FakeTransport(package, set(), public_available=False)
    staged = publication._stage_package(package, transport)
    published = publication._publish_package(package, transport)

    assert package.manifest_sha256 == semantic_digest
    assert len(package.files) == 27
    assert preflight["verified"] is True
    assert staged["uploaded"] == sorted(
        publication.PACKAGE_FILE_NAMES,
        key=lambda value: value.encode("utf-8"),
    )
    assert published["submitted"] is True
    assert published["byte_verified_file_count"] == 27
    assert sum(call[0] == "PUT" for call in transport.calls) == 27
    assert sum(call[0] == "GET-BYTES" and call[2] is True for call in transport.calls) == 54
    assert sum(call[0] == "GET-BYTES" and call[2] is False for call in transport.calls) == 27


def test_public_verifier_rejects_changed_direct_record(
    package: publication.ValidatedRegistrationPackage,
) -> None:
    class ChangedRecord(_FakeTransport):
        def get_bytes(self, url: str, *, authenticated: bool) -> bytes:
            encoded = super().get_bytes(url, authenticated=authenticated)
            if url == ZENODO_REGISTRY_URI:
                return encoded + b" "
            return encoded

    with pytest.raises(publication.ZenodoPublicationError, match="package bytes"):
        publication._verify_public(package, ChangedRecord(package, set(package.inventory)))


@pytest.mark.parametrize("mutation", ["id", "doi", "state", "extra-file"])
def test_public_verifier_rejects_record_drift(
    package: publication.ValidatedRegistrationPackage,
    mutation: str,
) -> None:
    class Drift(_FakeTransport):
        def get_json(self, url: str, *, authenticated: bool) -> dict[str, object]:
            value = super().get_json(url, authenticated=authenticated)
            if url != publication.ZENODO_PUBLIC_API_URI:
                return value
            if mutation == "id":
                value["id"] = ZENODO_RECORD_ID + 1
            elif mutation == "doi":
                value["doi"] = "10.5281/zenodo.1"
            elif mutation == "state":
                value["state"] = "unsubmitted"
            else:
                value["files"].append({"checksum": "md5:" + "0" * 32, "key": "extra", "size": 0})
            return value

    with pytest.raises(publication.ZenodoPublicationError):
        publication._verify_public(package, Drift(package, set(package.inventory)))


def test_token_is_fd_only_bounded_and_transport_redacts_and_zeroes() -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"0123456789abcdef0123456789abcdef\n")
    os.close(write_fd)
    token = publication._read_token_fd(read_fd)
    os.close(read_fd)
    assert bytes(token) == b"0123456789abcdef0123456789abcdef"
    transport = publication._ZenodoHttpsTransport(token)
    assert "012345" not in repr(transport)
    transport.close()
    assert not any(token)
    assert "--token" not in publication._parser().format_help()

    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"x" * (publication._MAX_TOKEN_BYTES + 2))
    os.close(write_fd)
    with pytest.raises(publication.ZenodoPublicationError, match="byte limit"):
        publication._read_token_fd(read_fd)
    os.close(read_fd)


def test_transport_rejects_other_hosts_queries_and_redirects() -> None:
    context = publication._verified_tls_context()
    assert context.verify_mode == publication.ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.minimum_version == publication.ssl.TLSVersion.TLSv1_2
    strict = getattr(publication.ssl, "VERIFY_X509_STRICT", 0)
    if strict:
        assert context.verify_flags & strict == 0
    with pytest.raises(publication.ZenodoPublicationError, match="zenodo.org"):
        publication._validate_transport_url("https://example.org/api/records/21361837")
    with pytest.raises(publication.ZenodoPublicationError, match="query-free"):
        publication._validate_transport_url(publication.ZENODO_DRAFT_API_URI + "?access_token=x")
    handler = publication._NoRedirectHandler()
    assert (
        handler.redirect_request(
            publication.urllib_request.Request(publication.ZENODO_DRAFT_API_URI),
            None,
            302,
            "redirect",
            {},
            publication.ZENODO_PUBLIC_API_URI,
        )
        is None
    )


class _HttpsResponse:
    def __init__(
        self,
        body: bytes,
        *,
        url: str = publication.ZENODO_DRAFT_API_URI,
        status: int = 200,
        content_type: str | None = "application/json; charset=utf-8",
        content_length: str | None = None,
    ) -> None:
        self._body = body
        self._url = url
        self.status = status
        self.headers: dict[str, str] = {"Content-Encoding": "identity"}
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        self.headers["Content-Length"] = content_length or str(len(body))

    def __enter__(self) -> _HttpsResponse:
        return self

    def __exit__(self, *_arguments: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._url

    def read(self, _maximum: int) -> bytes:
        return self._body


class _HttpsOpener:
    def __init__(self, response: _HttpsResponse) -> None:
        self.response = response
        self.requests: list[publication.urllib_request.Request] = []

    def open(
        self,
        request: publication.urllib_request.Request,
        *,
        timeout: float,
    ) -> _HttpsResponse:
        assert timeout == publication._TIMEOUT_SECONDS
        self.requests.append(request)
        return self.response


def test_https_transport_uses_bounded_json_put_with_exact_headers() -> None:
    response = _HttpsResponse(b'{"accepted":true}')
    opener = _HttpsOpener(response)
    token = bytearray(b"0123456789abcdef0123456789abcdef")
    transport = publication._ZenodoHttpsTransport(token)
    transport._opener = opener
    value = {"metadata": publication._fixed_protocol_metadata()}

    assert transport.put_json(publication.ZENODO_DRAFT_API_URI, value) == {"accepted": True}
    assert len(opener.requests) == 1
    request = opener.requests[0]
    assert request.full_url == publication.ZENODO_DRAFT_API_URI
    assert request.method == "PUT"
    assert request.data == publication._canonical_bytes(value)
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Content-length") == str(len(request.data))
    assert request.get_header("Authorization") == "Bearer " + token.decode("ascii")
    transport.close()


@pytest.mark.parametrize(
    ("response", "message"),
    (
        (_HttpsResponse(b"{}", content_type=None), "lacks Content-Type"),
        (_HttpsResponse(b"{}", content_type="text/plain"), "another Content-Type"),
        (_HttpsResponse(b"{}", status=202), "expected 200"),
        (
            _HttpsResponse(b"{}", url=publication.ZENODO_PUBLIC_API_URI),
            "redirects are forbidden",
        ),
        (
            _HttpsResponse(
                b"{}",
                content_length=str(publication._MAX_JSON_RESPONSE_BYTES + 1),
            ),
            "byte limit",
        ),
    ),
)
def test_https_metadata_put_rejects_response_boundary_drift(
    response: _HttpsResponse,
    message: str,
) -> None:
    transport = publication._ZenodoHttpsTransport(bytearray(b"0123456789abcdef0123456789abcdef"))
    transport._opener = _HttpsOpener(response)
    with pytest.raises(publication.ZenodoPublicationError, match=message):
        transport.put_json(
            publication.ZENODO_DRAFT_API_URI,
            {"metadata": publication._fixed_protocol_metadata()},
        )
    transport.close()


def test_cli_publish_guard_precedes_token_read(
    package: publication.ValidatedRegistrationPackage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publication, "validate_registration_package", lambda _path: package)
    assert (
        publication.main(
            ["publish", "--package", "/unused", "--confirm-record", "1", "--token-fd", "9999"]
        )
        == 2
    )
