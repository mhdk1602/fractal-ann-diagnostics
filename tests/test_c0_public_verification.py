from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import fractal_ann_diagnostics.c0_public_verification as public_verification
from fractal_ann_diagnostics.c0_evidence_release import (
    C0_APPARATUS_EVIDENCE_SCHEMA,
    C0_CANDIDATE_ASSEMBLY_RECEIPT_ARCHIVE_MEMBER_PATH,
    C0_CANDIDATE_MANIFEST_ARCHIVE_MEMBER_PATH,
    C0_EVIDENCE_RELEASE_SCHEMA,
    C0_EVIDENCE_RELEASE_TAG,
    C0_EVIDENCE_RELEASE_URL,
    C0_EVIDENCE_REPOSITORY,
    C0_EVIDENCE_VERIFICATION_SCHEMA,
    canonical_apparatus_evidence_bytes,
    canonical_verification_bytes,
)
from fractal_ann_diagnostics.c0_public_verification import (
    C0_PUBLIC_VERIFICATION_SCHEMA,
    C0PublicVerificationError,
    build_c0_public_verification_receipt,
    load_c0_public_verification_receipt,
    main,
    write_c0_public_verification_receipt,
)

COMMIT = "1" * 40
TAG_REF = f"refs/tags/{C0_EVIDENCE_RELEASE_TAG}"


def _canonical(value: object) -> bytes:
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


def _write(path: Path, encoded: bytes, *, mode: int = 0o600) -> Path:
    path.write_bytes(encoded)
    path.chmod(mode)
    return path


def _apparatus() -> dict[str, object]:
    return {
        "build_context_tree_sha256": "b" * 64,
        "c0_commit": COMMIT,
        "candidate_bootstrap_closure_sha256": "c" * 64,
        "candidate_image_closure_sha256": "d" * 64,
        "candidate_image_run_id": 101,
        "candidate_image_source_commit": "2" * 40,
        "candidate_manifest_archive_member_path": C0_CANDIDATE_MANIFEST_ARCHIVE_MEMBER_PATH,
        "candidate_manifest_assembly_receipt_archive_member_path": (
            C0_CANDIDATE_ASSEMBLY_RECEIPT_ARCHIVE_MEMBER_PATH
        ),
        "candidate_manifest_assembly_receipt_file_sha256": "a" * 64,
        "candidate_manifest_file_sha256": "9" * 64,
        "github_environment_control_receipt_file_sha256": "8" * 64,
        "oci_promotion_receipt_sha256": "e" * 64,
        "production_image_run_id": 202,
        "production_control_instantiation_receipt_file_sha256": "7" * 64,
        "provider_phase_plan_closure_sha256": "f" * 64,
        "provider_rehearsal_gate_sha256": "1" * 64,
        "provider_rehearsal_receipt_sha256": "2" * 64,
        "provider_rehearsal_run_id": 303,
        "rehearsal_attestation_verification_sha256": "3" * 64,
        "rehearsal_manifest_sha256": "4" * 64,
        "release_image_index_digest": f"sha256:{'5' * 64}",
        "schema_version": C0_APPARATUS_EVIDENCE_SCHEMA,
        "scientific_image_index_digest": f"sha256:{'6' * 64}",
    }


def _gh_result(
    *,
    asset_name: str,
    asset_sha256: str,
    checksum_name: str,
    checksum_sha256: str,
    marker: str,
) -> dict[str, object]:
    return {
        "attestation": {
            "bundle": {
                "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
                "marker": marker,
            },
            "bundleURL": f"https://api.github.com/attestations/{marker}",
        },
        "verificationResult": {
            "mediaType": ("application/vnd.dev.sigstore.verificationresult+json;version=0.1"),
            "statement": {
                "_type": "https://in-toto.io/Statement/v1",
                "predicateType": "https://github.com/Attestations/GitHubRelease/v1",
                "subject": [
                    {"digest": {"sha1": COMMIT}, "name": ""},
                    {"digest": {"sha256": asset_sha256}, "name": asset_name},
                    {"digest": {"sha256": checksum_sha256}, "name": checksum_name},
                ],
            },
            "verifiedTimestamps": [
                {
                    "timestamp": "2026-07-18T17:00:00Z",
                    "type": "TSA",
                    "uri": "https://timestamp.github.com",
                }
            ],
        },
    }


@dataclass
class EvidenceCase:
    root: Path
    binding: dict[str, object]
    release_api: dict[str, object]
    release_gh: dict[str, object]
    asset_gh: dict[str, object]
    binding_path: Path
    gh_version_path: Path
    release_api_path: Path
    tag_path: Path
    release_gh_path: Path
    asset_gh_path: Path
    archive_path: Path
    checksum_path: Path
    output_dir: Path

    def kwargs(self) -> dict[str, object]:
        return {
            "binding_path": self.binding_path,
            "c0_commit": COMMIT,
            "gh_version_path": self.gh_version_path,
            "release_api_path": self.release_api_path,
            "tag_ls_remote_path": self.tag_path,
            "release_verification_path": self.release_gh_path,
            "asset_verification_path": self.asset_gh_path,
            "archive_path": self.archive_path,
            "checksum_path": self.checksum_path,
        }

    def write_binding(self) -> None:
        _write(self.binding_path, _canonical(self.binding))

    def write_release_api(self, *, canonical: bool = True) -> None:
        encoded = (
            _canonical(self.release_api)
            if canonical
            else (json.dumps(self.release_api, indent=2) + "\n").encode("utf-8")
        )
        _write(self.release_api_path, encoded)

    def write_release_gh(self, *, pretty: bool = False) -> None:
        encoded = (
            (json.dumps(self.release_gh, indent=2) + "\n").encode("utf-8")
            if pretty
            else _canonical(self.release_gh)
        )
        _write(self.release_gh_path, encoded)

    def write_asset_gh(self, *, pretty: bool = False) -> None:
        encoded = (
            (json.dumps(self.asset_gh, indent=2) + "\n").encode("utf-8")
            if pretty
            else _canonical(self.asset_gh)
        )
        _write(self.asset_gh_path, encoded)


@pytest.fixture
def evidence(tmp_path: Path) -> EvidenceCase:
    archive_bytes = b"closed C0 evidence archive\n"
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    asset_name = f"fractal-ann-diagnostics-c0-evidence-{COMMIT}.tar.gz"
    checksum_name = f"{asset_name}.sha256"
    checksum_bytes = f"{archive_sha256}  {asset_name}\n".encode("ascii")
    checksum_sha256 = hashlib.sha256(checksum_bytes).hexdigest()
    asset_url = (
        f"https://github.com/{C0_EVIDENCE_REPOSITORY}/releases/download/"
        f"{C0_EVIDENCE_RELEASE_TAG}/{asset_name}"
    )
    release_id = 90210
    release_api: dict[str, object] = {
        "assets": [
            {
                "browser_download_url": asset_url,
                "digest": f"sha256:{archive_sha256}",
                "download_count": 7,
                "id": 901,
                "name": asset_name,
                "size": len(archive_bytes),
                "state": "uploaded",
                "url": (
                    f"https://api.github.com/repos/{C0_EVIDENCE_REPOSITORY}/releases/assets/901"
                ),
            },
            {
                "browser_download_url": f"{asset_url}.sha256",
                "digest": f"sha256:{checksum_sha256}",
                "download_count": 5,
                "id": 902,
                "name": checksum_name,
                "size": len(checksum_bytes),
                "state": "uploaded",
                "url": (
                    f"https://api.github.com/repos/{C0_EVIDENCE_REPOSITORY}/releases/assets/902"
                ),
            },
        ],
        "assets_url": (
            f"https://api.github.com/repos/{C0_EVIDENCE_REPOSITORY}/releases/{release_id}/assets"
        ),
        "draft": False,
        "html_url": C0_EVIDENCE_RELEASE_URL,
        "id": release_id,
        "immutable": True,
        "name": "Confirmatory apparatus C0 evidence",
        "prerelease": False,
        "published_at": "2026-07-18T17:00:00Z",
        "tag_name": C0_EVIDENCE_RELEASE_TAG,
        "target_commitish": COMMIT,
        "url": (f"https://api.github.com/repos/{C0_EVIDENCE_REPOSITORY}/releases/{release_id}"),
    }
    release_gh = _gh_result(
        asset_name=asset_name,
        asset_sha256=archive_sha256,
        checksum_name=checksum_name,
        checksum_sha256=checksum_sha256,
        marker="release",
    )
    asset_gh = _gh_result(
        asset_name=asset_name,
        asset_sha256=archive_sha256,
        checksum_name=checksum_name,
        checksum_sha256=checksum_sha256,
        marker="asset",
    )
    verification: dict[str, object] = {
        "anonymous_asset_sha256": archive_sha256,
        "anonymous_asset_size": len(archive_bytes),
        "anonymous_checksum_sha256": checksum_sha256,
        "anonymous_checksum_size": len(checksum_bytes),
        # These are the prior C0 observations.  Fresh C1 observations are independent.
        "asset_attestation_output_sha256": "8" * 64,
        "asset_attestation_verified": True,
        "release_api_output_sha256": "9" * 64,
        "release_attestation_output_sha256": "a" * 64,
        "release_attestation_verified": True,
        "release_tag_readback_sha256": "b" * 64,
        "release_tag_target_commit": COMMIT,
        "release_tag_target_verified": True,
        "schema_version": C0_EVIDENCE_VERIFICATION_SCHEMA,
    }
    apparatus = _apparatus()
    binding: dict[str, object] = {
        "apparatus_evidence": apparatus,
        "apparatus_evidence_sha256": hashlib.sha256(
            canonical_apparatus_evidence_bytes(apparatus)
        ).hexdigest(),
        "asset_name": asset_name,
        "asset_sha256": archive_sha256,
        "asset_size": len(archive_bytes),
        "asset_url": asset_url,
        "checksum_asset_name": checksum_name,
        "checksum_asset_sha256": checksum_sha256,
        "checksum_asset_size": len(checksum_bytes),
        "checksum_asset_url": f"{asset_url}.sha256",
        "immutable_release": True,
        "release_tag": C0_EVIDENCE_RELEASE_TAG,
        "release_url": C0_EVIDENCE_RELEASE_URL,
        "repository": C0_EVIDENCE_REPOSITORY,
        "schema_version": C0_EVIDENCE_RELEASE_SCHEMA,
        "target_commit": COMMIT,
        "verification_receipt": verification,
        "verification_receipt_sha256": hashlib.sha256(
            canonical_verification_bytes(verification)
        ).hexdigest(),
    }
    binding_path = _write(tmp_path / "c0-evidence-release-binding.json", _canonical(binding))
    gh_version_path = _write(
        tmp_path / "gh-version.txt",
        b"gh version 2.96.0 (2026-07-02)\nhttps://github.com/cli/cli/releases/tag/v2.96.0\n",
    )
    release_api_path = _write(tmp_path / "published-release-api.json", _canonical(release_api))
    tag_path = _write(tmp_path / "published-tag-ls-remote.txt", f"{COMMIT}\t{TAG_REF}\n".encode())
    release_gh_path = _write(tmp_path / "release-verification.json", _canonical(release_gh))
    asset_gh_path = _write(tmp_path / "asset-verification.json", _canonical(asset_gh))
    archive_path = _write(tmp_path / asset_name, archive_bytes)
    checksum_path = _write(tmp_path / checksum_name, checksum_bytes)
    output_dir = tmp_path / "private-receipts"
    output_dir.mkdir(mode=0o700)
    output_dir.chmod(0o700)
    return EvidenceCase(
        root=tmp_path,
        binding=binding,
        release_api=release_api,
        release_gh=release_gh,
        asset_gh=asset_gh,
        binding_path=binding_path,
        gh_version_path=gh_version_path,
        release_api_path=release_api_path,
        tag_path=tag_path,
        release_gh_path=release_gh_path,
        asset_gh_path=asset_gh_path,
        archive_path=archive_path,
        checksum_path=checksum_path,
        output_dir=output_dir,
    )


def test_build_write_and_reload_closed_receipt(evidence: EvidenceCase) -> None:
    receipt = build_c0_public_verification_receipt(**evidence.kwargs())
    assert receipt.schema_version == C0_PUBLIC_VERIFICATION_SCHEMA
    assert receipt.binding_source_kind == "c0-binding"
    assert receipt.tag_kind == "lightweight"
    assert receipt.archive_sha256 == evidence.binding["asset_sha256"]
    assert receipt.release_verification == evidence.release_gh
    output = evidence.output_dir / "c0-public-verification.json"
    assert write_c0_public_verification_receipt(receipt, output) == output
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert output.read_bytes() == receipt.canonical_file_bytes()
    assert load_c0_public_verification_receipt(output) == receipt


def test_fresh_semantic_outputs_need_not_match_prior_v2_raw_hashes(
    evidence: EvidenceCase,
) -> None:
    old = evidence.binding["verification_receipt"]
    assert isinstance(old, dict)
    evidence.release_api["body"] = "fresh API projection"
    assets = evidence.release_api["assets"]
    assert isinstance(assets, list) and isinstance(assets[0], dict)
    assets[0]["download_count"] = 999
    evidence.write_release_api()
    evidence.release_gh["attestation"]["freshProviderField"] = "retained"  # type: ignore[index]
    evidence.write_release_gh(pretty=True)
    evidence.asset_gh["attestation"]["freshProviderField"] = "retained"  # type: ignore[index]
    evidence.write_asset_gh(pretty=True)

    receipt = build_c0_public_verification_receipt(**evidence.kwargs())

    assert receipt.release_api_file_sha256 != old["release_api_output_sha256"]
    assert receipt.release_verification_file_sha256 != old["release_attestation_output_sha256"]
    assert receipt.asset_verification_file_sha256 != old["asset_attestation_output_sha256"]
    assert receipt.release_api["body"] == "fresh API projection"


def test_annotated_tag_is_admitted_by_its_peeled_commit(evidence: EvidenceCase) -> None:
    tag_object = "0" * 40
    rows = sorted(
        [
            f"{tag_object}\t{TAG_REF}",
            f"{COMMIT}\t{TAG_REF}^{{}}",
        ]
    )
    _write(evidence.tag_path, ("\n".join(rows) + "\n").encode("ascii"))
    receipt = build_c0_public_verification_receipt(**evidence.kwargs())
    assert receipt.tag_kind == "annotated"
    assert (
        receipt.tag_ls_remote_file_sha256
        != evidence.binding["verification_receipt"][  # type: ignore[index]
            "release_tag_readback_sha256"
        ]
    )


def test_frozen_manifest_source_is_fully_validated(evidence: EvidenceCase) -> None:
    from test_study import _manifest

    manifest = _manifest(frozen=True)
    sealed = manifest["sealed_execution"]
    assert isinstance(sealed, dict)
    sealed["c0_evidence_release"] = copy.deepcopy(evidence.binding)
    manifest_path = _write(evidence.root / "study-manifest.json", _canonical(manifest))
    inputs = evidence.kwargs()
    inputs.pop("binding_path")
    inputs.pop("c0_commit")
    inputs["manifest_path"] = manifest_path
    receipt = build_c0_public_verification_receipt(**inputs)
    assert receipt.binding_source_kind == "frozen-manifest"


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("tag_name", "other-tag"),
        ("target_commitish", "2" * 40),
        ("html_url", "https://github.com/example/forged"),
        ("name", "Forged title"),
        ("draft", True),
        ("prerelease", True),
        ("immutable", False),
    ],
)
def test_release_identity_forgery_is_rejected(
    evidence: EvidenceCase,
    field: str,
    forged: object,
) -> None:
    evidence.release_api[field] = forged
    evidence.write_release_api()
    with pytest.raises(C0PublicVerificationError, match=field):
        build_c0_public_verification_receipt(**evidence.kwargs())


@pytest.mark.parametrize("forgery", ["missing", "digest", "size", "url", "extra"])
def test_release_asset_forgery_is_rejected(evidence: EvidenceCase, forgery: str) -> None:
    assets = evidence.release_api["assets"]
    assert isinstance(assets, list) and isinstance(assets[0], dict)
    if forgery == "missing":
        assets.pop()
    elif forgery == "digest":
        assets[0]["digest"] = f"sha256:{'0' * 64}"
    elif forgery == "size":
        assets[0]["size"] = int(assets[0]["size"]) + 1
    elif forgery == "url":
        assets[0]["browser_download_url"] = "https://example.com/substitution"
    else:
        assets.append(copy.deepcopy(assets[0]))
    evidence.write_release_api()
    with pytest.raises(C0PublicVerificationError, match="asset"):
        build_c0_public_verification_receipt(**evidence.kwargs())


@pytest.mark.parametrize(
    "forgery",
    ["array", "extra-row", "empty-attestation", "missing-checksum", "duplicate-subject"],
)
def test_gh_release_output_requires_the_pinned_unambiguous_shape(
    evidence: EvidenceCase,
    forgery: str,
) -> None:
    if forgery == "array":
        encoded = _canonical([evidence.release_gh])
    elif forgery == "extra-row":
        encoded = _canonical([evidence.release_gh, evidence.release_gh])
    elif forgery == "empty-attestation":
        evidence.release_gh["attestation"] = {}
        encoded = _canonical(evidence.release_gh)
    else:
        result = evidence.release_gh["verificationResult"]
        assert isinstance(result, dict)
        statement = result["statement"]
        assert isinstance(statement, dict)
        subjects = statement["subject"]
        assert isinstance(subjects, list)
        if forgery == "missing-checksum":
            subjects.pop()
        else:
            subjects.append(copy.deepcopy(subjects[1]))
        encoded = _canonical(evidence.release_gh)
    _write(evidence.release_gh_path, encoded)
    with pytest.raises(C0PublicVerificationError):
        build_c0_public_verification_receipt(**evidence.kwargs())


def test_gh_asset_output_must_bind_the_archive(evidence: EvidenceCase) -> None:
    result = evidence.asset_gh["verificationResult"]
    assert isinstance(result, dict)
    statement = result["statement"]
    assert isinstance(statement, dict)
    subjects = statement["subject"]
    assert isinstance(subjects, list) and isinstance(subjects[1], dict)
    subjects[1]["digest"] = {"sha256": "0" * 64}
    evidence.write_asset_gh()
    with pytest.raises(C0PublicVerificationError, match="archive"):
        build_c0_public_verification_receipt(**evidence.kwargs())


@pytest.mark.parametrize(
    "encoded",
    [
        f"{'2' * 40}\t{TAG_REF}\n".encode(),
        f"{COMMIT} {TAG_REF}\n".encode(),
        f"{COMMIT}\t{TAG_REF}\n{COMMIT}\t{TAG_REF}\n".encode(),
        f"{'2' * 40}\t{TAG_REF}\n{'3' * 40}\t{TAG_REF}^{{}}\n".encode(),
        f"{COMMIT}\t{TAG_REF}\n{'2' * 40}\trefs/tags/other\n".encode(),
    ],
)
def test_forged_tag_rows_are_rejected(evidence: EvidenceCase, encoded: bytes) -> None:
    _write(evidence.tag_path, encoded)
    with pytest.raises(C0PublicVerificationError, match="tag"):
        build_c0_public_verification_receipt(**evidence.kwargs())


def test_anonymous_archive_substitution_is_rejected(evidence: EvidenceCase) -> None:
    _write(evidence.archive_path, b"substituted archive bytes\n")
    with pytest.raises(C0PublicVerificationError, match="archive"):
        build_c0_public_verification_receipt(**evidence.kwargs())


def test_checksum_substitution_is_rejected(evidence: EvidenceCase) -> None:
    _write(
        evidence.checksum_path,
        f"{'0' * 64}  {evidence.archive_path.name}\n".encode("ascii"),
    )
    with pytest.raises(C0PublicVerificationError, match="checksum"):
        build_c0_public_verification_receipt(**evidence.kwargs())


def test_noncanonical_release_api_is_rejected(evidence: EvidenceCase) -> None:
    evidence.write_release_api(canonical=False)
    with pytest.raises(C0PublicVerificationError, match="canonical"):
        build_c0_public_verification_receipt(**evidence.kwargs())


def test_duplicate_json_key_is_rejected(evidence: EvidenceCase) -> None:
    _write(
        evidence.release_gh_path,
        b'{"attestation":{},"attestation":{},"verificationResult":{}}\n',
    )
    with pytest.raises(C0PublicVerificationError, match="repeats key"):
        build_c0_public_verification_receipt(**evidence.kwargs())


def test_symlink_and_hardlink_inputs_are_rejected(evidence: EvidenceCase) -> None:
    original = evidence.root / "release-api-original.json"
    evidence.release_api_path.rename(original)
    evidence.release_api_path.symlink_to(original)
    with pytest.raises(C0PublicVerificationError, match="symlink"):
        build_c0_public_verification_receipt(**evidence.kwargs())

    evidence.release_api_path.unlink()
    os.link(original, evidence.release_api_path)
    with pytest.raises(C0PublicVerificationError, match="singly linked"):
        build_c0_public_verification_receipt(**evidence.kwargs())


def test_group_writable_input_is_rejected(evidence: EvidenceCase) -> None:
    evidence.release_gh_path.chmod(0o620)
    with pytest.raises(C0PublicVerificationError, match="writable by group"):
        build_c0_public_verification_receipt(**evidence.kwargs())


@pytest.mark.parametrize(
    "encoded",
    [
        b"gh version 2.95.0 (2026-06-18)\nhttps://github.com/cli/cli/releases/tag/v2.95.0\n",
        b"gh version 2.96.0 (not-a-date)\nhttps://github.com/cli/cli/releases/tag/v2.96.0\n",
        b"gh version 2.96.0 (2026-07-02)\nhttps://example.com/forged\n",
        b"gh version 2.96.0 (2026-07-02)\nhttps://github.com/cli/cli/releases/tag/v2.96.0\nextra\n",
    ],
)
def test_gh_version_transcript_is_closed_and_pinned(
    evidence: EvidenceCase,
    encoded: bytes,
) -> None:
    _write(evidence.gh_version_path, encoded)
    with pytest.raises(C0PublicVerificationError, match="gh version"):
        build_c0_public_verification_receipt(**evidence.kwargs())


def test_output_is_private_and_no_replace(evidence: EvidenceCase) -> None:
    receipt = build_c0_public_verification_receipt(**evidence.kwargs())
    output = evidence.output_dir / "receipt.json"
    output.write_text("sentinel", encoding="utf-8")
    with pytest.raises(C0PublicVerificationError, match="already exists"):
        write_c0_public_verification_receipt(receipt, output)
    assert output.read_text(encoding="utf-8") == "sentinel"

    evidence.output_dir.chmod(0o755)
    with pytest.raises(C0PublicVerificationError, match="0700"):
        write_c0_public_verification_receipt(receipt, evidence.output_dir / "other.json")


def test_input_change_during_final_revalidation_is_rejected(
    evidence: EvidenceCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = public_verification._OpenSnapshot.revalidate
    changed = False

    def mutate_then_revalidate(snapshot: Any) -> None:
        nonlocal changed
        if snapshot.label == "release API output" and not changed:
            changed = True
            snapshot.path.write_bytes(snapshot.path.read_bytes() + b" ")
            snapshot.path.chmod(0o600)
        original(snapshot)

    monkeypatch.setattr(
        public_verification._OpenSnapshot,
        "revalidate",
        mutate_then_revalidate,
    )
    with pytest.raises(C0PublicVerificationError, match="changed"):
        build_c0_public_verification_receipt(**evidence.kwargs())


def test_receipt_raw_gh_text_and_hash_cannot_be_decoupled(evidence: EvidenceCase) -> None:
    receipt = build_c0_public_verification_receipt(**evidence.kwargs())
    value = receipt.to_dict()
    value["release_verification_text"] = value["release_verification_text"] + " "
    with pytest.raises(C0PublicVerificationError, match="raw hash"):
        public_verification.C0PublicVerificationReceipt.from_dict(value)


def test_cli_writes_receipt_and_requires_binding_commit(
    evidence: EvidenceCase,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = evidence.output_dir / "cli-receipt.json"
    arguments = [
        "--binding",
        str(evidence.binding_path),
        "--c0-commit",
        COMMIT,
        "--gh-version",
        str(evidence.gh_version_path),
        "--release-api",
        str(evidence.release_api_path),
        "--tag-ls-remote",
        str(evidence.tag_path),
        "--release-verification",
        str(evidence.release_gh_path),
        "--asset-verification",
        str(evidence.asset_gh_path),
        "--archive",
        str(evidence.archive_path),
        "--checksum",
        str(evidence.checksum_path),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema_version"] == C0_PUBLIC_VERIFICATION_SCHEMA
    assert output.exists()

    without_commit = arguments.copy()
    position = without_commit.index("--c0-commit")
    del without_commit[position : position + 2]
    with pytest.raises(SystemExit, match="2"):
        main(without_commit)
