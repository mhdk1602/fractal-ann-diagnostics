from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from fractal_ann_diagnostics.tlock_release_provenance import (
    QUICKNET_CHAIN_HASH,
    QUICKNET_GENESIS_UNIX_SECONDS,
    QUICKNET_PERIOD_SECONDS,
    QUICKNET_PUBLIC_KEY,
    QUICKNET_SCHEME_ID,
    TLOCK_BUILDER_IMAGE,
    TLOCK_DEPENDENCY_DELTA_SHA256,
    TLOCK_GO_LINUX_ARM64_TARBALL_SHA256,
    TLOCK_GO_LINUX_ARM64_TOOL_SHA256,
    TLOCK_LINUX_ARM64_BINARY_BYTE_COUNT,
    TLOCK_LINUX_ARM64_BINARY_SHA256,
    TLOCK_OFFICIAL_INTEROP_FIXTURE_ARCHIVE_SHA256,
    TLOCK_OFFICIAL_INTEROP_FIXTURE_ARCHIVE_URL,
    TLOCK_OFFICIAL_INTEROP_FIXTURE_BINARY_BYTE_COUNT,
    TLOCK_OFFICIAL_INTEROP_FIXTURE_BINARY_SHA256,
    TLOCK_PATCHED_GO_MOD_SHA256,
    TLOCK_PATCHED_GO_SUM_SHA256,
    TLOCK_RELEASE_TAG_OBJECT_GIT_SHA1,
    TLOCK_SOURCE_ARCHIVE_SHA256,
    TLOCK_SOURCE_ARCHIVE_URL,
    TLOCK_SOURCE_COMMIT_GIT_SHA1,
    TLOCK_SOURCE_TREE_MANIFEST_SHA256,
    TlockReleaseProvenance,
    TlockReleaseProvenanceError,
    freeze_tlock_release_provenance,
    load_tlock_release_provenance,
    loads_tlock_release_provenance,
    verify_tlock_release_binary,
    write_tlock_release_provenance,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _prefreeze() -> TlockReleaseProvenance:
    return TlockReleaseProvenance.prefreeze_quicknet_v1_2_0_linux_arm64()


def test_prefreeze_record_pins_release_lineage_binary_and_quicknet_without_round() -> None:
    provenance = _prefreeze()

    assert (
        TLOCK_PATCHED_GO_SUM_SHA256
        == "988aeb96a135d5fc3cf7cd0d755ffc4bbc28a84fb114ea385843010073cd1b3c"
    )
    assert provenance.release_tag_object_git_sha1 == TLOCK_RELEASE_TAG_OBJECT_GIT_SHA1
    assert provenance.source_commit_git_sha1 == TLOCK_SOURCE_COMMIT_GIT_SHA1
    assert provenance.target_operating_system == "linux"
    assert provenance.target_architecture == "arm64"
    assert provenance.source_archive_url == TLOCK_SOURCE_ARCHIVE_URL
    assert provenance.source_archive_sha256 == TLOCK_SOURCE_ARCHIVE_SHA256
    assert provenance.source_tree_manifest_sha256 == TLOCK_SOURCE_TREE_MANIFEST_SHA256
    assert provenance.builder_image == TLOCK_BUILDER_IMAGE
    assert provenance.go_linux_arm64_tarball_sha256 == TLOCK_GO_LINUX_ARM64_TARBALL_SHA256
    assert provenance.go_linux_arm64_tool_sha256 == TLOCK_GO_LINUX_ARM64_TOOL_SHA256
    assert provenance.patched_go_mod_sha256 == TLOCK_PATCHED_GO_MOD_SHA256
    assert provenance.patched_go_sum_sha256 == TLOCK_PATCHED_GO_SUM_SHA256
    assert provenance.dependency_delta_sha256 == TLOCK_DEPENDENCY_DELTA_SHA256
    assert provenance.binary_sha256 == TLOCK_LINUX_ARM64_BINARY_SHA256
    assert provenance.binary_byte_count == TLOCK_LINUX_ARM64_BINARY_BYTE_COUNT
    assert (
        provenance.official_interop_fixture_archive_url
        == TLOCK_OFFICIAL_INTEROP_FIXTURE_ARCHIVE_URL
    )
    assert (
        provenance.official_interop_fixture_archive_sha256
        == TLOCK_OFFICIAL_INTEROP_FIXTURE_ARCHIVE_SHA256
    )
    assert (
        provenance.official_interop_fixture_binary_sha256
        == TLOCK_OFFICIAL_INTEROP_FIXTURE_BINARY_SHA256
    )
    assert (
        provenance.official_interop_fixture_binary_byte_count
        == TLOCK_OFFICIAL_INTEROP_FIXTURE_BINARY_BYTE_COUNT
    )
    assert provenance.chain_hash == QUICKNET_CHAIN_HASH
    assert provenance.chain_scheme_id == QUICKNET_SCHEME_ID
    assert provenance.chain_period_seconds == QUICKNET_PERIOD_SECONDS
    assert provenance.chain_genesis_unix_seconds == QUICKNET_GENESIS_UNIX_SECONDS
    assert provenance.chain_public_key == QUICKNET_PUBLIC_KEY
    assert provenance.drand_round is None
    assert not provenance.is_frozen
    with pytest.raises(TlockReleaseProvenanceError, match="no frozen drand round"):
        provenance.require_frozen()


def test_checked_in_prefreeze_artifact_is_exact_canonical_record() -> None:
    path = REPOSITORY_ROOT / "research" / "tlock-release-provenance.prefreeze.json"
    observed = load_tlock_release_provenance(path)
    expected = _prefreeze()

    assert observed == expected
    assert path.read_bytes() == expected.canonical_file_bytes()


def test_round_can_be_bound_once_only_and_must_be_positive_integer() -> None:
    provenance = _prefreeze()
    frozen = freeze_tlock_release_provenance(provenance, drand_round=123_456_789)

    assert frozen.drand_round == 123_456_789
    assert frozen.is_frozen
    frozen.require_frozen()
    with pytest.raises(TlockReleaseProvenanceError, match="already frozen"):
        freeze_tlock_release_provenance(frozen, drand_round=123_456_790)
    for invalid in (True, 0, -1):
        with pytest.raises(TlockReleaseProvenanceError, match="drand_round"):
            freeze_tlock_release_provenance(provenance, drand_round=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "substitute"),
    (
        ("release_tag_object_git_sha1", "0" * 40),
        ("source_commit_git_sha1", "1" * 40),
        ("source_archive_url", "https://example.test/tlock.tar.gz"),
        ("source_archive_sha256", "2" * 64),
        ("patched_go_mod_sha256", "5" * 64),
        ("dependency_delta_sha256", "6" * 64),
        ("binary_sha256", "3" * 64),
        ("binary_byte_count", 13_303_933),
        ("official_interop_fixture_binary_sha256", "7" * 64),
        ("chain_hash", "4" * 64),
        ("chain_scheme_id", "alternate-scheme"),
        ("chain_period_seconds", 30),
        ("chain_genesis_unix_seconds", 1_692_803_368),
        ("chain_public_key", "aa"),
    ),
)
def test_verified_release_and_chain_pins_reject_substitution(
    field: str,
    substitute: object,
) -> None:
    payload = _prefreeze().to_dict()
    payload[field] = substitute

    with pytest.raises(TlockReleaseProvenanceError, match="verified release pin"):
        TlockReleaseProvenance.from_dict(payload)


def test_closed_loader_rejects_unknown_missing_duplicate_and_noncanonical_records() -> None:
    canonical = _prefreeze().canonical_file_bytes()
    payload = _prefreeze().to_dict()

    payload["unknown"] = "field"
    with pytest.raises(TlockReleaseProvenanceError, match="keys differ"):
        loads_tlock_release_provenance(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        )

    payload = _prefreeze().to_dict()
    del payload["binary_sha256"]
    with pytest.raises(TlockReleaseProvenanceError, match="keys differ"):
        loads_tlock_release_provenance(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        )

    duplicate = canonical.replace(
        b'{"binary_byte_count":',
        b'{"binary_byte_count":13303934,"binary_byte_count":',
        1,
    )
    with pytest.raises(TlockReleaseProvenanceError, match="repeats key"):
        loads_tlock_release_provenance(duplicate)

    with pytest.raises(TlockReleaseProvenanceError, match="not canonical"):
        loads_tlock_release_provenance(canonical[:-1])
    with pytest.raises(TlockReleaseProvenanceError, match="not canonical"):
        loads_tlock_release_provenance(b" " + canonical)


def test_writer_is_exclusive_and_binary_verifier_fails_closed(tmp_path: Path) -> None:
    provenance = _prefreeze()
    target = (tmp_path / "provenance.json").resolve()

    write_tlock_release_provenance(provenance, target)
    assert load_tlock_release_provenance(target) == provenance
    with pytest.raises(TlockReleaseProvenanceError, match="cannot publish"):
        write_tlock_release_provenance(provenance, target)

    substituted_binary = (tmp_path / "tle").resolve()
    substituted_binary.write_bytes(b"substituted")
    with pytest.raises(TlockReleaseProvenanceError, match="byte count differs"):
        verify_tlock_release_binary(provenance, substituted_binary)


def test_dataclass_rejects_boolean_round_even_if_constructed_directly() -> None:
    with pytest.raises(TlockReleaseProvenanceError, match="drand_round"):
        replace(_prefreeze(), drand_round=True)
