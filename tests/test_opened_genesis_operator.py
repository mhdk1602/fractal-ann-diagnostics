from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fractal_ann_diagnostics.study import FIXED_CORPORA
from fractal_ann_diagnostics.suite_attempt import SuiteAttemptError

_OPERATOR_PATH = Path(__file__).parents[1] / "operators" / "opened_genesis.py"
_SPEC = importlib.util.spec_from_file_location("opened_genesis_operator", _OPERATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
operator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = operator
_SPEC.loader.exec_module(operator)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )


def _write(path: Path, value: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(value)
    path.chmod(mode)


def _publish_namespace(
    fixture: SimpleNamespace,
    descriptor: object,
    *,
    descriptor_bytes: bytes | None = None,
    descriptor_mode: int = 0o600,
    include_state: bool = True,
) -> None:
    fixture.namespace.mkdir(mode=0o700)
    (fixture.namespace / "online").mkdir(mode=0o700)
    _write(
        fixture.namespace / "attestation-descriptor.json",
        (
            descriptor.canonical_bytes() + b"\n"  # type: ignore[union-attr]
            if descriptor_bytes is None
            else descriptor_bytes
        ),
        mode=descriptor_mode,
    )
    if include_state:
        _write(fixture.namespace / "000.state.json", b"{}\n")


def _descriptor_bytes() -> bytes:
    return _canonical(
        {
            "expected_git_ref": "refs/tags/confirmatory-apparatus-c0",
            "expected_oidc_issuer": "https://token.actions.githubusercontent.com",
            "expected_repository": "mhdk1602/fractal-ann-diagnostics",
            "expected_signer_digest": "1" * 40,
            "expected_signer_identity": "confirmatory-suite-state",
            "expected_workflow": ".github/workflows/confirmatory-suite-state.yml",
            "schema_version": "fractal-suite-attestation-descriptor-v1",
            "state_key_prefix": "confirmatory/v0.3/",
            "state_service_identity": "github-git-data",
            "state_service_uri": "https://api.github.com/repos/mhdk1602/fractal-ann-diagnostics/git",
            "timestamp_authority_identity": "rekor-signed-entry-timestamp",
            "timestamp_authority_public_key_sha256": "2" * 64,
            "timestamp_authority_uri": "https://rekor.sigstore.dev/api/v1/log",
            "transparency_log_identity": "sigstore-rekor",
            "transparency_log_public_key_sha256": "3" * 64,
            "transparency_log_uri": "https://rekor.sigstore.dev/api/v1/log",
        }
    )


def _manifest_bytes() -> bytes:
    return _canonical(
        {
            "production_workloads": [
                {
                    "corpus_id": corpus_id,
                    "spec": {
                        "online_execution_plan_sha256": hashlib.sha256(
                            f"execution:{corpus_id}".encode()
                        ).hexdigest()
                    },
                }
                for corpus_id in FIXED_CORPORA
            ]
        }
    )


def _request_fixture(tmp_path: Path) -> SimpleNamespace:
    input_root = tmp_path / "inputs"
    package_root = input_root / "c1-package"
    paths: dict[str, Path] = {}
    values: dict[str, bytes] = {}
    for position, role in enumerate(operator._expected_roles()):
        if role.startswith(operator._PACKAGE_PREFIX):
            name = role.removeprefix(operator._PACKAGE_PREFIX)
            path = package_root / name
        else:
            path = input_root / "controls" / f"{position:03d}.json"
        if role == f"{operator._PACKAGE_PREFIX}study-manifest.json":
            value = _manifest_bytes()
        elif role == "suite-attestation-descriptor":
            value = _descriptor_bytes()
        else:
            value = _canonical({"fixture_role": role})
        _write(path, value)
        paths[role] = path
        values[role] = value

    bindings = tuple(
        operator.InputBinding(
            role=role,
            path=paths[role],
            file_sha256=_digest(values[role]),
        )
        for role in operator._expected_roles()
    )
    request = operator.OpenedGenesisRequest(inputs=bindings)
    request_path = tmp_path / "opened-genesis-request.json"
    _write(request_path, request.canonical_file_bytes())
    output_parent = tmp_path / "suite"
    output_parent.mkdir(mode=0o700)
    output_parent.chmod(0o700)
    suite_id = "4" * 64
    namespace = output_parent / f"suite-attempt-{suite_id}"
    return SimpleNamespace(
        bindings=bindings,
        namespace=namespace,
        package_root=package_root,
        paths=paths,
        request=request,
        request_path=request_path,
        request_sha256=_digest(request.canonical_file_bytes()),
        suite_id=suite_id,
        values=values,
    )


def _attempt_marker_path(fixture: SimpleNamespace) -> Path:
    return fixture.namespace.parent / f".opened-genesis-{fixture.suite_id}.attempted"


def _quarantine_path(fixture: SimpleNamespace) -> Path:
    return fixture.namespace.parent / f".opened-genesis-{fixture.suite_id}.quarantine"


def _attempt_marker_bytes(fixture: SimpleNamespace) -> bytes:
    return _canonical(
        {
            "attestation_descriptor_file_sha256": _digest(
                fixture.values["suite-attestation-descriptor"]
            ),
            "request_sha256": fixture.request_sha256,
            "schema_version": operator._ATTEMPT_MARKER_SCHEMA,
            "state": "ATTEMPTED",
            "suite_attempt_id": fixture.suite_id,
        }
    )


def _patch_typed_apparatus(
    monkeypatch: pytest.MonkeyPatch,
    fixture: SimpleNamespace,
    *,
    opener: object | None = None,
) -> SimpleNamespace:
    manifest_digest = "5" * 64
    registration = SimpleNamespace(manifest_sha256=manifest_digest)
    finalization_receipt = SimpleNamespace(
        canonical_suite_namespace=str(fixture.namespace),
        finalization_request_sha256=_digest(fixture.values["production-finalization-request"]),
        suite_attempt_id=fixture.suite_id,
    )
    finalization_request = SimpleNamespace(
        c1_package_root=fixture.package_root,
        protocol_registry_record_path=fixture.paths["protocol-registry-record"],
        protocol_registration_receipt_path=fixture.paths["protocol-registration-receipt"],
        sealed_run_receipt_path=fixture.paths["sealed-run-receipt"],
    )
    corpus_by_path = {
        fixture.paths[f"{corpus_id}/{suffix}"]: corpus_id
        for corpus_id in FIXED_CORPORA
        for suffix in operator._CORPUS_ROLE_SUFFIXES
    }

    monkeypatch.setattr(
        operator,
        "verify_production_protocol_registration",
        lambda *args, **kwargs: registration,
    )
    monkeypatch.setattr(
        operator,
        "load_production_control_finalization_receipt",
        lambda *args, **kwargs: finalization_receipt,
    )
    monkeypatch.setattr(
        operator,
        "load_production_control_finalization_request",
        lambda *args, **kwargs: finalization_request,
    )
    monkeypatch.setattr(
        operator,
        "load_preflight_launch_contract",
        lambda path: SimpleNamespace(
            geometry=SimpleNamespace(corpus_id=corpus_by_path[Path(path)])
        ),
    )
    monkeypatch.setattr(
        operator,
        "load_runtime_preflight_receipt",
        lambda path: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        operator,
        "load_runtime_plan_transition",
        lambda path: SimpleNamespace(corpus_id=corpus_by_path[Path(path)]),
    )
    monkeypatch.setattr(
        operator,
        "load_registered_plan_instantiation",
        lambda path: SimpleNamespace(corpus_id=corpus_by_path[Path(path)]),
    )
    monkeypatch.setattr(
        operator,
        "load_sealed_launch_contract",
        lambda path: SimpleNamespace(
            geometry=SimpleNamespace(corpus_id=corpus_by_path[Path(path)])
        ),
    )
    monkeypatch.setattr(
        operator,
        "verify_production_run_closure_authority",
        lambda **kwargs: SimpleNamespace(binding=kwargs),
    )

    record = SimpleNamespace(
        canonical_bytes=lambda: b"{}",
        manifest_sha256=manifest_digest,
        namespace_uri=fixture.namespace.as_uri(),
        previous_state_record_sha256=None,
        record_sha256="6" * 64,
        sequence=0,
        state="OPENED",
        suite_attempt_id=fixture.suite_id,
    )
    monkeypatch.setattr(operator, "load_suite_state_record", lambda path: record)

    def successful_open(manifest: object, **kwargs: object) -> Path:
        del manifest
        fixture.namespace.mkdir(mode=0o700)
        (fixture.namespace / "online").mkdir(mode=0o700)
        _write(
            fixture.namespace / "attestation-descriptor.json",
            kwargs["attestation_descriptor"].canonical_bytes() + b"\n",  # type: ignore[union-attr]
        )
        _write(fixture.namespace / "000.state.json", b"{}\n")
        return fixture.namespace

    monkeypatch.setattr(
        operator,
        "open_suite_attempt",
        successful_open if opener is None else opener,
    )
    return SimpleNamespace(
        finalization_receipt=finalization_receipt,
        finalization_request=finalization_request,
        manifest_digest=manifest_digest,
        record=record,
        registration=registration,
    )


def test_execute_derives_open_arguments_and_publishes_exact_genesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    captured: dict[str, object] = {}

    def opener(manifest: object, **kwargs: object) -> Path:
        captured["manifest"] = manifest
        captured.update(kwargs)
        fixture.namespace.mkdir(mode=0o700)
        (fixture.namespace / "online").mkdir(mode=0o700)
        descriptor = kwargs["attestation_descriptor"]
        _write(
            fixture.namespace / "attestation-descriptor.json",
            descriptor.canonical_bytes() + b"\n",  # type: ignore[union-attr]
        )
        _write(fixture.namespace / "000.state.json", b"{}\n")
        return fixture.namespace

    patched = _patch_typed_apparatus(monkeypatch, fixture, opener=opener)
    result = operator.execute(
        fixture.request_path,
        expected_request_sha256=fixture.request_sha256,
    )

    assert result == {
        "manifest_sha256": patched.manifest_digest,
        "namespace": str(fixture.namespace),
        "request_sha256": fixture.request_sha256,
        "schema_version": "fractal-opened-genesis-result-v1",
        "state": "OPENED",
        "state_record_sha256": "6" * 64,
        "suite_attempt_id": fixture.suite_id,
    }
    expected_artifacts = {
        corpus_id: hashlib.sha256(f"execution:{corpus_id}".encode()).hexdigest()
        for corpus_id in FIXED_CORPORA
    }
    assert captured["execution_artifacts"] == expected_artifacts
    assert captured["verified_protocol_registration"] is patched.registration
    assert captured["run_receipt_path"] == fixture.paths["sealed-run-receipt"]
    assert set(captured["verified_production_closures"]) == set(FIXED_CORPORA)
    assert stat.S_IMODE(fixture.namespace.stat().st_mode) == 0o700
    assert stat.S_IMODE((fixture.namespace / "online").stat().st_mode) == 0o700
    assert stat.S_IMODE((fixture.namespace / "attestation-descriptor.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((fixture.namespace / "000.state.json").stat().st_mode) == 0o600
    assert list(fixture.namespace.joinpath("online").iterdir()) == []
    lock = fixture.namespace.parent / f".opened-genesis-{fixture.suite_id}.lock"
    assert lock.read_bytes() == b""
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600
    marker = _attempt_marker_path(fixture)
    assert marker.read_bytes() == _attempt_marker_bytes(fixture)
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600


def test_request_and_output_may_share_one_private_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    fixture.namespace = tmp_path / f"suite-attempt-{fixture.suite_id}"
    _patch_typed_apparatus(monkeypatch, fixture)

    result = operator.execute(
        fixture.request_path,
        expected_request_sha256=fixture.request_sha256,
    )

    assert result["namespace"] == str(fixture.namespace)
    assert fixture.namespace.is_dir()
    assert _attempt_marker_path(fixture).read_bytes() == _attempt_marker_bytes(fixture)


def test_request_rejects_any_scientific_override(tmp_path: Path) -> None:
    fixture = _request_fixture(tmp_path)
    value = json.loads(fixture.request.canonical_file_bytes())
    value["execution_artifacts"] = {corpus_id: "7" * 64 for corpus_id in FIXED_CORPORA}

    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="unknown=\\['execution_artifacts'\\]",
    ):
        operator.OpenedGenesisRequest.from_bytes(_canonical(value))


def test_finalization_suite_id_must_be_one_lowercase_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    patched = _patch_typed_apparatus(monkeypatch, fixture)
    patched.finalization_receipt.suite_attempt_id = "../not-a-suite-id"

    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="suite attempt ID must be one lowercase SHA-256 digest",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert not fixture.namespace.exists()


def test_request_rejects_role_substitution_and_path_reuse(tmp_path: Path) -> None:
    fixture = _request_fixture(tmp_path)
    bindings = list(fixture.bindings)
    first = bindings[0]
    bindings[0] = operator.InputBinding(
        role="substituted-role",
        path=first.path,
        file_sha256=first.file_sha256,
    )
    with pytest.raises(operator.OpenedGenesisOperatorError, match="role set"):
        operator.OpenedGenesisRequest(inputs=tuple(bindings))

    bindings = list(fixture.bindings)
    bindings[1] = operator.InputBinding(
        role=bindings[1].role,
        path=bindings[0].path,
        file_sha256=bindings[1].file_sha256,
    )
    with pytest.raises(operator.OpenedGenesisOperatorError, match="reuse an input path"):
        operator.OpenedGenesisRequest(inputs=tuple(bindings))


def test_pinned_input_rejects_hard_links(tmp_path: Path) -> None:
    target = tmp_path / "control.json"
    alias = tmp_path / "alias.json"
    _write(target, b"control\n")
    os.link(target, alias)

    with pytest.raises(operator.OpenedGenesisOperatorError, match="singly linked"):
        operator._PinnedFile.open(
            target,
            label="fixture control",
            expected_sha256=_digest(b"control\n"),
            max_bytes=1024,
        )


@pytest.mark.parametrize("kind", ["fifo", "symlink"])
def test_pinned_input_rejects_special_files(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "control"
    if kind == "fifo":
        os.mkfifo(target, mode=0o600)
    else:
        source = tmp_path / "source"
        _write(source, b"control\n")
        target.symlink_to(source)

    with pytest.raises(operator.OpenedGenesisOperatorError):
        operator._PinnedFile.open(
            target,
            label="fixture control",
            expected_sha256=_digest(b"control\n"),
            max_bytes=1024,
        )


def test_fifo_admission_uses_nonblocking_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "control"
    os.mkfifo(target, mode=0o600)
    real_open = operator.os.open
    observed_flags: list[int] = []

    def recording_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if path == target.name:
            observed_flags.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(operator.os, "open", recording_open)
    with pytest.raises(operator.OpenedGenesisOperatorError, match="regular file"):
        operator._PinnedFile.open(
            target,
            label="fixture FIFO",
            expected_sha256=_digest(b""),
            max_bytes=1024,
        )
    assert observed_flags
    assert observed_flags[0] & os.O_NONBLOCK


def test_pinned_input_detects_path_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "control"
    replacement = tmp_path / "replacement"
    data = b"a" * (operator._READ_CHUNK_BYTES + 1)
    _write(target, data)
    _write(replacement, b"b" * len(data))
    original = operator.os.pread
    replaced = False

    def racing_pread(descriptor: int, count: int, offset: int) -> bytes:
        nonlocal replaced
        result = original(descriptor, count, offset)
        if not replaced:
            replaced = True
            os.replace(replacement, target)
        return result

    monkeypatch.setattr(operator.os, "pread", racing_pread)
    with pytest.raises(operator.OpenedGenesisOperatorError, match="changed while"):
        operator._PinnedFile.open(
            target,
            label="fixture control",
            expected_sha256=_digest(data),
            max_bytes=len(data) + 1,
        )


def test_pinned_input_detects_mutation_after_admission(tmp_path: Path) -> None:
    target = tmp_path / "control"
    _write(target, b"before\n")
    pin = operator._PinnedFile.open(
        target,
        label="fixture control",
        expected_sha256=_digest(b"before\n"),
        max_bytes=1024,
    )
    try:
        target.write_bytes(b"after!\n")
        target.chmod(0o600)
        with pytest.raises(operator.OpenedGenesisOperatorError, match="changed after"):
            pin.assert_current()
    finally:
        pin.close()


def test_pinned_input_detects_ancestor_substitution(tmp_path: Path) -> None:
    parent = tmp_path / "authority"
    target = parent / "control"
    _write(target, b"before\n")
    pin = operator._PinnedFile.open(
        target,
        label="fixture control",
        expected_sha256=_digest(b"before\n"),
        max_bytes=1024,
    )
    try:
        displaced = tmp_path / "displaced"
        parent.rename(displaced)
        parent.mkdir(mode=0o700)
        _write(parent / "control", b"before\n")
        with pytest.raises(operator.OpenedGenesisOperatorError, match="parent path"):
            pin.assert_current()
    finally:
        pin.close()


def test_failed_open_is_quarantined_before_error_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)

    def failing_open(*args: object, **kwargs: object) -> Path:
        del args
        fixture.namespace.mkdir(mode=0o700)
        (fixture.namespace / "online").mkdir(mode=0o700)
        descriptor = kwargs["attestation_descriptor"]
        _write(
            fixture.namespace / "attestation-descriptor.json",
            descriptor.canonical_bytes() + b"\n",  # type: ignore[union-attr]
        )
        raise SuiteAttemptError("injected write failure")

    _patch_typed_apparatus(monkeypatch, fixture, opener=failing_open)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="injected write failure",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert not fixture.namespace.exists()
    quarantine = _quarantine_path(fixture)
    assert quarantine.is_dir()
    assert (quarantine / "online").is_dir()
    assert (quarantine / "attestation-descriptor.json").read_bytes() == _descriptor_bytes()


def test_unrecognized_partial_output_is_preserved_as_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)

    def failing_open(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        fixture.namespace.mkdir(mode=0o700)
        _write(fixture.namespace / "unexpected", b"external\n")
        raise ValueError("injected write failure")

    _patch_typed_apparatus(monkeypatch, fixture, opener=failing_open)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="publication is indeterminate",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert (fixture.namespace / "unexpected").read_bytes() == b"external\n"


def test_wrong_output_mode_fails_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    patched = _patch_typed_apparatus(monkeypatch, fixture)
    original_loader = operator.load_suite_state_record

    def wrong_mode_open(manifest: object, **kwargs: object) -> Path:
        del manifest
        fixture.namespace.mkdir(mode=0o700)
        (fixture.namespace / "online").mkdir(mode=0o700)
        descriptor = kwargs["attestation_descriptor"]
        _write(
            fixture.namespace / "attestation-descriptor.json",
            descriptor.canonical_bytes() + b"\n",  # type: ignore[union-attr]
            mode=0o644,
        )
        _write(fixture.namespace / "000.state.json", b"{}\n")
        return fixture.namespace

    del patched, original_loader
    monkeypatch.setattr(operator, "open_suite_attempt", wrong_mode_open)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="attestation descriptor bytes or mode differ",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert not fixture.namespace.exists()


def test_partial_without_request_bound_provenance_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)

    def failing_open(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        fixture.namespace.mkdir(mode=0o700)
        (fixture.namespace / "online").mkdir(mode=0o700)
        raise SuiteAttemptError("injected partial write")

    _patch_typed_apparatus(monkeypatch, fixture, opener=failing_open)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="lacks its request-bound provenance file",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert fixture.namespace.is_dir()


def test_member_substitution_at_cleanup_boundary_is_never_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)

    def failing_open(*args: object, **kwargs: object) -> Path:
        del args
        _publish_namespace(
            fixture,
            kwargs["attestation_descriptor"],
            include_state=False,
        )
        raise SuiteAttemptError("injected partial write")

    _patch_typed_apparatus(monkeypatch, fixture, opener=failing_open)
    real_read_member = operator._read_member_at
    substituted = False

    def substituting_read_member(
        parent_descriptor: int,
        name: str,
        *,
        label: str,
        max_bytes: int,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal substituted
        result = real_read_member(
            parent_descriptor,
            name,
            label=label,
            max_bytes=max_bytes,
        )
        if label == "partial OPENED attestation descriptor" and not substituted:
            substituted = True
            replacement = fixture.namespace / ".replacement-descriptor"
            _write(replacement, result[0])
            os.replace(
                replacement,
                fixture.namespace / "attestation-descriptor.json",
            )
        return result

    monkeypatch.setattr(operator, "_read_member_at", substituting_read_member)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="attestation-descriptor.json changed before fail-clean removal",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert substituted
    assert fixture.namespace.is_dir()
    assert (fixture.namespace / "attestation-descriptor.json").read_bytes() == _descriptor_bytes()


def test_namespace_substitution_at_cleanup_boundary_is_never_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    displaced = fixture.namespace.parent / "displaced-cleanup-output"

    def failing_open(*args: object, **kwargs: object) -> Path:
        del args
        _publish_namespace(
            fixture,
            kwargs["attestation_descriptor"],
            include_state=False,
        )
        raise SuiteAttemptError("injected partial write")

    _patch_typed_apparatus(monkeypatch, fixture, opener=failing_open)
    real_read_member = operator._read_member_at
    substituted = False

    def substituting_read_member(
        parent_descriptor: int,
        name: str,
        *,
        label: str,
        max_bytes: int,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal substituted
        result = real_read_member(
            parent_descriptor,
            name,
            label=label,
            max_bytes=max_bytes,
        )
        if label == "partial OPENED attestation descriptor" and not substituted:
            substituted = True
            fixture.namespace.rename(displaced)
            shutil.copytree(displaced, fixture.namespace)
        return result

    monkeypatch.setattr(operator, "_read_member_at", substituting_read_member)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="namespace path changed before fail-clean removal",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert substituted
    assert fixture.namespace.is_dir()
    assert displaced.is_dir()
    assert (fixture.namespace / "attestation-descriptor.json").is_file()
    assert (displaced / "attestation-descriptor.json").is_file()


def test_check_then_quarantine_replacement_is_restored_without_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    displaced = fixture.namespace.parent / "displaced-admitted-partial"
    replacement_identity: tuple[int, int] | None = None

    def failing_open(*args: object, **kwargs: object) -> Path:
        del args
        _publish_namespace(
            fixture,
            kwargs["attestation_descriptor"],
            include_state=False,
        )
        raise SuiteAttemptError("injected partial write")

    _patch_typed_apparatus(monkeypatch, fixture, opener=failing_open)
    real_rename = operator._rename_no_replace_at
    injected = False

    def replace_at_quarantine_boundary(
        source_parent: int,
        source_name: str,
        target_parent: int,
        target_name: str,
    ) -> None:
        nonlocal injected, replacement_identity
        if (
            not injected
            and source_name == fixture.namespace.name
            and target_name == _quarantine_path(fixture).name
        ):
            injected = True
            fixture.namespace.rename(displaced)
            fixture.namespace.mkdir(mode=0o700)
            metadata = fixture.namespace.stat()
            replacement_identity = metadata.st_dev, metadata.st_ino
        real_rename(
            source_parent,
            source_name,
            target_parent,
            target_name,
        )

    monkeypatch.setattr(
        operator,
        "_rename_no_replace_at",
        replace_at_quarantine_boundary,
    )
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="namespace changed at the quarantine boundary; replacement entry restored",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert injected
    assert replacement_identity is not None
    restored = fixture.namespace.stat()
    assert (restored.st_dev, restored.st_ino) == replacement_identity
    assert list(fixture.namespace.iterdir()) == []
    assert displaced.is_dir()
    assert (displaced / "attestation-descriptor.json").read_bytes() == _descriptor_bytes()
    assert not _quarantine_path(fixture).exists()
    assert _attempt_marker_path(fixture).read_bytes() == _attempt_marker_bytes(fixture)


@pytest.mark.parametrize(
    "interrupt",
    [KeyboardInterrupt(), SystemExit(73)],
    ids=["keyboard-interrupt", "system-exit"],
)
def test_base_exception_quarantines_exact_provenance_partial_before_reraise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    fixture = _request_fixture(tmp_path)

    def interrupted_open(*args: object, **kwargs: object) -> Path:
        del args
        _publish_namespace(
            fixture,
            kwargs["attestation_descriptor"],
            include_state=False,
        )
        raise interrupt

    _patch_typed_apparatus(monkeypatch, fixture, opener=interrupted_open)
    with pytest.raises(type(interrupt)):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert not fixture.namespace.exists()
    assert _quarantine_path(fixture).is_dir()
    assert _attempt_marker_path(fixture).read_bytes() == _attempt_marker_bytes(fixture)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="prior OPENED-attempt evidence exists; OPENED replay is forbidden",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )


@pytest.mark.parametrize(
    "termination_signal",
    [
        value
        for value in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGTERM", None),
        )
        if isinstance(value, signal.Signals)
    ],
    ids=lambda value: value.name,
)
def test_termination_signal_quarantines_exact_partial_and_restores_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination_signal: signal.Signals,
) -> None:
    fixture = _request_fixture(tmp_path)
    previous = signal.getsignal(termination_signal)

    def interrupted_open(*args: object, **kwargs: object) -> Path:
        del args
        _publish_namespace(
            fixture,
            kwargs["attestation_descriptor"],
            include_state=False,
        )
        os.kill(os.getpid(), termination_signal)
        raise AssertionError("termination-signal handler returned")

    _patch_typed_apparatus(monkeypatch, fixture, opener=interrupted_open)
    with pytest.raises(operator._TerminationSignal, match="received signal"):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert not fixture.namespace.exists()
    assert _quarantine_path(fixture).is_dir()
    assert signal.getsignal(termination_signal) is previous


def test_suite_parent_replacement_during_core_is_indeterminate_and_never_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    displaced_parent = tmp_path / "displaced-suite-parent"

    def replacing_open(*args: object, **kwargs: object) -> Path:
        del args
        fixture.namespace.parent.rename(displaced_parent)
        fixture.namespace.parent.mkdir(mode=0o700)
        _publish_namespace(fixture, kwargs["attestation_descriptor"])
        return fixture.namespace

    _patch_typed_apparatus(monkeypatch, fixture, opener=replacing_open)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="indeterminate: suite namespace parent path was replaced",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert fixture.namespace.is_dir()
    assert (displaced_parent / f".opened-genesis-{fixture.suite_id}.lock").is_file()


def test_namespace_substitution_is_preserved_when_provenance_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    displaced_namespace = fixture.namespace.parent / "displaced-published-suite"

    def replacing_open(*args: object, **kwargs: object) -> Path:
        del args
        _publish_namespace(fixture, kwargs["attestation_descriptor"])
        fixture.namespace.rename(displaced_namespace)
        fixture.namespace.mkdir(mode=0o700)
        (fixture.namespace / "online").mkdir(mode=0o700)
        _write(
            fixture.namespace / "attestation-descriptor.json",
            b'{"unrelated":true}\n',
        )
        _write(fixture.namespace / "000.state.json", b"{}\n")
        return fixture.namespace

    _patch_typed_apparatus(monkeypatch, fixture, opener=replacing_open)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="partial namespace provenance differs",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert fixture.namespace.is_dir()
    assert displaced_namespace.is_dir()


def test_typed_loader_parent_swap_is_detected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    _patch_typed_apparatus(monkeypatch, fixture)
    controls = fixture.paths["production-finalization-request"].parent
    displaced = controls.parent / "displaced-controls"
    swapped = False

    def swapping_loader(path: str | Path) -> SimpleNamespace:
        nonlocal swapped
        if not swapped:
            swapped = True
            controls.rename(displaced)
            shutil.copytree(displaced, controls)
        return SimpleNamespace(path=path)

    monkeypatch.setattr(operator, "load_runtime_preflight_receipt", swapping_loader)
    with pytest.raises(operator.OpenedGenesisOperatorError, match="parent path was replaced"):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert not fixture.namespace.exists()


def test_typed_loader_transient_parent_substitution_cannot_escape_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    _patch_typed_apparatus(monkeypatch, fixture)
    controls = fixture.paths["production-finalization-request"].parent
    displaced = controls.parent / "displaced-original-controls"
    consumed = controls.parent / "consumed-substitute-controls"
    consumed_bytes: list[bytes] = []

    def transient_loader(path: str | Path) -> SimpleNamespace:
        if not consumed_bytes:
            controls.rename(displaced)
            controls.mkdir(mode=0o700)
            substitute = controls / Path(path).name
            _write(substitute, b'{"adversarial_substitute":true}\n')
            consumed_bytes.append(substitute.read_bytes())
            controls.rename(consumed)
            displaced.rename(controls)
        return SimpleNamespace(path=path)

    monkeypatch.setattr(operator, "load_runtime_preflight_receipt", transient_loader)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="changed after admission during typed consumption",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert consumed_bytes == [b'{"adversarial_substitute":true}\n']
    assert controls.is_dir()
    assert consumed.is_dir()
    assert not fixture.namespace.exists()
    assert not _attempt_marker_path(fixture).exists()


def test_core_input_swap_is_detected_and_exact_partial_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    target = fixture.paths["sealed-run-receipt"]

    def swapping_open(*args: object, **kwargs: object) -> Path:
        del args
        replacement = target.with_name("replacement.json")
        _write(replacement, fixture.values["sealed-run-receipt"])
        os.replace(replacement, target)
        _publish_namespace(fixture, kwargs["attestation_descriptor"])
        return fixture.namespace

    _patch_typed_apparatus(monkeypatch, fixture, opener=swapping_open)
    with pytest.raises(operator.OpenedGenesisOperatorError, match="changed after admission"):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert not fixture.namespace.exists()
    assert _quarantine_path(fixture).is_dir()


def test_attempt_marker_substitution_after_arm_preserves_published_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    marker = _attempt_marker_path(fixture)
    displaced = marker.with_name("displaced-attempt-marker")

    def substituting_open(*args: object, **kwargs: object) -> Path:
        del args
        _publish_namespace(fixture, kwargs["attestation_descriptor"])
        marker.rename(displaced)
        _write(marker, displaced.read_bytes())
        return fixture.namespace

    _patch_typed_apparatus(monkeypatch, fixture, opener=substituting_open)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="indeterminate: attempt evidence changed",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert fixture.namespace.is_dir()
    assert marker.read_bytes() == _attempt_marker_bytes(fixture)
    assert displaced.read_bytes() == _attempt_marker_bytes(fixture)


def test_typed_state_readback_cannot_swap_namespace_with_identical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    patched = _patch_typed_apparatus(monkeypatch, fixture)
    displaced = fixture.namespace.parent / "displaced-during-readback"

    def swapping_loader(path: str | Path) -> object:
        fixture.namespace.rename(displaced)
        shutil.copytree(displaced, fixture.namespace)
        return patched.record

    monkeypatch.setattr(operator, "load_suite_state_record", swapping_loader)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="namespace identity differs from the admitted output",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert fixture.namespace.is_dir()
    assert displaced.is_dir()


def test_typed_state_readback_cannot_substitute_identical_state_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    patched = _patch_typed_apparatus(monkeypatch, fixture)
    state_path = fixture.namespace / "000.state.json"

    def swapping_loader(path: str | Path) -> object:
        assert Path(path) == state_path
        replacement = fixture.namespace / ".replacement-state"
        _write(replacement, state_path.read_bytes())
        os.replace(replacement, state_path)
        return patched.record

    monkeypatch.setattr(operator, "load_suite_state_record", swapping_loader)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="000.state.json differs from the admitted output",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert fixture.namespace.is_dir()
    assert state_path.read_bytes() == b"{}\n"


def test_restrictive_inherited_umask_still_yields_exact_private_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)

    def raw_file(path: Path, data: bytes) -> None:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o666,
        )
        try:
            os.write(descriptor, data)
        finally:
            os.close(descriptor)

    def raw_open(*args: object, **kwargs: object) -> Path:
        del args
        os.mkdir(fixture.namespace, 0o777)
        os.mkdir(fixture.namespace / "online", 0o777)
        descriptor = kwargs["attestation_descriptor"]
        raw_file(
            fixture.namespace / "attestation-descriptor.json",
            descriptor.canonical_bytes() + b"\n",  # type: ignore[union-attr]
        )
        raw_file(fixture.namespace / "000.state.json", b"{}\n")
        return fixture.namespace

    _patch_typed_apparatus(monkeypatch, fixture, opener=raw_open)
    original_umask = os.umask(0o777)
    try:
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
        inherited_after = os.umask(0o777)
        assert inherited_after == 0o777
    finally:
        os.umask(original_umask)

    lock = fixture.namespace.parent / f".opened-genesis-{fixture.suite_id}.lock"
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600
    assert stat.S_IMODE(fixture.namespace.stat().st_mode) == 0o700
    assert stat.S_IMODE((fixture.namespace / "online").stat().st_mode) == 0o700
    assert stat.S_IMODE((fixture.namespace / "attestation-descriptor.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((fixture.namespace / "000.state.json").stat().st_mode) == 0o600


def test_persistent_lock_survives_success_and_forbids_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    _patch_typed_apparatus(monkeypatch, fixture)
    operator.execute(
        fixture.request_path,
        expected_request_sha256=fixture.request_sha256,
    )
    lock = fixture.namespace.parent / f".opened-genesis-{fixture.suite_id}.lock"
    before = lock.stat()

    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="OPENED replay is forbidden",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    after = lock.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert lock.read_bytes() == b""
    assert stat.S_IMODE(after.st_mode) == 0o600
    assert _attempt_marker_path(fixture).read_bytes() == _attempt_marker_bytes(fixture)


def test_existing_lock_held_by_another_publisher_blocks_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    _patch_typed_apparatus(monkeypatch, fixture)
    lock = fixture.namespace.parent / f".opened-genesis-{fixture.suite_id}.lock"
    _write(lock, b"")
    descriptor = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            operator.OpenedGenesisOperatorError,
            match="another OPENED-genesis publisher holds",
        ):
            operator.execute(
                fixture.request_path,
                expected_request_sha256=fixture.request_sha256,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert not fixture.namespace.exists()


def test_preexisting_cleanup_quarantine_blocks_without_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    core_calls = 0

    def opener(*args: object, **kwargs: object) -> Path:
        nonlocal core_calls
        del args, kwargs
        core_calls += 1
        raise AssertionError("pre-existing quarantine must block the opener")

    _patch_typed_apparatus(monkeypatch, fixture, opener=opener)
    quarantine = _quarantine_path(fixture)
    quarantine.mkdir(mode=0o700)
    _write(quarantine / "external-evidence", b"preserve\n")
    before = quarantine.stat()

    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="OPENED cleanup quarantine already exists; publication is forbidden",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    after = quarantine.stat()
    assert core_calls == 0
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert (quarantine / "external-evidence").read_bytes() == b"preserve\n"
    assert not _attempt_marker_path(fixture).exists()
    assert not fixture.namespace.exists()


def test_typed_state_readback_failure_is_fail_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    _patch_typed_apparatus(monkeypatch, fixture)

    def failing_loader(path: str | Path) -> object:
        del path
        raise SuiteAttemptError("injected readback rejection")

    monkeypatch.setattr(operator, "load_suite_state_record", failing_loader)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="published OPENED state is invalid",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert not fixture.namespace.exists()


def test_one_shot_post_publication_parent_fsync_failure_is_fail_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    _patch_typed_apparatus(monkeypatch, fixture)
    parent_identity = (
        fixture.namespace.parent.stat().st_dev,
        fixture.namespace.parent.stat().st_ino,
    )
    real_fsync = operator.os.fsync
    failed = False

    def flaky_fsync(descriptor: int) -> None:
        nonlocal failed
        metadata = os.fstat(descriptor)
        if (
            not failed
            and fixture.namespace.exists()
            and (metadata.st_dev, metadata.st_ino) == parent_identity
        ):
            failed = True
            raise OSError("injected parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(operator.os, "fsync", flaky_fsync)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="OPENED genesis was not published: injected parent fsync failure",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert failed
    assert not fixture.namespace.exists()
    assert _quarantine_path(fixture).is_dir()


def test_post_publication_fsync_failure_irrevocably_blocks_reinvoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    core_calls = 0

    def opener(manifest: object, **kwargs: object) -> Path:
        nonlocal core_calls
        del manifest
        core_calls += 1
        _publish_namespace(fixture, kwargs["attestation_descriptor"])
        return fixture.namespace

    _patch_typed_apparatus(monkeypatch, fixture, opener=opener)
    parent_identity = (
        fixture.namespace.parent.stat().st_dev,
        fixture.namespace.parent.stat().st_ino,
    )
    real_fsync = operator.os.fsync
    failed = False

    def flaky_fsync(descriptor: int) -> None:
        nonlocal failed
        metadata = os.fstat(descriptor)
        if (
            not failed
            and fixture.namespace.exists()
            and (metadata.st_dev, metadata.st_ino) == parent_identity
        ):
            failed = True
            raise OSError("injected post-publication parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(operator.os, "fsync", flaky_fsync)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="injected post-publication parent fsync failure",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert failed
    assert core_calls == 1
    assert not fixture.namespace.exists()
    assert _quarantine_path(fixture).is_dir()
    marker = _attempt_marker_path(fixture)
    assert marker.read_bytes() == _attempt_marker_bytes(fixture)

    monkeypatch.setattr(operator.os, "fsync", real_fsync)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="prior OPENED-attempt evidence exists; OPENED replay is forbidden",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert core_calls == 1
    assert not fixture.namespace.exists()
    assert marker.read_bytes() == _attempt_marker_bytes(fixture)


def test_attempt_marker_fsync_failure_leaves_poison_and_blocks_reinvoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    core_calls = 0

    def opener(*args: object, **kwargs: object) -> Path:
        nonlocal core_calls
        del args, kwargs
        core_calls += 1
        raise AssertionError("attempt marker failure must precede the suite opener")

    _patch_typed_apparatus(monkeypatch, fixture, opener=opener)
    marker = _attempt_marker_path(fixture)
    real_fsync = operator.os.fsync
    failed = False

    def flaky_fsync(descriptor: int) -> None:
        nonlocal failed
        if not failed and marker.exists():
            marker_metadata = marker.stat()
            descriptor_metadata = os.fstat(descriptor)
            if (
                descriptor_metadata.st_dev,
                descriptor_metadata.st_ino,
            ) == (
                marker_metadata.st_dev,
                marker_metadata.st_ino,
            ):
                failed = True
                raise OSError("injected attempt-marker fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(operator.os, "fsync", flaky_fsync)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="indeterminate: attempt evidence persistence failed",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert failed
    assert core_calls == 0
    assert marker.read_bytes() == b""
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600

    monkeypatch.setattr(operator.os, "fsync", real_fsync)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="attempt evidence is invalid; OPENED replay is forbidden",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert core_calls == 0
    assert marker.read_bytes() == b""


@pytest.mark.parametrize(
    "marker_kind",
    ["empty", "wrong-mode", "hardlink", "symlink", "fifo"],
)
def test_invalid_attempt_evidence_blocks_without_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_kind: str,
) -> None:
    fixture = _request_fixture(tmp_path)
    _patch_typed_apparatus(monkeypatch, fixture)
    marker = _attempt_marker_path(fixture)
    if marker_kind == "empty":
        _write(marker, b"")
    elif marker_kind == "wrong-mode":
        _write(marker, _attempt_marker_bytes(fixture), mode=0o640)
    elif marker_kind == "hardlink":
        backing = marker.with_name("attempt-evidence-backing")
        _write(backing, _attempt_marker_bytes(fixture))
        os.link(backing, marker)
    elif marker_kind == "symlink":
        target = marker.with_name("attempt-evidence-target")
        _write(target, _attempt_marker_bytes(fixture))
        marker.symlink_to(target)
    else:
        os.mkfifo(marker, mode=0o600)
    before = os.lstat(marker)

    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="attempt evidence is invalid; OPENED replay is forbidden",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    after = os.lstat(marker)
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_nlink) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
    )
    assert not fixture.namespace.exists()


def test_persistent_parent_fsync_failure_is_reported_as_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _request_fixture(tmp_path)
    _patch_typed_apparatus(monkeypatch, fixture)
    parent_identity = (
        fixture.namespace.parent.stat().st_dev,
        fixture.namespace.parent.stat().st_ino,
    )
    real_fsync = operator.os.fsync
    armed = False

    def failing_fsync(descriptor: int) -> None:
        nonlocal armed
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == parent_identity and (
            armed or fixture.namespace.exists()
        ):
            armed = True
            raise OSError("persistent parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(operator.os, "fsync", failing_fsync)
    with pytest.raises(
        operator.OpenedGenesisOperatorError,
        match="indeterminate: namespace quarantine synchronization failed",
    ):
        operator.execute(
            fixture.request_path,
            expected_request_sha256=fixture.request_sha256,
        )
    assert armed
    assert not fixture.namespace.exists()
    assert _quarantine_path(fixture).is_dir()


def test_unknown_cli_argument_is_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    code = operator.main(
        [
            "--request",
            "/controlled/opened-genesis-request.json",
            "--request-sha256",
            "0" * 64,
            "--unknown",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert captured.err.count("\n") == 1
    assert "unrecognized arguments: --unknown" in captured.err


def test_cli_rejects_prefix_abbreviation(capsys: pytest.CaptureFixture[str]) -> None:
    code = operator.main(
        [
            "--request",
            "/controlled/opened-genesis-request.json",
            "--request-sha256",
            "0" * 64,
            "--request-sha",
            "0" * 64,
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "unrecognized arguments: --request-sha" in captured.err
