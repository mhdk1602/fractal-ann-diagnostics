from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fractal_ann_diagnostics.tlock_reachability as reachability
from fractal_ann_diagnostics.tlock_reachability import (
    TlockReachabilityError,
    adjudicate_tlock_reachability,
)


def _stream(*, mode: str, package_reachable: bool = False) -> bytes:
    config = {
        "db": "https://vuln.go.dev",
        "db_last_modified": "2026-07-08T17:05:00Z",
        "protocol_version": "v1.0.0",
        "scan_level": "symbol",
        "scan_mode": mode,
        "scanner_name": "govulncheck",
        "scanner_version": "v1.6.0",
    }
    if mode == "source":
        config["go_version"] = "go1.26.5"
    trace: list[dict[str, str]] = [{"module": "golang.org/x/crypto", "version": "v0.54.0"}]
    if package_reachable:
        trace.append({"package": "golang.org/x/crypto/openpgp"})
    documents = (
        {"config": config},
        {
            "osv": {
                "database_specific": {
                    "review_status": "REVIEWED",
                    "url": "https://pkg.go.dev/vuln/GO-2026-5932",
                },
                "id": "GO-2026-5932",
            }
        },
        {"finding": {"osv": "GO-2026-5932", "trace": trace}},
    )
    return b"\n".join(json.dumps(document).encode() for document in documents) + b"\n"


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    govulncheck = b"govulncheck-v1.6.0"
    symbol_binary = b"symbol-bearing-tle"
    packages = b"archive/tar\ngithub.com/drand/tlock/cmd/tle\n"
    nm = b"0000000000401000 T main.main\n"
    monkeypatch.setattr(
        reachability, "_GOVULNCHECK_SHA256", hashlib.sha256(govulncheck).hexdigest()
    )
    monkeypatch.setattr(reachability, "_GOVULNCHECK_BYTE_COUNT", len(govulncheck))
    monkeypatch.setattr(
        reachability, "_SYMBOL_BINARY_SHA256", hashlib.sha256(symbol_binary).hexdigest()
    )
    monkeypatch.setattr(reachability, "_SYMBOL_BINARY_BYTE_COUNT", len(symbol_binary))
    monkeypatch.setattr(
        reachability, "_SOURCE_PACKAGE_LIST_SHA256", hashlib.sha256(packages).hexdigest()
    )
    monkeypatch.setattr(reachability, "_SOURCE_PACKAGE_COUNT", 2)
    monkeypatch.setattr(reachability, "_NM_SHA256", hashlib.sha256(nm).hexdigest())
    paths = {
        "source_scan_path": tmp_path / "source.json",
        "binary_scan_path": tmp_path / "binary.json",
        "source_packages_path": tmp_path / "packages.txt",
        "nm_path": tmp_path / "nm.txt",
        "govulncheck_binary_path": tmp_path / "govulncheck",
        "symbol_binary_path": tmp_path / "tle-symbols",
        "output_path": tmp_path / "receipt.json",
    }
    paths["source_scan_path"].write_bytes(_stream(mode="source"))
    paths["binary_scan_path"].write_bytes(_stream(mode="binary"))
    paths["source_packages_path"].write_bytes(packages)
    paths["nm_path"].write_bytes(nm)
    paths["govulncheck_binary_path"].write_bytes(govulncheck)
    paths["symbol_binary_path"].write_bytes(symbol_binary)
    return paths


def test_module_only_source_and_binary_findings_emit_one_closed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path, monkeypatch)

    receipt = adjudicate_tlock_reachability(**paths)

    assert receipt["schema_version"] == "fractal-tlock-govulncheck-reachability-v1"
    assert receipt["source_scan"]["finding_trace_level"] == "module"
    assert receipt["binary_scan"]["finding_trace_level"] == "module"
    assert receipt["finding"]["package_or_symbol_reachable"] is False
    assert receipt["vex_document"] is None
    assert json.loads(paths["output_path"].read_text()) == receipt


def test_package_level_trace_is_rejected_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path, monkeypatch)
    paths["source_scan_path"].write_bytes(_stream(mode="source", package_reachable=True))

    with pytest.raises(TlockReachabilityError, match="module-only"):
        adjudicate_tlock_reachability(**paths)
    assert not paths["output_path"].exists()


def test_openpgp_package_name_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path, monkeypatch)
    packages = b"archive/tar\ngolang.org/x/crypto/openpgp\n"
    paths["source_packages_path"].write_bytes(packages)
    monkeypatch.setattr(
        reachability, "_SOURCE_PACKAGE_LIST_SHA256", hashlib.sha256(packages).hexdigest()
    )

    with pytest.raises(TlockReachabilityError, match="closed set"):
        adjudicate_tlock_reachability(**paths)
    assert not paths["output_path"].exists()
