"""Admit retained govulncheck evidence for the source-built timelock binary."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TLOCK_REACHABILITY_SCHEMA = "fractal-tlock-govulncheck-reachability-v1"

_GOVULNCHECK_SHA256 = "1cf0bf22b6f9484c850380cd3065bffd9a6d6577181e281053ab2d6bcb8898f0"
_GOVULNCHECK_BYTE_COUNT = 9_633_918
_SYMBOL_BINARY_SHA256 = "69ca051a3d3e14f6f405875dfdcb976c6be78cab66dc24c7191a949bd8257ff7"
_SYMBOL_BINARY_BYTE_COUNT = 19_342_638
_SOURCE_PACKAGE_LIST_SHA256 = "323f5b5bb4147900ae4401b214cd9156e19f8690ad41a4db4fcef66dd5694265"
_SOURCE_PACKAGE_COUNT = 403
_NM_SHA256 = "d37ef9b9e10d3b3b17569653d5d3be68f5dba50f72d6494fcf63a360c952936b"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_FILE_BYTES = 512 * 1024 * 1024


class TlockReachabilityError(ValueError):
    """Retained reachability evidence is malformed or outside the closed claim."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read(path: Path, *, label: str) -> bytes:
    try:
        observed = path.lstat()
    except OSError as error:
        raise TlockReachabilityError(f"cannot open {label}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise TlockReachabilityError(f"{label} must be one singly linked regular file")
    if observed.st_size <= 0 or observed.st_size > _MAX_FILE_BYTES:
        raise TlockReachabilityError(f"{label} has an invalid byte count")
    payload = path.read_bytes()
    if len(payload) != observed.st_size:
        raise TlockReachabilityError(f"{label} changed while it was read")
    return payload


def _json_stream(payload: bytes, *, label: str) -> tuple[Mapping[str, Any], ...]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise TlockReachabilityError(f"{label} is not UTF-8") from error
    decoder = json.JSONDecoder()
    position = 0
    documents: list[Mapping[str, Any]] = []
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            raise TlockReachabilityError(f"{label} is not one JSON object stream") from error
        if not isinstance(value, Mapping) or len(value) != 1:
            raise TlockReachabilityError(f"{label} contains a non-protocol document")
        documents.append(value)
    if not documents:
        raise TlockReachabilityError(f"{label} contains no protocol documents")
    return tuple(documents)


def _scan_projection(
    payload: bytes,
    *,
    expected_mode: str,
    label: str,
) -> dict[str, object]:
    documents = _json_stream(payload, label=label)
    configs = [document["config"] for document in documents if "config" in document]
    if len(configs) != 1 or not isinstance(configs[0], Mapping):
        raise TlockReachabilityError(f"{label} must contain exactly one config")
    config = configs[0]
    required_config = {
        "protocol_version": "v1.0.0",
        "scan_level": "symbol",
        "scan_mode": expected_mode,
        "scanner_name": "govulncheck",
        "scanner_version": "v1.6.0",
    }
    if expected_mode == "source":
        required_config["go_version"] = "go1.26.5"
    for field, expected in required_config.items():
        if config.get(field) != expected:
            raise TlockReachabilityError(f"{label} has invalid config field {field}")
    if config.get("db") != "https://vuln.go.dev":
        raise TlockReachabilityError(f"{label} used another vulnerability database")
    db_last_modified = config.get("db_last_modified")
    if not isinstance(db_last_modified, str) or not db_last_modified.endswith("Z"):
        raise TlockReachabilityError(f"{label} lacks a UTC database revision")

    findings = [document["finding"] for document in documents if "finding" in document]
    expected_finding = {
        "osv": "GO-2026-5932",
        "trace": [{"module": "golang.org/x/crypto", "version": "v0.54.0"}],
    }
    if findings != [expected_finding]:
        raise TlockReachabilityError(f"{label} must contain one module-only GO-2026-5932 finding")
    osv_documents = [document["osv"] for document in documents if "osv" in document]
    matching_osv = [
        row for row in osv_documents if isinstance(row, Mapping) and row.get("id") == "GO-2026-5932"
    ]
    if len(matching_osv) != 1:
        raise TlockReachabilityError(f"{label} lacks the GO-2026-5932 advisory record")
    advisory = matching_osv[0]
    database_specific = advisory.get("database_specific")
    if not isinstance(database_specific, Mapping) or database_specific.get("url") != (
        "https://pkg.go.dev/vuln/GO-2026-5932"
    ):
        raise TlockReachabilityError(f"{label} advisory lacks its official reference")
    return {
        "db_last_modified": db_last_modified,
        "finding_count": 1,
        "finding_trace_level": "module",
        "package_or_symbol_reachable": False,
        "protocol_document_count": len(documents),
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "scan_mode": expected_mode,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
    except OSError as error:
        if error.errno == errno.EEXIST:
            raise TlockReachabilityError("reachability receipt already exists") from error
        raise TlockReachabilityError(f"cannot create reachability receipt: {error}") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def adjudicate_tlock_reachability(
    *,
    source_scan_path: Path,
    binary_scan_path: Path,
    source_packages_path: Path,
    nm_path: Path,
    govulncheck_binary_path: Path,
    symbol_binary_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Admit module-only findings and write one canonical immutable receipt."""
    source_scan = _read(source_scan_path, label="source govulncheck stream")
    binary_scan = _read(binary_scan_path, label="binary govulncheck stream")
    packages = _read(source_packages_path, label="source package list")
    nm = _read(nm_path, label="symbol-bearing binary nm output")
    govulncheck = _read(govulncheck_binary_path, label="govulncheck binary")
    symbol_binary = _read(symbol_binary_path, label="symbol-bearing tle binary")

    if (
        len(govulncheck) != _GOVULNCHECK_BYTE_COUNT
        or hashlib.sha256(govulncheck).hexdigest() != _GOVULNCHECK_SHA256
    ):
        raise TlockReachabilityError("govulncheck binary differs from its v1.6.0 pin")
    if (
        len(symbol_binary) != _SYMBOL_BINARY_BYTE_COUNT
        or hashlib.sha256(symbol_binary).hexdigest() != _SYMBOL_BINARY_SHA256
    ):
        raise TlockReachabilityError("symbol-bearing tle analysis twin differs from its pin")
    if hashlib.sha256(packages).hexdigest() != _SOURCE_PACKAGE_LIST_SHA256:
        raise TlockReachabilityError("source package list differs from its pin")
    package_names = packages.decode("utf-8", errors="strict").splitlines()
    if (
        len(package_names) != _SOURCE_PACKAGE_COUNT
        or len(set(package_names)) != len(package_names)
        or any("openpgp" in name.casefold() for name in package_names)
    ):
        raise TlockReachabilityError("source package closure is not the admitted closed set")
    if hashlib.sha256(nm).hexdigest() != _NM_SHA256:
        raise TlockReachabilityError("symbol-bearing tle nm output differs from its pin")
    if b"openpgp" in nm.lower():
        raise TlockReachabilityError("symbol-bearing tle contains an openpgp symbol")

    source = _scan_projection(source_scan, expected_mode="source", label="source scan")
    binary = _scan_projection(binary_scan, expected_mode="binary", label="binary scan")
    receipt: dict[str, object] = {
        "advisory": "https://pkg.go.dev/vuln/GO-2026-5932",
        "binary_scan": binary,
        "finding": {
            "module": "golang.org/x/crypto",
            "package_or_symbol_reachable": False,
            "version": "v0.54.0",
            "vulnerability_id": "GO-2026-5932",
        },
        "govulncheck_binary_byte_count": len(govulncheck),
        "govulncheck_binary_sha256": hashlib.sha256(govulncheck).hexdigest(),
        "govulncheck_version": "v1.6.0",
        "nm_sha256": hashlib.sha256(nm).hexdigest(),
        "schema_version": TLOCK_REACHABILITY_SCHEMA,
        "source_package_count": len(package_names),
        "source_package_list_sha256": hashlib.sha256(packages).hexdigest(),
        "source_scan": source,
        "symbol_binary_byte_count": len(symbol_binary),
        "symbol_binary_sha256": hashlib.sha256(symbol_binary).hexdigest(),
        "vex_document": None,
    }
    _write_exclusive(output_path, _canonical(receipt) + b"\n")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scan", required=True, type=Path)
    parser.add_argument("--binary-scan", required=True, type=Path)
    parser.add_argument("--source-packages", required=True, type=Path)
    parser.add_argument("--nm", required=True, type=Path)
    parser.add_argument("--govulncheck-binary", required=True, type=Path)
    parser.add_argument("--symbol-binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = adjudicate_tlock_reachability(
        source_scan_path=arguments.source_scan,
        binary_scan_path=arguments.binary_scan,
        source_packages_path=arguments.source_packages,
        nm_path=arguments.nm,
        govulncheck_binary_path=arguments.govulncheck_binary,
        symbol_binary_path=arguments.symbol_binary,
        output_path=arguments.output,
    )
    print(_canonical(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
