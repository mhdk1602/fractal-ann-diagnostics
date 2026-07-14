"""Command-line entry point for the reproducible development benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

from .artifact_integrity import (
    load_local_artifact_map,
    verify_local_artifacts,
    write_verification_receipt,
)
from .pilot import PilotConfig, write_pilot_artifacts
from .study import (
    begin_sealed_run,
    load_study_manifest,
    manifest_sha256,
    validate_study_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fractal-retrieval-governance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot", help="run the frozen synthetic development pilot")
    pilot.add_argument("--output", type=Path, default=Path("artifacts/pilot"))
    pilot.add_argument("--seed", type=int, default=PilotConfig.seed)
    pilot.add_argument("--documents", type=int, default=PilotConfig.n_documents)
    pilot.add_argument("--queries-per-role", type=int, default=PilotConfig.n_queries_per_role)

    validate = subparsers.add_parser(
        "validate-study",
        help="validate the confirmatory study manifest without opening sealed data",
    )
    validate.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/study-manifest.json"),
    )
    validate.add_argument(
        "--require-frozen",
        action="store_true",
        help="also enforce every sealed-execution prerequisite",
    )

    digest = subparsers.add_parser(
        "study-digest",
        help="print the canonical SHA-256 for a study manifest",
    )
    digest.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/study-manifest.json"),
    )

    verify = subparsers.add_parser(
        "verify-study-artifacts",
        help="verify every frozen manifest artifact through an explicit local path map",
    )
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--artifact-root", type=Path, required=True)
    verify.add_argument("--artifact-map", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)

    begin = subparsers.add_parser(
        "begin-sealed-run",
        help="verify frozen locks and atomically create the one-shot run receipt",
    )
    begin.add_argument("--manifest", type=Path, required=True)
    begin.add_argument("--lock", type=Path, required=True)
    begin.add_argument(
        "--artifact-verification-receipt",
        type=Path,
        required=True,
    )
    begin.add_argument("--artifact-root", type=Path, required=True)
    begin.add_argument("--artifact-map", type=Path, required=True)
    begin.add_argument(
        "--protocol-registration-receipt",
        type=Path,
        required=True,
    )
    begin.add_argument(
        "--protocol-registration-record",
        type=Path,
        required=True,
    )
    begin.add_argument("--runner-identity", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "pilot":
        config = PilotConfig(
            seed=args.seed,
            n_documents=args.documents,
            n_queries_per_role=args.queries_per_role,
        )
        _, summaries, metadata = write_pilot_artifacts(args.output, config)
        print(f"wrote {len(summaries)} summary cells to {args.output}")
        print(f"action outcomes: {metadata['n_action_outcomes']}")
        return 0
    if args.command == "validate-study":
        payload = load_study_manifest(args.manifest)
        validate_study_manifest(payload, require_frozen=args.require_frozen)
        print(f"valid study manifest: {args.manifest}")
        print(f"sha256: {manifest_sha256(payload)}")
        return 0
    if args.command == "study-digest":
        payload = load_study_manifest(args.manifest)
        validate_study_manifest(payload)
        print(manifest_sha256(payload))
        return 0
    if args.command == "verify-study-artifacts":
        payload = load_study_manifest(args.manifest)
        validate_study_manifest(payload, require_frozen=True)
        digest = manifest_sha256(payload)
        pins = {
            str(artifact["id"]): str(artifact["sha256"])
            for artifact in payload["artifacts"]
        }
        specs = load_local_artifact_map(
            args.artifact_map,
            expected_sha256_by_id=pins,
        )
        receipt = verify_local_artifacts(
            args.artifact_root,
            manifest_sha256=digest,
            artifacts=specs,
        )
        write_verification_receipt(receipt, args.receipt)
        print(f"verified {len(receipt.artifacts)} frozen study artifacts")
        print(f"manifest sha256: {receipt.manifest_sha256}")
        print(f"receipt sha256: {receipt.receipt_sha256}")
        print(f"receipt: {args.receipt}")
        return 0
    if args.command == "begin-sealed-run":
        receipt = begin_sealed_run(
            args.manifest,
            args.lock,
            runner_identity=args.runner_identity,
            artifact_verification_receipt_path=(
                args.artifact_verification_receipt
            ),
            artifact_root=args.artifact_root,
            local_artifact_map_path=args.artifact_map,
            protocol_registration_receipt_path=(
                args.protocol_registration_receipt
            ),
            protocol_registration_record_path=(
                args.protocol_registration_record
            ),
        )
        print(f"sealed run opened for protocol {receipt.protocol_version}")
        print(f"manifest sha256: {receipt.manifest_sha256}")
        print(f"receipt: {receipt.receipt_uri}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
