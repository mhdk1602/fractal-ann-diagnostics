"""Command-line entry point for the reproducible development benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .artifact_integrity import (
    load_local_artifact_map,
    verify_local_artifacts,
    write_verification_receipt,
)
from .custody import (
    admit_online_custody,
    custody_seal_receipt_from_manifest,
    encrypt_timelock_label,
    load_custody_seal_receipt,
    load_timelock_encryption_receipt,
    verify_custody_seal_receipt,
    verify_timelock_encryption_receipt,
    write_custody_seal_receipt,
    write_online_custody_admission_receipt,
    write_timelock_encryption_receipt,
)
from .execution_claim import (
    C1_REGISTRATION_PACKAGE_FILE_COUNT,
    ExecutionClaimError,
    loads_runtime_claim_receipt,
)
from .external_anchors import verify_prediction_completion_anchor
from .github_state_attestation import GitHubSuiteEvidenceVerifier
from .label_separation import load_prediction_completion_receipt
from .pilot import PilotConfig, write_pilot_artifacts
from .production_corpus_run import run_sealed_corpus_from_config
from .study import (
    begin_sealed_run,
    load_study_manifest,
    manifest_sha256,
    validate_study_manifest,
)
from .suite_attempt import verify_suite_state
from .timelock_release import (
    release_timelock_label,
    write_timelock_decryption_receipt,
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
    begin.add_argument(
        "--registration-package",
        type=Path,
        required=True,
        help=(
            f"closed {C1_REGISTRATION_PACKAGE_FILE_COUNT}-file C1 package verified "
            "against public Zenodo before run admission"
        ),
    )
    begin.add_argument("--runner-identity", required=True)

    create_custody = subparsers.add_parser(
        "create-custody-seal-receipt",
        help="commit pinned plaintext-label and timelock-ciphertext digests",
    )
    create_custody.add_argument("--manifest", type=Path, required=True)
    create_custody.add_argument("--drand-chain-hash", required=True)
    create_custody.add_argument("--drand-round", type=int, required=True)
    create_custody.add_argument("--receipt", type=Path, required=True)

    encrypt_label = subparsers.add_parser(
        "encrypt-timelock-label",
        help="encrypt one pinned plaintext-label file with the manifest-pinned tle binary",
    )
    encrypt_label.add_argument("--manifest", type=Path, required=True)
    encrypt_label.add_argument("--corpus-id", required=True)
    encrypt_label.add_argument("--plaintext", type=Path, required=True)
    encrypt_label.add_argument("--tle-binary", type=Path, required=True)
    encrypt_label.add_argument("--drand-network", required=True)
    encrypt_label.add_argument("--drand-chain-hash", required=True)
    encrypt_label.add_argument("--drand-round", type=int, required=True)
    encrypt_label.add_argument("--ciphertext", type=Path, required=True)
    encrypt_label.add_argument("--receipt", type=Path, required=True)
    encrypt_label.add_argument("--timeout-seconds", type=int, default=60)
    encrypt_label.add_argument(
        "--max-plaintext-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    encrypt_label.add_argument(
        "--max-ciphertext-bytes",
        type=int,
        default=128 * 1024 * 1024,
    )

    verify_encryption = subparsers.add_parser(
        "verify-timelock-encryption-receipt",
        help="verify one tle operation receipt against final manifest and custody pins",
    )
    verify_encryption.add_argument("--manifest", type=Path, required=True)
    verify_encryption.add_argument("--receipt", type=Path, required=True)
    verify_encryption.add_argument("--custody-seal", type=Path)
    verify_encryption.add_argument("--allow-draft", action="store_true")

    release_label = subparsers.add_parser(
        "release-timelock-label",
        help=(
            "verify the external completion anchor and target drand round before "
            "exclusive label decryption"
        ),
    )
    release_label.add_argument("--manifest", type=Path, required=True)
    release_label.add_argument("--corpus-id", required=True)
    release_label.add_argument("--custody-seal", type=Path, required=True)
    release_label.add_argument("--encryption-receipt", type=Path, required=True)
    release_label.add_argument("--completion-receipt", type=Path, required=True)
    release_label.add_argument("--completion-anchor-record", type=Path, required=True)
    release_label.add_argument("--completion-anchor-receipt", type=Path, required=True)
    release_label.add_argument(
        "--suite-namespace",
        type=Path,
        required=True,
        help="canonical suite namespace at externally attested ONLINE_COMPLETE",
    )
    release_label.add_argument("--ciphertext", type=Path, required=True)
    release_label.add_argument("--tle-binary", type=Path, required=True)
    release_label.add_argument("--plaintext-output", type=Path, required=True)
    release_label.add_argument("--receipt", type=Path, required=True)
    release_label.add_argument("--timeout-seconds", type=int, default=60)
    release_label.add_argument(
        "--max-ciphertext-bytes",
        type=int,
        default=128 * 1024 * 1024,
    )
    release_label.add_argument(
        "--max-plaintext-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )

    verify_custody = subparsers.add_parser(
        "verify-custody-seal-receipt",
        help="verify a canonical custody seal against its manifest pins",
    )
    verify_custody.add_argument("--manifest", type=Path, required=True)
    verify_custody.add_argument("--receipt", type=Path, required=True)
    verify_custody.add_argument(
        "--allow-draft",
        action="store_true",
        help="verify commitments before the manifest pins the receipt file",
    )

    online_custody = subparsers.add_parser(
        "verify-online-custody",
        help="admit an online runner without opening plaintext-label artifacts",
    )
    online_custody.add_argument("--manifest", type=Path, required=True)
    online_custody.add_argument("--custody-seal-receipt", type=Path, required=True)
    online_custody.add_argument("--sealed-run-receipt", type=Path, required=True)
    online_custody.add_argument(
        "--artifact-verification-receipt",
        type=Path,
        required=True,
    )
    online_custody.add_argument("--artifact-root", type=Path, required=True)
    online_custody.add_argument("--artifact-map", type=Path, required=True)
    online_custody.add_argument("--runner-identity", required=True)
    online_custody.add_argument("--receipt", type=Path, required=True)

    sealed_corpus = subparsers.add_parser(
        "run-sealed-corpus",
        help="execute the single config-bound confirmatory corpus attempt",
    )
    sealed_corpus.add_argument("--config", type=Path, required=True)
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
        pins = {str(artifact["id"]): str(artifact["sha256"]) for artifact in payload["artifacts"]}
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
        from .zenodo_publication import verify_production_protocol_registration

        verified_registration = verify_production_protocol_registration(
            args.registration_package,
            registration_receipt_path=args.protocol_registration_receipt,
            registration_record_path=args.protocol_registration_record,
        )
        receipt = begin_sealed_run(
            args.manifest,
            args.lock,
            runner_identity=args.runner_identity,
            artifact_verification_receipt_path=(args.artifact_verification_receipt),
            artifact_root=args.artifact_root,
            local_artifact_map_path=args.artifact_map,
            verified_protocol_registration=verified_registration,
        )
        print(f"sealed run opened for protocol {receipt.protocol_version}")
        print(f"manifest sha256: {receipt.manifest_sha256}")
        print(f"receipt: {receipt.receipt_uri}")
        return 0
    if args.command == "create-custody-seal-receipt":
        payload = load_study_manifest(args.manifest)
        receipt = custody_seal_receipt_from_manifest(
            payload,
            drand_chain_hash=args.drand_chain_hash,
            drand_round=args.drand_round,
        )
        write_custody_seal_receipt(receipt, args.receipt)
        print(f"wrote custody seal receipt: {args.receipt}")
        print(f"receipt sha256: {receipt.receipt_sha256}")
        print(f"manifest artifact sha256: {receipt.file_sha256}")
        return 0
    if args.command == "encrypt-timelock-label":
        payload = load_study_manifest(args.manifest)
        receipt = encrypt_timelock_label(
            payload,
            corpus_id=args.corpus_id,
            plaintext_path=args.plaintext,
            tle_binary_path=args.tle_binary,
            drand_network=args.drand_network,
            drand_chain_hash=args.drand_chain_hash,
            drand_round=args.drand_round,
            ciphertext_path=args.ciphertext,
            timeout_seconds=args.timeout_seconds,
            max_plaintext_bytes=args.max_plaintext_bytes,
            max_ciphertext_bytes=args.max_ciphertext_bytes,
        )
        write_timelock_encryption_receipt(receipt, args.receipt)
        print(f"wrote timelock ciphertext: {args.ciphertext}")
        print(f"ciphertext artifact sha256: {receipt.ciphertext_sha256}")
        print(f"encryption receipt sha256: {receipt.receipt_sha256}")
        print(f"encryption receipt: {args.receipt}")
        return 0
    if args.command == "verify-timelock-encryption-receipt":
        payload = load_study_manifest(args.manifest)
        receipt = load_timelock_encryption_receipt(args.receipt)
        custody_seal = (
            None if args.custody_seal is None else load_custody_seal_receipt(args.custody_seal)
        )
        verify_timelock_encryption_receipt(
            receipt,
            payload,
            custody_seal=custody_seal,
            require_frozen=not args.allow_draft,
        )
        print(f"valid timelock encryption receipt: {args.receipt}")
        print(f"ciphertext artifact sha256: {receipt.ciphertext_sha256}")
        return 0
    if args.command == "release-timelock-label":
        if os.path.lexists(args.receipt):
            raise ValueError("decryption receipt already exists; overwrite is forbidden")
        payload = load_study_manifest(args.manifest)
        suite_verifier = GitHubSuiteEvidenceVerifier(args.suite_namespace)
        verified_suite = verify_suite_state(
            args.suite_namespace,
            verifier=suite_verifier,
            expected_state="ONLINE_COMPLETE",
        )
        completion_receipt = load_prediction_completion_receipt(args.completion_receipt)
        verified_anchor = verify_prediction_completion_anchor(
            completion_receipt,
            anchor_record_path=args.completion_anchor_record,
            anchor_receipt_path=args.completion_anchor_receipt,
        )
        verified_release = release_timelock_label(
            payload,
            corpus_id=args.corpus_id,
            custody_seal=load_custody_seal_receipt(args.custody_seal),
            encryption_receipt=load_timelock_encryption_receipt(args.encryption_receipt),
            verified_completion_anchor=verified_anchor,
            verified_suite_completion=verified_suite,
            ciphertext_path=args.ciphertext,
            tle_binary_path=args.tle_binary,
            plaintext_output_path=args.plaintext_output,
            timeout_seconds=args.timeout_seconds,
            max_ciphertext_bytes=args.max_ciphertext_bytes,
            max_plaintext_bytes=args.max_plaintext_bytes,
        )
        write_timelock_decryption_receipt(verified_release.receipt, args.receipt)
        print(f"released canonical plaintext labels: {args.plaintext_output}")
        print(f"plaintext sha256: {verified_release.receipt.plaintext_sha256}")
        print(f"decryption receipt sha256: {verified_release.receipt.receipt_sha256}")
        print(f"decryption receipt: {args.receipt}")
        return 0
    if args.command == "verify-custody-seal-receipt":
        payload = load_study_manifest(args.manifest)
        receipt = load_custody_seal_receipt(args.receipt)
        verify_custody_seal_receipt(
            receipt,
            payload,
            require_frozen=not args.allow_draft,
            require_manifest_pin=not args.allow_draft,
        )
        print(f"valid custody seal receipt: {args.receipt}")
        print(f"receipt sha256: {receipt.receipt_sha256}")
        return 0
    if args.command == "verify-online-custody":
        receipt = admit_online_custody(
            args.manifest,
            custody_seal_receipt_path=args.custody_seal_receipt,
            sealed_run_receipt_path=args.sealed_run_receipt,
            artifact_verification_receipt_path=(args.artifact_verification_receipt),
            artifact_root=args.artifact_root,
            local_artifact_map_path=args.artifact_map,
            runner_identity=args.runner_identity,
        )
        write_online_custody_admission_receipt(receipt, args.receipt)
        print(f"verified {len(receipt.verified_artifact_ids)} online-safe artifacts")
        print(f"manifest sha256: {receipt.manifest_sha256}")
        print(f"receipt sha256: {receipt.receipt_sha256}")
        print(f"receipt: {args.receipt}")
        return 0
    if args.command == "run-sealed-corpus":
        claim_bytes = sys.stdin.buffer.read(256 * 1024 + 1)
        if len(claim_bytes) > 256 * 1024:
            raise ExecutionClaimError("runtime claim receipt exceeds its fixed byte limit")
        runtime_claim = loads_runtime_claim_receipt(claim_bytes)
        completed = run_sealed_corpus_from_config(args.config, runtime_claim)
        print(f"completed sealed corpus attempt: {completed.output_root}")
        print(f"attempt sha256: {completed.attempt_receipt.receipt_sha256}")
        print(f"result sha256: {completed.result_receipt.receipt_sha256}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
