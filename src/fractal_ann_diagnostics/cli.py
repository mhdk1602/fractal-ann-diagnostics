"""Command-line entry point for the reproducible development benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pilot import PilotConfig, write_pilot_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fractal-retrieval-governance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot", help="run the frozen synthetic development pilot")
    pilot.add_argument("--output", type=Path, default=Path("artifacts/pilot"))
    pilot.add_argument("--seed", type=int, default=PilotConfig.seed)
    pilot.add_argument("--documents", type=int, default=PilotConfig.n_documents)
    pilot.add_argument("--queries-per-role", type=int, default=PilotConfig.n_queries_per_role)
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
