"""Run the v0.2 authorization-first synthetic development pilot."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fractal_ann_diagnostics.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["pilot", *sys.argv[1:]]))
