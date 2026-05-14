"""Make ``src/fractal_ann_diagnostics`` importable in editable-free environments.

Adds the repo's ``src/`` directory to ``sys.path`` so tests run from a clean
checkout without requiring ``pip install -e .`` first.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
