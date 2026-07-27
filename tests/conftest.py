"""Make ``src/fractal_ann_diagnostics`` importable in editable-free environments.

Adds the repo's ``src/`` directory to ``sys.path`` so tests run from a clean
checkout without requiring ``pip install -e .`` first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import gettempdir

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

_DEBUG_TEMPROOT_OVERRIDE: str | None = None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Keep synthetic bind targets outside the launcher's fixed ``/tmp`` tmpfs."""

    global _DEBUG_TEMPROOT_OVERRIDE
    if config.option.basetemp is not None or "PYTEST_DEBUG_TEMPROOT" in os.environ:
        return
    fixed_tmpfs_root = Path("/tmp").resolve()
    default_temp_root = Path(gettempdir()).resolve()
    alternate_temp_root = Path("/var/tmp").resolve()
    if alternate_temp_root == fixed_tmpfs_root or not (
        default_temp_root == fixed_tmpfs_root or default_temp_root.is_relative_to(fixed_tmpfs_root)
    ):
        return
    if not alternate_temp_root.is_dir() or not os.access(
        alternate_temp_root,
        os.W_OK | os.X_OK,
    ):
        return
    _DEBUG_TEMPROOT_OVERRIDE = str(alternate_temp_root)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = _DEBUG_TEMPROOT_OVERRIDE


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore the environment after pytest has registered normal temp retention."""

    del config
    global _DEBUG_TEMPROOT_OVERRIDE
    if (
        _DEBUG_TEMPROOT_OVERRIDE is not None
        and os.environ.get("PYTEST_DEBUG_TEMPROOT") == _DEBUG_TEMPROOT_OVERRIDE
    ):
        del os.environ["PYTEST_DEBUG_TEMPROOT"]
    _DEBUG_TEMPROOT_OVERRIDE = None
