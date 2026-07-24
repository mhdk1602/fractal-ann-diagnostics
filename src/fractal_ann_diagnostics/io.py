"""Vector dataset loaders. HDF5 (ANN-benchmarks), fvecs / bvecs (BIGANN, SIFT)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_fvecs(path: Path) -> np.ndarray:
    """Read a .fvecs file (the canonical SIFT / GIST format).

    Each vector is stored as: int32 d, followed by d float32 values.
    """
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    n = len(raw) // (d + 1)
    return raw.reshape(n, d + 1)[:, 1:].view(np.float32).astype(np.float64)


def load_bvecs(path: Path) -> np.ndarray:
    """Read a .bvecs file (uint8 vectors, BIGANN base set).

    Each vector: int32 d, followed by d uint8 values.
    """
    with open(path, "rb") as f:
        d = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        f.seek(0)
        raw = np.fromfile(f, dtype=np.uint8)
    record_size = 4 + d
    n = len(raw) // record_size
    return raw.reshape(n, record_size)[:, 4:].astype(np.float64)


def load_hdf5_ann(path: Path) -> dict:
    """Load an ANN-benchmarks HDF5 file.

    Returns a dict with keys ``train``, ``test``, ``neighbors``, ``distances``,
    and the ``distance`` attribute. Requires the ``h5py`` extra:

        pip install fractal-ann-diagnostics[benchmarks]
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 loading. Install with: "
            "pip install fractal-ann-diagnostics[benchmarks]"
        ) from e

    with h5py.File(path, "r") as f:
        return {
            "train": np.asarray(f["train"]),
            "test": np.asarray(f["test"]),
            "neighbors": np.asarray(f["neighbors"]),
            "distances": np.asarray(f["distances"]),
            "distance": f.attrs.get("distance", "unknown"),
        }
