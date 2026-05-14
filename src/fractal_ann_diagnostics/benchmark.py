"""ANN-benchmarks corpus harness for descriptor calibration.

Reads the standard ANN-benchmarks HDF5 datasets (SIFT, GIST, GloVe, DEEP1B
subsets, MS-MARCO embeddings, etc.) and provides a uniform interface for
descriptor extraction and downstream evaluation.

Reference
---------
- Aumüller, M., Bernhardsson, E., Faithfull, A. (2020). ANN-Benchmarks:
  A benchmarking tool for approximate nearest neighbor algorithms.
  Information Systems, 87.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AnnDataset:
    """One ANN-benchmarks dataset, loaded and ready for descriptor extraction."""

    name: str
    train: np.ndarray
    test: np.ndarray
    neighbors: np.ndarray
    distances: np.ndarray
    distance_metric: str


CANONICAL_DATASETS: tuple[str, ...] = (
    "sift-128-euclidean",
    "gist-960-euclidean",
    "glove-25-angular",
    "glove-100-angular",
    "glove-200-angular",
    "deep-image-96-angular",
    "fashion-mnist-784-euclidean",
    "lastfm-64-dot",
    "mnist-784-euclidean",
    "nytimes-256-angular",
)

ANN_BENCHMARKS_BASE_URL = "https://ann-benchmarks.com"


_USER_AGENT = "fractal-ann-diagnostics/0.1.0 (+https://github.com/mhdk1602/fractal-ann-diagnostics)"


def _download(url: str, dst: Path) -> None:
    """Download ``url`` to ``dst`` atomically (write to a .part file first).

    Sends a non-default User-Agent because ann-benchmarks.com sits behind
    Cloudflare and 403s the stock Python urllib UA.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:  # noqa: S310
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except urllib.error.URLError as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    tmp.replace(dst)


def load_ann_benchmark(name: str, cache_dir: Path) -> AnnDataset:
    """Load a single ANN-benchmarks dataset by name.

    Downloads the HDF5 from ``https://ann-benchmarks.com/<name>.hdf5`` into
    ``cache_dir`` on first call. Subsequent calls reuse the cached file. The
    set of canonical names is in :data:`CANONICAL_DATASETS`.

    Parameters
    ----------
    name : str
        Dataset slug, e.g. ``"mnist-784-euclidean"``.
    cache_dir : Path
        Directory that holds the downloaded HDF5 file.

    Returns
    -------
    AnnDataset
        Parsed contents (``train``, ``test``, ``neighbors``, ``distances``,
        ``distance_metric``).
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for ANN-benchmarks loading. Install with: "
            "pip install h5py"
        ) from e

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.hdf5"
    if not path.exists():
        url = f"{ANN_BENCHMARKS_BASE_URL}/{name}.hdf5"
        _download(url, path)

    with h5py.File(path, "r") as f:
        distance_metric = f.attrs.get("distance", "unknown")
        if isinstance(distance_metric, bytes):
            distance_metric = distance_metric.decode("utf-8")
        return AnnDataset(
            name=name,
            train=np.asarray(f["train"], dtype=np.float32),
            test=np.asarray(f["test"], dtype=np.float32),
            neighbors=np.asarray(f["neighbors"], dtype=np.int64),
            distances=np.asarray(f["distances"], dtype=np.float32),
            distance_metric=str(distance_metric),
        )


def evaluate_recall(
    index_handle, queries: np.ndarray, ground_truth: np.ndarray, k: int = 10
) -> float:
    """Compute recall@k against ground-truth neighbour ids.

    Stub until index backends are wired up at v0.2.0.
    """
    raise NotImplementedError("evaluate_recall lands with the index backends at v0.2.0.")


def descriptor_panel_for_corpus(cache_dir: Path) -> dict[str, dict]:
    """Compute the descriptor panel for every canonical ANN-benchmarks dataset.

    Returns a nested dict[dataset_name][descriptor_name] -> value, suitable
    for the calibration step at v0.2.0.
    """
    raise NotImplementedError(
        "descriptor_panel_for_corpus is a v0.2.0 batch driver; for v0.1.0 see "
        "experiments/calibrate_v0_1_0.py for the two-dataset version."
    )
