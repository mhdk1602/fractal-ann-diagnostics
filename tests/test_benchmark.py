from __future__ import annotations

from pathlib import Path

import pytest

from fractal_ann_diagnostics.benchmark import _download, load_ann_benchmark


@pytest.mark.parametrize(
    "url",
    (
        "http://ann-benchmarks.com/example.hdf5",
        "file:///tmp/example.hdf5",
        "https://user:secret@ann-benchmarks.com/example.hdf5",
        "https://ann-benchmarks.com/example.hdf5#fragment",
    ),
)
def test_download_rejects_non_https_or_ambiguous_urls(tmp_path: Path, url: str) -> None:
    destination = tmp_path / "example.hdf5"

    with pytest.raises(ValueError, match="benchmark download URL"):
        _download(url, destination)

    assert not destination.exists()


@pytest.mark.parametrize("name", ("../escape", "nested/name", "", ".hidden"))
def test_dataset_name_cannot_escape_the_cache(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="safe dataset slug"):
        load_ann_benchmark(name, tmp_path)
