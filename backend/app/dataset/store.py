"""Read access to the installed benchmark (``data/raw/smap_msl``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.dataset.labels import ChannelLabels, LabelFile, load_label_file


class DatasetNotAvailable(RuntimeError):
    pass


def dataset_root() -> Path:
    return get_settings().raw_dir


def is_available() -> bool:
    root = dataset_root()
    return (root / "labeled_anomalies.csv").exists() and (root / "train").is_dir() and (root / "test").is_dir()


@lru_cache(maxsize=4)
def _labels_cached(path: str, mtime: float) -> LabelFile:
    return load_label_file(Path(path))


def labels() -> LabelFile:
    if not is_available():
        raise DatasetNotAvailable("Dataset not installed. Run setup.ps1 -DownloadData or POST /api/dataset/acquire.")
    p = dataset_root() / "labeled_anomalies.csv"
    return _labels_cached(str(p), p.stat().st_mtime)


def channel_labels(chan_id: str) -> ChannelLabels:
    lf = labels()
    if chan_id not in lf.channels:
        raise KeyError(chan_id)
    return lf.channels[chan_id]


@lru_cache(maxsize=48)
def _load_cached(path: str, mtime: float) -> np.ndarray:
    arr = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    arr.setflags(write=False)
    return arr


def load_split(chan_id: str, split: str) -> np.ndarray:
    """Return the (timesteps, features) float32 array. Column 0 is telemetry."""
    if split not in ("train", "test"):
        raise ValueError(split)
    if not is_available():
        raise DatasetNotAvailable("Dataset not installed.")
    p = dataset_root() / split / f"{chan_id}.npy"
    if not p.exists():
        raise KeyError(f"{split}/{chan_id}.npy")
    return _load_cached(str(p), p.stat().st_mtime)


def label_mask(chan_id: str, length: int | None = None) -> np.ndarray:
    lab = channel_labels(chan_id)
    n = length or lab.num_values
    mask = np.zeros(n, dtype=bool)
    for iv in lab.intervals:
        mask[iv.start : iv.end + 1] = True
    return mask
