"""Dataset acquisition: download, verify, record provenance, install read-only.

Downloads go to a fresh staging directory. Only ``.npy`` and ``.csv`` members are
extracted from archives, with path-traversal protection, and arrays are always
opened with ``allow_pickle=False``. The staged copy is validated against the
official label file before it replaces ``data/raw/smap_msl``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.core.config import get_settings
from app.dataset import sources
from app.dataset.labels import LabelFile, load_label_file
from app.dataset.validate import ValidationReport, validate_dataset

ProgressFn = Callable[[str, float], None]  # (message, fraction 0..1)

USER_AGENT = "heliograph-dataset-fetcher/1.0"


class AcquisitionError(RuntimeError):
    pass


@dataclass
class AcquisitionResult:
    ok: bool
    source: str | None
    message: str
    attempts: list[dict] = field(default_factory=list)
    report: ValidationReport | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _kaggle_auth_header() -> dict:
    """Optional Kaggle credentials (env vars or ~/.kaggle/kaggle.json). Public datasets
    usually download without them; credentials help if Kaggle requires sign-in."""
    import base64

    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not (user and key) and cfg.exists():
        try:
            d = json.loads(cfg.read_text())
            user, key = d.get("username"), d.get("key")
        except Exception:  # noqa: BLE001
            pass
    if user and key:
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()}
    return {}


def _download(url: str, dest: Path, timeout: float = 60.0, retries: int = 3, headers: dict | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            return
        except Exception as exc:  # noqa: BLE001 - network errors vary by platform
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise AcquisitionError(f"Download failed for {url}: {last_err}")


def _safe_extract_npy(zip_path: Path, staging: Path) -> int:
    """Extract ``train/*.npy`` and ``test/*.npy`` members (any nesting depth)."""
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if any(p in ("..", "") for p in parts) or Path(info.filename).is_absolute():
                continue
            name = parts[-1]
            parent = parts[-2] if len(parts) >= 2 else ""
            if not name.endswith(".npy") or parent not in ("train", "test"):
                continue
            target = staging / parent / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def _from_kaggle(staging: Path, labels: LabelFile, progress: ProgressFn) -> None:
    progress("Downloading archive from Kaggle", 0.1)
    zip_path = staging / "kaggle.zip"
    _download(sources.KAGGLE_URL, zip_path, timeout=300, retries=2, headers=_kaggle_auth_header())
    progress("Extracting archive", 0.6)
    n = _safe_extract_npy(zip_path, staging)
    zip_path.unlink(missing_ok=True)
    if n == 0:
        raise AcquisitionError("Kaggle archive contained no train/test .npy files")


def _from_mirror(staging: Path, labels: LabelFile, progress: ProgressFn) -> None:
    chans = sorted(labels.channels)
    total = len(chans) * 2
    done = 0
    for chan in chans:
        for split in ("train", "test"):
            _download(f"{sources.MIRROR_BASE}/{split}/{chan}.npy", staging / split / f"{chan}.npy")
            done += 1
            progress(f"Downloaded {split}/{chan}.npy", 0.05 + 0.85 * done / total)


def _from_manual(staging: Path, labels: LabelFile, progress: ProgressFn) -> None:
    settings = get_settings()
    incoming = settings.incoming_dir
    zips = sorted(incoming.glob("*.zip")) if incoming.exists() else []
    if zips:
        progress(f"Extracting {zips[0].name}", 0.3)
        if _safe_extract_npy(zips[0], staging) == 0:
            raise AcquisitionError(f"{zips[0].name} contained no train/test .npy files")
        return
    # Already-extracted folder: data/incoming/**/train/*.npy
    trains = [p for p in incoming.rglob("train") if p.is_dir()] if incoming.exists() else []
    if not trains:
        raise AcquisitionError("No data.zip or train/ folder found in data/incoming/")
    root = trains[0].parent
    for split in ("train", "test"):
        (staging / split).mkdir(parents=True, exist_ok=True)
        for f in (root / split).glob("*.npy"):
            shutil.copy2(f, staging / split / f.name)


_FETCHERS = {"kaggle": _from_kaggle, "github_mirror": _from_mirror, "manual": _from_manual}


def _make_readonly(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def _make_writable(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


def _write_manifest(staging: Path) -> dict:
    files = {}
    for p in sorted(staging.rglob("*")):
        if p.is_file() and p.name not in ("manifest.json",):
            rel = p.relative_to(staging).as_posix()
            files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    combined = hashlib.sha256(
        "".join(f"{k}:{v['sha256']}\n" for k, v in files.items()).encode()
    ).hexdigest()
    manifest = {"files": files, "dataset_checksum": combined, "n_files": len(files)}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def acquire(source: str = "auto", progress: ProgressFn | None = None) -> AcquisitionResult:
    """Fetch the benchmark into ``data/raw/smap_msl``. Existing valid data is replaced
    only if the new copy validates."""
    settings = get_settings()
    settings.ensure_dirs()
    progress = progress or (lambda msg, frac: None)
    order = ["kaggle", "github_mirror", "manual"] if source == "auto" else [source]
    for key in order:
        if key not in _FETCHERS:
            raise ValueError(f"Unknown source {key!r}")

    attempts: list[dict] = []
    raw_parent = settings.raw_dir.parent
    raw_parent.mkdir(parents=True, exist_ok=True)

    for key in order:
        staging = raw_parent / f".staging-{key}-{int(time.time())}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            progress("Downloading official label file (Telemanom, pinned commit)", 0.02)
            label_path = staging / "labeled_anomalies.csv"
            _download(sources.LABELS_URL, label_path)
            digest = sha256_file(label_path)
            if digest != sources.LABELS_SHA256:
                raise AcquisitionError(
                    f"labeled_anomalies.csv checksum mismatch ({digest[:12]}…); refusing to continue"
                )
            labels = load_label_file(label_path)

            _FETCHERS[key](staging, labels, progress)

            progress("Validating files against the label file", 0.92)
            report = validate_dataset(staging, labels)
            if not report.ok:
                raise AcquisitionError(
                    "Validation failed: " + "; ".join(report.errors[:5])
                )
            manifest = _write_manifest(staging)
            info = sources.SOURCES[key]
            provenance = {
                "source": key,
                "source_title": info.title,
                "source_url": info.url,
                "source_note": info.note,
                "labels_url": sources.LABELS_URL,
                "labels_sha256": sources.LABELS_SHA256,
                "telemanom_commit": sources.TELEMANOM_COMMIT,
                "mirror_commit": sources.MIRROR_COMMIT if key == "github_mirror" else None,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "dataset_checksum": manifest["dataset_checksum"],
                "license_note": sources.LICENSE_NOTE,
            }
            (staging / "provenance.json").write_text(json.dumps(provenance, indent=1))
            (staging / "validation_report.json").write_text(json.dumps(report.to_dict(), indent=1))

            # Swap in atomically-ish: move old copy aside, move staging in, delete old.
            if settings.raw_dir.exists():
                _make_writable(settings.raw_dir)
                old = raw_parent / f".old-{int(time.time())}"
                settings.raw_dir.rename(old)
                shutil.rmtree(old, ignore_errors=True)
            staging.rename(settings.raw_dir)
            _make_readonly(settings.raw_dir)
            attempts.append({"source": key, "ok": True, "message": "validated"})
            progress("Dataset ready", 1.0)
            return AcquisitionResult(True, key, f"Dataset acquired from {key}", attempts, report)
        except Exception as exc:  # noqa: BLE001 - report every failure reason to the user
            attempts.append({"source": key, "ok": False, "message": str(exc)})
            _make_writable(staging) if staging.exists() else None
            shutil.rmtree(staging, ignore_errors=True)
            progress(f"{key} failed: {exc}", 0.0)
    return AcquisitionResult(False, None, "All sources failed", attempts)


def load_provenance() -> dict | None:
    p = get_settings().raw_dir / "provenance.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_manifest() -> dict | None:
    p = get_settings().raw_dir / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None
