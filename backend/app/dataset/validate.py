"""Structural and quality validation of the benchmark against the label file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.dataset.labels import LabelFile


@dataclass
class ChannelCheck:
    chan_id: str
    spacecraft: str
    train_shape: list[int] | None = None
    test_shape: list[int] | None = None
    dtype: str | None = None
    test_len_matches_labels: bool | None = None
    nonfinite_train: int = 0
    nonfinite_test: int = 0
    value_min: float | None = None
    value_max: float | None = None
    train_value_std: float | None = None
    commands_binary: bool | None = None
    command_columns_active_train: int | None = None
    command_columns_active_test: int | None = None
    labelled_points: int = 0
    label_fraction: float = 0.0
    issues: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]
    channels: list[ChannelCheck]
    summary: dict

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self.summary,
            "channels": [asdict(c) for c in self.channels],
        }


def _load(path: Path) -> np.ndarray:
    # allow_pickle=False: never execute code from downloaded files.
    return np.load(path, allow_pickle=False)


def validate_dataset(root: Path, labels: LabelFile) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[ChannelCheck] = []

    errors.extend(f"labels: {i}" for i in labels.issues)
    for chan, n in labels.duplicates.items():
        warnings.append(
            f"labels: channel {chan} appears in {n} rows; intervals merged into one stream "
            f"(see docs/dataset.md)"
        )

    present = {
        split: {p.stem for p in (root / split).glob("*.npy")} if (root / split).exists() else set()
        for split in ("train", "test")
    }
    expected = set(labels.channels)
    for split in ("train", "test"):
        missing = sorted(expected - present[split])
        extra = sorted(present[split] - expected)
        if missing:
            errors.append(f"{split}: missing {len(missing)} channel files: {missing[:10]}")
        if extra:
            warnings.append(f"{split}: {len(extra)} files not referenced by labels (ignored): {extra[:10]}")

    features_by_sc: dict[str, set[int]] = {}
    for chan, lab in sorted(labels.channels.items()):
        c = ChannelCheck(chan_id=chan, spacecraft=lab.spacecraft)
        try:
            tr = _load(root / "train" / f"{chan}.npy")
            te = _load(root / "test" / f"{chan}.npy")
        except FileNotFoundError:
            checks.append(c)
            continue
        except Exception as exc:  # corrupted file
            c.issues.append(f"unreadable: {exc}")
            errors.append(f"{chan}: unreadable ({exc})")
            checks.append(c)
            continue

        c.train_shape, c.test_shape = list(tr.shape), list(te.shape)
        c.dtype = str(te.dtype)
        if tr.ndim != 2 or te.ndim != 2:
            errors.append(f"{chan}: expected 2-D arrays, got {tr.shape} / {te.shape}")
            checks.append(c)
            continue
        if tr.shape[1] != te.shape[1]:
            errors.append(f"{chan}: train/test feature mismatch {tr.shape[1]} vs {te.shape[1]}")
        features_by_sc.setdefault(lab.spacecraft, set()).add(te.shape[1])

        c.test_len_matches_labels = te.shape[0] == lab.num_values
        if not c.test_len_matches_labels:
            errors.append(f"{chan}: test length {te.shape[0]} != num_values {lab.num_values}")

        c.nonfinite_train = int((~np.isfinite(tr)).sum())
        c.nonfinite_test = int((~np.isfinite(te)).sum())
        if c.nonfinite_train or c.nonfinite_test:
            errors.append(f"{chan}: non-finite values (train {c.nonfinite_train}, test {c.nonfinite_test})")

        both = np.concatenate([tr[:, 0], te[:, 0]])
        c.value_min, c.value_max = float(np.nanmin(both)), float(np.nanmax(both))
        if c.value_min < -1.0001 or c.value_max > 1.0001:
            warnings.append(f"{chan}: telemetry outside [-1, 1] ({c.value_min:.3f}, {c.value_max:.3f})")
        c.train_value_std = float(np.nanstd(tr[:, 0]))
        if c.train_value_std == 0.0:
            c.issues.append("constant telemetry value in training split")
            warnings.append(f"{chan}: training telemetry is constant")

        cmds = np.concatenate([tr[:, 1:], te[:, 1:]])
        c.commands_binary = bool(np.isin(cmds, (0.0, 1.0)).all()) if cmds.size else True
        if not c.commands_binary:
            warnings.append(f"{chan}: command columns are not strictly 0/1")
        c.command_columns_active_train = int((tr[:, 1:] != 0).any(axis=0).sum())
        c.command_columns_active_test = int((te[:, 1:] != 0).any(axis=0).sum())

        c.labelled_points = sum(iv.length for iv in lab.intervals)
        c.label_fraction = c.labelled_points / max(1, lab.num_values)
        checks.append(c)

    for sc, feats in features_by_sc.items():
        if len(feats) != 1:
            warnings.append(f"{sc}: channels have differing feature counts {sorted(feats)}")

    by_sc: dict[str, dict] = {}
    for c in checks:
        s = by_sc.setdefault(
            c.spacecraft,
            {"channels": 0, "train_values": 0, "test_values": 0, "labelled_points": 0,
             "sequences": 0, "features": sorted(features_by_sc.get(c.spacecraft, []))},
        )
        s["channels"] += 1
        s["train_values"] += c.train_shape[0] if c.train_shape else 0
        s["test_values"] += c.test_shape[0] if c.test_shape else 0
        s["labelled_points"] += c.labelled_points
        s["sequences"] += len(labels.channels[c.chan_id].intervals)

    summary = {
        "label_rows": labels.raw_rows,
        "label_sequences_raw": labels.raw_sequences,
        "label_sequences_after_merge": sum(len(ch.intervals) for ch in labels.channels.values()),
        "unique_channels": len(labels.channels),
        "by_spacecraft": by_sc,
    }
    return ValidationReport(ok=not errors, errors=errors, warnings=warnings, channels=checks, summary=summary)
