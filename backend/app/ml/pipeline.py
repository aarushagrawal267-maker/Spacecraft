"""End-to-end detection for one channel: fit -> score -> smooth -> threshold -> evaluate.

This module has no database access so it can run inside a training subprocess.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.dataset import store
from app.ml import methods
from app.ml.evaluation.metrics import evaluate_channel
from app.ml.scoring.smoothing import ewma
from app.ml.scoring.thresholds import DynamicThreshold, Interval, StaticThreshold, ThresholdConfig
from app.ml.training import FitContext


@dataclass
class DetectionResult:
    eval_start: int
    smoothed: np.ndarray  # length N - eval_start
    threshold_high: np.ndarray
    threshold_low: np.ndarray
    flags: np.ndarray
    intervals: list[Interval]
    windows: list[dict]
    effective: dict = field(default_factory=dict)


def _fill_nan(e: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype=np.float64).copy()
    bad = ~np.isfinite(e)
    if bad.any():
        e[bad] = np.nanmean(e) if (~bad).any() else 0.0
    return e


def detect(
    y: np.ndarray,
    raw_scores: np.ndarray,
    thr: ThresholdConfig,
    eval_start: int = methods.EVAL_START,
    train_raw_scores: np.ndarray | None = None,
) -> DetectionResult:
    """Smooth and threshold ``raw_scores`` (aligned with the full test stream ``y``)."""
    e = _fill_nan(raw_scores[eval_start:])
    e_s = ewma(e, thr.span)
    y_eval = np.asarray(y[eval_start:], dtype=np.float64)
    if thr.mode == "static":
        if train_raw_scores is None:
            raise ValueError("Static thresholding needs training-split scores")
        tr = np.asarray(train_raw_scores, dtype=np.float64)
        tr = tr[np.isfinite(tr)]
        det = StaticThreshold.from_training_errors(ewma(tr, thr.span), len(e_s), thr.static_k)
        det.push(e_s)
        effective = {"mode": "static", "epsilon": det.epsilon, "k": thr.static_k}
    else:
        det = DynamicThreshold(len(e_s), thr)
        det.push(e_s, y_eval)
        det.finalize()
        effective = {"mode": "ndt", "window_size": det.ws, "batch_size": det.bs, "n_windows": det.n_windows + 1,
                     "span": thr.span}
    return DetectionResult(
        eval_start=eval_start,
        smoothed=e_s,
        threshold_high=det.threshold_high,
        threshold_low=det.threshold_low,
        flags=det.flags,
        intervals=det.intervals(offset=eval_start),
        windows=[asdict(w) for w in det.windows],
        effective=effective,
    )


def label_dicts(chan_id: str) -> list[dict]:
    lab = store.channel_labels(chan_id)
    return [{"start": iv.start, "end": iv.end, "cls": iv.cls, "merged_from": iv.merged_from} for iv in lab.intervals]


def annotate(intervals: list[Interval], labels: list[dict]) -> list[dict]:
    out = []
    for iv in intervals:
        matched = [lab for lab in labels if iv.start <= lab["end"] and lab["start"] <= iv.end]
        out.append({
            **asdict(iv),
            "outcome": "matches_label" if matched else "no_label_overlap",
            "matched_labels": [{"start": m["start"], "end": m["end"], "cls": m["cls"]} for m in matched],
        })
    return out


@dataclass
class ChannelOutcome:
    chan_id: str
    model_path: str | None
    arrays_path: str
    metrics: dict
    detections: list[dict]
    train_time_s: float
    infer_time_s: float
    epochs_run: int | None
    best_epoch: int | None
    best_val_loss: float | None
    effective_threshold: dict


def save_arrays(path: Path, *, actual, prediction, raw, det: DetectionResult) -> Path:
    n = len(actual)
    pad = det.eval_start

    def full(a):
        out = np.full(n, np.nan, dtype=np.float32)
        out[pad:] = a
        return out

    np.savez_compressed(
        path,
        prediction=(prediction.astype(np.float32) if prediction is not None else np.full(n, np.nan, np.float32)),
        raw_score=np.asarray(raw, dtype=np.float32),
        smoothed=full(det.smoothed),
        threshold_high=full(det.threshold_high),
        threshold_low=full(det.threshold_low),
        flags=np.concatenate([np.zeros(pad, bool), det.flags]),
        eval_start=np.array(pad),
    )
    return path


def run_channel(
    method: str,
    chan_id: str,
    cfg: dict,
    thr: ThresholdConfig,
    ctx: FitContext,
    out_dir: Path,
) -> ChannelOutcome:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = store.load_split(chan_id, "train")
    test = store.load_split(chan_id, "test")

    t0 = time.perf_counter()
    fitted = methods.fit(method, train, cfg, ctx)
    train_time = time.perf_counter() - t0
    model_path = methods.save(fitted, out_dir / chan_id)

    t1 = time.perf_counter()
    pred, raw = methods.score(fitted, test)
    train_raw = methods.score(fitted, train)[1] if thr.mode == "static" else None
    det = detect(test[:, 0], raw, thr, methods.EVAL_START, train_raw)
    infer_time = time.perf_counter() - t1

    labels = label_dicts(chan_id)
    metrics = evaluate_channel(
        [(iv.start, iv.end) for iv in det.intervals], labels, det.smoothed, len(test), det.eval_start
    )
    arrays_path = save_arrays(out_dir / f"{chan_id}.npz", actual=test[:, 0], prediction=pred, raw=raw, det=det)
    s = fitted.summary
    return ChannelOutcome(
        chan_id=chan_id,
        model_path=str(model_path) if model_path else None,
        arrays_path=str(arrays_path),
        metrics=metrics,
        detections=annotate(det.intervals, labels),
        train_time_s=train_time,
        infer_time_s=infer_time,
        epochs_run=s.epochs_run if s else None,
        best_epoch=s.best_epoch if s else None,
        best_val_loss=s.best_val_loss if s else None,
        effective_threshold=det.effective,
    )
