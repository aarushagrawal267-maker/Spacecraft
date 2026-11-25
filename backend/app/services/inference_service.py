"""Loading trained models, live inference, and series payloads for charts."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import get_settings, resolve_path
from app.dataset import store
from app.db.models import ChannelRun, Experiment
from app.ml import methods
from app.ml.device import resolve_device
from app.ml.evaluation.metrics import evaluate_channel
from app.ml.pipeline import annotate, detect, label_dicts
from app.ml.scoring.thresholds import ThresholdConfig


class InferenceError(RuntimeError):
    pass


class InvalidInferenceRequest(InferenceError):
    """The caller sent unusable settings (e.g. an out-of-range threshold override)."""


def to_list(a: np.ndarray | None, decimals: int = 5) -> list | None:
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    r = np.round(a, decimals)
    return [None if not np.isfinite(v) else float(v) for v in r]


def get_run(db: Session, run_id: int) -> tuple[ChannelRun, Experiment]:
    run = db.get(ChannelRun, run_id)
    if run is None:
        raise KeyError(run_id)
    return run, run.experiment


def load_fitted(run: ChannelRun, exp: Experiment) -> methods.Fitted:
    spec = methods.SPECS[exp.method]
    if spec.trainable:
        if not run.model_path or not resolve_path(run.model_path).exists():
            raise InferenceError(
                "This run has no saved model file on this machine. Re-run the experiment to train it locally."
            )
    device = resolve_device(get_settings().device)
    try:
        return methods.load(exp.method, exp.config, resolve_path(run.model_path), device, exp.seed)
    except Exception as exc:  # noqa: BLE001 - corrupt/truncated/incompatible checkpoint
        name = Path(run.model_path).name if run.model_path else "model"
        reason = (str(exc).strip().splitlines() or [""])[0][:160]
        raise InferenceError(
            f"The saved model file {name} could not be loaded ({type(exc).__name__}: {reason}). "
            "It may be corrupt or from an incompatible version; re-run the experiment to retrain it."
        ) from exc


def stored_series(run: ChannelRun, exp: Experiment) -> dict:
    if not run.arrays_path or not resolve_path(run.arrays_path).exists():
        raise InferenceError("Stored result arrays are not available for this run.")
    test = store.load_split(run.chan_id, "test")
    with np.load(resolve_path(run.arrays_path)) as a:
        arrays = {k: a[k] for k in a.files}
    return series_payload(
        run.chan_id, test, arrays, int(arrays["eval_start"]),
        detections=[
            {"id": d.id, "start": d.start, "end": d.end, "score": d.score, "direction": d.direction,
             "outcome": d.outcome, "threshold": d.threshold, "peak_error": d.peak_error}
            for d in run.detections
        ],
        source="stored",
        produces_prediction=methods.SPECS[exp.method].produces_prediction,
    )


def series_payload(chan_id, test, arrays: dict, eval_start: int, detections: list, source: str,
                   produces_prediction: bool) -> dict:
    cmd_activity = (test[:, 1:] != 0).sum(axis=1) if test.shape[1] > 1 else np.zeros(len(test))
    return {
        "chan_id": chan_id,
        "n": int(len(test)),
        "eval_start": eval_start,
        "source": source,
        "actual": to_list(test[:, 0]),
        "prediction": to_list(arrays.get("prediction")) if produces_prediction else None,
        "raw_score": to_list(arrays.get("raw_score")),
        "smoothed": to_list(arrays.get("smoothed")),
        "threshold_high": to_list(arrays.get("threshold_high")),
        "threshold_low": to_list(arrays.get("threshold_low")),
        "command_activity": cmd_activity.astype(int).tolist(),
        "detections": detections,
        "labels": label_dicts(chan_id),
    }


def live_inference(db: Session, run_id: int, threshold_overrides: dict | None) -> dict:
    """Execute the stored model on the test stream now, then threshold and evaluate."""
    run, exp = get_run(db, run_id)
    try:
        thr = ThresholdConfig.from_dict({**(exp.threshold_config or {}), **(threshold_overrides or {})})
    except (TypeError, ValueError) as exc:
        raise InvalidInferenceRequest(str(exc)) from exc
    test = store.load_split(run.chan_id, "test")
    t0 = time.perf_counter()
    fitted = load_fitted(run, exp)
    load_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    pred, raw = methods.score(fitted, test)
    score_ms = (time.perf_counter() - t1) * 1000
    train_raw = methods.score(fitted, store.load_split(run.chan_id, "train"))[1] if thr.mode == "static" else None
    t2 = time.perf_counter()
    det = detect(test[:, 0], raw, thr, methods.EVAL_START, train_raw)
    thr_ms = (time.perf_counter() - t2) * 1000
    labels = label_dicts(run.chan_id)
    metrics = evaluate_channel([(i.start, i.end) for i in det.intervals], labels, det.smoothed, len(test),
                               det.eval_start)
    n = len(test)

    def full(x):
        out = np.full(n, np.nan)
        out[det.eval_start:] = x
        return out

    arrays = {"prediction": pred, "raw_score": raw, "smoothed": full(det.smoothed),
              "threshold_high": full(det.threshold_high), "threshold_low": full(det.threshold_low)}
    dets = [{"id": None, **{k: v for k, v in d.items() if k in ("start", "end", "score", "direction", "outcome",
                                                               "threshold", "peak_error")}}
            for d in annotate(det.intervals, labels)]
    payload = series_payload(run.chan_id, test, arrays, det.eval_start, dets, "live_inference",
                             methods.SPECS[exp.method].produces_prediction)
    payload.update({
        "metrics": {k: metrics[k] for k in ("event", "point", "unscoreable_label_points", "n_evaluated")},
        "threshold_config": thr.to_dict(),
        "effective_threshold": det.effective,
        "timing_ms": {"load_model": load_ms, "score": score_ms, "threshold": thr_ms},
        "device": getattr(fitted.state, "device", "cpu") if fitted.state is not None else "cpu",
    })
    return payload


def detection_detail(db: Session, det_id: int, context: int = 300) -> dict:
    from app.db.models import Detection

    d = db.get(Detection, det_id)
    if d is None:
        raise KeyError(det_id)
    run, exp = d.run, d.run.experiment
    test = store.load_split(d.chan_id, "test")
    a0, a1 = max(0, d.start - context), min(len(test), d.end + context + 1)
    arrays = {}
    if run.arrays_path and resolve_path(run.arrays_path).exists():
        with np.load(resolve_path(run.arrays_path)) as a:
            arrays = {k: a[k][a0:a1] for k in a.files if k != "eval_start"}
    seg = slice(d.start, d.end + 1)
    actual_seg = test[seg, 0]
    stats = {
        "length": d.end - d.start + 1,
        "actual_min": float(actual_seg.min()),
        "actual_max": float(actual_seg.max()),
        "actual_mean": float(actual_seg.mean()),
    }
    if "smoothed" in arrays:
        s = arrays["smoothed"][d.start - a0 : d.end - a0 + 1]
        stats["smoothed_error_max"] = float(np.nanmax(s)) if np.isfinite(s).any() else None
    if "prediction" in arrays and np.isfinite(arrays["prediction"]).any():
        p = arrays["prediction"][d.start - a0 : d.end - a0 + 1]
        stats["mean_abs_forecast_error"] = float(np.nanmean(np.abs(actual_seg - p)))
    labels = label_dicts(d.chan_id)
    return {
        "id": d.id,
        "experiment_id": exp.id,
        "experiment_name": exp.name,
        "method": exp.method,
        "method_label": methods.SPECS[exp.method].label,
        "run_id": run.id,
        "chan_id": d.chan_id,
        "spacecraft": store.channel_labels(d.chan_id).spacecraft,
        "start": d.start,
        "end": d.end,
        "score": d.score,
        "peak_error": d.peak_error,
        "threshold": d.threshold,
        "direction": d.direction,
        "outcome": d.outcome,
        "matched_labels": d.matched_labels,
        "window": d.window,
        "stats": stats,
        "context": {
            "start": a0,
            "end": a1,
            "actual": to_list(test[a0:a1, 0]),
            "prediction": to_list(arrays.get("prediction")) if methods.SPECS[exp.method].produces_prediction else None,
            "raw_score": to_list(arrays.get("raw_score")),
            "smoothed": to_list(arrays.get("smoothed")),
            "threshold_high": to_list(arrays.get("threshold_high")),
            "threshold_low": to_list(arrays.get("threshold_low")),
        },
        "labels_in_context": [lab for lab in labels if lab["start"] <= a1 and lab["end"] >= a0],
        "confidence_note": (
            "The score is the margin of the smoothed error above the window threshold, divided by "
            "(window mean + window sd). It is a severity indicator, not a calibrated probability."
        ),
    }
