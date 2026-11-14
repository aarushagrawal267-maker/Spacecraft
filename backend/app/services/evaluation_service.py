"""Experiment-level aggregation of per-channel evaluation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.config import resolve_path
from app.dataset import store
from app.db.models import ChannelRun
from app.ml.evaluation.metrics import aggregate, clip_labels, intervals_to_mask, score_curves


def _pooled_curves(runs: list[ChannelRun]) -> dict | None:
    """Threshold-free PR/ROC over all evaluated points of the given runs.

    Each channel's smoothed score is standardised by its own mean/sd (label-free)
    so scores from different channels are comparable before pooling.
    """
    scores, masks = [], []
    for r in runs:
        if not r.arrays_path or not resolve_path(r.arrays_path).exists():
            continue
        with np.load(resolve_path(r.arrays_path)) as a:
            es = int(a["eval_start"])
            s = a["smoothed"][es:].astype(np.float64)
        if not np.isfinite(s).any():
            continue
        sd = np.nanstd(s)
        scores.append((s - np.nanmean(s)) / (sd if sd > 0 else 1.0))
        lab = store.channel_labels(r.chan_id)
        labs = clip_labels([{"start": iv.start, "end": iv.end} for iv in lab.intervals], es)
        masks.append(intervals_to_mask([(x["start"], x["end"]) for x in labs], len(s), es))
    if not scores:
        return None
    return score_curves(np.concatenate(scores), np.concatenate(masks))


def experiment_metrics(runs: list[ChannelRun], spacecraft_of: dict[str, str]) -> dict:
    done = [r for r in runs if r.status == "completed" and r.metrics]
    out: dict = {
        "n_channels": len(runs),
        "n_completed": len(done),
        "n_failed": sum(r.status == "failed" for r in runs),
        "overall": aggregate([r.metrics for r in done]) if done else None,
        "by_spacecraft": {},
        "curves": {},
    }
    try:
        out["curves"]["overall"] = _pooled_curves(done)
    except Exception as exc:  # noqa: BLE001 - curves are optional; never fail an experiment over them
        out["curves_error"] = str(exc)
    for sc in sorted({spacecraft_of.get(r.chan_id, "?") for r in done}):
        sub = [r for r in done if spacecraft_of.get(r.chan_id) == sc]
        out["by_spacecraft"][sc] = aggregate([r.metrics for r in sub])
        try:
            out["curves"][sc] = _pooled_curves(sub)
        except Exception:  # noqa: BLE001
            out["curves"][sc] = None
    return out
