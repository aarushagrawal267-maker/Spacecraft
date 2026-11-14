"""Evaluation of detected intervals against benchmark reference annotations.

Three views are kept strictly separate (docs/methodology.md):

Event-wise (the protocol of Hundman et al.)
    A labelled sequence overlapped by >= 1 detected sequence is a true positive
    (TP); otherwise a false negative (FN). A detected sequence overlapping no
    labelled sequence is a false positive (FP). True negatives are undefined,
    so a false-positive *rate* is not reported here; instead we give false
    alarms per 10,000 evaluated points.

Point-wise (strict)
    Every evaluated timestep is a sample: TP/FP/FN/TN, precision, recall, F1,
    TPR, FPR. No "point adjustment" is applied.

Interval quality
    Per labelled sequence: IoU with the union of overlapping detections,
    coverage, and detection delay (first detected index inside or touching the
    label minus label start; 0 if the detection began before the label).

Only points with index >= eval_start are evaluated; label points before
eval_start are reported separately as unscoreable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def _safe_div(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def f_beta(precision: float | None, recall: float | None, beta: float) -> float | None:
    if precision is None or recall is None:
        return None
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    return float((1 + b2) * precision * recall / (b2 * precision + recall))


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def clip_labels(labels: Sequence[dict], eval_start: int) -> list[dict]:
    """Labels restricted to the evaluated region (fully unscoreable labels dropped)."""
    out = []
    for lab in labels:
        if lab["end"] < eval_start:
            continue
        out.append({**lab, "start": max(lab["start"], eval_start)})
    return out


def event_metrics(
    detections: Sequence[tuple[int, int]],
    labels: Sequence[dict],
    n_evaluated: int,
) -> dict:
    """``labels``: dicts with start, end (inclusive), cls."""
    dets = [tuple(d) for d in detections]
    lab_iv = [(lab["start"], lab["end"]) for lab in labels]

    per_label = []
    tp = 0
    for lab, iv in zip(labels, lab_iv):
        hits = [d for d in dets if _overlaps(d, iv)]
        detected = bool(hits)
        tp += detected
        info = {"start": lab["start"], "end": lab["end"], "cls": lab.get("cls", "unknown"), "detected": detected}
        if detected:
            first = min(h[0] for h in hits)
            info["delay"] = int(max(0, first - iv[0]))
            union = np.zeros(max(iv[1], max(h[1] for h in hits)) + 1, dtype=bool)
            for h in hits:
                union[h[0] : h[1] + 1] = True
            lab_mask = np.zeros_like(union)
            lab_mask[iv[0] : iv[1] + 1] = True
            inter = int((union & lab_mask).sum())
            info["iou"] = inter / int((union | lab_mask).sum())
            info["coverage"] = inter / (iv[1] - iv[0] + 1)
        per_label.append(info)

    fp_dets = [d for d in dets if not any(_overlaps(d, iv) for iv in lab_iv)]
    fn = len(labels) - tp
    fp = len(fp_dets)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)

    by_class: dict[str, dict] = {}
    for info in per_label:
        c = by_class.setdefault(info["cls"], {"labels": 0, "detected": 0})
        c["labels"] += 1
        c["detected"] += int(info["detected"])
    for c in by_class.values():
        c["recall"] = _safe_div(c["detected"], c["labels"])

    delays = [i["delay"] for i in per_label if i["detected"]]
    ious = [i["iou"] for i in per_label if i["detected"]]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_labels": len(labels),
        "n_detections": len(dets),
        "precision": precision,
        "recall": recall,
        "f1": f_beta(precision, recall, 1.0),
        "f0_5": f_beta(precision, recall, 0.5),
        "false_alarms_per_10k": 1e4 * fp / n_evaluated if n_evaluated else None,
        "mean_delay": float(np.mean(delays)) if delays else None,
        "median_delay": float(np.median(delays)) if delays else None,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "by_class": by_class,
        "per_label": per_label,
        "false_positive_intervals": [list(d) for d in fp_dets],
    }


def point_metrics(pred_mask: np.ndarray, label_mask: np.ndarray) -> dict:
    p = np.asarray(pred_mask, bool)
    t = np.asarray(label_mask, bool)
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f_beta(precision, recall, 1.0),
        "f0_5": f_beta(precision, recall, 0.5),
        "tpr": recall,
        "fpr": _safe_div(fp, fp + tn),
        "n": int(t.size),
        "positives": int(t.sum()),
    }


def score_curves(scores: np.ndarray, label_mask: np.ndarray, max_points: int = 200) -> dict | None:
    """Threshold-free PR / ROC curves of a continuous score (down-sampled for display)."""
    s = np.asarray(scores, float)
    y = np.asarray(label_mask, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.sum() == 0 or y.sum() == y.size:
        return None
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)

    def thin(*arrs):
        n = len(arrs[0])
        if n <= max_points:
            return [a.tolist() for a in arrs]
        idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
        return [a[idx].tolist() for a in arrs]

    pr_r, pr_p = thin(rec, prec)
    roc_f, roc_t = thin(fpr, tpr)
    return {
        "average_precision": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "prevalence": float(y.mean()),
        "pr": {"recall": pr_r, "precision": pr_p},
        "roc": {"fpr": roc_f, "tpr": roc_t},
    }


def intervals_to_mask(intervals: Sequence[tuple[int, int]], n: int, offset: int = 0) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in intervals:
        a2, b2 = max(a - offset, 0), min(b - offset, n - 1)
        if b2 >= a2:
            m[a2 : b2 + 1] = True
    return m


def evaluate_channel(
    detections: Sequence[tuple[int, int]],
    labels: Sequence[dict],
    smoothed_scores: np.ndarray,
    n_total: int,
    eval_start: int,
) -> dict:
    """Full per-channel evaluation. ``smoothed_scores`` covers [eval_start, n_total)."""
    n_eval = n_total - eval_start
    lab_eval = clip_labels(labels, eval_start)
    unscoreable = sum(max(0, min(lab["end"], eval_start - 1) - lab["start"] + 1) for lab in labels)
    label_mask = intervals_to_mask([(lab["start"], lab["end"]) for lab in lab_eval], n_eval, eval_start)
    det_mask = intervals_to_mask(detections, n_eval, eval_start)
    return {
        "eval_start": eval_start,
        "n_evaluated": n_eval,
        "unscoreable_label_points": int(unscoreable),
        "event": event_metrics(detections, lab_eval, n_eval),
        "point": point_metrics(det_mask, label_mask),
        "curves": score_curves(smoothed_scores, label_mask),
    }


def aggregate(channel_metrics: Sequence[dict]) -> dict:
    """Pool counts across channels (micro-average) and average per-channel F1 (macro)."""
    ev = {"tp": 0, "fp": 0, "fn": 0, "n_labels": 0, "n_detections": 0}
    pt = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    by_class: dict[str, dict] = {}
    delays: list[float] = []
    ious: list[float] = []
    n_eval = 0
    macro_f1: list[float] = []
    aps: list[float] = []
    for m in channel_metrics:
        n_eval += m["n_evaluated"]
        for k in ev:
            ev[k] += m["event"][k]
        for k in pt:
            pt[k] += m["point"][k]
        for c, v in m["event"]["by_class"].items():
            agg = by_class.setdefault(c, {"labels": 0, "detected": 0})
            agg["labels"] += v["labels"]
            agg["detected"] += v["detected"]
        for lab in m["event"]["per_label"]:
            if lab["detected"]:
                delays.append(lab["delay"])
                ious.append(lab["iou"])
        if m["event"]["f1"] is not None:
            macro_f1.append(m["event"]["f1"])
        if m.get("curves"):
            aps.append(m["curves"]["average_precision"])
    for c in by_class.values():
        c["recall"] = _safe_div(c["detected"], c["labels"])
    e_prec = _safe_div(ev["tp"], ev["tp"] + ev["fp"])
    e_rec = _safe_div(ev["tp"], ev["tp"] + ev["fn"])
    p_prec = _safe_div(pt["tp"], pt["tp"] + pt["fp"])
    p_rec = _safe_div(pt["tp"], pt["tp"] + pt["fn"])
    return {
        "n_channels": len(channel_metrics),
        "n_evaluated": n_eval,
        "event": {
            **ev,
            "precision": e_prec,
            "recall": e_rec,
            "f1": f_beta(e_prec, e_rec, 1.0),
            "f0_5": f_beta(e_prec, e_rec, 0.5),
            "false_alarms_per_10k": 1e4 * ev["fp"] / n_eval if n_eval else None,
            "mean_delay": float(np.mean(delays)) if delays else None,
            "median_delay": float(np.median(delays)) if delays else None,
            "mean_iou": float(np.mean(ious)) if ious else None,
            "macro_f1": float(np.mean(macro_f1)) if macro_f1 else None,
            "by_class": by_class,
        },
        "point": {
            **pt,
            "precision": p_prec,
            "recall": p_rec,
            "f1": f_beta(p_prec, p_rec, 1.0),
            "f0_5": f_beta(p_prec, p_rec, 0.5),
            "tpr": p_rec,
            "fpr": _safe_div(pt["fp"], pt["fp"] + pt["tn"]),
        },
        "mean_average_precision": float(np.mean(aps)) if aps else None,
    }
