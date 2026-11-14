import numpy as np
import pytest

from app.ml.evaluation.metrics import (
    aggregate,
    clip_labels,
    evaluate_channel,
    event_metrics,
    f_beta,
    intervals_to_mask,
    point_metrics,
)


def test_f_beta():
    assert f_beta(0.5, 0.5, 1.0) == pytest.approx(0.5)
    assert f_beta(1.0, 0.5, 0.5) == pytest.approx(1.25 * 0.5 / (0.25 + 0.5))
    assert f_beta(0.0, 0.0, 1.0) == 0.0
    assert f_beta(None, 0.5, 1.0) is None


def test_event_metrics_hand_computed():
    labels = [
        {"start": 100, "end": 200, "cls": "point"},
        {"start": 500, "end": 600, "cls": "contextual"},
        {"start": 900, "end": 950, "cls": "point"},
    ]
    dets = [(150, 260), (190, 210), (700, 720), (940, 1000)]
    m = event_metrics(dets, labels, n_evaluated=10_000)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    # Protocol of Hundman et al.: 3 of 4 detections overlap a label, but the protocol counts
    # FP = detections overlapping no label (1) and TP = detected labels (2): 2 / (2 + 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_alarms_per_10k"] == pytest.approx(1.0)
    first = m["per_label"][0]
    assert first["delay"] == 50  # earliest overlapping detection starts at 150
    assert first["iou"] == pytest.approx(51 / 161)  # overlap 150..200, union 100..260
    assert m["per_label"][2]["delay"] == 40
    assert m["by_class"]["point"] == {"labels": 2, "detected": 2, "recall": 1.0}
    assert m["by_class"]["contextual"]["recall"] == 0.0


def test_event_metrics_no_detections():
    m = event_metrics([], [{"start": 1, "end": 2, "cls": "point"}], 100)
    assert m["precision"] is None and m["recall"] == 0.0 and m["fn"] == 1


def test_point_metrics():
    p = np.array([1, 1, 0, 0, 1, 0], bool)
    t = np.array([1, 0, 1, 0, 1, 0], bool)
    m = point_metrics(p, t)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["fpr"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_clip_and_mask():
    labs = clip_labels([{"start": 10, "end": 20}, {"start": 240, "end": 300}, {"start": 400, "end": 410}], 250)
    assert labs == [{"start": 250, "end": 300}, {"start": 400, "end": 410}]
    m = intervals_to_mask([(250, 252), (299, 305)], 50, offset=250)
    assert m[:3].all() and not m[3] and m[49]


def test_evaluate_channel_counts_unscoreable_label_points():
    labels = [{"start": 200, "end": 260, "cls": "point"}]
    s = np.zeros(750)
    s[10:20] = 5
    r = evaluate_channel([(255, 270)], labels, s, n_total=1000, eval_start=250)
    assert r["unscoreable_label_points"] == 50
    assert r["event"]["tp"] == 1
    assert r["point"]["n"] == 750


def test_aggregate_micro_and_macro():
    a = {"n_evaluated": 100, "event": {"tp": 1, "fp": 1, "fn": 0, "n_labels": 1, "n_detections": 2, "f1": 2 / 3,
         "by_class": {"point": {"labels": 1, "detected": 1}}, "per_label": [{"detected": True, "delay": 4, "iou": 0.5}]},
         "point": {"tp": 5, "fp": 5, "fn": 0, "tn": 90}, "curves": {"average_precision": 0.4}}
    b = {"n_evaluated": 100, "event": {"tp": 0, "fp": 0, "fn": 2, "n_labels": 2, "n_detections": 0, "f1": None,
         "by_class": {"point": {"labels": 2, "detected": 0}}, "per_label": [{"detected": False}, {"detected": False}]},
         "point": {"tp": 0, "fp": 0, "fn": 10, "tn": 90}, "curves": None}
    g = aggregate([a, b])
    assert g["event"]["precision"] == pytest.approx(0.5)
    assert g["event"]["recall"] == pytest.approx(1 / 3)
    assert g["event"]["macro_f1"] == pytest.approx(2 / 3)
    assert g["event"]["by_class"]["point"]["recall"] == pytest.approx(1 / 3)
    assert g["point"]["fpr"] == pytest.approx(5 / 185)
    assert g["mean_average_precision"] == pytest.approx(0.4)
