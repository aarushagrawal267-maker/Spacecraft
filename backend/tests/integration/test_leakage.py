"""Data-leakage guard: detections must not depend on the benchmark labels.

Labels are only allowed to influence *evaluation*. We run the full channel
pipeline twice — once with the real labels and once with fabricated labels —
and require identical detections (while the metrics, of course, differ).
"""

import pytest

from app.dataset import store
from app.dataset.labels import ChannelLabels, LabelInterval
from app.ml import methods, pipeline
from app.ml.scoring.thresholds import ThresholdConfig
from app.ml.training import FitContext

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize("method", ["ridge_ar", "persistence", "isolation_forest"])
def test_detections_independent_of_labels(method, tmp_path, monkeypatch):
    if not store.is_available():
        pytest.skip("dataset not installed")
    chan = "P-1"
    cfg = methods.resolve_config(method, None, None)
    real = pipeline.run_channel(method, chan, cfg, ThresholdConfig(), FitContext(seed=0), tmp_path / "a")

    original = store.channel_labels(chan)
    fake = ChannelLabels(chan, original.spacecraft, original.num_values,
                         [LabelInterval(300, 400, "point"), LabelInterval(7000, 7100, "contextual")], 1)
    monkeypatch.setattr(store, "channel_labels", lambda c: fake if c == chan else original)
    perturbed = pipeline.run_channel(method, chan, cfg, ThresholdConfig(), FitContext(seed=0), tmp_path / "b")

    assert [(d["start"], d["end"]) for d in real.detections] == [(d["start"], d["end"]) for d in perturbed.detections]
    assert real.metrics["event"]["n_labels"] != perturbed.metrics["event"]["n_labels"] or \
        real.metrics["point"] != perturbed.metrics["point"]


def test_training_code_never_reads_labels(monkeypatch, tmp_path):
    if not store.is_available():
        pytest.skip("dataset not installed")

    def forbidden(*a, **k):
        raise AssertionError("labels accessed during fit/score")

    monkeypatch.setattr(store, "channel_labels", forbidden)
    monkeypatch.setattr(store, "label_mask", forbidden)
    train, test = store.load_split("T-9", "train"), store.load_split("T-9", "test")
    for method in methods.SPECS:
        cfg = methods.resolve_config(method, "quick" if methods.SPECS[method].profiles else None,
                                     {"epochs": 1} if methods.SPECS[method].profiles else None)
        f = methods.fit(method, train, cfg, FitContext(seed=0))
        methods.score(f, test)
