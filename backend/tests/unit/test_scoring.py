"""Unit tests for smoothing and thresholding.

Synthetic arrays are used here *only* to test algorithm correctness; the
application itself never uses synthetic telemetry.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml.scoring.smoothing import EWMA, ewma
from app.ml.scoring.thresholds import DynamicThreshold, StaticThreshold, ThresholdConfig, runs


def test_ewma_matches_pandas_adjust_true():
    x = np.random.default_rng(0).random(500)
    expected = pd.Series(x).ewm(span=105, adjust=True).mean().to_numpy()
    np.testing.assert_allclose(ewma(x, 105), expected, rtol=1e-10, atol=1e-12)


def test_ewma_incremental_equals_batch():
    x = np.random.default_rng(1).random(777)
    full = ewma(x, 50)
    e = EWMA(50)
    parts = [e.update(x[a:b]) for a, b in [(0, 3), (3, 100), (100, 101), (101, 777)]]
    np.testing.assert_allclose(np.concatenate(parts), full, rtol=1e-12)


def test_runs():
    m = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
    assert runs(m) == [(1, 2), (4, 4), (7, 9)]
    assert runs(np.zeros(0, bool)) == []
    assert runs(np.ones(3, bool)) == [(0, 2)]


def _noise_with_spike(n=3000, spike_at=2000, width=40, height=1.5, seed=0):
    rng = np.random.default_rng(seed)
    e = np.abs(rng.normal(0.05, 0.02, n))
    e[spike_at : spike_at + width] += height
    y = rng.normal(0, 0.5, n)
    return ewma(e, 105), y


def test_dynamic_threshold_detects_spike():
    e_s, y = _noise_with_spike()
    dt = DynamicThreshold(len(e_s), ThresholdConfig())
    dt.push(e_s, y)
    dt.finalize()
    ivs = dt.intervals()
    assert ivs, "spike should be detected"
    assert any(iv.start <= 2000 <= iv.end + 100 for iv in ivs)
    assert all(iv.end > iv.start for iv in ivs)
    # every point has been given a verdict
    assert dt.processed_upto == len(e_s)


def test_dynamic_threshold_every_point_in_exactly_one_zone():
    for n in (2100, 2169, 2171, 3005, 5000, 900):
        dt = DynamicThreshold(n, ThresholdConfig())
        dt.push(np.random.default_rng(n).random(n), np.zeros(n))
        dt.finalize()
        cover = np.zeros(n, int)
        for w in dt.windows:
            cover[w.zone[0] : w.zone[1]] += 1
        assert (cover == 1).all(), f"n={n}: zones must tile the series exactly"


def test_streaming_equals_offline():
    e_s, y = _noise_with_spike(n=4321, spike_at=3100)
    cfg = ThresholdConfig()
    off = DynamicThreshold(len(e_s), cfg)
    off.push(e_s, y)
    off.finalize()
    st = DynamicThreshold(len(e_s), cfg)
    rng = np.random.default_rng(3)
    i = 0
    while i < len(e_s):
        k = int(rng.integers(1, 200))
        st.push(e_s[i : i + k], y[i : i + k])
        # verdicts already given never change later
        upto = st.processed_upto
        partial_flags = st.flags[:upto].copy()
        i += k
    st.finalize()
    np.testing.assert_array_equal(st.flags, off.flags)
    np.testing.assert_array_equal(partial_flags, off.flags[: len(partial_flags)])
    assert [(a.start, a.end) for a in st.intervals()] == [(a.start, a.end) for a in off.intervals()]


def test_inverse_pass_detects_unusually_low_errors():
    rng = np.random.default_rng(5)
    n = 3000
    e = 0.6 + np.abs(rng.normal(0, 0.03, n))
    e[2200:2260] = 0.0  # errors collapse
    e_s = ewma(e, 105)
    y = rng.normal(0, 1, n)
    dt = DynamicThreshold(n, ThresholdConfig(inverse=True))
    dt.push(e_s, y)
    dt.finalize()
    lows = [iv for iv in dt.intervals() if iv.direction in ("low", "both")]
    assert lows and any(iv.start <= 2260 and iv.end >= 2200 for iv in lows)
    dt2 = DynamicThreshold(n, ThresholdConfig(inverse=False))
    dt2.push(e_s, y)
    dt2.finalize()
    assert not any(iv.direction != "high" for iv in dt2.intervals())


def test_scale_guard_suppresses_tiny_errors():
    rng = np.random.default_rng(2)
    n = 3000
    e = np.abs(rng.normal(0.001, 0.0005, n))
    e[2000:2050] = 0.02  # a "spike" that is still below the 0.05 absolute floor
    dt = DynamicThreshold(n, ThresholdConfig())
    dt.push(ewma(e, 105), rng.normal(0, 1, n))
    dt.finalize()
    assert dt.intervals() == []


def test_pruning_removes_sequences_close_to_normal_maximum():
    dt = DynamicThreshold(100, ThresholdConfig(p=0.13))
    es = np.zeros(100)
    es[10:20] = 1.0  # clearly separated peak
    es[50:60] = 0.55  # peak only ~10% above the normal maximum (0.5)
    es[80] = 0.5
    flagged = np.zeros(100, bool)
    flagged[10:20] = True
    flagged[50:60] = True
    kept = dt._prune(flagged, es, non_anom_max=0.5)
    assert kept[10:20].all()
    assert not kept[50:60].any()


def test_short_sequence_window_shrinks():
    dt = DynamicThreshold(846, ThresholdConfig())  # e.g. T-9: 1096 - 250
    assert dt.ws < 30 and dt.n_windows == 0
    dt = DynamicThreshold(50, ThresholdConfig())  # shorter than one batch
    dt.push(np.zeros(50), np.zeros(50))
    dt.finalize()
    assert dt.processed_upto == 50


def test_static_threshold_uses_training_statistics_only():
    train = np.full(1000, 0.1)
    train[::2] = 0.2
    st = StaticThreshold.from_training_errors(train, 10, k=3.0)
    assert st.epsilon == pytest.approx(0.15 + 3 * 0.05)
    st.push(np.array([0, 0, 0.5, 0.5, 0, 0.5, 0, 0, 0, 0]))
    assert [(i.start, i.end) for i in st.intervals()] == [(2, 3)]  # single points dropped


@pytest.mark.parametrize("bad", [
    {"mode": "bogus"}, {"p": None}, {"p": "abc"}, {"p": 1.5}, {"batch_size": 0}, {"batch_size": 2.5},
    {"error_buffer": -1}, {"static_k": float("nan")}, {"z_min": 5, "z_max": 4}, {"inverse": "yes"},
])
def test_threshold_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        ThresholdConfig.from_dict(bad)


def test_threshold_config_accepts_defaults_and_coerces_numbers():
    assert ThresholdConfig.from_dict(None) == ThresholdConfig()
    cfg = ThresholdConfig.from_dict({"mode": "static", "static_k": 2, "error_buffer": 50.0, "unknown": 1})
    assert cfg.static_k == 2.0 and isinstance(cfg.error_buffer, int) and cfg.error_buffer == 50
