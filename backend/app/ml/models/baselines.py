"""Baseline scorers. Each has a specific comparison purpose:

* persistence      – naive forecast y_hat_t = y_{t-1}; shows what a *learned* forecaster adds.
* ridge_ar         – linear autoregression on the last ``lags`` telemetry values plus the
                     current command inputs; shows what the LSTM's non-linearity adds.
* rolling_zscore   – classic statistical thresholding: |y_t - mean| / sd over a trailing window;
                     needs no model at all.
* isolation_forest – a standard non-temporal anomaly detector on lagged-value features,
                     fitted on the training split only.
* random           – uniform random scores. A control that exposes metric inflation.

All scorers are causal (use only data at or before t) and return arrays for the
index range [start, end) with NaN where not enough history exists.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _lagged(y: np.ndarray, lags: int) -> np.ndarray:
    """Row i holds y[i : i+lags] (to predict / describe time i+lags-1 or i+lags)."""
    return sliding_window_view(y, lags)


# --------------------------------------------------------------------------- persistence
def persistence_score(data: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    y = data[:, 0]
    n = end - start
    pred = np.full(n, np.nan, dtype=np.float32)
    lo = max(start, 1)
    pred[lo - start :] = y[lo - 1 : end - 1]
    return pred, np.abs(y[start:end] - pred)


# --------------------------------------------------------------------------- ridge AR
@dataclass
class RidgeState:
    model: Ridge
    lags: int


def _ridge_features(data: np.ndarray, targets: np.ndarray, lags: int) -> np.ndarray:
    y = data[:, 0]
    lag_idx = targets[:, None] - lags + np.arange(lags)[None, :]
    return np.hstack([y[lag_idx], data[targets - 1, 1:]])


def ridge_fit(train: np.ndarray, cfg: dict) -> RidgeState:
    lags = cfg["lags"]
    targets = np.arange(lags, len(train))
    X = _ridge_features(train, targets, lags)
    m = Ridge(alpha=cfg.get("alpha", 1.0))
    m.fit(X, train[targets, 0])
    return RidgeState(m, lags)


def ridge_score(state: RidgeState, data: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    n = end - start
    pred = np.full(n, np.nan, dtype=np.float32)
    first = max(start, state.lags)
    if first < end:
        t = np.arange(first, end)
        pred[first - start :] = state.model.predict(_ridge_features(data, t, state.lags))
    return pred, np.abs(data[start:end, 0] - pred)


# --------------------------------------------------------------------------- rolling z
def rolling_zscore_score(data: np.ndarray, start: int, end: int, window: int, sd_floor: float) -> np.ndarray:
    y = data[:, 0].astype(np.float64)
    n = end - start
    out = np.full(n, np.nan, dtype=np.float32)
    first = max(start, window)
    if first < end:
        t = np.arange(first, end)
        hist = y[t[:, None] - window + np.arange(window)[None, :]]
        mu = hist.mean(axis=1)
        sd = np.maximum(hist.std(axis=1), sd_floor)
        out[first - start :] = np.abs(y[t] - mu) / sd
    return out


# --------------------------------------------------------------------------- isolation forest
@dataclass
class IForestState:
    model: IsolationForest
    scaler: StandardScaler
    lags: int


def _if_features(data: np.ndarray, t: np.ndarray, lags: int) -> np.ndarray:
    y = data[:, 0]
    win = y[t[:, None] - lags + 1 + np.arange(lags)[None, :]]
    diffs = np.diff(win, axis=1)
    cmd = data[t, 1:].sum(axis=1, keepdims=True)
    return np.hstack([win, diffs.std(axis=1, keepdims=True), np.abs(diffs).max(axis=1, keepdims=True), cmd])


def iforest_fit(train: np.ndarray, cfg: dict, seed: int) -> IForestState:
    lags = cfg["lags"]
    t = np.arange(lags - 1, len(train))
    X = _if_features(train, t, lags)
    scaler = StandardScaler().fit(X)
    model = IsolationForest(
        n_estimators=cfg.get("n_estimators", 100), max_samples=cfg.get("max_samples", 256),
        random_state=seed, n_jobs=1,
    ).fit(scaler.transform(X))
    return IForestState(model, scaler, lags)


def iforest_score(state: IForestState, data: np.ndarray, start: int, end: int) -> np.ndarray:
    n = end - start
    out = np.full(n, np.nan, dtype=np.float32)
    first = max(start, state.lags - 1)
    if first < end:
        t = np.arange(first, end)
        X = state.scaler.transform(_if_features(data, t, state.lags))
        out[first - start :] = -state.model.score_samples(X)  # higher = more anomalous
    return out


# --------------------------------------------------------------------------- random control
def random_score(n_total: int, start: int, end: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n_total).astype(np.float32)[start:end]


# --------------------------------------------------------------------------- persistence of sklearn models
def save_pickle(obj, path: Path) -> Path:
    path = path.with_suffix(".pkl")
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle(path: Path):
    # Only files written by this application into artifacts/ are ever loaded here.
    with path.open("rb") as f:
        return pickle.load(f)
