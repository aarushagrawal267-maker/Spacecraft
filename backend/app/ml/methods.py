"""Registry of detection methods with a uniform fit / score / save / load interface.

Every method turns a telemetry array into a per-timestep *raw score* (absolute
prediction error, reconstruction error, or a statistical score). The pipeline
then smooths and thresholds every method's scores the same way, so all methods
are compared under one protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.models import baselines
from app.ml.training import FitContext, FitSummary

# All methods are evaluated on test indices [EVAL_START, N). 250 is the history
# length used by Hundman et al.; no benchmark label starts before index 290.
EVAL_START = 250
MAX_WINDOW = EVAL_START


@dataclass
class MethodSpec:
    key: str
    label: str
    family: str  # forecast | reconstruction | statistical | control
    purpose: str
    trainable: bool
    defaults: dict
    profiles: dict[str, dict] = field(default_factory=dict)
    produces_prediction: bool = False


FORECASTER_BASE = {
    "window": 250, "horizon": 10, "hidden": 80, "layers": 2, "dropout": 0.3,
    "epochs": 35, "batch_size": 64, "lr": 1e-3, "patience": 10, "min_delta": 3e-4, "val_fraction": 0.2,
}

SPECS: dict[str, MethodSpec] = {
    "lstm_forecaster": MethodSpec(
        key="lstm_forecaster",
        label="LSTM forecaster",
        family="forecast",
        purpose="Principal model: learns normal temporal behaviour and forecasts the next value; "
        "large forecast errors indicate unusual behaviour (Hundman et al., 2018).",
        trainable=True,
        produces_prediction=True,
        defaults={**FORECASTER_BASE, "window": 250, "hidden": 64, "epochs": 20, "patience": 5},
        profiles={
            "quick": {**FORECASTER_BASE, "window": 100, "hidden": 48, "epochs": 8, "patience": 3},
            "standard": {**FORECASTER_BASE, "window": 250, "hidden": 64, "epochs": 20, "patience": 5},
            "paper": dict(FORECASTER_BASE),
        },
    ),
    "lstm_autoencoder": MethodSpec(
        key="lstm_autoencoder",
        label="LSTM autoencoder",
        family="reconstruction",
        purpose="Experimental: reconstructs windows of telemetry; large reconstruction errors "
        "indicate patterns unlike the training data.",
        trainable=True,
        defaults={"window": 100, "hidden": 64, "latent": 16, "dropout": 0.1, "epochs": 20,
                  "batch_size": 64, "lr": 1e-3, "patience": 5, "min_delta": 1e-4,
                  "val_fraction": 0.2, "train_stride": 1},
        profiles={
            "quick": {"window": 50, "hidden": 32, "latent": 8, "dropout": 0.1, "epochs": 8, "batch_size": 64,
                      "lr": 1e-3, "patience": 3, "min_delta": 1e-4, "val_fraction": 0.2, "train_stride": 2},
            "standard": {"window": 100, "hidden": 64, "latent": 16, "dropout": 0.1, "epochs": 20,
                         "batch_size": 64, "lr": 1e-3, "patience": 5, "min_delta": 1e-4,
                         "val_fraction": 0.2, "train_stride": 1},
        },
    ),
    "ridge_ar": MethodSpec(
        key="ridge_ar", label="Ridge autoregression", family="forecast",
        purpose="Linear forecasting baseline (same scoring as the LSTM): isolates the value of non-linearity.",
        trainable=True, produces_prediction=True, defaults={"lags": 30, "alpha": 1.0},
    ),
    "persistence": MethodSpec(
        key="persistence", label="Persistence forecast", family="forecast",
        purpose="Naive forecast (next value = current value): the floor any learned forecaster must beat.",
        trainable=False, produces_prediction=True, defaults={},
    ),
    "rolling_zscore": MethodSpec(
        key="rolling_zscore", label="Rolling z-score", family="statistical",
        purpose="Classic statistical thresholding on the raw telemetry, no learned model.",
        trainable=False, defaults={"window": 100, "sd_floor": 0.01},
    ),
    "isolation_forest": MethodSpec(
        key="isolation_forest", label="Isolation Forest", family="statistical",
        purpose="Standard non-temporal detector on lagged-value features, fitted on the training split.",
        trainable=True, defaults={"lags": 16, "n_estimators": 100, "max_samples": 256},
    ),
    "random": MethodSpec(
        key="random", label="Random scores (control)", family="control",
        purpose="Control: uniform random scores. Any protocol under which this looks good is inflated.",
        trainable=False, defaults={},
    ),
}


def resolve_config(method: str, profile: str | None, overrides: dict | None) -> dict:
    spec = SPECS[method]
    base = dict(spec.profiles.get(profile, spec.defaults)) if profile else dict(spec.defaults)
    base.update({k: v for k, v in (overrides or {}).items() if v is not None})
    w = base.get("window")
    if w is not None and not (1 <= int(w) <= MAX_WINDOW):
        raise ValueError(f"window must be between 1 and {MAX_WINDOW}")
    return base


@dataclass
class Fitted:
    method: str
    cfg: dict
    state: Any
    summary: FitSummary | None
    seed: int


def fit(method: str, train: np.ndarray, cfg: dict, ctx: FitContext) -> Fitted:
    if method == "lstm_forecaster":
        from app.ml.models import lstm_forecaster as m

        st = m.fit(train, cfg, ctx)
        return Fitted(method, cfg, st, st.summary, ctx.seed)
    if method == "lstm_autoencoder":
        from app.ml.models import lstm_autoencoder as m

        st = m.fit(train, cfg, ctx)
        return Fitted(method, cfg, st, st.summary, ctx.seed)
    if method == "ridge_ar":
        return Fitted(method, cfg, baselines.ridge_fit(train, cfg), None, ctx.seed)
    if method == "isolation_forest":
        return Fitted(method, cfg, baselines.iforest_fit(train, cfg, ctx.seed), None, ctx.seed)
    if method in ("persistence", "rolling_zscore", "random"):
        return Fitted(method, cfg, None, None, ctx.seed)
    raise KeyError(method)


def score(f: Fitted, data: np.ndarray, start: int = 0, end: int | None = None) -> tuple[np.ndarray | None, np.ndarray]:
    """(prediction or None, raw score) for indices [start, end)."""
    end = len(data) if end is None else end
    m = f.method
    if m == "lstm_forecaster":
        from app.ml.models import lstm_forecaster as mod

        return mod.score(f.state, data, start, end)
    if m == "lstm_autoencoder":
        from app.ml.models import lstm_autoencoder as mod

        return mod.score(f.state, data, start, end)
    if m == "ridge_ar":
        return baselines.ridge_score(f.state, data, start, end)
    if m == "persistence":
        return baselines.persistence_score(data, start, end)
    if m == "rolling_zscore":
        return None, baselines.rolling_zscore_score(data, start, end, f.cfg["window"], f.cfg["sd_floor"])
    if m == "isolation_forest":
        return None, baselines.iforest_score(f.state, data, start, end)
    if m == "random":
        return None, baselines.random_score(len(data), start, end, f.seed)
    raise KeyError(m)


def save(f: Fitted, path_stem: Path) -> Path | None:
    if f.method == "lstm_forecaster":
        from app.ml.models import lstm_forecaster as mod

        return mod.save(f.state, path_stem)
    if f.method == "lstm_autoencoder":
        from app.ml.models import lstm_autoencoder as mod

        return mod.save(f.state, path_stem)
    if f.method in ("ridge_ar", "isolation_forest"):
        return baselines.save_pickle(f.state, path_stem)
    return None  # stateless methods need no checkpoint


def load(method: str, cfg: dict, path: Path | None, device: str, seed: int) -> Fitted:
    if method == "lstm_forecaster":
        from app.ml.models import lstm_forecaster as mod

        return Fitted(method, cfg, mod.load(path, device), None, seed)
    if method == "lstm_autoencoder":
        from app.ml.models import lstm_autoencoder as mod

        return Fitted(method, cfg, mod.load(path, device), None, seed)
    if method in ("ridge_ar", "isolation_forest"):
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"No saved model for {method}")
        return Fitted(method, cfg, baselines.load_pickle(Path(path)), None, seed)
    return Fitted(method, cfg, None, None, seed)


ScoreFn = Callable[[np.ndarray, int, int], tuple[np.ndarray | None, np.ndarray]]
