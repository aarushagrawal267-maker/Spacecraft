"""Model 1 — LSTM telemetry forecaster (principal model).

Architecture follows the Telemanom defaults: two stacked LSTM layers with
dropout and a linear head that predicts the next ``horizon`` telemetry values
from a window of ``window`` past timesteps of all features (telemetry + one-hot
command inputs). The first predicted step is used for anomaly scoring, i.e. the
prediction for time t comes from the window [t-window, t).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.ml.training import FitContext, FitSummary, chronological_split, fit_torch, seed_everything


class LSTMForecasterNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float, horizon: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


@dataclass
class ForecasterState:
    net: LSTMForecasterNet
    cfg: dict
    n_features: int
    device: str
    summary: FitSummary | None = None


def _window_index(start_positions: np.ndarray, length: int) -> np.ndarray:
    return start_positions[:, None] + np.arange(length)[None, :]


def fit(train: np.ndarray, cfg: dict, ctx: FitContext) -> ForecasterState:
    seed_everything(ctx.seed)
    w, h = cfg["window"], cfg["horizon"]
    n_samples = len(train) - w - h + 1
    if n_samples < 8:
        raise ValueError(f"Training split too short ({len(train)}) for window {w} + horizon {h}")
    net = LSTMForecasterNet(train.shape[1], cfg["hidden"], cfg["layers"], cfg["dropout"], h)
    data = torch.from_numpy(np.array(train, dtype=np.float32))

    def make_batch(idx: np.ndarray):
        idx = np.asarray(idx)
        x = data[torch.from_numpy(_window_index(idx, w))]
        y = data[torch.from_numpy(_window_index(idx + w, h)), 0]
        return x, y

    tr_idx, va_idx = chronological_split(n_samples, cfg["val_fraction"], gap=w + h - 1)
    summary = fit_torch(
        net, make_batch, tr_idx, va_idx,
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        patience=cfg["patience"], min_delta=cfg["min_delta"], ctx=ctx,
    )
    return ForecasterState(net=net, cfg=cfg, n_features=train.shape[1], device=ctx.device, summary=summary)


@torch.no_grad()
def score(state: ForecasterState, data: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (prediction, abs error) for indices [start, end); NaN where t < window."""
    w = state.cfg["window"]
    n = end - start
    pred = np.full(n, np.nan, dtype=np.float32)
    first = max(start, w)
    if first < end:
        dev = torch.device(state.device)
        state.net.to(dev).eval()
        t_all = torch.from_numpy(np.array(data, dtype=np.float32))
        targets = np.arange(first, end)
        out = []
        for s in range(0, len(targets), 1024):
            tt = targets[s : s + 1024]
            x = t_all[torch.from_numpy(_window_index(tt - w, w))].to(dev)
            out.append(state.net(x)[:, 0].float().cpu().numpy())
        pred[first - start :] = np.concatenate(out)
    err = np.abs(data[start:end, 0] - pred)
    return pred, err


def save(state: ForecasterState, path: Path) -> Path:
    path = path.with_suffix(".pt")
    torch.save(
        {"kind": "lstm_forecaster", "state_dict": state.net.state_dict(), "cfg": state.cfg,
         "n_features": state.n_features},
        path,
    )
    return path


def load(path: Path, device: str) -> ForecasterState:
    blob = torch.load(path, map_location=device, weights_only=True)
    cfg = blob["cfg"]
    net = LSTMForecasterNet(blob["n_features"], cfg["hidden"], cfg["layers"], cfg["dropout"], cfg["horizon"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return ForecasterState(net=net, cfg=cfg, n_features=blob["n_features"], device=device)
