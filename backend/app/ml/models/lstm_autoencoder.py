"""Model 2 — LSTM autoencoder (experimental).

Encodes a window of ``window`` timesteps (all features) into a latent vector and
reconstructs the telemetry column of that window. The anomaly score at time t is
the mean absolute reconstruction error of the window ending at t, which keeps
the score causal (usable in the historical replay).

Caveat: the training split is not guaranteed to be anomaly-free, so the model
may partially learn to reconstruct unusual patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.ml.training import FitContext, FitSummary, chronological_split, fit_torch, seed_everything


class LSTMAutoencoderNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, latent: int, window: int, dropout: float) -> None:
        super().__init__()
        self.window = window
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.decoder = nn.LSTM(latent, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        z = torch.tanh(self.to_latent(h[-1]))
        rep = z.unsqueeze(1).expand(-1, x.shape[1], -1)
        dec, _ = self.decoder(rep)
        return self.out(self.dropout(dec)).squeeze(-1)


@dataclass
class AutoencoderState:
    net: LSTMAutoencoderNet
    cfg: dict
    n_features: int
    device: str
    summary: FitSummary | None = None


def _idx(starts: np.ndarray, length: int) -> np.ndarray:
    return starts[:, None] + np.arange(length)[None, :]


def fit(train: np.ndarray, cfg: dict, ctx: FitContext) -> AutoencoderState:
    seed_everything(ctx.seed)
    w = cfg["window"]
    stride = max(1, int(cfg.get("train_stride", 1)))
    starts = np.arange(0, len(train) - w + 1, stride)
    if len(starts) < 8:
        raise ValueError(f"Training split too short ({len(train)}) for window {w}")
    net = LSTMAutoencoderNet(train.shape[1], cfg["hidden"], cfg["latent"], w, cfg["dropout"])
    data = torch.from_numpy(np.array(train, dtype=np.float32))

    def make_batch(sample_idx: np.ndarray):
        st = starts[np.asarray(sample_idx)]
        x = data[torch.from_numpy(_idx(st, w))]
        return x, x[:, :, 0]

    tr, va = chronological_split(len(starts), cfg["val_fraction"], gap=int(np.ceil(w / stride)))
    summary = fit_torch(
        net, make_batch, tr, va,
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        patience=cfg["patience"], min_delta=cfg["min_delta"], ctx=ctx,
    )
    return AutoencoderState(net=net, cfg=cfg, n_features=train.shape[1], device=ctx.device, summary=summary)


@torch.no_grad()
def score(state: AutoencoderState, data: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (reconstruction of x_t from the window ending at t, window error) for [start, end)."""
    w = state.cfg["window"]
    n = end - start
    recon_last = np.full(n, np.nan, dtype=np.float32)
    err = np.full(n, np.nan, dtype=np.float32)
    first = max(start, w - 1)
    if first < end:
        dev = torch.device(state.device)
        state.net.to(dev).eval()
        t_all = torch.from_numpy(np.array(data, dtype=np.float32))
        ends = np.arange(first, end)
        r_out, e_out = [], []
        for s in range(0, len(ends), 512):
            ee = ends[s : s + 512]
            x = t_all[torch.from_numpy(_idx(ee - w + 1, w))].to(dev)
            rec = state.net(x)
            e_out.append((rec - x[:, :, 0]).abs().mean(dim=1).cpu().numpy())
            r_out.append(rec[:, -1].cpu().numpy())
        recon_last[first - start :] = np.concatenate(r_out)
        err[first - start :] = np.concatenate(e_out)
    return recon_last, err


def save(state: AutoencoderState, path: Path) -> Path:
    path = path.with_suffix(".pt")
    torch.save(
        {"kind": "lstm_autoencoder", "state_dict": state.net.state_dict(), "cfg": state.cfg,
         "n_features": state.n_features},
        path,
    )
    return path


def load(path: Path, device: str) -> AutoencoderState:
    blob = torch.load(path, map_location=device, weights_only=True)
    cfg = blob["cfg"]
    net = LSTMAutoencoderNet(blob["n_features"], cfg["hidden"], cfg["latent"], cfg["window"], cfg["dropout"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return AutoencoderState(net=net, cfg=cfg, n_features=blob["n_features"], device=device)
