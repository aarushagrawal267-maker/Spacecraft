"""Shared PyTorch training loop with chronological validation and early stopping."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


class TrainingCancelled(RuntimeError):
    pass


@dataclass
class FitContext:
    device: str = "cpu"
    seed: int = 0
    on_epoch: Callable[[int, float, float | None, float], None] | None = None
    should_stop: Callable[[], bool] | None = None


@dataclass
class FitSummary:
    epochs_run: int = 0
    best_epoch: int | None = None
    best_val_loss: float | None = None
    history: list[dict] = field(default_factory=list)
    n_train_samples: int = 0
    n_val_samples: int = 0


def chronological_split(n: int, val_fraction: float, gap: int, min_train: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices 0..n-1 in time order: train first, validation last.

    ``gap`` samples are dropped before the validation block so that training
    targets do not overlap validation inputs; it is reduced when the series is
    too short to keep ``min_train`` training samples.
    """
    if n <= 1 or val_fraction <= 0:
        return np.arange(n), np.array([], dtype=int)
    n_val = max(1, int(round(n * val_fraction)))
    k = n - n_val
    gap = int(max(0, min(gap, k - min_train)))
    return np.arange(0, k - gap), np.arange(k, n)


def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_torch(
    model,
    make_batch: Callable[[np.ndarray], tuple],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    min_delta: float,
    ctx: FitContext,
) -> FitSummary:
    """Train ``model`` (MSE) on samples built by ``make_batch(indices) -> (x, y)`` tensors."""
    import torch

    device = torch.device(ctx.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(ctx.seed)
    summary = FitSummary(n_train_samples=len(train_idx), n_val_samples=len(val_idx))
    best_state = None
    best = np.inf
    bad = 0

    def run_eval(idx: np.ndarray) -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                xb, yb = make_batch(idx[s : s + 1024])
                xb, yb = xb.to(device), yb.to(device)
                tot += float(loss_fn(model(xb), yb)) * len(xb)
                n += len(xb)
        return tot / max(n, 1)

    for epoch in range(1, epochs + 1):
        if ctx.should_stop and ctx.should_stop():
            raise TrainingCancelled()
        t0 = time.perf_counter()
        model.train()
        perm = rng.permutation(train_idx)
        tot, n = 0.0, 0
        for s in range(0, len(perm), batch_size):
            xb, yb = make_batch(perm[s : s + batch_size])
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
            if ctx.should_stop and ctx.should_stop():
                raise TrainingCancelled()
        train_loss = tot / max(n, 1)
        val_loss = run_eval(val_idx) if len(val_idx) else None
        secs = time.perf_counter() - t0
        summary.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": secs})
        summary.epochs_run = epoch
        if ctx.on_epoch:
            ctx.on_epoch(epoch, train_loss, val_loss, secs)

        monitored = val_loss if val_loss is not None else train_loss
        if monitored < best - min_delta:
            best = monitored
            bad = 0
            best_state = copy.deepcopy(model.state_dict())
            summary.best_epoch = epoch
            summary.best_val_loss = val_loss
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # restore best validation weights
    model.eval()
    return summary
