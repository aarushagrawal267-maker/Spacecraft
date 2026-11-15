"""Training/evaluation worker that runs in a separate process.

It never touches the database. Progress and results are sent to the parent
process over a multiprocessing queue; the parent is the single DB writer.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import asdict
from pathlib import Path


def run_worker(payload: dict, queue, cancel_event) -> None:
    exp_id = payload["experiment_id"]
    worker_idx = payload["worker_idx"]
    try:
        threads = payload.get("torch_threads")
        if threads:
            os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        import torch

        if threads:
            torch.set_num_threads(int(threads))

        from app.ml import pipeline
        from app.ml.device import resolve_device
        from app.ml.scoring.thresholds import ThresholdConfig
        from app.ml.training import FitContext, TrainingCancelled

        device = resolve_device(payload.get("device", "auto"))
        thr = ThresholdConfig.from_dict(payload["threshold_config"])
        out_dir = Path(payload["out_dir"])
        queue.put({"type": "worker_started", "experiment_id": exp_id, "worker": worker_idx, "device": device})

        for chan in payload["channels"]:
            if cancel_event.is_set():
                break
            queue.put({"type": "run_started", "experiment_id": exp_id, "chan_id": chan})

            def on_epoch(epoch, tl, vl, secs, _chan=chan):
                queue.put({"type": "epoch", "experiment_id": exp_id, "chan_id": _chan, "epoch": epoch,
                           "train_loss": tl, "val_loss": vl, "seconds": secs})

            ctx = FitContext(device=device, seed=payload["seed"], on_epoch=on_epoch,
                             should_stop=cancel_event.is_set)
            try:
                outcome = pipeline.run_channel(payload["method"], chan, payload["config"], thr, ctx, out_dir)
                queue.put({"type": "run_done", "experiment_id": exp_id, "chan_id": chan, "outcome": asdict(outcome)})
            except TrainingCancelled:
                queue.put({"type": "run_failed", "experiment_id": exp_id, "chan_id": chan, "error": "cancelled",
                           "cancelled": True})
                break
            except Exception as exc:  # noqa: BLE001 - report every per-channel failure
                queue.put({"type": "run_failed", "experiment_id": exp_id, "chan_id": chan,
                           "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-4000:]})
    except Exception as exc:  # noqa: BLE001
        queue.put({"type": "worker_error", "experiment_id": exp_id, "worker": worker_idx,
                   "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-4000:]})
    finally:
        queue.put({"type": "worker_done", "experiment_id": exp_id, "worker": worker_idx})
