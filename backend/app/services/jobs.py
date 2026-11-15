"""Experiment job manager.

* Experiments are queued in the database (status ``queued``).
* A scheduler thread starts at most ``max_concurrent_jobs`` experiments; each
  runs in 1..N spawned worker processes (Windows-safe ``spawn`` context).
* Workers send progress over one multiprocessing queue; a consumer thread in
  this process is the only writer to SQLite.
* Cancelling sets an event the training loop checks every batch; processes
  that do not exit within a grace period are terminated.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue as queue_mod
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Channel, ChannelRun, Detection, EpochLog, Experiment
from app.db.session import session_scope
from app.ml.device import system_info
from app.services.evaluation_service import experiment_metrics

log = logging.getLogger("heliograph.jobs")


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=get_settings().project_root,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class ActiveJob:
    experiment_id: int
    processes: list
    cancel_event: object
    workers_left: int
    run_ids: dict[str, int] = field(default_factory=dict)
    cancel_requested_at: float | None = None
    total: int = 0
    finished: int = 0


class JobManager:
    def __init__(self) -> None:
        self.mpctx = mp.get_context("spawn")
        self.queue = None  # created in start(); spawned workers import this module too
        self.active: dict[int, ActiveJob] = {}
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.started = False

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        # Recovery runs on every application start; experiments active in this process are skipped.
        self._recover_interrupted()
        if self.started:
            return
        if self.queue is None:
            self.queue = self.mpctx.Queue()
        self._stop.clear()
        self._threads = []
        for target in (self._scheduler_loop, self._consumer_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)
        self.started = True

    def shutdown(self) -> None:
        self._stop.set()
        with self.lock:
            for job in self.active.values():
                job.cancel_event.set()
                for p in job.processes:
                    p.join(timeout=3)
                    if p.is_alive():
                        p.terminate()
        self.started = False

    def _recover_interrupted(self) -> None:
        with session_scope() as db:
            with self.lock:
                active = set(self.active)
            for exp in db.scalars(select(Experiment).where(Experiment.status == "running")):
                if exp.id in active:
                    continue
                exp.status = "failed"
                exp.message = "Interrupted: the backend stopped while this experiment was running."
                exp.finished_at = datetime.now(timezone.utc)
                for r in exp.runs:
                    if r.status in ("pending", "running"):
                        r.status = "failed"
                        r.error = "interrupted"

    # ---------------------------------------------------------------- control
    def cancel(self, exp_id: int) -> str:
        with session_scope() as db:
            exp = db.get(Experiment, exp_id)
            if exp is None:
                return "not_found"
            if exp.status == "queued":
                exp.status = "cancelled"
                exp.finished_at = datetime.now(timezone.utc)
                return "cancelled"
        with self.lock:
            job = self.active.get(exp_id)
            if job is None:
                return "not_running"
            job.cancel_event.set()
            job.cancel_requested_at = time.time()
        return "cancelling"

    def is_active(self, exp_id: int) -> bool:
        with self.lock:
            return exp_id in self.active

    # ---------------------------------------------------------------- scheduler
    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._maybe_launch()
                self._enforce_cancellations()
            except Exception:  # noqa: BLE001
                log.exception("scheduler error")
            time.sleep(0.5)

    def _enforce_cancellations(self) -> None:
        with self.lock:
            for job in list(self.active.values()):
                if job.cancel_requested_at and time.time() - job.cancel_requested_at > 20:
                    for p in job.processes:
                        if p.is_alive():
                            p.terminate()

    def _maybe_launch(self) -> None:
        settings = get_settings()
        with self.lock:
            if len(self.active) >= settings.max_concurrent_jobs:
                return
        with session_scope() as db:
            exp = db.scalars(
                select(Experiment).where(Experiment.status == "queued").order_by(Experiment.id).limit(1)
            ).first()
            if exp is None:
                return
            env = dict(exp.environment or {})
            workers = max(1, min(int(env.get("workers", 1)), len(exp.channels), os.cpu_count() or 1))
            threads = env.get("torch_threads") or settings.torch_threads
            if threads is None and workers > 1:
                threads = max(1, (os.cpu_count() or 1) // workers)
            info = system_info()
            env.update({"workers": workers, "torch_threads": threads, "system": info})
            exp.environment = env
            exp.status = "running"
            exp.started_at = datetime.now(timezone.utc)
            exp.git_commit = exp.git_commit or git_commit()
            exp.progress = 0.0
            out_dir = settings.artifacts_dir / "experiments" / f"exp_{exp.id:05d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            test_len = {c.chan_id: c.test_len for c in db.scalars(select(Channel))}
            run_ids: dict[str, int] = {}
            for chan in exp.channels:
                run = ChannelRun(experiment_id=exp.id, chan_id=chan, status="pending")
                db.add(run)
                db.flush()
                run_ids[chan] = run.id
            # Balance workers: longest channels first, round-robin.
            ordered = sorted(exp.channels, key=lambda c: -test_len.get(c, 0))
            slices = [ordered[i::workers] for i in range(workers)]
            payload_base = {
                "experiment_id": exp.id, "method": exp.method, "config": exp.config,
                "threshold_config": exp.threshold_config, "seed": exp.seed,
                "device": settings.device, "torch_threads": threads, "out_dir": str(out_dir),
            }
            exp_id = exp.id

        cancel_event = self.mpctx.Event()
        procs = []
        for i, chans in enumerate(slices):
            p = self.mpctx.Process(
                target=_worker_entry, args=({**payload_base, "worker_idx": i, "channels": chans}, self.queue, cancel_event),
                daemon=True,
            )
            p.start()
            procs.append(p)
        with self.lock:
            self.active[exp_id] = ActiveJob(exp_id, procs, cancel_event, len(procs), run_ids, total=len(run_ids))

    # ---------------------------------------------------------------- consumer
    def _consumer_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self.queue.get(timeout=0.5)
            except queue_mod.Empty:
                self._reap_dead_workers()
                continue
            except (EOFError, OSError):
                time.sleep(0.5)
                continue
            try:
                self._handle(msg)
            except Exception:  # noqa: BLE001
                log.exception("failed to handle worker message %s", msg.get("type"))

    def _reap_dead_workers(self) -> None:
        """Finalise jobs whose processes died without sending worker_done (e.g. killed)."""
        with self.lock:
            jobs = list(self.active.values())
        for job in jobs:
            if all(not p.is_alive() for p in job.processes) and job.workers_left > 0:
                time.sleep(1.0)  # allow in-flight queue messages to drain
                if self.queue.empty():
                    job.workers_left = 0
                    self._finalize(job.experiment_id)

    def _handle(self, msg: dict) -> None:
        exp_id = msg["experiment_id"]
        with self.lock:
            job = self.active.get(exp_id)
        if job is None:
            return
        t = msg["type"]
        if t == "worker_done":
            job.workers_left -= 1
            if job.workers_left <= 0:
                self._finalize(exp_id)
            return
        with session_scope() as db:
            exp = db.get(Experiment, exp_id)
            run = db.get(ChannelRun, job.run_ids[msg["chan_id"]]) if "chan_id" in msg else None
            if t == "worker_started":
                exp.device = msg["device"]
            elif t == "run_started":
                run.status = "running"
                exp.current_channel = msg["chan_id"]
            elif t == "epoch":
                db.add(EpochLog(channel_run_id=run.id, epoch=msg["epoch"], train_loss=msg["train_loss"],
                                val_loss=msg["val_loss"], seconds=msg["seconds"]))
            elif t == "run_done":
                o = msg["outcome"]
                run.status = "completed"
                run.model_path = o["model_path"]
                run.arrays_path = o["arrays_path"]
                run.metrics = o["metrics"]
                run.train_time_s = o["train_time_s"]
                run.infer_time_s = o["infer_time_s"]
                run.epochs_run = o["epochs_run"]
                run.best_epoch = o["best_epoch"]
                run.best_val_loss = o["best_val_loss"]
                run.eval_start = o["metrics"]["eval_start"]
                run.n_detections = len(o["detections"])
                for d in o["detections"]:
                    db.add(Detection(
                        channel_run_id=run.id, experiment_id=exp_id, chan_id=run.chan_id,
                        start=d["start"], end=d["end"], score=d["score"], peak_error=d["peak_error"],
                        threshold=d["threshold"], direction=d["direction"], outcome=d["outcome"],
                        matched_labels=d["matched_labels"],
                        window={**d["window"], "effective": o["effective_threshold"]},
                    ))
                job.finished += 1
                exp.progress = job.finished / max(job.total, 1)
            elif t == "run_failed":
                run.status = "cancelled" if msg.get("cancelled") else "failed"
                run.error = msg["error"]
                job.finished += 1
                exp.progress = job.finished / max(job.total, 1)
            elif t == "worker_error":
                exp.message = ((exp.message or "") + f"\nWorker {msg['worker']} error: {msg['error']}").strip()

    def _finalize(self, exp_id: int) -> None:
        with self.lock:
            job = self.active.pop(exp_id, None)
        if job is None:
            return
        for p in job.processes:
            p.join(timeout=5)
        with session_scope() as db:
            exp = db.get(Experiment, exp_id)
            runs = list(exp.runs)
            for r in runs:
                if r.status in ("pending", "running"):
                    r.status = "cancelled" if job.cancel_event.is_set() else "failed"
                    r.error = r.error or ("cancelled" if job.cancel_event.is_set() else "worker exited")
            sc = {c.chan_id: c.spacecraft for c in db.scalars(select(Channel))}
            exp.metrics = experiment_metrics(runs, sc)
            exp.train_time_s = sum(r.train_time_s or 0 for r in runs)
            exp.infer_time_s = sum(r.infer_time_s or 0 for r in runs)
            n_ok = sum(r.status == "completed" for r in runs)
            if job.cancel_event.is_set():
                exp.status = "cancelled"
            elif n_ok == len(runs):
                exp.status = "completed"
            elif n_ok == 0:
                exp.status = "failed"
            else:
                exp.status = "partial"
            exp.current_channel = None
            exp.finished_at = datetime.now(timezone.utc)


def _worker_entry(payload: dict, q, cancel_event) -> None:
    from app.services.worker import run_worker

    run_worker(payload, q, cancel_event)


job_manager = JobManager()


def wait_for(exp_ids: list[int], poll: float = 2.0, on_poll=None) -> None:
    """Block until the given experiments are no longer queued/running (used by scripts)."""
    while True:
        with session_scope() as db:
            states = {e.id: e.status for e in db.scalars(select(Experiment).where(Experiment.id.in_(exp_ids)))}
        if on_poll:
            on_poll(states)
        if all(s not in ("queued", "running") for s in states.values()):
            return
        time.sleep(poll)


ARTIFACTS_NOTE = Path("artifacts") / "experiments"
