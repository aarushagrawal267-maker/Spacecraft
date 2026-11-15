"""Historical telemetry replay.

Reveals an existing *test* telemetry stream point by point. For each advance,
the trained model is executed on the newly revealed points only (causal
windows), the EWMA is updated incrementally and every threshold window is
evaluated as soon as it is complete. Verdicts therefore arrive with the same
latency a streaming deployment of this method would have. At the end of the
stream the result is identical to offline detection (tests/integration).

This is a replay of benchmark data, not a live spacecraft connection.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from app.dataset import store
from app.ml import methods
from app.ml.pipeline import label_dicts
from app.ml.scoring.smoothing import EWMA, ewma
from app.ml.scoring.thresholds import DynamicThreshold, StaticThreshold, ThresholdConfig
from app.services.inference_service import get_run, load_fitted, to_list

SESSION_TTL_S = 30 * 60
MAX_ADVANCE = 2000


@dataclass
class ReplaySession:
    id: str
    run_id: int
    chan_id: str
    method: str
    fitted: methods.Fitted
    test: np.ndarray
    thr: ThresholdConfig
    eval_start: int
    detector: object
    ewma: EWMA
    cursor: int = 0
    verdict_upto: int = 0  # in eval coordinates
    touched: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def n(self) -> int:
        return len(self.test)


class ReplayManager:
    def __init__(self) -> None:
        self.sessions: dict[str, ReplaySession] = {}
        self.lock = threading.Lock()

    def _gc(self) -> None:
        now = time.time()
        with self.lock:
            for sid in [s for s, v in self.sessions.items() if now - v.touched > SESSION_TTL_S]:
                del self.sessions[sid]

    def create(self, db: Session, run_id: int) -> dict:
        self._gc()
        run, exp = get_run(db, run_id)
        fitted = load_fitted(run, exp)
        test = store.load_split(run.chan_id, "test")
        thr = ThresholdConfig.from_dict(exp.threshold_config)
        es = methods.EVAL_START
        if thr.mode == "static":
            tr_raw = methods.score(fitted, store.load_split(run.chan_id, "train"))[1]
            tr_raw = tr_raw[np.isfinite(tr_raw)]
            det = StaticThreshold.from_training_errors(ewma(tr_raw, thr.span), len(test) - es, thr.static_k)
        else:
            det = DynamicThreshold(len(test) - es, thr)
        s = ReplaySession(
            id=uuid.uuid4().hex[:12], run_id=run.id, chan_id=run.chan_id, method=exp.method, fitted=fitted,
            test=test, thr=thr, eval_start=es, detector=det, ewma=EWMA(thr.span),
        )
        with self.lock:
            self.sessions[s.id] = s
        return {
            "session_id": s.id,
            "run_id": run.id,
            "experiment_id": exp.id,
            "chan_id": run.chan_id,
            "method": exp.method,
            "method_label": methods.SPECS[exp.method].label,
            "n": s.n,
            "eval_start": es,
            "threshold_mode": thr.mode,
            "batch_size": getattr(det, "bs", None),
            "window_batches": getattr(det, "ws", None),
            "labels": label_dicts(run.chan_id),
            "produces_prediction": methods.SPECS[exp.method].produces_prediction,
            "notice": "Historical replay of benchmark test telemetry — not a live spacecraft connection.",
        }

    def get(self, sid: str) -> ReplaySession:
        with self.lock:
            s = self.sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        s.touched = time.time()
        return s

    def advance(self, sid: str, k: int) -> dict:
        s = self.get(sid)
        k = int(max(1, min(k, MAX_ADVANCE)))
        with s.lock:
            t0 = time.perf_counter()
            start, end = s.cursor, min(s.n, s.cursor + k)
            if start >= end:
                return {"start": start, "end": end, "done": True, "cursor": s.cursor}
            pred, raw = methods.score(s.fitted, s.test, start, end)
            smoothed = np.full(end - start, np.nan)
            es = s.eval_start
            if end > es:
                lo = max(start, es)
                e = np.nan_to_num(np.asarray(raw[lo - start :], dtype=np.float64), nan=0.0)
                e_s = s.ewma.update(e)
                smoothed[lo - start :] = e_s
                y = s.test[lo:end, 0]
                if isinstance(s.detector, DynamicThreshold):
                    s.detector.push(e_s, y)
                    if end == s.n:
                        s.detector.finalize()
                else:
                    s.detector.push(e_s)
            s.cursor = end
            det = s.detector
            old, new = s.verdict_upto, det.processed_upto
            s.verdict_upto = new
            verdict = {
                "from": old + es,
                "to": new + es,
                "threshold_high": to_list(det.threshold_high[old:new]),
                "threshold_low": to_list(det.threshold_low[old:new]),
                "flags": det.flags[old:new].astype(int).tolist(),
            }
            intervals = [
                {"start": iv.start, "end": iv.end, "score": iv.score, "direction": iv.direction,
                 "peak_error": iv.peak_error, "threshold": iv.threshold}
                for iv in det.intervals(offset=es)
            ]
            return {
                "start": start,
                "end": end,
                "cursor": end,
                "done": end == s.n,
                "actual": to_list(s.test[start:end, 0]),
                "prediction": to_list(pred) if pred is not None and methods.SPECS[s.method].produces_prediction else None,
                "raw_score": to_list(raw),
                "smoothed": to_list(smoothed),
                "verdict": verdict,
                "intervals": intervals,
                "compute_ms": (time.perf_counter() - t0) * 1000,
            }

    def delete(self, sid: str) -> None:
        with self.lock:
            self.sessions.pop(sid, None)


replay_manager = ReplayManager()
