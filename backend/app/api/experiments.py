"""Experiments (training jobs), channel runs, live inference and progress streaming."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.core.config import get_settings, resolve_path
from app.db.models import ChannelRun, EpochLog, Experiment
from app.db.session import SessionLocal, get_db
from app.services import inference_service as inf
from app.services.experiment_service import ExperimentError, create_experiment, run_summary, summarize
from app.services.jobs import job_manager

router = APIRouter(tags=["experiments"])


class ExperimentCreate(BaseModel):
    method: str = Field(examples=["lstm_forecaster"])
    name: str | None = None
    profile: str | None = Field(None, examples=["quick"])
    config: dict | None = Field(None, description="Overrides of the method's hyper-parameters")
    threshold: dict | None = Field(None, description="Overrides of the thresholding configuration")
    spacecraft: str | None = Field(None, description="SMAP, MSL or ALL (used when channels is empty)")
    channels: list[str] | None = None
    seed: int = 0
    workers: int = Field(1, ge=1, le=16, description="Parallel worker processes for this experiment")
    notes: str | None = None


class InferenceRequest(BaseModel):
    threshold: dict | None = Field(None, description="Optional threshold overrides for this inference only")


@router.post("/experiments", summary="Queue a new experiment (training + evaluation job)", status_code=201)
def create(req: ExperimentCreate, db: Session = Depends(get_db)) -> dict:
    try:
        exp = create_experiment(
            db, method=req.method, name=req.name, profile=req.profile, config=req.config,
            threshold=req.threshold, spacecraft=req.spacecraft, channels=req.channels, seed=req.seed,
            workers=req.workers, notes=req.notes,
        )
    except ExperimentError as exc:
        raise HTTPException(422, str(exc)) from None
    return summarize(exp)


@router.get("/experiments", summary="Experiment history")
def list_experiments(
    status: str | None = None, method: str | None = None, origin: str | None = None,
    db: Session = Depends(get_db),
) -> list[dict]:
    q = select(Experiment).order_by(Experiment.id.desc())
    if status:
        q = q.where(Experiment.status == status)
    if method:
        q = q.where(Experiment.method == method)
    if origin:
        q = q.where(Experiment.origin == origin)
    return [summarize(e) for e in db.scalars(q)]


def _get(db: Session, exp_id: int) -> Experiment:
    exp = db.get(Experiment, exp_id)
    if exp is None:
        raise HTTPException(404, f"Experiment {exp_id} not found")
    return exp


@router.get("/experiments/{exp_id}", summary="Experiment detail with per-channel runs and metrics")
def get_experiment(exp_id: int, db: Session = Depends(get_db)) -> dict:
    return summarize(_get(db, exp_id), with_runs=True)


@router.post("/experiments/{exp_id}/cancel", summary="Cancel a queued or running experiment")
def cancel(exp_id: int, db: Session = Depends(get_db)) -> dict:
    _get(db, exp_id)
    return {"result": job_manager.cancel(exp_id)}


@router.delete("/experiments/{exp_id}", summary="Delete an experiment and its local artifacts")
def delete(exp_id: int, db: Session = Depends(get_db)) -> dict:
    exp = _get(db, exp_id)
    if exp.status in ("running",) or job_manager.is_active(exp_id):
        raise HTTPException(409, "Cancel the experiment before deleting it")
    art = get_settings().artifacts_dir / "experiments" / f"exp_{exp_id:05d}"
    db.delete(exp)
    db.commit()
    if art.exists():
        shutil.rmtree(art, ignore_errors=True)
    return {"deleted": exp_id}


@router.get("/experiments/{exp_id}/events", summary="Server-sent progress events")
async def events(exp_id: int, request: Request):
    async def gen():
        last_epoch_id = 0
        last_state = None
        while True:
            if await request.is_disconnected():
                break
            db = SessionLocal()
            try:
                exp = db.get(Experiment, exp_id)
                if exp is None:
                    yield {"event": "error", "data": json.dumps({"detail": "not found"})}
                    break
                rows = db.execute(
                    select(EpochLog, ChannelRun.chan_id)
                    .join(ChannelRun, ChannelRun.id == EpochLog.channel_run_id)
                    .where(ChannelRun.experiment_id == exp_id, EpochLog.id > last_epoch_id)
                    .order_by(EpochLog.id)
                ).all()
                for log, chan in rows:
                    last_epoch_id = log.id
                    yield {"event": "epoch", "data": json.dumps({
                        "chan_id": chan, "epoch": log.epoch, "train_loss": log.train_loss,
                        "val_loss": log.val_loss, "seconds": log.seconds})}
                state = {
                    "status": exp.status, "progress": exp.progress, "current_channel": exp.current_channel,
                    "runs": {r.chan_id: r.status for r in exp.runs},
                }
                if state != last_state:
                    last_state = state
                    yield {"event": "status", "data": json.dumps(state)}
                if exp.status not in ("queued", "running"):
                    yield {"event": "done", "data": json.dumps({"status": exp.status})}
                    break
            finally:
                db.close()
            await asyncio.sleep(1.0)

    return EventSourceResponse(gen())


@router.get("/runs/{run_id}", summary="One channel run: metrics, loss history, detections")
def get_run(run_id: int, db: Session = Depends(get_db)) -> dict:
    run = db.get(ChannelRun, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    d = run_summary(run)
    d["metrics"] = run.metrics
    d["method"] = run.experiment.method
    d["experiment_name"] = run.experiment.name
    d["config"] = run.experiment.config
    d["threshold_config"] = run.experiment.threshold_config
    d["epochs"] = [{"epoch": e.epoch, "train_loss": e.train_loss, "val_loss": e.val_loss, "seconds": e.seconds}
                   for e in run.epochs]
    d["detections"] = [
        {"id": x.id, "start": x.start, "end": x.end, "score": x.score, "peak_error": x.peak_error,
         "threshold": x.threshold, "direction": x.direction, "outcome": x.outcome,
         "matched_labels": x.matched_labels}
        for x in run.detections
    ]
    d["model_file"] = Path(run.model_path).name if run.model_path else None
    d["model_available"] = bool(run.model_path and resolve_path(run.model_path).exists())
    return d


@router.get("/runs/{run_id}/series", summary="Stored per-timestep series (actual, prediction, errors, thresholds)")
def run_series(run_id: int, db: Session = Depends(get_db)) -> dict:
    run = db.get(ChannelRun, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    try:
        return inf.stored_series(run, run.experiment)
    except inf.InferenceError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/runs/{run_id}/infer", summary="Execute the trained model now and re-threshold")
def run_infer(run_id: int, req: InferenceRequest, db: Session = Depends(get_db)) -> dict:
    try:
        return inf.live_inference(db, run_id, req.threshold)
    except KeyError:
        raise HTTPException(404, "Run not found") from None
    except inf.InvalidInferenceRequest as exc:
        raise HTTPException(422, str(exc)) from None
    except inf.InferenceError as exc:
        raise HTTPException(409, str(exc)) from None
