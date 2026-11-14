"""Detections, evaluation comparisons, historical replay and exports."""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Detection, Experiment
from app.db.session import get_db
from app.ml import methods
from app.services import inference_service as inf
from app.services.experiment_service import run_summary, summarize
from app.services.replay_service import replay_manager

router = APIRouter()


# ----------------------------------------------------------------------------- detections
@router.get("/detections", tags=["detections"], summary="Model-detected anomaly intervals")
def list_detections(
    experiment_id: int | None = None,
    chan_id: str | None = None,
    outcome: str | None = Query(None, description="matches_label | no_label_overlap"),
    limit: int = Query(500, le=5000),
    db: Session = Depends(get_db),
) -> list[dict]:
    q = select(Detection, Experiment.name, Experiment.method).join(Experiment, Experiment.id == Detection.experiment_id)
    if experiment_id:
        q = q.where(Detection.experiment_id == experiment_id)
    if chan_id:
        q = q.where(Detection.chan_id == chan_id)
    if outcome:
        q = q.where(Detection.outcome == outcome)
    q = q.order_by(Detection.score.desc()).limit(limit)
    return [
        {"id": d.id, "experiment_id": d.experiment_id, "experiment_name": name, "method": method,
         "method_label": methods.SPECS[method].label if method in methods.SPECS else method,
         "run_id": d.channel_run_id, "chan_id": d.chan_id, "start": d.start, "end": d.end,
         "length": d.end - d.start + 1, "score": d.score, "peak_error": d.peak_error, "threshold": d.threshold,
         "direction": d.direction, "outcome": d.outcome, "matched_labels": d.matched_labels}
        for d, name, method in db.execute(q).all()
    ]


@router.get("/detections/{det_id}", tags=["detections"], summary="Detected interval with surrounding context")
def get_detection(det_id: int, context: int = Query(300, ge=0, le=3000), db: Session = Depends(get_db)) -> dict:
    try:
        return inf.detection_detail(db, det_id, context)
    except KeyError:
        raise HTTPException(404, "Detection not found") from None


# ----------------------------------------------------------------------------- evaluation
@router.get("/evaluation/compare", tags=["evaluation"], summary="Side-by-side metrics of experiments")
def compare(ids: str = Query(..., description="Comma-separated experiment ids"), db: Session = Depends(get_db)) -> dict:
    try:
        id_list = [int(x) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(422, "ids must be integers") from None
    out = []
    for i in id_list:
        exp = db.get(Experiment, i)
        if exp is None:
            continue
        d = summarize(exp)
        d["metrics"] = exp.metrics
        d["runs"] = [run_summary(r) for r in exp.runs]
        out.append(d)
    return {"experiments": out}


# ----------------------------------------------------------------------------- replay
class ReplayCreate(BaseModel):
    run_id: int


class ReplayAdvance(BaseModel):
    n: int = Field(25, ge=1, le=2000)


@router.post("/replay", tags=["replay"], summary="Start a historical replay session for a trained run")
def replay_create(req: ReplayCreate, db: Session = Depends(get_db)) -> dict:
    try:
        return replay_manager.create(db, req.run_id)
    except KeyError:
        raise HTTPException(404, "Run not found") from None
    except inf.InferenceError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/replay/{sid}/advance", tags=["replay"], summary="Reveal the next n points and evaluate them")
def replay_advance(sid: str, req: ReplayAdvance) -> dict:
    try:
        return replay_manager.advance(sid, req.n)
    except KeyError:
        raise HTTPException(404, "Replay session not found or expired") from None


@router.delete("/replay/{sid}", tags=["replay"], summary="End a replay session")
def replay_delete(sid: str) -> dict:
    replay_manager.delete(sid)
    return {"deleted": sid}


# ----------------------------------------------------------------------------- export
def _csv(rows: list[dict], fields: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


@router.get("/export/experiments/{exp_id}.json", tags=["export"], summary="Full experiment record as JSON")
def export_json(exp_id: int, db: Session = Depends(get_db)) -> Response:
    exp = db.get(Experiment, exp_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    d = summarize(exp, with_runs=True)
    d["detections"] = [
        {"chan_id": x.chan_id, "start": x.start, "end": x.end, "score": x.score, "direction": x.direction,
         "outcome": x.outcome, "matched_labels": x.matched_labels}
        for r in exp.runs for x in r.detections
    ]
    return Response(json.dumps(d, indent=1, default=str), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="experiment_{exp_id}.json"'})


@router.get("/export/experiments/{exp_id}/channels.csv", tags=["export"], summary="Per-channel metrics as CSV")
def export_channels_csv(exp_id: int, db: Session = Depends(get_db)) -> Response:
    exp = db.get(Experiment, exp_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    rows = []
    for r in exp.runs:
        s = run_summary(r)
        rows.append({
            "chan_id": s["chan_id"], "spacecraft": s["spacecraft"], "status": s["status"],
            "event_tp": s["event"]["tp"], "event_fp": s["event"]["fp"], "event_fn": s["event"]["fn"],
            "event_precision": s["event"]["precision"], "event_recall": s["event"]["recall"],
            "event_f1": s["event"]["f1"], "point_precision": s["point"]["precision"],
            "point_recall": s["point"]["recall"], "point_f1": s["point"]["f1"], "point_fpr": s["point"]["fpr"],
            "average_precision": s["average_precision"], "train_time_s": s["train_time_s"],
            "infer_time_s": s["infer_time_s"], "epochs_run": s["epochs_run"],
        })
    fields = list(rows[0].keys()) if rows else ["chan_id"]
    return Response(_csv(rows, fields), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="experiment_{exp_id}_channels.csv"'})


@router.get("/export/experiments/{exp_id}/detections.csv", tags=["export"], summary="Detected intervals as CSV")
def export_detections_csv(exp_id: int, db: Session = Depends(get_db)) -> Response:
    exp = db.get(Experiment, exp_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    rows = [
        {"chan_id": x.chan_id, "start": x.start, "end": x.end, "score": x.score, "peak_error": x.peak_error,
         "threshold": x.threshold, "direction": x.direction, "outcome": x.outcome,
         "matched_labels": json.dumps(x.matched_labels)}
        for r in exp.runs for x in r.detections
    ]
    fields = ["chan_id", "start", "end", "score", "peak_error", "threshold", "direction", "outcome", "matched_labels"]
    return Response(_csv(rows, fields), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="experiment_{exp_id}_detections.csv"'})
