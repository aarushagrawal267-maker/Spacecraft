"""Health, system information, method catalogue and the mission overview."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import __version__
from app.core.config import get_settings
from app.dataset import store
from app.db.models import Channel, ChannelRun, Detection, Experiment, LabelInterval
from app.db.session import get_db
from app.ml import methods
from app.ml.device import system_info
from app.ml.scoring.thresholds import ThresholdConfig
from app.services.dataset_service import current_snapshot
from app.services.experiment_service import summarize

router = APIRouter(tags=["system"])


@router.get("/health", summary="Liveness check")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/system", summary="Compute environment (device, CPU, RAM)")
def system() -> dict:
    info = system_info()
    s = get_settings()
    info["device_preference"] = s.device
    info["max_concurrent_jobs"] = s.max_concurrent_jobs
    ram = info.get("ram_available_gb")
    info["recommendation"] = (
        "Low available memory: keep to the Quick profile and a few channels per experiment."
        if ram is not None and ram < 4 else
        "Start with the Quick profile on one channel, then scale up."
    )
    return info


@router.get("/methods", summary="Available detection methods, profiles and threshold defaults")
def list_methods() -> dict:
    return {
        "eval_start": methods.EVAL_START,
        "methods": [
            {
                "key": s.key, "label": s.label, "family": s.family, "purpose": s.purpose,
                "trainable": s.trainable, "produces_prediction": s.produces_prediction,
                "defaults": s.defaults, "profiles": s.profiles,
            }
            for s in methods.SPECS.values()
        ],
        "threshold_defaults": ThresholdConfig().to_dict(),
    }


@router.get("/overview", summary="Mission overview figures (all from the database)")
def overview(db: Session = Depends(get_db)) -> dict:
    snap = current_snapshot(db)
    chans = db.execute(
        select(Channel.spacecraft, func.count(), func.sum(Channel.test_len), func.sum(Channel.train_len))
        .group_by(Channel.spacecraft)
    ).all()
    seqs = dict(
        db.execute(
            select(Channel.spacecraft, func.count(LabelInterval.id))
            .join(LabelInterval, LabelInterval.chan_id == Channel.chan_id)
            .group_by(Channel.spacecraft)
        ).all()
    )
    status_counts = dict(db.execute(select(Experiment.status, func.count()).group_by(Experiment.status)).all())
    completed_runs = db.scalar(select(func.count()).select_from(ChannelRun).where(ChannelRun.status == "completed"))
    analysed = db.scalar(
        select(func.count(func.distinct(ChannelRun.chan_id))).where(ChannelRun.status == "completed")
    )
    n_det = db.scalar(select(func.count()).select_from(Detection))
    recent = db.scalars(select(Experiment).order_by(Experiment.id.desc()).limit(6)).all()

    # Latest completed all-channel experiment per method → leaderboard
    board = []
    for key in methods.SPECS:
        exp = db.scalars(
            select(Experiment)
            .where(Experiment.method == key, Experiment.status.in_(["completed", "partial"]),
                   Experiment.spacecraft == "ALL")
            .order_by(Experiment.id.desc()).limit(1)
        ).first()
        if exp:
            board.append(summarize(exp))

    return {
        "dataset": {
            "available": store.is_available(),
            "source": snap.source if snap else None,
            "retrieved_at": snap.retrieved_at if snap else None,
            "validation_ok": snap.validation_ok if snap else None,
            "checksum": snap.dataset_checksum if snap else None,
        },
        "spacecraft": {
            sc: {"channels": n, "test_values": int(t or 0), "train_values": int(tr or 0),
                 "label_sequences": int(seqs.get(sc, 0))}
            for sc, n, t, tr in chans
        },
        "experiments": {"by_status": status_counts, "total": sum(status_counts.values())},
        "completed_channel_runs": completed_runs or 0,
        "channels_analysed": analysed or 0,
        "detections_total": n_det or 0,
        "recent_experiments": [summarize(e) for e in recent],
        "leaderboard": board,
    }
