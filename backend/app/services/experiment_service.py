"""Creation, querying and serialisation of experiments."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Channel, ChannelRun, Experiment
from app.dataset import store
from app.ml import methods
from app.ml.scoring.thresholds import ThresholdConfig


class ExperimentError(ValueError):
    pass


def resolve_channels(db: Session, spacecraft: str | None, channels: list[str] | None) -> tuple[str, list[str]]:
    known = {c.chan_id: c.spacecraft for c in db.scalars(select(Channel))}
    if not known:
        raise ExperimentError("Dataset not loaded: no channels registered.")
    if channels:
        unknown = [c for c in channels if c not in known]
        if unknown:
            raise ExperimentError(f"Unknown channels: {unknown}")
        chans = list(dict.fromkeys(channels))
        scs = {known[c] for c in chans}
        return (scs.pop() if len(scs) == 1 else "MIXED"), chans
    sc = (spacecraft or "ALL").upper()
    if sc not in ("SMAP", "MSL", "ALL"):
        raise ExperimentError("spacecraft must be SMAP, MSL or ALL")
    chans = sorted(c for c, s in known.items() if sc == "ALL" or s == sc)
    return sc, chans


def create_experiment(
    db: Session,
    *,
    method: str,
    name: str | None = None,
    profile: str | None = None,
    config: dict | None = None,
    threshold: dict | None = None,
    spacecraft: str | None = None,
    channels: list[str] | None = None,
    seed: int = 0,
    workers: int = 1,
    origin: str = "local",
    notes: str | None = None,
) -> Experiment:
    if method not in methods.SPECS:
        raise ExperimentError(f"Unknown method {method!r}")
    spec = methods.SPECS[method]
    if profile and spec.profiles and profile not in spec.profiles:
        raise ExperimentError(f"Unknown profile {profile!r} for {method}")
    try:
        cfg = methods.resolve_config(method, profile, config)
    except ValueError as exc:
        raise ExperimentError(str(exc)) from exc
    thr = ThresholdConfig.from_dict(threshold)
    if thr.mode not in ("ndt", "static"):
        raise ExperimentError("threshold mode must be 'ndt' or 'static'")
    sc, chans = resolve_channels(db, spacecraft, channels)
    if not chans:
        raise ExperimentError("No channels selected")
    snapshot_checksum = None
    try:
        from app.dataset.acquire import load_manifest

        m = load_manifest()
        snapshot_checksum = m["dataset_checksum"] if m else None
    except Exception:  # noqa: BLE001
        pass
    label = spec.label + (f" · {profile}" if profile else "")
    exp = Experiment(
        name=name or f"{label} · {sc} · {len(chans)} ch",
        method=method,
        profile=profile,
        spacecraft=sc,
        channels=chans,
        config=cfg,
        threshold_config=thr.to_dict(),
        status="queued",
        seed=seed,
        dataset_checksum=snapshot_checksum,
        origin=origin,
        environment={"workers": max(1, int(workers))},
        notes=notes,
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


def summarize(exp: Experiment, with_runs: bool = False) -> dict:
    overall = (exp.metrics or {}).get("overall") or {}
    d = {
        "id": exp.id,
        "name": exp.name,
        "method": exp.method,
        "method_label": methods.SPECS[exp.method].label if exp.method in methods.SPECS else exp.method,
        "profile": exp.profile,
        "spacecraft": exp.spacecraft,
        "channels": exp.channels,
        "n_channels": len(exp.channels),
        "config": exp.config,
        "threshold_config": exp.threshold_config,
        "status": exp.status,
        "progress": exp.progress,
        "current_channel": exp.current_channel,
        "message": exp.message,
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "started_at": exp.started_at.isoformat() if exp.started_at else None,
        "finished_at": exp.finished_at.isoformat() if exp.finished_at else None,
        "device": exp.device,
        "seed": exp.seed,
        "git_commit": exp.git_commit,
        "dataset_checksum": exp.dataset_checksum,
        "train_time_s": exp.train_time_s,
        "infer_time_s": exp.infer_time_s,
        "origin": exp.origin,
        "environment": exp.environment,
        "notes": exp.notes,
        "headline": {
            "event_precision": (overall.get("event") or {}).get("precision"),
            "event_recall": (overall.get("event") or {}).get("recall"),
            "event_f1": (overall.get("event") or {}).get("f1"),
            "event_f0_5": (overall.get("event") or {}).get("f0_5"),
            "point_f1": (overall.get("point") or {}).get("f1"),
            "map": overall.get("mean_average_precision"),
            "n_detections": (overall.get("event") or {}).get("n_detections"),
        },
    }
    if with_runs:
        d["metrics"] = exp.metrics
        d["runs"] = [run_summary(r) for r in exp.runs]
    return d


def run_summary(r: ChannelRun) -> dict:
    m = r.metrics or {}
    ev = m.get("event") or {}
    pt = m.get("point") or {}
    return {
        "id": r.id,
        "experiment_id": r.experiment_id,
        "chan_id": r.chan_id,
        "spacecraft": _spacecraft(r.chan_id),
        "status": r.status,
        "epochs_run": r.epochs_run,
        "best_epoch": r.best_epoch,
        "best_val_loss": r.best_val_loss,
        "train_time_s": r.train_time_s,
        "infer_time_s": r.infer_time_s,
        "n_detections": r.n_detections,
        "has_model": bool(r.model_path),
        "has_arrays": bool(r.arrays_path),
        "error": r.error,
        "event": {k: ev.get(k) for k in ("tp", "fp", "fn", "precision", "recall", "f1", "f0_5", "mean_delay")},
        "point": {k: pt.get(k) for k in ("precision", "recall", "f1", "fpr")},
        "average_precision": (m.get("curves") or {}).get("average_precision"),
    }


def _spacecraft(chan_id: str) -> str | None:
    try:
        return store.channel_labels(chan_id).spacecraft
    except Exception:  # noqa: BLE001
        return None
