"""Dataset status/acquisition, channel inventory and telemetry access."""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dataset import acquire as acq
from app.dataset import sources, store
from app.dataset.validate import validate_dataset
from app.db.models import Channel, ChannelRun, Experiment
from app.db.session import get_db
from app.services import dataset_service
from app.services.inference_service import to_list

router = APIRouter(tags=["dataset"])


class AcquireRequest(BaseModel):
    source: Literal["auto", "kaggle", "github_mirror", "manual"] = "auto"


@router.get("/dataset/status", summary="Installed dataset, provenance and validation summary")
def dataset_status(db: Session = Depends(get_db)) -> dict:
    snap = dataset_service.current_snapshot(db)
    return {
        "available": store.is_available(),
        "acquisition": dataset_service.acquisition_job.state,
        "snapshot": None if snap is None else {
            "id": snap.id, "source": snap.source, "source_url": snap.source_url,
            "retrieved_at": snap.retrieved_at, "dataset_checksum": snap.dataset_checksum,
            "labels_sha256": snap.labels_sha256, "n_files": snap.n_files, "total_bytes": snap.total_bytes,
            "validation_ok": snap.validation_ok,
            "summary": snap.report.get("summary"),
            "errors": snap.report.get("errors"),
            "warnings": snap.report.get("warnings"),
            "provenance": snap.provenance,
        },
        "sources": [s.__dict__ for s in sources.SOURCES.values()],
        "disclosures": [
            "Telemetry channel identifiers are anonymised; they are not real instrument names. "
            "Only the first letter is documented as a channel type (e.g. P = power, R = radiation).",
            "Time is anonymised: values are indexed by position; there are no timestamps or engineering units.",
            "Values are pre-processed. The Telemanom README states telemetry was scaled to (-1, 1) using the "
            "test-set min/max; several channels nevertheless exceed that range (see warnings).",
            "Inputs other than column 0 are one-hot encoded command information.",
            "Anomaly intervals are expert benchmark reference annotations, not confirmed hardware failures. "
            "Unlabelled regions are not guaranteed to be normal.",
            "This is an independent research implementation inspired by Hundman et al. (KDD 2018); it is not "
            "affiliated with, commissioned or used by NASA/JPL.",
        ],
        "license_note": sources.LICENSE_NOTE,
    }


@router.post("/dataset/acquire", summary="Download, verify and install the benchmark (background)")
def dataset_acquire(req: AcquireRequest) -> dict:
    started = dataset_service.acquisition_job.start(req.source)
    if not started:
        raise HTTPException(409, "An acquisition is already running")
    return {"started": True, "source": req.source}


@router.post("/dataset/validate", summary="Re-run validation on the installed files")
def dataset_validate() -> dict:
    if not store.is_available():
        raise HTTPException(404, "Dataset not installed")
    rep = validate_dataset(store.dataset_root(), store.labels())
    manifest = acq.load_manifest() or {"files": {}}
    mismatched = []
    for rel, info in manifest.get("files", {}).items():
        p = store.dataset_root() / rel
        if not p.exists() or acq.sha256_file(p) != info["sha256"]:
            mismatched.append(rel)
    return {"ok": rep.ok and not mismatched, "checksum_mismatches": mismatched, "errors": rep.errors,
            "warnings": rep.warnings, "summary": rep.summary}


@router.get("/dataset/report", summary="Full stored validation report")
def dataset_report(db: Session = Depends(get_db)) -> dict:
    snap = dataset_service.current_snapshot(db)
    if snap is None:
        raise HTTPException(404, "Dataset not registered")
    return snap.report


def _channel_dict(c: Channel, runs_by_chan: dict | None = None) -> dict:
    d = {
        "chan_id": c.chan_id, "spacecraft": c.spacecraft, "type_letter": c.chan_id.split("-")[0],
        "train_len": c.train_len, "test_len": c.test_len, "n_features": c.n_features,
        "n_intervals": c.n_intervals, "labelled_points": c.labelled_points, "label_fraction": c.label_fraction,
        "value_min": c.value_min, "value_max": c.value_max, "train_std": c.train_std,
        "commands_active_train": c.commands_active_train, "commands_active_test": c.commands_active_test,
        "label_rows": c.label_rows, "issues": c.issues,
        "intervals": [{"start": i.start, "end": i.end, "cls": i.cls, "merged_from": i.merged_from}
                      for i in c.intervals],
    }
    if runs_by_chan is not None:
        d["runs"] = runs_by_chan.get(c.chan_id, [])
    return d


@router.get("/channels", summary="Channel inventory with label intervals and trained runs")
def list_channels(spacecraft: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    q = select(Channel).order_by(Channel.spacecraft.desc(), Channel.chan_id)
    if spacecraft:
        q = q.where(Channel.spacecraft == spacecraft.upper())
    rows = db.execute(
        select(ChannelRun.chan_id, ChannelRun.id, Experiment.id, Experiment.method, Experiment.name,
               ChannelRun.status, ChannelRun.model_path.is_not(None))
        .join(Experiment, Experiment.id == ChannelRun.experiment_id)
        .where(ChannelRun.status == "completed")
        .order_by(ChannelRun.id.desc())
    ).all()
    runs_by_chan: dict[str, list] = {}
    for chan, rid, eid, method, ename, status, has_model in rows:
        runs_by_chan.setdefault(chan, []).append(
            {"run_id": rid, "experiment_id": eid, "method": method, "experiment_name": ename, "has_model": has_model}
        )
    return [_channel_dict(c, runs_by_chan) for c in db.scalars(q)]


@router.get("/channels/{chan_id}", summary="One channel")
def get_channel(chan_id: str, db: Session = Depends(get_db)) -> dict:
    c = db.get(Channel, chan_id)
    if c is None:
        raise HTTPException(404, f"Unknown channel {chan_id}")
    return _channel_dict(c)


def _decimate(y: np.ndarray, max_points: int) -> tuple[list[int], list]:
    """Min/max bucket decimation (keeps spikes visible)."""
    n = len(y)
    if n <= max_points:
        return list(range(n)), to_list(y)
    buckets = max_points // 2
    edges = np.linspace(0, n, buckets + 1).astype(int)
    idx: list[int] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        i1, i2 = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        idx.extend(sorted({int(i1), int(i2)}))
    return idx, to_list(y[idx])


@router.get("/telemetry/{chan_id}", summary="Telemetry values for a channel split")
def telemetry(
    chan_id: str,
    split: Literal["train", "test"] = "test",
    start: int = 0,
    end: int | None = None,
    max_points: int | None = Query(None, ge=50, le=20000),
) -> dict:
    try:
        arr = store.load_split(chan_id, split)
    except KeyError:
        raise HTTPException(404, f"Unknown channel {chan_id}") from None
    except store.DatasetNotAvailable as exc:
        raise HTTPException(404, str(exc)) from None
    n = len(arr)
    end = n if end is None else min(end, n)
    start = max(0, min(start, end))
    y = arr[start:end, 0]
    cmd = (arr[start:end, 1:] != 0).sum(axis=1)
    if max_points:
        idx, vals = _decimate(y, max_points)
        index = [start + i for i in idx]
        cmd_vals = cmd[idx].astype(int).tolist()
    else:
        index, vals, cmd_vals = list(range(start, end)), to_list(y), cmd.astype(int).tolist()
    lab = store.channel_labels(chan_id)
    return {
        "chan_id": chan_id,
        "spacecraft": lab.spacecraft,
        "split": split,
        "n": n,
        "start": start,
        "end": end,
        "index": index,
        "values": vals,
        "command_activity": cmd_vals,
        "n_features": int(arr.shape[1]),
        "labels": [{"start": i.start, "end": i.end, "cls": i.cls, "merged_from": i.merged_from}
                   for i in lab.intervals] if split == "test" else [],
        "stats": {
            "min": float(arr[:, 0].min()), "max": float(arr[:, 0].max()), "mean": float(arr[:, 0].mean()),
            "std": float(arr[:, 0].std()),
        },
    }
