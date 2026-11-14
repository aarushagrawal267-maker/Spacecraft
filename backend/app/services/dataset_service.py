"""Synchronises the installed dataset into the database and runs acquisition jobs."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.dataset import acquire as acq
from app.dataset import store
from app.dataset.validate import validate_dataset
from app.db.models import Channel, DatasetSnapshot, LabelInterval
from app.db.session import session_scope


def current_snapshot(db: Session) -> DatasetSnapshot | None:
    return db.scalars(select(DatasetSnapshot).order_by(DatasetSnapshot.id.desc()).limit(1)).first()


def sync_from_disk(db: Session) -> DatasetSnapshot | None:
    """Register the installed dataset (if any) in the database. Idempotent."""
    if not store.is_available():
        return None
    prov = acq.load_provenance() or {}
    manifest = acq.load_manifest() or {"files": {}, "dataset_checksum": "unknown", "n_files": 0}
    snap = current_snapshot(db)
    if snap and snap.dataset_checksum == manifest["dataset_checksum"] and db.scalar(select(Channel).limit(1)):
        return snap

    report_path = store.dataset_root() / "validation_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = validate_dataset(store.dataset_root(), store.labels()).to_dict()

    snap = DatasetSnapshot(
        source=prov.get("source", "unknown"),
        source_url=prov.get("source_url", ""),
        retrieved_at=prov.get("retrieved_at", ""),
        dataset_checksum=manifest["dataset_checksum"],
        labels_sha256=prov.get("labels_sha256", ""),
        n_files=manifest.get("n_files", 0),
        total_bytes=sum(f["bytes"] for f in manifest["files"].values()),
        validation_ok=bool(report.get("ok")),
        report=report,
        provenance=prov,
    )
    db.add(snap)

    db.execute(delete(LabelInterval))
    db.execute(delete(Channel))
    labels = store.labels()
    for c in report["channels"]:
        lab = labels.channels[c["chan_id"]]
        ch = Channel(
            chan_id=c["chan_id"],
            spacecraft=c["spacecraft"],
            train_len=c["train_shape"][0],
            test_len=c["test_shape"][0],
            n_features=c["test_shape"][1],
            n_intervals=len(lab.intervals),
            labelled_points=c["labelled_points"],
            label_fraction=c["label_fraction"],
            value_min=c["value_min"],
            value_max=c["value_max"],
            train_std=c["train_value_std"],
            commands_active_train=c["command_columns_active_train"],
            commands_active_test=c["command_columns_active_test"],
            label_rows=lab.source_rows,
            issues=c["issues"],
        )
        ch.intervals = [
            LabelInterval(start=iv.start, end=iv.end, cls=iv.cls, merged_from=iv.merged_from)
            for iv in lab.intervals
        ]
        db.add(ch)
    db.commit()
    return snap


class AcquisitionJob:
    """Single background acquisition with pollable progress."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: dict = {"running": False, "message": "idle", "progress": 0.0, "result": None}

    def start(self, source: str) -> bool:
        with self.lock:
            if self.state["running"]:
                return False
            self.state = {"running": True, "message": "starting", "progress": 0.0, "result": None,
                          "source": source, "started_at": datetime.now(timezone.utc).isoformat()}
        threading.Thread(target=self._run, args=(source,), daemon=True).start()
        return True

    def _progress(self, msg: str, frac: float) -> None:
        with self.lock:
            self.state["message"] = msg
            self.state["progress"] = frac

    def _run(self, source: str) -> None:
        try:
            res = acq.acquire(source, self._progress)
            if res.ok:
                with session_scope() as db:
                    sync_from_disk(db)
            result = {"ok": res.ok, "source": res.source, "message": res.message, "attempts": res.attempts}
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "source": None, "message": str(exc), "attempts": []}
        with self.lock:
            self.state.update(running=False, result=result, message=result["message"])


acquisition_job = AcquisitionJob()
