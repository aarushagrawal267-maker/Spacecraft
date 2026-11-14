"""Export / import of the committed reference bundle (``reference/``).

The bundle holds results that were really computed with this code on the full
benchmark (see reference/manifest.json for when, where and with which commit).
It is imported into an empty local database on first start so the application
shows genuine results before the user has trained anything locally. Imported
experiments are marked ``origin = "reference"`` everywhere in the UI.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings, resolve_path
from app.db.models import ChannelRun, Detection, EpochLog, Experiment
from app.db.session import session_scope

BUNDLE_VERSION = 1


def _iso(dt):
    return dt.isoformat() if dt else None


def _parse(s):
    return datetime.fromisoformat(s) if s else None


def export_bundle(exp_ids: list[int], out_dir: Path, include_models: set[str], include_arrays: set[str]) -> dict:
    """Write the selected experiments to ``out_dir`` (relative file paths inside)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"bundle_version": BUNDLE_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                "experiments": []}
    with session_scope() as db:
        for eid in exp_ids:
            exp = db.get(Experiment, eid)
            key = f"{exp.method}{'_' + exp.profile if exp.profile else ''}"
            runs = []
            for r in exp.runs:
                model_rel = arrays_rel = None
                if exp.method in include_models and r.model_path and resolve_path(r.model_path).exists():
                    src = resolve_path(r.model_path)
                    dst = out_dir / "models" / key / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    model_rel = dst.relative_to(get_settings().project_root).as_posix()
                if exp.method in include_arrays and r.arrays_path and resolve_path(r.arrays_path).exists():
                    src = resolve_path(r.arrays_path)
                    dst = out_dir / "arrays" / key / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    arrays_rel = dst.relative_to(get_settings().project_root).as_posix()
                runs.append({
                    "chan_id": r.chan_id, "status": r.status, "epochs_run": r.epochs_run,
                    "best_epoch": r.best_epoch, "best_val_loss": r.best_val_loss,
                    "train_time_s": r.train_time_s, "infer_time_s": r.infer_time_s,
                    "model_path": model_rel, "arrays_path": arrays_rel, "eval_start": r.eval_start,
                    "metrics": r.metrics, "n_detections": r.n_detections, "error": r.error,
                    "epochs": [{"epoch": e.epoch, "train_loss": e.train_loss, "val_loss": e.val_loss,
                                "seconds": e.seconds} for e in r.epochs],
                    "detections": [{"start": d.start, "end": d.end, "score": d.score, "peak_error": d.peak_error,
                                    "threshold": d.threshold, "direction": d.direction, "outcome": d.outcome,
                                    "matched_labels": d.matched_labels, "window": d.window}
                                   for d in r.detections],
                })
            rec = {
                "key": key, "name": exp.name, "method": exp.method, "profile": exp.profile,
                "spacecraft": exp.spacecraft, "channels": exp.channels, "config": exp.config,
                "threshold_config": exp.threshold_config, "status": exp.status,
                "created_at": _iso(exp.created_at), "started_at": _iso(exp.started_at),
                "finished_at": _iso(exp.finished_at), "device": exp.device, "seed": exp.seed,
                "git_commit": exp.git_commit, "dataset_checksum": exp.dataset_checksum,
                "train_time_s": exp.train_time_s, "infer_time_s": exp.infer_time_s, "metrics": exp.metrics,
                "environment": exp.environment, "notes": exp.notes, "runs": runs,
            }
            path = out_dir / "experiments" / f"{key}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, separators=(",", ":"), default=str))
            manifest["experiments"].append({"key": key, "file": f"experiments/{key}.json", "name": exp.name,
                                            "status": exp.status, "git_commit": exp.git_commit,
                                            "has_models": exp.method in include_models,
                                            "has_arrays": exp.method in include_arrays})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def import_if_needed(bundle_dir: Path | None = None) -> int:
    """Import the bundle when the database has no reference experiments yet."""
    bundle_dir = bundle_dir or get_settings().reference_dir
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return 0
    with session_scope() as db:
        if db.scalar(select(Experiment.id).where(Experiment.origin == "reference").limit(1)):
            return 0
    manifest = json.loads(manifest_path.read_text())
    n = 0
    for entry in manifest["experiments"]:
        rec = json.loads((bundle_dir / entry["file"]).read_text())
        with session_scope() as db:
            exp = Experiment(
                name=rec["name"], method=rec["method"], profile=rec["profile"], spacecraft=rec["spacecraft"],
                channels=rec["channels"], config=rec["config"], threshold_config=rec["threshold_config"],
                status=rec["status"], progress=1.0, created_at=_parse(rec["created_at"]),
                started_at=_parse(rec["started_at"]), finished_at=_parse(rec["finished_at"]),
                device=rec["device"], seed=rec["seed"], git_commit=rec["git_commit"],
                dataset_checksum=rec["dataset_checksum"], train_time_s=rec["train_time_s"],
                infer_time_s=rec["infer_time_s"], metrics=rec["metrics"], origin="reference",
                environment=rec["environment"], notes=rec["notes"],
            )
            db.add(exp)
            db.flush()
            for r in rec["runs"]:
                run = ChannelRun(
                    experiment_id=exp.id, chan_id=r["chan_id"], status=r["status"], epochs_run=r["epochs_run"],
                    best_epoch=r["best_epoch"], best_val_loss=r["best_val_loss"], train_time_s=r["train_time_s"],
                    infer_time_s=r["infer_time_s"], model_path=r["model_path"], arrays_path=r["arrays_path"],
                    eval_start=r["eval_start"], metrics=r["metrics"], n_detections=r["n_detections"],
                    error=r["error"],
                )
                db.add(run)
                db.flush()
                for e in r["epochs"]:
                    db.add(EpochLog(channel_run_id=run.id, **e))
                for d in r["detections"]:
                    db.add(Detection(channel_run_id=run.id, experiment_id=exp.id, chan_id=r["chan_id"], **d))
            n += 1
    return n
