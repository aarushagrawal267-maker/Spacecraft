"""Build (and optionally publish) the Heliograph browser demo: a static viewer of the stored reference results.

    python deploy/browser/build.py --out build/browser-demo
    python deploy/browser/build.py --out build/browser-demo --space <hf-user>/heliograph   # needs huggingface_hub

Nothing is trained or computed for display here. The page shows what is committed in reference/: for each of the
seven methods and 81 channels, the stored model outputs (forecast, smoothed error, dynamic thresholds), the
detected intervals and the evaluation metrics. The telemetry itself is not copied: the page downloads a channel's
test stream from the public GitHub copy of the benchmark (the same pinned source setup uses) when it is opened.
The full app (training, live inference, replay, the model laboratory) runs locally.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.dataset.labels import parse_label_text  # noqa: E402
from app.dataset.sources import LABELS_SHA256, LABELS_URL, MIRROR_BASE  # noqa: E402
from app.ml.methods import SPECS  # noqa: E402

ARRAYS = ("prediction", "smoothed", "threshold_high", "threshold_low")
METHOD_ORDER = ["lstm_forecaster_standard", "lstm_autoencoder_standard", "ridge_ar", "persistence", "rolling_zscore",
                "isolation_forest", "random"]


def label_text() -> str:
    """The official label file: from the installed benchmark, else downloaded (pinned commit); SHA-256 checked."""
    from app.dataset import store

    local = store.dataset_root() / "labeled_anomalies.csv"  # the installed benchmark, if any
    if local.is_file():
        data = local.read_bytes()
    else:
        with urllib.request.urlopen(LABELS_URL, timeout=60) as r:  # noqa: S310 - pinned https URL
            data = r.read()
    import hashlib

    if hashlib.sha256(data).hexdigest() != LABELS_SHA256:
        raise SystemExit("labeled_anomalies.csv does not match the pinned SHA-256")
    return data.decode()


def pick(d: dict, keys) -> dict:
    return {k: d.get(k) for k in keys}


def summarise_metrics(m: dict) -> dict:
    ev = m.get("event") or {}
    pt = m.get("point") or {}
    return {
        "event": pick(ev, ("tp", "fp", "fn", "n_labels", "n_detections", "precision", "recall", "f1", "f0_5",
                           "false_alarms_per_10k", "mean_delay", "median_delay", "mean_iou")),
        "point": pick(pt, ("precision", "recall", "f1")),
        "mean_average_precision": m.get("mean_average_precision"),
    }


def build(out: Path) -> None:
    manifest = json.loads((ROOT / "reference" / "manifest.json").read_text())
    exps = {e["key"]: json.loads((ROOT / "reference" / e["file"]).read_text()) for e in manifest["experiments"]}
    order = [k for k in METHOD_ORDER if k in exps] + [k for k in exps if k not in METHOD_ORDER]
    labels = parse_label_text(label_text())

    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "channels").mkdir(parents=True)
    methods = []
    for key in order:
        e = exps[key]
        methods.append({
            "key": key, "name": e["name"].split(" · ALL")[0], "method": e["method"], "profile": e.get("profile"),
            "family": SPECS[e["method"]].family, "produces_prediction": SPECS[e["method"]].produces_prediction,
            "config": e.get("config"), "threshold_mode": (e.get("threshold_config") or {}).get("mode"),
            "train_time_s": e.get("train_time_s"), "git_commit": e.get("git_commit"),
            "overall": summarise_metrics(e["metrics"]["overall"]),
            "by_spacecraft": {sc: summarise_metrics(v) for sc, v in e["metrics"]["by_spacecraft"].items()},
            "curves": {k: pick(v, ("average_precision", "roc_auc")) for k, v in e["metrics"].get("curves", {}).items()},
        })

    channels = []
    total_bytes = 0
    for chan in sorted(labels.channels, key=lambda c: (labels.channels[c].spacecraft, c)):
        lab = labels.channels[chan]
        blobs, meta = [], {}
        offset = 0
        for key in order:
            run = next((r for r in exps[key]["runs"] if r["chan_id"] == chan), None)
            if run is None or not run.get("arrays_path"):
                continue
            with np.load(ROOT / run["arrays_path"]) as a:
                arrays = {}
                for name in ARRAYS:
                    if name in a.files and a[name].size:
                        v = a[name].astype("<f4")
                        if len(v) != lab.num_values:
                            raise SystemExit(f"{key}/{chan}: {name} has {len(v)} values, expected {lab.num_values}")
                        blobs.append(v.tobytes())
                        arrays[name] = offset
                        offset += len(v)
                eval_start = int(a["eval_start"])
            m = run["metrics"]
            meta[key] = {
                "eval_start": eval_start, "arrays": arrays,
                "epochs_run": run.get("epochs_run"), "best_epoch": run.get("best_epoch"),
                "train_time_s": run.get("train_time_s"),
                "metrics": {**summarise_metrics(m), "average_precision": (m.get("curves") or {}).get("average_precision")},
                "per_label": (m.get("event") or {}).get("per_label", []),
                "detections": [pick(d, ("start", "end", "score", "direction", "outcome", "threshold", "peak_error"))
                               for d in run.get("detections", [])],
            }
        (out / "data" / "channels" / f"{chan}.bin").write_bytes(b"".join(blobs))
        total_bytes += offset * 4
        (out / "data" / "channels" / f"{chan}.json").write_text(json.dumps({
            "chan_id": chan, "spacecraft": lab.spacecraft, "n": lab.num_values,
            "labels": [{"start": i.start, "end": i.end, "cls": i.cls, "merged_from": i.merged_from} for i in lab.intervals],
            "methods": meta,
        }, separators=(",", ":")))
        channels.append({"chan_id": chan, "spacecraft": lab.spacecraft, "n": lab.num_values, "n_labels": len(lab.intervals)})

    (out / "data" / "summary.json").write_text(json.dumps({
        "reference_created_at": manifest.get("created_at"), "methods": methods, "channels": channels,
        "telemetry_base": MIRROR_BASE, "labels_url": LABELS_URL,
    }, indent=1))
    for name in ("index.html", "viewer.js", "viewer.css"):
        shutil.copy2(HERE / name, out / name)
    shutil.copy2(ROOT / "frontend" / "public" / "favicon.svg", out / "favicon.svg")
    shutil.copy2(HERE / "README.md", out / "README.md")
    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out}: {len(methods)} methods, {len(channels)} channels, {n_files} files, {size / 1e6:.1f} MB "
          f"({total_bytes / 1e6:.1f} MB of stored model outputs)")


def publish(out: Path, space: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=str(out), commit_message="Publish browser demo",
                      delete_patterns=["*"])
    owner, name = space.split("/", 1)
    print(f"Space page: https://huggingface.co/spaces/{space}")
    print(f"App URL:    https://{owner.lower()}-{name.lower().replace('_', '-').replace('.', '-')}.static.hf.space")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "build" / "browser-demo"))
    ap.add_argument("--space", help="also publish to this Hugging Face Space (static SDK, free)")
    a = ap.parse_args()
    build(Path(a.out))
    if a.space:
        publish(Path(a.out), a.space)


if __name__ == "__main__":
    main()
