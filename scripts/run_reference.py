"""Run the reference experiments over all 81 benchmark channels.

Uses exactly the same job system as the web UI (queued experiments, worker
processes, single DB writer). Results land in the local SQLite database and
artifacts/; scripts/export_reference.py then packages them into reference/.

Usage (from the project root, with the backend virtualenv):
    backend/.venv/bin/python scripts/run_reference.py --workers 3
    backend\\.venv\\Scripts\\python scripts\\run_reference.py --workers 2   (Windows)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

PLAN = [
    # (method, profile, workers_override)
    ("persistence", None, 1),
    ("random", None, 1),
    ("rolling_zscore", None, 1),
    ("ridge_ar", None, 2),
    ("isolation_forest", None, 2),
    ("lstm_forecaster", "standard", None),
    ("lstm_autoencoder", "standard", None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="subset of methods to run")
    ap.add_argument("--channels", nargs="*", help="subset of channels (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from app.db.session import migrate, session_scope
    from app.services import dataset_service
    from app.services.experiment_service import create_experiment
    from app.services.jobs import job_manager, wait_for

    migrate()
    with session_scope() as db:
        if dataset_service.sync_from_disk(db) is None:
            sys.exit("Dataset not installed. Run the acquisition first.")
        ids = []
        for method, profile, w in PLAN:
            if args.only and method not in args.only:
                continue
            exp = create_experiment(
                db, method=method, profile=profile, spacecraft="ALL", channels=args.channels,
                seed=args.seed, workers=w or args.workers, origin="reference",
                name=None, notes="Reference run (all channels), produced with scripts/run_reference.py",
            )
            ids.append(exp.id)
            print(f"queued experiment {exp.id}: {exp.name}", flush=True)

    job_manager.start()
    last = {}

    def report(states):
        nonlocal last
        if states != last:
            print(time.strftime("%H:%M:%S"), states, flush=True)
            last = dict(states)

    try:
        wait_for(ids, poll=5.0, on_poll=report)
    finally:
        job_manager.shutdown()
    print("done", flush=True)


if __name__ == "__main__":
    main()
