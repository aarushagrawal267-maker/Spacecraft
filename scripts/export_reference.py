"""Package completed reference experiments into reference/ (committed to git).

Model checkpoints are included for the methods listed in --models (small torch /
ridge files); per-timestep arrays for --arrays. Methods without arrays can be
re-run locally; their metrics and detections are always included.

    python scripts/export_reference.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["lstm_forecaster", "lstm_autoencoder", "ridge_ar"])
    ap.add_argument("--arrays", nargs="*", default=["lstm_forecaster", "lstm_autoencoder", "ridge_ar", "persistence",
                                                     "rolling_zscore", "isolation_forest", "random"])
    ap.add_argument("--out", default=str(ROOT / "reference"))
    args = ap.parse_args()

    from sqlalchemy import select

    from app.db.models import Experiment
    from app.db.session import session_scope
    from app.services.reference_service import export_bundle

    with session_scope() as db:
        ids = [e.id for e in db.scalars(
            select(Experiment).where(Experiment.origin == "reference", Experiment.status.in_(["completed", "partial"]))
            .order_by(Experiment.id)
        )]
    if not ids:
        sys.exit("No completed reference experiments found.")
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    m = export_bundle(ids, out, set(args.models), set(args.arrays))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"Exported {len(m['experiments'])} experiments to {out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
