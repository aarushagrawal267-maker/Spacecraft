"""Print a Markdown table of reference results straight from the database (no hand-typed numbers).

    python scripts/results_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

ORDER = ["lstm_forecaster", "lstm_autoencoder", "ridge_ar", "persistence", "rolling_zscore", "isolation_forest", "random"]


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def num(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def dur(sec):
    if not sec:
        return "< 1 s"
    if sec < 60:
        return f"{sec:.1f} s"
    if sec < 3600:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.2f} h"


def main() -> None:
    from sqlalchemy import select

    from app.db.models import Experiment
    from app.db.session import session_scope
    from app.ml.methods import SPECS

    with session_scope() as db:
        exps = list(db.scalars(select(Experiment).where(Experiment.origin == "reference",
                                                         Experiment.status.in_(["completed", "partial"]))))
        exps.sort(key=lambda e: ORDER.index(e.method) if e.method in ORDER else 99)
        print("| Method | Event P | Event R | Event F0.5 | Event TP / FP / FN | Point P | Point R | Point F1 | Point FPR | mean AP | Training time (sum over channels) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for e in exps:
            o = (e.metrics or {}).get("overall") or {}
            ev, pt = o.get("event", {}), o.get("point", {})
            label = SPECS[e.method].label + (f" ({e.profile})" if e.profile else "")
            print(f"| {label} | {pct(ev.get('precision'))} | {pct(ev.get('recall'))} | {num(ev.get('f0_5'))} | "
                  f"{ev.get('tp')} / {ev.get('fp')} / {ev.get('fn')} | {pct(pt.get('precision'))} | {pct(pt.get('recall'))} | "
                  f"{num(pt.get('f1'))} | {pct(pt.get('fpr'))} | {num(o.get('mean_average_precision'))} | "
                  f"{dur(e.train_time_s)} |")
        print()
        print("| Method | SMAP event F0.5 | MSL event F0.5 | SMAP point F1 | MSL point F1 | point-anomaly recall | contextual recall |")
        print("|---|---|---|---|---|---|---|")
        for e in exps:
            m = e.metrics or {}
            bs = m.get("by_spacecraft", {})
            ov = (m.get("overall") or {}).get("event", {}).get("by_class", {})
            label = SPECS[e.method].label
            print(f"| {label} | {num(bs.get('SMAP', {}).get('event', {}).get('f0_5'))} | "
                  f"{num(bs.get('MSL', {}).get('event', {}).get('f0_5'))} | {num(bs.get('SMAP', {}).get('point', {}).get('f1'))} | "
                  f"{num(bs.get('MSL', {}).get('point', {}).get('f1'))} | {pct(ov.get('point', {}).get('recall'))} | "
                  f"{pct(ov.get('contextual', {}).get('recall'))} |")


if __name__ == "__main__":
    main()
