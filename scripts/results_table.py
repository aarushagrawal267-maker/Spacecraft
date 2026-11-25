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
        print_bootstrap(exps)


def print_bootstrap(exps) -> None:
    """Channel-bootstrap 95 % intervals for event F0.5, paired against the first method in ORDER."""
    import numpy as np

    from app.ml.evaluation.metrics import bootstrap_event_f_beta
    from app.ml.methods import SPECS

    per = {}
    for e in exps:
        per[e.method] = {r.chan_id: r.metrics["event"] for r in e.runs if r.status == "completed" and r.metrics}
    if not per:
        return
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    counts = {m: np.array([[v[c]["tp"], v[c]["fp"], v[c]["fn"]] for c in common]) for m, v in per.items()}
    ref = exps[0].method
    b = bootstrap_event_f_beta(counts, reference=ref, n_boot=10_000, seed=0)
    print()
    print(f"Event F0.5 with 95 % channel-bootstrap intervals ({b['n_channels']} channels resampled with replacement, "
          f"{b['n_boot']:,} draws, seed {b['seed']}; differences are paired against {SPECS[ref].label}):")
    print()
    print(f"| Method | Event F0.5 [95 % CI] | {SPECS[ref].label} minus method [95 % CI] |")
    print("|---|---|---|")
    for e in exps:
        r = b["methods"][e.method]
        d = r.get("diff_vs_reference")
        diff = "—" if d is None else f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
        print(f"| {SPECS[e.method].label} | {r['estimate']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] | {diff} |")


if __name__ == "__main__":
    main()
