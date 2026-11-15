"""Download, verify and install the SMAP/MSL benchmark into data/raw/smap_msl.

    python scripts/acquire_data.py                 # auto: Kaggle -> GitHub mirror -> data/incoming/data.zip
    python scripts/acquire_data.py --source kaggle
    python scripts/acquire_data.py --force         # re-download even if a valid copy exists
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto", choices=["auto", "kaggle", "github_mirror", "manual"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from app.dataset import store
    from app.dataset.acquire import acquire

    if store.is_available() and not args.force:
        print("Dataset already installed at", store.dataset_root(), "(use --force to re-download).")
        return 0

    last = [0.0]

    def progress(msg: str, frac: float) -> None:
        if time.time() - last[0] > 3 or frac >= 0.9 or "fail" in msg:
            print(f"[{frac * 100:5.1f}%] {msg}", flush=True)
            last[0] = time.time()

    res = acquire(args.source, progress)
    for a in res.attempts:
        print(f"  {'OK ' if a['ok'] else 'FAIL'} {a['source']}: {a['message']}")
    if not res.ok:
        print("\nAcquisition failed. Alternative: download the dataset zip manually from")
        print("  https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
        print("save it as data/incoming/data.zip and run:  python scripts/acquire_data.py --source manual")
        return 1
    s = res.report.summary
    print(f"\nInstalled from {res.source}: {s['unique_channels']} channels, {s['label_sequences_after_merge']} label "
          f"sequences ({s['label_rows']} label rows). {len(res.report.warnings)} validation warnings, 0 errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
