"""Where the SMAP/MSL benchmark can be obtained from.

The benchmark accompanies Hundman et al., "Detecting Spacecraft Anomalies Using
LSTMs and Nonparametric Dynamic Thresholding" (KDD 2018). The original S3 link
in the Telemanom README no longer works (HTTP 403 as of 2026-11), so the
acquisition pipeline tries, in order:

1. ``kaggle``  – the Kaggle dataset that the Telemanom README now points to.
2. ``github_mirror`` – the copy of the original Telemanom ``.npy`` files kept in
   the public TranAD research repository (pinned commit). Its
   ``labeled_anomalies.csv`` is byte-identical to the official Telemanom file.
3. ``manual`` – a ``data.zip`` the user places in ``data/incoming/``.

The anomaly labels always come from the official Telemanom repository (pinned
commit), regardless of which source supplied the telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass

TELEMANOM_COMMIT = "2e6c5b6c3558e7835601519b7bdef37c649bdbdc"
LABELS_URL = (
    f"https://raw.githubusercontent.com/khundman/telemanom/{TELEMANOM_COMMIT}/labeled_anomalies.csv"
)
# SHA-256 of labeled_anomalies.csv at TELEMANOM_COMMIT (verified 2026-11-14).
LABELS_SHA256 = "057ce2d6c8875982bf4e5404aefea14efdcbce413d80826d2b737c95b59b7539"

KAGGLE_DATASET = "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
KAGGLE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"

MIRROR_REPO = "imperial-qore/TranAD"
MIRROR_COMMIT = "7ffb98d0c18189cc3d9ab732b4cb0278200a0af0"
MIRROR_BASE = f"https://raw.githubusercontent.com/{MIRROR_REPO}/{MIRROR_COMMIT}/data/SMAP_MSL"


@dataclass(frozen=True)
class SourceInfo:
    key: str
    title: str
    url: str
    note: str


SOURCES: dict[str, SourceInfo] = {
    "kaggle": SourceInfo(
        key="kaggle",
        title="Kaggle: NASA Anomaly Detection Dataset SMAP & MSL",
        url=f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
        note="Download source named in the Telemanom README (v2).",
    ),
    "github_mirror": SourceInfo(
        key="github_mirror",
        title="GitHub mirror of the Telemanom data (TranAD repository, pinned commit)",
        url=f"https://github.com/{MIRROR_REPO}/tree/{MIRROR_COMMIT}/data/SMAP_MSL",
        note=(
            "Third-party redistribution of the original Telemanom .npy files. Its "
            "labeled_anomalies.csv is byte-identical (SHA-256) to the official file; every "
            "telemetry file is validated against the official label file after download."
        ),
    ),
    "manual": SourceInfo(
        key="manual",
        title="Manually supplied data.zip",
        url="data/incoming/",
        note="User-provided archive; validated with the same checks as downloaded data.",
    ),
}

LICENSE_NOTE = (
    "Telemanom (code, labels and the redistributed benchmark): the README states Apache 2.0; "
    "the repository's LICENSE.txt is a Caltech/JPL BSD-style copyright assertion "
    "(Copyright (c) 2018, California Institute of Technology). The mirror repository "
    "(TranAD) is BSD 3-Clause. Kaggle redistribution terms are shown on the Kaggle page. "
    "This project is an independent research implementation and is not affiliated with "
    "or endorsed by NASA, JPL or Caltech."
)
