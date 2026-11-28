# Dataset documentation

## What it is

The **NASA SMAP / MSL telemetry anomaly benchmark** released with

> K. Hundman, V. Constantinou, C. Laporte, I. Colwell, T. Soderstrom.
> *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD 2018.
> arXiv:1802.04431 · NASA NTRS 20210008188 · code: https://github.com/khundman/telemanom

It contains expert-labelled telemetry from two missions:

| | SMAP (Soil Moisture Active Passive satellite) | MSL (Mars Science Laboratory, Curiosity rover) |
|---|---|---|
| Label-file rows | 55 | 27 |
| Unique channel IDs | **54** (`P-2` is listed twice) | 27 |
| Labelled sequences (raw) | 69 (43 point, 26 contextual) | 36 (19 point, 17 contextual) |
| Labelled sequences after merging `P-2` | 68 | 36 |
| Features per timestep | 25 | 55 |
| Training values (sum of lengths) | 138,004 | 58,317 |
| Test values (unique channels) | 435,826 | 73,729 |
| Test length range | 4,453 – 8,640 | 1,096 – 6,100 |
| Shortest training split | 312 (`D-12`) | 439 (`T-9`) |

All figures above were measured by this project's validator on the files actually obtained
(see `data/raw/smap_msl/validation_report.json` after installation). They agree with the
Telemanom README: summing *test length − 260* over the 82 label rows gives exactly the README's
"telemetry values evaluated" — 429,735 (SMAP) and 66,709 (MSL).

## File layout

```
data/raw/smap_msl/            (read-only after installation)
├── train/<chan_id>.npy       float64, shape (timesteps, features)
├── test/<chan_id>.npy        float64, shape (timesteps, features)
├── labeled_anomalies.csv     official Telemanom label file (pinned commit, SHA-256 verified)
├── manifest.json             SHA-256 + size of every file, combined dataset checksum
├── provenance.json           source, URL, retrieval time, commits, licence note
└── validation_report.json    full structural / quality report
```

* **Column 0** is the telemetry value (the quantity the models forecast).
* **Columns 1…** are one-hot encoded command information for spacecraft modules.

### `labeled_anomalies.csv`

| column | meaning |
|---|---|
| `chan_id` | anonymised channel ID; only the first letter is documented as a channel type (e.g. `P` = power, `R` = radiation) |
| `spacecraft` | `SMAP` or `MSL` |
| `anomaly_sequences` | list of inclusive `[start, end]` index pairs into the **test** sequence |
| `class` | one class per sequence: `point` or `contextual` |
| `num_values` | length of the test sequence |

## Integrity disclosures (also shown in the app)

* Channel identifiers are anonymised and are **not** real instrument names.
* Time is anonymised: positions only, no timestamps, no engineering units.
* Values are pre-processed. The Telemanom README says telemetry was scaled to (−1, 1) using the
  **test-set** min/max. In the obtained files, 16 channels exceed that range by more than 1e-4
  (e.g. `M-6` reaches 258.1, `F-5` 6.95), and 16 channels have a **constant** training value:
  14 at −1 (`B-1`, `C-2`, `D-2`, `D-7`, `D-8`, `D-9`, `D-12`, `D-13`, `D-14`, `G-2`, `M-6`, `P-4`,
  `S-2`, `T-5`) and 2 at 0.999 (`A-1`, `R-1`).
* Anomaly intervals are benchmark reference annotations derived from expert review, not confirmed
  hardware failures; unlabelled regions are not guaranteed normal.
* No missing (non-finite) values occur in any file.

## Duplicate channel `P-2`

`P-2` appears in two label rows with overlapping intervals `[5350, 6575]` and `[5300, 6420]`
(identical `num_values`). This project treats a channel as one stream and **merges** the rows into
the single interval `[5300, 6575]`; the merge is stored with the interval (`merged_from`) and shown
in the UI. Telemanom itself evaluates the channel twice.

## Acquisition

The original S3 link in the Telemanom README returns HTTP 403 (checked November 2026). The pipeline
(`backend/app/dataset/acquire.py`, CLI `scripts/acquire_data.py`) tries in order:

1. **Kaggle** — `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`, the source the Telemanom
   README now names. Optional credentials: `KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json`.
2. **GitHub mirror** — the original Telemanom `.npy` files kept in the public TranAD research
   repository (`imperial-qore/TranAD`, pinned commit `7ffb98d0`). Its `labeled_anomalies.csv` is
   byte-identical to the official one (SHA-256 `057ce2d6…`).
3. **Manual** — a `data.zip` placed in `data/incoming/` (or an extracted folder containing
   `train/` and `test/`).

The label file is always downloaded from the official Telemanom repository at commit `2e6c5b6c` and
its SHA-256 is checked. Then every telemetry file is validated against it: channel inventory (no
missing files), 2-D arrays, matching feature counts, test length = `num_values`, finite values,
value ranges, binary command columns, constant channels. Only a copy that passes is installed
(atomic swap of a staging directory), made read-only, and recorded with a SHA-256 manifest.
Archives are extracted with path-traversal protection and arrays are loaded with
`allow_pickle=False`.

The reference results in this repository were computed on data obtained from the **GitHub mirror**.

## Licensing

Telemanom's README states Apache 2.0; its `LICENSE.txt` is a Caltech/JPL BSD-style copyright
assertion (Copyright © 2018 California Institute of Technology). The TranAD mirror is BSD 3-Clause.
The telemetry is **not** committed to this repository; it is downloaded by `setup.ps1 -DownloadData`.
