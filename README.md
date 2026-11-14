# Heliograph — Spacecraft Telemetry Anomaly Detection

Heliograph is a locally runnable research platform for **anomaly detection in real spacecraft telemetry**
from NASA's **SMAP** satellite and the **Mars Science Laboratory (Curiosity)** rover. It trains per-channel
**LSTM forecasters**, scores their prediction errors, applies **nonparametric dynamic thresholding**, and lets
you inspect every detected interval against the benchmark's **expert-labelled anomalies**. It also provides
an experimental LSTM autoencoder, five baselines, a historical telemetry replay, experiment tracking and a
full evaluation suite.

> **Independent research implementation.** Inspired by Hundman et al., *Detecting Spacecraft Anomalies
> Using LSTMs and Nonparametric Dynamic Thresholding* (KDD 2018). Not affiliated with, commissioned by,
> or used by NASA/JPL. A detected anomaly is an unusual pattern that warrants investigation. It is not
> proof of a hardware failure.

![Detection Centre](docs/img/detection-centre.png)

---

## Contents

1. [Quick start (Windows)](#quick-start-windows)
2. [What you can do](#what-you-can-do)
3. [Reference results](#reference-results)
4. [The data](#the-data)
5. [How it works](#how-it-works)
6. [Project structure](#project-structure)
7. [Testing](#testing)
8. [Reproducing the experiments](#reproducing-the-experiments)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [Acceptance status](#acceptance-status)
12. [Limitations, references and licence](#limitations-references-and-licence)

---

## Quick start (Windows)

**Prerequisites**

* Windows 10 or 11
* **Python 3.12** (3.11–3.13, 64-bit) from <https://www.python.org/downloads/>, with "Add python.exe to PATH" ticked
* **Node.js 18+ LTS** from <https://nodejs.org/>
* About 4 GB of free disk space (PyTorch is the largest dependency) and an internet connection for setup

Open **PowerShell** in the project folder. If scripts are blocked, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**1. Set up** (creates `backend\.venv`, installs dependencies, builds the UI, downloads and verifies the NASA dataset):

```powershell
.\setup.ps1 -DownloadData
```

If you have an NVIDIA GPU and want CUDA acceleration, run this instead:

```powershell
.\setup.ps1 -DownloadData -Cuda
```

**2. Start** (opens the browser):

```powershell
.\start.ps1
```

| What | Address |
|---|---|
| Web application | <http://localhost:5173> |
| Backend API | <http://localhost:8000> |
| Interactive API documentation | <http://localhost:8000/docs> |

**3. Stop:**

```powershell
.\stop.ps1
```

Nothing trains automatically. On first start, the database imports the **reference results** shipped in
`reference/`, so every page shows genuine results right away (marked **Reference**). Your own runs are
marked **Local**. CPU works everywhere. CUDA is used only if it is available, and you can force CPU with
`.\start.ps1 -Device cpu`.

---

## What you can do

| Page | Purpose |
|---|---|
| **Mission Overview** | SMAP and Curiosity summaries, dataset status, analysed channels, detected intervals, latest result per method, and a per-channel map of detections vs labels |
| **Telemetry Explorer** | Browse training and test telemetry for any channel, zoom, overlay expert labels, and compare value distributions inside vs outside labelled regions |
| **Detection Centre** | Actual vs predicted telemetry, raw and smoothed score, dynamic threshold, detected intervals and expert labels on one shared axis (the Trace Stack); re-run the stored model live with different threshold settings |
| **Telemetry Replay** | Historical replay that reveals a test stream point by point while the model runs on each new point; start, pause, resume, reset and speed controls. **Not a live spacecraft link.** |
| **Investigation** | Every detected interval ranked by score: indices, values, prediction, score, threshold window, matching label and model |
| **Model Laboratory** | Configure and train any method on chosen channels (Quick / Standard / Paper profiles), watch live training and validation loss, cancel jobs |
| **Model Evaluation** | Event-wise and point-wise metrics, PR and ROC curves, confusion matrix, SMAP vs MSL, detection delay and IoU, per-channel table, timings |
| **Experiment History** | All runs with configuration, seed, code commit, dataset checksum and metrics; compare, export (JSON/CSV) and delete |
| **Dataset Explorer** | Provenance, checksums, validation report, SMAP and MSL breakdown, split lengths, label shares, channel inventory, data-quality warnings |
| **Research & Docs** | Background, LSTM explanation, dataset documentation, architecture, thresholding, evaluation, limitations, reproducibility and references |

![Mission Overview](docs/img/overview.png)

---

## Reference results

<!-- RESULTS:START -->
_Results are generated from the database by `scripts/results_table.py` after the reference run._
<!-- RESULTS:END -->

---

## The data

The **NASA SMAP / MSL telemetry anomaly benchmark** (Telemanom):

* 81 unique channels: 54 SMAP and 27 MSL. The label file has 82 rows because `P-2` is listed twice; this project merges the two rows.
* 105 labelled anomaly sequences in the file (104 after merging `P-2`): 62 point and 43 contextual.
* 25 input features per timestep for SMAP and 55 for MSL. Column 0 is telemetry; the rest are one-hot command inputs.
* Anonymised channel IDs, anonymised time, pre-processed values. There are no timestamps or engineering units.

Acquisition tries, in order: **Kaggle** (the source named in the Telemanom README), a **pinned GitHub copy**
of the original Telemanom files, and a manually supplied `data.zip`. The label file always comes from the
official Telemanom repository and is checked by SHA-256. Every file is validated against it before
installation. Full details, measured statistics and the integrity disclosures are in
[docs/dataset.md](docs/dataset.md).

---

## How it works

```
train split ──► fit model (LSTM forecaster | LSTM autoencoder | baselines) ──► saved checkpoint
test split  ──► causal raw scores (forecast / reconstruction error) ──► EWMA smoothing
            ──► nonparametric dynamic thresholding (windows of 30×70, z-search, inverse pass, pruning)
            ──► detected intervals ──► compared with expert labels (event-wise, point-wise, delay, IoU)
```

* Models never see test data or labels during training, and thresholds use no labels. Automated tests enforce both.
* Every method is scored on test indices ≥ 250 and goes through the same smoothing, thresholding and evaluation.
* Event-wise and point-wise metrics are always reported separately, with no point adjustment.
* Deviations from the original Telemanom code are deliberate and documented.

Read more: [docs/methodology.md](docs/methodology.md) · [docs/architecture.md](docs/architecture.md) ·
[docs/limitations.md](docs/limitations.md)

**Stack:** FastAPI · Pydantic · SQLAlchemy + Alembic · SQLite (WAL) · PyTorch · NumPy · pandas · SciPy ·
scikit-learn · React 18 · TypeScript · Vite · Tailwind CSS 4 · Plotly.js · TanStack Query · Pytest ·
Vitest · Playwright.

---

## Project structure

```
Spacecraft/
├── setup.ps1 / start.ps1 / stop.ps1     Windows setup, start and stop
├── backend/
│   ├── app/
│   │   ├── api/          FastAPI routers (system, dataset, experiments, analysis)
│   │   ├── services/     jobs (worker processes), inference, replay, evaluation, experiments, reference bundle
│   │   ├── ml/           models (LSTM, autoencoder, baselines), training, scoring (EWMA, NDT), evaluation, pipeline
│   │   ├── dataset/      sources, acquisition, validation, label parsing, data access
│   │   ├── db/           SQLAlchemy models and session (SQLite, WAL)
│   │   └── core/         settings
│   ├── alembic/          database migrations
│   └── tests/            unit + integration tests (pytest)
├── frontend/
│   ├── src/pages/        the ten application pages
│   ├── src/components/   Trace Stack, Plot wrapper, UI kit, pickers
│   ├── src/test/         Vitest component tests
│   └── e2e/              Playwright end-to-end workflows
├── scripts/              acquire_data, init_db, run_reference, export_reference, results_table
├── reference/            committed reference results (metrics, detections, checkpoints, arrays)
├── docs/                 dataset, methodology, architecture, limitations, references
├── data/                 (not in git) raw benchmark, SQLite database
└── artifacts/            (not in git) models and arrays from your local experiments
```

---

## Testing

Run these from the project root after setup. Backend unit and integration tests (the integration tests use
the real dataset and a throwaway database):

```powershell
cd backend; .\.venv\Scripts\python -m pytest; cd ..
```

Frontend component tests:

```powershell
cd frontend; npm test; cd ..
```

End-to-end browser workflows (start an isolated backend on :8001 and UI on :5174, train a small real model,
then test inference, replay, history and evaluation):

```powershell
cd frontend; npx playwright install chromium; npm run e2e -- --trace off; cd ..
```

---

## Reproducing the experiments

```powershell
backend\.venv\Scripts\python scripts\run_reference.py --workers 2
backend\.venv\Scripts\python scripts\results_table.py
```

`run_reference.py` queues the same seven experiments as the shipped reference run (all 81 channels) through
the application's own job system. Every experiment stores its configuration, seed, code commit, dataset
checksum, device and timings. The LSTM steps take hours on a laptop CPU. Exact numbers can differ slightly
across platforms and PyTorch builds.

---

## Configuration

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `HELIOGRAPH_DEVICE` | `auto` | `auto`, `cpu` or `cuda` (also `start.ps1 -Device`) |
| `HELIOGRAPH_TORCH_THREADS` | torch default | CPU threads per training process |
| `HELIOGRAPH_MAX_CONCURRENT_JOBS` | `1` | experiments running at the same time |
| `HELIOGRAPH_DATA_DIR` / `HELIOGRAPH_DB_PATH` / `HELIOGRAPH_ARTIFACTS_DIR` | project folders | storage locations |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | — | optional Kaggle credentials for the download |

For low-memory laptops, use the **Quick** profile and a few channels per experiment, with one worker process.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "running scripts is disabled" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| "Port 8000/5173 is already in use" | Run `.\stop.ps1`, or close the other program using the port |
| Kaggle download fails | Setup falls back to the GitHub copy automatically. Or download the zip from the Kaggle page, save it as `data\incoming\data.zip`, then run `backend\.venv\Scripts\python scripts\acquire_data.py --source manual` |
| UI shows "Backend unreachable" | Check `logs\backend.err.log`, then restart with `.\stop.ps1` and `.\start.ps1` |
| Project in OneDrive | Move it outside OneDrive; syncing can lock the database |
| CUDA not used | Run `.\setup.ps1 -Cuda` (needs an NVIDIA driver); check the CPU/CUDA badge in the top bar |

---

## Acceptance status

<!-- ACCEPTANCE:START -->
_Filled in at delivery._
<!-- ACCEPTANCE:END -->

---

## Limitations, references and licence

* Limitations and interpretation: [docs/limitations.md](docs/limitations.md)
* References: [docs/references.md](docs/references.md)
* Dataset licensing: Telemanom (Caltech/JPL copyright; README states Apache 2.0); the mirror used is BSD 3-Clause.
  The telemetry is not stored in this repository; it is downloaded during setup.
