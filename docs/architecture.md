# Architecture

```
            Browser  ──►  http://localhost:5173   (Vite preview / dev server, React UI)
                               │  /api/*  proxied
                               ▼
                         http://localhost:8000   FastAPI (uvicorn, single worker)
                               │
     ┌─────────────────────────┼───────────────────────────────────────┐
     │ api/       routers: system, dataset, experiments, analysis      │
     │ services/  dataset · jobs · worker · inference · replay ·       │
     │            evaluation · experiment · reference                  │
     │ ml/        methods · models (LSTM, AE, baselines) · training ·  │
     │            scoring (EWMA, NDT, static) · evaluation · pipeline  │
     │ dataset/   sources · acquire · validate · labels · store        │
     │ db/        SQLAlchemy models · session (WAL) · Alembic          │
     └─────────────────────────┬───────────────────────────────────────┘
          spawn                │ multiprocessing queue (progress, results)
   ┌───────────────┐           │
   │ worker procs  │ ──────────┘   only the API process writes to SQLite
   │ fit/score/eval│
   └───────────────┘
  data/raw/smap_msl (read-only .npy) · data/heliograph.db · artifacts/ (checkpoints, arrays) · reference/
```

## Storage

* **SQLite** (`data/heliograph.db`) via SQLAlchemy 2 + Alembic migrations. WAL journal mode,
  `busy_timeout` 30 s, foreign keys on. Tables: `dataset_snapshots`, `channels`, `label_intervals`,
  `experiments` (also the job record), `channel_runs`, `epoch_logs`, `detections`.
* **Files**: model checkpoints (`.pt` for torch, `.pkl` for scikit-learn models written by this app) and
  compressed per-timestep arrays (`.npz`: prediction, raw score, smoothed score, thresholds, flags) under
  `artifacts/experiments/exp_NNNNN/`. Paths are stored in `channel_runs`; reference bundle paths are
  project-relative so the bundle works wherever the project lives.

## Jobs

`POST /api/experiments` queues an experiment. A scheduler thread starts at most
`max_concurrent_jobs` (default 1) experiments, each in 1…N worker processes (`spawn` context, safe on
Windows). Workers load data, fit, score, threshold and evaluate each channel, and push messages
(`run_started`, `epoch`, `run_done`, `run_failed`, `worker_done`) to a queue. A consumer thread in the
API process writes them to SQLite, so there is a single writer. Cancellation sets an event checked
after every mini-batch; stragglers are terminated after 20 s. On restart, experiments left in
`running` are marked as interrupted. Progress reaches the browser via Server-Sent Events
(`GET /api/experiments/{id}/events`).

## API overview (full, interactive docs at http://localhost:8000/docs)

| group | endpoints |
|---|---|
| system | `GET /api/health`, `/api/system`, `/api/methods`, `/api/overview` |
| dataset | `GET /api/dataset/status`, `POST /api/dataset/acquire`, `POST /api/dataset/validate`, `GET /api/dataset/report`, `GET /api/channels`, `/api/channels/{id}`, `/api/telemetry/{id}?split=&start=&end=&max_points=` |
| experiments | `POST/GET /api/experiments`, `GET/DELETE /api/experiments/{id}`, `POST /api/experiments/{id}/cancel`, `GET /api/experiments/{id}/events` (SSE) |
| runs | `GET /api/runs/{id}`, `GET /api/runs/{id}/series`, `POST /api/runs/{id}/infer` (executes the stored model now, optional threshold overrides) |
| detections | `GET /api/detections?experiment_id=&chan_id=&outcome=`, `GET /api/detections/{id}?context=` |
| evaluation | `GET /api/evaluation/compare?ids=1,2,3` |
| replay | `POST /api/replay`, `POST /api/replay/{sid}/advance`, `DELETE /api/replay/{sid}` |
| export | `GET /api/export/experiments/{id}.json`, `/channels.csv`, `/detections.csv` |

## Frontend

React 18 + TypeScript + Vite 6 + Tailwind CSS 4 + Plotly.js (one charting library for consistent
styling and WebGL lines for 8,000-point streams) + TanStack Query + React Router. Recharts from the
original suggestion was not used, to keep a single chart system. API types are hand-written in
`src/api/types.ts` to mirror the FastAPI responses.

The signature component is the **Trace Stack** (`src/components/TraceStack.tsx`): telemetry,
score/threshold and an annotation lane on one shared index axis. Data colours were validated for
colour-vision deficiency against the panel surface (`src/lib/palette.ts`).
