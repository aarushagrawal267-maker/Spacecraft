"""FastAPI application. API docs: http://localhost:8000/docs"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import analysis, dataset, experiments, system
from app.core.config import get_settings
from app.db.session import migrate, session_scope

log = logging.getLogger("heliograph")

DESCRIPTION = """
**Heliograph** — spacecraft telemetry anomaly detection on the NASA SMAP / MSL benchmark.

Independent research implementation inspired by Hundman et al., *Detecting Spacecraft Anomalies
Using LSTMs and Nonparametric Dynamic Thresholding* (KDD 2018). Not affiliated with NASA/JPL.
All figures returned by this API come from the installed dataset and stored experiment results.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    try:
        import torch

        torch.set_num_threads(max(1, settings.api_torch_threads))
    except Exception:  # noqa: BLE001
        log.exception("Could not set torch threads")
    migrate()
    from app.services import dataset_service, reference_service

    with session_scope() as db:
        try:
            dataset_service.sync_from_disk(db)
        except Exception:  # noqa: BLE001
            log.exception("Dataset sync failed")
    try:
        reference_service.import_if_needed()
    except Exception:  # noqa: BLE001
        log.exception("Reference import failed")
    if os.environ.get("HELIOGRAPH_DISABLE_JOBS") != "1":
        from app.services.jobs import job_manager

        job_manager.start()
    yield
    if os.environ.get("HELIOGRAPH_DISABLE_JOBS") != "1":
        from app.services.jobs import job_manager

        job_manager.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Heliograph API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"],
    )
    for r in (system.router, dataset.router, experiments.router, analysis.router):
        app.include_router(r, prefix="/api")
    return app


app = create_app()
