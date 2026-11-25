"""FastAPI application. API docs: http://localhost:8000/docs"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import analysis, dataset, experiments, system
from app.core.config import get_settings
from app.dataset import store
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
    # When FastAPI also serves the UI (single-container deployments), the UI owns /docs
    # (the "Research & Docs" page), so the interactive API documentation moves under /api.
    api_docs_prefix = "/api" if _serve_frontend(settings) else ""
    app = FastAPI(
        title="Heliograph API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url=f"{api_docs_prefix}/docs",
        redoc_url=f"{api_docs_prefix}/redoc",
        openapi_url=f"{api_docs_prefix}/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"],
    )
    for r in (system.router, dataset.router, experiments.router, analysis.router):
        app.include_router(r, prefix="/api")

    @app.exception_handler(store.DatasetNotAvailable)
    async def _dataset_missing(_request: Request, exc: store.DatasetNotAvailable) -> JSONResponse:
        # Any endpoint that needs telemetry explains how to install it instead of a bare 500.
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    _mount_frontend(app, settings)
    return app


def _serve_frontend(settings) -> bool:
    dist = settings.project_root / "frontend" / "dist"
    return os.environ.get("HELIOGRAPH_SERVE_FRONTEND") == "1" and (dist / "index.html").exists()


def _mount_frontend(app: FastAPI, settings) -> None:
    """Serve the built UI from the same origin when frontend/dist exists (single-container deployments)."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    dist = settings.project_root / "frontend" / "dist"
    if not _serve_frontend(settings):
        return
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = (dist / path).resolve()
        if path and f.is_file() and dist.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(dist / "index.html")  # client-side routes


app = create_app()
