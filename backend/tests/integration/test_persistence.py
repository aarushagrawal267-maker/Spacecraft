"""Experiments survive a backend restart; jobs cut off by a restart are marked as interrupted."""

import os

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.dataset


def test_experiments_persist_across_restart(tmp_path, monkeypatch):
    from app.dataset import store

    if not store.is_available():
        pytest.skip("dataset not installed")
    monkeypatch.setenv("HELIOGRAPH_DB_PATH", str(tmp_path / "restart.db"))
    monkeypatch.setenv("HELIOGRAPH_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("HELIOGRAPH_REFERENCE_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("HELIOGRAPH_DISABLE_JOBS", "1")
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.db.models import Experiment
    from app.db.session import reset_engine, session_scope
    from app.main import create_app

    get_settings.cache_clear()
    reset_engine()
    try:
        with TestClient(create_app()) as c:  # first "process"
            done = c.post("/api/experiments", json={"method": "persistence", "channels": ["T-9"], "name": "kept"}).json()
            running = c.post("/api/experiments", json={"method": "persistence", "channels": ["C-2"], "name": "cut off"}).json()
        with session_scope() as db:  # simulate: one finished, one was running when the backend died
            db.get(Experiment, done["id"]).status = "completed"
            db.get(Experiment, running["id"]).status = "running"
        reset_engine()

        os.environ.pop("HELIOGRAPH_DISABLE_JOBS")  # second start runs the job manager's recovery
        with TestClient(create_app()) as c:
            exps = {e["name"]: e for e in c.get("/api/experiments").json()}
            assert exps["kept"]["status"] == "completed"
            assert exps["kept"]["config"] == done["config"]
            assert exps["cut off"]["status"] == "failed"
            assert "Interrupted" in (exps["cut off"]["message"] or "")
        with session_scope() as db:
            assert db.scalar(select(Experiment).where(Experiment.name == "kept")) is not None
    finally:
        from app.services.jobs import job_manager

        job_manager.shutdown()
        reset_engine()
        get_settings.cache_clear()
