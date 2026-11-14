"""Integration fixtures: a FastAPI app on a throwaway database + artifacts folder.

These tests need the real benchmark in data/raw/smap_msl (they are skipped
otherwise). They never touch the user's own database.
"""

import os
import time

import pytest


def _dataset_present() -> bool:
    from app.dataset import store

    return store.is_available()


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    if not _dataset_present():
        pytest.skip("SMAP/MSL dataset not installed (run setup.ps1 -DownloadData)")
    tmp = tmp_path_factory.mktemp("heliograph")
    os.environ["HELIOGRAPH_DB_PATH"] = str(tmp / "test.db")
    os.environ["HELIOGRAPH_ARTIFACTS_DIR"] = str(tmp / "artifacts")
    os.environ["HELIOGRAPH_REFERENCE_DIR"] = str(tmp / "no_reference")
    from app.core.config import get_settings
    from app.db.session import reset_engine

    get_settings.cache_clear()
    reset_engine()
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    reset_engine()
    get_settings.cache_clear()


def wait_experiment(client, exp_id: int, timeout: float = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = client.get(f"/api/experiments/{exp_id}").json()
        if d["status"] not in ("queued", "running"):
            return d
        time.sleep(0.5)
    raise TimeoutError(f"experiment {exp_id} still {d['status']}")
