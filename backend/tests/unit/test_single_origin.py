"""Single-container mode (HELIOGRAPH_SERVE_FRONTEND=1): FastAPI serves the built UI.

The UI has its own client-side route ``/docs`` (Research & Docs), so in this mode the
interactive API documentation must live under ``/api/docs`` instead, otherwise a page
refresh on the Research page would show Swagger UI.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def single_origin_app(tmp_path, monkeypatch):
    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Heliograph UI</title><div id=root></div>")
    (dist / "assets" / "app.js").write_text("console.log('ui')")
    monkeypatch.setenv("HELIOGRAPH_SERVE_FRONTEND", "1")
    monkeypatch.setenv("HELIOGRAPH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("HELIOGRAPH_DB_PATH", str(tmp_path / "t.db"))

    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    try:
        # No context manager: the lifespan (dataset sync, job workers) is not needed here.
        yield TestClient(create_app())
    finally:
        get_settings.cache_clear()


def test_ui_routes_serve_index(single_origin_app):
    c = single_origin_app
    for path in ("/", "/docs", "/detect", "/experiments/3"):
        r = c.get(path)
        assert r.status_code == 200, path
        assert "Heliograph UI" in r.text, path
    assert c.get("/assets/app.js").text == "console.log('ui')"


def test_api_docs_move_under_api(single_origin_app):
    c = single_origin_app
    assert "swagger" in c.get("/api/docs").text.lower()
    assert c.get("/api/openapi.json").json()["info"]["title"] == "Heliograph API"
    assert c.get("/api/does-not-exist").status_code == 404


def test_default_mode_keeps_docs_at_root(monkeypatch):
    monkeypatch.delenv("HELIOGRAPH_SERVE_FRONTEND", raising=False)
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    c = TestClient(create_app())
    assert "swagger" in c.get("/docs").text.lower()
    assert c.get("/openapi.json").status_code == 200
