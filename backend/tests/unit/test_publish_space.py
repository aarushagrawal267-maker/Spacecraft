"""deploy/huggingface/publish.py stages a Space repository with the Dockerfile and card at its root."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load_publish():
    spec = importlib.util.spec_from_file_location("hf_publish", ROOT / "deploy" / "huggingface" / "publish.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage_places_dockerfile_and_space_card_at_root(tmp_path, monkeypatch):
    publish = _load_publish()
    picked = [Path("backend/requirements.txt"), Path("reference/manifest.json"), Path("README.md")]
    monkeypatch.setattr(publish, "project_files", lambda: picked)

    n = publish.stage(tmp_path)

    assert n == len(picked)
    assert (tmp_path / "backend" / "requirements.txt").is_file()
    assert (tmp_path / "reference" / "manifest.json").is_file()
    assert (tmp_path / "Dockerfile").read_text() == (ROOT / "deploy/huggingface/Dockerfile").read_text()
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---") and "sdk: docker" in card and "app_port: 7860" in card


def test_project_files_excludes_local_environments_and_data():
    publish = _load_publish()
    files = publish.project_files()
    assert files, "expected at least the tracked project files"
    for f in files:
        assert not set(f.parts) & {".venv", "node_modules", "data", "artifacts", ".git"}, f


def test_credential_files_are_never_staged(tmp_path, monkeypatch):
    publish = _load_publish()
    for name in (".env", "backend/.env.local", "kaggle.json", "deploy/space.pem", "README.md", "backend/app/main.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    monkeypatch.setattr(publish, "ROOT", tmp_path)
    # Without git (e.g. a downloaded zip) the fallback walk is used.
    monkeypatch.setattr(publish.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))

    picked = {p.as_posix() for p in publish.project_files()}

    assert picked == {"README.md", "backend/app/main.py"}
    assert publish._is_secret_file(Path(".env.production")) and not publish._is_secret_file(Path("env.py"))
