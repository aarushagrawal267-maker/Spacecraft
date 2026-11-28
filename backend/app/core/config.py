"""Application settings.

All paths are resolved relative to the project root so the app behaves the same
whether it is started from ``start.ps1`` on Windows or from a shell on Linux.
Every setting can be overridden with an environment variable prefixed
``HELIOGRAPH_`` (for example ``HELIOGRAPH_DEVICE=cpu``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> parents[3] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HELIOGRAPH_", env_file=None, extra="ignore")

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    reference_dir: Path = PROJECT_ROOT / "reference"
    db_path: Path | None = None  # defaults to data_dir / "heliograph.db"

    # "auto" uses CUDA when torch reports it available, otherwise CPU.
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # CPU threads torch may use inside each training worker
    # (None = min(2, CPU cores / workers), see services.jobs.default_torch_threads).
    torch_threads: int | None = None
    # Threads PyTorch may use inside the API process (live inference / replay). One thread
    # avoids severe oversubscription when training workers are busy on the same CPU.
    api_torch_threads: int = 1
    # Only one training job runs at a time by default (laptop-friendly).
    max_concurrent_jobs: int = 1

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw" / "smap_msl"

    @property
    def incoming_dir(self) -> Path:
        return self.data_dir / "incoming"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def database_path(self) -> Path:
        return self.db_path or (self.data_dir / "heliograph.db")

    @property
    def database_url(self) -> str:
        # as_posix() keeps the URL valid on Windows (sqlite:///C:/...).
        return f"sqlite:///{self.database_path.resolve().as_posix()}"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.artifacts_dir, self.incoming_dir, self.processed_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_path(p: str | Path | None) -> Path | None:
    """Stored artifact paths may be absolute or relative to the project root
    (the committed reference bundle uses relative paths so it is portable)."""
    if p is None or p == "":
        return None
    path = Path(p)
    return path if path.is_absolute() else get_settings().project_root / path
