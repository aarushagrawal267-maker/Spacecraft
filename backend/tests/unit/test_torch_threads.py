"""Training workers use a small number of torch threads unless configured otherwise."""

from app.core.config import get_settings
from app.services.jobs import DEFAULT_MAX_TORCH_THREADS, default_torch_threads


def test_default_threads_are_capped():
    assert DEFAULT_MAX_TORCH_THREADS == 2
    assert default_torch_threads(1, cpu_count=16) == 2
    assert default_torch_threads(1, cpu_count=4) == 2
    assert default_torch_threads(1, cpu_count=1) == 1


def test_default_threads_split_cores_between_workers():
    assert default_torch_threads(4, cpu_count=4) == 1
    assert default_torch_threads(2, cpu_count=4) == 2
    assert default_torch_threads(3, cpu_count=2) == 1  # never below one thread
    assert default_torch_threads(0, cpu_count=4) == 2  # defensive: treated as one worker


def test_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("HELIOGRAPH_TORCH_THREADS", "3")
    get_settings.cache_clear()
    try:
        assert get_settings().torch_threads == 3
    finally:
        get_settings.cache_clear()
