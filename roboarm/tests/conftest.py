import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _metrics_log_in_tmp(tmp_path, monkeypatch):
    """Tests must not write into the user's results/metrics_log.csv."""
    import metrics
    monkeypatch.setattr(metrics, "LOG_PATH", str(tmp_path / "metrics_log.csv"))
