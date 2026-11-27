"""paths.py - the one place that makes the roboarm/ modules importable.

roboarm/ is not a package: its modules import each other as top-level names
(`import config`, `from planner import ...`). Importing this module adds the
roboarm/ folder to sys.path so we can reuse that code without copying it:

    from roboarm_mujoco import paths  # noqa: F401  (side effect: roboarm on sys.path)
    import config as rcfg
    from planner import RRTConnect
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOARM_DIR = REPO_ROOT / "roboarm"


def add_roboarm_to_path():
    path = str(ROBOARM_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


add_roboarm_to_path()
