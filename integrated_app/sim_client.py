"""Thin client that runs sim_runner.py in a subprocess and returns parsed results."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

RUNNER = Path(__file__).with_name("sim_runner.py")
COBUILD_DIR = Path(__file__).resolve().parents[1] / "cobuild_gymnasium_starter"
OUT_DIR = Path(__file__).with_name("outputs")


class SimError(RuntimeError):
    pass


def _run(mode: str, sim_profile: dict, out: Path, extra=(), timeout=900) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(sim_profile, fh)
        prof_path = fh.name
    try:
        # -I would drop the script dir from sys.path; sim_runner adds the Cobuild dir itself.
        proc = subprocess.run(
            [sys.executable, str(RUNNER), mode, "--profile-json", prof_path, "--out", str(out), *map(str, extra)],
            cwd=str(COBUILD_DIR), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SimError(f"Simulation timed out after {timeout}s") from exc
    finally:
        Path(prof_path).unlink(missing_ok=True)
    if proc.returncode != 0 or not out.exists():
        raise SimError((proc.stderr or proc.stdout or "unknown error").strip()[-1500:])


def run_evaluation(sim_profile: dict, runs: int = 20) -> dict:
    out = OUT_DIR / "last_eval.json"
    out.unlink(missing_ok=True)
    _run("eval", sim_profile, out, ["--runs", runs])
    return json.loads(out.read_text(encoding="utf-8"))


def render_snapshot(sim_profile: dict, at_tick: int = 60) -> Path:
    out = OUT_DIR / "compare_snapshot.png"
    out.unlink(missing_ok=True)
    _run("snapshot", sim_profile, out, ["--at-tick", at_tick])
    return out


def render_video(sim_profile: dict) -> Path:
    out = OUT_DIR / "compare_video.mp4"
    out.unlink(missing_ok=True)
    _run("video", sim_profile, out, timeout=1800)
    return out
