"""Benchmark script: one run plans, executes and writes a CSV row."""
import csv

import pytest

from roboarm_mujoco import benchmark


def test_one_run_writes_csv(tmp_path):
    out = tmp_path / "bench.csv"
    rows = benchmark.main(["--runs", "1", "--seed", "0", "--out", str(out)])
    r = rows[0]
    assert r["plan_success"] and r["task_success"], r["message"]
    assert r["collision_steps"] == 0
    assert r["place_error_m"] < benchmark.PLACE_TOL
    with open(out, newline="") as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == 1 and list(saved[0]) == benchmark.FIELDS
    with pytest.raises(SystemExit):                       # never overwrites an existing results file
        benchmark.main(["--runs", "1", "--out", str(out)])
