"""Optional OMPL backend (skipped automatically when `pip install ompl` has not been run)."""
import numpy as np
import pytest

pytest.importorskip("ompl")

import config as cfg  # noqa: E402
from collision import CollisionChecker  # noqa: E402
from environment import Environment  # noqa: E402
from ompl_backend import OMPLPlanner  # noqa: E402
from pick_place import TaskPlanner  # noqa: E402
from planner import PlanningProblem  # noqa: E402
from robot import PlanarArm  # noqa: E402


def test_ompl_rrtconnect_same_interface():
    arm, env = PlanarArm(), Environment.default()
    cc = CollisionChecker(arm, env, include_free_object=False)
    prob = PlanningProblem(cc, arm.lower, arm.upper)
    qa, qb = np.radians([122.8, 45.5, 101.7]), np.radians([49.9, -26.3, -113.7])
    res = OMPLPlanner("RRTConnect", time_limit=20).plan(qa, [qb], prob)
    assert res.success, res.message
    assert np.allclose(res.path[0], qa) and np.allclose(res.path[-1], qb)
    assert cc.path_valid(res.path)


def test_ompl_full_task():
    arm, env = PlanarArm(), Environment.default()
    plan = TaskPlanner(arm, env, rng=np.random.default_rng(0), backend="ompl").plan_pick_and_place(
        np.radians(cfg.HOME_CONFIG_DEG))
    assert plan.success, plan.message
    assert plan.metrics["planner"].startswith("OMPL")
