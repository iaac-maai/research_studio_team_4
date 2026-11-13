"""Stage C: RRT-Connect between two joint configurations."""
import numpy as np

from collision import CollisionChecker
from environment import Environment
from planner import PlanningProblem, RRTConnect, path_length, shortcut_path
from robot import PlanarArm

Q_PICK_A = np.radians([122.8, 45.5, 101.7])     # above A, arm threaded through the left window
Q_PLACE_B = np.radians([49.9, -26.3, -113.7])   # above B


def problem():
    arm, env = PlanarArm(), Environment.default()
    cc = CollisionChecker(arm, env, include_free_object=False)
    return arm, cc, PlanningProblem(cc, arm.lower, arm.upper)


def test_direct_motion_is_blocked():
    """The straight joint-space motion collides -> this really needs a planner."""
    _, cc, _ = problem()
    assert cc.is_valid(Q_PICK_A) and cc.is_valid(Q_PLACE_B)
    assert not cc.edge_valid(Q_PICK_A, Q_PLACE_B)


def test_rrt_connect_finds_valid_path():
    arm, cc, prob = problem()
    res = RRTConnect().plan(Q_PICK_A, [Q_PLACE_B], prob, 20000, np.random.default_rng(0))
    assert res.success, res.message
    assert np.allclose(res.path[0], Q_PICK_A) and np.allclose(res.path[-1], Q_PLACE_B)
    assert cc.path_valid(res.path)
    assert all(arm.within_limits(q) for q in res.path)
    assert res.nodes > 2 and len(res.trees) == 2


def test_shortcut_keeps_validity_and_shortens():
    arm, cc, prob = problem()
    rng = np.random.default_rng(1)
    res = RRTConnect().plan(Q_PICK_A, [Q_PLACE_B], prob, 20000, rng)
    short = shortcut_path(res.path, cc.edge_valid, 300, rng)
    assert cc.path_valid(short)
    assert path_length(short) <= path_length(res.path) + 1e-9
    assert np.allclose(short[0], res.path[0]) and np.allclose(short[-1], res.path[-1])


def test_invalid_start_reports_failure():
    arm, cc, prob = problem()
    res = RRTConnect().plan(np.radians([5, -80, 0]), [Q_PLACE_B], prob, 100, np.random.default_rng(2))
    assert not res.success and "Start" in res.message


def test_multiple_goals_reports_which_one():
    arm, cc, prob = problem()
    bad = np.radians([5, -80, 0])        # in collision: ignored
    res = RRTConnect().plan(Q_PICK_A, [bad, Q_PLACE_B], prob, 20000, np.random.default_rng(3))
    assert res.success and res.goal_index == 1


def test_iteration_limit_gives_understandable_failure():
    arm, cc, prob = problem()
    res = RRTConnect().plan(Q_PICK_A, [Q_PLACE_B], prob, 3, np.random.default_rng(4))
    assert not res.success and "No path found" in res.message
