"""Stage D: full pick-and-place task planning."""
import numpy as np
import pytest

import config as cfg
from collision import CollisionChecker
from environment import Environment
from geometry import rect_points
from pick_place import GripperAction, Move, TaskPlanner
from robot import PlanarArm, wrap_angle

HOME = np.radians(cfg.HOME_CONFIG_DEG)


@pytest.fixture(scope="module")
def plan_and_world():
    arm, env = PlanarArm(), Environment.default()
    plan = TaskPlanner(arm, env, rng=np.random.default_rng(0)).plan_pick_and_place(HOME)
    return arm, env, plan


def test_plan_succeeds_with_expected_sequence(plan_and_world):
    _, _, plan = plan_and_world
    assert plan.success, plan.message
    kinds = [s.kind if isinstance(s, Move) else s.action for s in plan.steps]
    assert kinds[:8] == ["transit", "approach", "close", "retreat", "carry", "approach", "open", "retreat"]


def test_moves_are_continuous(plan_and_world):
    """No teleporting: every move starts where the previous one ended."""
    _, _, plan = plan_and_world
    moves = plan.moves
    assert np.allclose(moves[0].path[0], HOME)
    for a, b in zip(moves[:-1], moves[1:]):
        assert np.allclose(a.path[-1], b.path[0], atol=1e-9)


def test_every_move_is_collision_free_in_its_scene(plan_and_world):
    arm, env, plan = plan_and_world
    placed = env.copy()
    placed.place_object_at("B")
    carrying = env.copy()
    carrying.object_station = None
    released = False
    for step in plan.steps:
        if isinstance(step, GripperAction):
            released = released or step.action == "open"
            continue
        scene = carrying if step.held is not None else (placed if released else env)
        cc = CollisionChecker(arm, scene, step.held)
        assert cc.path_valid(step.path), step.label


def test_object_ends_on_table_b(plan_and_world):
    arm, env, plan = plan_and_world
    carry_moves = [m for m in plan.moves if m.held is not None]
    q_release = carry_moves[-1].path[-1]
    center, angle = arm.object_pose(q_release, carry_moves[-1].held)
    target = env.stations["B"].object_rest_center(env.object_size)
    assert np.allclose(center, target, atol=1e-3)
    assert abs(wrap_angle(angle)) < 1e-3     # same orientation as on table A


def test_grasp_is_at_the_object(plan_and_world):
    arm, env, plan = plan_and_world
    q_grasp = plan.moves[1].path[-1]
    assert np.allclose(arm.tcp(q_grasp), env.object_center, atol=1e-3)


def test_metrics_present(plan_and_world):
    _, _, plan = plan_and_world
    m = plan.metrics
    for key in ("planning_time_s", "tree_nodes", "samples", "joint_path_length_rad",
                "tcp_path_length_m", "min_clearance_m", "free_space_clearance_m", "collision_checks"):
        assert key in m
    assert m["free_space_clearance_m"] >= cfg.PLANNING_MARGIN - 0.035   # discretisation tolerance
    assert m["min_clearance_m"] > 0


def test_blocked_pick_gives_clear_message():
    arm, env = PlanarArm(), Environment.default()
    i = env.movable_obstacle_at(env.obstacles[-1].points.mean(axis=0))
    c = env.object_center
    env.obstacles[i].points = rect_points(c[0] - 0.4, c[1] + 0.2, c[0] + 0.4, c[1] + 0.5)  # lid over object
    plan = TaskPlanner(arm, env, rng=np.random.default_rng(1)).plan_pick_and_place(HOME)
    assert not plan.success
    assert "grasp" in plan.message.lower() and "A" in plan.message


def test_unreachable_station_gives_clear_message():
    arm = PlanarArm(link_lengths=[1.0, 0.8, 0.5])
    env = Environment.default()
    plan = TaskPlanner(arm, env, rng=np.random.default_rng(2)).plan_pick_and_place(np.radians([90, 0, 0]))
    assert not plan.success and "out of reach" in plan.message


def test_start_in_collision_gives_clear_message():
    arm, env = PlanarArm(), Environment.default()
    plan = TaskPlanner(arm, env, rng=np.random.default_rng(3)).plan_pick_and_place(np.radians([5, -80, 0]))
    assert not plan.success and "collision" in plan.message


def test_joint_goal_mode():
    arm, env = PlanarArm(), Environment.default()
    goal = np.radians([49.9, -26.3, -113.7])
    plan = TaskPlanner(arm, env, rng=np.random.default_rng(4)).plan_joint_goal(HOME, goal)
    assert plan.success and np.allclose(plan.moves[0].path[-1], goal)
