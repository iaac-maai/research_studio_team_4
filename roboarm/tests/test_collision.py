"""Stage B: geometry and collision checking."""
import numpy as np
import pytest

import config as cfg
from collision import CollisionChecker
from environment import Environment
from geometry import point_in_convex_polygon, rect_points, segment_segment_distance
from robot import HeldObject, PlanarArm


def test_segment_distance_basic():
    a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert np.isclose(segment_segment_distance(a, b, np.array([0.5, 1.0]), np.array([0.5, 2.0])), 1.0)
    # crossing segments
    assert segment_segment_distance(a, b, np.array([0.5, -1.0]), np.array([0.5, 1.0])) == 0.0
    # endpoint-to-endpoint
    assert np.isclose(segment_segment_distance(a, b, np.array([2.0, 0.0]), np.array([3.0, 0.0])), 1.0)


def test_point_in_polygon():
    r = rect_points(0, 0, 1, 1)
    assert point_in_convex_polygon([0.5, 0.5], r)
    assert not point_in_convex_polygon([1.5, 0.5], r)


@pytest.fixture
def setup():
    arm, env = PlanarArm(), Environment.default()
    return arm, env, CollisionChecker(arm, env)


def test_home_is_valid(setup):
    arm, env, cc = setup
    assert cc.is_valid(np.radians(cfg.HOME_CONFIG_DEG))


def test_every_link_is_checked(setup):
    """Put an obstacle on each link in turn: the configuration must become invalid."""
    arm, env, _ = setup
    q = np.radians(cfg.HOME_CONFIG_DEG)
    pts = arm.forward_kinematics(q)
    for k in range(arm.n_joints):
        mid = (pts[k] + pts[k + 1]) / 2
        e = env.copy()
        i = e.movable_obstacle_at(e.obstacles[-1].points.mean(axis=0))
        e.obstacles[i].points = rect_points(mid[0] - 0.05, mid[1] - 0.05, mid[0] + 0.05, mid[1] + 0.05)
        cc = CollisionChecker(arm, e)
        assert not cc.is_valid(q), f"link {k + 1} collision not detected"
        assert k in cc.report(q)["colliding"]


def test_obstacle_fully_inside_link_capsule_is_detected(setup):
    arm, env, _ = setup
    q = np.radians(cfg.HOME_CONFIG_DEG)
    p = arm.forward_kinematics(q)[1]
    env.obstacles[-1].points = rect_points(p[0] - 0.01, p[1] - 0.01, p[0] + 0.01, p[1] + 0.01)
    assert not CollisionChecker(arm, env).is_valid(q)


def test_floor_and_self_collision(setup):
    arm, env, cc = setup
    assert not cc.is_valid(np.radians([5, -80, 0]))         # forearm into the floor
    assert not cc.is_valid(np.radians([90, 150, 150]))      # arm folded onto itself
    assert any("hits link" in s for s in cc.report(np.radians([90, 150, 150]))["pairs"])


def test_fast_check_equals_exact_clearance(setup):
    arm, env, cc = setup
    Q = arm.sample(np.random.default_rng(0), 3000)
    exact = arm.within_limits(Q) & (cc.clearance_batch(Q) > cc.margin)
    assert np.array_equal(cc.valid_batch(Q), exact)


def test_held_object_is_checked():
    arm, env = PlanarArm(), Environment.default()
    q = np.radians(cfg.HOME_CONFIG_DEG)
    tcp, angle = arm.tcp_pose(q)
    big = HeldObject((0.6, 0.6), np.zeros(2), 0.0)             # a huge box: must hit something
    env.object_station = None
    assert CollisionChecker(arm, env).is_valid(q)
    assert not CollisionChecker(arm, env, big).is_valid(q)


def test_edge_check_catches_thin_obstacle(setup):
    """Both end poses are free but the motion between them sweeps through a wall."""
    arm, env, _ = setup
    e = env.copy()
    for o in e.obstacles:
        if o.movable:
            o.points = rect_points(10, 10, 10.1, 10.1)   # move all obstacles away
    e.obstacles[-1].points = rect_points(-0.05, 4.0, 0.05, 5.0)
    cc = CollisionChecker(arm, e)
    qa, qb = np.radians([120, 0, 0]), np.radians([60, 0, 0])
    assert cc.is_valid(qa) and cc.is_valid(qb)
    assert not cc.edge_valid(qa, qb)
