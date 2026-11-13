"""Stage A/B: forward kinematics, joint limits and inverse kinematics."""
import numpy as np
import pytest

import config as cfg
from robot import PlanarArm, wrap_angle


@pytest.fixture
def arm():
    return PlanarArm()


def test_straight_arm_points_along_x(arm):
    pts = arm.forward_kinematics(np.zeros(arm.n_joints))
    expected_x = cfg.ROBOT_BASE[0] + np.concatenate([[0], np.cumsum(arm.lengths)])
    assert np.allclose(pts[:, 0], expected_x)
    assert np.allclose(pts[:, 1], cfg.ROBOT_BASE[1])


def test_relative_joint_angles(arm):
    # link 1 straight up, link 2 turned 90 deg clockwise -> pointing right
    q = np.radians([90, -90, 0])
    pts = arm.forward_kinematics(q)
    base = np.array(cfg.ROBOT_BASE)
    assert np.allclose(pts[1], base + [0, arm.lengths[0]])
    assert np.allclose(pts[2], pts[1] + [arm.lengths[1], 0])
    assert np.allclose(pts[3], pts[2] + [arm.lengths[2], 0])


def test_batch_matches_single(arm):
    rng = np.random.default_rng(0)
    Q = arm.sample(rng, 20)
    batch = arm.forward_kinematics(Q)
    for q, pts in zip(Q, batch):
        assert np.allclose(arm.forward_kinematics(q), pts)


def test_link_lengths_are_preserved(arm):
    q = arm.sample(np.random.default_rng(1))
    pts = arm.forward_kinematics(q)
    assert np.allclose(np.linalg.norm(np.diff(pts, axis=0), axis=1), arm.lengths)


def test_joint_limits(arm):
    q = arm.upper + 0.5
    assert not arm.within_limits(q)
    assert arm.within_limits(arm.clip(q))
    assert np.all(arm.within_limits(arm.sample(np.random.default_rng(2), 100)))


def test_jacobian_matches_finite_differences(arm):
    q = np.radians([100, -40, -70])
    J = arm.jacobian(q)
    eps = 1e-6
    for i in range(arm.n_joints):
        dq = np.zeros(arm.n_joints)
        dq[i] = eps
        p1, a1 = arm.tcp_pose(q + dq)
        p0, a0 = arm.tcp_pose(q - dq)
        assert np.allclose(J[:2, i], (p1 - p0) / (2 * eps), atol=1e-5)
        assert np.isclose(J[2, i], (a1 - a0) / (2 * eps))


def test_ik_reaches_target_with_orientation(arm):
    rng = np.random.default_rng(3)
    target, angle = np.array([-2.9, 1.5]), np.radians(-90)
    sols = arm.ik_solutions(target, angle, rng)
    assert sols, "IK should find at least one solution"
    for q in sols:
        p, a = arm.tcp_pose(q)
        assert np.allclose(p, target, atol=1e-4)
        assert abs(wrap_angle(a - angle)) < 1e-4
        assert arm.within_limits(q)


def test_ik_reports_unreachable_target(arm):
    far = np.array([0.0, 50.0])
    assert arm.ik_solutions(far, 0.0, np.random.default_rng(4), restarts=5) == []


def test_cartesian_line_is_straight(arm):
    rng = np.random.default_rng(5)
    q0 = arm.ik_solutions(np.array([3.0, 1.3]), np.radians(-90), rng)[0]
    path = arm.cartesian_line(q0, np.array([3.0, 1.65]), np.radians(-90))
    assert path is not None
    tcp = arm.tcp(path)
    assert np.allclose(tcp[:, 0], 3.0, atol=1e-4)
    assert np.all(np.diff(tcp[:, 1]) > 0)


def test_works_with_more_joints():
    arm = PlanarArm(link_lengths=[1.5, 1.2, 1.0, 0.6], link_radii=[0.1, 0.09, 0.08, 0.06],
                    joint_limits_deg=[(0, 180)] + [(-150, 150)] * 3)
    sols = arm.ik_solutions(np.array([-2.5, 1.5]), np.radians(-90), np.random.default_rng(6))
    assert sols and arm.forward_kinematics(sols[0]).shape == (5, 2)
