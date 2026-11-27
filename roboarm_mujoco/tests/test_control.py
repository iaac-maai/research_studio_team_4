"""Step 2: position-controlled joints under gravity."""
import numpy as np

from roboarm_mujoco import settings as S
from roboarm_mujoco.robot_api import RobotSim
from roboarm_mujoco.trajectory import drive, resample
from roboarm_mujoco.tests.helpers import open_layout
import config as rcfg


def test_gravity_is_on():
    sim = RobotSim()
    assert sim.model.opt.gravity[2] == S.GRAVITY < 0


def test_step_advances_control_dt():
    sim = RobotSim()
    t0 = sim.time
    sim.step(sim.home)
    assert abs(sim.time - t0 - S.CONTROL_DT) < 1e-9


def test_holds_home_pose_under_gravity():
    sim = RobotSim()
    for _ in range(100):
        s = sim.step(sim.home)
    assert np.max(np.abs(s["q"] - sim.home)) < 1e-3


def test_reaches_a_new_target():
    sim = RobotSim.from_layout(open_layout())  # (in the default layout this pose hits the right beam)
    goal = sim.home + np.array([-0.3, 0.3, 0.4])
    for _ in range(100):                       # 2 s
        s = sim.step(goal)
    assert np.max(np.abs(s["q"] - goal)) < 2e-3


def test_obstacles_physically_block_the_arm():
    sim = RobotSim()
    goal = sim.home + np.array([-0.3, 0.3, 0.4])
    for _ in range(100):
        s = sim.step(goal)
    assert sim.in_collision() and np.max(np.abs(s["q"] - goal)) > 0.1


def test_follows_moving_target_without_lag():
    """At the planner's joint speed the arm stays close to the moving target."""
    sim = RobotSim.from_layout(open_layout())
    path = [sim.home, sim.home + np.array([-0.8, 0.6, 0.8])]
    errors = [np.max(np.abs(s["q"] - t)) for s, t in
              zip(drive(sim, path, rcfg.JOINT_SPEED), resample(path, rcfg.JOINT_SPEED * S.CONTROL_DT))]
    assert max(errors[10:]) < 0.01           # after the first 0.2 s


def test_targets_are_clipped_to_joint_limits():
    sim = RobotSim()
    far = sim.upper + 1.0
    for _ in range(150):
        s = sim.step(far)
    assert np.all(s["q"] <= sim.upper + 0.02)
