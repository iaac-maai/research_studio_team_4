"""Shared test helpers."""
import mujoco
import numpy as np

from roboarm_mujoco import paths  # noqa: F401
from roboarm_mujoco import scene
from roboarm_mujoco.trajectory import drive
import config as rcfg
from robot import PlanarArm


def open_layout():
    """Default layout with only the two tables and the box (no obstacles)."""
    return [r for r in scene.default_layout() if r["type"] != "obstacle"]


def put_arm(sim, q):
    """Teleport the arm (no physics) and refresh contacts."""
    sim.data.qpos[sim._arm_q] = q
    mujoco.mj_forward(sim.model, sim.data)


def grasp_pose(sim, rng=None):
    """IK pose with the TCP on the box centre, gripper pointing down, plus the
    pre-grasp pose APPROACH_HEIGHT above it. Picks the solution closest to home."""
    arm = PlanarArm()
    rng = rng or np.random.default_rng(0)
    box = sim.get_state()["box_pos"]
    angle = np.radians(rcfg.GRASP_ANGLE_DEG)
    best = None
    for q in arm.ik_solutions(box, angle, rng):
        line = arm.cartesian_line(q, box + [0, rcfg.APPROACH_HEIGHT], angle)
        if line is not None and (best is None or np.linalg.norm(q - sim.home) < np.linalg.norm(best[0] - sim.home)):
            best = (q, line)
    assert best is not None, "no grasp pose found"
    return best[0], best[1]          # grasp pose, path grasp -> pre-grasp (going up)


def pick_up(sim, lift_steps=0):
    """Open-layout pick: home -> above box -> down -> close -> lift.
    Returns the states recorded while lifting."""
    q_grasp, up = grasp_pose(sim)
    drive(sim, [sim.home, up[-1]], speed=0.9, settle_steps=25)
    drive(sim, up[::-1], speed=0.3, settle_steps=25)
    sim.set_gripper(True)
    drive(sim, [up[0], up[0]], speed=1.0, settle_steps=30)
    states = drive(sim, up, speed=0.3, settle_steps=lift_steps)
    return states
