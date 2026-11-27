"""Step 4: what counts as a collision (agreed rules)."""
import mujoco
import numpy as np
import pytest

from roboarm_mujoco import scene
from roboarm_mujoco.robot_api import RobotSim
from roboarm_mujoco.tests.helpers import open_layout, put_arm


def _sim(layout):
    return RobotSim.from_layout(layout)


def test_home_pose_is_collision_free():
    sim = RobotSim()
    assert not sim.in_collision()


def test_box_resting_on_table_is_not_a_collision():
    sim = RobotSim()
    for _ in range(25):
        sim.step(sim.home)
    assert sim.data.ncon > 0                  # the box does touch the table ...
    assert not sim.in_collision()             # ... but that is allowed


@pytest.mark.parametrize("part", scene.ROBOT_GEOMS)
def test_every_robot_part_detects_an_obstacle(part):
    """A small block placed on each part of the arm at home is detected."""
    probe = RobotSim()
    g = mujoco.mj_name2id(probe.model, mujoco.mjtObj.mjOBJ_GEOM, part)
    x, _, z = probe.data.geom_xpos[g]
    block = {"name": "Probe", "type": "obstacle", "shape": "box", "x": x, "y": z, "width": 0.05, "height": 0.05}
    sim = _sim(open_layout() + [block])
    assert any("obstacle:Probe" in pair and part in pair for pair in sim.collisions())


def test_arm_in_floor_is_a_collision():
    sim = RobotSim()
    put_arm(sim, np.array([0.0, -np.pi / 2, 0.0]))
    assert sim.in_collision()
    assert all("floor" in pair for pair in sim.collisions())


def test_self_collision_is_detected():
    sim = RobotSim()
    put_arm(sim, np.array([np.pi / 2, 2.6, 2.6]))      # folded: wrist back onto link 1
    assert sim.in_collision()
    assert ("link1", "link3") in sim.collisions()


def test_arm_on_table_is_a_collision():
    sim = RobotSim()
    a = next(r for r in sim.layout if r["name"] == "Table A")
    block_q = None
    from robot import PlanarArm
    arm = PlanarArm()
    for q in arm.ik_solutions(np.array([a["x"] + 0.4, a["y"] + a["height"] / 2 - 0.05]), -np.pi / 2,
                              np.random.default_rng(0)):
        block_q = q
        break
    put_arm(sim, block_q)                              # fingertips inside the table top
    assert any(p[0].startswith("table:") or p[1].startswith("table:") for p in sim.collisions())


def test_box_touching_obstacle_is_a_collision_box_on_floor_is_not():
    obj = scene.object_row(scene.default_layout())
    wall = {"name": "Wall", "type": "obstacle", "shape": "box", "x": obj["x"] + obj["width"] / 2 + 0.05,
            "y": obj["y"], "width": 0.1, "height": 0.1}
    assert _sim(open_layout() + [wall]).in_collision()

    sim = RobotSim()
    sim.data.qpos[sim._obj_q] = [0.9, obj["height"] / 2, 0.0]   # box on the floor
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.data.ncon > 0 and not sim.in_collision()
