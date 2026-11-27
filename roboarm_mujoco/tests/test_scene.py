"""Step 1-2: the MuJoCo model matches roboarm's robot and the layout rules."""
import mujoco
import numpy as np
import pytest

from roboarm_mujoco import scene
from roboarm_mujoco.robot_api import RobotSim
from roboarm_mujoco.tests.helpers import put_arm
from robot import PlanarArm


def test_roboarm_modules_importable():
    import config
    import planner
    assert hasattr(planner, "RRTConnect") and hasattr(config, "LINK_LENGTHS")


def test_default_layout_is_valid_and_complete():
    rows = scene.validate_layout(scene.default_layout())
    names = {r["name"] for r in rows}
    assert {"Table A", "Table B", "Object"} <= names
    assert sum(r["type"] == "box" for r in rows) == 1


def test_every_layout_row_becomes_a_geom():
    sim = RobotSim()
    geom_names = {sim.model.geom(i).name for i in range(sim.model.ngeom)}
    for name in scene.unique_names(sim.layout):
        assert name in geom_names


def test_forward_kinematics_matches_roboarm():
    """TCP position and gripper angle agree with roboarm/robot.py for random poses."""
    sim, arm = RobotSim(), PlanarArm()
    rng = np.random.default_rng(1)
    gripper = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    for _ in range(50):
        q = arm.sample(rng)
        put_arm(sim, q)
        np.testing.assert_allclose(sim.get_state()["tcp"], arm.tcp(q), atol=1e-9)
        x_axis = sim.data.xmat[gripper].reshape(3, 3)[:, 0]          # gripper axis u
        np.testing.assert_allclose(np.arctan2(x_axis[2], x_axis[0]),
                                   np.arctan2(np.sin(q.sum()), np.cos(q.sum())), atol=1e-9)


def test_joint_limits_come_from_config():
    sim, arm = RobotSim(), PlanarArm()
    np.testing.assert_allclose(sim.lower, arm.lower)
    np.testing.assert_allclose(sim.upper, arm.upper)


def test_layout_validation_errors():
    rows = scene.default_layout()
    with pytest.raises(scene.LayoutError):
        scene.validate_layout(rows + [dict(rows[-1])])                  # two boxes
    with pytest.raises(scene.LayoutError):
        scene.validate_layout([{**rows[0], "type": "chair"}] + rows[1:])
    with pytest.raises(scene.LayoutError):
        scene.validate_layout([{**rows[0], "shape": "triangle"}] + rows[1:])


def test_circle_obstacle_is_a_cylinder():
    rows = scene.default_layout() + [{"name": "Disc", "type": "obstacle", "shape": "circle",
                                      "x": 0.0, "y": 3.0, "width": 0.4, "height": 0.4}]
    m = mujoco.MjModel.from_xml_string(scene.build_xml(rows))
    g = m.geom("obstacle:Disc")
    assert g.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert g.size[0] == pytest.approx(0.2)
