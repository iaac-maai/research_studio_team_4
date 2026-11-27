"""Step 6: MuJoCo collision checking for roboarm's planner, and the full task."""
import mujoco
import numpy as np
import pytest

from roboarm_mujoco import scene
from roboarm_mujoco.pick_place import PlanExecutor, plan_for
from roboarm_mujoco.planning import MujocoChecker
from roboarm_mujoco.robot_api import RobotSim
from roboarm_mujoco.tests.helpers import open_layout
from robot import HeldObject, PlanarArm


def test_checker_agrees_with_the_simulation():
    """Arm-vs-world validity equals RobotSim.in_collision() for random poses (margin 0, no box)."""
    sim, arm, rng = RobotSim(), PlanarArm(), np.random.default_rng(2)
    chk = MujocoChecker(sim.layout, arm, 0.0, object_pose=None)
    sim.data.qpos[sim._obj_q] = [0.0, 50.0, 0.0]           # box out of the way, as in the checker
    for _ in range(300):
        q = arm.sample(rng)
        sim.data.qpos[sim._arm_q] = q
        mujoco.mj_forward(sim.model, sim.data)
        assert chk.is_valid(q) == (not sim.in_collision())


def _pointing_down(arm, tcp=(0.0, 2.5)):
    """A pose with the TCP at `tcp` and the gripper pointing straight down."""
    q = arm.ik_solutions(np.array(tcp), -np.pi / 2, np.random.default_rng(0))[0]
    return q, arm.tcp(q)


def test_margin_means_clearance():
    """A block 3 cm below the fingertips: fine with a 2 cm margin, invalid with 4 cm."""
    arm = PlanarArm()
    q, tcp = _pointing_down(arm)
    tip_y = tcp[1] - (arm.finger_length - arm.tcp_offset)          # fingertips 0.10 below the TCP
    block = {"name": "Block", "type": "obstacle", "shape": "box", "x": tcp[0], "y": tip_y - 0.03 - 0.05,
             "width": 0.8, "height": 0.10}                       # wider than the open fingers
    rows = open_layout() + [block]
    assert MujocoChecker(rows, arm, 0.0).is_valid(q)
    assert MujocoChecker(rows, arm, 0.02).is_valid(q)
    assert not MujocoChecker(rows, arm, 0.04).is_valid(q)


def test_held_box_is_checked_against_obstacles():
    """A block just under the held box's bottom face (clear of the fingers)."""
    arm = PlanarArm()
    q, tcp = _pointing_down(arm)
    held = HeldObject((0.15, 0.15), np.zeros(2), 0.0)
    block = {"name": "Block", "type": "obstacle", "shape": "box", "x": tcp[0], "y": tcp[1] - 0.17,
             "width": 0.2, "height": 0.06}                       # top at tcp - 0.14, box bottom at tcp - 0.15
    rows = open_layout() + [block]
    assert MujocoChecker(rows, arm, 0.0, gripper_closed=True).is_valid(q)
    assert not MujocoChecker(rows, arm, 0.0, held=held, gripper_closed=True).is_valid(q)


def test_edge_check_catches_thin_obstacle_between_poses():
    sim, arm = RobotSim.from_layout(open_layout()), PlanarArm()
    qa = sim.home
    qb = qa + np.array([-0.6, 0.0, 0.0])
    mid = arm.tcp((qa + qb) / 2)
    post = {"name": "Post", "type": "obstacle", "shape": "box", "x": mid[0], "y": mid[1],
            "width": 0.06, "height": 0.06}
    chk = MujocoChecker(open_layout() + [post], arm, 0.0)
    assert chk.is_valid(qa) and chk.is_valid(qb)
    assert not chk.edge_valid(qa, qb)


@pytest.fixture(scope="module")
def executed_task():
    sim = RobotSim(seed=0)
    plan = plan_for(sim)
    ex = PlanExecutor(sim, plan) if plan.success else None
    state = ex.run() if ex else None
    return sim, plan, ex, state


def test_default_task_is_planned(executed_task):
    _, plan, _, _ = executed_task
    assert plan.success, plan.message
    assert plan.pick_table == "Table A" and plan.place_table == "Table B"
    assert plan.metrics["joint_path_rad"] <= plan.metrics["raw_joint_path_rad"]


def test_default_task_executes_physically(executed_task):
    sim, plan, ex, s = executed_task
    assert plan.success
    assert ex.collisions == []
    b = next(r for r in sim.layout if r["name"] == "Table B")
    rest_z = b["y"] + b["height"] / 2 + 0.15
    assert abs(s["box_pos"][0] - plan.place_center[0]) < 0.05
    assert abs(s["box_pos"][1] - rest_z) < 0.01                 # resting on Table B
    assert abs(s["box_angle"]) < 0.05                            # upright
    assert not s["holding"]
    np.testing.assert_allclose(s["q"], sim.home, atol=0.01)      # returned home
