"""Step 3: the gripper holds the box by friction only."""
import numpy as np

from roboarm_mujoco.robot_api import RobotSim
from roboarm_mujoco.tests.helpers import open_layout, pick_up


def _sim(grip_force=40.0):
    return RobotSim.from_layout(open_layout(), grip_force=grip_force)


def test_no_weld_or_magnet_constraints():
    """The only equality constraint couples the two finger joints; nothing
    attaches the box to the gripper."""
    import mujoco
    sim = _sim()
    for i in range(sim.model.neq):
        assert sim.model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
        joints = {sim.model.joint(int(j)).name for j in (sim.model.eq_obj1id[i], sim.model.eq_obj2id[i])}
        assert joints == {"finger_left_slide", "finger_right_slide"}
    assert sim.model.body("object").weldid[0] != sim.model.body("gripper").weldid[0]


def test_firm_grip_lifts_and_holds_the_box():
    sim = _sim()
    z0 = sim.get_state()["box_pos"][1]
    states = pick_up(sim, lift_steps=100)     # lift, then hold 2 s in the air
    s = states[-1]
    assert s["holding"]
    assert s["box_pos"][1] > z0 + 0.3        # lifted by ~APPROACH_HEIGHT
    assert np.linalg.norm(s["box_pos"] - s["tcp"]) < 0.03     # still between the fingers
    assert not sim.in_collision()             # fingers on the box are not collisions


def test_weak_grip_lets_the_box_slip():
    """2 x friction x grip force below the box weight: the box cannot be lifted."""
    sim = _sim(grip_force=2.0)                # 2 * 1.0 * 2 N = 4 N < 9.81 N
    z0 = sim.get_state()["box_pos"][1]
    s = pick_up(sim, lift_steps=50)[-1]
    assert s["box_pos"][1] < z0 + 0.05       # stayed on (or fell back to) the table
    assert s["tcp"][1] - s["box_pos"][1] > 0.2


def test_opening_the_gripper_drops_the_box():
    sim = _sim()
    pick_up(sim, lift_steps=10)
    sim.set_gripper(False)
    for _ in range(60):
        s = sim.step(sim.target)
    assert not s["holding"]
    assert s["tcp"][1] - s["box_pos"][1] > 0.2


def test_box_does_not_walk_along_the_fingers():
    """Rotating the hand back and forth must not make the box creep along the
    fingers (with MuJoCo's default soft contacts it moved ~2.8 mm per move)."""
    from roboarm_mujoco.trajectory import drive
    sim = _sim()
    pick_up(sim)
    drive(sim, [sim.get_state()["q"], sim.home], speed=0.2, settle_steps=50)
    m, d = sim.model, sim.data
    finger, box = m.body("finger_left").id, m.body("object").id

    def along_finger():
        return (d.xmat[finger].reshape(3, 3).T @ (d.xpos[box] - d.xpos[finger]))[0]

    start = along_finger()
    for target in [sim.home + np.radians([0, 0, 40]), sim.home] * 2:
        for q in np.linspace(sim.get_state()["q"], target, 100):
            sim.step(q)
    assert sim.holding()
    assert abs(along_finger() - start) < 0.003          # 4 moves: was ~11 mm, now ~1.6 mm
