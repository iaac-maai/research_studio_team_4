"""Step 5: RobotSim follows the agreed interface (keys, shapes, types).

The team's contract test lives in roboarm_mujoco/rl/test_contract.py (RL
person). This file checks the same agreement until that exists.
"""
import numpy as np
import pytest

from roboarm_mujoco.robot_api import STATE_KEYS, RobotSim


def _check_state(s):
    assert set(s) == set(STATE_KEYS)
    for key, shape in (("q", (3,)), ("tcp", (2,)), ("box_pos", (2,))):
        assert isinstance(s[key], np.ndarray) and s[key].shape == shape and s[key].dtype == float
    assert isinstance(s["box_angle"], float)
    assert isinstance(s["gripper_closed"], bool)
    assert isinstance(s["holding"], bool)


def test_interface_methods_exist():
    for name in ("reset", "step", "set_gripper", "get_state", "in_collision"):
        assert callable(getattr(RobotSim, name))


def test_reset_step_and_state_types():
    sim = RobotSim()
    _check_state(sim.reset(seed=0))
    _check_state(sim.step(np.zeros(3) + sim.home))
    sim.set_gripper(True)
    s = sim.get_state()
    _check_state(s)
    assert s["gripper_closed"] is True
    assert isinstance(sim.in_collision(), bool)


def test_reset_restores_the_start():
    sim = RobotSim()
    first = sim.reset(seed=3)
    for _ in range(20):
        sim.step(sim.home + 0.3)
    sim.set_gripper(True)
    again = sim.reset(seed=3)
    for k in ("q", "tcp", "box_pos"):
        np.testing.assert_allclose(again[k], first[k])
    assert again["gripper_closed"] is False and sim.time == 0.0


def test_step_accepts_lists_and_rejects_wrong_size():
    sim = RobotSim()
    sim.step(list(sim.home))
    with pytest.raises(ValueError):
        sim.step(np.zeros(4))


def test_reset_from_csv():
    layout = pytest.importorskip("roboarm_mujoco.environments.layout",
                                 reason="environments/layout.py not written yet (environments person)")
    from pathlib import Path
    csvs = sorted((Path(layout.__file__).parent).glob("*.csv"))
    if not csvs:
        pytest.skip("no environment CSV files yet")
    sim = RobotSim()
    s = sim.reset(layout_csv=str(csvs[0]), seed=0)
    rows = layout.load_layout(str(csvs[0]))
    obj = next(r for r in rows if r["type"] == "box")
    np.testing.assert_allclose(s["box_pos"], [obj["x"], obj["y"]])
