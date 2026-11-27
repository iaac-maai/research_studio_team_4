"""Step 8: the viewer app's keyboard logic, run without a window."""
import time

import numpy as np

from roboarm_mujoco.main import KEY_GRIPPER, KEY_PAUSE, KEY_PLAN, KEY_RESET, App
from roboarm_mujoco.robot_api import RobotSim


def _run_until(app, condition, max_steps=6000, wall_timeout=120):
    t0 = time.time()
    for _ in range(max_steps):
        app.update()
        if condition():
            return True
        if app.planning:
            time.sleep(0.01)
        if time.time() - t0 > wall_timeout:
            break
    return False


def test_gripper_key_toggles():
    app = App(RobotSim())
    app.key_callback(KEY_GRIPPER)
    app.update()
    assert app.sim.get_state()["gripper_closed"]
    app.key_callback(KEY_GRIPPER)
    app.update()
    assert not app.sim.get_state()["gripper_closed"]


def test_plan_key_runs_the_task_and_pause_works():
    app = App(RobotSim(seed=0), speed=2.0)
    app.key_callback(KEY_PLAN)
    assert _run_until(app, lambda: app.executor is not None), app.status
    app.key_callback(KEY_PAUSE)
    app.update()
    q = app.sim.get_state()["q"]
    for _ in range(10):
        app.update()
    np.testing.assert_allclose(app.sim.get_state()["q"], q, atol=0.02)    # paused: holds still
    app.key_callback(KEY_PAUSE)
    assert _run_until(app, lambda: app.executor is None)
    assert "Done" in app.status and "No collisions" in app.status
    assert app.sim.get_state()["box_pos"][0] > 2.0                       # on Table B


def test_reset_key_restores_start_and_cancels():
    app = App(RobotSim(seed=0))
    start = app.sim.get_state()
    app.key_callback(KEY_PLAN)
    app.update()
    app.key_callback(KEY_RESET)
    app.update()
    assert app.executor is None and "Reset" in app.status
    np.testing.assert_allclose(app.sim.get_state()["box_pos"], start["box_pos"], atol=1e-3)
    _run_until(app, lambda: not app.planning, max_steps=2000)
    app.update()
    assert app.executor is None                                         # stale plan was discarded
