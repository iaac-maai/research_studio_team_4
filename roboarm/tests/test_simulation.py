"""Stage D/E: execution, attachment / release, replanning and interaction (no window)."""
import numpy as np
import pytest

import config as cfg
from simulation import Simulation


def plan_now(sim):
    sim.request_plan()
    sim._worker.join()
    sim.update(0.0)


def run_until(sim, condition, max_seconds=200.0, dt=1 / 30):
    t = 0.0
    while not condition() and t < max_seconds:
        prev_q = sim.q.copy()
        prev_obj = np.array(sim.object_pose()[0])
        sim.update(dt)
        # no teleporting: per frame, joints and object move only a little
        assert np.max(np.abs(sim.q - prev_q)) < 0.2
        assert np.linalg.norm(np.array(sim.object_pose()[0]) - prev_obj) < 0.5
        t += dt
    return t


@pytest.fixture
def sim():
    s = Simulation(seed=0)
    s.speed = 4.0
    return s


def test_full_task_a_to_b_and_back(sim):
    plan_now(sim)
    assert sim.state == "ready", sim.status
    sim.start_pause()
    run_until(sim, lambda: sim.state == "done")
    assert sim.state == "done"
    assert sim.env.object_station == "B" and sim.held is None
    target = sim.env.stations["B"].object_rest_center(sim.env.object_size)
    assert np.allclose(sim.env.object_center, target, atol=1e-3)
    assert np.allclose(sim.q, sim.home)
    # and back again
    plan_now(sim)
    assert sim.state == "ready", sim.status
    sim.start_pause()
    run_until(sim, lambda: sim.state == "done")
    assert sim.env.object_station == "A"


def test_object_moves_with_gripper(sim):
    plan_now(sim)
    sim.start_pause()
    run_until(sim, lambda: sim.held is not None)
    assert sim.held is not None
    for _ in range(30):
        sim.update(1 / 30)
        c, _ = sim.object_pose()
        assert np.linalg.norm(c - sim.robot.tcp(sim.q)) < 0.01


def test_dragging_obstacle_invalidates_and_replans_mid_task(sim):
    plan_now(sim)
    sim.start_pause()
    run_until(sim, lambda: sim.held is not None and sim.plan.steps[sim.step_index].kind == "carry")
    i = sim.env.movable_obstacle_at(np.array([0.0, 5.0]))    # centre stalactite
    sim.env.move_obstacle(i, np.array([0.0, 0.3]))
    sim.on_environment_changed()
    assert sim.state == "paused" and not sim.plan_valid
    sim.start_pause()                                         # refuses to run a stale plan
    assert sim.state == "paused"
    plan_now(sim)                                             # replan while holding the object
    assert sim.state == "ready", sim.status
    assert sim.plan.moves[0].held is not None
    sim.start_pause()
    run_until(sim, lambda: sim.state == "done")
    assert sim.env.object_station == "B"


def test_link_length_change_and_failure_message(sim):
    sim.set_link_length(1, 0.6)
    plan_now(sim)
    assert sim.state == "failed"
    assert "Planning failed" in sim.status


def test_manual_jog_respects_limits(sim):
    sim.jog(0, 10.0)
    assert np.isclose(sim.q[0], sim.robot.upper[0])


def test_joint_goal_flow(sim):
    sim.store_custom_goal()
    sim.jog(1, 0.4)
    sim.jog(2, -0.3)
    assert sim.current_checker().is_valid(sim.q)
    sim.request_joint_plan()
    sim._worker.join()
    sim.update(0.0)
    assert sim.state == "ready", sim.status
    sim.start_pause()
    run_until(sim, lambda: sim.state == "done")
    assert np.allclose(sim.q, sim.custom_goal)


def test_regenerate_produces_solvable_layout(sim):
    before = [o.points.copy() for o in sim.env.obstacles]
    sim.request_regenerate()
    sim._worker.join()
    sim.update(0.0)
    moved = any(not np.allclose(a, o.points) for a, o in zip(before, sim.env.obstacles))
    if sim.state == "ready":
        assert moved and sim.plan.success
    else:
        assert "No solvable random layout" in sim.status


def test_render_smoke(sim):
    import pygame
    from main import build_widgets
    from render import Renderer
    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
    r = Renderer(screen)
    for key in ("tree", "collision", "cspace", "help"):
        r.show[key] = True
    widgets = build_widgets(sim, r)
    plan_now(sim)
    r.draw(sim, widgets)
    sim.start_pause()
    run_until(sim, lambda: sim.held is not None)
    r.draw(sim, widgets)
    pygame.quit()
