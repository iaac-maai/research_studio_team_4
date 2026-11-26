"""Naive "try anyway" motion shown when planning fails (attempt.py)."""
import numpy as np

import config as cfg
from attempt import crash_message, naive_attempt
from collision import CollisionChecker
from environment import Environment
from robot import PlanarArm


def test_crash_message_wording():
    assert crash_message(["palm hits Left beam"]) == "Oops! Crashed into Left beam (palm hit it)."
    assert crash_message(["link 1 hits link 3"]).startswith("Oops! The arm hit itself")
    assert crash_message([]) == "Oops! Crashed into something."


def test_attempt_stops_at_first_collision():
    robot, env = PlanarArm(), Environment.default()
    att = naive_attempt(robot, env, np.radians(cfg.HOME_CONFIG_DEG))
    checker = CollisionChecker(robot, env)
    ok = checker.valid_batch(att.path)
    if att.outcome == "crash":
        assert ok[:-1].all() and not ok[-1]       # free right up to the impact pose
    else:
        assert ok.all()


def test_unreachable_target_stretches_toward_it():
    robot, env = PlanarArm(), Environment.default()
    env.obstacles = [o for o in env.obstacles if o.kind not in ("hanging", "rising", "free")]
    for i in range(3):
        robot.set_link_length(i, 0.6)             # far too short for either table
    att = naive_attempt(robot, env, np.radians([90.0, 0.0, 0.0]))
    assert att.outcome == "unreachable", att.message
    assert "missed it by" in att.message
    assert CollisionChecker(robot, env).valid_batch(att.path).all()
