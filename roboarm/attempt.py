"""
attempt.py - what happens if the arm just TRIES, without a motion planner?

When planning fails (arm too short, every grasp pose blocked, no path found),
the simulator still shows a motion so you can see *why* it fails: the arm
moves in a straight joint-space line toward the task target - no obstacle
avoidance at all - and stops at the first collision ("Oops! Crashed into ...")
or at the closest pose it can reach ("Oops! Can't reach ..."). The simulation
then drives the arm back to where it started.

This is a deliberately naive baseline, not a planner: it shows the difference
between "moving toward the goal" and "finding a collision-free path".
"""
from dataclasses import dataclass

import numpy as np

import config as cfg
from collision import CollisionChecker
from robot import rotation


@dataclass
class Attempt:
    path: np.ndarray      # (K, N) configurations from the start pose to where the arm stops
    outcome: str          # "crash", "unreachable", "reached" or "stuck"
    message: str          # short, friendly explanation for the status bar / bubble
    target: np.ndarray    # TCP point the arm was trying to reach
    station: str


def crash_message(pairs):
    """'link 2 hits Right pillar' -> 'Oops! Crashed into Right pillar (link 2 hit it).'"""
    if not pairs:
        return "Oops! Crashed into something."
    part, _, other = pairs[0].partition(" hits ")
    if not other:
        return f"Oops! Crashed ({pairs[0]})."
    names = ("link", "palm", "finger", "object")
    if other.split()[0].lower() in names and part.split()[0].lower() in names:
        return f"Oops! The arm hit itself ({part} hit {other})."
    return f"Oops! Crashed into {other} ({part} hit it)."


def _target(robot, env, held):
    """(TCP target, station): grasp the object, or put it down if already holding it."""
    angle = np.radians(cfg.GRASP_ANGLE_DEG)
    if held is None:
        station = env.object_station or "A"
        return env.object_center.copy(), station
    station = "B" if held.origin == "A" else "A"
    center = env.stations[station].object_rest_center(env.object_size)
    return center - rotation(angle) @ held.rel_pos, station


def _goal(robot, target, q_start, rng):
    """Joint goal for the target and how far the TCP ends up from it.

    Uses a proper IK solution if one exists (the one nearest the start pose -
    the naive choice). Otherwise the arm stretches as close as it can: the IK
    solver's best non-converged result over several seeds.
    """
    angle = np.radians(cfg.GRASP_ANGLE_DEG)
    sols = robot.ik_solutions(target, angle, rng, q_seed=q_start)
    if sols:
        q = min(sols, key=lambda s: np.linalg.norm(s - q_start))
        return q, 0.0
    best, best_err = None, np.inf
    seeds = [q_start] + [robot.sample(rng) for _ in range(cfg.IK_RESTARTS)]
    for seed in seeds:
        q, _ = robot.solve_ik(target, angle, seed)
        err = float(np.linalg.norm(robot.tcp(q) - target))
        if err < best_err:
            best, best_err = q, err
    return best, best_err


def naive_attempt(robot, env, q_start, held=None, rng=None):
    """Straight joint-space motion toward the task target, cut at the first collision."""
    rng = rng if rng is not None else np.random.default_rng(0)
    q_start = np.asarray(q_start, float)
    target, station = _target(robot, env, held)
    scene = env.copy()
    if held is not None:
        scene.object_station = None              # the carried object is not an obstacle
    checker = CollisionChecker(robot, scene, held, gripper_closed=held is not None)
    where = "the object on table " + station if held is None else "table " + station

    if not checker.is_valid(q_start):
        pairs = checker.report(q_start)["pairs"]
        what = pairs[0] if pairs else "an obstacle"
        return Attempt(q_start[None], "stuck", f"Oops! Already stuck ({what}) - can't even start.",
                       target, station)

    q_goal, miss = _goal(robot, target, q_start, rng)
    path = checker.interpolate(q_start, q_goal, include_start=True)
    ok = checker.valid_batch(path)
    bad = np.flatnonzero(~ok)
    if bad.size:
        k = int(bad[0])                          # first configuration that touches something
        pairs = checker.report(path[k])["pairs"]
        return Attempt(path[:k + 1], "crash", crash_message(pairs), target, station)
    if miss > 0.01:
        return Attempt(path, "unreachable",
                       f"Oops! Can't reach {where} - missed it by {miss:.2f} m.", target, station)
    return Attempt(path, "reached",
                   f"Reached {where} by moving straight - but the rest of the task can't be planned.",
                   target, station)
