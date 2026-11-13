"""
metrics.py - numbers that describe a trajectory, so different planners
(today RRT-Connect, later OMPL planners or a reinforcement-learning policy)
can be compared on the same task.
"""
import csv
import datetime
import os

import numpy as np

from collision import CollisionChecker
from planner import path_length


def densify(path, max_step=0.01):
    """Insert intermediate configurations so no joint-space step exceeds max_step (rad)."""
    path = np.asarray(path, float)
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / max_step)))
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.extend(a + t * (b - a))
    return np.array(out)


def trajectory_metrics(robot, env, moves, place_station=None):
    """Joint-space length, tool path length and clearance of a list of Moves.

    Clearance is evaluated against the scene as it is during each move: the
    resting object is an obstacle unless it is being carried. "free-space"
    clearance ignores the approach/retreat moves, where the gripper is
    intentionally close to the table.
    """
    joint_len = raw_len = tcp_len = 0.0
    min_clear, free_clear = np.inf, np.inf
    placed_env = None
    for k, mv in enumerate(moves):
        joint_len += path_length(mv.path)
        raw_len += path_length(mv.raw_path if mv.raw_path is not None else mv.path)
        dense = densify(mv.path)
        tcp = robot.tcp(dense)
        tcp_len += float(np.sum(np.linalg.norm(np.diff(tcp, axis=0), axis=1)))
        # Scene for this move: the object is on its original table before the
        # grasp, carried in between, and on the target table afterwards.
        scene = env.copy()
        if mv.held is not None:
            scene.object_station = None
        elif place_station is not None and any(m.held is not None for m in moves[:k]):
            if placed_env is None:
                placed_env = env.copy()
                placed_env.place_object_at(place_station)
            scene = placed_env
        checker = CollisionChecker(robot, scene, mv.held,
                                   gripper_closed=mv.held is not None)
        c = float(np.min(checker.clearance_batch(dense)))
        min_clear = min(min_clear, c)
        if mv.kind in ("transit", "carry"):
            free_clear = min(free_clear, c)
    return {
        "joint_path_length_rad": joint_len,
        "raw_joint_path_length_rad": raw_len,
        "tcp_path_length_m": tcp_len,
        "min_clearance_m": min_clear,
        "free_space_clearance_m": free_clear,
        "waypoints": int(sum(len(m.path) for m in moves)),
    }


LOG_FIELDS = ["timestamp", "planner", "success", "planning_time_s", "rrt_time_s", "rrt_queries",
              "samples", "tree_nodes", "collision_checks", "joint_path_length_rad",
              "raw_joint_path_length_rad", "tcp_path_length_m", "min_clearance_m",
              "free_space_clearance_m", "waypoints", "link_lengths", "max_iterations", "message"]


LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "metrics_log.csv")


def append_log(metrics, path=None, extra=None):
    """Append one planning run to a CSV file (created with a header if needed)."""
    path = path or LOG_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    row.update(metrics)
    row.update(extra or {})
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    return path
