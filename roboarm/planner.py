"""
planner.py - sampling-based motion planning in joint (configuration) space.

The planner never looks at pixels or obstacles directly. It only asks two
questions, answered by a PlanningProblem:
    is_valid(q)        is this configuration collision-free and within limits?
    edge_valid(qa, qb) is the straight joint-space motion from qa to qb valid?
That separation is what makes the planner swappable: ompl_backend.py offers
the same `plan()` interface using the OMPL library.

RRT-Connect (Kuffner & LaValle, 2000)
------------------------------------
Two trees grow towards each other, one rooted at the start configuration and
one rooted at the goal configuration(s):
  1. pick a random configuration q_rand;
  2. EXTEND tree A one step (RRT_STEP_SIZE) from its nearest node towards q_rand;
  3. if that worked, CONNECT tree B: extend it repeatedly towards the new node
     until it reaches it or hits an obstacle;
  4. if tree B reached the node, the two trees are joined -> path found;
  5. swap the roles of A and B and repeat.
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

import config as cfg

TRAPPED, ADVANCED, REACHED = 0, 1, 2


class PlanningProblem:
    """Everything a planner needs to know about the robot's world."""

    def __init__(self, checker, lower, upper, cancel_event=None):
        self.checker = checker
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.cancel_event = cancel_event

    @property
    def dim(self):
        return len(self.lower)

    def is_valid(self, q):
        return self.checker.is_valid(q)

    def edge_valid(self, qa, qb):
        return self.checker.edge_valid(qa, qb)

    def sample(self, rng):
        return rng.uniform(self.lower, self.upper)

    def cancelled(self):
        return self.cancel_event is not None and self.cancel_event.is_set()


@dataclass
class PlannerResult:
    success: bool
    path: list = field(default_factory=list)       # list of configurations
    message: str = ""
    iterations: int = 0                            # random samples drawn
    nodes: int = 0                                 # configurations added to the trees
    time: float = 0.0
    goal_index: int = -1                           # which goal configuration was reached
    trees: list = field(default_factory=list)      # [(nodes, parents, label), ...] for drawing


class MotionPlanner(ABC):
    name = "abstract"

    @abstractmethod
    def plan(self, start, goals, problem, max_iterations, rng):
        """Return a PlannerResult with a path from `start` to one of `goals`."""


class Tree:
    """A tree of configurations stored in growable NumPy arrays."""

    def __init__(self, roots, dim):
        self.nodes = np.empty((max(1024, 4 * len(roots)), dim))
        self.parents = np.empty(len(self.nodes), dtype=int)
        self.root_of = np.empty(len(self.nodes), dtype=int)
        self.size = 0
        for k, r in enumerate(roots):
            self.add(r, -1, k)

    def add(self, q, parent, root=None):
        if self.size == len(self.nodes):
            self.nodes = np.concatenate([self.nodes, np.empty_like(self.nodes)])
            self.parents = np.concatenate([self.parents, np.empty_like(self.parents)])
            self.root_of = np.concatenate([self.root_of, np.empty_like(self.root_of)])
        i = self.size
        self.nodes[i] = q
        self.parents[i] = parent
        self.root_of[i] = root if root is not None else self.root_of[parent]
        self.size += 1
        return i

    def nearest(self, q):
        d = np.sum((self.nodes[:self.size] - q) ** 2, axis=1)
        return int(np.argmin(d))

    def branch(self, i):
        """Configurations from node i back to its root."""
        out = []
        while i != -1:
            out.append(self.nodes[i].copy())
            i = self.parents[i]
        return out

    def export(self, label):
        return (self.nodes[:self.size].copy(), self.parents[:self.size].copy(), label)


class RRTConnect(MotionPlanner):
    name = "RRT-Connect"

    def __init__(self, step_size=cfg.RRT_STEP_SIZE):
        self.step_size = step_size

    def _extend(self, tree, q_target, problem):
        i_near = tree.nearest(q_target)
        q_near = tree.nodes[i_near]
        delta = q_target - q_near
        dist = np.linalg.norm(delta)
        if dist < 1e-9:
            return REACHED, i_near
        reached = dist <= self.step_size
        q_new = q_target.copy() if reached else q_near + delta * (self.step_size / dist)
        if not problem.edge_valid(q_near, q_new):
            return TRAPPED, -1
        i_new = tree.add(q_new, i_near)
        return (REACHED if reached else ADVANCED), i_new

    def _connect(self, tree, q_target, problem):
        status, i = ADVANCED, -1
        while status == ADVANCED:
            status, i = self._extend(tree, q_target, problem)
        return status, i

    def plan(self, start, goals, problem, max_iterations=cfg.MAX_ITERATIONS, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        t0 = time.perf_counter()
        start = np.asarray(start, float)
        if not problem.is_valid(start):
            return PlannerResult(False, message="Start configuration is in collision.", time=time.perf_counter() - t0)
        goals = [np.asarray(g, float) for g in goals]
        goal_ids = [k for k, g in enumerate(goals) if problem.is_valid(g)]
        if not goal_ids:
            return PlannerResult(False, message="Every goal configuration is in collision.",
                                 time=time.perf_counter() - t0)

        # Shortcut: maybe the straight joint-space motion is already free.
        for k in goal_ids:
            if problem.edge_valid(start, goals[k]):
                return PlannerResult(True, [start, goals[k]], "Direct motion is collision-free.",
                                     0, 2, time.perf_counter() - t0, k,
                                     [(np.array([start]), np.array([-1]), "start"),
                                      (np.array([goals[k]]), np.array([-1]), "goal")])

        tree_start = Tree([start], problem.dim)
        tree_goal = Tree([goals[k] for k in goal_ids], problem.dim)
        tree_a, tree_b = tree_start, tree_goal
        for it in range(1, max_iterations + 1):
            if problem.cancelled():
                return self._result(False, "Planning cancelled.", it, tree_start, tree_goal, t0)
            q_rand = problem.sample(rng)
            status, i_new = self._extend(tree_a, q_rand, problem)
            if status != TRAPPED:
                q_new = tree_a.nodes[i_new]
                status_b, i_b = self._connect(tree_b, q_new, problem)
                if status_b == REACHED:
                    # Join the two branches into one start -> goal path.
                    i_s, i_g = (i_new, i_b) if tree_a is tree_start else (i_b, i_new)
                    path = tree_start.branch(i_s)[::-1] + tree_goal.branch(i_g)[1:]
                    res = self._result(True, f"Trees connected after {it} samples.", it,
                                       tree_start, tree_goal, t0)
                    res.path = path
                    res.goal_index = goal_ids[tree_goal.root_of[i_g]]
                    return res
            tree_a, tree_b = tree_b, tree_a
        return self._result(False, f"No path found in {max_iterations} samples "
                                   "(try more iterations, or move obstacles to widen the passage).",
                            max_iterations, tree_start, tree_goal, t0)

    @staticmethod
    def _result(success, message, iterations, tree_start, tree_goal, t0):
        return PlannerResult(success, [], message, iterations, tree_start.size + tree_goal.size,
                             time.perf_counter() - t0, -1,
                             [tree_start.export("start"), tree_goal.export("goal")])


# --------------------------------------------------------------------------- #
# Path post-processing
# --------------------------------------------------------------------------- #
def path_length(path):
    path = np.asarray(path, float)
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0


def shortcut_path(path, edge_valid, iterations, rng):
    """Random shortcutting: repeatedly pick two points along the path and, if
    the straight joint-space motion between them is collision-free, replace
    the part of the path in between. Removes the RRT's random zig-zags."""
    path = [np.asarray(q, float) for q in path]
    for _ in range(iterations):
        if len(path) < 3:
            break
        seg = np.linalg.norm(np.diff(np.array(path), axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total < 1e-9:
            break
        s1, s2 = np.sort(rng.uniform(0.0, total, 2))
        i1 = int(np.searchsorted(cum, s1, side="right") - 1)
        i2 = int(np.searchsorted(cum, s2, side="right") - 1)
        i1, i2 = min(i1, len(path) - 2), min(i2, len(path) - 2)
        if i1 == i2:
            continue
        q1 = path[i1] + (path[i1 + 1] - path[i1]) * (s1 - cum[i1]) / max(seg[i1], 1e-12)
        q2 = path[i2] + (path[i2 + 1] - path[i2]) * (s2 - cum[i2]) / max(seg[i2], 1e-12)
        if edge_valid(q1, q2):
            path = path[:i1 + 1] + [q1, q2] + path[i2 + 1:]
    # drop near-duplicate waypoints created by shortcutting
    clean = [path[0]]
    for q in path[1:]:
        if np.linalg.norm(q - clean[-1]) > 1e-6:
            clean.append(q)
    return clean


def make_planner(backend=None):
    """Planner factory. backend: "rrt_connect" (built-in) or "ompl" (needs `pip install ompl`)."""
    backend = backend or cfg.PLANNER_BACKEND
    if backend == "ompl":
        from ompl_backend import OMPLPlanner
        return OMPLPlanner()
    if backend != "rrt_connect":
        raise ValueError(f"Unknown planner backend {backend!r}")
    return RRTConnect()
