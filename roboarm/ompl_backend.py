"""
ompl_backend.py - OPTIONAL: use the OMPL library (ompl.kavrakilab.org) as the
motion planner instead of the built-in RRT-Connect.

OMPL is the planning library used inside MoveIt 2. Since version 1.7 it can be
installed with pip, without ROS:

    pip install ompl          (macOS arm64 / Linux wheels, Python 3.10+)

Then run    python main.py --planner ompl
or set      PLANNER_BACKEND = "ompl"   in config.py
Any OMPL geometric planner can be tried, e.g. --ompl-algorithm PRM, BITstar, KPIECE1.

How the adapter works
---------------------
OMPL only needs the same two answers our own planner uses:
  * state validity  -> PlanningProblem.is_valid(q)
  * motion validity -> PlanningProblem.edge_valid(qa, qb)
We register them as OMPL's StateValidityChecker and a custom MotionValidator,
so OMPL plans in exactly the same configuration space, with exactly the same
collision model, as the built-in planner. That makes the comparison fair.

Note: when Python exits, the OMPL 2.0 wheel may print "nanobind: leaked ..."
messages. They are harmless (a known issue of the bindings).
"""
import time

import numpy as np

import config as cfg
from planner import MotionPlanner, PlannerResult


def _import_ompl():
    try:
        from ompl import base as ob, geometric as og, util as ou
    except ImportError as exc:
        raise ImportError("The OMPL backend needs the 'ompl' package: run  pip install ompl  "
                          "(or use the built-in planner: PLANNER_BACKEND = 'rrt_connect').") from exc
    ou.setLogLevel(ou.LOG_WARN)
    return ob, og, ou


def _to_array(state, n):
    return np.array([state[i] for i in range(n)])


class OMPLPlanner(MotionPlanner):
    def __init__(self, algorithm=None, step_size=cfg.RRT_STEP_SIZE, time_limit=cfg.OMPL_TIME_LIMIT):
        algorithm = algorithm or cfg.OMPL_ALGORITHM
        self.algorithm = algorithm
        self.step_size = step_size
        self.time_limit = time_limit
        self.name = f"OMPL {algorithm}"

    def plan(self, start, goals, problem, max_iterations=None, rng=None):
        ob, og, ou = _import_ompl()
        t0 = time.perf_counter()
        n = problem.dim
        start = np.asarray(start, float)
        if not problem.is_valid(start):
            return PlannerResult(False, message="Start configuration is in collision.")
        goals = [np.asarray(g, float) for g in goals]
        goal_ids = [k for k, g in enumerate(goals) if problem.is_valid(g)]
        if not goal_ids:
            return PlannerResult(False, message="Every goal configuration is in collision.")

        space = ob.RealVectorStateSpace(n)
        bounds = ob.RealVectorBounds(n)
        for i in range(n):
            bounds.setLow(i, float(problem.lower[i]))
            bounds.setHigh(i, float(problem.upper[i]))
        space.setBounds(bounds)
        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(lambda s: bool(problem.is_valid(_to_array(s, n))))

        class MotionValidator(ob.MotionValidator):
            def checkMotion(self, s1, s2, *last_valid):
                return bool(problem.edge_valid(_to_array(s1, n), _to_array(s2, n)))

        validator = MotionValidator(si)          # keep a reference while planning
        si.setMotionValidator(validator)
        si.setup()

        def make_state(q):
            s = space.allocState()
            for i in range(n):
                s[i] = float(q[i])
            return s

        pdef = ob.ProblemDefinition(si)
        pdef.addStartState(make_state(start))
        goal = ob.GoalStates(si)
        for k in goal_ids:
            goal.addState(make_state(goals[k]))
        pdef.setGoal(goal)

        planner = getattr(og, self.algorithm)(si)
        if hasattr(planner, "setRange"):
            planner.setRange(self.step_size)
        planner.setProblemDefinition(pdef)
        planner.setup()
        ptc = ob.plannerOrTerminationCondition(
            ob.PlannerTerminationCondition(lambda: bool(problem.cancelled())),
            ob.timedPlannerTerminationCondition(float(self.time_limit)))
        planner.solve(ptc)

        trees = self._trees(ob, si, planner, n)
        nodes = sum(len(t[0]) for t in trees)
        elapsed = time.perf_counter() - t0
        if not pdef.hasExactSolution():
            msg = "Planning cancelled." if problem.cancelled() else \
                f"OMPL {self.algorithm} found no exact path within {self.time_limit:.0f} s."
            return PlannerResult(False, [], msg, nodes, nodes, elapsed, -1, trees)
        path_obj = pdef.getSolutionPath()
        path = [_to_array(path_obj.getState(i), n) for i in range(path_obj.getStateCount())]
        end = path[-1]
        k = goal_ids[int(np.argmin([np.linalg.norm(goals[j] - end) for j in goal_ids]))]
        path[-1] = goals[k].copy()
        return PlannerResult(True, path, f"OMPL {self.algorithm} found a path.", nodes, nodes, elapsed, k, trees)

    @staticmethod
    def _trees(ob, si, planner, n):
        """Planner graph as [(nodes, parents, label)] for drawing (best effort)."""
        try:
            pd = ob.PlannerData(si)
            planner.getPlannerData(pd)
            count = pd.numVertices()
            nodes = np.array([_to_array(pd.getVertex(i).getState(), n) for i in range(count)])
            tags = np.array([pd.getVertex(i).getTag() for i in range(count)])
            parents = np.full(count, -1)
            for i in range(count):
                for j in pd.getEdges(i):
                    if parents[j] == -1:
                        parents[j] = i
            out = []
            for tag, label in ((1, "start"), (2, "goal")):
                idx = np.nonzero(tags == tag)[0] if tag in tags else np.arange(count) * (tag == 1)
                if len(idx) == 0:
                    continue
                remap = {int(old): new for new, old in enumerate(idx)}
                par = np.array([remap.get(int(parents[i]), -1) for i in idx])
                out.append((nodes[idx], par, label))
            return out
        except Exception:
            return []
