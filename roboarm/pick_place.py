"""
pick_place.py - turns "move the object from A to B" into a sequence of
collision-free robot motions (the "task plan").

Sequence (each move is a list of joint configurations):
  1. TRANSIT   current pose -> pre-grasp pose above A           (RRT-Connect)
  2. APPROACH  straight down onto the object                    (Cartesian line + IK)
  3. GRASP     close the fingers, attach the object
  4. RETREAT   straight up, now carrying the object
  5. CARRY     pre-grasp above A -> pre-place above B           (RRT-Connect, object included)
  6. APPROACH  straight down to the place pose
  7. RELEASE   open the fingers, the object stays on table B
  8. RETREAT   straight up with the empty gripper
  9. TRANSIT   back to the home pose                            (RRT-Connect, optional)

Inverse kinematics provides several candidate grasp poses (e.g. "elbow up"
and "elbow down"); every candidate whose straight approach is collision-free
becomes a goal for RRT-Connect, which simply uses whichever it reaches first.
"""
import time
from dataclasses import dataclass, field

import numpy as np

import config as cfg
from collision import CollisionChecker
from planner import PlanningProblem, make_planner, path_length, shortcut_path
from robot import HeldObject, rotation


@dataclass
class Move:
    kind: str                 # transit / approach / retreat / carry
    label: str
    path: np.ndarray          # (K, N) joint-space waypoints
    held: HeldObject = None   # object carried during this move (None = empty gripper)
    raw_path: np.ndarray = None   # RRT path before smoothing (if any)

    @property
    def speed_factor(self):
        return cfg.APPROACH_SPEED_FACTOR if self.kind in ("approach", "retreat") else 1.0


@dataclass
class GripperAction:
    action: str               # "close" or "open"
    station: str
    label: str = ""


@dataclass
class TaskPlan:
    success: bool
    message: str
    steps: list = field(default_factory=list)
    start_q: np.ndarray = None
    trees: list = field(default_factory=list)    # [(nodes, parents, label, held)] for drawing
    metrics: dict = field(default_factory=dict)
    target_station: str = None

    @property
    def moves(self):
        return [s for s in self.steps if isinstance(s, Move)]


class PlanningFailed(Exception):
    pass


class TaskPlanner:
    def __init__(self, robot, env, max_iterations=cfg.MAX_ITERATIONS, rng=None,
                 cancel_event=None, shortcut_iterations=cfg.SHORTCUT_ITERATIONS,
                 return_home=cfg.RETURN_HOME, backend=None):
        self.robot = robot
        self.env = env
        self.max_iterations = int(max_iterations)
        self.rng = rng if rng is not None else np.random.default_rng(cfg.RANDOM_SEED)
        self.cancel_event = cancel_event
        self.shortcut_iterations = shortcut_iterations
        self.return_home = return_home
        self.planner = make_planner(backend)
        self.grasp_angle = np.radians(cfg.GRASP_ANGLE_DEG)
        self.home = np.radians(cfg.HOME_CONFIG_DEG[:robot.n_joints])
        self._stats = {"iterations": 0, "nodes": 0, "rrt_time": 0.0, "queries": 0}
        self._checkers = []
        self._trees = []

    # ------------------------------------------------------------ helpers
    def _checker(self, env, held=None, gripper_closed=None, include_free_object=True,
                 margin=cfg.SAFETY_MARGIN):
        cc = CollisionChecker(self.robot, env, held, gripper_closed, include_free_object, margin)
        self._checkers.append(cc)
        return cc

    def _describe_collision(self, checker, q):
        pairs = checker.report(q)["pairs"]
        return ", ".join(pairs[:3]) if pairs else "unknown"

    def _rrt(self, start, goals, checker, label, held, fallback=None):
        """Run the motion planner. `checker` uses the larger free-space margin;
        if the start or every goal is closer than that to an obstacle (e.g. the
        arm was stopped next to a table) the tighter `fallback` checker is used."""
        if fallback is not None and (not checker.is_valid(start) or
                                     not any(checker.is_valid(g) for g in goals)):
            checker = fallback
        problem = PlanningProblem(checker, self.robot.lower, self.robot.upper, self.cancel_event)
        res = self.planner.plan(start, goals, problem, self.max_iterations, self.rng)
        self._stats["iterations"] += res.iterations
        self._stats["nodes"] += res.nodes
        self._stats["rrt_time"] += res.time
        self._stats["queries"] += 1
        for nodes, parents, tree_label in res.trees:
            self._trees.append((nodes, parents, f"{label}:{tree_label}", held))
        if not res.success:
            raise PlanningFailed(f"{label}: {res.message}")
        raw = np.array(res.path)
        if self.shortcut_iterations > 0:
            smooth = np.array(shortcut_path(res.path, checker.edge_valid, self.shortcut_iterations, self.rng))
        else:
            smooth = raw
        return smooth, raw, res.goal_index

    def _station_candidates(self, station, object_center, held_rel, q_seed,
                            down_checker, up_checker, what):
        """IK poses at a station plus their vertical approach paths.

        down_checker validates the approach (moving down), up_checker the
        retreat (moving up after the gripper action). Returns a list of
        (pre_pose, approach_path, retreat_path).
        """
        # Where must the TCP be so that the held object ends up at object_center?
        rel_pos, rel_angle = held_rel
        tool_angle = self.grasp_angle
        tcp_target = object_center - rotation(tool_angle) @ rel_pos
        solutions = self.robot.ik_solutions(tcp_target, tool_angle, self.rng, q_seed=q_seed)
        if not solutions:
            raise PlanningFailed(f"{what} at {station} is out of reach: no joint angles put the gripper "
                                 f"there within the joint limits (check link lengths).")
        lift = np.array([0.0, cfg.APPROACH_HEIGHT])
        candidates, reasons = [], []
        for q in solutions:
            if not down_checker.is_valid(q):
                reasons.append(self._describe_collision(down_checker, q))
                continue
            line = self.robot.cartesian_line(q, tcp_target + lift, tool_angle)
            if line is None:
                reasons.append("no straight vertical approach")
                continue
            approach = line[::-1]
            if not (down_checker.path_valid(approach) and up_checker.path_valid(line)):
                reasons.append("vertical approach blocked")
                continue
            candidates.append((line[-1], approach, line))
        if not candidates:
            raise PlanningFailed(f"No collision-free {what.lower()} pose at {station} "
                                 f"({'; '.join(sorted(set(reasons))) or 'all IK solutions invalid'}).")
        return candidates

    # ------------------------------------------------------------ main entry
    def plan_pick_and_place(self, q_start, held=None, pick=None, place=None):
        """Plan the full task from configuration q_start.

        If `held` is given the robot is already carrying the object (replanning
        in the middle of a task) and the pick phase is skipped.
        """
        t0 = time.perf_counter()
        robot, env = self.robot, self.env
        q_start = np.asarray(q_start, float)
        steps = []
        try:
            if held is None:
                pick = pick or env.object_station
                if pick is None:
                    raise PlanningFailed("The object is not on a table.")
                place = place or ("B" if pick == "A" else "A")
                q_lifted, held = self._plan_pick(q_start, pick, steps)
            else:
                place = place or ("B" if held.origin == "A" else "A")
                q_lifted = q_start
            self._plan_place(q_lifted, held, place, steps)
        except PlanningFailed as exc:
            plan = TaskPlan(False, str(exc), steps, q_start, self._trees, target_station=place)
            plan.metrics = self._metrics(plan, time.perf_counter() - t0)
            return plan
        plan = TaskPlan(True, f"Plan found: pick-and-place to {place}.", steps, q_start, self._trees,
                        target_station=place)
        plan.metrics = self._metrics(plan, time.perf_counter() - t0)
        return plan

    def _plan_pick(self, q_start, pick, steps):
        robot, env = self.robot, self.env
        empty = self._checker(env)                                  # object on table = obstacle
        empty_free = self._checker(env, margin=cfg.PLANNING_MARGIN)
        if not empty.is_valid(q_start):
            raise PlanningFailed("The arm's current pose is in collision "
                                 f"({self._describe_collision(empty, q_start)}). Move the obstacle or the arm.")
        center = env.object_center
        half = env.object_half_size()
        # Held-object transform (exact one is measured after IK, at the grasp pose)
        provisional = HeldObject(half, np.zeros(2), 0.0, pick)
        carrying_env = env.copy()
        carrying_env.object_station = None                          # the object leaves the table
        holding = self._checker(carrying_env, provisional)
        cands = self._station_candidates(pick, center, (np.zeros(2), 0.0), q_start,
                                         down_checker=empty, up_checker=holding, what="Grasp")
        path, raw, k = self._rrt(q_start, [c[0] for c in cands], empty_free, f"Transit to {pick}",
                                 None, fallback=empty)
        pre, approach, retreat = cands[k]
        # Exact object pose relative to the gripper at the grasp configuration.
        q_grasp = approach[-1]
        tcp, angle = robot.tcp_pose(q_grasp)
        rel = rotation(-angle) @ (center - tcp)
        held = HeldObject(half, rel, env.object_angle - angle, pick)
        steps.append(Move("transit", f"Move above {pick}", path, None, raw))
        steps.append(Move("approach", f"Descend to object at {pick}", approach, None))
        steps.append(GripperAction("close", pick, f"Grasp object at {pick}"))
        steps.append(Move("retreat", f"Lift object from {pick}", retreat, held))
        return retreat[-1], held

    def _plan_place(self, q_from, held, place, steps):
        robot, env = self.robot, self.env
        carrying_env = env.copy()
        carrying_env.object_station = None
        holding = self._checker(carrying_env, held)
        if not holding.is_valid(q_from):
            raise PlanningFailed("The arm (with the object) is in collision at its current pose "
                                 f"({self._describe_collision(holding, q_from)}).")
        st = env.stations[place]
        target_center = st.object_rest_center(env.object_size)
        # After release the object rests on table B and becomes an obstacle.
        placed_env = env.copy()
        placed_env.place_object_at(place)
        released = self._checker(placed_env, None, gripper_closed=False)
        cands = self._station_candidates(place, target_center, (held.rel_pos, held.rel_angle), q_from,
                                         down_checker=holding, up_checker=released, what="Place")
        holding_free = self._checker(carrying_env, held, margin=cfg.PLANNING_MARGIN)
        path, raw, k = self._rrt(q_from, [c[0] for c in cands], holding_free, f"Carry to {place}", held,
                                 fallback=holding)
        pre, approach, retreat = cands[k]
        steps.append(Move("carry", f"Carry object to {place}", path, held, raw))
        steps.append(Move("approach", f"Lower object onto {place}", approach, held))
        steps.append(GripperAction("open", place, f"Release object at {place}"))
        steps.append(Move("retreat", f"Retreat from {place}", retreat, None))
        if self.return_home:
            released_free = self._checker(placed_env, None, gripper_closed=False, margin=cfg.PLANNING_MARGIN)
            if released.is_valid(self.home):
                path, raw, _ = self._rrt(retreat[-1], [self.home], released_free, "Return home", None,
                                         fallback=released)
                steps.append(Move("transit", "Return home", path, None, raw))

    def plan_joint_goal(self, q_start, q_goal, held=None):
        """Stage C helper: plan a single motion between two joint configurations."""
        t0 = time.perf_counter()
        env = self.env.copy()
        if held is not None:
            env.object_station = None
        checker = self._checker(env, held)
        free = self._checker(env, held, margin=cfg.PLANNING_MARGIN)
        try:
            for name, q in (("Start", q_start), ("Goal", q_goal)):
                if not checker.is_valid(q):
                    raise PlanningFailed(f"{name} configuration is invalid ({self._describe_collision(checker, q)}).")
            path, raw, _ = self._rrt(q_start, [q_goal], free, "Joint goal", held, fallback=checker)
            steps = [Move("transit" if held is None else "carry", "Move to joint goal", path, held, raw)]
            plan = TaskPlan(True, "Plan found between the two configurations.", steps, np.asarray(q_start), self._trees)
        except PlanningFailed as exc:
            plan = TaskPlan(False, str(exc), [], np.asarray(q_start), self._trees)
        plan.metrics = self._metrics(plan, time.perf_counter() - t0)
        return plan

    # ------------------------------------------------------------ metrics
    def _metrics(self, plan, total_time):
        from metrics import trajectory_metrics
        m = {
            "success": plan.success,
            "planner": self.planner.name,
            "planning_time_s": total_time,
            "rrt_time_s": self._stats["rrt_time"],
            "rrt_queries": self._stats["queries"],
            "samples": self._stats["iterations"],
            "tree_nodes": self._stats["nodes"],
            "collision_checks": sum(c.checks for c in self._checkers),
        }
        if plan.moves:
            m.update(trajectory_metrics(self.robot, self.env, plan.moves, plan.target_station))
        return m
