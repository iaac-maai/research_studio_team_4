"""
pick_place.py - plan a pick-and-place task and execute it in the physics simulation.

Same sequence as roboarm/pick_place.py:
  1. TRANSIT   current pose -> above the box                 (RRT-Connect)
  2. APPROACH  straight down                                 (Cartesian line + IK)
  3. GRASP     close the fingers (friction grip)
  4. RETREAT   straight up with the box
  5. CARRY     -> above the other table                      (RRT-Connect, box included)
  6. APPROACH  straight down
  7. RELEASE   open the fingers
  8. RETREAT   straight up
  9. TRANSIT   back home                                     (RRT-Connect)

Reused from roboarm/ without copying: PlanarArm (IK, straight lines, reach),
RRTConnect, PlanningProblem, shortcut_path, path_length and all planner
settings in config.py. Collision checks come from MuJoCo (planning.py).

Planning is kinematic; execution is physical. PlanExecutor feeds the plan to
RobotSim.step() one control step at a time, so the box can still slip, and
collisions during execution are recorded.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from roboarm_mujoco import paths  # noqa: F401
from roboarm_mujoco import scene
from roboarm_mujoco import settings as S
from roboarm_mujoco.planning import MujocoChecker
from roboarm_mujoco.trajectory import resample
import config as rcfg
from geometry import rotation
from planner import PlanningProblem, make_planner, path_length, shortcut_path
from robot import HeldObject, PlanarArm


@dataclass
class Move:
    kind: str                 # transit / approach / retreat / carry
    label: str
    path: np.ndarray          # (K, 3) joint waypoints
    raw_path: np.ndarray = None

    @property
    def speed_factor(self):
        return rcfg.APPROACH_SPEED_FACTOR if self.kind in ("approach", "retreat") else 1.0


@dataclass
class GripperAction:
    closed: bool
    label: str


@dataclass
class TaskPlan:
    success: bool
    message: str
    steps: list = field(default_factory=list)
    pick_table: str = None
    place_table: str = None
    place_center: np.ndarray = None
    metrics: dict = field(default_factory=dict)

    @property
    def moves(self):
        return [s for s in self.steps if isinstance(s, Move)]


class PlanningFailed(Exception):
    pass


def table_top(row):
    return row["y"] + row["height"] / 2


def table_under(layout, box_pos, tol=0.06):
    """The table the box is resting on (or None)."""
    obj = scene.object_row(layout)
    bottom = box_pos[1] - obj["height"] / 2
    for row in scene.table_rows(layout):
        if abs(row["x"] - box_pos[0]) <= row["width"] / 2 and abs(bottom - table_top(row)) <= tol:
            return row
    return None


class TaskPlanner:
    def __init__(self, layout, rng=None, max_iterations=rcfg.MAX_ITERATIONS,
                 shortcut_iterations=rcfg.SHORTCUT_ITERATIONS, return_home=rcfg.RETURN_HOME):
        self.layout = layout
        self.robot = PlanarArm()
        self.rng = rng if rng is not None else np.random.default_rng(rcfg.RANDOM_SEED)
        self.max_iterations = int(max_iterations)
        self.shortcut_iterations = shortcut_iterations
        self.return_home = return_home
        self.planner = make_planner("rrt_connect")
        self.grasp_angle = np.radians(rcfg.GRASP_ANGLE_DEG)
        self.home = np.radians(rcfg.HOME_CONFIG_DEG)
        obj = scene.object_row(layout)
        self.half = (obj["width"] / 2, obj["height"] / 2)
        self.robot.finger_closed = obj["width"] / 2 + rcfg.FINGER_RADIUS + rcfg.FINGER_CLEARANCE
        self._checkers = []
        self._stats = {"samples": 0, "nodes": 0, "rrt_time_s": 0.0, "queries": 0}

    # ------------------------------------------------------------ helpers
    def _checker(self, margin=rcfg.SAFETY_MARGIN, **kw):
        c = MujocoChecker(self.layout, self.robot, margin, **kw)
        self._checkers.append(c)
        return c

    def _rrt(self, start, goals, checker, label, fallback=None):
        """RRT-Connect + shortcutting. Uses the tighter `fallback` checker if the
        start or every goal is too close to something for the free-space margin."""
        if fallback is not None and (not checker.is_valid(start) or not any(checker.is_valid(g) for g in goals)):
            checker = fallback
        problem = PlanningProblem(checker, self.robot.lower, self.robot.upper)
        res = self.planner.plan(start, goals, problem, self.max_iterations, self.rng)
        self._stats["samples"] += res.iterations
        self._stats["nodes"] += res.nodes
        self._stats["rrt_time_s"] += res.time
        self._stats["queries"] += 1
        if not res.success:
            raise PlanningFailed(f"{label}: {res.message}")
        raw = np.array(res.path)
        smooth = np.array(shortcut_path(res.path, checker.edge_valid, self.shortcut_iterations, self.rng)) \
            if self.shortcut_iterations > 0 else raw
        return smooth, raw, res.goal_index

    def _station(self, tcp_target, q_seed, down, up, what, where):
        """IK poses at a station with collision-free vertical approach and retreat.
        Returns [(pre_pose, approach_path, retreat_path)]."""
        sols = self.robot.ik_solutions(tcp_target, self.grasp_angle, self.rng, q_seed=q_seed)
        if not sols:
            raise PlanningFailed(f"{what} at {where} is out of reach (no IK solution within the joint limits).")
        lift = np.array([0.0, rcfg.APPROACH_HEIGHT])
        cands, reasons = [], []
        for q in sols:
            if not down.is_valid(q):
                reasons.append("pose collides: " + ", ".join(down.report(q)[:2]))
                continue
            line = self.robot.cartesian_line(q, tcp_target + lift, self.grasp_angle)
            if line is None:
                reasons.append("no straight vertical approach")
                continue
            if not (down.path_valid(line[::-1]) and up.path_valid(line)):
                reasons.append("vertical approach blocked")
                continue
            cands.append((line[-1], line[::-1], line))
        if not cands:
            raise PlanningFailed(f"No collision-free {what.lower()} pose at {where} "
                                 f"({'; '.join(sorted(set(reasons)))}).")
        return cands

    # ------------------------------------------------------------ main entry
    def plan(self, q_start, box_pos, box_angle=0.0, place=None):
        """Plan moving the box from the table it is on to the other table
        (or to the table named `place`)."""
        t0 = time.perf_counter()
        q_start, box_pos = np.asarray(q_start, float), np.asarray(box_pos, float)
        plan = TaskPlan(False, "")
        try:
            pick_row = table_under(self.layout, box_pos)
            if pick_row is None:
                raise PlanningFailed("The box is not resting on a table.")
            others = [r for r in scene.table_rows(self.layout) if r["name"] != pick_row["name"]]
            place_row = next((r for r in others if r["name"] == place), None) if place else None
            place_row = place_row or next((r for r in others if r["name"] in ("Table A", "Table B")), others[0])
            plan.pick_table, plan.place_table = pick_row["name"], place_row["name"]
            pick_geom = scene.unique_names(self.layout)[self.layout.index(pick_row)]
            self._plan_pick(plan, q_start, box_pos, box_angle, pick_geom)
            self._plan_place(plan, place_row)
            plan.success = True
            plan.message = f"Plan found: {plan.pick_table} -> {plan.place_table}."
        except PlanningFailed as exc:
            plan.message = str(exc)
        plan.metrics = self._metrics(plan, time.perf_counter() - t0)
        return plan

    def _plan_pick(self, plan, q_start, box_pos, box_angle, pick_geom):
        resting = (box_pos, box_angle)
        empty = self._checker(object_pose=resting)
        empty_free = self._checker(rcfg.PLANNING_MARGIN, object_pose=resting)
        if not empty.is_valid(q_start):
            raise PlanningFailed(f"The arm's current pose is in collision ({', '.join(empty.report(q_start)[:3])}).")
        provisional = HeldObject(self.half, np.zeros(2), 0.0, plan.pick_table)
        lifting = self._checker(held=provisional, gripper_closed=True, ignore_table=pick_geom)
        cands = self._station(box_pos, q_start, empty, lifting, "Grasp", plan.pick_table)
        path, raw, k = self._rrt(q_start, [c[0] for c in cands], empty_free, f"Transit to {plan.pick_table}", empty)
        _, approach, retreat = cands[k]
        tcp, angle = self.robot.tcp_pose(approach[-1])
        self.held = HeldObject(self.half, rotation(-angle) @ (box_pos - tcp), box_angle - angle, plan.pick_table)
        self._pick_geom = pick_geom
        plan.steps += [Move("transit", f"Move above {plan.pick_table}", path, raw),
                       Move("approach", "Descend to the box", approach),
                       GripperAction(True, "Grasp the box"),
                       Move("retreat", f"Lift the box off {plan.pick_table}", retreat)]
        self._q = retreat[-1]

    def _plan_place(self, plan, place_row):
        held = self.held
        holding = self._checker(held=held, gripper_closed=True, ignore_table=self._pick_geom)
        holding_free = self._checker(rcfg.PLANNING_MARGIN, held=held, gripper_closed=True)
        target = np.array([place_row["x"], table_top(place_row) + self.half[1] + S.PLACE_DROP_GAP])
        plan.place_center = target
        tcp_target = target - rotation(self.grasp_angle) @ held.rel_pos
        released = self._checker(object_pose=(target, 0.0))
        placing = self._checker(held=held, gripper_closed=True)
        cands = self._station(tcp_target, self._q, placing, released, "Place", plan.place_table)
        path, raw, k = self._rrt(self._q, [c[0] for c in cands], holding_free, f"Carry to {plan.place_table}",
                                 holding)
        _, approach, retreat = cands[k]
        plan.steps += [Move("carry", f"Carry the box to {plan.place_table}", path, raw),
                       Move("approach", f"Lower the box onto {plan.place_table}", approach),
                       GripperAction(False, "Release the box"),
                       Move("retreat", f"Retreat from {plan.place_table}", retreat)]
        if self.return_home and released.is_valid(self.home):
            released_free = self._checker(rcfg.PLANNING_MARGIN, object_pose=(target, 0.0))
            path, raw, _ = self._rrt(retreat[-1], [self.home], released_free, "Return home", released)
            plan.steps.append(Move("transit", "Return home", path, raw))

    def _metrics(self, plan, total):
        m = {"success": plan.success, "planner": self.planner.name, "planning_time_s": total,
             **self._stats, "collision_checks": sum(c.checks for c in self._checkers)}
        if plan.moves:
            joints = np.concatenate([mv.path for mv in plan.moves])
            tcp = self.robot.tcp(joints)
            m["joint_path_rad"] = path_length(joints)
            m["raw_joint_path_rad"] = sum(path_length(mv.raw_path if mv.raw_path is not None else mv.path)
                                          for mv in plan.moves)
            m["tcp_path_m"] = float(np.sum(np.linalg.norm(np.diff(tcp, axis=0), axis=1)))
        return m


def plan_for(sim, rng=None, **kw):
    """Plan from the RobotSim's current state."""
    s = sim.get_state()
    return TaskPlanner(sim.layout, rng=rng if rng is not None else sim.rng, **kw) \
        .plan(s["q"], s["box_pos"], s["box_angle"])


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
class PlanExecutor:
    """Plays a TaskPlan on a RobotSim, one control step per tick()."""

    def __init__(self, sim, plan, speed=1.0, settle_time=0.5):
        self.sim = sim
        self.plan = plan
        self.commands = []                    # ("q", target, label) or ("grip", closed, label)
        dt = S.CONTROL_DT
        settle = int(round(S.GRIPPER_SETTLE_TIME / dt))
        for step in plan.steps:
            if isinstance(step, Move):
                ds = rcfg.JOINT_SPEED * speed * step.speed_factor * dt
                self.commands += [("q", q, step.label) for q in resample(step.path, ds)]
            else:
                self.commands.append(("grip", step.closed, step.label))
                self.commands += [("hold", None, step.label)] * settle
        self.commands += [("hold", None, "Done")] * int(round(settle_time / dt))
        self.index = 0
        self.collisions = []                  # (time, pairs) seen during execution
        self.label = ""

    @property
    def done(self):
        return self.index >= len(self.commands)

    @property
    def progress(self):
        return self.index / max(1, len(self.commands))

    def tick(self):
        if self.done:
            return self.sim.step(self.sim.target)
        kind, value, self.label = self.commands[self.index]
        self.index += 1
        if kind == "grip":
            self.sim.set_gripper(value)
            state = self.sim.step(self.sim.target)
        elif kind == "q":
            state = self.sim.step(value)
        else:
            state = self.sim.step(self.sim.target)
        pairs = self.sim.collisions()
        if pairs:
            self.collisions.append((self.sim.time, pairs))
        return state

    def run(self):
        while not self.done:
            state = self.tick()
        return state
