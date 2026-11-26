"""
simulation.py - the state of the running simulator: robot pose, object,
current plan and its animation. No drawing happens here (see render.py).

States
  idle      nothing planned yet (manual jogging allowed)
  planning  a background thread is computing a plan
  ready     a valid plan is waiting to be executed
  running   the plan is being animated
  paused    animation paused (or stopped because something changed)
  done      the plan finished
  failed    the last planning attempt failed
  attempt   planning failed; the arm is showing a naive straight-line try
            ("Oops!") and then returns to where it started (see attempt.py)

Planning runs in a background thread so the window stays responsive. The
thread works on COPIES of the robot and environment, so dragging things
around cannot corrupt a plan in progress.
"""
import queue
import threading
import time

import numpy as np

import config as cfg
from attempt import naive_attempt
from collision import CollisionChecker
from environment import Environment
from metrics import append_log, densify
from pick_place import GripperAction, Move, TaskPlanner
from robot import HeldObject, PlanarArm, wrap_angle


class Simulation:
    def __init__(self, seed=cfg.RANDOM_SEED, backend=None):
        self.backend = backend or cfg.PLANNER_BACKEND
        self.robot = PlanarArm()
        self.env = Environment.default()
        self.home = np.radians(cfg.HOME_CONFIG_DEG[:self.robot.n_joints])
        self.q = self.home.copy()
        self.held = None
        self.finger = self.robot.finger_open
        self.rng = np.random.default_rng(seed)

        self.state = "idle"
        self.status = "Press PLAN (P) to compute a pick-and-place motion."
        self.status_level = "info"
        self.plan = None
        self.plan_valid = False
        self.scene_version = 0          # bumps when obstacles / robot geometry change
        self.metrics = {}
        self.visual = None              # cached drawing data for the current plan

        self.speed = 1.0
        self.max_iterations = cfg.MAX_ITERATIONS
        self.smoothing = cfg.SHORTCUT_ITERATIONS > 0
        self.ik_compare = cfg.IK_SELECTION == "compare"
        self.attempt = None             # naive "try anyway" motion after a failed plan
        self.attempt_phase = ""         # "out" -> "oops" -> "back"
        self._att_s = 0.0
        self._att_cum = None
        self._att_timer = 0.0
        self._att_reason = ""
        self.selected_joint = 0
        self.custom_goal = None

        self.step_index = 0
        self.step_s = 0.0
        self._cum = None
        self._worker = None
        self._results = queue.Queue()
        self._cancel = threading.Event()
        self._job = 0                   # results of older (reset) planning jobs are ignored
        self.planning_started = 0.0
        self.planning_what = ""

    # ------------------------------------------------------------ helpers
    def set_status(self, text, level="info"):
        self.status, self.status_level = text, level

    @property
    def busy(self):
        return self.state == "planning"

    def object_pose(self):
        """(centre, angle) of the object, whether resting or carried."""
        if self.held is not None:
            c, a = self.robot.object_pose(self.q, self.held)
            return c, float(a)
        return self.env.object_center, self.env.object_angle

    def current_checker(self):
        closed = self.held is not None
        return CollisionChecker(self.robot, self.env, self.held, gripper_closed=closed)

    def current_report(self):
        return self.current_checker().report(self.q)

    def invalidate(self, reason):
        """Something changed (obstacle, link length, manual move): the plan is stale."""
        if self.plan is not None and self.plan_valid:
            self.plan_valid = False
            if self.state in ("ready", "running", "paused"):
                self.state = "paused" if self.state == "running" or self.step_index > 0 else "idle"
            self.set_status(f"{reason} The plan is no longer valid - press PLAN to replan from here.", "warn")
        elif self.state == "done":
            self.state = "idle"

    # ------------------------------------------------------------ editing
    def on_environment_changed(self):
        self.scene_version += 1
        self.invalidate("Environment changed.")

    def safe_home(self):
        """(home pose, adjusted) for the current link lengths and obstacles.

        HOME_CONFIG_DEG is a fixed set of joint angles. With other link lengths
        that pose can put the arm inside an obstacle, so Reset would always
        return the arm to a colliding pose and planning could not start.
        If the configured pose is collision-free it is returned unchanged;
        otherwise the search moves outwards from it in growing joint-space rings
        and returns the closest collision-free pose found (preferring poses with
        at least PLANNING_MARGIN clearance). Deterministic (fixed seed).
        """
        configured = np.radians(cfg.HOME_CONFIG_DEG[:self.robot.n_joints])
        env = self.env.copy()
        env.place_object_at("A")                       # Reset puts the object back on table A
        checker = CollisionChecker(self.robot, env, gripper_closed=False)
        if checker.is_valid(configured):
            return configured, False
        rng = np.random.default_rng(0)
        for radius in np.linspace(0.15, np.pi, cfg.HOME_SEARCH_RINGS):
            offsets = rng.uniform(-radius, radius, (cfg.HOME_SEARCH_SAMPLES, self.robot.n_joints))
            Q = self.robot.clip(configured + offsets)
            ok = checker.valid_batch(Q)
            if not ok.any():
                continue
            V = Q[ok]
            roomy = checker.clearance_batch(V) >= cfg.PLANNING_MARGIN
            if roomy.any():
                V = V[roomy]
            return V[int(np.argmin(np.linalg.norm(V - configured, axis=1)))], True
        return configured, False                       # nothing found: keep the configured pose

    def _update_home(self):
        """Recompute home; if the arm was resting at the old home, move it too."""
        at_home = (self.held is None and self.state in ("idle", "failed", "done")
                   and np.allclose(self.q, self.home))
        self.home, adjusted = self.safe_home()
        if at_home:
            self.q = self.home.copy()
        return adjusted

    def set_link_length(self, index, length):
        if self.busy:
            return
        self.robot.set_link_length(index, length)
        self.scene_version += 1
        self.visual = None          # drawings were computed with the old link lengths
        self.invalidate(f"Link {index + 1} length changed.")
        if self._update_home():
            self.set_status(f"Link {index + 1} length changed. The configured home pose would collide "
                            "with these link lengths, so home was moved to the nearest collision-free pose.",
                            "warn")

    def jog(self, joint, delta):
        if self.busy or self.state == "running":
            return
        self._abort_attempt()
        q = self.q.copy()
        q[joint] += delta
        self.q = self.robot.clip(q)
        if self.plan_valid:
            self.invalidate("Arm moved manually.")

    def reset(self):
        self.cancel_planning()
        self._job += 1
        self.attempt = None
        self.held = None
        self.finger = self.robot.finger_open
        self.env.place_object_at("A")
        self.home, _ = self.safe_home()   # stays collision-free after link-length / obstacle edits
        self.q = self.home.copy()
        self.plan, self.plan_valid, self.visual = None, False, None
        self.step_index, self.step_s = 0, 0.0
        self.state = "idle"
        self.set_status("Reset: arm at home, object at A. Press PLAN (P).")

    def restore_default_layout(self):
        if self.busy:
            return
        self.env.restore_default_obstacles()
        self.on_environment_changed()
        self.set_status("Default obstacle layout restored.")

    def store_custom_goal(self):
        self.custom_goal = self.q.copy()
        self.set_status("Current pose stored as joint goal. Move the arm, then press J to plan to it.")

    # ------------------------------------------------------------ planning
    def _start_worker(self, what, fn):
        if self.busy:
            return
        self._abort_attempt()
        cancel = self._cancel = threading.Event()   # every job has its own cancel flag
        self._job += 1
        job = self._job
        self.state = "planning"
        self.planning_what = what
        self.planning_started = time.perf_counter()
        self.set_status(f"Planning ({what})...", "info")
        version = self.scene_version

        def run():
            try:
                self._results.put((job, what, fn(cancel), version))
            except Exception as exc:   # never let a worker die silently
                self._results.put((job, what, exc, version))

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _planner(self, robot, env, seed, cancel, max_iterations=None, return_home=cfg.RETURN_HOME):
        return TaskPlanner(robot, env, max_iterations or self.max_iterations, np.random.default_rng(seed),
                           cancel, cfg.SHORTCUT_ITERATIONS if self.smoothing else 0, return_home,
                           backend=self.backend, ik_mode="compare" if self.ik_compare else "first")

    def toggle_ik_compare(self):
        self.ik_compare = not self.ik_compare
        self.set_status("IK postures: " + ("compare - each posture is planned separately and the best is "
                                           "kept (see the IK table)" if self.ik_compare else
                                           "first reached (RRT-Connect uses whichever posture it reaches "
                                           "first)") + ". Applies to the next plan.")

    def _pause_for_planning(self):
        if self.state == "running":
            self.state = "paused"

    def request_plan(self):
        """Plan the pick-and-place task from the current state."""
        self._pause_for_planning()
        robot, env, q, held = self.robot.copy(), self.env.copy(), self.q.copy(), self.held
        seed = int(self.rng.integers(2 ** 31))
        self._start_worker("pick and place",
                           lambda cancel: self._planner(robot, env, seed, cancel).plan_pick_and_place(q, held))

    def request_joint_plan(self):
        if self.custom_goal is None:
            self.set_status("No joint goal stored yet: move the arm and press G first.", "warn")
            return
        self._pause_for_planning()
        robot, env, q, held, goal = self.robot.copy(), self.env.copy(), self.q.copy(), self.held, self.custom_goal
        seed = int(self.rng.integers(2 ** 31))
        self._start_worker("joint goal",
                           lambda cancel: self._planner(robot, env, seed, cancel).plan_joint_goal(q, goal, held))

    def request_regenerate(self):
        """Random obstacle layout, verified to be solvable by actually planning it."""
        if self.state == "running":
            self.set_status("Pause the animation before regenerating obstacles.", "warn")
            return
        robot, env, q, held = self.robot.copy(), self.env.copy(), self.q.copy(), self.held
        seed = int(self.rng.integers(2 ** 31))

        def work(cancel):
            rng = np.random.default_rng(seed)
            for attempt in range(1, cfg.REGENERATE_ATTEMPTS + 1):
                if cancel.is_set():
                    return None
                candidate = env.copy()
                candidate.randomize(rng)
                checker = CollisionChecker(robot, candidate, held, gripper_closed=held is not None)
                if not checker.is_valid(q):
                    continue
                tp = self._planner(robot, candidate, int(rng.integers(2 ** 31)), cancel,
                                   max_iterations=min(self.max_iterations, 8000))
                plan = tp.plan_pick_and_place(q, held)
                if plan.success:
                    return candidate, plan, attempt
            return None

        self._start_worker("regenerate obstacles", work)

    def cancel_planning(self):
        if self.busy:
            self._cancel.set()

    def poll_worker(self):
        try:
            job, what, result, version = self._results.get_nowait()
        except queue.Empty:
            return
        if job != self._job:
            return                      # result of a job started before a Reset
        self.state = "idle"
        if isinstance(result, Exception):
            self.set_status(f"Internal error while planning: {result!r}", "error")
            self.state = "failed"
            return
        if what == "regenerate obstacles":
            if result is None:
                msg = "Planning cancelled." if self._cancel.is_set() else \
                    f"No solvable random layout found in {cfg.REGENERATE_ATTEMPTS} attempts; kept the current one."
                self.set_status(msg, "warn")
                return
            candidate, plan, attempt = result
            for mine, new in zip(self.env.obstacles, candidate.obstacles):
                mine.points = new.points.copy()
            self.scene_version += 1
            self._accept_plan(plan, self.scene_version)
            self.set_status(f"New layout generated (verified solvable, attempt {attempt}). "
                            "Press START (Space) to run the verified plan.", "ok")
            return
        self._accept_plan(result, version)

    def _accept_plan(self, plan, version):
        self.plan = plan
        self.metrics = plan.metrics
        self.visual = build_visual(self.robot, plan)
        try:
            append_log(plan.metrics, extra={
                "link_lengths": " ".join(f"{v:.2f}" for v in self.robot.lengths),
                "max_iterations": self.max_iterations, "message": plan.message})
        except OSError:
            pass
        if version != self.scene_version:
            self.plan_valid = False
            self.state = "failed"
            self.set_status("The scene changed while planning; the result was discarded. Press PLAN again.", "warn")
            return
        if not plan.success:
            self.plan_valid = False
            self.state = "failed"
            self.set_status(f"Planning failed. {plan.message}", "error")
            if cfg.SHOW_FAILED_ATTEMPT and "cancelled" not in plan.message.lower():
                self._start_attempt(plan.message)
            return
        self.plan_valid = True
        self.state = "ready"
        self.step_index, self.step_s = 0, 0.0
        self._prepare_step()
        m = plan.metrics
        self.set_status(f"{plan.message} {m['planning_time_s']:.2f} s, {m['tree_nodes']} tree nodes. "
                        "Press START (Space).", "ok")

    # ------------------------------------------------------------ execution
    def start_pause(self):
        if self.state == "running":
            self.state = "paused"
            self.set_status("Paused.")
        elif self.state in ("ready", "paused") and self.plan_valid:
            self.state = "running"
            self.set_status(self._step_label())
        elif self.state == "paused":
            self.set_status("The plan is out of date - press PLAN to replan from the current pose.", "warn")
        elif self.state == "done":
            self.set_status("Task finished. Press PLAN for the return trip, or RESET.", "info")
        elif self.state != "planning":
            self.set_status("Nothing to run yet - press PLAN first.", "warn")

    def _step_label(self):
        if self.plan is None or self.step_index >= len(self.plan.steps):
            return ""
        s = self.plan.steps[self.step_index]
        return f"Step {self.step_index + 1}/{len(self.plan.steps)}: {s.label}"

    def _prepare_step(self):
        self._cum = None
        if self.plan and self.step_index < len(self.plan.steps):
            step = self.plan.steps[self.step_index]
            if isinstance(step, Move):
                seg = np.linalg.norm(np.diff(step.path, axis=0), axis=1)
                self._cum = np.concatenate([[0.0], np.cumsum(seg)])

    # ------------------------------------------------------------ naive attempt
    def _start_attempt(self, reason):
        """Planning failed: show the arm trying anyway (see attempt.py)."""
        att = naive_attempt(self.robot, self.env, self.q, self.held, np.random.default_rng(0))
        if len(att.path) < 2:
            self.set_status(f"{att.message}  (Planning failed: {reason})", "error")
            return
        self.attempt, self.attempt_phase, self._att_reason = att, "out", reason
        seg = np.linalg.norm(np.diff(att.path, axis=0), axis=1)
        self._att_cum = np.concatenate([[0.0], np.cumsum(seg)])
        self._att_s = 0.0
        self.state = "attempt"
        self.set_status("Planning failed - let's see what happens if the arm just tries anyway...", "warn")

    def _attempt_pose(self, s):
        path, cum = self.attempt.path, self._att_cum
        i = min(int(np.searchsorted(cum, s, side="right") - 1), len(path) - 2)
        t = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)
        return path[i] + np.clip(t, 0.0, 1.0) * (path[i + 1] - path[i])

    def _update_attempt(self, dt):
        att, total = self.attempt, self._att_cum[-1]
        step = dt * self.speed * cfg.JOINT_SPEED * cfg.ATTEMPT_SPEED_FACTOR
        if self.attempt_phase == "out":
            self._att_s = min(total, self._att_s + step)
            self.q = self._attempt_pose(self._att_s)
            if self._att_s >= total:
                self.attempt_phase, self._att_timer = "oops", cfg.ATTEMPT_PAUSE
                self.set_status(att.message + " Going back...", "error")
        elif self.attempt_phase == "oops":
            self._att_timer -= dt * self.speed
            if self._att_timer <= 0:
                self.attempt_phase = "back"
        else:   # back to where the attempt started
            self._att_s = max(0.0, self._att_s - step)
            self.q = self._attempt_pose(self._att_s)
            if self._att_s <= 0:
                self.q = att.path[0].copy()
                self.attempt = None
                self.state = "failed"
                self.set_status(f"{att.message} Back at the start. Planner: {self._att_reason}", "error")

    def _abort_attempt(self):
        """Stop a running attempt and put the arm back at its starting pose."""
        if self.attempt is not None:
            self.q = self.attempt.path[0].copy()
            self.attempt = None
            if self.state == "attempt":
                self.state = "failed"

    def update(self, dt):
        self.poll_worker()
        if self.state == "attempt" and self.attempt is not None:
            self._update_attempt(dt)
            return
        if self.state != "running":
            return
        steps = self.plan.steps
        budget = dt * self.speed
        while budget > 0 and self.step_index < len(steps):
            step = steps[self.step_index]
            if isinstance(step, Move):
                target_finger = self.robot.finger_closed if step.held is not None else self.robot.finger_open
                self.finger += np.clip(target_finger - self.finger, -cfg.GRIPPER_SPEED * budget,
                                       cfg.GRIPPER_SPEED * budget)
                v = cfg.JOINT_SPEED * step.speed_factor
                total = self._cum[-1]
                self.step_s += budget * v
                if self.step_s >= total:
                    budget = (self.step_s - total) / v
                    self.q = step.path[-1].copy()
                    self._next_step()
                else:
                    budget = 0.0
                    i = int(np.searchsorted(self._cum, self.step_s, side="right") - 1)
                    i = min(i, len(step.path) - 2)
                    t = (self.step_s - self._cum[i]) / max(self._cum[i + 1] - self._cum[i], 1e-12)
                    self.q = step.path[i] + t * (step.path[i + 1] - step.path[i])
            else:  # gripper action
                if step.action == "open" and self.held is not None:
                    # release: the object stays exactly where the gripper left it
                    c, a = self.robot.object_pose(self.q, self.held)
                    self.env.set_object_pose(c, float(wrap_angle(a)), step.station)
                    self.held = None
                target = self.robot.finger_closed if step.action == "close" else self.robot.finger_open
                move = cfg.GRIPPER_SPEED * budget
                if abs(target - self.finger) <= move:
                    budget -= abs(target - self.finger) / cfg.GRIPPER_SPEED
                    self.finger = target
                    if step.action == "close":
                        nxt = steps[self.step_index + 1] if self.step_index + 1 < len(steps) else None
                        self.held = nxt.held if isinstance(nxt, Move) and nxt.held is not None else \
                            self._measure_held(step.station)
                        self.env.object_station = None
                    self._next_step()
                else:
                    self.finger += np.sign(target - self.finger) * move
                    budget = 0.0
        if self.step_index >= len(steps):
            self.state = "done"
            self.plan_valid = False
            where = self.env.object_station
            self.set_status(f"Done! Object placed at {where}. Press PLAN to carry it back, or RESET."
                            if where and self.plan.target_station else "Done.", "ok")

    def _measure_held(self, station):
        tcp, angle = self.robot.tcp_pose(self.q)
        rel = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]) @ \
            (self.env.object_center - tcp)
        return HeldObject(self.env.object_half_size(), rel, self.env.object_angle - angle, station)

    def _next_step(self):
        self.step_index += 1
        self.step_s = 0.0
        self._prepare_step()
        if self.step_index < len(self.plan.steps):
            self.set_status(self._step_label())


def build_visual(robot, plan):
    """Pre-computed drawing data for a plan (tool path, ghost arms, tree projections)."""
    moves = []
    for step in plan.steps:
        if isinstance(step, Move):
            dense = densify(step.path, 0.02)
            if step.kind in ("transit", "carry"):
                fine = densify(step.path, 0.01)
                ghosts = fine[np.linspace(0, len(fine) - 1, cfg.GHOSTS_PER_MOVE).astype(int)]
            else:
                ghosts = step.path[[0, -1]]
            raw_tcp = robot.tcp(densify(step.raw_path, 0.02)) if step.raw_path is not None else None
            moves.append({"kind": step.kind, "tcp": robot.tcp(dense), "ghosts": ghosts,
                          "raw_tcp": raw_tcp, "path": step.path, "held": step.held})
    trees = []
    for nodes, parents, label, held in plan.trees:
        if len(nodes) == 0:
            continue
        tcp = robot.tcp(nodes)
        trees.append({"label": label, "nodes": nodes, "parents": parents, "tcp": tcp,
                      "is_start": label.endswith(":start")})
    return {"moves": moves, "trees": trees}
