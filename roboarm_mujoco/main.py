"""
main.py - interactive pick-and-place in MuJoCo's built-in viewer.

    python -m roboarm_mujoco.main                       # default layout (roboarm/config.py)
    python -m roboarm_mujoco.main --layout roboarm_mujoco/environments/my_env.csv --seed 3

Keys (MuJoCo's viewer already uses almost every letter for its own display
toggles; these were chosen because their built-in toggles change nothing
visible in this scene):
    P       plan the pick-and-place from the current state, then run it
            (after the box reaches the other table, P carries it back)
    Enter   pause / resume the running motion
    E       open / close the gripper (when no motion is running)
    Home    reset: arm home, gripper open, box back at its layout position

The viewer's own controls still work: mouse to orbit / pan / zoom,
double-click to select, Ctrl + drag to push bodies, Tab / F1 for its menus.
"""
import argparse
import queue
import threading
import time

import numpy as np

from roboarm_mujoco import settings as S
from roboarm_mujoco.pick_place import PlanExecutor, TaskPlanner
from roboarm_mujoco.robot_api import RobotSim

KEY_PLAN = ord("P")
KEY_GRIPPER = ord("E")
KEY_PAUSE = 257          # GLFW_KEY_ENTER
KEY_RESET = 268          # GLFW_KEY_HOME
HELP = "P  plan + run      Enter  pause      E  gripper      Home  reset"


class App:
    """Keyboard handling and task state, independent of the window (testable)."""

    def __init__(self, sim, speed=1.0):
        self.sim = sim
        self.speed = speed
        self.keys = queue.SimpleQueue()
        self.executor = None
        self.paused = False
        self.status = "Press P to plan and run the pick-and-place."
        self.last_plan = None
        self._thread = None
        self._result = None
        self._generation = 0           # bumped by reset: discards plans started before it

    # ------------------------------------------------------------ input
    def key_callback(self, keycode):
        """Called by the viewer thread; the main loop handles the key."""
        self.keys.put(keycode)

    def handle_key(self, key):
        if key == KEY_PLAN:
            self.request_plan()
        elif key == KEY_PAUSE:
            if self.executor is not None:
                self.paused = not self.paused
                self.status = "Paused (Enter resumes)." if self.paused else "Running."
        elif key == KEY_GRIPPER:
            if self.busy:
                self.status = "Busy: wait for the motion to finish (or press Home)."
            else:
                closed = not self.sim.get_state()["gripper_closed"]
                self.sim.set_gripper(closed)
                self.status = "Gripper closing." if closed else "Gripper opening."
        elif key == KEY_RESET:
            self._generation += 1
            self.executor, self.paused = None, False
            self.sim.reset()
            self.status = "Reset. Press P to plan."

    # ------------------------------------------------------------ planning
    @property
    def planning(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def busy(self):
        return self.planning or self.executor is not None

    def request_plan(self):
        if self.busy:
            self.status = "Busy: already planning or moving."
            return
        s = self.sim.get_state()
        layout, rng, gen = self.sim.layout, self.sim.rng, self._generation

        def work():
            plan = TaskPlanner(layout, rng=rng).plan(s["q"], s["box_pos"], s["box_angle"])
            self._result = (gen, plan)

        self._result = None
        self.status = "Planning (RRT-Connect with MuJoCo collision checks)..."
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def _collect_plan(self):
        if self._result is None or self.planning:
            return
        gen, plan = self._result
        self._result = None
        if gen != self._generation:
            return
        self.last_plan = plan
        m = plan.metrics
        if plan.success:
            self.executor = PlanExecutor(self.sim, plan, speed=self.speed)
            self.status = (f"{plan.message} {m['planning_time_s']:.1f} s, "
                           f"{m['joint_path_rad']:.1f} rad, {m['collision_checks']} collision checks.")
        else:
            self.status = f"Planning failed: {plan.message}"

    # ------------------------------------------------------------ loop
    def update(self):
        """One control step (0.02 s of simulated time)."""
        while not self.keys.empty():
            self.handle_key(self.keys.get())
        self._collect_plan()
        ex = self.executor
        if ex is not None and not self.paused:
            ex.tick()
            if ex.done:
                self._report(ex)
                self.executor = None
        else:
            self.sim.step(self.sim.target)

    def _report(self, ex):
        s = self.sim.get_state()
        hits = len(ex.collisions)
        self.status = (f"Done: box at x={s['box_pos'][0]:.2f}, y={s['box_pos'][1]:.2f} m "
                       f"(target {ex.plan.place_table}). "
                       + ("No collisions." if not hits else f"{hits} control steps with collisions!")
                       + " Press P to carry it back.")

    def texts(self):
        s = self.sim.get_state()
        ex = self.executor
        phase = f"{ex.label}  ({100 * ex.progress:.0f} %)" if ex is not None else ""
        info = (f"t = {self.sim.time:6.1f} s   q = {np.degrees(s['q']).round(1)} deg   "
                f"holding: {s['holding']}   collision: {self.sim.in_collision()}")
        return [(None, "top", "roboarm_mujoco", self.status + ("\n" + phase if phase else "")),
                (None, "bottom", "Keys", HELP + "\n" + info)]


def main(argv=None):
    import mujoco
    import mujoco.viewer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", help="environment CSV (default: the layout in roboarm/config.py)")
    ap.add_argument("--seed", type=int, default=None, help="seed for the planner's random choices")
    ap.add_argument("--speed", type=float, default=1.0, help="motion speed factor (1 = config.JOINT_SPEED)")
    args = ap.parse_args(argv)

    sim = RobotSim(layout_csv=args.layout, seed=args.seed)
    app = App(sim, speed=args.speed)
    grid = {"top": mujoco.mjtGridPos.mjGRID_TOPLEFT, "bottom": mujoco.mjtGridPos.mjGRID_BOTTOMLEFT}
    print(HELP)
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=app.key_callback) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.0, 0.0, 2.6]
            viewer.cam.distance = 11.0
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -5.0
        while viewer.is_running():
            t0 = time.perf_counter()
            app.update()
            viewer.set_texts([(font, grid[pos], a, b) for font, pos, a, b in app.texts()])
            viewer.sync()
            wait = S.CONTROL_DT - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)


if __name__ == "__main__":
    main()
