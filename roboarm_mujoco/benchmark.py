"""
benchmark.py - classical baseline: plan AND physically execute the
pick-and-place many times without a window, then summarise.

    python -m roboarm_mujoco.benchmark                      # 10 runs, default layout, seeds 0..9
    python -m roboarm_mujoco.benchmark --runs 20 --seed 100
    python -m roboarm_mujoco.benchmark --layout a.csv --layout b.csv   # each layout x each seed

Each run uses its own seed (base seed + run number), so any run can be
repeated exactly. Planning is RRT-Connect + shortcutting (pick_place.py) with
MuJoCo collision checks; execution runs the physics with friction grip, so
the box can slip or drop even when the plan is valid.

Results are saved to a NEW file every time (existing results are never
overwritten): roboarm_mujoco/results/benchmark_<date>_<time>.csv
Units: seconds, radians, metres.
"""
import argparse
import csv
import datetime
import os
import platform

import numpy as np

from roboarm_mujoco import paths  # noqa: F401  (puts roboarm/ on sys.path)
from roboarm_mujoco import settings as S
from roboarm_mujoco.pick_place import GripperAction, Move, PlanExecutor, TaskPlanner
from roboarm_mujoco.robot_api import RobotSim

import config as rcfg                     # roboarm/config.py
from geometry import rotation             # roboarm/geometry.py

PLACE_TOL = 0.05        # m: box centre within this distance of the planned place point
UPRIGHT_TOL = np.radians(10)

FIELDS = ["layout", "seed", "max_iterations", "grip_force_n", "speed",
          "plan_success", "task_success", "message",
          "planning_time_s", "rrt_time_s", "queries", "samples", "nodes", "collision_checks",
          "joint_path_rad", "raw_joint_path_rad", "tcp_path_m",
          "exec_sim_time_s", "collision_steps", "first_collision",
          # slip = how far the box moved in the hand while carried: max_slip_m / _rad relative to
          # the tool centre point (a few mm of noise), max_slip_finger_m relative to the finger itself
          "lost_grip_steps", "max_slip_m", "max_slip_rad", "max_slip_finger_m",
          "final_box_x", "final_box_y", "final_box_angle_rad", "place_error_m"]


def _hand_frame(state):
    """Box position and angle relative to the gripper (planar: hand angle = q1+q2+q3)."""
    hand = float(np.sum(state["q"]))
    rel = rotation(-hand) @ (state["box_pos"] - state["tcp"])
    return rel, state["box_angle"] - hand


def run_one(layout_csv, seed, max_iterations, grip_force=S.GRIP_FORCE, speed=1.0):
    sim = RobotSim(layout_csv=layout_csv, seed=seed, grip_force=grip_force)
    s = sim.get_state()
    plan = TaskPlanner(sim.layout, rng=sim.rng, max_iterations=max_iterations) \
        .plan(s["q"], s["box_pos"], s["box_angle"])
    m = plan.metrics
    row = {"layout": layout_csv or "default", "seed": seed, "max_iterations": max_iterations,
           "grip_force_n": grip_force, "plan_success": plan.success, "task_success": False,
           "message": plan.message,
           **{k: m.get(k) for k in ("planning_time_s", "rrt_time_s", "queries", "samples", "nodes",
                                    "collision_checks", "joint_path_rad", "raw_joint_path_rad",
                                    "tcp_path_m")}}
    if not plan.success:
        return row
    row.update(execute(sim, plan, speed))
    return row


def execute(sim, plan, speed=1.0):
    """Play a successful plan on `sim` (physics) and measure the outcome.
    Returns the execution columns of FIELDS plus task_success and message."""
    m, d = sim.model, sim.data
    finger, box = m.body("finger_left").id, m.body("object").id

    def along_finger():                                   # box centre in the finger's frame
        return d.xmat[finger].reshape(3, 3).T @ (d.xpos[box] - d.xpos[finger])

    # Between "grasp" and "release" the box should stay fixed in the hand.
    ex = PlanExecutor(sim, plan, speed=speed)
    t_start = sim.time
    in_hand, ref = False, None
    lost, slip_m, slip_rad, slip_finger = 0, 0.0, 0.0, 0.0
    while not ex.done:
        kind, value, _ = ex.commands[ex.index]
        if kind == "grip" and not value:
            in_hand = False                                  # release command
        state = ex.tick()
        if kind == "grip" and value:
            in_hand, ref = True, None                        # grasp command; reference set at first motion
        if in_hand and kind == "q":
            rel, ang = _hand_frame(state)
            f = along_finger()
            if ref is None:
                ref = (rel, ang, f)
            slip_m = max(slip_m, float(np.linalg.norm(rel - ref[0])))
            dang = (ang - ref[1] + np.pi) % (2 * np.pi) - np.pi     # wrapped: 2*pi is no rotation
            slip_rad = max(slip_rad, abs(float(dang)))
            slip_finger = max(slip_finger, float(np.linalg.norm(f - ref[2])))
            lost += not state["holding"]

    s = sim.get_state()
    rest = plan.place_center - [0.0, S.PLACE_DROP_GAP]           # released above the table, rests on it
    err = float(np.linalg.norm(s["box_pos"] - rest))
    ok = err < PLACE_TOL and abs(s["box_angle"]) < UPRIGHT_TOL and not ex.collisions
    out = dict(task_success=ok, speed=speed, exec_sim_time_s=sim.time - t_start,
               collision_steps=len(ex.collisions),
               first_collision=" / ".join("-".join(p) for p in ex.collisions[0][1][:2]) if ex.collisions else "",
               lost_grip_steps=lost, max_slip_m=slip_m, max_slip_rad=slip_rad, max_slip_finger_m=slip_finger,
               final_box_x=s["box_pos"][0], final_box_y=s["box_pos"][1],
               final_box_angle_rad=s["box_angle"], place_error_m=err, message=plan.message)
    if not ok:
        out["message"] += " Execution failed: " + (
            "collision." if ex.collisions else f"box {err:.3f} m from target.")
    return out


def summarise(rows):
    n = len(rows)
    planned = [r for r in rows if r["plan_success"]]
    done = [r for r in rows if r["task_success"]]
    print(f"\nplanning success: {len(planned)}/{n} ({100 * len(planned) / max(1, n):.0f} %)")
    print(f"task success (box placed, upright, no collisions): {len(done)}/{n} "
          f"({100 * len(done) / max(1, n):.0f} %)")
    if not planned:
        return
    keys = ["planning_time_s", "collision_checks", "joint_path_rad", "raw_joint_path_rad", "tcp_path_m",
            "exec_sim_time_s", "collision_steps", "lost_grip_steps", "max_slip_m", "place_error_m"]
    print(f"\n{'metric (planned runs)':24s} {'mean':>11s} {'std':>11s} {'min':>11s} {'max':>11s}")
    for k in keys:
        v = np.array([r[k] for r in planned if r.get(k) is not None], float)
        if v.size:
            print(f"{k:24s} {v.mean():11.4g} {v.std():11.4g} {v.min():11.4g} {v.max():11.4g}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10, help="runs per layout")
    ap.add_argument("--seed", type=int, default=0, help="seed of the first run (run k uses seed + k)")
    ap.add_argument("--layout", action="append", help="environment CSV (repeatable; default: config layout)")
    ap.add_argument("--max-iterations", type=int, default=rcfg.MAX_ITERATIONS)
    ap.add_argument("--grip-force", type=float, default=S.GRIP_FORCE, help="finger force limit, N")
    ap.add_argument("--speed", type=float, default=1.0, help="motion speed factor (1 = config.JOINT_SPEED)")
    ap.add_argument("--out", help="output CSV (default: a new timestamped file in roboarm_mujoco/results/)")
    args = ap.parse_args(argv)

    out = args.out
    if out is None:
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(folder, exist_ok=True)
        out = os.path.join(folder, f"benchmark_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; not overwriting it.")

    import mujoco
    print(f"MuJoCo {mujoco.__version__}, Python {platform.python_version()}, planner RRT-Connect, "
          f"max_iterations {args.max_iterations}, shortcut {rcfg.SHORTCUT_ITERATIONS}, "
          f"margins {rcfg.PLANNING_MARGIN}/{rcfg.SAFETY_MARGIN} m, grip {args.grip_force} N, speed x{args.speed}")
    rows = []
    layouts = args.layout or [None]
    for layout in layouts:
        for k in range(args.runs):
            seed = args.seed + k
            r = run_one(layout, seed, args.max_iterations, args.grip_force, args.speed)
            rows.append(r)
            tag = "ok  " if r["task_success"] else ("EXEC" if r["plan_success"] else "PLAN")
            extra = (f"  slip {r['max_slip_m'] * 1000:5.1f} mm  err {r['place_error_m'] * 1000:5.1f} mm"
                     if r["plan_success"] else "")
            print(f"{r['layout']}  seed {seed:4d}  {tag}  plan {r['planning_time_s']:6.2f} s{extra}"
                  + ("" if r["task_success"] else f"   ({r['message']})"), flush=True)
    summarise(rows)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")
    return rows


if __name__ == "__main__":
    main()
