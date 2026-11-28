"""
grip_sweep.py - how weak can the grip be, and how fast can the arm move,
before the box slips or falls?

    python -m roboarm_mujoco.grip_sweep                          # default grid, seeds 0..4
    python -m roboarm_mujoco.grip_sweep --seeds 3 --grips 5 10 40 --speeds 1 2

Grip force and speed do not change the plan, so each seed is planned ONCE
(RRT-Connect, MuJoCo collision checks) and that same plan is then executed
with every grip force x speed combination. Differences between cells are
therefore caused by the physics only.

Physics reference: the box (1 kg, 9.81 N) is held by two fingers with
friction 1.0, so standing still it needs 2 x 1.0 x F > 9.81 N, i.e.
F > ~4.9 N per finger. Moving adds inertial forces.

Saves a NEW file every time: roboarm_mujoco/results/grip_sweep_<date>_<time>.csv
Units: N, x JOINT_SPEED (rad/s), metres, seconds.
"""
import argparse
import csv
import datetime
import os
import platform

import numpy as np

from roboarm_mujoco import paths  # noqa: F401  (puts roboarm/ on sys.path)
from roboarm_mujoco import settings as S
from roboarm_mujoco.benchmark import FIELDS, execute
from roboarm_mujoco.pick_place import TaskPlanner
from roboarm_mujoco.robot_api import RobotSim

import config as rcfg                     # roboarm/config.py

GRIPS = [2.5, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 40.0, 80.0]     # N per finger
SPEEDS = [0.5, 1.0, 2.0, 3.0]                                  # x JOINT_SPEED
SWEEP_FIELDS = FIELDS + ["outcome"]
CLEAN_SLIP = 0.02        # m: a "clean" success keeps the box within 2 cm of where it was grasped


def outcome(r):
    """One word per run: placed / dropped / collision / misplaced.
    Dropped is checked first: a box that slips out and then hits an obstacle
    is a grip failure, not a planning failure."""
    if r["task_success"]:
        return "placed"
    if r["lost_grip_steps"] > 5 or r["max_slip_finger_m"] > 0.3:
        return "dropped"
    if r["collision_steps"]:
        return "collision"
    return "misplaced"


def sweep(seeds, grips, speeds, layout_csv=None, max_iterations=rcfg.MAX_ITERATIONS):
    rows = []
    for seed in seeds:
        sim = RobotSim(layout_csv=layout_csv, seed=seed)
        s = sim.get_state()
        plan = TaskPlanner(sim.layout, rng=sim.rng, max_iterations=max_iterations) \
            .plan(s["q"], s["box_pos"], s["box_angle"])
        print(f"seed {seed}: {plan.message} ({plan.metrics['planning_time_s']:.1f} s)", flush=True)
        if not plan.success:
            continue
        for g in grips:
            for v in speeds:
                sim = RobotSim(layout_csv=layout_csv, seed=seed, grip_force=g)
                r = {"layout": layout_csv or "default", "seed": seed, "max_iterations": max_iterations,
                     "grip_force_n": g, "plan_success": True,
                     **{k: plan.metrics.get(k) for k in ("planning_time_s", "joint_path_rad", "tcp_path_m")},
                     **execute(sim, plan, speed=v)}
                r["outcome"] = outcome(r)
                rows.append(r)
    return rows


def table(rows, grips, speeds, title, cell):
    print(f"\n{title}")
    print(f"{'grip N':>8s} | " + " | ".join(f"speed x{v:<4g}" for v in speeds))
    for g in grips:
        cells = []
        for v in speeds:
            sel = [r for r in rows if r["grip_force_n"] == g and r["speed"] == v]
            cells.append(f"{cell(sel):>11s}" if sel else f"{'-':>11s}")
        print(f"{g:8g} | " + " | ".join(cells))


def report(rows, grips, speeds):
    table(rows, grips, speeds, "Task success (box placed upright on the target table, no collisions):",
          lambda sel: f"{sum(r['task_success'] for r in sel)}/{len(sel)}")
    table(rows, grips, speeds, "Clean success (placed AND the box slipped < 2 cm in the hand):",
          lambda sel: f"{sum(r['task_success'] and r['max_slip_finger_m'] < CLEAN_SLIP for r in sel)}/{len(sel)}")

    def slip(sel):
        kept = [r["max_slip_finger_m"] for r in sel if r["outcome"] != "dropped"]
        return f"{1000 * np.mean(kept):.1f}" if kept else "dropped"
    table(rows, grips, speeds,
          "Max slip of the box along the finger, mean over seeds where it was not dropped (mm):", slip)
    table(rows, grips, speeds, "Failures (d = dropped, c = collision, m = misplaced):",
          lambda sel: " ".join(f"{k[0]}{sum(r['outcome'] == k for r in sel)}"
                               for k in ("dropped", "collision", "misplaced")
                               if any(r["outcome"] == k for r in sel)) or "none")


def load(path):
    """Rows of a saved sweep CSV, with numbers and booleans converted back."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if v in ("True", "False"):
                    r[k] = v == "True"
                else:
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            r["outcome"] = outcome(r)
            rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (0 .. n-1, plus --seed)")
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--grips", type=float, nargs="+", default=GRIPS, help="grip forces, N per finger")
    ap.add_argument("--speeds", type=float, nargs="+", default=SPEEDS, help="speed factors")
    ap.add_argument("--layout", help="environment CSV (default: config layout)")
    ap.add_argument("--max-iterations", type=int, default=rcfg.MAX_ITERATIONS)
    ap.add_argument("--out", help="output CSV (default: a new timestamped file in roboarm_mujoco/results/)")
    ap.add_argument("--summarise", metavar="CSV", help="only print the tables of an earlier sweep CSV")
    args = ap.parse_args(argv)

    if args.summarise:
        rows = load(args.summarise)
        report(rows, sorted({r["grip_force_n"] for r in rows}), sorted({r["speed"] for r in rows}))
        return rows

    out = args.out
    if out is None:
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(folder, exist_ok=True)
        out = os.path.join(folder, f"grip_sweep_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; not overwriting it.")

    import mujoco
    seeds = list(range(args.seed, args.seed + args.seeds))
    print(f"MuJoCo {mujoco.__version__}, Python {platform.python_version()}, planner RRT-Connect, "
          f"max_iterations {args.max_iterations}; seeds {seeds}; box {S.OBJECT_MASS} kg, friction {S.FRICTION[0]}, "
          f"contact solref {S.CONTACT_SOLREF}; JOINT_SPEED {rcfg.JOINT_SPEED} rad/s, "
          f"motor speed cap {S.MAX_JOINT_SPEED} rad/s")
    rows = sweep(seeds, args.grips, args.speeds, args.layout, args.max_iterations)
    report(rows, args.grips, args.speeds)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")
    return rows


if __name__ == "__main__":
    main()
