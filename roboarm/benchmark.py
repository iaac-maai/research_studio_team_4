"""
benchmark.py - run the pick-and-place task many times without graphics and
summarise the metrics. This is the tool for comparing planners (and, later,
a reinforcement-learning policy) on the same task.

    python benchmark.py                       # 20 runs, built-in RRT-Connect
    python benchmark.py --runs 50 --planner ompl --ompl-algorithm PRM
    python benchmark.py --random-layouts      # a new random obstacle layout per run

Results are printed and saved to results/benchmark_<planner>.csv
"""
import argparse
import csv
import os

import numpy as np

import config as cfg
from collision import CollisionChecker
from environment import Environment
from metrics import LOG_FIELDS
from pick_place import TaskPlanner
from robot import PlanarArm

SUMMARY = ["planning_time_s", "samples", "tree_nodes", "collision_checks", "joint_path_length_rad",
           "tcp_path_length_m", "free_space_clearance_m"]


def run(runs, backend, seed, random_layouts, max_iterations):
    rng = np.random.default_rng(seed)
    robot = PlanarArm()
    home = np.radians(cfg.HOME_CONFIG_DEG[:robot.n_joints])
    rows = []
    for k in range(runs):
        env = Environment.default()
        if random_layouts:
            # random jitter, redrawn until the start (home) pose is collision-free;
            # the layout is NOT guaranteed to be solvable (that is part of the test)
            while True:
                env.randomize(rng)
                if CollisionChecker(robot, env).is_valid(home):
                    break
        planner = TaskPlanner(robot, env, max_iterations, np.random.default_rng(rng.integers(2 ** 31)),
                              backend=backend)
        plan = planner.plan_pick_and_place(home)
        m = dict(plan.metrics, message=plan.message, max_iterations=max_iterations,
                 link_lengths=" ".join(f"{v:.2f}" for v in robot.lengths))
        rows.append(m)
        status = "ok  " if plan.success else "FAIL"
        print(f"run {k + 1:3d}/{runs}  {status}  {m['planning_time_s']:6.2f} s  "
              f"nodes {m['tree_nodes']:5d}  joint len {m.get('joint_path_length_rad', float('nan')):6.2f} rad"
              + ("" if plan.success else f"   ({plan.message})"))
    return rows


def summarise(rows):
    ok = [r for r in rows if r["success"]]
    print(f"\nsuccess rate: {len(ok)}/{len(rows)} ({100 * len(ok) / max(1, len(rows)):.0f} %)")
    if not ok:
        return
    print(f"{'metric (successful runs)':32s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
    for key in SUMMARY:
        v = np.array([r[key] for r in ok], float)
        print(f"{key:32s} {v.mean():10.3f} {v.std():10.3f} {v.min():10.3f} {v.max():10.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--planner", choices=["rrt_connect", "ompl"], default=cfg.PLANNER_BACKEND)
    p.add_argument("--ompl-algorithm", default=cfg.OMPL_ALGORITHM)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iterations", type=int, default=cfg.MAX_ITERATIONS)
    p.add_argument("--random-layouts", action="store_true")
    args = p.parse_args()
    cfg.OMPL_ALGORITHM = args.ompl_algorithm
    rows = run(args.runs, args.planner, args.seed, args.random_layouts, args.max_iterations)
    summarise(rows)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    name = args.planner if args.planner != "ompl" else f"ompl_{args.ompl_algorithm}"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"benchmark_{name}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[c for c in LOG_FIELDS if c != "timestamp"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
