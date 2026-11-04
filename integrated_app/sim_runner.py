"""Runs the Cobuild drywall simulation for ONE worker profile. Executed as a subprocess.

Why a subprocess: the camera project and the Cobuild project both have a top-level `config.py`.
Running the simulation in its own process keeps the two import namespaces apart and leaves
the Cobuild code completely unmodified.

    python sim_runner.py eval     --profile-json p.json --runs 20 --out result.json
    python sim_runner.py snapshot --profile-json p.json --out s.png [--at-tick 60]
    python sim_runner.py video    --profile-json p.json --out v.mp4

The profile JSON holds the Cobuild WorkerProfile fields (see bridge.py).
"""
import argparse
import json
import sys
from pathlib import Path

COBUILD_DIR = Path(__file__).resolve().parents[1] / "cobuild_gymnasium_starter"
sys.path.insert(0, str(COBUILD_DIR))

# Rule-based strategies need no trained model (PPO would need torch + stable-baselines3).
RULE_STRATEGIES = ["no_robot", "A_reactive", "B_proactive_generic", "C_cobuild_rule"]


def load_profile(path):
    from profiles import WorkerProfile
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return WorkerProfile(**data)


def do_eval(profile, runs, out):
    import numpy as np
    import evaluate as EV
    from strategies import LABELS, make_strategy

    rows = {k: [] for k in RULE_STRATEGIES}
    for seed in range(1000, 1000 + runs):          # same seeds for every strategy => paired
        for k in RULE_STRATEGIES:
            rows[k].append(EV.run_episode(make_strategy(k), profile, seed))

    metrics = ["minutes", "worker_wait_minutes", "interruptions", "strain",
               "heavy_minutes", "cost_minutes", "unnecessary_commands", "holds"]
    summary = {}
    for k, rs in rows.items():
        summary[k] = {"label": LABELS[k], **{m: float(np.mean([r[m] for r in rs])) for m in metrics},
                      **{m + "_std": float(np.std([r[m] for r in rs])) for m in ("minutes", "strain", "cost_minutes")}}

    def paired(ref, other, metric):
        d = np.array([a[metric] - b[metric] for a, b in zip(rows[ref], rows[other])])
        half = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        return {"mean_saving": float(d.mean()), "ci95": float(half),
                "pct_runs_better": float((d > 1e-9).mean() * 100),
                "pct_runs_worse": float((d < -1e-9).mean() * 100)}

    paired_out = {}
    for ref, other in (("no_robot", "B_proactive_generic"), ("no_robot", "C_cobuild_rule"),
                       ("B_proactive_generic", "C_cobuild_rule")):
        paired_out[f"{ref}__vs__{other}"] = {m: paired(ref, other, m)
                                              for m in ("minutes", "strain", "cost_minutes")}
    result = {"profile": profile.as_dict(), "runs": runs, "strategies": summary, "paired": paired_out}
    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def do_visual(mode, profile, out, at_tick, top, bottom, width, height):
    """Reuse visualize_3d.main() unchanged, swapping only its profile lookup."""
    import visualize_3d as V
    V.resolve_profile = lambda name, seed: profile
    argv = ["visualize_3d.py", "--compare", "--top", top, "--bottom", bottom,
            "--width", str(width), "--height", str(height)]
    argv += ["--snapshot", out, "--at-tick", str(at_tick)] if mode == "snapshot" else ["--video", out]
    sys.argv = argv
    V.main()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["eval", "snapshot", "video"])
    ap.add_argument("--profile-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--at-tick", type=int, default=60)
    ap.add_argument("--top", default="B_proactive_generic")
    ap.add_argument("--bottom", default="C_cobuild_rule")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()
    profile = load_profile(a.profile_json)
    if a.mode == "eval":
        do_eval(profile, a.runs, a.out)
    else:
        do_visual(a.mode, profile, a.out, a.at_tick, a.top, a.bottom, a.width, a.height)


if __name__ == "__main__":
    main()
