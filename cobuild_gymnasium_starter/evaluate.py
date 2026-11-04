"""Install the same wall many times with every strategy and measure the difference.

    python evaluate.py                  # 9 personas x 50 runs + 100 unseen workers x 5 runs, then plots
    python evaluate.py --runs 20        # fewer repetitions (faster)
    python evaluate.py --no-ppo         # rule-based strategies only (no trained model needed)

Fair comparison: for a given (worker, run seed) every strategy faces exactly the same worker and the
same random variability in step durations ("common random numbers"), so differences between strategies
are PAIRED. We report the mean paired difference with a 95 % confidence interval.

"Combined cost" = minutes + 0.05 x strain points. It is the planners' own trade-off (one strain point is
worth half a 6-second tick) expressed in minutes, so that time and physical strain can be read together.

Outputs (in results/): episodes.csv (one row per worker x seed x strategy), summary.md, fig*.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import ergonomics as E
from cobuild_env import CobuildDrywallEnv
from profiles import GENERIC, PERSONAS, sample_population
from strategies import LABELS, STRATEGY_KEYS, make_strategy

RESULTS = Path(__file__).with_name('results')
UNSEEN_SEED = 20261104  # different stream from training: these workers were never seen
COST_PER_STRAIN_MIN = -C.R_STRAIN * C.TICK_SECONDS / 60.0   # minutes per strain point
# (reference, challenger): how much does the challenger save compared with the reference?
COMPARISONS = [
    ('no_robot', 'A_reactive'),
    ('no_robot', 'B_proactive_generic'),
    ('no_robot', 'C_cobuild_rule'),
    ('no_robot', 'C_cobuild_ppo'),
    ('B_proactive_generic', 'C_cobuild_rule'),       # value of the profile, same planner
    ('B_proactive_generic', 'C_cobuild_ppo'),
    ('B_proactive_generic', 'ablation_ppo_no_profile'),
    ('ablation_ppo_no_profile', 'C_cobuild_ppo'),    # value of the profile, same learning algorithm
]


def run_episode(strategy, profile, seed):
    env = CobuildDrywallEnv(profile=profile, **strategy.env_kwargs())
    obs, info = env.reset(seed=seed)
    while True:
        obs, _, terminated, truncated, info = env.step(strategy.act(env, obs))
        if terminated or truncated:
            break
    m = C.TICK_SECONDS / 60.0
    row = {
        'minutes': info['minutes'],
        'worker_wait_minutes': info['worker_wait_ticks'] * m,
        'interruptions': info['interruptions'],
        'strain': info['strain'],
        'heavy_minutes': info['heavy_ticks'] * m,
        'overhead_minutes': info['overhead_ticks'] * m,
        'unnecessary_commands': info['unnecessary_commands'],
        'robot_exposure_minutes': info['robot_exposure_ticks'] * m,
        'holds': info['holds'], 'boards_brought': info['boards_brought'], 'steadied': info['steadied'],
        'tool_handovers': info['handovers'], 'awkward_handovers': info['awkward_handovers'],
        'wrong_side_handovers': info['wrong_side_handovers'],
        'final_fatigue': info['final_fatigue'], 'completed': int(terminated),
    }
    row['cost_minutes'] = row['minutes'] + COST_PER_STRAIN_MIN * row['strain']
    for cat, ticks in info['activity_ticks'].items():
        row[f'min_{cat}'] = ticks * m
    return row


def build_workers(n_unseen):
    workers = [('persona', p) for p in [GENERIC, *PERSONAS.values()]]
    rng = np.random.default_rng(UNSEEN_SEED)
    workers += [('unseen', sample_population(rng, name=f'unseen_{i:03d}')) for i in range(n_unseen)]
    return workers


def evaluate(strategy_keys, runs, n_unseen, unseen_runs):
    strategies = [make_strategy(k) for k in strategy_keys]
    rows = []
    for group, profile in build_workers(n_unseen):
        for seed in range(1000, 1000 + (runs if group == 'persona' else unseen_runs)):
            for s in strategies:
                rows.append({'group': group, 'profile': profile.name, 'seed': seed, 'strategy': s.key,
                             'height_m': profile.height_m, 'reach_m': profile.reach_m,
                             'lift_capacity_kg': profile.lift_capacity_kg, 'pace': profile.pace,
                             'handedness': profile.handedness, 'fatigue_rate': profile.fatigue_rate,
                             **run_episode(s, profile, seed)})
    return pd.DataFrame(rows)


def paired(df, ref, other, metric):
    """Mean of (ref - other) over identical (worker, seed) pairs. Positive = `other` is lower (better).
    Returns (mean difference, 95 % CI half-width, % of the reference mean, % runs better, % runs worse)."""
    wide = df.pivot_table(index=['profile', 'seed'], columns='strategy', values=metric)
    d = wide[ref] - wide[other]
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    base = wide[ref].mean()
    return d.mean(), half, 100 * d.mean() / base if base else 0.0, (d > 1e-9).mean() * 100, (d < -1e-9).mean() * 100


def summarize(df):
    keys = [k for k in STRATEGY_KEYS if k in df.strategy.unique()]
    L = ['# Cobuild drywall installation: evaluation results', '',
         f'Synthetic simulation. A wall of {C.N_BOARDS} boards (2 whole + 2 trimmed), 1 tick = {C.TICK_SECONDS} s. '
         'Numbers follow from the assumptions in config.py (see docs/RESEARCH.md for what is calibrated to '
         'published data and what is assumed); they are not field measurements.', '']
    for group, title in [('persona', 'Named personas (incl. the generic average worker)'),
                         ('unseen', 'Unseen random workers (never used in training)')]:
        g = df[df.group == group]
        if g.empty:
            continue
        n_pairs = g[['profile', 'seed']].drop_duplicates().shape[0]
        L += [f'## {title}', '', f'{g.profile.nunique()} workers, {n_pairs} paired runs per strategy.', '',
              '| Strategy | Time (min) | Worker waiting (min) | Interruptions | Strain (pts) | Heavy handling (min) '
              '| Combined cost (min) | Boards held | Unnecessary cmds |', '|---|---|---|---|---|---|---|---|---|']
        for k in keys:
            s = g[g.strategy == k]
            L.append(f'| {LABELS[k]} | {s.minutes.mean():.1f} | {s.worker_wait_minutes.mean():.2f} | '
                     f'{s.interruptions.mean():.1f} | {s.strain.mean():.0f} | {s.heavy_minutes.mean():.1f} | '
                     f'{s.cost_minutes.mean():.1f} | {s.holds.mean():.1f} | {s.unnecessary_commands.mean():.2f} |')
        L += ['', 'Where does the time go? Mean minutes per wall by activity:', '',
              '| Strategy | ' + ' | '.join(E.CATEGORIES) + ' |', '|---' * (len(E.CATEGORIES) + 1) + '|']
        for k in keys:
            s = g[g.strategy == k]
            L.append(f'| {LABELS[k]} | ' + ' | '.join(f'{s[f"min_{c}"].mean():.1f}' for c in E.CATEGORIES) + ' |')
        L += ['', 'Paired savings (positive = the second strategy needs less):', '',
              '| Comparison | Time saved | Strain saved | Heavy handling saved | Combined cost saved | Runs better / worse (cost) |',
              '|---|---|---|---|---|---|']
        for ref, other in COMPARISONS:
            if ref in keys and other in keys:
                t, th, tp, _, _ = paired(g, ref, other, 'minutes')
                _, _, sp, _, _ = paired(g, ref, other, 'strain')
                _, _, hp, _, _ = paired(g, ref, other, 'heavy_minutes')
                c, ch, cp, better, worse = paired(g, ref, other, 'cost_minutes')
                L.append(f'| {LABELS[ref]} → {LABELS[other]} | {t:.2f} ± {th:.2f} min ({tp:.1f} %) | {sp:.1f} % | '
                         f'{hp:.1f} % | {c:.2f} ± {ch:.2f} min ({cp:.1f} %) | {better:.0f} % / {worse:.0f} % |')
        L.append('')
    p = df[df.group == 'persona']
    if not p.empty:
        L += ['## Per persona: mean time (min) / strain (pts)', '',
              '| Persona | ' + ' | '.join(LABELS[k] for k in keys) + ' |', '|---' * (len(keys) + 1) + '|']
        for name in p.profile.unique():
            q = p[p.profile == name]
            L.append(f'| {name} | ' + ' | '.join(
                f'{q[q.strategy == k].minutes.mean():.1f} / {q[q.strategy == k].strain.mean():.0f}' for k in keys) + ' |')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=int, default=50, help='repetitions per persona')
    parser.add_argument('--unseen', type=int, default=100, help='number of unseen random workers')
    parser.add_argument('--unseen-runs', type=int, default=5, help='repetitions per unseen worker')
    parser.add_argument('--no-ppo', action='store_true', help='only the rule-based strategies')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    keys = [k for k in STRATEGY_KEYS if not (args.no_ppo and 'ppo' in k)]
    df = evaluate(keys, args.runs, args.unseen, args.unseen_runs)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / 'episodes.csv', index=False)
    text = summarize(df)
    (RESULTS / 'summary.md').write_text(text)
    print(text)
    print(f'Saved {RESULTS / "episodes.csv"} and {RESULTS / "summary.md"}')
    if not args.no_plots:
        import plots
        plots.make_all(df, RESULTS)
