"""Repeat the drywall installation many times with every strategy and measure the difference.

    python evaluate.py                  # 8 personas x 50 runs + 100 unseen workers x 5 runs, then plots
    python evaluate.py --runs 20        # fewer repetitions (faster)
    python evaluate.py --no-ppo         # rule-based strategies only (no trained model needed)

Fair comparison: for a given (worker, run seed) every strategy faces exactly the
same worker and the same random variability in their step durations ("common
random numbers"). Differences between strategies are therefore *paired*, and we
report the mean paired difference with a 95 % confidence interval.

Outputs (in results/):
    episodes.csv     one row per (worker, seed, strategy) with every metric
    summary.md       tables: means per strategy, savings vs the baselines, per persona
    fig*.png         figures (see plots.py)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from cobuild_env import CobuildDrywallEnv
from profiles import GENERIC, PERSONAS, sample_population
from strategies import LABELS, STRATEGY_KEYS, make_strategy

RESULTS = Path(__file__).with_name('results')
METRICS = ['minutes', 'worker_wait_minutes', 'interruptions', 'strain', 'unnecessary_deliveries',
           'robot_exposure_minutes', 'supported_lifts', 'awkward_handovers', 'wrong_side_handovers',
           'final_fatigue', 'completed']
UNSEEN_SEED = 20261104  # different stream from training, so these workers were never seen
# (reference, challenger): how much does the challenger save compared with the reference?
COMPARISONS = [
    ('A_reactive', 'B_proactive_generic'),
    ('A_reactive', 'C_cobuild_rule'),
    ('A_reactive', 'C_cobuild_ppo'),
    ('B_proactive_generic', 'C_cobuild_rule'),      # value of the profile, same planner
    ('B_proactive_generic', 'C_cobuild_ppo'),
    ('B_proactive_generic', 'ablation_ppo_no_profile'),
    ('ablation_ppo_no_profile', 'C_cobuild_ppo'),   # value of the profile, same learning algorithm
]


def run_episode(strategy, profile, seed):
    env = CobuildDrywallEnv(profile=profile, observe_profile=strategy.observe_profile)
    obs, info = env.reset(seed=seed)
    while True:
        obs, _, terminated, truncated, info = env.step(strategy.act(env, obs))
        if terminated or truncated:
            break
    m = C.TICK_SECONDS / 60.0
    return {
        'minutes': info['ticks'] * m,
        'worker_wait_minutes': info['worker_wait_ticks'] * m,
        'interruptions': info['interruptions'],
        'strain': info['strain'],
        'unnecessary_deliveries': info['unnecessary_deliveries'],
        'robot_exposure_minutes': info['robot_exposure_ticks'] * m,
        'supported_lifts': info['supported_lifts'],
        'awkward_handovers': info['awkward_handovers'],
        'wrong_side_handovers': info['wrong_side_handovers'],
        'final_fatigue': info['final_fatigue'],
        'completed': int(terminated),
    }


def build_workers(n_unseen):
    workers = [('persona', p) for p in [GENERIC, *PERSONAS.values()]]
    rng = np.random.default_rng(UNSEEN_SEED)
    workers += [('unseen', sample_population(rng, name=f'unseen_{i:03d}')) for i in range(n_unseen)]
    return workers


def evaluate(strategy_keys, runs, n_unseen, unseen_runs):
    strategies = [make_strategy(k) for k in strategy_keys]
    rows = []
    for group, profile in build_workers(n_unseen):
        n = runs if group == 'persona' else unseen_runs
        for seed in range(1000, 1000 + n):
            for s in strategies:
                rows.append({'group': group, 'profile': profile.name, 'seed': seed, 'strategy': s.key,
                             'height_m': profile.height_m, 'reach_m': profile.reach_m,
                             'lift_capacity_kg': profile.lift_capacity_kg, 'pace': profile.pace,
                             'handedness': profile.handedness, 'fatigue_rate': profile.fatigue_rate,
                             **run_episode(s, profile, seed)})
    return pd.DataFrame(rows)


def paired_savings(df, ref, other, metric):
    """Mean of (ref - other) over identical (worker, seed) pairs, with a 95 % CI.
    Positive = `other` is better (lower) than `ref`."""
    wide = df.pivot_table(index=['profile', 'seed'], columns='strategy', values=metric)
    d = wide[ref] - wide[other]
    mean, half = d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return mean, half, 100 * mean / wide[ref].mean(), (d > 0).mean() * 100, (d < 0).mean() * 100


def summarize(df):
    keys = [k for k in STRATEGY_KEYS if k in df.strategy.unique()]
    lines = ['# Cobuild drywall evaluation — results', '',
             f'Synthetic simulation, {C.N_BOARDS} boards per wall, 1 tick = {C.TICK_SECONDS} s. '
             'These numbers follow from the assumptions in config.py; they are not field measurements.', '']
    for group, title in [('persona', 'Named personas (incl. the generic average worker)'),
                         ('unseen', 'Unseen random workers (never used in training)')]:
        g = df[df.group == group]
        if g.empty:
            continue
        n_pairs = g[['profile', 'seed']].drop_duplicates().shape[0]
        lines += [f'## {title}', '', f'{g.profile.nunique()} workers, {n_pairs} paired runs per strategy.', '',
                  '| Strategy | Time (min) | Worker waiting (min) | Interruptions | Strain (pts) '
                  '| Unnecessary deliveries | Robot exposure (min) | Supported lifts | Completed |',
                  '|---|---|---|---|---|---|---|---|---|']
        for k in keys:
            s = g[g.strategy == k]
            lines.append(f'| {LABELS[k]} | {s.minutes.mean():.1f} | {s.worker_wait_minutes.mean():.2f} | '
                         f'{s.interruptions.mean():.2f} | {s.strain.mean():.1f} | {s.unnecessary_deliveries.mean():.2f} | '
                         f'{s.robot_exposure_minutes.mean():.2f} | {s.supported_lifts.mean():.2f} | '
                         f'{100 * s.completed.mean():.0f} % |')
        lines += ['', 'Paired savings (positive = the second strategy needs less):', '',
                  '| Comparison | Time saved (min) | Time saved (%) | Strain saved (%) | Runs better / worse |',
                  '|---|---|---|---|---|']
        for ref, other in COMPARISONS:
            if ref in keys and other in keys:
                t, th, tp, better, worse = paired_savings(g, ref, other, 'minutes')
                _, _, sp, _, _ = paired_savings(g, ref, other, 'strain')
                lines.append(f'| {LABELS[ref]} → {LABELS[other]} | {t:.2f} ± {th:.2f} | {tp:.1f} % | {sp:.1f} % '
                             f'| {better:.0f} % / {worse:.0f} % |')
        lines.append('')
    p = df[df.group == 'persona']
    if not p.empty:
        lines += ['## Per persona: mean time (min) and strain', '',
                  '| Persona | ' + ' | '.join(LABELS[k] for k in keys) + ' |', '|---' * (len(keys) + 1) + '|']
        for name in p.profile.unique():
            q = p[p.profile == name]
            cells = [f'{q[q.strategy == k].minutes.mean():.1f} min / {q[q.strategy == k].strain.mean():.0f} pts' for k in keys]
            lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


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
