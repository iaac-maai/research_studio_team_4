"""Two follow-up analyses that use the transparent rule-based planner (fast, no training needed).

    python analysis.py

1. TRAIT CONTRIBUTION - which part of the profile produces the savings?
   The planner is given the generic average worker, except for ONE trait group
   copied from the real worker (e.g. only their handedness). Comparing with B
   (nothing personalised) and with the full profile shows how much each
   physical characteristic or preference is worth.

2. SENSITIVITY - do the conclusions survive different assumptions?
   The B vs. Cobuild-rule comparison is repeated after changing one modelling
   assumption at a time (fatigue, variability, ergonomic penalties, robot
   speed, a more conservative robot B...). If the sign and rough size of the
   savings hold, the result is not an artefact of one arbitrary number.

Outputs: results/analysis.md, results/fig5_trait_contribution.png, results/fig6_sensitivity.png
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from cobuild_env import CobuildDrywallEnv
from evaluate import build_workers
from profiles import GENERIC
from strategies import planner_action

RESULTS = Path(__file__).with_name('results')

TRAIT_GROUPS = {
    'nothing (= robot B)': [],
    'handedness only': ['handedness'],
    'height + reach only': ['height_m', 'reach_m'],
    'lifting capacity only': ['lift_capacity_kg'],
    'pace only': ['pace'],
    'all physical traits (height, reach, capacity)': ['height_m', 'reach_m', 'lift_capacity_kg'],
    'preferences only (handedness, pace)': ['handedness', 'pace'],
    'full profile (= Cobuild rule)': ['handedness', 'height_m', 'reach_m', 'lift_capacity_kg', 'pace', 'fatigue_rate'],
}


def run(worker, seed, believed):
    """One episode where the planner believes the worker looks like `believed`."""
    env = CobuildDrywallEnv(profile=worker)
    env.reset(seed=seed)
    while True:
        _, _, terminated, truncated, info = env.step(planner_action(env, believed))
        if terminated or truncated:
            return info['ticks'] * C.TICK_SECONDS / 60.0, info['strain']


def partial_profile(worker, traits, base=GENERIC):
    return replace(base, **{t: getattr(worker, t) for t in traits})


def trait_contribution(workers, seeds):
    rows = []
    for _, w in workers:
        for seed in seeds:
            for group, traits in TRAIT_GROUPS.items():
                minutes, strain = run(w, seed, partial_profile(w, traits))
                rows.append({'worker': w.name, 'seed': seed, 'group': group, 'minutes': minutes, 'strain': strain})
    df = pd.DataFrame(rows)
    base = df[df.group == 'nothing (= robot B)'].set_index(['worker', 'seed'])
    out = []
    for group in TRAIT_GROUPS:
        g = df[df.group == group].set_index(['worker', 'seed'])
        out.append({'personalised': group,
                    'time_saved_pct': 100 * (base.minutes - g.minutes).mean() / base.minutes.mean(),
                    'strain_saved_pct': 100 * (base.strain - g.strain).mean() / base.strain.mean()})
    return pd.DataFrame(out)


# name -> (config overrides, profile robot B is designed for)
SCENARIOS = {
    'baseline assumptions': ({}, GENERIC),
    'no fatigue': ({'FATIGUE_SCALE': 0.0}, GENERIC),
    'double fatigue': ({'FATIGUE_SCALE': 2.0}, GENERIC),
    'low variability (sigma 0.05)': ({'DURATION_NOISE_SIGMA': 0.05}, GENERIC),
    'high variability (sigma 0.30)': ({'DURATION_NOISE_SIGMA': 0.30}, GENERIC),
    'ergonomic penalties halved': ({'WRONG_SIDE_STRAIN': 0.5, 'OUT_OF_REACH_STRAIN': 0.75,
                                    'HEIGHT_STRAIN_PER_10CM': 1.0, 'LIFT_STRAIN_BASE': 1.5}, GENERIC),
    'wrong side costs no time': ({'WRONG_SIDE_TICKS': 0}, GENERIC),
    'strain weight halved (R_STRAIN -0.25)': ({'R_STRAIN': -0.25}, GENERIC),
    'strain weight doubled (R_STRAIN -1.0)': ({'R_STRAIN': -1.0}, GENERIC),
    'faster robot (fetch board 2, screws 1)': ({'FETCH_TICKS': {'board': 2, 'screws': 1}}, GENERIC),
    'B designed for short reach (always near)': ({}, replace(GENERIC, reach_m=0.56)),
}


def sensitivity(workers, seeds):
    rows = []
    for name, (overrides, b_profile) in SCENARIOS.items():
        saved = {k: getattr(C, k) for k in overrides}
        for k, v in overrides.items():
            setattr(C, k, v)
        try:
            b_t, b_s, c_t, c_s = [], [], [], []
            for _, w in workers:
                for seed in seeds:
                    t, s = run(w, seed, b_profile)
                    b_t.append(t), b_s.append(s)
                    t, s = run(w, seed, w)
                    c_t.append(t), c_s.append(s)
        finally:
            for k, v in saved.items():
                setattr(C, k, v)
        b_t, b_s, c_t, c_s = map(np.array, (b_t, b_s, c_t, c_s))
        d = b_t - c_t
        rows.append({'scenario': name, 'B_minutes': b_t.mean(), 'C_minutes': c_t.mean(),
                     'time_saved_pct': 100 * d.mean() / b_t.mean(),
                     'strain_saved_pct': 100 * (b_s - c_s).mean() / b_s.mean(),
                     'runs_C_better_pct': 100 * (d > 0).mean(), 'runs_C_worse_pct': 100 * (d < 0).mean()})
    return pd.DataFrame(rows)


def plot(contrib, sens, out):
    import plots  # applies the shared figure style
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    y = np.arange(len(contrib))[::-1]
    for ax, col, title in [(axes[0], 'time_saved_pct', 'Time saved vs. B'),
                           (axes[1], 'strain_saved_pct', 'Physical strain saved vs. B')]:
        ax.barh(y, contrib[col], height=0.6, color=plots.COLORS['C_cobuild_rule'], edgecolor=plots.SURFACE, linewidth=2)
        for yi, v in zip(y, contrib[col]):
            ax.text(v + (0.5 if v >= 0 else -0.5), yi, f'{v:.1f} %', va='center', ha='left' if v >= 0 else 'right', fontsize=9)
        ax.axvline(0, color=plots.INK2, linewidth=1)
        ax.set_title(title, loc='left')
        ax.set_xlabel('% less than robot B')
        ax.grid(axis='y', visible=False)
        ax.set_xlim(min(0, contrib[col].min()) - 3, contrib[col].max() * 1.2 + 2)
    axes[0].set_yticks(y, ['Planner knows: ' + p for p in contrib.personalised])
    fig.suptitle('Which part of the worker profile matters? (same planner, one trait group personalised at a time)',
                 fontsize=12, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / 'fig5_trait_contribution.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(sens) + 1.8), sharey=True)
    y = np.arange(len(sens))[::-1]
    for ax, col, title in [(axes[0], 'time_saved_pct', 'Time saved: Cobuild rule vs. B'),
                           (axes[1], 'strain_saved_pct', 'Strain saved: Cobuild rule vs. B')]:
        colors = ['#8a8984' if i == 0 else plots.COLORS['C_cobuild_rule'] for i in range(len(sens))]  # grey = baseline
        ax.barh(y, sens[col], height=0.6, color=colors, edgecolor=plots.SURFACE, linewidth=2)
        for yi, v in zip(y, sens[col]):
            ax.text(v + 0.5, yi, f'{v:.1f} %', va='center', fontsize=8.5)
        ax.axvline(0, color=plots.INK2, linewidth=1)
        ax.set_title(title, loc='left')
        ax.set_xlabel('% less than robot B')
        ax.grid(axis='y', visible=False)
        ax.set_xlim(min(0, sens[col].min()) - 2, sens[col].max() * 1.2 + 2)
    axes[0].set_yticks(y, sens.scenario)
    fig.suptitle('Sensitivity: do the savings survive other assumptions? (first bar = baseline assumptions)',
                 fontsize=12, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / 'fig6_sensitivity.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    workers = build_workers(n_unseen=60)
    seeds = range(1000, 1010)
    contrib = trait_contribution(workers, seeds)
    sens = sensitivity(workers, seeds)
    RESULTS.mkdir(exist_ok=True)
    text = ('# Cobuild — trait contribution and sensitivity\n\n'
            f'Rule-based planner, {len(workers)} workers (personas + unseen) x {len(seeds)} paired runs.\n\n'
            '## 1. Which part of the profile produces the savings?\n\n'
            + contrib.to_markdown(index=False, floatfmt='.1f')
            + '\n\nThe rows do not add up to the full profile: traits interact (e.g. height changes both the '
              'best handover height and whether holding the board pays off).\n\n'
            '## 2. Sensitivity to modelling assumptions\n\n'
            + sens.to_markdown(index=False, floatfmt='.1f') + '\n')
    (RESULTS / 'analysis.md').write_text(text)
    print(text)
    plot(contrib, sens, RESULTS)
    print(f'Saved {RESULTS / "analysis.md"}, fig5_trait_contribution.png, fig6_sensitivity.png')
