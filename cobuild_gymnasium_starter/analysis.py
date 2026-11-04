"""Two follow-up analyses with the transparent rule-based planner (fast, no training needed).

    python analysis.py

1. TRAIT CONTRIBUTION: which part of the profile produces the gain?
   The planner is given the generic average worker, except for ONE trait group copied from the real
   worker. Comparing with B (nothing personalised) and with the full profile shows what each
   characteristic is worth.

2. SENSITIVITY: do the conclusions survive different assumptions?
   Many robot-side numbers are assumptions (docs/RESEARCH.md §6). Here each one is changed, one at a
   time, and two questions are asked again:
     - what does the robot save compared with the worker alone?     (robot B vs no robot)
     - what does the profile add on top of that?                      (Cobuild rule vs B)

Outputs: results/analysis.md, results/fig6_trait_contribution.png, results/fig7_sensitivity.png
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
COST_PER_STRAIN_MIN = 0.05   # fixed at the baseline weight so scenarios stay comparable


def run(worker, seed, believed):
    """One episode in which the planner believes the worker looks like `believed` (None = no robot)."""
    env = CobuildDrywallEnv(profile=worker, robot_present=believed is not None)
    env.reset(seed=seed)
    while True:
        a = 0 if believed is None else planner_action(env, believed)
        _, _, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            minutes, strain = info['minutes'], info['strain']
            return minutes, strain, minutes + COST_PER_STRAIN_MIN * strain


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


def partial_profile(worker, traits):
    return replace(GENERIC, **{t: getattr(worker, t) for t in traits})


def trait_contribution(workers, seeds):
    rows = []
    for _, w in workers:
        for seed in seeds:
            for group, traits in TRAIT_GROUPS.items():
                m, s, c = run(w, seed, partial_profile(w, traits))
                rows.append({'worker': w.name, 'seed': seed, 'group': group, 'minutes': m, 'strain': s, 'cost': c})
    df = pd.DataFrame(rows)
    base = df[df.group == 'nothing (= robot B)'].set_index(['worker', 'seed'])
    out = []
    for group in TRAIT_GROUPS:
        g = df[df.group == group].set_index(['worker', 'seed'])
        out.append({'personalised': group,
                    'time_saved_pct': 100 * (base.minutes - g.minutes).mean() / base.minutes.mean(),
                    'strain_saved_pct': 100 * (base.strain - g.strain).mean() / base.strain.mean(),
                    'cost_saved_pct': 100 * (base.cost - g.cost).mean() / base.cost.mean()})
    return pd.DataFrame(out)


STRONG, WEAK = replace(GENERIC, lift_capacity_kg=45.0), replace(GENERIC, lift_capacity_kg=24.0)
# name -> (config overrides, profile that robot B is designed for)
SCENARIOS = {
    'baseline assumptions': ({}, GENERIC),
    'no fatigue': ({'FATIGUE_SCALE': 0.0}, GENERIC),
    'double fatigue': ({'FATIGUE_SCALE': 1.2}, GENERIC),
    'low variability (sigma 0.05)': ({'DURATION_NOISE_SIGMA': 0.05}, GENERIC),
    'high variability (sigma 0.30)': ({'DURATION_NOISE_SIGMA': 0.30}, GENERIC),
    'light board (26 kg, 1/2")': ({'BOARD_KG': 26.0}, GENERIC),
    'heavy board (42 kg, Yuan density)': ({'BOARD_KG': 42.0}, GENERIC),
    'lifting never slows the worker': ({'LIFT_LOAD_SLOWDOWN': 0.0}, GENERIC),
    'lifting slows 2x more': ({'LIFT_LOAD_SLOWDOWN': 3.0}, GENERIC),
    'no overhead penalty': ({'OVERHEAD_FACTOR_PER_M': 0.0, 'STRAIN_SCREW_OVERHEAD_PER_M': 0.0,
                             'SCREW_OVERHEAD_SLOWDOWN_PER_M': 0.0}, GENERIC),
    'robot alignment takes 1 tick': ({'ALIGN_TICKS': 1.0}, GENERIC),
    'robot alignment takes 4 ticks': ({'ALIGN_TICKS': 4.0}, GENERIC),
    'slower robot (+2 s per carry, tools 3 ticks)': (
        {'BRING_TICKS': {k: v + 2 for k, v in C.BRING_TICKS.items()}, 'FETCH_TOOL_TICKS': 3}, GENERIC),
    'steadying at the cut station has no effect': (
        {'STEADY_CUT_TIME_FACTOR': 1.0, 'STEADY_CUT_STRAIN_FACTOR': 1.0}, GENERIC),
    'tool-handover penalties off': (
        {'WRONG_SIDE_TICKS': 0.0, 'WRONG_SIDE_STRAIN': 0.0, 'OUT_OF_REACH_TICKS': 0.0, 'OUT_OF_REACH_STRAIN': 0.0,
         'HEIGHT_MISMATCH_TICKS': 0.0, 'HEIGHT_STRAIN_PER_10CM': 0.0}, GENERIC),
    'planner values strain half as much (R_STRAIN -0.25)': ({'R_STRAIN': -0.25}, GENERIC),
    'planner values strain twice as much (R_STRAIN -1.0)': ({'R_STRAIN': -1.0}, GENERIC),
    'B designed for a strong worker (45 kg)': ({}, STRONG),
    'B designed for a weak worker (24 kg)': ({}, WEAK),
}


def sensitivity(workers, seeds):
    rows = []
    for name, (overrides, b_profile) in SCENARIOS.items():
        saved = {k: getattr(C, k) for k in overrides}
        for k, v in overrides.items():
            setattr(C, k, v)
        try:
            res = {'none': [], 'B': [], 'C': []}
            for _, w in workers:
                for seed in seeds:
                    res['none'].append(run(w, seed, None))
                    res['B'].append(run(w, seed, b_profile))
                    res['C'].append(run(w, seed, w))
        finally:
            for k, v in saved.items():
                setattr(C, k, v)
        n, b, c = (np.array(res[k]) for k in ('none', 'B', 'C'))
        pct = lambda ref, oth, i: 100 * (ref[:, i] - oth[:, i]).mean() / ref[:, i].mean()
        rows.append({'scenario': name,
                     'robot_time_saved_pct': pct(n, b, 0), 'robot_strain_saved_pct': pct(n, b, 1),
                     'robot_cost_saved_pct': pct(n, b, 2),
                     'profile_time_saved_pct': pct(b, c, 0), 'profile_strain_saved_pct': pct(b, c, 1),
                     'profile_cost_saved_pct': pct(b, c, 2),
                     'profile_better_pct': 100 * (b[:, 2] - c[:, 2] > 1e-9).mean(),
                     'profile_worse_pct': 100 * (b[:, 2] - c[:, 2] < -1e-9).mean()})
    return pd.DataFrame(rows)


def plot(contrib, sens, out):
    import matplotlib.pyplot as plt
    import plots  # shared figure style and colours

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    y = np.arange(len(contrib))[::-1]
    for ax, col, title in [(axes[0], 'time_saved_pct', 'Time saved vs. B'),
                           (axes[1], 'strain_saved_pct', 'Strain saved vs. B'),
                           (axes[2], 'cost_saved_pct', 'Combined cost saved vs. B')]:
        ax.barh(y, contrib[col], height=0.6, color=plots.COLORS['C_cobuild_rule'], edgecolor=plots.SURFACE, linewidth=2)
        for yi, v in zip(y, contrib[col]):
            ax.text(v + (0.4 if v >= 0 else -0.4), yi, f'{v:.1f} %', va='center', ha='left' if v >= 0 else 'right', fontsize=8.5)
        ax.axvline(0, color=plots.INK2, linewidth=1)
        ax.set_title(title, loc='left')
        ax.set_xlabel('% less than robot B')
        ax.grid(axis='y', visible=False)
        ax.set_xlim(min(0, contrib[col].min()) - 2, contrib[col].max() * 1.25 + 2)
    axes[0].set_yticks(y, ['Planner knows: ' + p for p in contrib.personalised], fontsize=8.5)
    fig.suptitle('Which part of the worker profile matters? (same planner, one trait group personalised at a time)',
                 fontsize=12, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / 'fig6_trait_contribution.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(sens) + 2.0), sharey=True)
    y = np.arange(len(sens))[::-1]
    for ax, col, title, color in [
            (axes[0], 'robot_cost_saved_pct', 'What the robot saves\n(robot B vs. worker alone, combined cost)', plots.COLORS['B_proactive_generic']),
            (axes[1], 'profile_cost_saved_pct', 'What the profile adds\n(Cobuild rule vs. robot B, combined cost)', plots.COLORS['C_cobuild_rule'])]:
        colors = ['#8a8984' if i == 0 else color for i in range(len(sens))]
        ax.barh(y, sens[col], height=0.6, color=colors, edgecolor=plots.SURFACE, linewidth=2)
        for yi, v in zip(y, sens[col]):
            ax.text(v + (0.4 if v >= 0 else -0.4), yi, f'{v:.1f} %', va='center', ha='left' if v >= 0 else 'right', fontsize=8.5)
        ax.axvline(0, color=plots.INK2, linewidth=1)
        ax.set_title(title, loc='left')
        ax.set_xlabel('% of combined cost (grey bar = baseline assumptions)')
        ax.grid(axis='y', visible=False)
        ax.set_xlim(min(0, sens[col].min()) - 2, sens[col].max() * 1.2 + 2)
    axes[0].set_yticks(y, sens.scenario, fontsize=8.5)
    fig.suptitle('Sensitivity: do the conclusions survive other assumptions?', fontsize=12, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / 'fig7_sensitivity.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    workers = build_workers(n_unseen=60)
    seeds = range(1000, 1010)
    contrib = trait_contribution(workers, seeds)
    sens = sensitivity(workers, seeds)
    RESULTS.mkdir(exist_ok=True)
    text = ('# Cobuild: trait contribution and sensitivity\n\n'
            f'Rule-based planner, {len(workers)} workers (personas + unseen) x {len(seeds)} paired runs. '
            'Combined cost = minutes + 0.05 x strain points.\n\n'
            '## 1. Which part of the profile produces the gain?\n\n'
            + contrib.to_markdown(index=False, floatfmt='.1f')
            + '\n\nThe rows do not add up to the full profile: traits interact (for example, height changes both the '
              'best handover height and whether holding the board pays off).\n\n'
            '## 2. Sensitivity to modelling assumptions\n\n'
            + sens.to_markdown(index=False, floatfmt='.1f') + '\n')
    (RESULTS / 'analysis.md').write_text(text)
    print(text)
    plot(contrib, sens, RESULTS)
    print(f'Saved {RESULTS / "analysis.md"}, fig6_trait_contribution.png, fig7_sensitivity.png')
