"""Figures for the Cobuild evaluation. Called by evaluate.py, or on its own:

    python plots.py            # re-draw the figures from results/episodes.csv

fig1_overview.png        time, waiting, strain and robot exposure per strategy (unseen workers)
fig2_savings_by_worker.png  % time and % strain saved vs. robot B, for every persona
fig3_traits.png          which physical traits drive the savings (unseen workers)
fig4_paired_runs.png     distribution of per-run time saved vs. B (is it always better?)
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies import LABELS, STRATEGY_KEYS

# Validated categorical palette (fixed order, one colour per strategy in every figure)
COLORS = dict(zip(STRATEGY_KEYS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']))
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SHORT = {
    'A_reactive': 'A\nReactive',
    'B_proactive_generic': 'B\nProactive\n(average)',
    'C_cobuild_rule': 'C\nCobuild rule\n(profile)',
    'C_cobuild_ppo': 'C\nCobuild PPO\n(profile)',
    'ablation_ppo_no_profile': 'PPO\nno profile\n(ablation)',
}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.axisbelow': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'font.size': 10, 'legend.frameon': False,
})


def _ci(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _keys(df):
    return [k for k in STRATEGY_KEYS if k in df.strategy.unique()]


def _cobuild_key(df):
    return 'C_cobuild_ppo' if 'C_cobuild_ppo' in df.strategy.unique() else 'C_cobuild_rule'


def fig_overview(df, out):
    g = df[df.group == 'unseen'] if (df.group == 'unseen').any() else df
    keys = _keys(g)
    panels = [('minutes', 'Total installation time', 'minutes per wall'),
              ('worker_wait_minutes', 'Worker waiting for material / robot', 'minutes per wall'),
              ('strain', 'Physical strain on the worker', 'strain points (relative)'),
              ('robot_exposure_minutes', 'Robot hovering in the worker\'s space', 'minutes per wall')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (col, title, unit) in zip(axes.flat, panels):
        means = [g[g.strategy == k][col].mean() for k in keys]
        cis = [_ci(g[g.strategy == k][col]) for k in keys]
        x = np.arange(len(keys))
        ax.bar(x, means, width=0.62, color=[COLORS[k] for k in keys], edgecolor=SURFACE, linewidth=2,
               yerr=cis, error_kw={'ecolor': INK2, 'elinewidth': 1, 'capsize': 3})
        for xi, m, c in zip(x, means, cis):
            ax.text(xi, m + c + max(means) * 0.02, f'{m:.1f}', ha='center', va='bottom', fontsize=9, color=INK)
        ax.set_xticks(x, [SHORT[k] for k in keys], fontsize=8.5)
        ax.set_title(title, loc='left')
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(np.array(means) + np.array(cis)) * 1.18)
        ax.grid(axis='x', visible=False)
    n_workers, n_runs = g.profile.nunique(), g[['profile', 'seed']].drop_duplicates().shape[0]
    fig.suptitle(f'Installing a 4-board drywall wall: {n_workers} unseen simulated workers, '
                 f'{n_runs} paired runs per strategy (mean ± 95 % CI)', fontsize=13, fontweight='bold', x=0.01, ha='left')
    fig.text(0.01, 0.005, 'Synthetic simulation: absolute values depend on the assumptions in config.py. '
             'Compare strategies with each other, not with real site times.', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out / 'fig1_overview.png', dpi=160)
    plt.close(fig)


def _pct_saved(df, ref, other, metric):
    wide = df.pivot_table(index=['profile', 'seed'], columns='strategy', values=metric)
    return 100 * (wide[ref] - wide[other]).mean() / wide[ref].mean()


def fig_savings_by_worker(df, out):
    ref = 'B_proactive_generic'
    challengers = [k for k in ['C_cobuild_rule', 'C_cobuild_ppo'] if k in df.strategy.unique()]
    rows = [(name, df[(df.group == 'persona') & (df.profile == name)]) for name in df[df.group == 'persona'].profile.unique()]
    if (df.group == 'unseen').any():
        rows.append(('UNSEEN WORKERS (average)', df[df.group == 'unseen']))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * len(rows) + 2.2), sharey=True)
    y = np.arange(len(rows))[::-1]
    h = 0.8 / len(challengers)
    for ax, (metric, title) in zip(axes, [('minutes', 'Time saved vs. B'), ('strain', 'Physical strain saved vs. B')]):
        for j, k in enumerate(challengers):
            vals = [_pct_saved(sub, ref, k, metric) for _, sub in rows]
            pos = y + (len(challengers) - 1) * h / 2 - j * h
            ax.barh(pos, vals, height=h, color=COLORS[k], edgecolor=SURFACE, linewidth=2, label=LABELS[k])
            for p, v in zip(pos, vals):
                ax.text(v + (1 if v >= 0 else -1), p, f'{v:.0f} %', va='center',
                        ha='left' if v >= 0 else 'right', fontsize=8, color=INK)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_title(title, loc='left')
        ax.set_xlabel('% less than robot B (proactive, tuned for the average worker)')
        ax.grid(axis='y', visible=False)
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, -5) - 4, hi + 8)
    axes[0].set_yticks(y, [name for name, _ in rows])
    axes[0].legend(loc='lower right')
    fig.suptitle('Who benefits from a worker profile? (paired runs, same worker and same variability for every strategy)',
                 fontsize=12, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'fig2_savings_by_worker.png', dpi=160)
    plt.close(fig)


def fig_traits(df, out):
    g = df[df.group == 'unseen']
    if g.empty:
        return
    k = _cobuild_key(df)
    wide = g.pivot_table(index=['profile', 'seed'], columns='strategy', values='minutes')
    saved = (wide['B_proactive_generic'] - wide[k]).groupby('profile').mean()
    traits = g.drop_duplicates('profile').set_index('profile').loc[saved.index]
    panels = [('height_m', 'Height (m)'), ('lift_capacity_kg', 'Lifting capacity (kg)'),
              ('reach_m', 'Comfortable reach (m)'), ('pace', 'Pace multiplier (higher = slower)')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, (col, label) in zip(axes, panels):
        for hand, filled in [('right', True), ('left', False)]:
            m = traits.handedness == hand
            ax.scatter(traits[col][m], saved[m], s=34, linewidths=1.3, edgecolors=INK2,
                       facecolors=INK2 if filled else 'none', label=f'{hand}-handed')
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xlabel(label)
    axes[0].set_ylabel('minutes saved per wall vs. B')
    axes[0].legend(loc='upper right')
    fig.suptitle(f'Which physical traits drive the savings? {LABELS[k]} vs. B, '
                 f'{len(saved)} unseen workers (each dot = mean of its runs)',
                 fontsize=12, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / 'fig3_traits.png', dpi=160)
    plt.close(fig)


def fig_paired_runs(df, out):
    wide = df.pivot_table(index=['profile', 'seed'], columns='strategy', values='minutes')
    challengers = [k for k in ['C_cobuild_rule', 'C_cobuild_ppo', 'ablation_ppo_no_profile'] if k in wide]
    diffs = {k: wide['B_proactive_generic'] - wide[k] for k in challengers}
    lo = min(d.min() for d in diffs.values())
    hi = max(d.max() for d in diffs.values())
    bins = np.linspace(np.floor(lo), np.ceil(hi), 50)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, d in diffs.items():
        ax.hist(d, bins=bins, histtype='step', linewidth=2, color=COLORS[k],
                label=f'{LABELS[k]}: better in {100 * (d > 0).mean():.0f} % of runs, worse in {100 * (d < 0).mean():.0f} %')
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel('minutes saved in one run vs. B (negative = slower than B)')
    ax.set_ylabel('number of runs')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f'Run-by-run comparison with robot B ({len(wide)} paired runs, personas + unseen workers)', loc='left')
    fig.tight_layout()
    fig.savefig(out / 'fig4_paired_runs.png', dpi=160)
    plt.close(fig)


def make_all(df, out):
    out = Path(out)
    out.mkdir(exist_ok=True)
    fig_overview(df, out)
    if 'B_proactive_generic' in df.strategy.unique():
        fig_savings_by_worker(df, out)
        fig_traits(df, out)
        fig_paired_runs(df, out)
    print(f'Saved figures to {out}/')


if __name__ == '__main__':
    results = Path(__file__).with_name('results')
    make_all(pd.read_csv(results / 'episodes.csv'), results)
