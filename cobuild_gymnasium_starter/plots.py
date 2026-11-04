"""Figures for the Cobuild evaluation. Called by evaluate.py, or on its own:  python plots.py

fig1_overview.png          time, waiting, strain and heavy handling per strategy (unseen workers)
fig2_time_by_activity.png  where the time goes: minutes per activity, per strategy
fig3_work_saved.png        work saved by the robot vs. no robot, per persona (time and strain)
fig4_profile_value.png     what the worker profile adds on top of robot B, per persona
fig5_paired_runs.png       run-by-run distribution of the saving vs. robot B
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ergonomics as E
from strategies import LABELS, STRATEGY_KEYS

# One colour per strategy in every figure (validated categorical order; "no robot" is neutral grey)
COLORS = {'no_robot': '#8a8984', 'A_reactive': '#2a78d6', 'B_proactive_generic': '#eb6834',
          'C_cobuild_rule': '#1baf7a', 'C_cobuild_ppo': '#eda100', 'ablation_ppo_no_profile': '#e87ba4'}
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SHORT = {'no_robot': '0\nNo robot', 'A_reactive': 'A\nReactive', 'B_proactive_generic': 'B\nProactive\n(average)',
         'C_cobuild_rule': 'C\nCobuild\nrule', 'C_cobuild_ppo': 'C\nCobuild\nPPO',
         'ablation_ppo_no_profile': 'PPO\nno profile'}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.titlesize': 11.5, 'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'font.size': 10,
    'legend.frameon': False,
})
TICK_LABEL = {'no_robot': '0', 'A_reactive': 'A', 'B_proactive_generic': 'B', 'C_cobuild_rule': 'C\nrule',
              'C_cobuild_ppo': 'C\nPPO', 'ablation_ppo_no_profile': 'PPO\nno prof.'}
NOTE = ('Synthetic simulation: absolute values depend on the assumptions in config.py. '
        'Compare strategies with each other, not with real site times.')


def _ci(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _keys(df):
    return [k for k in STRATEGY_KEYS if k in df.strategy.unique()]


def _pct(df, ref, other, metric):
    wide = df.pivot_table(index=['profile', 'seed'], columns='strategy', values=metric)
    return 100 * (wide[ref] - wide[other]).mean() / wide[ref].mean()


def _bars(ax, g, keys, col, title, unit, fmt='{:.1f}'):
    means = [g[g.strategy == k][col].mean() for k in keys]
    cis = [_ci(g[g.strategy == k][col]) for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, means, width=0.62, color=[COLORS[k] for k in keys], edgecolor=SURFACE, linewidth=2,
           yerr=cis, error_kw={'ecolor': INK2, 'elinewidth': 1, 'capsize': 3})
    top = max(np.array(means) + np.array(cis)) if max(means) > 0 else 1
    for xi, m, c in zip(x, means, cis):
        ax.text(xi, m + c + top * 0.02, fmt.format(m), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x, [SHORT[k] for k in keys], fontsize=8)
    ax.set_title(title, loc='left')
    ax.set_ylabel(unit)
    ax.set_ylim(0, top * 1.18)
    ax.grid(axis='x', visible=False)


def fig_overview(df, out):
    g = df[df.group == 'unseen'] if (df.group == 'unseen').any() else df
    keys = _keys(g)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    _bars(axes[0, 0], g, keys, 'minutes', 'Total installation time', 'minutes per wall')
    _bars(axes[0, 1], g, keys, 'worker_wait_minutes', 'Worker waiting for the robot / material', 'minutes per wall')
    _bars(axes[1, 0], g, keys, 'strain', 'Physical strain on the worker', 'strain points (relative)', '{:.0f}')
    _bars(axes[1, 1], g, keys, 'heavy_minutes', 'Time the worker carries the board\'s weight', 'minutes per wall')
    n_w, n_r = g.profile.nunique(), g[['profile', 'seed']].drop_duplicates().shape[0]
    fig.suptitle(f'Installing a 4-board drywall wall: {n_w} unseen simulated workers, {n_r} paired runs per '
                 'strategy (mean ± 95 % CI)', fontsize=13, fontweight='bold', x=0.01, ha='left')
    fig.text(0.01, 0.005, NOTE, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out / 'fig1_overview.png', dpi=160)
    plt.close(fig)


def fig_activity(df, out):
    g = df[df.group == 'unseen'] if (df.group == 'unseen').any() else df
    keys = _keys(g)
    names = {'prep': 'Prep, marking, stool', 'measure': 'Measure', 'cut': 'Cut', 'haul': 'Lift & carry the board',
             'place': 'Place the board', 'screw': 'Screw', 'reload': 'Reload the screw gun',
             'waiting': 'Waiting'}
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, cat in zip(axes.flat, E.CATEGORIES):
        _bars(ax, g, keys, f'min_{cat}', names[cat], 'min per wall')
        ax.set_xticks(np.arange(len(keys)), [TICK_LABEL[k] for k in keys], fontsize=8.5)
    fig.suptitle('Where the worker\'s time goes (minutes per wall, by activity; mean ± 95 % CI). '
                 '0 = no robot, A = reactive, B = proactive (average worker), C = Cobuild',
                 fontsize=12, fontweight='bold', x=0.01, ha='left')
    fig.text(0.01, 0.005, NOTE, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out / 'fig2_time_by_activity.png', dpi=160)
    plt.close(fig)


def _grouped_h(df, challengers, ref, metrics, titles, xlabel, out_name, title):
    rows = [(n, df[(df.group == 'persona') & (df.profile == n)]) for n in df[df.group == 'persona'].profile.unique()]
    if (df.group == 'unseen').any():
        rows.append(('UNSEEN WORKERS (average)', df[df.group == 'unseen']))
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.6 * len(metrics), 0.62 * len(rows) + 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(rows))[::-1]
    h = 0.8 / len(challengers)
    for ax, metric, t in zip(axes, metrics, titles):
        for j, k in enumerate(challengers):
            vals = [_pct(sub, ref, k, metric) for _, sub in rows]
            pos = y + (len(challengers) - 1) * h / 2 - j * h
            ax.barh(pos, vals, height=h, color=COLORS[k], edgecolor=SURFACE, linewidth=2, label=LABELS[k])
            for p, v in zip(pos, vals):
                ax.text(v + (1 if v >= 0 else -1), p, f'{v:.0f} %', va='center', ha='left' if v >= 0 else 'right',
                        fontsize=8)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_title(t, loc='left')
        ax.set_xlabel(xlabel)
        ax.grid(axis='y', visible=False)
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, -5) - 4, hi + 9)
    axes[0].set_yticks(y, [n for n, _ in rows])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(challengers), fontsize=9)
    fig.suptitle(title, fontsize=12, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out_name, dpi=160)
    plt.close(fig)


def fig_work_saved(df, out):
    ch = [k for k in ['A_reactive', 'B_proactive_generic', 'C_cobuild_rule', 'C_cobuild_ppo'] if k in df.strategy.unique()]
    _grouped_h(df, ch, 'no_robot', ['minutes', 'strain', 'heavy_minutes'],
               ['Installation time saved', 'Physical strain saved', 'Time carrying the board saved'],
               '% less than the worker alone (no robot)', out / 'fig3_work_saved.png',
               'How much work does the robot save? Compared with the same worker installing the wall alone')


def fig_profile_value(df, out):
    if 'B_proactive_generic' not in df.strategy.unique():
        return
    ch = [k for k in ['C_cobuild_rule', 'C_cobuild_ppo'] if k in df.strategy.unique()]
    _grouped_h(df, ch, 'B_proactive_generic', ['minutes', 'strain', 'cost_minutes'],
               ['Time saved vs. B', 'Strain saved vs. B', 'Combined cost saved vs. B'],
               '% less than robot B (proactive, tuned for the average worker)', out / 'fig4_profile_value.png',
               'What does the worker profile add? Same planner / same learner, with and without the profile')


def fig_paired(df, out):
    wide_c = df.pivot_table(index=['profile', 'seed'], columns='strategy', values='cost_minutes')
    wide_t = df.pivot_table(index=['profile', 'seed'], columns='strategy', values='minutes')
    ch = [k for k in ['C_cobuild_rule', 'C_cobuild_ppo', 'ablation_ppo_no_profile'] if k in wide_c]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, wide, title in [(axes[0], wide_t, 'Minutes saved in one run vs. B'),
                            (axes[1], wide_c, 'Combined cost (min) saved in one run vs. B')]:
        diffs = {k: wide['B_proactive_generic'] - wide[k] for k in ch}
        lo, hi = min(d.min() for d in diffs.values()), max(d.max() for d in diffs.values())
        bins = np.linspace(np.floor(lo), np.ceil(hi), 45)
        for k, d in diffs.items():
            ax.hist(d, bins=bins, histtype='step', linewidth=2, color=COLORS[k],
                    label=f'{LABELS[k]}: better {100 * (d > 1e-9).mean():.0f} %, worse {100 * (d < -1e-9).mean():.0f} %')
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_xlabel(title + ' (negative = worse than B)')
        ax.set_ylabel('number of runs')
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(f'Run-by-run comparison with robot B ({len(wide_c)} paired runs, personas + unseen workers)',
                 fontsize=12, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / 'fig5_paired_runs.png', dpi=160)
    plt.close(fig)


def make_all(df, out):
    out = Path(out)
    out.mkdir(exist_ok=True)
    fig_overview(df, out)
    fig_activity(df, out)
    if 'no_robot' in df.strategy.unique():
        fig_work_saved(df, out)
    if 'B_proactive_generic' in df.strategy.unique():
        fig_profile_value(df, out)
        fig_paired(df, out)
    print(f'Saved figures to {out}/')


if __name__ == '__main__':
    results = Path(__file__).with_name('results')
    make_all(pd.read_csv(results / 'episodes.csv'), results)
