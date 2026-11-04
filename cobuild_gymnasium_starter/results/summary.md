# Cobuild drywall installation: evaluation results

Synthetic simulation. A wall of 4 boards (2 whole + 2 trimmed), 1 tick = 6 s. Numbers follow from the assumptions in config.py (see docs/RESEARCH.md for what is calibrated to published data and what is assumed); they are not field measurements.

## Named personas (incl. the generic average worker)

9 workers, 450 paired runs per strategy.

| Strategy | Time (min) | Worker waiting (min) | Interruptions | Strain (pts) | Heavy handling (min) | Combined cost (min) | Boards held | Unnecessary cmds |
|---|---|---|---|---|---|---|---|---|
| 0 · No robot | 21.2 | 0.00 | 0.0 | 128 | 7.8 | 27.6 | 0.0 | 0.00 |
| A · Reactive | 20.1 | 5.60 | 15.0 | 36 | 0.0 | 21.9 | 4.0 | 0.00 |
| B · Proactive (average worker) | 15.4 | 1.23 | 2.2 | 55 | 1.6 | 18.2 | 2.0 | 0.00 |
| C · Cobuild rule (profile) | 15.2 | 1.22 | 2.1 | 44 | 1.3 | 17.4 | 2.2 | 0.00 |
| C · Cobuild PPO (profile) | 15.1 | 1.05 | 2.1 | 46 | 1.5 | 17.4 | 1.9 | 0.00 |
| PPO without profile (ablation) | 15.9 | 1.24 | 2.2 | 39 | 0.7 | 17.9 | 3.0 | 0.00 |

Where does the time go? Mean minutes per wall by activity:

| Strategy | prep | measure | cut | haul | place | screw | reload | waiting |
|---|---|---|---|---|---|---|---|---|
| 0 · No robot | 3.9 | 1.5 | 2.6 | 4.5 | 1.4 | 3.9 | 3.4 | 0.0 |
| A · Reactive | 3.4 | 1.5 | 2.5 | 0.0 | 0.8 | 3.7 | 2.5 | 5.6 |
| B · Proactive (average worker) | 3.4 | 1.5 | 2.2 | 0.0 | 1.1 | 3.8 | 2.2 | 1.2 |
| C · Cobuild rule (profile) | 3.4 | 1.5 | 2.2 | 0.0 | 0.9 | 3.7 | 2.3 | 1.2 |
| C · Cobuild PPO (profile) | 3.4 | 1.5 | 2.2 | 0.0 | 1.0 | 3.7 | 2.3 | 1.1 |
| PPO without profile (ablation) | 3.4 | 1.5 | 2.2 | 0.0 | 0.9 | 3.7 | 3.0 | 1.2 |

Paired savings (positive = the second strategy needs less):

| Comparison | Time saved | Strain saved | Heavy handling saved | Combined cost saved | Runs better / worse (cost) |
|---|---|---|---|---|---|
| 0 · No robot → A · Reactive | 1.08 ± 0.17 min (5.1 %) | 72.0 % | 100.0 % | 5.70 ± 0.35 min (20.7 %) | 100 % / 0 % |
| 0 · No robot → B · Proactive (average worker) | 5.77 ± 0.16 min (27.2 %) | 57.0 % | 79.4 % | 9.42 ± 0.30 min (34.2 %) | 100 % / 0 % |
| 0 · No robot → C · Cobuild rule (profile) | 6.00 ± 0.14 min (28.4 %) | 65.9 % | 83.4 % | 10.23 ± 0.34 min (37.1 %) | 100 % / 0 % |
| 0 · No robot → C · Cobuild PPO (profile) | 6.08 ± 0.13 min (28.7 %) | 64.0 % | 80.8 % | 10.18 ± 0.33 min (36.9 %) | 100 % / 0 % |
| B · Proactive (average worker) → C · Cobuild rule (profile) | 0.24 ± 0.06 min (1.5 %) | 20.7 % | 19.1 % | 0.81 ± 0.06 min (4.4 %) | 78 % / 2 % |
| B · Proactive (average worker) → C · Cobuild PPO (profile) | 0.31 ± 0.04 min (2.0 %) | 16.2 % | 6.6 % | 0.76 ± 0.06 min (4.2 %) | 85 % / 15 % |
| B · Proactive (average worker) → PPO without profile (ablation) | -0.50 ± 0.04 min (-3.2 %) | 28.6 % | 57.6 % | 0.29 ± 0.08 min (1.6 %) | 51 % / 49 % |
| PPO without profile (ablation) → C · Cobuild PPO (profile) | 0.81 ± 0.06 min (5.1 %) | -17.5 % | -120.3 % | 0.46 ± 0.04 min (2.6 %) | 98 % / 2 % |

## Unseen random workers (never used in training)

100 workers, 500 paired runs per strategy.

| Strategy | Time (min) | Worker waiting (min) | Interruptions | Strain (pts) | Heavy handling (min) | Combined cost (min) | Boards held | Unnecessary cmds |
|---|---|---|---|---|---|---|---|---|
| 0 · No robot | 21.1 | 0.00 | 0.0 | 126 | 7.8 | 27.4 | 0.0 | 0.00 |
| A · Reactive | 20.0 | 5.60 | 15.0 | 34 | 0.0 | 21.7 | 4.0 | 0.00 |
| B · Proactive (average worker) | 15.3 | 1.23 | 2.2 | 52 | 1.5 | 17.9 | 2.0 | 0.00 |
| C · Cobuild rule (profile) | 15.1 | 1.24 | 2.1 | 43 | 1.3 | 17.3 | 2.2 | 0.00 |
| C · Cobuild PPO (profile) | 15.0 | 1.06 | 2.2 | 45 | 1.6 | 17.3 | 1.8 | 0.00 |
| PPO without profile (ablation) | 15.8 | 1.23 | 2.2 | 38 | 0.7 | 17.7 | 3.0 | 0.00 |

Where does the time go? Mean minutes per wall by activity:

| Strategy | prep | measure | cut | haul | place | screw | reload | waiting |
|---|---|---|---|---|---|---|---|---|
| 0 · No robot | 4.0 | 1.5 | 2.6 | 4.6 | 1.3 | 3.8 | 3.4 | 0.0 |
| A · Reactive | 3.5 | 1.5 | 2.5 | 0.0 | 0.8 | 3.6 | 2.4 | 5.6 |
| B · Proactive (average worker) | 3.5 | 1.5 | 2.2 | 0.0 | 1.1 | 3.6 | 2.2 | 1.2 |
| C · Cobuild rule (profile) | 3.5 | 1.4 | 2.2 | 0.0 | 1.0 | 3.6 | 2.2 | 1.2 |
| C · Cobuild PPO (profile) | 3.5 | 1.4 | 2.2 | 0.0 | 1.0 | 3.6 | 2.2 | 1.1 |
| PPO without profile (ablation) | 3.5 | 1.5 | 2.1 | 0.0 | 0.9 | 3.6 | 3.0 | 1.2 |

Paired savings (positive = the second strategy needs less):

| Comparison | Time saved | Strain saved | Heavy handling saved | Combined cost saved | Runs better / worse (cost) |
|---|---|---|---|---|---|
| 0 · No robot → A · Reactive | 1.11 ± 0.17 min (5.2 %) | 73.3 % | 100.0 % | 5.73 ± 0.36 min (20.9 %) | 100 % / 0 % |
| 0 · No robot → B · Proactive (average worker) | 5.85 ± 0.16 min (27.7 %) | 58.8 % | 80.9 % | 9.56 ± 0.31 min (34.9 %) | 100 % / 0 % |
| 0 · No robot → C · Cobuild rule (profile) | 6.00 ± 0.14 min (28.4 %) | 65.9 % | 83.3 % | 10.15 ± 0.33 min (37.0 %) | 100 % / 0 % |
| 0 · No robot → C · Cobuild PPO (profile) | 6.11 ± 0.13 min (29.0 %) | 64.0 % | 80.1 % | 10.15 ± 0.32 min (37.0 %) | 100 % / 0 % |
| B · Proactive (average worker) → C · Cobuild rule (profile) | 0.15 ± 0.07 min (1.0 %) | 17.1 % | 12.4 % | 0.60 ± 0.04 min (3.3 %) | 84 % / 6 % |
| B · Proactive (average worker) → C · Cobuild PPO (profile) | 0.26 ± 0.05 min (1.7 %) | 12.5 % | -4.5 % | 0.59 ± 0.04 min (3.3 %) | 94 % / 6 % |
| B · Proactive (average worker) → PPO without profile (ablation) | -0.57 ± 0.03 min (-3.7 %) | 27.3 % | 51.8 % | 0.14 ± 0.05 min (0.8 %) | 59 % / 41 % |
| PPO without profile (ablation) → C · Cobuild PPO (profile) | 0.83 ± 0.05 min (5.3 %) | -20.3 % | -116.7 % | 0.45 ± 0.03 min (2.5 %) | 93 % / 1 % |

## Per persona: mean time (min) / strain (pts)

| Persona | 0 · No robot | A · Reactive | B · Proactive (average worker) | C · Cobuild rule (profile) | C · Cobuild PPO (profile) | PPO without profile (ablation) |
|---|---|---|---|---|---|---|
| generic_average | 20.0 / 109 | 19.7 / 32 | 14.8 / 47 | 14.8 / 47 | 14.9 / 45 | 15.7 / 38 |
| expert_tall_right | 14.9 / 63 | 16.1 / 21 | 11.6 / 28 | 10.5 / 37 | 10.5 / 38 | 12.5 / 24 |
| expert_average_left | 16.9 / 86 | 17.9 / 29 | 13.1 / 41 | 12.7 / 38 | 12.3 / 44 | 13.4 / 31 |
| novice_average_right | 26.4 / 142 | 24.0 / 42 | 19.0 / 62 | 19.1 / 64 | 19.2 / 60 | 20.1 / 49 |
| short_left | 21.7 / 154 | 20.6 / 45 | 16.0 / 71 | 16.4 / 38 | 16.0 / 45 | 16.0 / 46 |
| short_strong_right | 18.2 / 94 | 18.8 / 37 | 14.0 / 50 | 13.8 / 42 | 13.2 / 51 | 14.6 / 39 |
| tall_novice_left | 23.5 / 103 | 22.2 / 33 | 17.2 / 46 | 15.8 / 56 | 17.1 / 41 | 17.9 / 35 |
| low_capacity_right | 23.1 / 196 | 19.4 / 35 | 15.1 / 68 | 15.5 / 31 | 15.1 / 41 | 15.3 / 43 |
| low_capacity_short_left | 25.9 / 208 | 22.1 / 48 | 17.8 / 84 | 18.0 / 41 | 17.5 / 50 | 17.6 / 51 |
