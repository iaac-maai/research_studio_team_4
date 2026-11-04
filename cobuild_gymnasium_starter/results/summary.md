# Cobuild drywall installation: evaluation results

Synthetic simulation. A wall of 4 boards (2 whole + 2 trimmed), 1 tick = 6 s. Numbers follow from the assumptions in config.py (see docs/RESEARCH.md for what is calibrated to published data and what is assumed); they are not field measurements.

## Named personas (incl. the generic average worker)

9 workers, 450 paired runs per strategy.

| Strategy | Time (min) | Worker waiting (min) | Interruptions | Strain (pts) | Heavy handling (min) | Combined cost (min) | Boards held | Unnecessary cmds | Safety stops | Contacts |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 · No robot | 21.2 | 0.00 | 0.0 | 128 | 7.8 | 27.6 | 0.0 | 0.00 | 0.00 | 0.00 |
| A · Reactive | 20.3 | 5.80 | 15.0 | 36 | 0.0 | 22.1 | 4.0 | 0.00 | 2.00 | 0.00 |
| B · Proactive (average worker) | 15.6 | 1.43 | 2.2 | 55 | 1.6 | 18.4 | 2.0 | 0.00 | 2.06 | 0.02 |
| C · Cobuild rule (profile) | 15.4 | 1.43 | 2.2 | 44 | 1.3 | 17.6 | 2.2 | 0.00 | 2.83 | 0.14 |
| C · Cobuild PPO (profile) | 15.5 | 1.35 | 2.4 | 44 | 1.3 | 17.7 | 2.2 | 0.00 | 2.01 | 0.00 |
| PPO without profile (ablation) | 16.2 | 1.47 | 2.5 | 39 | 0.7 | 18.1 | 3.0 | 0.00 | 2.00 | 0.00 |

Where does the time go? Mean minutes per wall by activity:

| Strategy | prep | measure | cut | haul | place | screw | reload | waiting |
|---|---|---|---|---|---|---|---|---|
| 0 · No robot | 3.9 | 1.5 | 2.6 | 4.5 | 1.4 | 3.9 | 3.4 | 0.0 |
| A · Reactive | 3.4 | 1.5 | 2.5 | 0.0 | 0.8 | 3.7 | 2.5 | 5.8 |
| B · Proactive (average worker) | 3.4 | 1.5 | 2.2 | 0.0 | 1.1 | 3.8 | 2.2 | 1.4 |
| C · Cobuild rule (profile) | 3.4 | 1.5 | 2.2 | 0.0 | 0.9 | 3.7 | 2.2 | 1.4 |
| C · Cobuild PPO (profile) | 3.4 | 1.5 | 2.2 | 0.0 | 1.0 | 3.7 | 2.5 | 1.3 |
| PPO without profile (ablation) | 3.5 | 1.5 | 2.2 | 0.0 | 0.9 | 3.7 | 3.0 | 1.5 |

Paired savings (positive = the second strategy needs less):

| Comparison | Time saved | Strain saved | Heavy handling saved | Combined cost saved | Runs better / worse (cost) |
|---|---|---|---|---|---|
| 0 · No robot → A · Reactive | 0.88 ± 0.17 min (4.2 %) | 72.0 % | 100.0 % | 5.50 ± 0.35 min (19.9 %) | 100 % / 0 % |
| 0 · No robot → B · Proactive (average worker) | 5.57 ± 0.16 min (26.3 %) | 57.0 % | 79.3 % | 9.23 ± 0.30 min (33.5 %) | 100 % / 0 % |
| 0 · No robot → C · Cobuild rule (profile) | 5.80 ± 0.15 min (27.4 %) | 65.9 % | 83.3 % | 10.03 ± 0.34 min (36.4 %) | 100 % / 0 % |
| 0 · No robot → C · Cobuild PPO (profile) | 5.68 ± 0.14 min (26.8 %) | 66.0 % | 83.9 % | 9.92 ± 0.34 min (35.9 %) | 100 % / 0 % |
| B · Proactive (average worker) → C · Cobuild rule (profile) | 0.23 ± 0.06 min (1.5 %) | 20.8 % | 19.4 % | 0.80 ± 0.06 min (4.4 %) | 78 % / 2 % |
| B · Proactive (average worker) → C · Cobuild PPO (profile) | 0.11 ± 0.04 min (0.7 %) | 21.1 % | 22.2 % | 0.69 ± 0.06 min (3.7 %) | 81 % / 19 % |
| B · Proactive (average worker) → PPO without profile (ablation) | -0.60 ± 0.04 min (-3.8 %) | 29.3 % | 57.9 % | 0.21 ± 0.08 min (1.1 %) | 46 % / 54 % |
| PPO without profile (ablation) → C · Cobuild PPO (profile) | 0.71 ± 0.06 min (4.4 %) | -11.6 % | -84.6 % | 0.48 ± 0.03 min (2.6 %) | 97 % / 3 % |

## Unseen random workers (never used in training)

100 workers, 500 paired runs per strategy.

| Strategy | Time (min) | Worker waiting (min) | Interruptions | Strain (pts) | Heavy handling (min) | Combined cost (min) | Boards held | Unnecessary cmds | Safety stops | Contacts |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 · No robot | 21.1 | 0.00 | 0.0 | 126 | 7.8 | 27.4 | 0.0 | 0.00 | 0.00 | 0.00 |
| A · Reactive | 20.2 | 5.80 | 15.0 | 34 | 0.0 | 21.9 | 4.0 | 0.00 | 2.00 | 0.00 |
| B · Proactive (average worker) | 15.5 | 1.43 | 2.2 | 52 | 1.5 | 18.1 | 2.0 | 0.00 | 2.00 | 0.00 |
| C · Cobuild rule (profile) | 15.3 | 1.44 | 2.1 | 43 | 1.3 | 17.5 | 2.2 | 0.00 | 2.41 | 0.04 |
| C · Cobuild PPO (profile) | 15.4 | 1.33 | 2.5 | 42 | 1.2 | 17.5 | 2.2 | 0.00 | 2.00 | 0.00 |
| PPO without profile (ablation) | 16.1 | 1.47 | 2.5 | 38 | 0.7 | 18.0 | 3.0 | 0.00 | 2.00 | 0.00 |

Where does the time go? Mean minutes per wall by activity:

| Strategy | prep | measure | cut | haul | place | screw | reload | waiting |
|---|---|---|---|---|---|---|---|---|
| 0 · No robot | 4.0 | 1.5 | 2.6 | 4.6 | 1.3 | 3.8 | 3.4 | 0.0 |
| A · Reactive | 3.5 | 1.5 | 2.5 | 0.0 | 0.8 | 3.6 | 2.4 | 5.8 |
| B · Proactive (average worker) | 3.5 | 1.5 | 2.2 | 0.0 | 1.1 | 3.6 | 2.2 | 1.4 |
| C · Cobuild rule (profile) | 3.5 | 1.4 | 2.2 | 0.0 | 1.0 | 3.6 | 2.2 | 1.4 |
| C · Cobuild PPO (profile) | 3.5 | 1.4 | 2.1 | 0.0 | 1.0 | 3.6 | 2.5 | 1.3 |
| PPO without profile (ablation) | 3.5 | 1.5 | 2.1 | 0.0 | 0.9 | 3.6 | 3.0 | 1.5 |

Paired savings (positive = the second strategy needs less):

| Comparison | Time saved | Strain saved | Heavy handling saved | Combined cost saved | Runs better / worse (cost) |
|---|---|---|---|---|---|
| 0 · No robot → A · Reactive | 0.91 ± 0.17 min (4.3 %) | 73.3 % | 100.0 % | 5.53 ± 0.36 min (20.2 %) | 100 % / 0 % |
| 0 · No robot → B · Proactive (average worker) | 5.65 ± 0.16 min (26.8 %) | 58.8 % | 80.9 % | 9.36 ± 0.31 min (34.1 %) | 100 % / 0 % |
| 0 · No robot → C · Cobuild rule (profile) | 5.80 ± 0.14 min (27.5 %) | 65.8 % | 83.2 % | 9.95 ± 0.33 min (36.3 %) | 100 % / 0 % |
| 0 · No robot → C · Cobuild PPO (profile) | 5.66 ± 0.14 min (26.8 %) | 66.6 % | 84.1 % | 9.86 ± 0.33 min (36.0 %) | 100 % / 0 % |
| B · Proactive (average worker) → C · Cobuild rule (profile) | 0.15 ± 0.07 min (1.0 %) | 17.0 % | 12.1 % | 0.59 ± 0.04 min (3.3 %) | 84 % / 6 % |
| B · Proactive (average worker) → C · Cobuild PPO (profile) | 0.01 ± 0.04 min (0.1 %) | 19.0 % | 16.9 % | 0.50 ± 0.04 min (2.8 %) | 88 % / 12 % |
| B · Proactive (average worker) → PPO without profile (ablation) | -0.65 ± 0.02 min (-4.2 %) | 27.7 % | 51.9 % | 0.06 ± 0.05 min (0.4 %) | 49 % / 51 % |
| PPO without profile (ablation) → C · Cobuild PPO (profile) | 0.67 ± 0.04 min (4.1 %) | -12.0 % | -73.0 % | 0.44 ± 0.02 min (2.4 %) | 97 % / 3 % |

## Per persona: mean time (min) / strain (pts)

| Persona | 0 · No robot | A · Reactive | B · Proactive (average worker) | C · Cobuild rule (profile) | C · Cobuild PPO (profile) | PPO without profile (ablation) |
|---|---|---|---|---|---|---|
| generic_average | 20.0 / 109 | 19.9 / 32 | 15.0 / 47 | 15.0 / 47 | 15.1 / 45 | 15.9 / 37 |
| expert_tall_right | 14.9 / 63 | 16.3 / 21 | 11.7 / 29 | 10.7 / 37 | 10.8 / 37 | 12.8 / 23 |
| expert_average_left | 16.9 / 86 | 18.1 / 29 | 13.3 / 41 | 12.9 / 39 | 13.0 / 37 | 13.9 / 31 |
| novice_average_right | 26.4 / 142 | 24.2 / 42 | 19.2 / 62 | 19.3 / 64 | 19.6 / 60 | 20.3 / 48 |
| short_left | 21.7 / 154 | 20.8 / 45 | 16.2 / 71 | 16.6 / 38 | 16.2 / 45 | 16.4 / 46 |
| short_strong_right | 18.2 / 94 | 19.0 / 37 | 14.2 / 50 | 14.1 / 42 | 14.0 / 45 | 14.8 / 37 |
| tall_novice_left | 23.5 / 103 | 22.4 / 33 | 17.4 / 46 | 16.0 / 56 | 17.3 / 41 | 18.3 / 36 |
| low_capacity_right | 23.1 / 196 | 19.6 / 35 | 15.3 / 68 | 15.7 / 31 | 15.3 / 41 | 15.5 / 41 |
| low_capacity_short_left | 25.9 / 208 | 22.3 / 48 | 18.0 / 84 | 18.2 / 41 | 18.2 / 41 | 18.0 / 52 |
