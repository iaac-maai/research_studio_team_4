# Cobuild drywall evaluation — results

Synthetic simulation, 4 boards per wall, 1 tick = 15 s. These numbers follow from the assumptions in config.py; they are not field measurements.

## Named personas (incl. the generic average worker)

9 workers, 450 paired runs per strategy.

| Strategy | Time (min) | Worker waiting (min) | Interruptions | Strain (pts) | Unnecessary deliveries | Robot exposure (min) | Supported lifts | Completed |
|---|---|---|---|---|---|---|---|---|
| A · Reactive | 30.5 | 5.00 | 8.00 | 38.0 | 0.00 | 0.00 | 0.00 | 100 % |
| B · Proactive (average worker) | 25.7 | 0.18 | 0.66 | 38.0 | 0.00 | 7.12 | 0.00 | 100 % |
| C · Cobuild rule (profile) | 22.0 | 0.79 | 1.50 | 17.3 | 0.00 | 3.16 | 0.89 | 100 % |
| C · Cobuild PPO (profile) | 23.2 | 0.20 | 0.69 | 25.7 | 0.00 | 3.65 | 0.00 | 100 % |
| PPO without profile (ablation) | 24.7 | 0.19 | 0.78 | 32.7 | 0.00 | 4.41 | 0.00 | 100 % |

Paired savings (positive = the second strategy needs less):

| Comparison | Time saved (min) | Time saved (%) | Strain saved (%) | Runs better / worse |
|---|---|---|---|---|
| A · Reactive → B · Proactive (average worker) | 4.82 ± 0.03 | 15.8 % | 0.0 % | 100 % / 0 % |
| A · Reactive → C · Cobuild rule (profile) | 8.44 ± 0.31 | 27.7 % | 54.5 % | 100 % / 0 % |
| A · Reactive → C · Cobuild PPO (profile) | 7.23 ± 0.22 | 23.7 % | 32.4 % | 100 % / 0 % |
| B · Proactive (average worker) → C · Cobuild rule (profile) | 3.63 ± 0.30 | 14.1 % | 54.5 % | 89 % / 0 % |
| B · Proactive (average worker) → C · Cobuild PPO (profile) | 2.41 ± 0.21 | 9.4 % | 32.4 % | 72 % / 18 % |
| B · Proactive (average worker) → PPO without profile (ablation) | 0.92 ± 0.19 | 3.6 % | 13.9 % | 63 % / 35 % |
| PPO without profile (ablation) → C · Cobuild PPO (profile) | 1.49 ± 0.06 | 6.0 % | 21.5 % | 100 % / 0 % |

## Unseen random workers (never used in training)

100 workers, 500 paired runs per strategy.

| Strategy | Time (min) | Worker waiting (min) | Interruptions | Strain (pts) | Unnecessary deliveries | Robot exposure (min) | Supported lifts | Completed |
|---|---|---|---|---|---|---|---|---|
| A · Reactive | 29.3 | 5.00 | 8.00 | 31.5 | 0.00 | 0.00 | 0.00 | 100 % |
| B · Proactive (average worker) | 24.4 | 0.09 | 0.36 | 31.5 | 0.00 | 6.14 | 0.00 | 100 % |
| C · Cobuild rule (profile) | 21.9 | 0.38 | 0.86 | 17.6 | 0.00 | 3.50 | 0.52 | 100 % |
| C · Cobuild PPO (profile) | 22.6 | 0.09 | 0.35 | 22.5 | 0.00 | 3.46 | 0.00 | 100 % |
| PPO without profile (ablation) | 24.0 | 0.11 | 0.45 | 28.7 | 0.00 | 3.82 | 0.00 | 100 % |

Paired savings (positive = the second strategy needs less):

| Comparison | Time saved (min) | Time saved (%) | Strain saved (%) | Runs better / worse |
|---|---|---|---|---|
| A · Reactive → B · Proactive (average worker) | 4.91 ± 0.02 | 16.7 % | 0.0 % | 100 % / 0 % |
| A · Reactive → C · Cobuild rule (profile) | 7.48 ± 0.18 | 25.5 % | 44.2 % | 100 % / 0 % |
| A · Reactive → C · Cobuild PPO (profile) | 6.73 ± 0.16 | 22.9 % | 28.5 % | 100 % / 0 % |
| B · Proactive (average worker) → C · Cobuild rule (profile) | 2.58 ± 0.18 | 10.5 % | 44.2 % | 85 % / 0 % |
| B · Proactive (average worker) → C · Cobuild PPO (profile) | 1.82 ± 0.16 | 7.5 % | 28.5 % | 65 % / 30 % |
| B · Proactive (average worker) → PPO without profile (ablation) | 0.47 ± 0.14 | 1.9 % | 9.1 % | 64 % / 35 % |
| PPO without profile (ablation) → C · Cobuild PPO (profile) | 1.35 ± 0.04 | 5.6 % | 21.4 % | 99 % / 0 % |

## Per persona: mean time (min) and strain

| Persona | A · Reactive | B · Proactive (average worker) | C · Cobuild rule (profile) | C · Cobuild PPO (profile) | PPO without profile (ablation) |
|---|---|---|---|---|---|
| generic_average | 26.4 min / 19 pts | 21.5 min / 19 pts | 21.5 min / 19 pts | 21.9 min / 21 pts | 22.7 min / 26 pts |
| expert_tall_right | 20.1 min / 20 pts | 16.2 min / 20 pts | 15.3 min / 19 pts | 16.1 min / 24 pts | 17.2 min / 26 pts |
| expert_average_left | 24.9 min / 28 pts | 20.1 min / 28 pts | 17.7 min / 18 pts | 17.8 min / 18 pts | 19.1 min / 22 pts |
| novice_average_right | 33.2 min / 19 pts | 28.2 min / 19 pts | 27.1 min / 13 pts | 28.3 min / 19 pts | 29.7 min / 26 pts |
| short_left | 36.1 min / 66 pts | 31.1 min / 66 pts | 23.6 min / 15 pts | 25.8 min / 37 pts | 27.3 min / 44 pts |
| short_strong_right | 27.0 min / 40 pts | 22.1 min / 40 pts | 20.0 min / 23 pts | 20.0 min / 24 pts | 21.8 min / 36 pts |
| tall_novice_left | 31.6 min / 37 pts | 26.7 min / 37 pts | 23.6 min / 15 pts | 24.1 min / 27 pts | 25.3 min / 31 pts |
| low_capacity_right | 33.3 min / 45 pts | 28.4 min / 45 pts | 23.0 min / 16 pts | 25.0 min / 26 pts | 27.7 min / 40 pts |
| low_capacity_short_left | 41.5 min / 67 pts | 36.6 min / 67 pts | 26.4 min / 18 pts | 30.2 min / 36 pts | 31.9 min / 44 pts |
