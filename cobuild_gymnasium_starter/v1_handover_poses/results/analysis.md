# Cobuild — trait contribution and sensitivity

Rule-based planner, 69 workers (personas + unseen) x 10 paired runs.

## 1. Which part of the profile produces the savings?

| personalised                                  |   time_saved_pct |   strain_saved_pct |
|:----------------------------------------------|-----------------:|-------------------:|
| nothing (= robot B)                           |              0.0 |                0.0 |
| handedness only                               |              5.0 |               16.8 |
| height + reach only                           |              3.6 |               21.7 |
| lifting capacity only                         |              2.1 |                6.6 |
| pace only                                     |              0.6 |                2.0 |
| all physical traits (height, reach, capacity) |              5.3 |               27.5 |
| preferences only (handedness, pace)           |              5.6 |               18.8 |
| full profile (= Cobuild rule)                 |             10.7 |               45.5 |

The rows do not add up to the full profile: traits interact (e.g. height changes both the best handover height and whether holding the board pays off).

## 2. Sensitivity to modelling assumptions

| scenario                                 |   B_minutes |   C_minutes |   time_saved_pct |   strain_saved_pct |   runs_C_better_pct |   runs_C_worse_pct |
|:-----------------------------------------|------------:|------------:|-----------------:|-------------------:|--------------------:|-------------------:|
| baseline assumptions                     |        25.0 |        22.4 |             10.7 |               45.5 |                84.8 |                0.0 |
| no fatigue                               |        23.3 |        21.5 |              7.8 |               45.5 |                85.1 |                0.0 |
| double fatigue                           |        26.7 |        23.2 |             13.2 |               45.5 |                84.5 |                0.0 |
| low variability (sigma 0.05)             |        24.5 |        21.9 |             10.7 |               45.5 |                83.9 |                0.0 |
| high variability (sigma 0.30)            |        26.4 |        23.6 |             10.5 |               45.5 |                85.5 |                0.6 |
| ergonomic penalties halved               |        24.4 |        22.2 |              8.9 |               34.6 |                82.0 |                0.0 |
| wrong side costs no time                 |        24.0 |        22.4 |              7.0 |               45.5 |                81.6 |                0.0 |
| strain weight halved (R_STRAIN -0.25)    |        25.0 |        22.5 |             10.2 |               43.2 |                81.9 |                0.0 |
| strain weight doubled (R_STRAIN -1.0)    |        24.6 |        22.3 |              9.5 |               39.3 |                82.3 |                1.6 |
| faster robot (fetch board 2, screws 1)   |        24.0 |        21.8 |              9.2 |               43.2 |                80.9 |                0.9 |
| B designed for short reach (always near) |        24.4 |        22.4 |              8.5 |               36.4 |                88.4 |                0.1 |
