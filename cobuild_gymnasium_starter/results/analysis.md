# Cobuild: trait contribution and sensitivity

Rule-based planner, 69 workers (personas + unseen) x 10 paired runs. Combined cost = minutes + 0.05 x strain points.

## 1. Which part of the profile produces the gain?

| personalised                                  |   time_saved_pct |   strain_saved_pct |   cost_saved_pct |
|:----------------------------------------------|-----------------:|-------------------:|-----------------:|
| nothing (= robot B)                           |              0.0 |                0.0 |              0.0 |
| handedness only                               |              1.0 |                2.2 |              1.2 |
| height + reach only                           |             -0.1 |                7.9 |              1.1 |
| lifting capacity only                         |             -1.0 |               15.1 |              1.4 |
| pace only                                     |             -0.0 |               -0.1 |             -0.0 |
| all physical traits (height, reach, capacity) |              0.2 |               14.1 |              2.3 |
| preferences only (handedness, pace)           |              1.0 |                2.1 |              1.2 |
| full profile (= Cobuild rule)                 |              1.4 |               16.3 |              3.6 |

The rows do not add up to the full profile: traits interact (for example, height changes both the best handover height and whether holding the board pays off).

## 2. Sensitivity to modelling assumptions

| scenario                                            |   robot_time_saved_pct |   robot_strain_saved_pct |   robot_cost_saved_pct |   profile_time_saved_pct |   profile_strain_saved_pct |   profile_cost_saved_pct |   profile_better_pct |   profile_worse_pct |
|:----------------------------------------------------|-----------------------:|-------------------------:|-----------------------:|-------------------------:|---------------------------:|-------------------------:|---------------------:|--------------------:|
| baseline assumptions                                |                   27.5 |                     58.2 |                   34.6 |                      1.4 |                       16.3 |                      3.6 |                 84.1 |                 4.6 |
| no fatigue                                          |                   25.6 |                     58.2 |                   33.3 |                      1.2 |                       16.0 |                      3.4 |                 83.3 |                 5.8 |
| double fatigue                                      |                   29.3 |                     58.2 |                   35.8 |                      1.5 |                       16.4 |                      3.7 |                 84.3 |                 4.2 |
| low variability (sigma 0.05)                        |                   27.3 |                     58.2 |                   34.4 |                      1.3 |                       16.4 |                      3.5 |                 83.5 |                 4.9 |
| high variability (sigma 0.30)                       |                   27.7 |                     57.8 |                   34.6 |                      0.9 |                       12.7 |                      2.6 |                 77.5 |                15.4 |
| light board (26 kg, 1/2")                           |                   29.3 |                     35.7 |                   30.5 |                     -0.6 |                       24.0 |                      3.8 |                 82.6 |                 4.9 |
| heavy board (42 kg, Yuan density)                   |                   27.2 |                     80.8 |                   41.8 |                      3.8 |                      -20.4 |                      1.7 |                 77.1 |                 1.3 |
| lifting never slows the worker                      |                   25.4 |                     55.6 |                   32.1 |                      1.4 |                       11.5 |                      2.8 |                 79.9 |                 7.4 |
| lifting slows 2x more                               |                   29.5 |                     60.3 |                   36.9 |                      1.4 |                       20.9 |                      4.4 |                 85.9 |                 4.1 |
| no overhead penalty                                 |                   29.5 |                     52.0 |                   34.3 |                      1.5 |                       22.2 |                      4.7 |                 84.2 |                 2.0 |
| robot alignment takes 1 tick                        |                   26.4 |                     66.8 |                   35.7 |                      2.0 |                       10.0 |                      3.0 |                 90.4 |                 1.2 |
| robot alignment takes 4 ticks                       |                   30.8 |                     40.3 |                   33.0 |                     -0.5 |                       27.0 |                      5.2 |                 84.1 |                 0.7 |
| slower robot (+2 s per carry, tools 3 ticks)        |                   24.9 |                     55.2 |                   31.9 |                      1.6 |                       10.8 |                      3.0 |                 83.6 |                 5.4 |
| steadying at the cut station has no effect          |                   25.7 |                     56.4 |                   32.8 |                      1.3 |                       15.6 |                      3.5 |                 84.6 |                 4.1 |
| tool-handover penalties off                         |                   28.7 |                     60.2 |                   36.0 |                     -0.3 |                       12.1 |                      1.5 |                 65.4 |                16.1 |
| planner values strain half as much (R_STRAIN -0.25) |                   30.8 |                     40.3 |                   33.0 |                      1.6 |                       21.4 |                      5.7 |                 82.6 |                 0.7 |
| planner values strain twice as much (R_STRAIN -1.0) |                   22.9 |                     74.7 |                   34.8 |                      0.8 |                        1.2 |                      0.8 |                 70.4 |                 1.0 |
| B designed for a strong worker (45 kg)              |                   30.8 |                     40.3 |                   33.0 |                     -3.3 |                       41.3 |                      5.9 |                 85.8 |                12.8 |
| B designed for a weak worker (24 kg)                |                   22.9 |                     74.7 |                   34.8 |                      7.3 |                      -38.3 |                      3.2 |                 87.5 |                 0.9 |
