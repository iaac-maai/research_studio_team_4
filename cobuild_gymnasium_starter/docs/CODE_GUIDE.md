# Code guide: how to read, modify and understand the code

For anyone who needs to explain or change the Cobuild code (v2, drywall installation).

Related documents:
- How to run things: [../README.md](../README.md)
- How the model works and why the comparison is fair: [MODEL.md](MODEL.md)
- Where the numbers come from: [RESEARCH.md](RESEARCH.md)

Run every command from the `cobuild_gymnasium_starter/` folder.

---

## Step 0 · The idea

A **simulator** of a worker installing 4 drywall boards on a wall, and a **robot** that can help. The installation is repeated many times with different robots and different workers, and we measure time and physical strain.

```
profiles.py ──► cobuild_env.py ◄── config.py        (the world: worker, task, numbers)
                    │   ▲
                    │   └── ergonomics.py            (body → time and strain)
                    ▼
              strategies.py                          (the robot's "brains")
              ┌─────┴──────┐
        train.py          evaluate.py ──► plots.py   (train / measure / plot)
      (creates models/)   analysis.py                (which traits matter, sensitivity)
                          visualize_3d.py + scene3d.py   (watch the simulation in 3D)
```

---

## Step 1 · File map (in the order worth reading them)

| # | File | Lines | What it contains | Touch it when… |
|---|---|---|---|---|
| 1 | [config.py](../config.py) | ~105 | **Every number** in the model (durations, weights, reward), tagged `[DATA]`, `[LIT]` or `[ASSUME]` | you want to change an assumption |
| 2 | [profiles.py](../profiles.py) | ~86 | The `WorkerProfile` class, the 8 personas, the generic worker and the random worker generator | you want another profile or other traits |
| 3 | [ergonomics.py](../ergonomics.py) | ~179 | Turns a body into time and strain for each task; decides whether holding the board pays off | you change how the body affects a task |
| 4 | [cobuild_env.py](../cobuild_env.py) | ~520 | **The Gymnasium environment**: state, actions, observation, reward, metrics | you change the simulation rules |
| 5 | [strategies.py](../strategies.py) | ~125 | The 6 strategies (no robot, A, B, C rule, C PPO, PPO without profile) | you change how the robot decides |
| 6 | [train.py](../train.py) | ~64 | Trains the PPO | you want to retrain |
| 7 | [evaluate.py](../evaluate.py) | ~169 | Repeats the installation many times and saves the results | you want more runs or more metrics |
| 8 | [plots.py](../plots.py) | ~195 | Figures 1–5 | you want another plot |
| 9 | [analysis.py](../analysis.py) | ~202 | Trait contribution + sensitivity (figures 6–7) | you want to test another assumption |
| 10 | [scene3d.py](../scene3d.py), [visualize_3d.py](../visualize_3d.py) | ~299 / ~393 | The 3D scene and the video director | you want to change how it looks |
| 11 | [collisions.py](../collisions.py), [audit_3d.py](../audit_3d.py) | ~190 / ~110 | Shape-vs-shape distances in the 3D scene; the audit report; `yield_robot` (the robot gives way) | the 3D shows things passing through each other |
| 12 | [tests/](../tests/) | ~144 + ~30 | 53 automated tests (`test_env.py`, `test_3d.py`) | always, after changing anything |

**Recommended reading order:** `config.py` → `profiles.py` → the docstrings of `ergonomics.py` → `cobuild_env.py` (`reset`, `step`, `_observation`) → `strategies.py` → `evaluate.py`. That covers 90 % of it. You can ignore the 3D at first: it does not affect any result.

---

## Step 2 · The existing tools we use

Installed with [requirements.txt](../requirements.txt). **We do not modify any library**; we write our own code on top of them.

| Tool | What it is for | How we use it | Where |
|---|---|---|---|
| **Gymnasium** | Industry standard for RL environments: defines `reset()`, `step()`, `action_space`, `observation_space` | We write our own class `CobuildDrywallEnv(gym.Env)` | `cobuild_env.py` |
| **Stable-Baselines3** | Tested implementations of RL algorithms | Only its utilities: `make_vec_env` (8 parallel environments) and `BaseCallback` (progress printing) | `train.py` |
| **sb3-contrib → `MaskablePPO`** | PPO with **action masking**: the agent cannot pick impossible actions | The algorithm we train. The environment exposes `action_masks()` and the model uses it | `train.py`, `strategies.py` |
| **NumPy** | Arrays and random numbers | Duration noise (`lognormal`), the observation, worker sampling | everywhere |
| **pandas** | Tables | `evaluate.py` saves one row per episode in `results/episodes.csv`; the plots read it | `evaluate.py`, `plots.py` |
| **tabulate** | Markdown tables | Writes `results/summary.md` and `analysis.md` | `evaluate.py`, `analysis.py` |
| **matplotlib** | Plots | The 7 figures in `results/` | `plots.py`, `analysis.py` |
| **MuJoCo** | 3D physics / rendering engine | Used only to **draw**. Bodies are "mocap" (we move them ourselves), no physics | `scene3d.py` |
| **OpenCV** | Video | Writes the mp4 (H.264 codec `avc1`) and the `--live` window | `visualize_3d.py` |
| **Pillow** | Text on images | The panel with clock and metrics on every frame | `visualize_3d.py` |
| **pytest** | Automated tests | `python -m pytest -q tests` | `tests/` |

### What is ours, not from any library
- **The task model** (what happens in each tick): `cobuild_env.py` and `ergonomics.py`.
- **The calibration** against Yuan et al. 2016 and Dasgupta et al. 2016: `config.py` and the test in `tests/`.
- **The rule-based strategies** (B and C rule): `strategies.py`.
- **The paired comparison** with common random numbers: `evaluate.py`.

---

## Step 3 · What each piece does, in detail

### 3.1 `config.py`: the numbers
A single place with every assumption. Each line carries a tag:
- `[DATA]`: calibrated to a published measurement.
- `[LIT]`: direction or size supported by the literature.
- `[ASSUME]`: our own assumption. **These are the ones to defend or measure.**

Main blocks:
- **Task:** `TICK_SECONDS = 6`, `N_BOARDS = 4`, `BOARD_KG = 34`.
- **`BASE`**: duration in ticks of each phase for the average worker alone (measure 7, cut 12, reload 4.5…). `DURATION_NOISE_SIGMA = 0.15` is the random variability.
- **Body:** `LIFT_LOAD_KNEE`, `OVERHEAD_FACTOR_PER_M`, `FATIGUE_SCALE`.
- **Robot:** `BRING_TICKS`, `ALIGN_TICKS`, `FETCH_TOOL_TICKS`, `STEADY_CUT_TIME_FACTOR`, `HOLD_RELEASE_AFTER`.
- **Tool handover poses:** 2 sides × 3 heights × 2 distances = 12.
- **Reward:** `R_TICK`, `R_WORKER_WAIT`, `R_STRAIN`, `R_EXPOSURE`, `R_UNNECESSARY`, `R_IGNORED`, `R_COMPLETE`.

> Golden rule: **if you want to change a number, change it here, not in another file.**

### 3.2 `profiles.py`: the workers
- `WorkerProfile` is a dataclass with `height_m`, `reach_m`, `lift_capacity_kg`, `pace`, `handedness`, `fatigue_rate`.
- `GENERIC` is the average worker the non-personalised robot (B) is designed for.
- `PERSONAS` are 8 named cases (for example `short_left`, `low_capacity_right`).
- `sample_population(rng)` creates a random worker (normal height, normal capacity, uniform pace 0.8–1.3, 50/50 handedness). It is used for **training** and for the **unseen workers** in the evaluation.

### 3.3 `ergonomics.py`: from body to time and strain
Pure functions (no state). The central ones:
- `load_ratio(P, b)`: board weight ÷ worker capacity.
- `overhead_deficit(P, b)`: how much reach the worker lacks for the upper row.
- `phase_cost(P, b, name, mode)`: **the most important function**. Returns the extra time and the strain of a phase, given the worker's body and who does it (alone or with the robot).
- `hold_is_worth_it(P, b)`: does it pay for the robot to hold the board for this worker? This is what makes a personalised robot different.
- `best_tool_pose(P)`, `handover_cost(P, pose)`: the best pose for handing a tool and what a bad pose costs.

### 3.4 `cobuild_env.py`: the heart
The class `CobuildDrywallEnv(gym.Env)`.

**Episode flow**
1. `reset(seed)`: picks the profile (fixed or random), draws the noise for all phases **once**, and starts board 0. The noise is drawn before any robot action, which is why every strategy with the same seed faces the same worker (**common random numbers**).
2. `step(action)` on every 6 s tick:
   1. adds `R_TICK`,
   2. `_robot_command(action)`: executes what the robot asked for (or penalises it if it makes no sense),
   3. `_advance_robot()`: advances the robot's state,
   4. `_advance_worker()`: advances the worker's phase,
   5. adds `R_EXPOSURE`, `R_STRAIN × (new strain)` and `R_COMPLETE` if finished.
3. It ends when all 4 boards are fastened (`terminated`) or `MAX_TICKS` is reached (`truncated`).

**The 16 actions:** 0 wait · 1 bring board · 2 steady at the cut · 3 lift and hold · 4–15 hand a tool at one of 12 poses.

**Masks:** `action_masks()` marks inapplicable actions as impossible (for example, holding when no board is ready). Also: there is **only one arm**, so while it holds the board it cannot hand tools.

**Observation:** `_observation()` builds 43 numbers between 0 and 1: task progress, where the board is, robot state, expected ticks until the worker needs something (`time_to`), the profile, and what it implies (whether holding pays off).

**The `observe_profile` switch:** if `False`, the profile the agent sees is always the generic one. This is how the "PPO without profile" control model is built.

**Metrics:** `_info()` returns minutes, waiting, strain, time carrying, minutes per activity, etc. `evaluate.py` saves them.

### 3.5 `strategies.py`: the brains
Each strategy is a `Strategy` object with an `act(env, obs) → action` function.

| Key | How it decides |
|---|---|
| `no_robot` | always waits; the worker does everything alone |
| `A_reactive` | `reactive_action`: waits for the worker to ask for help |
| `B_proactive_generic` | `planner_action(env, GENERIC)` |
| `C_cobuild_rule` | `planner_action(env, env.profile)`: **the same function as B**, with the real profile |
| `C_cobuild_ppo` | loads `models/cobuild_ppo_profile.zip` |
| `ablation_ppo_no_profile` | loads `models/cobuild_ppo_no_profile.zip` and does not see the profile |

`planner_action(env, P)` is the rule-based planner. Priority: hold if it pays off → bring the next board just in time → hand the tool just in time at the best pose → steady. "Just in time" uses `env.time_to(...)` plus a safety margin (`SAFETY_MARGIN_TICKS = 3`).

> That is why B vs C rule is a clean comparison: **only the profile `P` changes**.

### 3.6 `train.py`: training
- Creates 8 parallel environments with `make_vec_env`.
- `MaskablePPO('MlpPolicy', …)`, two layers of 128, `gamma=0.995`, `ent_coef=0.01`.
- Every episode uses a new random worker, so the agent learns a general rule instead of memorising people.
- Saves `models/cobuild_ppo_profile.zip` and `models/cobuild_ppo_no_profile.zip`.

```bash
python train.py --timesteps 3000000          # both models, 3 M steps
python train.py --variant profile --seed 7   # only the profile-aware model, another seed
```
> Note 1: the file's default is **2 M** steps (`--timesteps`). The README results are from 3 M; pass it explicitly to reproduce them.
> Note 2: `--seed` changes the seed, but each model is saved under the same name and **overwrites** the previous one. To try several seeds, copy or rename the models between runs.

### 3.7 `evaluate.py`: measuring
- `run_episode(strategy, profile, seed)` runs one full wall and returns a row of metrics.
- Groups: the 8 personas × 50 runs and 100 unseen workers × 5 runs (900 runs per strategy).
- For each (worker, seed) it runs **all** strategies with the same noise.
- Saves `results/episodes.csv` (one row per episode) and `results/summary.md`, and calls `plots.py`.
- `paired(...)` computes paired savings with a 95 % confidence interval.
- **Combined cost** = minutes + 0.05 × strain points.

### 3.8 `plots.py` and `analysis.py`
- `plots.py` reads `episodes.csv` and produces figures 1–5. Each figure is a function (`fig_overview`, `fig_activity`, `fig_work_saved`, `fig_profile_value`, `fig_paired`). You can regenerate them without retraining: `python plots.py`.
- `analysis.py` does two things: **(a)** `trait_contribution`: the planner knows only one group of traits and we measure how much that helps; **(b)** `sensitivity`: 19 scenarios with different assumptions. It writes `results/analysis.md` and figures 6–7.

### 3.9 `scene3d.py` and `visualize_3d.py`: watching the simulation
- `scene3d.py` builds the scene in MuJoCo: wall with studs, pallet, cutting cart, worker (with the profile's height), robot, boards and tools. It only draws.
- `visualize_3d.py` runs the **same environment and the same strategies** as the evaluation, interpolates positions between ticks and writes the video. `pose_of(env, P)` translates the environment state into 3D positions.
- **Important:** the 3D changes no result. If a video looks odd, the bug is in the drawing, not in the numbers.

### 3.10 `tests/test_env.py`: the safety net
`test_env.py` has 48 tests; `test_3d.py` adds 5 (nothing through the floor or the set, worker/robot/board overlaps stay small). The most useful ones for understanding the design:
- the environment follows the Gymnasium API (`check_env`),
- the action masks are correct,
- same seed ⇒ same worker in every strategy,
- B and C rule give **identical** results with the generic worker,
- the worker-alone time matches Yuan et al.,
- the decision to hold the board changes by persona.

---

## Step 4 · How to run everything, in order

```bash
python -m pytest -q tests             # 1. is everything fine?
python train.py --timesteps 3000000   # 2. train (~10 min per model)
python evaluate.py                    # 3. measure + figures + tables
python analysis.py                    # 4. traits + sensitivity (~3 min)
python visualize_3d.py --profile short_left --compare --top no_robot --bottom C_cobuild_ppo --video videos/demo.mp4
```

Without training: `python evaluate.py --no-ppo` uses only the rule-based strategies.

---

## Step 5 · Recipes: I want to change X

| I want to… | What to touch | Afterwards |
|---|---|---|
| Change the duration of a worker phase | `BASE` in `config.py` | tests (there is a calibration test) → `evaluate.py` |
| Change how fast the robot is | `BRING_TICKS`, `ALIGN_TICKS`, `FETCH_TOOL_TICKS`, `SWAP_TICKS` in `config.py` | **retrain** the PPO → `evaluate.py` |
| Change the board weight | `BOARD_KG` | retrain → evaluate |
| Change how much we value strain vs. time | `R_STRAIN` in `config.py`. In `evaluate.py` the combined cost is derived from `R_STRAIN` automatically; in `analysis.py` `COST_PER_STRAIN_MIN` is fixed at 0.05 on purpose (so the scenarios stay comparable) | retrain → evaluate |
| Add a named profile | `PERSONAS` in `profiles.py` | add it to the tests if relevant; it shows up automatically in `evaluate.py` and `--profile` |
| Add a profile trait (for example, age) | `WorkerProfile` and `sample_population` in `profiles.py`; its effect in `ergonomics.py`; add it to `_observation()` in `cobuild_env.py` (changes `OBS_DIM`, defined in `cobuild_env.py`) | **retraining is mandatory** (the network changes size) |
| Add a robot task (for example, fastening screws) | new action count and state in `cobuild_env.py`, its mask in `action_masks()`, its logic in `planner_action` | retrain, tests, evaluate |
| Add a new strategy | `STRATEGY_KEYS`, `LABELS` and `make_strategy` in `strategies.py`; colour in `plots.py` | evaluate |
| Evaluate with more runs | `python evaluate.py --runs 100 --unseen 200` | — |
| Test another assumption in the sensitivity | the scenario list in `sensitivity()` in `analysis.py` | `python analysis.py` |
| Change only how the plots look | `plots.py` | `python plots.py` (no retraining) |
| Change only how the 3D looks | `scene3d.py` / `visualize_3d.py` | regenerate the video |

### When do I have to retrain?
Retrain if you change **anything the PPO sees or feels**: numbers in `config.py` that affect the simulation, the reward, the observation, the actions or the profile. The rule-based strategies do not need retraining but do need re-evaluating. If you only change plots or the 3D, you do not need to.

> Danger: an old PPO with a new environment **does not fail with a clear error** (either it fails on size, or worse, it runs and gives misleading results). If you change the model, retrain and re-run `evaluate.py`.

---

## Step 6 · Pitfalls and details worth knowing

1. **One tick is 6 seconds**; durations in the code are in ticks. Multiply by `TICK_SECONDS / 60` to get minutes.
2. **The robot does not see the noise or the accumulated strain**; it plans with expected values.
3. **Strain is "relative points"**, not a clinical unit.
4. **A single training seed** (42) today. For more reliable PPO results several are needed; see the discussion in the README.
5. **Models are not versioned**: `models/*.zip` is overwritten when you retrain.
6. **Videos** are written with the H.264 codec (`avc1`); with `mp4v` they did not play in QuickTime or VS Code.
7. What is **not** RL: B and C rule are hand-written rules (on purpose: they are the clean reference).

---

## Step 7 · Mini glossary

| Term | Meaning here |
|---|---|
| **Tick** | 6 seconds of simulated work |
| **Episode** | one full wall (4 boards) |
| **Persona** | a named worker profile |
| **Unseen worker** | a random profile not used among the personas |
| **Action mask** | list of actions allowed right now |
| **Ablation** | removing one piece (the profile) to see how much it mattered |
| **Common random numbers** | same variability for every strategy, so the comparison is paired |
| **Combined cost** | minutes + 0.05 × strain points |
| **B / C rule** | same planner; B assumes the average worker, C uses the real profile |
