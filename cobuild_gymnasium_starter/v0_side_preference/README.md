# Cobuild RL — minimal Gymnasium prototype

> **Legacy (v0).** This is the first prototype: pace + left/right preference only. It is kept so it can still be run
> (`cd v0_side_preference && python evaluate.py`, including the 3D robosuite viewer). The current model, with physical
> profiles and a real drywall procedure, is one folder up. See `../docs/MODEL.md` §10 for what was changed and why.


**Goal:** Demonstrate a *trainable* worker-profile-informed assistance planner, not a physical robot or a validated construction model.

## Quick start (Mac / VS Code)

Recommended: Python 3.11 or 3.12 in a new environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python evaluate.py
python visualize.py --profile fast_left --strategy ppo
```

The trained agent is stored as `cobuild_ppo.zip`.

## Project structure

- `cobuild_env.py` — the custom Gymnasium environment (observations, actions, reward, episode logic).
- `profiles.py` — the named synthetic worker profiles (pace + preferred handover side).
- `baselines.py` — the two rule-based strategies (reactive, fixed proactive) plus a transparent profile-aware rule used as a reference point.
- `train.py` — trains the PPO agent and saves `cobuild_ppo.zip`.
- `evaluate.py` — runs all four strategies across all profiles and prints the comparison table.
- `visualize.py` — 2D Pygame view of one episode, for watching a trained (or rule-based) policy run.
- `scene_3d.py` — robosuite/MuJoCo scene: Panda arm, panel stack, worker mannequin, wall with six panel slots.
- `visualize_3d.py` — acts out an episode in the 3D scene (interactive window or `.mp4`).

## What the agent learns

A single PPO policy trains over four synthetic worker profiles, with different work paces (3 or 5 ticks per component) and left/right material handover preferences.

- **Observation:** worker's remaining work time, material status, in-transit status, preferred side, pace, remaining components.
- **Actions:** wait, deliver to left, deliver to right.
- **Reward:** negative elapsed time, extra penalty for worker idle time and wrong-side deliveries, completion bonus.
- **Episode:** supply six components.
- **Evaluation:** reactive / fixed proactive / profile-informed rule / trained PPO.

The worker begins the first operation without requiring an additional supply. At the end of each work cycle, the next component must already have arrived to avoid waiting. Deliveries can be staged in advance; a wrong-side delivery takes three ticks instead of one.

**Important limitations:** synthetic timings, no site observations, no 3D kinematics, no physical safety proof. Work profiles are directly observed by the agent. This demonstrates the Gymnasium RL workflow but **does not prove** personalized RL improves over sensible baselines. Report actual results after running. The profile-rule policy is deliberately strong and may match or outperform PPO.

## Watching an episode

```bash
python visualize.py --profile fast_left --strategy ppo
python visualize.py --profile random --strategy profile_rule --fps 4
```

- `--profile`: `fast_left`, `fast_right`, `slow_left`, `slow_right`, or `random` (a new profile drawn each reset, same as training).
- `--strategy`: `ppo` (needs `cobuild_ppo.zip`, run `train.py` first), `reactive`, `fixed_proactive`, or `profile_rule`.
- `--fps`: simulation ticks per second (default 3 — slow enough to follow).

While the window is open: `1`/`2`/`3`/`4` switch profile and restart, `r` restarts the current profile, `space` pauses, `esc` or closing the window quits.

The view is schematic (circles, squares, bars) rather than a drawing of a real worker, robot, or site — it exists to make the scheduling decisions visible, not to look realistic. If you see `objc[...]: Class SDL... is implemented in both ...` warnings on macOS when running it, that's pygame and OpenCV (loaded by Stable-Baselines3) both bundling SDL2; it's a harmless duplicate-symbol warning, not a crash.

## Watching an episode in 3D (robosuite)

```bash
mjpython visualize_3d.py --profile fast_left --strategy ppo              # interactive window
mjpython visualize_3d.py --profile fast_left --strategy fixed_proactive  # baseline B, for contrast
python visualize_3d.py --profile slow_right --strategy reactive --video reactive.mp4   # save a video
```

On macOS the interactive MuJoCo window must be started with `mjpython` (installed with `mujoco`), not `python`. Video mode works with plain `python`. `--profile` and `--strategy` take the same values as `visualize.py`; `--speed` sets playback speed (default 1.5× real time). In the window you can rotate (left drag), pan (right drag) and zoom (scroll).

What you see:
- **Robot (Panda)** picks a drywall panel (beige) from the stack on the table and brings it to a handover point next to the worker's hip. The **green sphere** marks the worker's preferred side, the grey one the other side.
- **Worker** faces the wall and prepares the next slot (the slot slowly turns orange). When their preparation is done they need a panel: if it is ready on their preferred side they take it and install it (the slot turns solid beige). If not, they turn **red** and hold their hand out on the preferred side: that is worker waiting time.
- **Wrong-side delivery** (e.g. baseline B always delivers on the RIGHT): the robot first goes to the wrong side and then has to reposition, which costs the extra ticks modelled in `CobuildEnv`.
- The text panel shows the profile, the strategy, the tick counter, and the same metrics `evaluate.py` reports.

How this relates to the RL model: the decision of *when* and *on which side* to deliver comes from exactly the same `CobuildEnv` + strategy (PPO or a baseline) used in training and evaluation. The 3D arm motion is scripted (operational-space control to a few waypoints) and only acts that decision out; it is not learned, and the panel is carried kinematically rather than physically grasped. The metrics come from `CobuildEnv`, and how long the 3D motions take is illustrative only.

## Next steps

1. Add multiple material types and sequence constraints from a real drywall subprocess.
2. Ground step durations and interruptions using measured data or literature.
3. Keep all profiles in training and evaluate on previously unseen profiles or noisy profiles.
4. Add delivery travel time, occupancy constraints, safety constraints and tool-change penalties.
5. Model comfortable reaching distance as a third profile trait (currently only pace and handover side are modeled).
6. Only after the planning experiment works, link to robosuite or MuJoCo for visual robot motion.
