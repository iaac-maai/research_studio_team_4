# Learning Efficient Robot Motion Under Constraints

An IAAC Smart Construction Research Studio project investigating:

Can reinforcement learning find efficient, collision-free trajectories
for a robotic arm, and how does its performance compare with classical
motion-planning algorithms?

## Current implementation

The active project is `roboarm/`: an interactive, kinematic 2D robotic-arm
simulation with static obstacles, joint limits, collision checking,
pick-and-place sequences, RRT-Connect planning and path shortcutting.
An optional OMPL adapter provides alternative classical planners.

A Gymnasium environment and reinforcement-learning comparison are planned;
they are not yet implemented in this project.

## Run

From the repository root:

```bash
python -m pip install -r roboarm/requirements.txt
cd roboarm
python main.py
```

Press P to plan and Space to start or pause.

## Validate

From `roboarm/`:

```bash
python -m pytest -q tests
```

For benchmark runs:

```bash
python benchmark.py --runs 20
```

Benchmark runs write CSV results under `roboarm/results/`.

## Research and technical documentation

- [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md): question, scope, methodology and next steps.
- [roboarm/README.md](roboarm/README.md): implementation, controls and technical explanations.
- [AGENTS.md](AGENTS.md): repository instructions for coding agents.

## Limitations

The simulation is planar and kinematic. It does not model dynamics,
motor torque, realistic contact or physical-robot execution.
Changing link lengths has a reported issue requiring reproduction
and diagnosis.

## Previous research

Earlier CO-BUILD work is preserved on GitHub:

- `archive/cobuild-original`
- `archive/cobuild-integrated-app`

These archives preserve research history; the active project is `roboarm/`.
