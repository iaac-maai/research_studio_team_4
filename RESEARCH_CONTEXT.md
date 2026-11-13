# Research Context

## Setting

IAAC Barcelona, Master's in AI Applied to Architecture.
Smart Construction Research Studio — Operable Futures.
Team: Emma, Amelia and Karthik.

Working title: Learning Efficient Robot Motion Under Constraints.

The project evolved from CO-BUILD, which explored personalised
human–robot construction assistance. Following midterm feedback,
the research scope narrowed to motion planning under constraints.

## Research question

Can reinforcement learning find efficient, collision-free trajectories
for a robotic arm, and how does its performance compare with classical
motion-planning algorithms?

## Scope

Start with static, fully known 2D environments. Study how obstacles,
link lengths and joint limits affect reachability and feasible paths.

Physical-robot experiments are optional future work.
Human tracking, worker personalisation and construction-task modelling
are outside the current scope.

## Methodology

1. Study point navigation in 2D to understand classical planning.
2. Study a fixed planar arm and planning in configuration space.
3. Build an equivalent Gymnasium environment.
4. Select and evaluate an RL algorithm; PPO and SAC are candidates,
   not confirmed choices.
5. Compare RL and classical planning using matched tasks, geometry,
   collision rules, start/goal conditions and evaluation budgets.

Before experiments, define success, collision, timeout and failure.
Use multiple recorded seeds and distinguish training scenarios from
held-out evaluation scenarios.

Measure success rate, collision rate, joint-space path length,
tool-path length, planning or inference time, and generalisation.
Report RL training cost separately from inference time.
Report path-quality metrics alongside failures and success rates.

## Current progress

Implemented in `roboarm/`:
- Planar arm kinematics and joint limits.
- Static obstacle and self-collision checking.
- RRT-Connect and path shortcutting.
- Pick-and-place sequencing.
- Interactive visualisation.
- Metrics, benchmark tooling and automated tests.
- Optional OMPL adapter.

A separate point-navigation implementation has not been established
by this audit. The arm implementation already exists, despite the
conceptual stages above.

Not implemented:
- A Gymnasium environment for this research direction.
- RL training or an RL-versus-classical evaluation.

Existing numerical claims in technical documentation must be reproduced
before being treated as validated research results.

## Known issue

Changing link lengths reportedly causes problems.
The cause has not been diagnosed. Investigate reachability, collisions,
initial poses and plan invalidation before selecting a fix.

## Next steps

1. Complete and validate the repository transition.
2. Reproduce the link-length issue with a recorded configuration.
3. Establish a reproducible classical-planning baseline.
4. Agree the benchmark protocol and minimum RL task.
5. Implement further research stages only after explicit approval.

## History

Previous research is preserved in:
- `archive/cobuild-original` at `262575e`.
- `archive/cobuild-integrated-app` at `d18fd18`.

Neither archive alone contains both development lines.
