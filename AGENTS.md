# Repository Instructions

## Project

The active research project is `roboarm/`.
Read `RESEARCH_CONTEXT.md` and `roboarm/README.md` before proposing changes.
Keep earlier CO-BUILD research separate from the active implementation.

## Working approach

Inspect the actual code before claiming a feature exists.
Distinguish implemented behaviour, planned work and unverified claims.
Explain changes in plain language and work in small, reviewable steps.

Preserve working behaviour. Avoid restructuring, new dependencies,
research features or RL training unless requested.
Ask before structural changes or deleting or moving files.

## Git safety

Before changing files, inspect the current branch and working-tree status.
Do not overwrite, discard or revert teammates' or user changes.
Work on feature branches.

Do not modify main, merge, push, delete branches, reset or rewrite history
without explicit authorization.
Keep archive branches unchanged unless explicitly instructed otherwise.
Archive branches do not preserve ignored or untracked local files.

## Code layout

Keep implementation, tests, dependencies and technical documentation
inside `roboarm/`.
Preserve its current local-import arrangement unless a packaging change
is explicitly approved.
Use `roboarm/requirements.txt`; OMPL is optional.

## Validation

Run relevant existing checks for code changes.
From `roboarm/`, the test command is:

    python -m pytest -q tests

Report failures and skipped optional-dependency tests accurately.
Do not claim application behaviour was verified from inspection alone.
Avoid unnecessary benchmarks or training runs.
Planning and benchmark execution can generate results files;
preserve existing research outputs.

## Reporting

State what changed, why, what was checked and any remaining uncertainty.
For research results, record configuration, seeds, planner and units.
Do not present documentation examples as reproduced experimental evidence.
