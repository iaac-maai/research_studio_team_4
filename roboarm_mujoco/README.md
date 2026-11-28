# roboarm_mujoco: the pick-and-place arm with real physics (MuJoCo)

This is the same 3-joint planar arm and pick-and-place task as [roboarm/](../roboarm/README.md), rebuilt in the **MuJoCo** physics engine with no Pygame. The differences from roboarm:

- **Gravity and motors.** The joints and fingers are driven by position-controlled motors.
- **A real grip.** The fingers hold the box only by friction, so a weak grip lets it slip or fall.
- **MuJoCo's collision detection** answers the planner's "is this pose / motion free?" questions.
- **MuJoCo's built-in viewer** replaces the Pygame window, and the keyboard replaces buttons.

roboarm/ is unchanged. Its kinematics, IK and planner (RRT-Connect + shortcutting) are imported, not copied.

---

## 1. Quick start

Run everything from the **repository root**:

```bash
pip install -r roboarm_mujoco/requirements.txt          # mujoco, numpy, pytest
python -m roboarm_mujoco.main                           # opens the viewer, default layout
python -m roboarm_mujoco.main --layout roboarm_mujoco/environments/<file>.csv --seed 3
python -m pytest -q roboarm_mujoco/tests                # 44 tests (1 skips until environments/layout.py exists), under a minute
```

| Key | What it does |
|---|---|
| `P` | Plan the pick-and-place from the current state, then run it. When the box is on the other table, `P` carries it back. |
| `Enter` | Pause / resume the running motion. |
| `E` | Open / close the gripper (when nothing is running). |
| `Home` | Reset: arm home, gripper open, box back at its layout position. |
| Mouse | MuJoCo viewer's own controls: drag to orbit / pan, scroll to zoom, double-click to select a body, Ctrl + drag to push it (see `F1` in the viewer). |

MuJoCo's viewer uses almost every letter key for its own display toggles (for example `C` contact points, `T` transparency, `W` wireframe). `P` and `E` were picked because their built-in toggles (contact split, equality constraints) change nothing visible here.

Planning takes about 7 to 13 s, during which the window may stutter. The status line at the top says what is happening.

Options: `--layout FILE.csv`, `--seed N` (repeatable planning), `--speed F` (motion speed, 1 = `config.JOINT_SPEED`).

---

## 2. Files

| File | Role |
|---|---|
| [robot_api.py](robot_api.py) | **`RobotSim`**, the agreed interface (section 4). Owns the MuJoCo model and data. |
| [scene.py](scene.py) | Builds the MuJoCo model (MJCF XML) from a layout plus `roboarm/config.py`. Also holds `default_layout()`, layout validation and `load_csv_layout()`. |
| [planning.py](planning.py) | `MujocoChecker`: the `is_valid` / `edge_valid` answers for roboarm's planner, using MuJoCo collision detection on a separate copy of the scene. |
| [pick_place.py](pick_place.py) | `TaskPlanner` (the roboarm task sequence: transit, approach, grasp, lift, carry, lower, release, retreat, home) and `PlanExecutor` (plays a plan through `RobotSim.step()`). |
| [trajectory.py](trajectory.py) | Turns a joint path into one target per control step. |
| [main.py](main.py) | The viewer app and its keys. |
| [benchmark.py](benchmark.py) | Plans and physically executes the task for many seeds without a window; prints a summary and saves a new CSV in `results/` (section 7). |
| [settings.py](settings.py) | Physics-only parameters: masses, motor gains, grip force, friction, contact stiffness, time step, depths. |
| [paths.py](paths.py) | The one place that puts `roboarm/` on `sys.path`. |
| [tests/](tests/) | One file per step: scene, control, grasp, collision, API, planning, app, benchmark. |

Not owned by this folder: `environments/` (layout CSVs and `layout.py`, environments person) and `rl/` (RL person).

---

## 3. Coordinates and layouts

- **2D world:** x right, y up, metres, origin on the floor below the robot (as in roboarm).
- **In MuJoCo** the arm moves in the **x-z plane**: layout `(x, y)` becomes MuJoCo `(x, 0, z = y)`. All hinge axes are −y, so a joint angle means the same as in roboarm (q1 measured from +x, q2 and q3 relative to the previous link).
- **Depth.** Every shape needs a thickness along MuJoCo y: 1.0 m for tables and obstacles, 0.30 m for the box, 0.12 m for the finger pads.
- **Layout CSV** (agreed format, read only through `environments/layout.py`'s `load_layout`):
  - Columns: `name,type,shape,x,y,width,height`, with x, y the centre of the shape.
  - `type` is `obstacle`, `table` or `box`. There must be exactly one `box` row (`Object`), and tables are named `Table A` and `Table B`.
  - `shape` is `box` or `circle`. A circle becomes a cylinder lying along y, with diameter = width.
- **Without a CSV**, the default layout is `roboarm/config.py` converted to rows. The CSV format has no trapezoids, so the trapezoid "Centre stalactite" becomes a rectangle of the same height and average width (same area). roboarm's walls and ceiling are not part of the CSV format and are left out. The floor is always there.
- **The box** starts at its CSV position. Put its centre at `table top + height/2` so it rests on the table.

---

## 4. `RobotSim` (the agreed interface)

```python
from roboarm_mujoco.robot_api import RobotSim
sim = RobotSim()                                   # default layout
state = sim.reset(layout_csv=None, seed=None)      # None keeps the current layout
state = sim.step(np.array([1.6, -0.9, -2.2]))      # absolute joint targets (rad), advances 0.02 s
sim.set_gripper(True)                              # close (squeeze) / False = open
sim.get_state()                                    # dict, see below
sim.in_collision()                                 # bool
```

`get_state()` returns exactly these keys:

| Key | Type | Meaning |
|---|---|---|
| `q` | ndarray (3,) | Joint angles, rad |
| `tcp` | ndarray (2,) | Tool centre point (between the fingers), x, y in m |
| `box_pos` | ndarray (2,) | Box centre, x, y in m |
| `box_angle` | float | Box rotation in the plane, rad, wrapped to [−π, π) |
| `gripper_closed` | bool | Last gripper command |
| `holding` | bool | Gripper closed **and** both fingers touching the box |

`in_collision()` is True only when:

- the arm or gripper touches an obstacle, a table, the floor or the robot's pedestal, or itself; or
- the box touches an obstacle.

The box resting on a table (or on the floor), and the fingers touching the box, are **not** collisions.

Details:

- `step()` clips targets to the joint limits.
- Each joint has a position servo plus a velocity servo. The velocity servo gets the speed implied by consecutive targets (capped at 2 rad/s), so a smoothly moving target is followed without lag.
- Large jumps between consecutive targets make fast, violent motions. That is real physics, worth knowing for RL action design.
- `reset(seed=...)` stores `sim.rng` for the planner. The physics itself is deterministic.

Extra (not part of the contract):

- `RobotSim.from_layout(rows)` builds a sim from rows instead of a CSV.
- `sim.collisions()` lists the colliding geom pairs.
- `sim.set_grip_force(newtons)` changes the grip strength.
- `sim.target`, `sim.time`, `sim.model`, `sim.data` are also available.

---

## 5. The physics model

| Item | Value | Why |
|---|---|---|
| Time step | 2 ms physics; one `step()` = 10 physics steps = 0.02 s | |
| Gravity | −9.81 m/s² | |
| Link masses | 4.0, 3.0, 1.5 kg; palm 0.6 kg; fingers 0.15 kg each; box 1.0 kg | Chosen, not measured. |
| Gravity compensation | On for the arm bodies (`gravcomp`), **off for the box** | A position servo alone settles slightly short of its target under gravity. The carried box is still a real load. |
| Joint servos | kp = 60000, 25000, 4000; kv = 4000, 1500, 160 | About critically damped for the measured joint inertias (84, 18, 1.4 kg·m² stretched out). |
| Gripper | Two sliding fingers, **mechanically coupled** (a joint-equality constraint between the two finger slides), position servos with force capped at **40 N each** | Coupling keeps the box centred, like a real parallel gripper. Nothing attaches the box to the gripper. |
| Friction | μ = 1.0 (sliding), elliptic friction cones, `impratio = 10` | The usual MuJoCo settings for stable grasps. |
| Contact stiffness | `solref = 0.004 1` on every geom (`CONTACT_SOLREF`; MuJoCo's default is 0.02) | With the default, the box sank ~1 mm into each finger, rocked at every motion and "walked" ~4 cm along the fingers during one carry. Stiffer contacts cut this about tenfold (see section 7). |
| Box | Moves in the plane only (slide x, slide z, hinge about y) | It can slide, tip or fall, but not leave the 2D world. |

**Holding by friction.** The box stays up while 2 × μ × grip force > weight + inertial forces. Here that is 2 × 1.0 × 40 N = 80 N against 9.8 N. With `sim.set_grip_force(2.0)` (4 N < 9.8 N), the box slips out (tested in `tests/test_grasp.py`).

---

## 6. Planning

Reused from roboarm:

- `PlanarArm`: IK, straight-line approach motions, reach radii, held-object pose.
- `RRTConnect`, `PlanningProblem`, `shortcut_path`, `path_length`.
- Every planner setting in `config.py`: step size, iterations, margins, approach height, joint speed.

New here:

- **`MujocoChecker`** compiles its own copy of the scene, sets the joint angles, runs MuJoCo kinematics plus collision detection and reads the contact list. The live simulation is never touched.
- **Clearance margins** use MuJoCo's geom `margin`: a robot geom with margin m reports contact whenever it is closer than m. As in roboarm, the margin is 5 cm in free space (`PLANNING_MARGIN`) and 1 cm near the tables (`SAFETY_MARGIN`).
  - Pitfall we hit: the margin must be in the XML **before compiling**. Changing `model.geom_margin` afterwards is ignored by MuJoCo's broad phase (tested in `tests/test_planning.py::test_margin_means_clearance`).
- **Motion checks** are spaced so no robot point moves more than 3 cm between checks (`COLLISION_STEP`), as in roboarm.
- **Planning is kinematic.** The held box is placed rigidly in the gripper for checking. **Execution is physical**: `PlanExecutor` sends one target per 0.02 s step and records any collision that happens during execution.
- **While lifting**, the held box may touch the table it is lifted from. The box is released 2 cm above the place table and drops onto it.

---

## 7. Measured results (this machine)

Configuration:

| | |
|---|---|
| Layout | Default (`roboarm/config.py`, converted as in section 3) |
| Planner | Built-in RRT-Connect, `MAX_ITERATIONS = 15000`, `SHORTCUT_ITERATIONS = 400`, margins 0.05 / 0.01 m |
| Physics | Settings in section 5 |
| Machine | Windows 11 laptop, Python 3.12.4, MuJoCo 4.0.1 |

Each seed ran once (2026-11-26/27):

| Seed | Planned | Planning time (s) | Collision checks | Joint path (rad), smoothed / raw | TCP path (m) | Collisions during execution | Final box (x, y) m |
|---|---|---|---|---|---|---|---|
| 0 | yes | 11.6 | 121,705 | 24.0 / 36.0 | 19.3 | 0 | (3.000, 1.250) |
| 1 | yes | 12.7 | 125,660 | 23.2 / 34.2 | 19.9 | 0 | (3.000, 1.250) |
| 2 | yes | 7.2 | 139,283 | 24.5 / 34.9 | 21.6 | 0 | (3.000, 1.250) |

- (3.000, 1.250) is the box resting on Table B: table top 1.10 m + half the box height. The box ended upright (|angle| < 0.001 rad).
- Executing a full task (about 31 s of simulated time) takes about 0.5 s of computer time.
- One MuJoCo collision check takes about 50 µs, compared with about 8 µs for roboarm's vectorised checker. That is why planning is slower than roboarm's reported 2.2 s.
- Seed 2's first run reported 10,736 s because the computer slept during planning (wall-clock timer). It was re-run and gave 7.2 s with the identical check count and path.

These are three single runs, made before the contact-stiffness change (that change does not affect planning).

### Benchmark

`python -m roboarm_mujoco.benchmark --runs 10 --seed 0` plans **and physically executes** the task once per seed (seeds 0–9, default layout, same planner settings and machine as above) and writes a new timestamped CSV to `results/` (existing files are never overwritten). "Slip" is how far the box moved in the hand between grasp and release.

| | Contacts 0.02 (MuJoCo default), 2026-11-27 | Contacts 0.004, 2026-11-28 |
|---|---|---|
| Planning / task success | 10/10 / 10/10 | 10/10 / 10/10 |
| Collisions during execution | 0 | 0 |
| Max slip in the hand, mean (min–max) | 40 mm (25–47) | 7 mm (6–8) |
| Final placement error, mean | 0.5 mm | 1.0 mm |
| Joint path, smoothed, mean | 25.6 rad | 25.6 rad (identical paths) |
| Planning time, mean | 7.6 s | 4.4 s |

Results: `results/benchmark_20261127_124821.csv` and `results/benchmark_20261128_153714.csv`. Collision-check counts and paths are identical in both runs; the difference in planning time is laptop speed variation, not the contact change. The slip measure (box relative to the tool centre point) carries a few mm of measurement noise; measured against the finger itself, the slip on seed 0 dropped from 40.0 to 4.0 mm. Use more seeds and layouts before comparing with RL.

---

## 8. What this simulation does NOT represent

- **Still 2D.** The arm and box are confined to a plane; real objects tumble in 3D.
- **Chosen numbers.** Masses, motor gains, friction and grip force are plausible values, not measured from a real robot. Motor torques are not limited.
- **The planner does not know about physics.** It plans rigid, kinematic motions; small tracking errors and slip are only seen during execution. There is no replanning if the box slips.
- **No time-parameterisation.** The motion is constant joint speed along the path, with corners. The velocity servo hides most of the lag.
- **Perfect knowledge.** The planner reads exact positions from the simulation; there are no sensors.

---

## 9. Open items

- **`environments/layout.py`** (environments person) does not exist yet. CSV loading is wired in (`RobotSim.reset(layout_csv=...)` calls `load_layout`), but untested: `tests/test_api.py::test_reset_from_csv` skips until the file and at least one CSV exist.
- **`rl/test_contract.py`** (RL person) does not exist yet. `tests/test_api.py` checks the same agreed keys, shapes and types in the meantime.
- **Kinematic adapter.** The rules say a kinematic backend must also pass the contract test. It is not part of this folder; whoever owns it should confirm.
- **No ceiling or walls** in the default layout: the CSV format has no rows for them.
