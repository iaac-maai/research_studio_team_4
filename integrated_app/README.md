# Cobuild · Live Assist (merged app)

Merges the two projects on this branch (`merged-app`):

| Part | Folder | Role |
|---|---|---|
| Human tracking + calibration | `../human_robot_coordination/` | webcam -> MediaPipe pose -> 10 s calibration -> anonymous worker profile |
| Drywall-installation simulation | `../cobuild_gymnasium_starter/` | Gymnasium env, robot strategies, MuJoCo 3D (unchanged) |
| **Integration** | `integrated_app/` | bridge, subprocess runner, desktop app |

```
camera -> pose -> calibration -> worker profile --bridge--> Cobuild WorkerProfile
       -> simulate: no robot | A reactive | B generic | C personalised -> tiles, charts, 3D
```

## Run (Windows)

```powershell
cd human_robot_coordination
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r ..\integrated_app\requirements.txt
cd ..\integrated_app
..\human_robot_coordination\.venv\Scripts\python.exe app.py
```

No camera? Click **Demo worker (no camera)**, then **Run simulation**.

Pages: **Live Operation** (camera view with callouts, status cards, task flow, robot status, expected value),
**Configure** (profile -> simulation mapping with provenance, manual overrides, stated preferences),
**Analytics** (table, charts, 3D snapshot/video), **Settings**, **Overview**.

## What is real, what is not

* Every number on screen comes from tracking, calibration or the simulation. "Expected value" is
  *simulated* time/strain/cost vs working alone or vs the generic robot; it is blank until you run it.
* The bridge (`bridge.py`) labels each simulated trait as measured / stated / manual / default.
  Lifting capacity and fatigue cannot be measured by camera and stay default unless you enter them.
  Pace comes from walking speed, which is only a proxy (assumption documented in the Configure page).
* The camera-side safety engine (clearance, prediction, motion-consistency check, three experimental
  conditions) is **not built yet**. The only live link to the robot today is the tracking-lost
  SAFE-HOLD indicator.
* PPO strategies are excluded (need PyTorch); rule-based planners are used.
* Research prototype. Not certified; never connect it to a physical robot.

## Design note

Both projects have a top-level `config.py`, so the simulation runs in a subprocess (`sim_runner.py`)
and the Cobuild code is not modified. `sim_client.py` wraps it.

## Tests

```powershell
..\human_robot_coordination\.venv\Scripts\python.exe -m pytest tests ..\human_robot_coordination\tests -q
```
