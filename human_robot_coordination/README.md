# Personalised Human–Robot Coordination (research prototype)

Academic simulation project. **Not certified, not for real construction sites, never to be
connected to a physical robot.** Anonymous worker profiles only; no face recognition.

## Status

| Stage | Content | State |
|---|---|---|
| 1 | Webcam tracking, pose, movement, calibration, profile save/load | **implemented** |
| 2 | Robot simulation + coordination engine + prediction | not started |
| 3 | Three experimental conditions, metrics, graphs | not started |
| 4 | Full UI | not started |

## Windows setup (Python 3.12)

From the repository root in PowerShell:

```powershell
cd human_robot_coordination
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Notes
* MediaPipe 1.x removed the old `mp.solutions.pose` API. This project uses the **Tasks
  `PoseLandmarker`**, which needs a model file. It is downloaded automatically on first run
  (~9 MB, from Google's MediaPipe model storage) into `models/`.
* In VS Code: *Python: Select Interpreter* -> `human_robot_coordination\.venv\Scripts\python.exe`.

## Run Stage 1

```powershell
.venv\Scripts\python.exe main.py                 # live camera window
.venv\Scripts\python.exe main.py --auto-calibrate
.venv\Scripts\python.exe main.py --profile profiles\worker-xxxxxx.json
.venv\Scripts\python.exe main.py --pref-clearance 1.5 --pref-speed-scale 0.6   # stated preferences
```

Window keys: **C** calibrate (10 s, then saved to `profiles/`), **L** load latest profile,
**N** clear profile, **Q/Esc** quit. Other options: `--camera N`, `--video file.mp4`,
`--calib-seconds S`, `--hfov DEG`, `--headless --max-seconds S`.

### How to calibrate well
* Camera at about chest height, **whole body in view from head to feet** (shoulders and hips
  are mandatory; legs are needed for height). Roughly 2.5–4 m from the camera.
* Good lighting; one person only.
* During the 10 s: stand briefly, **walk a few steps** left/right, and **reach out both arms**.
  Without walking, direction is not estimated; without reaching, arm reach is underestimated
  (the profile records warnings for this).

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```
Tests use a synthetic stick-figure worker with known position/speed/reach (no camera needed).

## Design notes / limitations

* Position/speed are **approximate**: floor-plane position uses a pinhole model with an
  *assumed* horizontal FOV (60 deg, `--hfov`) and depth from torso size. Lateral motion is far
  more accurate than motion toward/away from the camera.
* Thresholds in `config.py` (walking speed, reach ratio, posture limits) are starting values,
  validated on synthetic data only.
* `profiles/*.json` separates `measured` (from observation) and `stated_preferences` (explicit
  worker input). In Stage 2 preferences will only be allowed to make the robot *more*
  cautious; mandatory clearance and emergency stops stay independent of them.
* Tracking loss: gap < 0.6 s -> `DEGRADED` (stale data); longer -> `LOST` with
  `safe_hold_required=True`, shown on screen. The robot that consumes it arrives in Stage 2.
