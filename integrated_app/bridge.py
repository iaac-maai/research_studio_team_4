"""Bridge: camera-calibrated worker profile  ->  Cobuild simulation WorkerProfile.

Every simulated field records its PROVENANCE so the research record stays honest:
    measured  - estimated by the camera during calibration
    stated    - explicitly told by the worker
    manual    - typed in by the researcher in the app
    default   - NOT known; the simulation's generic-average value is used

A webcam cannot measure lifting capacity or fatigue, so those are always default/manual.
Mapping assumptions are listed in MAPPING_NOTES and are shown in the app.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HRC_DIR = Path(__file__).resolve().parents[1] / "human_robot_coordination"
if str(HRC_DIR) not in sys.path:
    sys.path.insert(0, str(HRC_DIR))

from worker_profile import WorkerProfile as CameraProfile  # noqa: E402  (camera project)

# Generic-average worker used by the Cobuild simulation (cobuild_gymnasium_starter/profiles.py).
GENERIC = dict(height_m=1.75, reach_m=0.665, lift_capacity_kg=34.0, pace=1.0,
               handedness="right", fatigue_rate=0.0008)

HAND_ALLOWANCE_M = 0.09       # shoulder->wrist arm length + hand = reach used by the simulation
REFERENCE_WALK_MPS = 1.2      # typical comfortable walking speed; ASSUMPTION
PACE_LIMITS = (0.80, 1.30)    # range the simulation was built for
HEIGHT_LIMITS = (1.50, 2.00)
REACH_LIMITS = (0.50, 0.80)

MAPPING_NOTES = [
    "height_m   <- camera height estimate (legs + torso + head allowance), clipped to 1.50-2.00 m",
    "reach_m    <- measured arm length + 0.09 m hand, clipped to 0.50-0.80 m",
    "pace       <- ASSUMPTION: (1.2 m/s / measured walking speed)^0.5, clipped to 0.80-1.30. "
    "Walking speed is only a proxy for work pace; override it if you know better.",
    "handedness <- the worker's STATED preferred side; default right",
    "lift_capacity_kg, fatigue_rate: not measurable by camera -> default or manual override",
]


@dataclass
class SimProfile:
    values: dict
    provenance: dict
    warnings: list = field(default_factory=list)

    def sim_json(self) -> dict:
        """Exactly the fields of Cobuild's WorkerProfile dataclass."""
        return dict(self.values)


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def to_sim_profile(profile: CameraProfile, overrides: Optional[dict] = None) -> SimProfile:
    """Map a camera profile to a simulation profile. `overrides` hold manual researcher values."""
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    m, s = profile.measured, profile.stated_preferences
    vals, prov, warn = {}, {}, []

    if m.height_m is not None:
        vals["height_m"], prov["height_m"] = round(_clip(m.height_m, *HEIGHT_LIMITS), 3), "measured"
        if m.height_m != vals["height_m"]:
            warn.append(f"Height {m.height_m:.2f} m clipped to {vals['height_m']:.2f} m.")
    else:
        vals["height_m"], prov["height_m"] = GENERIC["height_m"], "default"
        warn.append("Height not measured (legs not visible): using generic 1.75 m.")

    arm = m.arm_length_m if m.arm_length_m is not None else m.arm_reach_m
    if arm is not None:
        vals["reach_m"] = round(_clip(arm + HAND_ALLOWANCE_M, *REACH_LIMITS), 3)
        prov["reach_m"] = "measured"
    else:
        vals["reach_m"], prov["reach_m"] = GENERIC["reach_m"], "default"
        warn.append("Arm length not measured: using generic reach.")

    if m.walking_speed_mps:
        vals["pace"] = round(_clip((REFERENCE_WALK_MPS / m.walking_speed_mps) ** 0.5, *PACE_LIMITS), 3)
        prov["pace"] = "measured"
    else:
        vals["pace"], prov["pace"] = GENERIC["pace"], "default"
        warn.append("No walking observed: pace left at the average (1.0).")

    if s.preferred_approach_side in ("left", "right"):
        vals["handedness"], prov["handedness"] = s.preferred_approach_side, "stated"
    else:
        vals["handedness"], prov["handedness"] = GENERIC["handedness"], "default"

    for key in ("lift_capacity_kg", "fatigue_rate"):
        vals[key], prov[key] = GENERIC[key], "default"

    for key, value in overrides.items():          # researcher overrides win, and are labelled
        if key in vals:
            vals[key], prov[key] = value, "manual"

    vals["name"] = profile.profile_id or "camera_worker"
    vals["description"] = "Built from camera calibration (see provenance)."
    prov["name"] = "measured"
    return SimProfile(vals, prov, warn)
