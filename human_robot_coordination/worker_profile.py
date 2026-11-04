"""Anonymous worker profile: data model plus JSON save/load.

The profile keeps two things strictly apart:
  * `measured`            - values computed from camera observation during calibration.
  * `stated_preferences`  - values the worker explicitly told us. Never inferred.

No face data, images or identity are stored. The ID is a random token.
"""
from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import PROFILES_DIR

SCHEMA_VERSION = 1


@dataclass
class MeasuredCharacteristics:
    """Everything here comes from observation. Distances in metres, speeds in m/s."""
    # Movement
    average_speed_mps: float = 0.0           # mean over all tracked time (incl. standing)
    walking_speed_mps: Optional[float] = None  # mean over walking frames only
    peak_speed_mps: float = 0.0              # high percentile, not the raw max
    movement_range_x_m: float = 0.0          # lateral spread (p5-p95) of body centre
    movement_range_depth_m: float = 0.0      # toward/away-from-camera spread (noisier)
    # Direction: mean heading of walking, 0 deg = away from camera, +90 = to image right.
    typical_heading_deg: Optional[float] = None
    heading_concentration: Optional[float] = None  # 0 (all directions) .. 1 (one direction)
    # Variability
    speed_std_mps: float = 0.0
    speed_cv: Optional[float] = None         # std / mean speed
    heading_circular_std_deg: Optional[float] = None
    # Body
    height_m: Optional[float] = None         # approximate
    shoulder_width_m: Optional[float] = None
    arm_length_m: Optional[float] = None
    arm_reach_m: Optional[float] = None      # peak shoulder-to-wrist distance
    envelope_width_m: Optional[float] = None
    envelope_depth_m: Optional[float] = None
    envelope_radius_m: Optional[float] = None
    # Working space = p95 distance travelled from the mean position + arm reach.
    working_space_radius_m: Optional[float] = None
    # Behaviour mix
    activity_fractions: dict = field(default_factory=dict)  # standing/walking/reaching
    posture_fractions: dict = field(default_factory=dict)
    posture_changes_per_min: float = 0.0


@dataclass
class StatedPreferences:
    """Explicitly provided by the worker. None means 'not stated'.

    SAFETY: these are requests only. The coordination engine (Stage 2) must clamp
    them so they can never reduce mandatory safety distances or emergency stops.
    """
    preferred_clearance_m: Optional[float] = None
    preferred_robot_speed_scale: Optional[float] = None  # e.g. 0.5 = slower robot
    preferred_approach_side: Optional[str] = None        # "left" | "right" | None
    notes: str = ""
    source: str = "stated_by_worker"


@dataclass
class CalibrationInfo:
    duration_s: float = 0.0
    frames_total: int = 0
    frames_tracked: int = 0
    tracked_fraction: float = 0.0
    walking_time_s: float = 0.0
    reaching_time_s: float = 0.0
    warnings: list = field(default_factory=list)
    camera_hfov_deg_assumed: float = 60.0


@dataclass
class WorkerProfile:
    profile_id: str = ""
    created_utc: str = ""
    schema_version: int = SCHEMA_VERSION
    measured: MeasuredCharacteristics = field(default_factory=MeasuredCharacteristics)
    stated_preferences: StatedPreferences = field(default_factory=StatedPreferences)
    calibration: CalibrationInfo = field(default_factory=CalibrationInfo)
    disclaimer: str = ("Research prototype. Measured values are camera estimates, not "
                       "certified measurements. Contains no identity or face data.")

    # ---- construction ------------------------------------------------------------
    @staticmethod
    def new_anonymous() -> "WorkerProfile":
        return WorkerProfile(
            profile_id="worker-" + secrets.token_hex(3),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # ---- (de)serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "WorkerProfile":
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported profile schema_version {version!r} "
                             f"(expected {SCHEMA_VERSION}).")

        def build(cls, values):
            # Ignore unknown keys, keep defaults for missing ones.
            known = cls.__dataclass_fields__.keys()
            return cls(**{k: v for k, v in (values or {}).items() if k in known})

        return WorkerProfile(
            profile_id=data.get("profile_id", ""),
            created_utc=data.get("created_utc", ""),
            schema_version=version,
            measured=build(MeasuredCharacteristics, data.get("measured")),
            stated_preferences=build(StatedPreferences, data.get("stated_preferences")),
            calibration=build(CalibrationInfo, data.get("calibration")),
            disclaimer=data.get("disclaimer", ""),
        )

    def summary_lines(self) -> list:
        """Short human-readable lines for the on-screen panel / console."""
        m, p = self.measured, self.stated_preferences

        def f(v, unit="", nd=2):
            return "n/a" if v is None else f"{v:.{nd}f}{unit}"

        return [
            f"Profile: {self.profile_id}",
            f"Avg speed {f(m.average_speed_mps, ' m/s')}  walk {f(m.walking_speed_mps, ' m/s')}",
            f"Arm reach {f(m.arm_reach_m, ' m')}  shoulders {f(m.shoulder_width_m, ' m')}",
            f"Envelope {f(m.envelope_width_m, ' m')} x {f(m.envelope_depth_m, ' m')}",
            f"Work radius {f(m.working_space_radius_m, ' m')}",
            f"Speed CV {f(m.speed_cv)}  heading {f(m.typical_heading_deg, ' deg', 0)}",
            f"Stated: clearance {f(p.preferred_clearance_m, ' m')}, "
            f"robot speed x{f(p.preferred_robot_speed_scale)}",
        ]


def save_profile(profile: WorkerProfile, directory: Path = PROFILES_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{profile.profile_id}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile.to_dict(), fh, indent=2)
    return path


def load_profile(path: Path) -> WorkerProfile:
    with open(path, "r", encoding="utf-8") as fh:
        return WorkerProfile.from_dict(json.load(fh))


def list_profiles(directory: Path = PROFILES_DIR) -> list:
    """Profile files, newest first."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("worker-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
