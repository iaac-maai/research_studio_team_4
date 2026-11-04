"""Individual worker calibration: collects TrackingFrames, builds a WorkerProfile.

Nothing from one worker is assumed for another: every number in the profile comes
from the calibration window of that worker (or from explicit stated preferences).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from config import CalibrationConfig
from human_tracking import Activity, Posture, TrackingFrame, TrackingStatus
from worker_profile import StatedPreferences, WorkerProfile


class CalibrationError(RuntimeError):
    """Raised when the calibration data is not good enough to build a profile."""


class WorkerCalibrator:
    def __init__(self, cfg: CalibrationConfig, hfov_deg_assumed: float = 60.0):
        self.cfg = cfg
        self.hfov = hfov_deg_assumed
        self.reset()

    def reset(self) -> None:
        self._start_t: Optional[float] = None
        self._last_t: Optional[float] = None
        self._frames = []        # tracked frames only
        self._n_total = 0
        self._walking_s = 0.0
        self._reaching_s = 0.0
        self._activity_s = {a.value: 0.0 for a in (Activity.STANDING, Activity.WALKING, Activity.REACHING)}
        self._posture_s = {p.value: 0.0 for p in (Posture.UPRIGHT, Posture.LEANING, Posture.CROUCHED)}
        self._posture_changes = 0
        self._prev_posture: Optional[Posture] = None
        self.active = False

    # ---------------------------------------------------------------- lifecycle --
    def start(self, t: float) -> None:
        self.reset()
        self._start_t = self._last_t = t
        self.active = True

    @property
    def progress(self) -> float:
        if not self.active or self._start_t is None or self._last_t is None:
            return 0.0
        return min(1.0, (self._last_t - self._start_t) / self.cfg.duration_s)

    @property
    def finished(self) -> bool:
        return self.active and self.progress >= 1.0

    @property
    def tracked_fraction(self) -> float:
        return len(self._frames) / self._n_total if self._n_total else 0.0

    def add(self, frame: TrackingFrame) -> None:
        """Feed one tracking frame while calibrating."""
        if not self.active:
            return
        dt = max(0.0, frame.timestamp - self._last_t)
        self._last_t = frame.timestamp
        self._n_total += 1
        if frame.status != TrackingStatus.TRACKING or frame.position_m is None:
            return                                  # lost/degraded frames carry no new data
        self._frames.append(frame)
        if frame.activity.value in self._activity_s:
            self._activity_s[frame.activity.value] += dt
        if frame.posture.value in self._posture_s:
            self._posture_s[frame.posture.value] += dt
        if self._prev_posture not in (None, Posture.UNKNOWN) and frame.posture != self._prev_posture \
                and frame.posture != Posture.UNKNOWN:
            self._posture_changes += 1
        if frame.posture != Posture.UNKNOWN:
            self._prev_posture = frame.posture

    # ------------------------------------------------------------------ profile --
    def build_profile(self, stated: Optional[StatedPreferences] = None) -> WorkerProfile:
        cfg = self.cfg
        duration = (self._last_t - self._start_t) if self._start_t is not None else 0.0
        if self._n_total == 0 or not self._frames:
            raise CalibrationError("No worker was tracked during calibration.")
        if self.tracked_fraction < cfg.min_tracked_fraction:
            raise CalibrationError(
                f"Worker was tracked only {self.tracked_fraction:.0%} of the time "
                f"(need {cfg.min_tracked_fraction:.0%}). Stay fully in view and retry.")

        frames = self._frames
        pct = cfg.peak_percentile
        profile = WorkerProfile.new_anonymous()
        m = profile.measured
        warnings = []

        speeds = np.array([f.speed_mps for f in frames])
        m.average_speed_mps = float(speeds.mean())
        m.speed_std_mps = float(speeds.std())
        m.peak_speed_mps = float(np.percentile(speeds, pct))
        m.speed_cv = float(m.speed_std_mps / m.average_speed_mps) if m.average_speed_mps > 1e-3 else None

        walking = [f for f in frames if f.activity == Activity.WALKING]
        if walking:
            m.walking_speed_mps = float(np.mean([f.speed_mps for f in walking]))

        xs = np.array([f.position_m[0] for f in frames])
        zs = np.array([f.position_m[1] for f in frames])
        m.movement_range_x_m = float(np.percentile(xs, 95) - np.percentile(xs, 5))
        m.movement_range_depth_m = float(np.percentile(zs, 95) - np.percentile(zs, 5))

        # Typical direction: circular mean of walking headings.
        headings = np.radians([f.heading_deg for f in walking if f.heading_deg is not None])
        if self._activity_s["walking"] >= cfg.min_walking_time_s and len(headings) >= 5:
            s, c = np.sin(headings).mean(), np.cos(headings).mean()
            r = float(math.hypot(s, c))
            m.typical_heading_deg = float(math.degrees(math.atan2(s, c)))
            m.heading_concentration = r
            m.heading_circular_std_deg = float(math.degrees(math.sqrt(-2.0 * math.log(max(r, 1e-6)))))
        else:
            warnings.append("Not enough walking observed: typical direction not estimated.")

        # Body dimensions: medians are robust to single-frame landmark glitches.
        def med(values):
            vals = [v for v in values if v is not None]
            return float(np.median(vals)) if vals else None

        m.shoulder_width_m = med(f.shoulder_width_m for f in frames)
        m.arm_length_m = med(f.arm_length_m for f in frames)
        m.height_m = med(f.height_m for f in frames)
        if m.height_m is None:
            warnings.append("Legs not visible: height not estimated. Step back so the full body is in view.")
        reaches = [f.arm_reach_m for f in frames if f.arm_reach_m is not None]
        if reaches:
            reach = float(np.percentile(reaches, pct))
            # Reach cannot physically exceed the arm length; clip landmark noise.
            m.arm_reach_m = min(reach, m.arm_length_m) if m.arm_length_m else reach
        else:
            warnings.append("Arms not visible: arm reach not estimated.")
        if self._activity_s["reaching"] < 0.3:
            warnings.append("No clear reaching observed: arm reach may be underestimated. "
                            "Reach out with both arms during calibration.")

        m.envelope_width_m = med(f.envelope_width_m for f in frames)
        m.envelope_depth_m = med(f.envelope_depth_m for f in frames)
        if m.envelope_width_m is not None and m.envelope_depth_m is not None:
            m.envelope_radius_m = 0.5 * math.hypot(m.envelope_width_m, m.envelope_depth_m)

        # Working space: how far the body centre roams + how far the arms reach from it.
        centre = np.array([xs.mean(), zs.mean()])
        dist = np.hypot(xs - centre[0], zs - centre[1])
        roam = float(np.percentile(dist, 95))
        if m.arm_reach_m is not None:
            m.working_space_radius_m = roam + m.arm_reach_m
        else:
            m.working_space_radius_m = roam + (m.envelope_radius_m or 0.0)

        total_activity = sum(self._activity_s.values()) or 1.0
        m.activity_fractions = {k: round(v / total_activity, 3) for k, v in self._activity_s.items()}
        total_posture = sum(self._posture_s.values()) or 1.0
        m.posture_fractions = {k: round(v / total_posture, 3) for k, v in self._posture_s.items()}
        tracked_time = sum(self._activity_s.values())
        m.posture_changes_per_min = float(self._posture_changes / (tracked_time / 60.0)) if tracked_time > 1 else 0.0

        if self.tracked_fraction < 0.85:
            warnings.append(f"Tracked only {self.tracked_fraction:.0%} of the time; profile may be less reliable.")

        info = profile.calibration
        info.duration_s = float(duration)
        info.frames_total = self._n_total
        info.frames_tracked = len(frames)
        info.tracked_fraction = float(self.tracked_fraction)
        info.walking_time_s = float(self._activity_s["walking"])
        info.reaching_time_s = float(self._activity_s["reaching"])
        info.warnings = warnings
        info.camera_hfov_deg_assumed = self.hfov

        if stated is not None:
            profile.stated_preferences = stated
        self.active = False
        return profile
