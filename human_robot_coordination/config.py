"""Central configuration for the human-robot coordination prototype (Stage 1).

All tunable numbers live here so experiments are reproducible. Values were chosen
as reasonable starting points and have NOT been tuned on real worker data.

RESEARCH PROTOTYPE ONLY: simulation, not certified for real construction sites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PROFILES_DIR = PROJECT_DIR / "profiles"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

# MediaPipe Tasks pose model (downloaded on first run). "full" balances speed/accuracy.
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    # Assumed horizontal field of view of a typical webcam. Used to convert image
    # position into approximate metres. This is an ASSUMPTION; adjust per camera.
    horizontal_fov_deg: float = 60.0


@dataclass
class TrackingConfig:
    # Minimum landmark visibility (0-1) to trust a landmark.
    min_landmark_visibility: float = 0.5
    # Pose detector confidence thresholds.
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # If no usable body is seen for this long, tracking is LOST -> safe-hold request.
    lost_timeout_s: float = 0.6
    # Positions are smoothed with an exponential filter (1.0 = no smoothing).
    position_smoothing: float = 0.35
    depth_smoothing: float = 0.2          # depth from a single camera is noisier
    # Velocity is a least-squares slope over this time window.
    velocity_window_s: float = 0.5
    # Activity thresholds (hysteresis avoids flicker between standing and walking).
    walk_speed_on_mps: float = 0.35
    walk_speed_off_mps: float = 0.20
    reach_extension_ratio: float = 0.80    # shoulder-wrist distance / arm length
    reach_min_angle_from_down_deg: float = 45.0  # straight arm hanging down != reaching
    # Posture classification.
    lean_threshold_deg: float = 20.0
    crouch_ratio_threshold: float = 0.75   # hip-ankle height / leg length
    posture_min_hold_s: float = 0.4        # ignore posture flickers shorter than this
    # Rough allowance from shoulder line to top of head, used for approx. height.
    head_allowance_m: float = 0.30


@dataclass
class CalibrationConfig:
    duration_s: float = 10.0
    # Minimum share of calibration time with the worker tracked; otherwise reject.
    min_tracked_fraction: float = 0.60
    # Minimum seconds of walking needed before a "typical direction" is reported.
    min_walking_time_s: float = 1.0
    # Percentile used for "peak" quantities, to reject single-frame landmark glitches.
    peak_percentile: float = 98.0


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)


# --- Safety note (enforced in Stage 2) -------------------------------------------
# Worker preferences may only make the robot MORE cautious. They must never reduce
# the mandatory clearance or override an emergency stop. Stage 2 will define
# MANDATORY_CLEARANCE_M here and always use max(mandatory, anything personalised).
