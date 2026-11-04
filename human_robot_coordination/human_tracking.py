"""Human detection, pose estimation and movement analysis.

Pipeline:   camera frame --PoseEstimator--> RawPose --MotionAnalyzer--> TrackingFrame

* PoseEstimator wraps the MediaPipe Tasks PoseLandmarker (single person).
* MotionAnalyzer turns landmarks into metric quantities (position, velocity, reach,
  body size, posture, activity) and decides when tracking is LOST.

The analyser takes plain NumPy arrays, so it can be tested without a camera.
No face recognition or identification is done; only body landmarks are used.
"""
from __future__ import annotations

import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import POSE_MODEL_PATH, POSE_MODEL_URL, CameraConfig, TrackingConfig

# MediaPipe pose landmark indices (33-point BlazePose topology).
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
CORE = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)

# Skeleton lines for drawing (kept local so we do not depend on a MediaPipe drawing API).
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]


class TrackingStatus(str, Enum):
    TRACKING = "TRACKING"   # body fully usable
    DEGRADED = "DEGRADED"   # brief gap or some limbs missing; data is stale/partial
    LOST = "LOST"           # gone longer than lost_timeout_s -> safe-hold requested


class Activity(str, Enum):
    STANDING = "standing"
    WALKING = "walking"
    REACHING = "reaching"
    UNKNOWN = "unknown"


class Posture(str, Enum):
    UPRIGHT = "upright"
    LEANING = "leaning"
    CROUCHED = "crouched"
    UNKNOWN = "unknown"


@dataclass
class RawPose:
    """One pose detection. Arrays are (33, ...) in MediaPipe landmark order."""
    image_xy: np.ndarray       # (33, 2) normalised image coords (0..1)
    visibility: np.ndarray     # (33,)
    world_xyz: np.ndarray      # (33, 3) metres, origin at hip centre, y points DOWN


@dataclass
class TrackingFrame:
    """Everything downstream modules need about the worker at one instant."""
    timestamp: float
    status: TrackingStatus = TrackingStatus.LOST
    safe_hold_required: bool = True        # True whenever the worker cannot be tracked
    time_since_seen_s: float = 0.0
    reason: str = ""                       # why tracking is degraded/lost (for the HUD)
    # Floor-plane position of the body centre: x = lateral (right +), z = distance
    # from camera. Approximate, derived from a pinhole camera assumption.
    position_m: Optional[tuple] = None
    velocity_mps: Optional[tuple] = None   # (vx, vz)
    speed_mps: float = 0.0
    heading_deg: Optional[float] = None    # 0 = away from camera, +90 = image right
    activity: Activity = Activity.UNKNOWN
    posture: Posture = Posture.UNKNOWN
    lean_deg: Optional[float] = None
    # Body dimensions from this frame (metres).
    shoulder_width_m: Optional[float] = None
    arm_length_m: Optional[float] = None
    arm_reach_m: Optional[float] = None    # current max shoulder->wrist distance
    height_m: Optional[float] = None
    envelope_width_m: Optional[float] = None
    envelope_depth_m: Optional[float] = None
    # Pixel coordinates (x, y, visible) for drawing the skeleton.
    landmarks_px: Optional[np.ndarray] = None


def ensure_pose_model(path: Path = POSE_MODEL_PATH, url: str = POSE_MODEL_URL) -> Path:
    """Download the pose model on first run (about 9 MB)."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {path} ...")
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    except Exception as exc:  # network failure, proxy, disk...
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the pose model from {url}: {exc}\n"
            f"Download it manually and save it as {path}") from exc
    return path


class PoseEstimator:
    """MediaPipe PoseLandmarker in VIDEO mode, tracking at most one person."""

    def __init__(self, cfg: TrackingConfig, model_path: Path = POSE_MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        model_path = ensure_pose_model(model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,                       # one worker only
            min_pose_detection_confidence=cfg.min_detection_confidence,
            min_pose_presence_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_s: float) -> Optional[RawPose]:
        """Return the pose in this frame, or None if nobody is detected."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(timestamp_s * 1000)
        ts_ms = max(ts_ms, self._last_ts_ms + 1)  # MediaPipe needs strictly increasing times
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.pose_landmarks or not result.pose_world_landmarks:
            return None
        lm, wl = result.pose_landmarks[0], result.pose_world_landmarks[0]
        return RawPose(
            image_xy=np.array([[p.x, p.y] for p in lm], dtype=float),
            visibility=np.array([p.visibility for p in lm], dtype=float),
            world_xyz=np.array([[p.x, p.y, p.z] for p in wl], dtype=float),
        )

    def close(self) -> None:
        self._landmarker.close()


class Camera:
    """Thin wrapper that tolerates a missing / failing camera."""

    def __init__(self, cfg: CameraConfig, video_path: Optional[str] = None):
        self.cfg = cfg
        self.is_file = video_path is not None
        self.cap = None
        sources = [(video_path, None)] if video_path else [
            (cfg.index, None), (cfg.index, cv2.CAP_DSHOW), (cfg.index, cv2.CAP_MSMF)]
        for src, backend in sources:
            cap = cv2.VideoCapture(src) if backend is None else cv2.VideoCapture(src, backend)
            ok, _ = cap.read() if cap.isOpened() else (False, None)
            if ok:
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            raise RuntimeError(
                "Could not open the camera" + (f" / video '{video_path}'" if video_path else
                f" (index {cfg.index})") + ". Check it is connected, not used by another "
                "app, and that Windows Settings > Privacy > Camera allows desktop apps.")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


class MotionAnalyzer:
    """Converts RawPose objects into TrackingFrame objects (stateful)."""

    def __init__(self, cfg: TrackingConfig, camera_cfg: CameraConfig):
        self.cfg = cfg
        self.cam = camera_cfg
        self.reset()

    def reset(self) -> None:
        self._history = deque()          # (t, x, z) smoothed positions
        self._last_seen_t: Optional[float] = None
        self._smooth_px = None           # smoothed hip-centre pixel x
        self._smooth_depth = None
        self._world_torso = None         # smoothed metric torso length
        self._walking = False
        self._posture = Posture.UNKNOWN
        self._pending_posture: Optional[tuple] = None   # (posture, since_t)
        self._last_frame: Optional[TrackingFrame] = None

    # ------------------------------------------------------------------ helpers --
    def _usable(self, raw: Optional[RawPose]) -> bool:
        if raw is None:
            return False
        return bool(np.all(raw.visibility[list(CORE)] >= self.cfg.min_landmark_visibility))

    def _visible(self, raw: RawPose, idx) -> bool:
        return bool(np.all(raw.visibility[list(idx)] >= self.cfg.min_landmark_visibility))

    def _velocity(self) -> Optional[tuple]:
        """Least-squares slope of position over the recent window."""
        if len(self._history) < 4:
            return None
        t = np.array([h[0] for h in self._history])
        if t[-1] - t[0] < 0.15:
            return None
        t = t - t[-1]
        vx = np.polyfit(t, [h[1] for h in self._history], 1)[0]
        vz = np.polyfit(t, [h[2] for h in self._history], 1)[0]
        return float(vx), float(vz)

    def _debounce_posture(self, candidate: Posture, t: float) -> Posture:
        if self._posture == Posture.UNKNOWN or candidate == self._posture:
            self._pending_posture = None
            if self._posture == Posture.UNKNOWN:
                self._posture = candidate
            return self._posture
        if self._pending_posture is None or self._pending_posture[0] != candidate:
            self._pending_posture = (candidate, t)
        elif t - self._pending_posture[1] >= self.cfg.posture_min_hold_s:
            self._posture, self._pending_posture = candidate, None
        return self._posture

    # --------------------------------------------------------------------- main --
    def update(self, raw: Optional[RawPose], t: float, frame_wh: tuple) -> TrackingFrame:
        """Process one camera frame. `t` is a monotonic time in seconds."""
        if not self._usable(raw):
            reason = ("No person detected" if raw is None else
                      "Shoulders/hips not visible - step back so the whole body is in view")
            return self._handle_missing(t, reason)

        cfg, w, h = self.cfg, frame_wh[0], frame_wh[1]
        # A gap longer than the timeout invalidates old motion history.
        if self._last_seen_t is not None and t - self._last_seen_t > cfg.lost_timeout_s:
            self._history.clear()
            self._smooth_px = self._smooth_depth = None
            self._walking = False
        self._last_seen_t = t

        img, vis, world = raw.image_xy, raw.visibility, raw.world_xyz
        px = img * np.array([w, h])                       # pixel coordinates
        shoulder_mid_px = px[[L_SHOULDER, R_SHOULDER]].mean(axis=0)
        hip_mid_px = px[[L_HIP, R_HIP]].mean(axis=0)
        shoulder_mid_w = world[[L_SHOULDER, R_SHOULDER]].mean(axis=0)
        hip_mid_w = world[[L_HIP, R_HIP]].mean(axis=0)

        # ---- floor-plane position via a pinhole camera model --------------------
        torso_world = float(np.linalg.norm(shoulder_mid_w - hip_mid_w))
        torso_px = float(np.linalg.norm(shoulder_mid_px - hip_mid_px))
        self._world_torso = torso_world if self._world_torso is None else \
            0.9 * self._world_torso + 0.1 * torso_world
        focal_px = (w / 2.0) / math.tan(math.radians(self.cam.horizontal_fov_deg) / 2.0)
        depth = focal_px * self._world_torso / max(torso_px, 1.0)
        a_p, a_d = cfg.position_smoothing, cfg.depth_smoothing
        self._smooth_px = hip_mid_px[0] if self._smooth_px is None else \
            a_p * hip_mid_px[0] + (1 - a_p) * self._smooth_px
        self._smooth_depth = depth if self._smooth_depth is None else \
            a_d * depth + (1 - a_d) * self._smooth_depth
        x_m = (self._smooth_px - w / 2.0) * self._smooth_depth / focal_px
        z_m = self._smooth_depth
        self._history.append((t, x_m, z_m))
        while self._history and t - self._history[0][0] > cfg.velocity_window_s:
            self._history.popleft()

        vel = self._velocity()
        speed = math.hypot(*vel) if vel else 0.0
        heading = None
        if vel and speed > cfg.walk_speed_off_mps:
            heading = math.degrees(math.atan2(vel[0], vel[1]))   # 0 = +z (away), +90 = +x

        # Walking with hysteresis.
        if self._walking:
            self._walking = speed > cfg.walk_speed_off_mps
        else:
            self._walking = speed > cfg.walk_speed_on_mps

        # ---- arms / reach -------------------------------------------------------
        arm_lengths, reaches, reaching = [], [], False
        for sh, el, wr in ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)):
            if not self._visible(raw, (sh, el, wr)):
                continue
            upper = np.linalg.norm(world[el] - world[sh])
            fore = np.linalg.norm(world[wr] - world[el])
            vec = world[wr] - world[sh]
            dist = float(np.linalg.norm(vec))
            arm_lengths.append(float(upper + fore))
            reaches.append(dist)
            down = np.array([0.0, 1.0, 0.0])             # world y points down
            ratio = dist / (upper + fore) if upper + fore > 1e-6 else 0.0
            if ratio >= cfg.reach_extension_ratio and \
                    _angle_deg(vec, down) >= cfg.reach_min_angle_from_down_deg:
                reaching = True

        if self._walking:
            activity = Activity.WALKING
        elif reaching:
            activity = Activity.REACHING
        else:
            activity = Activity.STANDING

        # ---- posture ------------------------------------------------------------
        lean = _angle_deg(shoulder_mid_w - hip_mid_w, np.array([0.0, -1.0, 0.0]))
        candidate = Posture.UPRIGHT
        crouch_ratio = None
        if self._visible(raw, (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)):
            ankle_mid = world[[L_ANKLE, R_ANKLE]].mean(axis=0)
            knee_mid = world[[L_KNEE, R_KNEE]].mean(axis=0)
            leg_len = np.linalg.norm(hip_mid_w - knee_mid) + np.linalg.norm(knee_mid - ankle_mid)
            if leg_len > 1e-6:
                crouch_ratio = float((ankle_mid[1] - hip_mid_w[1]) / leg_len)
        if crouch_ratio is not None and crouch_ratio < cfg.crouch_ratio_threshold:
            candidate = Posture.CROUCHED
        elif not math.isnan(lean) and lean > cfg.lean_threshold_deg:
            candidate = Posture.LEANING
        posture = self._debounce_posture(candidate, t)

        # ---- body dimensions ----------------------------------------------------
        shoulder_w = float(np.linalg.norm(world[L_SHOULDER] - world[R_SHOULDER]))
        height = None
        if self._visible(raw, (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)):
            leg = 0.5 * sum(
                np.linalg.norm(world[hp] - world[kn]) + np.linalg.norm(world[kn] - world[an])
                for hp, kn, an in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)))
            height = float(leg + torso_world + cfg.head_allowance_m)
        shown = vis >= cfg.min_landmark_visibility
        env_w = float(np.ptp(world[shown, 0])) if shown.sum() > 2 else None
        env_d = float(np.ptp(world[shown, 2])) if shown.sum() > 2 else None

        landmarks_px = np.column_stack([px, shown.astype(float)])
        # DEGRADED when arms/legs are largely missing, otherwise fully TRACKING.
        limb_ok = self._visible(raw, (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)) or len(reaches) > 0
        status = TrackingStatus.TRACKING if limb_ok else TrackingStatus.DEGRADED

        frame = TrackingFrame(
            timestamp=t, status=status, safe_hold_required=False, time_since_seen_s=0.0,
            position_m=(float(x_m), float(z_m)), velocity_mps=vel, speed_mps=speed,
            heading_deg=heading, activity=activity, posture=posture,
            lean_deg=None if math.isnan(lean) else lean,
            shoulder_width_m=shoulder_w,
            arm_length_m=max(arm_lengths) if arm_lengths else None,
            arm_reach_m=max(reaches) if reaches else None,
            height_m=height, envelope_width_m=env_w, envelope_depth_m=env_d,
            landmarks_px=landmarks_px)
        self._last_frame = frame
        return frame

    def _handle_missing(self, t: float, reason: str = "") -> TrackingFrame:
        """No usable body this frame: DEGRADED for a short gap, LOST after the timeout."""
        since = math.inf if self._last_seen_t is None else t - self._last_seen_t
        lost = since > self.cfg.lost_timeout_s
        if lost:
            return TrackingFrame(timestamp=t, status=TrackingStatus.LOST,
                                 safe_hold_required=True, reason=reason,
                                 time_since_seen_s=min(since, 1e6))
        # Short gap: keep last known state but flag it as stale; no velocity extrapolation.
        last = self._last_frame
        return TrackingFrame(
            timestamp=t, status=TrackingStatus.DEGRADED, safe_hold_required=False,
            time_since_seen_s=since, reason=reason,
            position_m=last.position_m if last else None,
            velocity_mps=last.velocity_mps if last else None,
            speed_mps=last.speed_mps if last else 0.0,
            heading_deg=last.heading_deg if last else None,
            activity=last.activity if last else Activity.UNKNOWN,
            posture=last.posture if last else Posture.UNKNOWN)
