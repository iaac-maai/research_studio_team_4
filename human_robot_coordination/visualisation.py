"""OpenCV drawing helpers for the camera feed (skeleton + status HUD)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from human_tracking import SKELETON, TrackingFrame, TrackingStatus
from worker_profile import WorkerProfile

_COLOURS = {
    TrackingStatus.TRACKING: (80, 200, 80),
    TrackingStatus.DEGRADED: (0, 200, 255),
    TrackingStatus.LOST: (60, 60, 230),
}


def _text(img, text, org, scale=0.5, colour=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _panel(img, x0, y0, x1, y1, alpha=0.6):
    """Dark translucent box so text stays readable on any background."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha)).astype(roi.dtype)


def draw_skeleton(img: np.ndarray, tf: TrackingFrame, colour=None) -> None:
    """Draw only the body skeleton (no text), in place."""
    colour = colour or _COLOURS[tf.status]
    if tf.landmarks_px is not None and tf.status != TrackingStatus.LOST:
        pts = tf.landmarks_px
        for a, b in SKELETON:
            if pts[a, 2] > 0 and pts[b, 2] > 0:
                cv2.line(img, tuple(pts[a, :2].astype(int)), tuple(pts[b, :2].astype(int)), colour, 3)
        for x, y, v in pts:
            if v > 0:
                cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), -1)


def draw_tracking(img: np.ndarray, tf: TrackingFrame, fps: float = 0.0) -> None:
    """Draw skeleton and tracking readout on the image in place."""
    draw_skeleton(img, tf)

    lines = [f"Status: {tf.status.value}   FPS {fps:.0f}"]
    if tf.status == TrackingStatus.LOST:
        lines.append("WORKER LOST -> SAFE-HOLD sent to (simulated) robot")
        if tf.reason:
            lines.append(tf.reason[:70])
    else:
        if tf.position_m:
            lines.append(f"Position x={tf.position_m[0]:+.2f} m  dist={tf.position_m[1]:.2f} m")
        head = f"{tf.heading_deg:+.0f} deg" if tf.heading_deg is not None else "-"
        lines.append(f"Speed {tf.speed_mps:.2f} m/s  heading {head}")
        lines.append(f"Activity: {tf.activity.value}   Posture: {tf.posture.value}")
        reach = f"{tf.arm_reach_m:.2f} m" if tf.arm_reach_m is not None else "-"
        shoulders = f"{tf.shoulder_width_m:.2f} m" if tf.shoulder_width_m is not None else "-"
        lines.append(f"Arm reach {reach}  shoulders {shoulders}")
    _panel(img, 0, 0, 480, 10 + 20 * len(lines))
    for i, line in enumerate(lines):
        c = (110, 110, 255) if "SAFE-HOLD" in line else (255, 255, 255)
        _text(img, line, (10, 22 + 20 * i), 0.5, c)

    if tf.safe_hold_required:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (60, 60, 230), 4)


def draw_calibration(img: np.ndarray, progress: float, tracked_fraction: float, duration_s: float) -> None:
    h, w = img.shape[:2]
    _panel(img, 0, h - 62, w, h)
    cv2.rectangle(img, (10, h - 40), (w - 10, h - 20), (50, 50, 50), -1)
    cv2.rectangle(img, (10, h - 40), (10 + int((w - 20) * progress), h - 20), (0, 180, 255), -1)
    remaining = max(0.0, duration_s * (1 - progress))
    _text(img, f"CALIBRATING {remaining:4.1f}s  tracked {tracked_fraction:.0%}  "
               "- stand, walk a few steps, reach out both arms", (10, h - 48), 0.45, (0, 220, 255))


def draw_profile_panel(img: np.ndarray, profile: Optional[WorkerProfile]) -> None:
    h, w = img.shape[:2]
    lines = ["No profile loaded (C = calibrate, L = load latest)"] if profile is None \
        else profile.summary_lines()
    y0 = h - 40 - 18 * len(lines)          # bottom-left, above the message line
    _panel(img, 0, y0 - 16, 400, h - 30)
    for i, line in enumerate(lines):
        _text(img, line, (10, y0 + 18 * i), 0.42, (200, 255, 200) if profile else (220, 220, 220))
