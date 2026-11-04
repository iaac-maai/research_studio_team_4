"""Synthetic stick-figure worker for testing without a camera or person.

Builds MediaPipe-style landmarks (world metres, y down, hip-centre origin) and
projects them through the same pinhole model the analyser assumes.
"""
from __future__ import annotations

import math

import numpy as np

from human_tracking import RawPose

W, H, FPS = 640, 480, 30.0
HFOV = 60.0
FOCAL = (W / 2) / math.tan(math.radians(HFOV / 2))

ARM = 0.55          # shoulder -> wrist (upper 0.28 + fore 0.27)
SHOULDER_HALF = 0.20


def make_pose(x_m: float, z_m: float, arms: str = "down", crouch: bool = False,
              visible_core: bool = True) -> RawPose:
    """arms: 'down' | 'out' (both arms horizontal) | 'up' (forward-up reach)."""
    w = np.zeros((33, 3))
    leg_y = 0.9 if not crouch else 0.55           # ankle height below hip
    w[11] = (-SHOULDER_HALF, -0.5, 0); w[12] = (SHOULDER_HALF, -0.5, 0)
    w[23] = (-0.10, 0, 0); w[24] = (0.10, 0, 0)
    # Crouch = legs fold with knees forward (leg length stays ~0.9 m, hip drops toward ankles).
    knee = (0.2, 0.35) if crouch else (leg_y / 2, 0.0)
    w[25] = (-0.10, knee[0], knee[1]); w[26] = (0.10, knee[0], knee[1])
    w[27] = (-0.10, leg_y, 0); w[28] = (0.10, leg_y, 0)
    w[0] = (0, -0.7, 0)
    for side, sx in ((0, -1), (1, 1)):
        sh = w[11 + side]
        if arms == "down":
            el, wr = sh + (0, 0.28, 0), sh + (0, ARM, 0)
        elif arms == "out":
            el, wr = sh + (sx * 0.28, 0, 0), sh + (sx * ARM, 0, 0)
        else:
            el, wr = sh + (0, -0.1, 0.27), sh + (0, -0.2, 0.5)
        w[13 + side], w[15 + side] = el, wr
    # Perspective projection: camera at hip height looking along +z.
    depth = z_m + w[:, 2]
    u = W / 2 + FOCAL * (x_m + w[:, 0]) / depth
    v = H / 2 + FOCAL * w[:, 1] / depth
    img = np.column_stack([u / W, v / H])
    vis = np.full(33, 0.95)
    if not visible_core:
        vis[[11, 12, 23, 24]] = 0.1
    return RawPose(image_xy=img, visibility=vis, world_xyz=w)
