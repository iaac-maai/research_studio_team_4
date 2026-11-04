"""Stage 1 application: webcam tracking + individual worker calibration.

Keys (focus the video window):
    C  start calibration (new anonymous profile, saved to profiles/)
    L  load the most recent saved profile
    N  clear the loaded profile
    Q / Esc  quit

RESEARCH PROTOTYPE: simulation only. Never connect to a physical robot.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import cv2

from config import AppConfig, PROFILES_DIR
from human_tracking import Camera, MotionAnalyzer, PoseEstimator, TrackingStatus
from visualisation import draw_calibration, draw_profile_panel, draw_tracking
from worker_calibration import CalibrationError, WorkerCalibrator
from worker_profile import (StatedPreferences, WorkerProfile, list_profiles,
                            load_profile, save_profile)

WINDOW = "Human tracking & calibration (Stage 1)"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--video", type=str, default=None, help="use a video file instead of a camera")
    p.add_argument("--profile", type=str, default=None, help="path to a profile JSON to load at start")
    p.add_argument("--calib-seconds", type=float, default=None, help="calibration duration (default 10)")
    p.add_argument("--hfov", type=float, default=None, help="assumed camera horizontal FOV in degrees")
    p.add_argument("--auto-calibrate", action="store_true", help="start calibrating immediately")
    p.add_argument("--max-seconds", type=float, default=None, help="exit automatically after N seconds (testing)")
    p.add_argument("--headless", action="store_true", help="no window (testing)")
    # Explicit worker preferences (kept separate from measured values).
    p.add_argument("--pref-clearance", type=float, default=None, help="preferred clearance in metres")
    p.add_argument("--pref-speed-scale", type=float, default=None, help="preferred robot speed scale, e.g. 0.5")
    p.add_argument("--pref-side", choices=["left", "right"], default=None, help="preferred robot approach side")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = AppConfig()
    cfg.camera.index = args.camera
    if args.calib_seconds:
        cfg.calibration.duration_s = args.calib_seconds
    if args.hfov:
        cfg.camera.horizontal_fov_deg = args.hfov

    stated = StatedPreferences(preferred_clearance_m=args.pref_clearance,
                               preferred_robot_speed_scale=args.pref_speed_scale,
                               preferred_approach_side=args.pref_side)

    try:
        camera = Camera(cfg.camera, args.video)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    estimator = PoseEstimator(cfg.tracking)
    analyzer = MotionAnalyzer(cfg.tracking, cfg.camera)
    calibrator = WorkerCalibrator(cfg.calibration, cfg.camera.horizontal_fov_deg)

    profile: Optional[WorkerProfile] = None
    if args.profile:
        try:
            profile = load_profile(args.profile)
            print(f"Loaded profile {profile.profile_id}")
        except (OSError, ValueError) as exc:
            print(f"Could not load profile '{args.profile}': {exc}")
    message, message_until = "", 0.0
    if args.auto_calibrate:
        calibrator.start(time.monotonic())

    t0 = time.monotonic()
    last_t, fps = t0, 0.0
    try:
        while True:
            frame = camera.read()
            if frame is None:
                if camera.is_file:
                    break
                print("Camera frame read failed; stopping.")
                break
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3)
            last_t = now

            try:
                raw = estimator.detect(frame, now - t0)
            except Exception as exc:     # a bad frame must never crash the safety loop
                print(f"Pose estimation error: {exc}")
                raw = None
            tf = analyzer.update(raw, now, (frame.shape[1], frame.shape[0]))

            if calibrator.active:
                calibrator.add(tf)
                if calibrator.finished:
                    try:
                        profile = calibrator.build_profile(stated)
                        path = save_profile(profile)
                        message = f"Profile saved: {path.name}"
                        print(message)
                        for line in profile.summary_lines():
                            print("  " + line)
                        for w in profile.calibration.warnings:
                            print("  WARNING: " + w)
                    except CalibrationError as exc:
                        message = f"Calibration failed: {exc}"
                        print(message)
                        calibrator.active = False
                    message_until = now + 8

            if not args.headless:
                draw_tracking(frame, tf, fps)
                if calibrator.active:
                    draw_calibration(frame, calibrator.progress, calibrator.tracked_fraction,
                                     cfg.calibration.duration_s)
                else:
                    draw_profile_panel(frame, profile)
                if message and now < message_until:
                    cv2.putText(frame, message[:70], (10, frame.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    calibrator.start(now)
                    message = ""
                elif key == ord("l"):
                    files = list_profiles()
                    if files:
                        profile = load_profile(files[0])
                        message = f"Loaded {profile.profile_id}"
                    else:
                        message = f"No profiles in {PROFILES_DIR}"
                    message_until = now + 4
                elif key == ord("n"):
                    profile = None

            if args.max_seconds and now - t0 > args.max_seconds:
                break
    finally:
        camera.release()
        estimator.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
