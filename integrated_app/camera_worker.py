"""Background thread: camera -> pose -> tracking -> (optional) calibration.

The Tk GUI polls `snapshot()` for the latest annotated frame and tracking state, so slow
pose inference never freezes the interface.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

HRC_DIR = Path(__file__).resolve().parents[1] / "human_robot_coordination"
if str(HRC_DIR) not in sys.path:
    sys.path.insert(0, str(HRC_DIR))

from config import AppConfig  # noqa: E402
from human_tracking import Camera, MotionAnalyzer, PoseEstimator, TrackingFrame  # noqa: E402
from visualisation import draw_skeleton  # noqa: E402
from worker_calibration import CalibrationError, WorkerCalibrator  # noqa: E402
from worker_profile import StatedPreferences, WorkerProfile  # noqa: E402


class CameraWorker:
    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or AppConfig()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame_rgb = None
        self._tracking: Optional[TrackingFrame] = None
        self._fps = 0.0
        self._error: Optional[str] = None
        self._calibrate_request: Optional[StatedPreferences] = None
        self._calibration_result: Optional[WorkerProfile] = None
        self._calibration_error: Optional[str] = None
        self._calibration_progress = 0.0
        self._calibrating = False
        self.running = False

    # --------------------------------------------------------------- control --
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.running = False

    def request_calibration(self, stated: Optional[StatedPreferences] = None) -> None:
        with self._lock:
            self._calibration_result = None
            self._calibration_error = None
            self._calibrate_request = stated or StatedPreferences()

    # ----------------------------------------------------------------- polling --
    def snapshot(self) -> dict:
        with self._lock:
            return dict(frame_rgb=self._frame_rgb, tracking=self._tracking, fps=self._fps,
                        error=self._error, running=self.running, calibrating=self._calibrating,
                        progress=self._calibration_progress)

    def take_calibration_result(self):
        """Returns (profile | None, error | None) once, then clears."""
        with self._lock:
            res, err = self._calibration_result, self._calibration_error
            self._calibration_result = self._calibration_error = None
        return res, err

    # -------------------------------------------------------------------- loop --
    def _loop(self) -> None:
        import cv2
        try:
            camera = Camera(self.cfg.camera)
            estimator = PoseEstimator(self.cfg.tracking)
        except Exception as exc:       # missing camera, missing model download, etc.
            with self._lock:
                self._error = str(exc)
            return
        analyzer = MotionAnalyzer(self.cfg.tracking, self.cfg.camera)
        calibrator = WorkerCalibrator(self.cfg.calibration, self.cfg.camera.horizontal_fov_deg)
        stated = StatedPreferences()
        self.running = True
        t0 = last = time.monotonic()
        fps = 0.0
        try:
            while not self._stop.is_set():
                frame = camera.read()
                if frame is None:
                    with self._lock:
                        self._error = "Camera stopped delivering frames."
                    break
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
                last = now
                try:
                    raw = estimator.detect(frame, now - t0)
                except Exception:
                    raw = None          # one bad frame must not kill the loop
                tf = analyzer.update(raw, now, (frame.shape[1], frame.shape[0]))

                with self._lock:
                    request, self._calibrate_request = self._calibrate_request, None
                if request is not None:
                    stated = request
                    calibrator.start(now)

                if calibrator.active:
                    calibrator.add(tf)
                    if calibrator.finished:
                        try:
                            result, err = calibrator.build_profile(stated), None
                        except CalibrationError as exc:
                            result, err = None, str(exc)
                        calibrator.active = False
                        with self._lock:
                            self._calibration_result, self._calibration_error = result, err

                # Skeleton only: the GUI draws its own callouts, status and progress overlays.
                draw_skeleton(frame, tf, (24, 197, 245))      # BGR yellow
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame_rgb, self._tracking, self._fps = rgb, tf, fps
                    self._calibrating = calibrator.active
                    self._calibration_progress = calibrator.progress
        finally:
            camera.release()
            estimator.close()
            self.running = False
            with self._lock:
                self._calibrating = False
