"""Stage 1 tests: tracking analysis, lost-tracking handling, calibration, profile I/O."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CalibrationConfig, CameraConfig, TrackingConfig  # noqa: E402
from human_tracking import Activity, MotionAnalyzer, Posture, TrackingStatus  # noqa: E402
from synthetic import FPS, H, HFOV, W, make_pose  # noqa: E402
from worker_calibration import CalibrationError, WorkerCalibrator  # noqa: E402
from worker_profile import (StatedPreferences, WorkerProfile, load_profile,  # noqa: E402
                            save_profile)

DT = 1.0 / FPS


def analyzer():
    return MotionAnalyzer(TrackingConfig(), CameraConfig(horizontal_fov_deg=HFOV))


def run(an, seconds, pose_fn, t0=0.0):
    """Feed `seconds` of frames; pose_fn(t) returns RawPose or None. Returns frames."""
    out, n = [], int(seconds * FPS)
    for i in range(n):
        t = t0 + i * DT
        out.append(an.update(pose_fn(t), t, (W, H)))
    return out


def test_standing_position_and_activity():
    frames = run(analyzer(), 2, lambda t: make_pose(0.5, 3.0))
    f = frames[-1]
    assert f.status == TrackingStatus.TRACKING and not f.safe_hold_required
    assert f.activity == Activity.STANDING
    assert f.speed_mps < 0.1
    assert f.position_m[0] == pytest.approx(0.5, abs=0.1)
    assert f.position_m[1] == pytest.approx(3.0, rel=0.1)
    assert f.shoulder_width_m == pytest.approx(0.40, abs=0.01)
    assert f.posture == Posture.UPRIGHT


def test_walking_speed_and_heading():
    frames = run(analyzer(), 3, lambda t: make_pose(-1.0 + 1.0 * t, 3.0))
    f = frames[-1]
    assert f.activity == Activity.WALKING
    assert f.speed_mps == pytest.approx(1.0, abs=0.15)
    assert f.heading_deg == pytest.approx(90.0, abs=10)   # moving to image right


def test_walking_away_from_camera_heading_zero():
    frames = run(analyzer(), 3, lambda t: make_pose(0.0, 2.0 + 0.8 * t))
    f = frames[-1]
    assert f.activity == Activity.WALKING
    assert f.heading_deg == pytest.approx(0.0, abs=20)


def test_reaching_vs_arms_hanging():
    down = run(analyzer(), 1, lambda t: make_pose(0, 3, arms="down"))[-1]
    out = run(analyzer(), 1, lambda t: make_pose(0, 3, arms="out"))[-1]
    assert down.activity == Activity.STANDING        # straight arm hanging is NOT reaching
    assert out.activity == Activity.REACHING
    assert out.arm_reach_m == pytest.approx(0.55, abs=0.01)


def test_crouch_and_posture_debounce():
    an = analyzer()
    run(an, 1, lambda t: make_pose(0, 3))
    # A 0.2 s crouch flicker must be ignored (min hold 0.4 s)...
    short = run(an, 0.2, lambda t: make_pose(0, 3, crouch=True), t0=1.0)
    assert short[-1].posture == Posture.UPRIGHT
    # ...a sustained crouch is accepted.
    long = run(an, 1.0, lambda t: make_pose(0, 3, crouch=True), t0=1.2)
    assert long[-1].posture == Posture.CROUCHED


def test_tracking_lost_triggers_safe_hold_and_recovers():
    an = analyzer()
    present = lambda t: make_pose(0, 3)
    run(an, 1, present)
    # 0.3 s gap: degraded but not yet lost.
    gap = run(an, 0.3, lambda t: None, t0=1.0)
    assert gap[-1].status == TrackingStatus.DEGRADED and not gap[-1].safe_hold_required
    # Beyond the timeout: LOST + safe-hold.
    gap2 = run(an, 1.0, lambda t: None, t0=1.3)
    assert gap2[-1].status == TrackingStatus.LOST and gap2[-1].safe_hold_required
    # Reappears.
    back = run(an, 0.5, present, t0=2.3)
    assert back[-1].status == TrackingStatus.TRACKING and not back[-1].safe_hold_required
    # No huge speed spike from the stale pre-gap history.
    assert max(f.speed_mps for f in back) < 0.5


def test_never_seen_is_lost():
    f = analyzer().update(None, 0.0, (W, H))
    assert f.status == TrackingStatus.LOST and f.safe_hold_required


def test_low_visibility_core_landmarks_treated_as_missing():
    an = analyzer()
    run(an, 0.5, lambda t: make_pose(0, 3))
    frames = run(an, 1.0, lambda t: make_pose(0, 3, visible_core=False), t0=0.5)
    assert frames[-1].status == TrackingStatus.LOST


# ------------------------------------------------------------------ calibration --
def scripted_worker(t):
    """3 s stand, 4 s walk right then left, 3 s arms out."""
    if t < 3:
        return make_pose(0.0, 3.0)
    if t < 5:
        return make_pose(1.0 * (t - 3), 3.0)
    if t < 7:
        return make_pose(2.0 - 1.0 * (t - 5), 3.0)
    return make_pose(0.0, 3.0, arms="out")


def calibrate(pose_fn, duration=10.0):
    an = analyzer()
    cal = WorkerCalibrator(CalibrationConfig(duration_s=duration), HFOV)
    cal.start(0.0)
    for i in range(int(duration * FPS) + 1):
        t = i * DT
        cal.add(an.update(pose_fn(t), t, (W, H)))
    assert cal.finished
    return cal


def test_calibration_builds_sensible_profile():
    prof = calibrate(scripted_worker).build_profile(StatedPreferences(preferred_clearance_m=1.2))
    m = prof.measured
    assert prof.profile_id.startswith("worker-")
    assert m.arm_reach_m == pytest.approx(0.55, abs=0.02)
    assert m.shoulder_width_m == pytest.approx(0.40, abs=0.01)
    assert m.walking_speed_mps == pytest.approx(1.0, abs=0.2)
    assert 0.0 < m.average_speed_mps < m.walking_speed_mps
    assert m.speed_cv is not None and m.speed_cv > 0.5      # stop-and-go => high variability
    assert m.movement_range_x_m > 1.0
    assert m.working_space_radius_m > m.arm_reach_m
    assert m.height_m == pytest.approx(0.9 + 0.5 + 0.30, abs=0.05)
    assert prof.calibration.tracked_fraction > 0.95
    assert prof.calibration.walking_time_s > 2.0
    # Measured and stated stay separate.
    assert prof.stated_preferences.preferred_clearance_m == 1.2
    assert not hasattr(m, "preferred_clearance_m")


def test_calibration_differs_between_workers():
    slow = lambda t: make_pose(0.4 * math.sin(t), 3.0)
    fast = lambda t: make_pose(1.5 * math.sin(2 * t), 3.0)
    ps, pf = calibrate(slow).build_profile(), calibrate(fast).build_profile()
    assert pf.measured.average_speed_mps > 2 * ps.measured.average_speed_mps
    assert pf.measured.movement_range_x_m > 2 * ps.measured.movement_range_x_m


def test_calibration_rejects_poor_tracking():
    mostly_gone = lambda t: make_pose(0, 3) if (t % 1.0) < 0.2 else None
    with pytest.raises(CalibrationError):
        calibrate(mostly_gone).build_profile()


def test_calibration_with_nobody_fails_cleanly():
    with pytest.raises(CalibrationError):
        calibrate(lambda t: None).build_profile()


def test_no_walking_gives_warning_not_direction():
    prof = calibrate(lambda t: make_pose(0, 3)).build_profile()
    assert prof.measured.typical_heading_deg is None
    assert any("walking" in w.lower() for w in prof.calibration.warnings)


# ---------------------------------------------------------------------- profile --
def test_profile_roundtrip(tmp_path):
    prof = calibrate(scripted_worker).build_profile(StatedPreferences(preferred_approach_side="left"))
    path = save_profile(prof, tmp_path)
    assert path.exists()
    loaded = load_profile(path)
    assert loaded.to_dict() == prof.to_dict()
    raw = json.loads(path.read_text())
    assert set(raw) >= {"measured", "stated_preferences", "calibration", "profile_id"}
    keys = set()

    def collect(d):
        for k, v in d.items():
            keys.add(k.lower())
            if isinstance(v, dict):
                collect(v)
    collect(raw)
    assert not any(w in k for k in keys for w in ("face", "image", "name", "landmark"))


def test_profile_unknown_schema_rejected(tmp_path):
    bad = tmp_path / "worker-bad.json"
    bad.write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(ValueError):
        load_profile(bad)


def test_profile_tolerates_missing_and_extra_fields():
    p = WorkerProfile.from_dict({"schema_version": 1, "profile_id": "worker-x",
                                 "measured": {"arm_reach_m": 0.7, "future_field": 1}})
    assert p.measured.arm_reach_m == 0.7 and p.stated_preferences.preferred_clearance_m is None
