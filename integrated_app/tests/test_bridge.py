"""Tests for the camera-profile -> simulation-profile bridge and the simulation round trip."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge  # noqa: E402
from worker_profile import StatedPreferences, WorkerProfile  # noqa: E402


def camera_profile(**measured):
    p = WorkerProfile.new_anonymous()
    for k, v in measured.items():
        setattr(p.measured, k, v)
    return p


def test_measured_values_mapped_with_provenance():
    p = camera_profile(height_m=1.62, arm_length_m=0.52, walking_speed_mps=1.2)
    p.stated_preferences = StatedPreferences(preferred_approach_side="left")
    sp = bridge.to_sim_profile(p)
    assert sp.values["height_m"] == 1.62 and sp.provenance["height_m"] == "measured"
    assert sp.values["reach_m"] == pytest.approx(0.52 + bridge.HAND_ALLOWANCE_M)
    assert sp.values["pace"] == pytest.approx(1.0)          # walking at the reference speed
    assert sp.values["handedness"] == "left" and sp.provenance["handedness"] == "stated"
    # Not measurable by camera: must be flagged as default, never "measured".
    assert sp.provenance["lift_capacity_kg"] == "default" and sp.provenance["fatigue_rate"] == "default"


def test_missing_measurements_fall_back_to_default_and_warn():
    sp = bridge.to_sim_profile(camera_profile())
    assert sp.values["height_m"] == bridge.GENERIC["height_m"] and sp.provenance["height_m"] == "default"
    assert sp.provenance["pace"] == "default" and sp.provenance["reach_m"] == "default"
    assert len(sp.warnings) >= 3


def test_values_clipped_to_simulation_range():
    sp = bridge.to_sim_profile(camera_profile(height_m=2.4, arm_length_m=1.2, walking_speed_mps=0.1))
    assert sp.values["height_m"] == 2.0
    assert sp.values["reach_m"] == 0.80
    assert sp.values["pace"] == 1.30


def test_manual_override_wins_and_is_labelled():
    sp = bridge.to_sim_profile(camera_profile(walking_speed_mps=0.9),
                               {"pace": 0.85, "lift_capacity_kg": 22.0, "ignored": 1})
    assert sp.values["pace"] == 0.85 and sp.provenance["pace"] == "manual"
    assert sp.values["lift_capacity_kg"] == 22.0 and sp.provenance["lift_capacity_kg"] == "manual"
    assert "ignored" not in sp.values


def test_stated_clearance_is_not_a_simulation_input():
    p = camera_profile()
    p.stated_preferences = StatedPreferences(preferred_clearance_m=0.1)
    assert "clearance" not in " ".join(bridge.to_sim_profile(p).values)


def test_simulation_roundtrip_through_subprocess():
    import sim_client
    sp = bridge.to_sim_profile(camera_profile(height_m=1.6, arm_length_m=0.5, walking_speed_mps=1.0))
    res = sim_client.run_evaluation(sp.sim_json(), runs=3)
    assert set(res["strategies"]) == {"no_robot", "A_reactive", "B_proactive_generic", "C_cobuild_rule"}
    s = res["strategies"]
    assert s["B_proactive_generic"]["minutes"] < s["no_robot"]["minutes"]     # robot helps
    assert res["profile"]["height_m"] == 1.6
    assert sp.sim_json()["handedness"] in ("left", "right")
