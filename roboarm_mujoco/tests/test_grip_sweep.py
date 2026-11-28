"""Grip sweep: one plan, executed with a too-weak and a firm grip."""
from roboarm_mujoco import grip_sweep


def test_weak_grip_drops_and_firm_grip_places(tmp_path):
    out = tmp_path / "sweep.csv"
    rows = grip_sweep.main(["--seeds", "1", "--grips", "2.5", "40", "--speeds", "1", "--out", str(out)])
    by_grip = {r["grip_force_n"]: r for r in rows}
    assert by_grip[2.5]["outcome"] == "dropped"           # 2 x 1.0 x 2.5 N = 5 N < 9.81 N box weight
    assert by_grip[40.0]["outcome"] == "placed"
    assert by_grip[40.0]["max_slip_finger_m"] < grip_sweep.CLEAN_SLIP
    # both runs executed the same plan
    assert by_grip[2.5]["joint_path_rad"] == by_grip[40.0]["joint_path_rad"]
    # the saved CSV reads back with the same outcomes
    again = grip_sweep.main(["--summarise", str(out)])
    assert [r["outcome"] for r in again] == [r["outcome"] for r in rows]
