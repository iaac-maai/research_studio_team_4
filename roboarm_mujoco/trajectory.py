"""trajectory.py - turn a joint-space path into one target per control step."""
import numpy as np


def resample(path, step):
    """Points along a piecewise-linear joint path, at most `step` rad apart
    (Euclidean joint-space distance). Always ends exactly at the last waypoint;
    the first waypoint is not included."""
    path = np.atleast_2d(np.asarray(path, float))
    out = []
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.array(out).reshape(-1, path.shape[1])


def drive(sim, path, speed, dt=None, settle_steps=0):
    """Send `path` to a RobotSim at `speed` rad/s, then hold the last pose.
    Returns the list of states (one per control step)."""
    from roboarm_mujoco import settings as S
    dt = dt or S.CONTROL_DT
    states = [sim.step(q) for q in resample(path, speed * dt)]
    states += [sim.step(path[-1]) for _ in range(settle_steps)]
    return states
