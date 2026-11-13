"""
robot.py - the planar robotic arm: kinematics, joint limits, gripper geometry
and inverse kinematics.

Configuration q = list of joint angles (radians).
  q[0]  angle of link 1 measured from the +x axis (the shoulder)
  q[i]  angle of link i+1 relative to link i (elbow, wrist, ...)
The arm works with any number of joints: the number of joints is simply
len(config.LINK_LENGTHS).

Most functions accept a single configuration (shape (N,)) or a batch of
configurations (shape (B, N)) so the collision checker can test many poses
in one vectorised call.
"""
import copy
from dataclasses import dataclass

import numpy as np

import config as cfg
from geometry import rotation


def wrap_angle(a):
    """Map an angle (or array of angles) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class HeldObject:
    """An object rigidly attached to the gripper.

    rel_pos   : object centre expressed in the tool frame
                (component along the gripper axis, component along the palm)
    rel_angle : object angle minus gripper angle
    These are measured at the moment of grasping, so attaching never moves
    ("teleports") the object.
    """
    half_size: tuple
    rel_pos: np.ndarray
    rel_angle: float
    origin: str = "A"   # station the object was picked from


class PlanarArm:
    def __init__(self, base=cfg.ROBOT_BASE, link_lengths=cfg.LINK_LENGTHS,
                 link_radii=cfg.LINK_RADII, joint_limits_deg=cfg.JOINT_LIMITS_DEG):
        self.base = np.asarray(base, float)
        self.lengths = np.asarray(link_lengths, float)
        self.radii = np.asarray(link_radii, float)
        limits = np.radians(np.asarray(joint_limits_deg, float))
        if not (len(self.lengths) == len(self.radii) == len(limits)):
            raise ValueError("LINK_LENGTHS, LINK_RADII and JOINT_LIMITS_DEG need one entry per joint")
        self.lower = limits[:, 0]
        self.upper = limits[:, 1]
        # gripper
        self.tcp_offset = cfg.TCP_OFFSET
        self.finger_length = cfg.FINGER_LENGTH
        self.finger_radius = cfg.FINGER_RADIUS
        self.palm_radius = cfg.PALM_RADIUS
        self.finger_open = cfg.FINGER_OPEN
        self.finger_closed = cfg.OBJECT_SIZE[0] / 2 + cfg.FINGER_RADIUS + cfg.FINGER_CLEARANCE

    # ------------------------------------------------------------------ basics
    @property
    def n_joints(self):
        return len(self.lengths)

    def copy(self):
        return copy.deepcopy(self)

    def set_link_length(self, index, length):
        self.lengths[index] = float(length)

    def within_limits(self, q):
        q = np.asarray(q, float)
        return np.all((q >= self.lower - 1e-9) & (q <= self.upper + 1e-9), axis=-1)

    def clip(self, q):
        return np.clip(q, self.lower, self.upper)

    def sample(self, rng, count=None):
        """Uniform random configuration(s) inside the joint limits."""
        shape = (self.n_joints,) if count is None else (count, self.n_joints)
        return rng.uniform(self.lower, self.upper, size=shape)

    # ------------------------------------------------------ forward kinematics
    def forward_kinematics(self, q):
        """Positions of the shoulder, every joint and the flange (end of last link).

        Each link adds  L_i * (cos(theta_i), sin(theta_i)),  where theta_i is the
        sum of the joint angles up to joint i (the link's absolute angle).
        q (N,) -> (N+1, 2);  q (B, N) -> (B, N+1, 2)
        """
        q = np.asarray(q, float)
        single = q.ndim == 1
        Q = np.atleast_2d(q)
        theta = np.cumsum(Q, axis=1)
        steps = np.stack([np.cos(theta), np.sin(theta)], axis=-1) * self.lengths[None, :, None]
        pts = np.empty((Q.shape[0], self.n_joints + 1, 2))
        pts[:, 0] = self.base
        pts[:, 1:] = self.base + np.cumsum(steps, axis=1)
        return pts[0] if single else pts

    def tool_frame(self, q):
        """(flange position, gripper axis u, palm direction n, gripper angle)."""
        q = np.asarray(q, float)
        pts = self.forward_kinematics(q)
        angle = np.sum(q, axis=-1)
        u = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        n = np.stack([-np.sin(angle), np.cos(angle)], axis=-1)
        return pts[..., -1, :], u, n, angle

    def tcp(self, q):
        """Tool centre point: the point between the fingers where objects are held."""
        flange, u, _, _ = self.tool_frame(q)
        return flange + self.tcp_offset * u

    def tcp_pose(self, q):
        flange, u, _, angle = self.tool_frame(q)
        return flange + self.tcp_offset * u, angle

    def gripper_segments(self, q, finger_offset):
        """Palm and two fingers as segments: arrays (..., 3, 2) of start / end points."""
        flange, u, n, _ = self.tool_frame(q)
        fo = finger_offset
        base_l, base_r = flange + fo * n, flange - fo * n
        starts = np.stack([base_l, base_l, base_r], axis=-2)
        ends = np.stack([base_r, base_l + self.finger_length * u, base_r + self.finger_length * u], axis=-2)
        return starts, ends

    def object_pose(self, q, held):
        """Centre and angle of a held object for configuration(s) q."""
        tcp, angle = self.tcp_pose(q)
        _, u, n, _ = self.tool_frame(q)
        center = tcp + held.rel_pos[0] * u + held.rel_pos[1] * n
        return center, angle + held.rel_angle

    @staticmethod
    def box_corners(center, angle, half_size):
        """Counter-clockwise corners of a rectangle: (..., 4, 2)."""
        hw, hh = half_size
        local = np.array([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
        center = np.asarray(center, float)
        angle = np.asarray(angle, float)
        c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
        x = c * local[:, 0] - s * local[:, 1]
        y = s * local[:, 0] + c * local[:, 1]
        return center[..., None, :] + np.stack([x, y], axis=-1)

    # ------------------------------------------------- collision representation
    def collision_parts(self, q, finger_offset, held=None):
        """Every robot part as a thick segment ("capsule").

        Returns (starts (B,S,2), ends (B,S,2), radii (S,)). A held object is
        added as its four edges with radius 0.
        """
        Q = np.atleast_2d(np.asarray(q, float))
        pts = self.forward_kinematics(Q)
        g_start, g_end = self.gripper_segments(Q, finger_offset)
        starts = [pts[:, :-1], g_start]
        ends = [pts[:, 1:], g_end]
        radii = [self.radii, [self.palm_radius, self.finger_radius, self.finger_radius]]
        if held is not None:
            center, angle = self.object_pose(Q, held)
            corners = self.box_corners(center, angle, held.half_size)
            starts.append(corners)
            ends.append(np.roll(corners, -1, axis=1))
            radii.append(np.zeros(4))
        return np.concatenate(starts, 1), np.concatenate(ends, 1), np.concatenate(radii)

    def part_names(self, held=None):
        names = [f"link {i + 1}" for i in range(self.n_joints)] + ["palm", "left finger", "right finger"]
        if held is not None:
            names += ["object"] * 4
        return names

    def self_collision_pairs(self, held=None):
        """Pairs of parts that must not touch.

        Skipped: parts of the same rigid body (the last link, the gripper and a
        held object never move relative to each other) and consecutive links,
        which always overlap at the joint they share.
        """
        last = self.n_joints - 1
        body = list(range(self.n_joints)) + [last] * 3 + ([last] * 4 if held is not None else [])
        is_link = [True] * self.n_joints + [False] * (len(body) - self.n_joints)
        pairs = [(i, j) for i in range(len(body)) for j in range(i + 1, len(body))
                 if body[i] != body[j] and not (is_link[i] and is_link[j] and abs(body[i] - body[j]) == 1)]
        return np.array(pairs, dtype=int).reshape(-1, 2)

    def reach_radii(self, finger_offset, held=None):
        """For each joint: distance to the farthest robot point beyond it.

        Rotating joint i by dtheta moves no point more than R_i * dtheta, which
        lets the collision checker choose a safe interpolation step.
        """
        tool = np.hypot(self.finger_length, finger_offset)
        if held is not None:
            tool = max(tool, self.tcp_offset + np.linalg.norm(held.rel_pos) + np.hypot(*held.half_size))
        return np.cumsum(self.lengths[::-1])[::-1] + tool

    # ------------------------------------------------------ inverse kinematics
    def jacobian(self, q):
        """3 x N matrix: how TCP x, y and gripper angle change per joint."""
        pts = self.forward_kinematics(q)
        p = self.tcp(q)
        r = p - pts[:-1]
        return np.vstack([-r[:, 1], r[:, 0], np.ones(self.n_joints)])

    def solve_ik(self, target, angle, q_seed, max_iter=200, tol=1e-5, damping=0.05, max_step=0.2):
        """Damped-least-squares IK from one seed. Returns (q, converged)."""
        q = self.clip(np.asarray(q_seed, float).copy())
        target = np.asarray(target, float)
        for _ in range(max_iter):
            p, a = self.tcp_pose(q)
            err = np.array([target[0] - p[0], target[1] - p[1], wrap_angle(angle - a)])
            if np.all(np.abs(err) < tol):
                return q, True
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), err)
            biggest = np.max(np.abs(dq))
            if biggest > max_step:
                dq *= max_step / biggest
            q = self.clip(q + dq)
        p, a = self.tcp_pose(q)
        ok = np.linalg.norm(target - p) < 1e-4 and abs(wrap_angle(angle - a)) < 1e-4
        return q, bool(ok)

    def ik_solutions(self, target, angle, rng, q_seed=None, restarts=cfg.IK_RESTARTS, dedupe_tol=0.05):
        """All distinct IK solutions found from several random seeds."""
        seeds = ([np.asarray(q_seed, float)] if q_seed is not None else []) + \
            [self.sample(rng) for _ in range(restarts)]
        solutions = []
        for seed in seeds:
            q, ok = self.solve_ik(target, angle, seed)
            if ok and self.within_limits(q) and all(np.max(np.abs(q - s)) > dedupe_tol for s in solutions):
                solutions.append(q)
        return solutions

    def cartesian_line(self, q_start, p_end, angle, step=cfg.CARTESIAN_STEP, max_jump=0.15):
        """Joint path that moves the TCP along a straight line, keeping the gripper angle.

        Each intermediate point is solved with IK seeded by the previous
        configuration. Returns an array (K, N) or None if IK fails or jumps.
        """
        p_start = self.tcp(q_start)
        p_end = np.asarray(p_end, float)
        n = max(1, int(np.ceil(np.linalg.norm(p_end - p_start) / step)))
        path = [np.asarray(q_start, float)]
        for k in range(1, n + 1):
            target = p_start + (p_end - p_start) * k / n
            q, ok = self.solve_ik(target, angle, path[-1], max_iter=60)
            if not ok or np.max(np.abs(q - path[-1])) > max_jump:
                return None
            path.append(q)
        return np.array(path)
