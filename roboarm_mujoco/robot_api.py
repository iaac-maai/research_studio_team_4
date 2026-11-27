"""
robot_api.py - RobotSim: the physics robot behind the team's agreed interface.

    sim = RobotSim()                      # default layout from roboarm/config.py
    state = sim.reset(layout_csv="roboarm_mujoco/environments/some.csv", seed=0)
    state = sim.step(np.array([1.6, -0.9, -2.2]))   # absolute joint targets (rad), 0.02 s
    sim.set_gripper(True)
    sim.get_state()  -> {"q", "tcp", "box_pos", "box_angle", "gripper_closed", "holding"}
    sim.in_collision() -> bool

Physics: gravity on; position-controlled joint and finger motors; the box is
held only by friction between the finger pads and the box (no weld). The box
moves in the arm's plane (two slide joints + one hinge), so it can slide,
tip over or fall, but cannot leave the plane.
"""
import mujoco
import numpy as np

from roboarm_mujoco import paths  # noqa: F401
from roboarm_mujoco import scene
from roboarm_mujoco import settings as S
import config as rcfg

STATE_KEYS = ("q", "tcp", "box_pos", "box_angle", "gripper_closed", "holding")


class GeomClasses:
    """Which geoms belong to the robot, the environment and the box."""

    def __init__(self, model):
        def gid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

        self.robot = {gid(n) for n in scene.ROBOT_GEOMS}
        self.links = {gid(n) for n in scene.LINK_GEOMS}
        self.fingers = {gid(n) for n in scene.FINGER_GEOMS}
        self.object = gid(scene.OBJECT_GEOM)
        self.tables, self.obstacles = set(), set()
        for i in range(model.ngeom):
            name = model.geom(i).name
            if name.startswith(scene.TABLE_PREFIX):
                self.tables.add(i)
            elif name.startswith(scene.OBSTACLE_PREFIX):
                self.obstacles.add(i)
        self.floor = {gid("floor"), gid("pedestal")}       # the robot's mount counts like the floor
        self.environment = self.tables | self.obstacles | self.floor


def contact_pairs(data):
    """(geom1, geom2) of every active contact."""
    n = data.ncon
    if n == 0:
        return []
    geom = data.contact.geom[:n]
    return [(int(a), int(b)) for a, b in geom]


class RobotSim:
    def __init__(self, layout_csv=None, seed=None, grip_force=S.GRIP_FORCE):
        self.layout = None
        self.model = None
        self.data = None
        self.grip_force = float(grip_force)
        self.reset(layout_csv, seed)

    @classmethod
    def from_layout(cls, rows, seed=None, grip_force=S.GRIP_FORCE):
        """A RobotSim for a layout given as rows (dicts) instead of a CSV file."""
        sim = cls.__new__(cls)
        sim.model = sim.data = None
        sim.grip_force = float(grip_force)
        sim._build(scene.validate_layout(rows))
        sim.reset(seed=seed)
        return sim

    # ------------------------------------------------------------ setup
    def _build(self, layout):
        self.layout = layout
        self.xml = scene.build_xml(layout)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model

        def jadr(name):
            return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])

        def aid(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        self._arm_q = np.array([jadr(j) for j in scene.ARM_JOINTS])
        self._finger_q = np.array([jadr(j) for j in scene.FINGER_JOINTS])
        self._obj_q = np.array([jadr(j) for j in scene.OBJECT_JOINTS])
        self._pos_act = np.array([aid(f"{j}_pos") for j in scene.ARM_JOINTS])
        self._vel_act = np.array([aid(f"{j}_vel") for j in scene.ARM_JOINTS])
        self._finger_act = np.array([aid(f"{j}_pos") for j in scene.FINGER_JOINTS])
        self._tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.geoms = GeomClasses(m)
        self.lower = m.jnt_range[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in scene.ARM_JOINTS], 0]
        self.upper = m.jnt_range[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in scene.ARM_JOINTS], 1]
        self.finger_closed_ctrl = float(m.actuator_ctrlrange[self._finger_act[0], 0])
        self.substeps = int(round(S.CONTROL_DT / m.opt.timestep))
        self.set_grip_force(self.grip_force)

    def set_grip_force(self, force):
        """Maximum squeezing force of each finger (N). Lower = weaker grip."""
        self.grip_force = float(force)
        self.model.actuator_forcerange[self._finger_act] = [-self.grip_force, self.grip_force]

    @property
    def home(self):
        return np.radians(rcfg.HOME_CONFIG_DEG)

    # ------------------------------------------------------------ interface
    def reset(self, layout_csv=None, seed=None):
        """Arm at home, gripper open, box at its layout position.

        layout_csv: path of an environment CSV. None keeps the current layout
        (the default layout from roboarm/config.py on the first call).
        seed: stored as self.rng for anything random (e.g. the planner);
        the physics itself is deterministic.
        """
        if layout_csv is not None:
            layout = scene.load_csv_layout(layout_csv)
        else:
            layout = self.layout if self.layout is not None else scene.default_layout()
        if self.model is None or layout != self.layout:
            self._build(layout)
        self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        obj = scene.object_row(self.layout)
        d.qpos[self._arm_q] = self.home
        d.qpos[self._obj_q] = [obj["x"], obj["y"], 0.0]
        d.ctrl[self._pos_act] = self.home
        d.ctrl[self._vel_act] = 0.0
        self._target = self.home.copy()
        self.set_gripper(False)
        mujoco.mj_forward(m, d)
        return self.get_state()

    def step(self, joint_targets):
        """One control step (0.02 s) towards absolute joint angles (rad), shape (3,).

        Targets are clipped to the joint limits. The velocity servo gets the
        speed implied by the change of target, so a smoothly moving target is
        followed without lag.
        """
        target = np.clip(np.asarray(joint_targets, float).reshape(3), self.lower, self.upper)
        velocity = np.clip((target - self._target) / S.CONTROL_DT, -S.MAX_JOINT_SPEED, S.MAX_JOINT_SPEED)
        self._target = target
        d = self.data
        d.ctrl[self._pos_act] = target
        d.ctrl[self._vel_act] = velocity
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, d)
        return self.get_state()

    def set_gripper(self, closed):
        """closed=True squeezes the fingers together (force capped at grip_force)."""
        self._gripper_closed = bool(closed)
        self.data.ctrl[self._finger_act] = self.finger_closed_ctrl if closed else 0.0

    def get_state(self):
        d = self.data
        tcp = d.site_xpos[self._tcp_site]
        obj = d.qpos[self._obj_q]
        return {
            "q": d.qpos[self._arm_q].copy(),
            "tcp": np.array([tcp[0], tcp[2]]),
            "box_pos": np.array([obj[0], obj[1]]),
            "box_angle": float((obj[2] + np.pi) % (2 * np.pi) - np.pi),   # wrapped to [-pi, pi)
            "gripper_closed": self._gripper_closed,
            "holding": self.holding(),
        }

    def in_collision(self):
        """True if the arm or gripper touches an obstacle, a table, the floor
        (or pedestal) or itself, or the box touches an obstacle. The box on a
        table and the fingers on the box are not collisions."""
        return bool(self.collisions())

    # ------------------------------------------------------------ helpers
    @property
    def target(self):
        """Last joint targets sent with step()."""
        return self._target.copy()

    @property
    def time(self):
        return float(self.data.time)

    def finger_offsets(self):
        """Distance of each finger from the gripper axis (m)."""
        return rcfg.FINGER_OPEN + self.data.qpos[self._finger_q]

    def collisions(self):
        """Names of the geom pairs that count as collisions right now."""
        g = self.geoms
        out = []
        for a, b in contact_pairs(self.data):
            ra, rb = a in g.robot, b in g.robot
            hit = (ra and rb) or (ra and b in g.environment) or (rb and a in g.environment) or \
                  (a == g.object and b in g.obstacles) or (b == g.object and a in g.obstacles)
            if hit:
                out.append((self.model.geom(a).name, self.model.geom(b).name))
        return out

    def holding(self):
        """Gripper closed and both fingers touching the box."""
        if not self._gripper_closed:
            return False
        g = self.geoms
        touching = set()
        for a, b in contact_pairs(self.data):
            if a == g.object and b in g.fingers:
                touching.add(b)
            elif b == g.object and a in g.fingers:
                touching.add(a)
        return touching == g.fingers
