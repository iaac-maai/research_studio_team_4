"""
planning.py - MuJoCo collision checking for roboarm's motion planner.

roboarm/planner.py (RRT-Connect + shortcutting) only asks two questions:
    is_valid(q)        is this pose collision-free?
    edge_valid(qa, qb) is the straight joint-space motion collision-free?
MujocoChecker answers them with MuJoCo's own collision detection on a
separate copy of the scene (the live simulation is never touched):
set the joint angles, run kinematics, list the contacts.

Clearance margins work through MuJoCo's geom `margin`: a robot geom with
margin m reports a "contact" whenever it comes closer than m to anything, so
"no contacts" means "at least m clearance" (SAFETY_MARGIN near the tables,
PLANNING_MARGIN in free space, both from roboarm/config.py). The margin is
written into the XML before compiling, because MuJoCo's broad phase does not
see margins changed on a compiled model.

Planning is kinematic (no physics): the held box is placed rigidly in the
gripper, as in roboarm. The physics simulation then has to actually carry it.
"""
import mujoco
import numpy as np

from roboarm_mujoco import paths  # noqa: F401
from roboarm_mujoco import scene
from roboarm_mujoco.robot_api import GeomClasses, contact_pairs
import config as rcfg


class MujocoChecker:
    """Pose / motion validity for one planning situation.

    layout         : the scene's layout rows (RobotSim.layout)
    robot          : roboarm PlanarArm (joint limits, reach radii, held-box pose)
    margin         : required clearance (m)
    held           : roboarm HeldObject if the box is carried (None = box rests)
    object_pose    : (centre (2,), angle) of a resting box; None = no box in the scene
    gripper_closed : finger position used for the check
    ignore_table   : table geom name the HELD box may touch (lifting it off that table)
    """

    def __init__(self, layout, robot, margin, held=None, object_pose=None, gripper_closed=False,
                 ignore_table=None):
        xml = scene.build_xml(layout, robot_margin=margin, object_margin=margin if held is not None else 0.0)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.robot = robot
        self.held = held
        self.margin = margin
        m = self.model
        self.g = GeomClasses(m)
        self.ignore = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, ignore_table) if ignore_table else -1
        # A held box moves rigidly with link 3 and the gripper; it can only hit links 1-2.
        self._upper_links = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in scene.LINK_GEOMS[:2]}

        def adr(name):
            return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])

        self._arm = np.array([adr(j) for j in scene.ARM_JOINTS])
        self._obj = np.array([adr(j) for j in scene.OBJECT_JOINTS])
        fingers = np.array([adr(j) for j in scene.FINGER_JOINTS])
        offset = robot.finger_closed if gripper_closed else robot.finger_open
        self.finger_offset = offset
        self.data.qpos[fingers] = offset - rcfg.FINGER_OPEN
        if held is None:
            if object_pose is None:
                self.data.qpos[self._obj] = [0.0, 50.0, 0.0]            # parked far above the scene: no box
            else:
                (cx, cz), ang = object_pose
                self.data.qpos[self._obj] = [cx, cz, ang]
        self.reach = robot.reach_radii(offset, held)
        self.checks = 0

    # ------------------------------------------------------------ queries
    def _bad_pairs(self, q):
        d = self.data
        d.qpos[self._arm] = q
        if self.held is not None:
            c, a = self.robot.object_pose(np.asarray(q, float), self.held)
            d.qpos[self._obj] = [c[0], c[1], a]
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        g, out = self.g, []
        for a, b in contact_pairs(d):
            if b in g.robot and a not in g.robot:
                a, b = b, a                                   # robot geom first
            if a in g.robot:
                if b in g.robot or b in g.environment:
                    out.append((a, b))
                elif b == g.object:
                    # a resting box is an obstacle; a held box only for links 1-2
                    if self.held is None or a in self._upper_links:
                        out.append((a, b))
            elif g.object in (a, b) and self.held is not None:
                other = b if a == g.object else a
                if other in g.environment and other != self.ignore:
                    out.append((a, b))
        return out

    def is_valid(self, q):
        self.checks += 1
        q = np.asarray(q, float)
        if not self.robot.within_limits(q):
            return False
        return not self._bad_pairs(q)

    def edge_valid(self, qa, qb):
        """Check poses along the straight joint motion so that no robot point
        moves more than config.COLLISION_STEP between two checks."""
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        travel = float(np.sum(np.abs(qb - qa) * self.reach))
        n = max(1, int(np.ceil(travel / rcfg.COLLISION_STEP)))
        for k in range(1, n + 1):
            if not self.is_valid(qa + (qb - qa) * k / n):
                return False
        return True

    def path_valid(self, path):
        path = np.atleast_2d(path)
        return self.is_valid(path[0]) and all(self.edge_valid(a, b) for a, b in zip(path[:-1], path[1:]))

    def report(self, q):
        """Names of the geom pairs that make q invalid."""
        return [f"{self.model.geom(a).name} / {self.model.geom(b).name}" for a, b in self._bad_pairs(q)]
