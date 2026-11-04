"""3D robosuite/MuJoCo scene for Cobuild: a Panda arm hands drywall panels to a worker.

Layout (world frame, metres):
    robot base (x < 0)  ->  table with panel stack  ->  worker (x = 0.5, facing +x)  ->  wall (x = 0.95)

The worker faces the wall, so their LEFT hand is on +y and their RIGHT hand on -y.
The robot hands panels over at one of two points behind the worker's hips.

The worker, wall, and panels are visual-only (no collisions). The carried
panel is a mocap body that is attached to the gripper or the worker's hand
kinematically — this scene illustrates the decisions made by the abstract
CobuildEnv; it is not a grasping or contact simulation.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

TABLE_SIZE = (0.5, 0.9, 0.05)
TABLE_TOP_Z = 0.8

STACK_XY = np.array([-0.08, 0.0])
PANEL_HALF = np.array([0.12, 0.03, 0.01])
STACK_COUNT = 5
PIECE_ON_STACK = np.array([*STACK_XY, TABLE_TOP_Z + PANEL_HALF[2] * (2 * STACK_COUNT + 1)])

HANDOVER = {0: np.array([0.28, 0.33, 1.05]),   # worker's left
            1: np.array([0.28, -0.33, 1.05])}  # worker's right

WORKER_X = 0.5
SHOULDER = {0: np.array([WORKER_X, 0.19, 1.42]), 1: np.array([WORKER_X, -0.19, 1.42])}
WALL_X = 0.95
SLOT_HALF = 0.14
SLOTS = [np.array([WALL_X - 0.03, y, z]) for z in (0.95, 1.27) for y in (0.32, 0.0, -0.32)]

SKIN = "0.85 0.68 0.55 1"
VEST = "0.95 0.55 0.1 1"
WAIT_RGBA = np.array([0.85, 0.15, 0.15, 1.0])
VEST_RGBA = np.array([0.95, 0.55, 0.1, 1.0])
FRAME_RGBA = np.array([0.55, 0.42, 0.28, 0.35])
PREP_RGBA = np.array([0.95, 0.7, 0.2, 0.5])
DRYWALL_RGBA = np.array([0.95, 0.82, 0.55, 1.0])


def _el(tag, **attrs):
    return ET.Element(tag, {k: str(v) for k, v in attrs.items()})


def _vis_geom(name, **attrs):
    return _el("geom", name=name, contype=0, conaffinity=0, group=1, **attrs)


def _fmt(v):
    return " ".join(f"{x:.4f}" for x in v)


def _mocap(name, pos, geom_attrs):
    body = _el("body", name=name, mocap="true", pos=_fmt(pos))
    body.append(_vis_geom(f"{name}_g", **geom_attrs))
    return body


def build_static_scene():
    """Worker torso/legs/head, wall, panel slots, panel stack, handover markers."""
    bodies = []
    world = _el("body", name="cobuild_static", pos="0 0 0")
    x = WORKER_X
    world.append(_vis_geom("worker_leg_l", type="capsule", size="0.06", fromto=f"{x} 0.1 0.05 {x} 0.1 0.88", rgba="0.2 0.25 0.35 1"))
    world.append(_vis_geom("worker_leg_r", type="capsule", size="0.06", fromto=f"{x} -0.1 0.05 {x} -0.1 0.88", rgba="0.2 0.25 0.35 1"))
    world.append(_vis_geom("worker_torso", type="capsule", size="0.14", fromto=f"{x} 0 0.95 {x} 0 1.35", rgba=VEST))
    world.append(_vis_geom("worker_head", type="sphere", size="0.1", pos=f"{x} 0 1.62", rgba=SKIN))
    world.append(_vis_geom("worker_helmet", type="ellipsoid", size="0.115 0.115 0.06", pos=f"{x} 0 1.69", rgba="0.98 0.85 0.1 1"))
    world.append(_vis_geom("wall", type="box", size="0.02 0.75 0.9", pos=f"{WALL_X + 0.02} 0 0.9", rgba="0.95 0.94 0.9 1"))
    for i, p in enumerate(SLOTS):
        world.append(_vis_geom(f"slot_{i}", type="box", size=f"0.008 {SLOT_HALF} {SLOT_HALF}", pos=_fmt(p), rgba=_fmt(FRAME_RGBA)))
    for i in range(STACK_COUNT):
        z = TABLE_TOP_Z + PANEL_HALF[2] * (2 * i + 1)
        world.append(_vis_geom(f"stack_{i}", type="box", size=_fmt(PANEL_HALF), pos=_fmt([*STACK_XY, z]), rgba=_fmt(DRYWALL_RGBA)))
    for side, p in HANDOVER.items():
        world.append(_vis_geom(f"handover_{side}", type="sphere", size="0.035", pos=_fmt(p), rgba="0.5 0.5 0.5 0.25"))
    bodies.append(world)

    # moving parts: the carried panel and the worker's arms/hands
    bodies.append(_mocap("piece", PIECE_ON_STACK, dict(type="box", size=_fmt(PANEL_HALF), rgba=_fmt(DRYWALL_RGBA))))
    for side in (0, 1):
        bodies.append(_mocap(f"arm_{side}", SHOULDER[side], dict(type="capsule", size="0.045 0.2", rgba=VEST)))
        bodies.append(_mocap(f"hand_{side}", SHOULDER[side], dict(type="sphere", size="0.05", rgba=SKIN)))
    return bodies


class CobuildHandover(ManipulationEnv):
    def __init__(self, robots="Panda", **kwargs):
        kwargs.setdefault("has_renderer", False)
        kwargs.setdefault("has_offscreen_renderer", False)
        kwargs.setdefault("use_camera_obs", False)
        kwargs.setdefault("control_freq", 20)
        kwargs.setdefault("horizon", 100_000)
        kwargs.setdefault("hard_reset", False)
        kwargs.setdefault("initialization_noise", None)
        super().__init__(robots=robots, **kwargs)

    def reward(self, action=None):
        return 0.0

    def _load_model(self):
        super()._load_model()
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(robot_model.base_xpos_offset["table"](TABLE_SIZE[0]))
        arena = TableArena(table_full_size=TABLE_SIZE, table_offset=(0, 0, TABLE_TOP_Z))
        arena.set_origin([0, 0, 0])
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[robot_model])
        for body in build_static_scene():
            self.model.worldbody.append(body)

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model._model
        # the arena's lights come from above, leaving the wall face dark; light from the camera too
        m.vis.headlight.ambient[:] = 0.25
        m.vis.headlight.diffuse[:] = 0.5
        self.mocap_ids = {name: m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)]
                          for name in ("piece", "arm_0", "arm_1", "hand_0", "hand_1")}
        self.geom_ids = {name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                         for name in ["worker_torso", "handover_0", "handover_1", "arm_0_g", "arm_1_g"]
                         + [f"slot_{i}" for i in range(len(SLOTS))]}

    # --- helpers used by the visualizer -------------------------------------------------

    @property
    def mj_model(self):
        return self.sim.model._model

    @property
    def mj_data(self):
        return self.sim.data._data

    def eef_pos(self):
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])

    def eef_mat(self):
        return np.array(self.sim.data.site_xmat[self.robots[0].eef_site_id["right"]]).reshape(3, 3)

    def set_mocap(self, name, pos, quat=None):
        i = self.mocap_ids[name]
        self.mj_data.mocap_pos[i] = pos
        if quat is not None:
            self.mj_data.mocap_quat[i] = quat

    def set_rgba(self, geom, rgba):
        self.mj_model.geom_rgba[self.geom_ids[geom]] = rgba

    def set_worker_arm(self, side, hand_pos):
        shoulder = SHOULDER[side]
        vec = hand_pos - shoulder
        length = max(np.linalg.norm(vec), 1e-3)
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, vec / length)
        self.set_mocap(f"arm_{side}", shoulder + vec / 2, quat)
        self.mj_model.geom_size[self.geom_ids[f"arm_{side}_g"]][1] = length / 2
        self.set_mocap(f"hand_{side}", hand_pos)
