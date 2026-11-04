"""Physics for the loose objects of the 3D scene (MuJoCo rigid bodies with gravity, friction and contact).

The environment decides WHO holds each tool (robot tray, gripper, worker's hand, belt). While somebody holds a tool it
simply follows that holder. When the worker puts a tool down (the cut tools at the end of the cut) it is released
into this small physics world: it falls under gravity, lands on the cutting cart, bounces a little and comes to rest
there. Nothing else is animated by hand: a released tool cannot float.

This world only knows the static set that can support a tool (floor, cart). The worker and the robot are not
simulated: they are kinematic and tools do not collide with them (see docs/MODEL.md §9).
"""
import mujoco
import numpy as np

import geometry as G

TOOL_BOX = (0.14, 0.03, 0.03)          # half sizes of the cut set (same as the drawn one)
SUBSTEPS = 10                           # 10 x 5 ms per drawn frame (1/20 s)


def _xml(names):
    bodies = ''.join(
        f'<body name="{n}" pos="0 0 -9"><freejoint name="j_{n}"/>'
        f'<geom name="g_{n}" type="box" size="{TOOL_BOX[0]} {TOOL_BOX[1]} {TOOL_BOX[2]}" mass="0.4" '
        f'contype="0" conaffinity="0" friction="0.9 0.02 0.002" solref="0.01 1" solimp="0.9 0.95 0.001"/></body>'
        for n in names)
    return f'''<mujoco><option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="12 8 0.1" pos="2.3 -1.5 0" friction="0.9 0.02 0.002"/>
    <geom name="cart_top" type="box" size="1.4 0.12 0.02" pos="{G.CART_XY[0]} {G.CART_XY[1]} 0.1" friction="0.9 0.02 0.002"/>
    {bodies}
  </worldbody></mujoco>'''


class PropWorld:
    def __init__(self, names):
        self.names = list(names)
        self.model = mujoco.MjModel.from_xml_string(_xml(self.names))
        self.data = mujoco.MjData(self.model)
        self.adr = {n: self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'j_{n}')]
                    for n in self.names}
        self.vadr = {n: self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'j_{n}')]
                     for n in self.names}
        self.gid = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'g_{n}') for n in self.names}
        self.active = set()
        mujoco.mj_forward(self.model, self.data)

    def release(self, name, pos, quat, vel=(0.0, 0.0, 0.0)):
        """Let go of a tool at this pose: from now on gravity and contact decide where it is."""
        a, v = self.adr[name], self.vadr[name]
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = quat
        self.data.qvel[v:v + 6] = (*vel, 0, 0, 0)
        self.model.geom_contype[self.gid[name]] = 1             # from now on it collides
        self.model.geom_conaffinity[self.gid[name]] = 1
        bid = self.model.geom_bodyid[self.gid[name]]            # MuJoCo also caches the filter per body
        self.model.body_contype[bid] = self.model.body_conaffinity[bid] = 1
        self.active.add(name)

    def step(self):
        if not self.active:
            return
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            for n in self.names:                                  # tools nobody has released yet stay parked
                if n not in self.active:
                    a, v = self.adr[n], self.vadr[n]
                    self.data.qpos[a:a + 3] = (0, 0, -9)
                    self.data.qvel[v:v + 6] = 0

    def pose(self, name):
        a = self.adr[name]
        return self.data.qpos[a:a + 3].copy(), self.data.qpos[a + 3:a + 7].copy()

    def speed(self, name):
        v = self.vadr[name]
        return float(np.linalg.norm(self.data.qvel[v:v + 6]))

    def resting(self, name):
        """True once the tool has stopped moving."""
        return name in self.active and self.speed(name) < 0.05
