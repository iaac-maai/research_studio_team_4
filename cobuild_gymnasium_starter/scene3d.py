"""Schematic 3D scene of the drywall wall (MuJoCo, visual only: no physics, no collisions).

Layout in metres. The wall lies in the plane y = 0 and the room is at y < 0; the worker faces the wall
(+y). Seen from the worker's back, their RIGHT hand is on +x and their LEFT hand on -x.

    wall: x 0..4.5, z 0..2.44, 2 columns x 2 rows of boards (the right-hand column is trimmed to 2.06 m)
    left of the wall: the board cart (a trestle about 0.7 m high) with the board stack lying flat on it
    right of the wall: the cutting cart (the board stands on it)
    in front: the robot (mobile base + lift column + two-link arm + suction frame) and the worker

Everything that moves is a "mocap" body whose pose is written every frame by Scene.apply(); the worker
mannequin is built from the profile's height, so different profiles are different bodies.
The arm is a stand-in: a real robot able to lift a 34 kg board to 2.4 m needs a far higher payload than
a Franka Panda, so the robot is drawn schematically (see docs/RESEARCH.md §6).
"""
import numpy as np
import mujoco

import config as C
import ergonomics as E

WALL_W, WALL_H = 4.5, C.WALL_TOP_M
BOARD_H, BOARD_T = 1.22, 0.016
FULL_W, CUT_W = 2.44, 2.06
from geometry import CART_XY, HOME_XY, STACK_XY, STAGE_X, STAGE_Y      # the floor plan lives in geometry.py

STACK_TOP = 0.7          # boards are delivered on a trestle cart so they can be picked up without bending to the floor

BOARD_RGBA = (0.93, 0.84, 0.62, 1.0)
VEST_ON, VEST_WAIT = (0.95, 0.55, 0.10, 1.0), (0.85, 0.15, 0.15, 1.0)
ROBOT_RGBA = (0.92, 0.41, 0.20, 1.0)
DARK = (0.18, 0.18, 0.2, 1.0)
SKIN = (0.85, 0.68, 0.55, 1.0)

# anthropometric fractions of stature (Drillis & Contini)
HIP, SHOULDER, HEAD = 0.53, 0.82, 0.935
UPPER_ARM, FOREARM = 0.186, 0.254


def board_width(b, final=True):
    return CUT_W if (E.needs_cut(b) and final) else FULL_W


def board_wall_pose(b):
    col = b % 2
    x0 = 0.0 if col == 0 else FULL_W
    cx = x0 + board_width(b) / 2
    cz = 1.83 if E.is_upper(b) else 0.61
    return np.array([cx, -BOARD_T / 2, cz])


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def q_axis(axis, ang):
    s = np.sin(ang / 2)
    return np.array([np.cos(ang / 2), *(np.asarray(axis, float) * s)])


Q_ID = np.array([1.0, 0, 0, 0])
Q_FLAT = q_axis([1, 0, 0], np.pi / 2)   # board lying flat (thickness along z)


TOOL_NAMES = ('K0', 'K1', 'G0', 'G1')


def _geom(name, kind, size, rgba, pos=None):
    p = f' pos="{pos[0]} {pos[1]} {pos[2]}"' if pos is not None else ''
    return (f'<geom name="{name}" type="{kind}" size="{" ".join(str(s) for s in size)}" '
            f'rgba="{" ".join(str(c) for c in rgba)}" contype="0" conaffinity="0"{p}/>')


def _mocap(name, geoms, pos="0 0 -5"):
    return f'<body name="{name}" mocap="true" pos="{pos}">{"".join(geoms)}</body>'


def build_xml(profile):
    H = profile.height_m
    studs = []
    n_studs = int(WALL_W / C.STUD_SPACING_M) + 1
    for i in range(n_studs):
        studs.append(_geom(f'stud{i}', 'box', (0.02, 0.045, WALL_H / 2), (0.62, 0.48, 0.3, 1),
                           pos=(i * C.STUD_SPACING_M, 0.06, WALL_H / 2)))
    for z, nm in ((0.02, 'plate_lo'), (WALL_H - 0.02, 'plate_hi')):
        studs.append(_geom(nm, 'box', (WALL_W / 2, 0.045, 0.02), (0.62, 0.48, 0.3, 1), pos=(WALL_W / 2, 0.06, z)))
    ghosts = []
    for b in range(C.N_BOARDS):
        c = board_wall_pose(b)
        ghosts.append(_geom(f'ghost{b}', 'box', (board_width(b) / 2, 0.004, BOARD_H / 2), (0.55, 0.55, 0.6, 0.12),
                            pos=(c[0], 0.012, c[2])))
    static = [
        _geom('floor', 'plane', (12, 8, 0.1), (0.74, 0.73, 0.70, 1), pos=(2.3, -1.5, 0)),
        _geom('pallet', 'box', (1.3, 0.65, 0.04), (0.55, 0.42, 0.28, 1), pos=(STACK_XY[0], STACK_XY[1], STACK_TOP - 0.04)),
        *[_geom(f'pallet_leg{i}', 'box', (0.04, 0.04, (STACK_TOP - 0.08) / 2), DARK,
                pos=(STACK_XY[0] + sx * 1.2, STACK_XY[1] + sy * 0.55, (STACK_TOP - 0.08) / 2))
          for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])],
        _geom('cart_top', 'box', (1.4, 0.12, 0.02), (0.35, 0.36, 0.4, 1), pos=(CART_XY[0], CART_XY[1], 0.1)),
        _geom('cart_leg1', 'box', (0.03, 0.1, 0.05), DARK, pos=(CART_XY[0] - 1.2, CART_XY[1], 0.04)),
        _geom('cart_leg2', 'box', (0.03, 0.1, 0.05), DARK, pos=(CART_XY[0] + 1.2, CART_XY[1], 0.04)),
        _geom('tooltable', 'box', (0.6, 0.3, 0.02), (0.5, 0.5, 0.52, 1), pos=(HOME_XY[0] + 1.6, HOME_XY[1] - 0.3, 0.78)),
        _geom('tooltable_l1', 'box', (0.02, 0.02, 0.39), DARK, pos=(HOME_XY[0] + 1.1, HOME_XY[1] - 0.3, 0.39)),
        _geom('tooltable_l2', 'box', (0.02, 0.02, 0.39), DARK, pos=(HOME_XY[0] + 2.1, HOME_XY[1] - 0.3, 0.39)),
        _geom('backwall', 'box', (WALL_W / 2 + 0.2, 0.02, WALL_H / 2), (0.75, 0.74, 0.72, 1), pos=(WALL_W / 2, 0.13, WALL_H / 2)),
        *studs, *ghosts,
    ]
    bodies = []
    for b in range(C.N_BOARDS):
        bodies.append(_mocap(f'board{b}', [_geom(f'board{b}_g', 'box', (FULL_W / 2, BOARD_T / 2, BOARD_H / 2), BOARD_RGBA)]))
    bodies.append(_mocap('stool', [_geom('stool_g', 'box', (0.25, 0.2, 0.1), (0.4, 0.42, 0.5, 1))]))
    r = 0.045 * H / 1.75
    for nm, rgba, size in [('torso', VEST_ON, (0.085 * H, 0.1)), ('leg_l', (0.2, 0.25, 0.35, 1), (0.04 * H, 0.1)),
                           ('leg_r', (0.2, 0.25, 0.35, 1), (0.04 * H, 0.1)), ('arm_l_up', VEST_ON, (r, 0.1)),
                           ('arm_l_lo', VEST_ON, (r * 0.9, 0.1)), ('arm_r_up', VEST_ON, (r, 0.1)),
                           ('arm_r_lo', VEST_ON, (r * 0.9, 0.1))]:
        bodies.append(_mocap(nm, [_geom(nm + '_g', 'capsule', size, rgba)]))
    bodies.append(_mocap('head', [_geom('head_g', 'sphere', (0.062 * H,), SKIN)]))
    bodies.append(_mocap('helmet', [_geom('helmet_g', 'ellipsoid', (0.07 * H, 0.07 * H, 0.04 * H), (0.98, 0.85, 0.1, 1))]))
    for s in ('l', 'r'):
        bodies.append(_mocap(f'hand_{s}', [_geom(f'hand_{s}_g', 'sphere', (0.035 * H / 1.75,), SKIN)]))
    bodies.append(_mocap('robot_base', [
        _geom('rb_body', 'box', (0.5, 0.32, 0.14), ROBOT_RGBA, pos=(0, 0, 0.22)),
        _geom('rb_column', 'box', (0.07, 0.07, 1.45), DARK, pos=(0, -0.12, 1.55)),
        _geom('rb_tray', 'box', (0.42, 0.26, 0.012), (0.3, 0.3, 0.32, 1), pos=(0, 0, 0.372)),
        *[_geom(f'rb_wheel{i}', 'cylinder', (0.1, 0.04), DARK, pos=(sx * 0.4, sy * 0.3, 0.1))
          for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])]]))
    bodies.append(_mocap('carriage', [_geom('carriage_g', 'box', (0.12, 0.12, 0.1), DARK)]))
    bodies.append(_mocap('link1', [_geom('link1_g', 'capsule', (0.055, 0.4), ROBOT_RGBA)]))
    bodies.append(_mocap('link2', [_geom('link2_g', 'capsule', (0.045, 0.4), ROBOT_RGBA)]))
    bodies.append(_mocap('frame', [_geom('frame_g', 'box', (0.4, 0.03, 0.15), DARK),
                                   *[_geom(f'cup{i}', 'cylinder', (0.045, 0.02), (0.1, 0.1, 0.1, 1),
                                           pos=(sx * 0.3, -0.03, sz * 0.1)) for i, (sx, sz) in
                                     enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])]]))
    bodies.append(_mocap('gripper', [_geom('gripper_g', 'box', (0.07, 0.05, 0.05), DARK)]))
    for j in (0, 1):                # two cut sets and two screw guns: the robot's and the worker's swap roles
        bodies.append(_mocap(f'tool_K{j}', [_geom(f'tK{j}_g', 'box', (0.14, 0.03, 0.03), (0.95, 0.85, 0.1, 1)),
                                           _geom(f'tK{j}_t', 'cylinder', (0.05, 0.02), (0.9, 0.9, 0.9, 1), pos=(0.1, 0, 0.04))]))
        bodies.append(_mocap(f'tool_G{j}', [_geom(f'tG{j}_g', 'box', (0.1, 0.035, 0.04), (0.75, 0.15, 0.15, 1)),
                                           _geom(f'tG{j}_h', 'box', (0.03, 0.03, 0.08), DARK, pos=(-0.04, 0, -0.07))]))
    bodies.append(_mocap('marker', [_geom('marker_g', 'sphere', (0.06,), (0.2, 0.8, 0.3, 0.55))]))
    return f'''<mujoco model="cobuild_wall">
  <visual><global offwidth="1600" offheight="900"/><quality shadowsize="2048"/>
          <headlight ambient="0.42 0.42 0.42" diffuse="0.3 0.3 0.3" specular="0.0 0.0 0.0"/></visual>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.92 0.94 0.97" rgb2="0.72 0.76 0.84" width="64" height="64"/></asset>
  <worldbody>
    <light directional="true" pos="2 -4 6" dir="0.2 0.5 -1" diffuse="0.4 0.4 0.4" specular="0 0 0" castshadow="false"/>
    {''.join(static)}
    {''.join(bodies)}
  </worldbody>
</mujoco>'''


class Scene:
    def __init__(self, profile, width=800, height=600):
        self.P = profile
        self.model = mujoco.MjModel.from_xml_string(build_xml(profile))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.mocap = {}
        for i in range(m.nbody):
            if m.body_mocapid[i] >= 0:
                self.mocap[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)] = m.body_mocapid[i]
        self.gid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(m.ngeom)
                    if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)}
        self.renderer = mujoco.Renderer(m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [2.0, -1.5, 1.4]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 9.2, 66, -17

    # ---------------------------------------------------------------- primitives
    def put(self, name, pos, quat=None):
        i = self.mocap[name]
        self.data.mocap_pos[i] = pos
        self.data.mocap_quat[i] = Q_ID if quat is None else quat

    def hide(self, name):
        self.put(name, (0, 0, -5))

    def capsule(self, name, p0, p1):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        v = p1 - p0
        length = max(np.linalg.norm(v), 1e-3)
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, v / length)
        self.put(name, (p0 + p1) / 2, quat)
        self.model.geom_size[self.gid[name + '_g']][1] = length / 2

    def color(self, geom, rgba):
        self.model.geom_rgba[self.gid[geom]] = rgba

    # ---------------------------------------------------------------- the worker
    @staticmethod
    def _elbow(S, T, l1, l2, hint):
        v = T - S
        d = float(np.linalg.norm(v))
        d = max(min(d, l1 + l2 - 1e-3), 1e-3)
        u = v / max(np.linalg.norm(v), 1e-9)
        T = S + u * d
        a = (l1 ** 2 - l2 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(max(l1 ** 2 - a ** 2, 0.0))
        n = np.asarray(hint, float) - u * np.dot(hint, u)
        n = n / max(np.linalg.norm(n), 1e-9)
        return S + u * a + n * h, T

    def worker(self, w):
        H = self.P.height_m
        px, py, pz = w['pos']
        lean = w['lean']
        hip = np.array([px, py, pz + HIP * H])
        fwd = np.array([0, np.sin(lean), np.cos(lean)])
        sh_c = hip + fwd * (SHOULDER - HIP) * H
        neck = sh_c + fwd * 0.05 * H
        head = neck + fwd * 0.075 * H
        sw = 0.125 * H
        step = w['walk']
        for side, sx in (('l', -1), ('r', 1)):
            # the leg capsule has rounded ends: lift the foot target by its radius so the shoe rests ON the floor
            foot = np.array([px + sx * 0.05 * H, py + sx * 0.14 * np.sin(step) * (1 if w['moving'] else 0),
                             pz + 0.04 * H])
            self.capsule(f'leg_{side}', hip + [sx * 0.05 * H, 0, 0], foot)
            S = sh_c + [sx * sw, 0, 0]
            T = np.asarray(w['hand_' + side], float)
            E_, T = self._elbow(S, T, UPPER_ARM * H, FOREARM * H, (sx * 0.5, -0.3, -1.0))
            self.capsule(f'arm_{side}_up', S, E_)
            self.capsule(f'arm_{side}_lo', E_, T)
            self.put(f'hand_{side}', T)
        self.capsule('torso', hip, neck)
        self.put('head', head)
        self.put('helmet', head + [0, 0, 0.055 * H])
        vest = VEST_WAIT if w['waiting'] else VEST_ON
        for g in ('torso_g', 'arm_l_up_g', 'arm_l_lo_g', 'arm_r_up_g', 'arm_r_lo_g'):
            self.color(g, vest)
        if w['stool']:
            self.put('stool', (px, py, 0.1))
        else:
            self.hide('stool')

    # ---------------------------------------------------------------- the robot
    def robot(self, r):
        bx, by = r['base']
        yaw = r['yaw']
        quat = q_axis([0, 0, 1], yaw - np.pi / 2)   # local +y is the robot's front
        self.put('robot_base', (bx, by, 0.0), quat)
        f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        ee = np.asarray(r['ee'], float)
        zc = float(np.clip(r['carriage_z'], 0.55, 2.7))
        col = np.array([bx, by, zc]) + f * -0.12 * 0
        S = np.array([bx, by, zc]) + f * 0.0
        self.put('carriage', S)
        L1, L2 = 0.95, 0.9
        # keep the arm within reach: pull the target in if needed
        rel = ee - S
        dist = np.linalg.norm(rel)
        if dist > L1 + L2 - 0.02:
            ee = S + rel / dist * (L1 + L2 - 0.02)
        horiz = np.array([ee[0] - S[0], ee[1] - S[1], 0.0])
        hn = np.linalg.norm(horiz)
        up = np.array([0, 0, 1.0])
        hint = up + np.asarray(r.get('elbow_push', (0.0, 0.0, 0.0)), float)     # bend the elbow away from obstacles
        ft = float(r.get('frame_t', 1.0))
        board_ee = ee.copy()
        if r['kind'] == 'board':     # the forearm ends at the back of the suction frame, not inside the board
            back = 0.038 + 0.03 + 0.05
            wrist = ee + np.array([0, -back * ft, back * (1 - ft)])
            elbow, wrist = self._elbow(S, wrist, L1, L2, hint)
            ee = board_ee + (wrist - (board_ee + np.array([0, -back * ft, back * (1 - ft)])))   # if out of reach
        else:
            elbow, ee = self._elbow(S, ee, L1, L2, hint)
            wrist = ee
        self.capsule('link1', S, elbow)
        self.capsule('link2', elbow, wrist)
        kind = r['kind']
        if kind == 'board':
            # the suction frame follows the board it carries: lying on top of a flat board (frame_t 0) and in front
            # of an upright one (frame_t 1)
            q = (1 - ft) * Q_FLAT + ft * Q_ID
            self.put('frame', ee + np.array([0, -0.038 * ft, 0.038 * (1 - ft)]), q / np.linalg.norm(q))   # touching the board
            self.hide('gripper')
        else:
            self.hide('frame')
            self.put('gripper', ee)

    # ---------------------------------------------------------------- the rest
    def boards(self, bs):
        for b, d in enumerate(bs):
            self.put(f'board{b}', d['pos'], d['quat'])
            hx, hy, hz = d['size']
            self.model.geom_size[self.gid[f'board{b}_g']][:] = (hx, hy, hz)

    def tools(self, t):
        """t: {name: (pos, quat) or None}, name in TOOL_NAMES."""
        for name in TOOL_NAMES:
            v = t.get(name)
            if v is None:
                self.hide('tool_' + name)
            else:
                self.put('tool_' + name, v[0], v[1])

    def marker(self, m):
        if m is None:
            self.hide('marker')
        else:
            self.put('marker', m['pos'])
            self.color('marker_g', m['rgba'])

    def apply(self, pose):
        self.worker(pose['worker'])
        self.robot(pose['robot'])
        self.boards(pose['boards'])
        self.tools(pose['tools'])
        self.marker(pose['marker'])
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()
