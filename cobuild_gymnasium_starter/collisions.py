"""Shape-vs-shape distances in the 3D scene (used by audit_3d.py and by the choreography to stay clear).

The environment has no geometry (docs/MODEL.md), so "collisions" are only about the DRAWN shapes: the worker's
mannequin, the robot, the boards, the tools and the static set. This module measures the signed distance between
them with MuJoCo (mj_geomDistance) and knows which contacts are intended (a hand on a tool, the suction frame
on the board it carries).
"""
import mujoco
import numpy as np

TOL = 0.02          # metres of overlap that still counts as "touching"
WORKER_BODY = ('torso_g', 'leg_l_g', 'leg_r_g', 'head_g', 'helmet_g')
WORKER_ARMS = ('arm_l_up_g', 'arm_l_lo_g', 'arm_r_up_g', 'arm_r_lo_g')
WORKER_HANDS = ('hand_l_g', 'hand_r_g')
ROBOT_HOLDERS = ('frame_g', 'cup0', 'cup1', 'cup2', 'cup3', 'gripper_g')      # may touch what they carry
ROBOT_REST = ('rb_body', 'rb_column', 'rb_wheel0', 'rb_wheel1', 'rb_wheel2', 'rb_wheel3', 'carriage_g',
              'link1_g', 'link2_g')
TOOLS = ('tc_g', 'tc_tape', 'tg_g', 'tg_h', 'wtc_g', 'wtg_g', 'wtg_h')
STATIC = ('pallet', 'cart_top', 'cart_leg1', 'cart_leg2', 'tooltable', 'tooltable_l1', 'tooltable_l2', 'backwall')


class Checker:
    def __init__(self, scene):
        self.s = scene
        m = scene.model
        self.names = {i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)}
        self.id = {v: k for k, v in self.names.items() if v}
        self.floor = self.id['floor']
        self.fromto = np.zeros(6)

    def dist(self, a, b):
        return mujoco.mj_geomDistance(self.s.model, self.s.data, self.id[a], self.id[b], 0.5, self.fromto)

    def lowest(self, name):
        """Lowest world z of a geom (exact enough: bounding sphere for capsules/spheres, box corners for boxes)."""
        m, d = self.s.model, self.s.data
        i = self.id[name]
        pos, mat, size, kind = d.geom_xpos[i], d.geom_xmat[i].reshape(3, 3), m.geom_size[i], m.geom_type[i]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return pos[2] - np.abs(mat[2]) @ size[:3]
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return pos[2] - abs(mat[2, 2]) * size[1] - size[0]
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return pos[2] - abs(mat[2, 2]) * size[1] - np.sqrt(max(1 - mat[2, 2] ** 2, 0)) * size[0]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return pos[2] - size[0]
        if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return pos[2] - np.sqrt(((mat[2] * size[:3]) ** 2).sum())
        return pos[2]

    def check(self, pose, kinds=None):
        """List of (kind, a, b, depth) for the current scene (only the given kinds, if any)."""
        out = []
        boards = [f'board{b}_g' for b in range(len(pose['boards']))]
        movers = [*WORKER_BODY, *WORKER_ARMS, *WORKER_HANDS, *ROBOT_HOLDERS, *ROBOT_REST, *TOOLS, *boards]
        for g in (movers if kinds is None or 'floor' in kinds else ()):
            if g in self.id and self.s.data.geom_xpos[self.id[g]][2] > -4:     # parked things (z=-5) are hidden
                low = self.lowest(g)
                if low < -TOL:
                    out.append(('floor', g, 'floor', -low))
        wr = kinds is None or 'worker-robot' in kinds
        for a in (*WORKER_BODY, *WORKER_ARMS) if wr else ():
            for r in (*ROBOT_REST, *ROBOT_HOLDERS):
                d = self.dist(a, r)
                if d < -TOL:
                    out.append(('worker-robot', a, r, -d))
        for a in WORKER_HANDS if wr else ():
            for r in ROBOT_REST:
                d = self.dist(a, r)
                if d < -TOL:
                    out.append(('worker-robot', a, r, -d))
        for a in WORKER_BODY if kinds is None or 'worker-board' in kinds else ():
            for b in boards:
                d = self.dist(a, b)
                if d < -TOL:
                    out.append(('worker-board', a, b, -d))
        for r in ROBOT_REST if kinds is None or 'robot-board' in kinds else ():
            for b in boards:
                d = self.dist(r, b)
                if d < -TOL:
                    out.append(('robot-board', r, b, -d))
        for r in (*ROBOT_REST, *ROBOT_HOLDERS) if kinds is None or 'robot-static' in kinds else ():
            for s in (*STATIC, *[n for n in self.id if n.startswith(('stud', 'plate'))]):
                if r in ROBOT_HOLDERS and s in ('backwall',) and False:
                    continue
                d = self.dist(r, s)
                if d < -TOL:
                    out.append(('robot-static', r, s, -d))
        for b in boards if kinds is None or 'board-static' in kinds else ():
            for s in (*STATIC, *[n for n in self.id if n.startswith(('stud', 'plate'))]):
                d = self.dist(b, s)
                if d < -TOL:
                    out.append(('board-static', b, s, -d))
        for i, a in enumerate(boards if kinds is None or 'board-board' in kinds else ()):
            for b in boards[i + 1:]:
                d = self.dist(a, b)
                if d < -TOL:
                    out.append(('board-board', a, b, -d))
        return out




ROBOT_BODY_PARTS = {'rb_body', 'rb_column', 'rb_wheel0', 'rb_wheel1', 'rb_wheel2', 'rb_wheel3', 'carriage_g'}
YIELD = ('worker-robot', 'robot-board', 'robot-static')


def yield_robot(scene, pose, checker, iterations=6):
    """Make the robot give way: where one of its parts overlaps the worker, a board or the set, slide its base away
    (body parts) or bend its elbow away (arm links). Returns the adjusted pose. A safe robot yields to people."""
    pose = dict(pose, robot=dict(pose['robot']))
    r = pose['robot']
    r['base'] = np.array(r['base'], float)
    r['elbow_push'] = np.array(r['elbow_push'], float)
    for _ in range(iterations):
        scene.apply(pose)
        hits = [h for h in checker.check(pose, YIELD) if h[3] > TOL]
        if not hits:
            break
        data = scene.data
        for kind, g1, g2, depth in hits:
            robot_part, other = (g1, g2) if g1 in ROBOT_BODY_PARTS | {'link1_g', 'link2_g', *ROBOT_HOLDERS} else (g2, g1)
            n = data.geom_xpos[checker.id[robot_part]] - data.geom_xpos[checker.id[other]]
            if robot_part in ('link1_g', 'link2_g'):
                r['elbow_push'] += 1.5 * n / max(np.linalg.norm(n), 1e-6)
            else:
                nh = n[:2] / max(np.linalg.norm(n[:2]), 1e-6)
                r['base'] += nh * (depth + 0.02)
    scene.apply(pose)
    return pose
