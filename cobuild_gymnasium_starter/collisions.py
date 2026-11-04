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
ROBOT_REST = ('rb_body', 'rb_tray', 'rb_column', 'rb_wheel0', 'rb_wheel1', 'rb_wheel2', 'rb_wheel3', 'carriage_g',
              'link1_g', 'link2_g')
TOOLS = tuple(f't{n}' for j in (0, 1) for n in (f'K{j}_g', f'K{j}_t', f'G{j}_g', f'G{j}_h'))
STATIC = ('pallet', 'pallet_leg0', 'pallet_leg1', 'pallet_leg2', 'pallet_leg3', 'cart_top', 'cart_leg1', 'cart_leg2', 'tooltable', 'tooltable_l1', 'tooltable_l2', 'backwall')


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




ROBOT_BODY_PARTS = {'rb_body', 'rb_tray', 'rb_column', 'rb_wheel0', 'rb_wheel1', 'rb_wheel2', 'rb_wheel3', 'carriage_g'}
YIELD = ('worker-robot', 'robot-board', 'robot-static')


def yield_robot(scene, pose, checker, iterations=12, move_base=True):
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
            elif move_base:              # (not while it carries a board: the board would be left behind)
                nh = n[:2] / max(np.linalg.norm(n[:2]), 1e-6)
                r['base'] += nh * (depth + 0.02)
    scene.apply(pose)
    return pose


ROBOT_ALL = (*ROBOT_REST, *ROBOT_HOLDERS)
WORKER_ALL = (*WORKER_BODY, *WORKER_ARMS, *WORKER_HANDS)
CLEAR = 0.003            # what "no contact" means for separate(): at most 3 mm of overlap


def robot_worker_contacts(checker):
    """(robot geom, worker geom, depth, unit vector from the worker to the robot) for every overlap.
    Only a hand touching the gripper (taking a tool) is allowed."""
    out = []
    d = checker.s.data
    for w in WORKER_ALL:
        for r in ROBOT_ALL:
            if w in WORKER_HANDS and r == 'gripper_g':
                continue
            dist = mujoco.mj_geomDistance(checker.s.model, d, checker.id[w], checker.id[r], 0.5, checker.fromto)
            if -dist > CLEAR:
                n = d.geom_xpos[checker.id[r]] - d.geom_xpos[checker.id[w]]
                out.append((r, w, -dist, n / max(np.linalg.norm(n), 1e-6)))
    return out


def separate(scene, pose, checker, move_base=True, hands_fixed=False, iterations=24):
    """Last word on every frame: as long as any robot part is inside the worker, the robot bends its elbow (arm
    links), slides its base (base parts, unless it carries a board) and, when its tool or suction frame is the part
    that touches, the worker steps back or moves the hand away. A safe robot never pushes into a person."""
    pose = dict(pose, robot=dict(pose['robot']), worker=dict(pose['worker']))
    r, w = pose['robot'], pose['worker']
    r['base'] = np.array(r['base'], float)
    r['elbow_push'] = np.array(r['elbow_push'], float)
    w['pos'] = np.array(w['pos'], float)
    for k in ('hand_l', 'hand_r'):
        w[k] = np.array(w[k], float)
    for it in range(iterations):
        scene.apply(pose)
        hits = robot_worker_contacts(checker)
        if not hits:
            break
        stubborn = it >= iterations // 3            # the elbow alone did not clear it: the person moves too
        for rg, wg, depth, n in hits:
            step = depth + 0.01
            nh = np.array([n[0], n[1], 0.0])
            nh = nh / max(np.linalg.norm(nh), 1e-6)
            if rg in ('link1_g', 'link2_g') and not stubborn:
                r['elbow_push'] += 2.0 * n
                continue
            if rg in ROBOT_BODY_PARTS and move_base:
                r['base'][:2] += nh[:2] * step
            elif wg in ('hand_l_g', 'arm_l_lo_g', 'arm_l_up_g') and not hands_fixed:
                w['hand_l'] = w['hand_l'] - n * step
            elif wg in ('hand_r_g', 'arm_r_lo_g', 'arm_r_up_g') and not hands_fixed:
                w['hand_r'] = w['hand_r'] - n * step
            else:                                   # the worker steps back (hands stay on a board they carry)
                w['pos'][:2] -= nh[:2] * step
                if not hands_fixed:
                    for k in ('hand_l', 'hand_r'):
                        w[k][:2] -= nh[:2] * step
    scene.apply(pose)
    return pose
