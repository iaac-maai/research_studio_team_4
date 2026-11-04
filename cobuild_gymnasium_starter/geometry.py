"""The floor plan of the job site: where things are, where the worker and the robot base stand, and the
separation rule that keeps them apart. Pure numpy, no MuJoCo: it is used by BOTH the environment (so the
RL agent feels collisions) and the 3D viewer (so what you see is what was simulated).

Coordinates in metres. The wall lies in the plane y = 0, the room is at y < 0, the worker faces +y.

    pallet with the board stack (left) | wall 4.5 m wide | cutting cart (right)

Only the 2D footprints on the floor are modelled: the worker is a circle (WORKER_RADIUS), the robot base a
circle (ROBOT_RADIUS). ZONE = the sum is the minimum centre-to-centre distance between them.
The arm reach, the carried board and the 3D shapes are NOT part of the model (see docs/MODEL.md).
"""
import numpy as np

import config as C
import ergonomics as E

# --- the room ----------------------------------------------------------------------------------
WALL_W = 4.5
BOARD_H, BOARD_T = 1.22, 0.016
FULL_W, CUT_W = 2.44, 2.06
STACK_XY = np.array([-3.0, -2.4])
CART_XY = np.array([5.3, -1.05])
STAGE_X, STAGE_Y = -2.6, -0.45
HOME_XY = np.array([1.0, -3.6])
WALL_Y = -0.55                     # where the worker stands in front of the wall

# Footprints nobody's feet or wheels go through: x0, x1, y0, y1
OBSTACLES = [(-4.3, -1.7, -3.05, -1.75),        # pallet with the board stack
             (3.9, 6.7, -1.17, -0.93),          # cutting cart
             (2.0, 3.2, -4.2, -3.6)]            # tool table
ROBOT_RADIUS, WORKER_RADIUS = 0.6, 0.35
ZONE = ROBOT_RADIUS + WORKER_RADIUS             # minimum distance between worker and robot base
WALL_CLEARANCE_Y = -0.3
SUBSTEPS = 4                                    # a tick (6 s) is checked at this many points in time
MAX_STALL_TICKS = 3                             # after this many stopped ticks in a row, contact happens


def board_width(b, final=True):
    return CUT_W if (E.needs_cut(b) and final) else FULL_W


def board_wall_pose(b):
    col = b % 2
    x0 = 0.0 if col == 0 else FULL_W
    cx = x0 + board_width(b) / 2
    cz = 1.83 if E.is_upper(b) else 0.61
    return np.array([cx, -BOARD_T / 2, cz])


def push_out(xy, radius):
    """Move a floor point out of every obstacle (inflated by `radius`) and off the wall: continuous in xy."""
    x, y = float(xy[0]), float(xy[1])
    for _ in range(3):
        for x0, x1, y0, y1 in OBSTACLES:
            x0, x1, y0, y1 = x0 - radius, x1 + radius, y0 - radius, y1 + radius
            if x0 < x < x1 and y0 < y < y1:
                d = {'l': x - x0, 'r': x1 - x, 'b': y - y0, 't': y1 - y}
                k = min(d, key=d.get)
                if k == 'l': x = x0
                elif k == 'r': x = x1
                elif k == 'b': y = y0
                else: y = y1
    return np.array([x, min(y, WALL_CLEARANCE_Y - radius + 0.3)])


# --- where the worker stands -------------------------------------------------------------------
def station(name, b):
    if name == 'wall':
        return np.array([board_wall_pose(b)[0] - 0.95, WALL_Y])
    if name == 'wall_right':                                # working on a board the robot holds from the left
        return np.array([board_wall_pose(b)[0] + 0.95, WALL_Y])
    if name == 'cut':
        return np.array([CART_XY[0], CART_XY[1] - 0.65])
    if name == 'staged':                                    # next to the boards the robot left on the left of the wall
        return np.array([STAGE_X + 0.8, -1.1])
    if name == 'aside':                                     # out of the way while the robot lifts board b in
        return np.array([board_wall_pose(b)[0] + board_width(b) / 2 + 0.6, WALL_Y])
    return np.array([STACK_XY[0], STACK_XY[1] - 1.06])      # 'stack'


HAULS = {'haul_stack_cut': ('stack', 'cut'), 'haul_stack_wall': ('stack', 'staged'), 'haul_cut_wall': ('cut', 'staged')}


def worker_xy(env, advance=0):
    """Floor position of the worker. advance=1 -> where the worker will be at the end of this tick
    (the current phase one tick further along)."""
    b = min(env.board, C.N_BOARDS - 1)
    ph = env.phase if env.worker_state == 'working' else None
    if ph and ph.startswith('haul'):
        total = max(env.elapsed + env.remaining, 1)
        p = min(1.0, (env.elapsed + advance) / total)
        a, c = HAULS[ph]
        return station(a, b) * (1 - p) + station(c, b) * p       # carrying: not pushed out of the pallet
    if ph == 'place' and env.phase_mode != 'aligned':        # alone: walk to the staged board, then carry it to the wall
        total = max(env.elapsed + env.remaining, 1)
        p = min(1.0, (env.elapsed + advance) / total)
        st, wl = station('staged', b), push_out(station('wall', b), WORKER_RADIUS)
        return st + (wl - st) * ((p - 0.4) / 0.6) if p >= 0.4 else wl + (st - wl) * (p / 0.4)
    name = ph or (env.queue[0] if (env.queue and not env.done) else 'prep')
    if env.robot_state == 'positioning_hold' and name != 'cut':
        return push_out(station('aside', b), WORKER_RADIUS)   # the board comes in along the wall: step aside
    if name == 'cut':
        return push_out(station('cut', b), WORKER_RADIUS)
    if env.robot_state == 'holding' or env.boards[b]['loc'] == 'held':
        return push_out(station('wall_right', b), WORKER_RADIUS)
    return push_out(station('wall', b), WORKER_RADIUS)


# --- where the robot base stands for each job ----------------------------------------------------
def spot_for_board(loc, rank=0):
    """Where the base stands to put a board down at / pick it from `loc`."""
    if loc == 'cutstation':
        base = np.array([CART_XY[0] - 0.9, CART_XY[1] - 0.06 - 1.2])
    elif loc == 'stack':
        base = np.array([STACK_XY[0] + 0.9, STACK_XY[1] - 0.06 - 1.2])
    else:                                                           # staged
        base = np.array([STAGE_X + 0.9, STAGE_Y - 0.22 * rank - 0.06 - 1.2])
    return push_out(base, ROBOT_RADIUS + 0.1)


def spot_steady():
    return push_out(np.array([CART_XY[0] - 0.55 - 0.95, CART_XY[1] - 0.06 - 1.1]), ROBOT_RADIUS + 0.1)


def spot_hold(b):
    """The robot holds a board from its LEFT (the side it comes from); the worker fastens it from the right."""
    return push_out(np.array([board_wall_pose(b)[0] - 0.75, -0.06 - 1.25]), ROBOT_RADIUS + 0.1)


def spot_handover(wxy, pose_idx):
    side, _, d = E.POSES[pose_idx]
    sgn = 1 if side == 1 else -1
    return push_out(np.array([wxy[0] + sgn * (d + 0.7), wxy[1] - 1.25]), ROBOT_RADIUS + 0.1)


def robot_progress(env, advance=0):
    if env.robot_total <= 0:
        return 1.0
    return float(np.clip(1 - (env.robot_ticks - advance) / env.robot_total, 0.0, 1.0))


MOVING = ('bringing', 'positioning_steady', 'positioning_hold', 'fetching_tool')


def route(a, b, w):
    """Waypoint that takes a robot going a -> b around a worker standing at w, or None if the line is clear."""
    a, b, w = (np.asarray(v, float) for v in (a, b, w))
    ab = b - a
    t = float(np.clip(np.dot(w - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0))
    c = a + t * ab
    if np.linalg.norm(c - w) >= ZONE + 0.2:
        return None
    n = c - w
    n = n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-6 else np.array([0.0, -1.0])
    return push_out(w + n * (ZONE + 0.3), ROBOT_RADIUS + 0.1)


def _hits_box(a, b, box):
    """Does the segment a-b pass through the (open) box x0, x1, y0, y1? (Liang-Barsky)"""
    x0, x1, y0, y1 = box
    t0, t1 = 0.0, 1.0
    d = b - a
    for p, q in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
        if abs(p) < 1e-12:
            if q <= 0:
                return False
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 >= t1:
                return False
    return t1 - t0 > 1e-6


def clear_path(pts, radius=ROBOT_RADIUS + 0.1):
    """Insert corner waypoints so that no leg of the path drives the robot base through the pallet, cart or table."""
    pts = [np.asarray(p, float) for p in pts]
    for _ in range(6):
        changed = False
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            detour = []
            for x0, x1, y0, y1 in OBSTACLES:
                box = (x0 - radius + 0.02, x1 + radius - 0.02, y0 - radius + 0.02, y1 + radius - 0.02)
                if not _hits_box(a, b, box):
                    continue
                c = [np.array(v) for v in ((x0 - radius - 0.05, y0 - radius - 0.05), (x1 + radius + 0.05, y0 - radius - 0.05),
                                           (x1 + radius + 0.05, y1 + radius + 0.05), (x0 - radius - 0.05, y1 + radius + 0.05))]
                options = [[k] for k in c] + [[c[i], c[(i + s) % 4]] for i in range(4) for s in (1, 3)]
                ok = [o for o in options if not any(_hits_box(p, q, box) for p, q in zip([a, *o], [*o, b]))]
                if ok:
                    detour = min(ok, key=lambda o: sum(np.linalg.norm(q - p) for p, q in zip([a, *o], [*o, b])))
                    break
            out += detour + [b]
            changed |= bool(detour)
        pts = out
        if not changed:
            break
    return pts


def carry_lane(src, dest):
    """Waypoints for carrying a board from `src` to `dest`. A cut board going back west would sweep through the
    worker standing at the wall, so it is driven along the front lane first (the board is 2.4 m wide)."""
    if src == 'cutstation' and dest == 'staged':
        return [np.array([0.9, -2.5]), np.array([0.9, -1.05])]
    return []


def cumulative(wps):
    d = np.r_[0.0, np.cumsum([np.linalg.norm(np.asarray(wps[i + 1]) - np.asarray(wps[i])) for i in range(len(wps) - 1)])]
    return d, max(float(d[-1]), 1e-9)


def point_along(wps, t):
    d, total = cumulative(wps)
    s = float(np.clip(t, 0.0, 1.0)) * total
    i = int(np.clip(np.searchsorted(d, s, side='right') - 1, 0, len(wps) - 2))
    seg = max(d[i + 1] - d[i], 1e-9)
    return np.asarray(wps[i], float) + (np.asarray(wps[i + 1], float) - np.asarray(wps[i], float)) * ((s - d[i]) / seg)


def robot_xy(env, advance=0):
    """Floor position of the robot base now (advance=0) or at the end of this tick (advance=1)."""
    if env.robot_state in MOVING:
        return push_out(point_along(env.robot_wps, robot_progress(env, advance)), ROBOT_RADIUS + 0.1)
    return env.robot_pos.copy()


BOARD_MARGIN = WORKER_RADIUS + 0.10     # worker body + a hand of clearance around the board the robot carries


def board_xy(loc):
    return {'stack': STACK_XY, 'cutstation': CART_XY}.get(loc, np.array([STAGE_X, STAGE_Y]))


def carry_offset(loc, b=0):
    """Vector from the robot base to the centre of the board it holds when standing at `loc` ('wall' = holding
    board b against the studs)."""
    if loc == 'wall':
        return board_wall_pose(b)[:2] - spot_hold(b)
    return board_xy(loc) - spot_for_board(loc)


def carry_fraction(env, advance=0):
    """0..1 = how far along the carrying part of the trip the robot is; None before pickup / when not carrying."""
    if env.robot_state not in ('bringing', 'positioning_hold') or not env.job or env.robot_pick is None:
        return None
    d, total = cumulative(env.robot_wps)
    fp = d[env.robot_pick] / total
    t = robot_progress(env, advance)
    if t < fp - 1e-9:
        return None
    return (t - fp) / max(1 - fp, 1e-9)


def carry_offset_now(env, advance=0):
    """Vector from the base to the board while the robot is carrying one (None otherwise)."""
    u = carry_fraction(env, advance)
    if u is None:
        return None
    b = env.job['board']
    return carry_offset(env.job['src'], b) * (1 - u) + carry_offset(env.job['dest'], b) * u


def _seg_dist(p, w0, w1):
    seg = w1 - w0
    t = float(np.clip(np.dot(p - w0, seg) / max(np.dot(seg, seg), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (w0 + t * seg)))


def away_from_path(r, w0, w1, dist):
    """Nearest free place for the robot base that is at least `dist` from the worker's walking segment w0 -> w1
    (where a robot at r steps to get out of the worker's way), outside every obstacle."""
    r, w0, w1 = (np.asarray(v, float) for v in (r, w0, w1))
    cands = []
    for centre in (w0, w1, (w0 + w1) / 2):
        for k in range(16):
            a = 2 * np.pi * k / 16
            cands.append(push_out(centre + dist * np.array([np.cos(a), np.sin(a)]), ROBOT_RADIUS + 0.1))
    stage0, stage1 = np.array([STAGE_X - FULL_W / 2, STAGE_Y]), np.array([STAGE_X + FULL_W / 2, STAGE_Y + 0.2])
    ok = [c for c in cands if _seg_dist(c, w0, w1) >= dist - 0.05
          and _seg_dist(c, stage0, stage1) >= ROBOT_RADIUS + 0.15]      # never park on the boards waiting there
    pool = ok or cands
    # nearest place, but stay out of the narrow strip along the wall (a robot parked there blocks itself in)
    return min(pool, key=lambda c: (np.linalg.norm(c - r) + 2.0 * max(0.0, c[1] + 1.4) if ok else -_seg_dist(c, w0, w1)))


def separation_violated(w0, w1, r0, r1, c0=None, c1=None, half_w=0.0):
    """True if, at any sub-step of the tick, the worker comes closer than ZONE to the robot base or, when the
    robot carries a board (offsets c0, c1 from the base, half width half_w), closer than BOARD_MARGIN to it."""
    for k in range(SUBSTEPS + 1):
        t = k / SUBSTEPS
        w, r = w0 * (1 - t) + w1 * t, r0 * (1 - t) + r1 * t
        if np.linalg.norm(w - r) < ZONE:
            return True
        if c0 is not None:
            c = r + c0 * (1 - t) + c1 * t
            dx = max(abs(w[0] - c[0]) - half_w, 0.0)
            if np.hypot(dx, w[1] - c[1]) < BOARD_MARGIN:
                return True
    return False
