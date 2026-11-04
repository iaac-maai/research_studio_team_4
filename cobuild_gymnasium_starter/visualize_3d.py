"""3D view of the drywall installation: watch a strategy assist a worker, as a video or live.

    python visualize_3d.py --profile short_left --strategy C_cobuild_rule --video demo.mp4
    python visualize_3d.py --profile low_capacity_short_left --compare --video compare.mp4
    python visualize_3d.py --profile random --compare --top no_robot --bottom C_cobuild_ppo --video x.mp4
    python visualize_3d.py --profile short_left --strategy C_cobuild_ppo --live      # window (cv2), no video file

What you see (schematic, to scale): the wall with studs every 16 in and 4 board positions, the board pallet
(left), the cutting cart (right), the worker as a mannequin built from the profile's HEIGHT, and the robot
(mobile base + lift column + 2-link arm + suction frame). Worker turns red while waiting. The sphere is the
planned tool-handover point (green = comfortable for this worker, orange = awkward).

The decisions come from exactly the same CobuildDrywallEnv and strategies used in training and evaluation.
The 3D motion only acts them out: positions are interpolated between simulation ticks (1 tick = 6 s of
simulated work), it is not a physics or grasping simulation.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config as C
import ergonomics as E
import scene3d as S
from collisions import Checker, yield_robot
from cobuild_env import CobuildDrywallEnv, action_label
from profiles import GENERIC, PERSONAS, sample_population
from strategies import LABELS, STRATEGY_KEYS, make_strategy

FRAMES_PER_TICK = 8
FPS = 20
WALL_Y = -0.55          # where the worker stands in front of the wall
GOOD, AWK = (0.15, 0.75, 0.3, 0.55), (0.95, 0.6, 0.1, 0.6)


# ----------------------------------------------------------------------------- geometry helpers
def station(name, b):
    if name == 'wall':
        return np.array([S.board_wall_pose(b)[0] - 0.95, WALL_Y])
    if name == 'cut':
        return np.array([S.CART_XY[0], S.CART_XY[1] - 0.65])
    return np.array([S.STACK_XY[0], S.STACK_XY[1] - 1.06])      # 'stack'


def board_pose(env, b, loc, rank=0):
    cut_done = env.boards[b]['cut_done']
    w = S.board_width(b, final=cut_done)
    size = np.array([w / 2, S.BOARD_T / 2, S.BOARD_H / 2])
    if loc == 'stack':
        return dict(pos=np.array([S.STACK_XY[0], S.STACK_XY[1], 0.1 + 0.016 * (rank + 0.5)]), quat=S.Q_FLAT, size=size)
    if loc == 'cutstation':
        return dict(pos=np.array([S.CART_XY[0], S.CART_XY[1], 0.12 + S.BOARD_H / 2]), quat=S.Q_ID, size=size)
    if loc == 'staged':
        return dict(pos=np.array([S.STAGE_X, S.STAGE_Y - 0.22 * rank, S.BOARD_H / 2 + 0.01]), quat=S.Q_ID, size=size)
    return dict(pos=S.board_wall_pose(b), quat=S.Q_ID, size=size)       # held / wall


def blend(p, q, t):
    quat = p['quat'] * (1 - t) + q['quat'] * t
    if np.dot(p['quat'], q['quat']) < 0:
        quat = p['quat'] * (1 - t) - q['quat'] * t
    return dict(pos=p['pos'] + (q['pos'] - p['pos']) * t, quat=quat / np.linalg.norm(quat), size=q['size'])


def robot_at(ee, side=0.8, depth=1.1, kind='board', lift=0.35, frame_t=1.0):
    ee = np.asarray(ee, float)
    base = np.array([ee[0] + side, ee[1] - depth])
    yaw = float(np.arctan2(ee[1] - base[1], ee[0] - base[0]))
    return dict(base=base, yaw=yaw, carriage_z=float(ee[2] + lift), ee=ee, kind=kind, frame_t=frame_t,
                elbow_push=np.zeros(3))


def handover_target(P, wx, wy, pose_idx):
    side, h, d = E.POSES[pose_idx]
    return np.array([wx + (1 if side == 1 else -1) * d, wy, h])


# ----------------------------------------------------------------------------- keeping things apart
# Footprints on the floor that the robot base and the worker must stay out of: x0, x1, y0, y1 (metres).
OBSTACLES = [(-3.0, -0.4, -3.05, -1.75),        # pallet with the board stack
             (3.9, 6.7, -1.17, -0.93),          # cutting cart
             (2.0, 3.2, -4.2, -3.6)]            # tool table
ROBOT_RADIUS, WORKER_RADIUS = 0.75, 0.35
WALL_CLEARANCE_Y = -0.3                          # nobody's feet go beyond this towards the wall


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


def board_lowest(p):
    """Lowest world z of a board pose (centre z minus the half-extent along the vertical)."""
    w, x, y, z = p['quat']
    row = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])      # third row of R(quat)
    return p['pos'][2] - np.abs(row) @ p['size']


def blend_clear(p, q, t):
    """Blend two board poses, lifting the board so a rotating corner never dips below the line between its start
    and end heights (a flat board turning upright would otherwise cut through the pallet and the floor)."""
    out = blend(p, q, t)
    floor = (1 - t) * board_lowest(p) + t * board_lowest(q) + 0.05 * np.sin(np.pi * t)
    low = board_lowest(out)
    if low < floor:
        out['pos'] = out['pos'] + np.array([0, 0, floor - low])
    return out


def settle(robot):
    """Keep the robot's base clear of the pallet, cart and tool table."""
    robot['base'] = push_out(robot['base'], ROBOT_RADIUS)
    return robot


# ----------------------------------------------------------------------------- the pose of everything
def pose_of(env, P):
    N = C.N_BOARDS
    b = min(env.board, N - 1)
    ph = env.phase if env.worker_state == 'working' else None
    pend = env.queue[0] if (env.queue and not env.done) else None
    name = ph or pend or 'prep'
    p = env.elapsed / max(env.elapsed + env.remaining, 1) if ph else 0.0
    H = P.height_m

    # ---- boards
    stack = [i for i in range(N) if env.boards[i]['loc'] == 'stack']
    staged = [i for i in range(N) if env.boards[i]['loc'] == 'staged']
    boards = []
    for i in range(N):
        loc = env.boards[i]['loc']
        rank = stack.index(i) if loc == 'stack' else (staged.index(i) if loc == 'staged' else 0)
        bp = board_pose(env, i, loc, len(stack) - 1 - rank if loc == 'stack' else rank)
        if env.robot_state == 'bringing' and env.job and env.job['board'] == i:
            t = 1 - env.robot_ticks / max(env.robot_total, 1)
            bp = blend_clear(bp, board_pose(env, i, env.job['dest']), t)
        elif env.robot_state == 'positioning_hold' and env.job and env.job['board'] == i:
            t = 1 - env.robot_ticks / max(env.robot_total, 1)
            bp = blend_clear(bp, board_pose(env, i, 'wall'), t)
        elif i == b and ph and ph.startswith('haul'):
            src = 'stack' if ph in ('haul_stack_wall', 'haul_stack_cut') else 'cutstation'
            dst = 'cutstation' if ph == 'haul_stack_cut' else 'staged'
            bp = blend_clear(board_pose(env, i, src, 0), board_pose(env, i, dst), p)
        elif i == b and ph == 'place' and env.phase_mode != 'aligned':
            bp = blend_clear(board_pose(env, i, 'staged'), board_pose(env, i, 'wall'), min(1.0, p * 1.2))
        boards.append(bp)

    # ---- worker position and what the hands do
    if ph and ph.startswith('haul'):
        a = {'haul_stack_cut': ('stack', 'cut'), 'haul_stack_wall': ('stack', 'wall'), 'haul_cut_wall': ('cut', 'wall')}[ph]
        xy = station(a[0], b) * (1 - p) + station(a[1], b) * p
    elif name == 'cut':
        xy = station('cut', b)
    else:
        xy = station('wall', b)
    on_stool = E.needs_stool(P, b) and name in ('stool', 'place', 'screw_a', 'screw_b', 'reload_a', 'reload_b') and not env.done
    z0 = 0.2 if on_stool else 0.0
    sh_z = z0 + S.SHOULDER * H
    if env.robot_state == 'bringing' and env.job and env.job['dest'] == 'cutstation':
        xy = np.asarray(xy, float) + np.array([1.0, 0.0])       # the worker steps aside (away from the robot's side)
    carrying = bool(ph and ph.startswith('haul'))
    px, py = (float(xy[0]), float(xy[1])) if carrying else push_out(xy, WORKER_RADIUS)
    if ph and (ph.startswith('haul') or (ph == 'place' and env.phase_mode != 'aligned')):
        # a carried board is in front of the worker, never through them
        wq = boards[b]['quat']
        row = np.array([2 * (wq[1] * wq[2] + wq[0] * wq[3]), 1 - 2 * (wq[1] ** 2 + wq[3] ** 2),
                        2 * (wq[2] * wq[3] - wq[0] * wq[1])])
        py = min(py, boards[b]['pos'][1] - np.abs(row) @ boards[b]['size'] - 0.5)
    idle_l = np.array([px - 0.17 * H / 1.75, py, z0 + 0.5 * H])
    idle_r = np.array([px + 0.17 * H / 1.75, py, z0 + 0.5 * H])
    hl, hr, lean = idle_l, idle_r, 0.0
    bp = boards[b]
    hw = bp['size'][0]
    wtools = {'wtool_cut': None, 'wtool_gun': None}
    robot_busy = env.robot_state in ('bringing', 'positioning_hold', 'holding', 'positioning_steady', 'steadying')
    swing = np.sin(p * 2 * np.pi * 3)
    if env.done:
        pass
    elif name == 'prep' or name == 'stool':
        hr = np.array([px + 0.3, -0.04, 1.45 + 0.08 * swing]) if ph else idle_r
        lean = 0.0
    elif name == 'tools_fetch':
        hr = np.array([px + 0.3, py + 0.3, z0 + 0.9])
    elif name == 'measure':        # measuring the opening on the wall
        c = S.board_wall_pose(b)
        hr = np.array([c[0] - 1.0 + 2.0 * p, -0.04, 1.15 + 0.1 * swing])
        hl = np.array([c[0] - 1.0, -0.04, 1.15])
        lean, wtools['wtool_cut'] = 0.05, hr.copy()
    elif name == 'cut':
        front = S.CART_XY[1] - 0.04
        hr = np.array([S.CART_XY[0] + 0.3, front, 1.3 - 1.1 * p])
        hl = np.array([S.CART_XY[0] - 0.8, front, 0.45 if robot_busy else 0.9])
        lean, wtools['wtool_cut'] = 0.15, hr.copy()
    elif name.startswith('haul') or (name == 'place' and ph):
        c = bp['pos']
        if robot_busy:       # the robot holds the middle of the board: the worker directs it from the left edge
            hl = c + np.array([-hw * 0.85, -0.05, -0.25])
            hr = c + np.array([-hw * 0.55, -0.05, -0.1])
        else:
            hl = c + np.array([-hw * 0.7, -0.05, -0.3])
            hr = c + np.array([hw * 0.7, -0.05, -0.3])
        lean = 0.3 if name.startswith('haul') else (-0.1 if E.is_upper(b) else 0.2)
    elif name in ('screw_a', 'screw_b'):
        c = S.board_wall_pose(b)
        wp = 0.5 * p + (0.5 if name == 'screw_b' else 0.0)
        hold_band = robot_busy                     # the suction frame covers the upper part of the board
        hr = np.array([0.0, -0.06, c[2] + (-0.3 + 0.25 * np.sin(wp * 2 * np.pi * 4) if hold_band
                                           else 0.45 * np.sin(wp * 2 * np.pi * 4))])
        hr[0] = c[0] + (-hw + 2 * hw * wp) * 0.9
        held = env.boards[b]['loc'] == 'held'
        hl = idle_l if held or name == 'screw_b' else np.array([c[0] - hw * 0.4, -0.06, c[2] - 0.3])
        lean = -0.1 if E.is_upper(b) else 0.12
        wtools['wtool_gun'] = hr.copy()
    elif name.startswith('reload'):
        hl = np.array([px - 0.1, py + 0.3, z0 + 1.1])
        hr = np.array([px + 0.1, py + 0.32, z0 + 1.1])
        wtools['wtool_gun'] = hr.copy()
    elif name == 'place':          # waiting for a board
        pass

    # ---- the robot
    rs = env.robot_state
    marker, tools = None, {'tool_cut': None, 'tool_gun': None}
    home_ee = np.array([S.HOME_XY[0], S.HOME_XY[1] + 0.7, 1.3])
    robot = dict(base=S.HOME_XY.copy(), yaw=np.pi / 2, carriage_z=1.5, ee=home_ee, kind='tool', frame_t=1.0,
                 elbow_push=np.zeros(3))
    if rs == 'bringing' and env.job:
        i = env.job['board']
        c = boards[i]['pos']
        flat_t = 1.0 if env.boards[i]['loc'] != 'stack' else float(1 - env.robot_ticks / max(env.robot_total, 1))
        side = -0.9 if env.job['dest'] == 'cutstation' else 0.9      # at the cart the robot works from the left
        robot = robot_at(c + np.array([0, -0.06, 0]), side=side, depth=1.2, frame_t=flat_t)
    elif rs in ('positioning_steady', 'steadying'):
        c = boards[b]['pos']
        robot = robot_at(c + np.array([-0.55, -0.06, 0.2]), side=-0.95, depth=1.1)
    elif rs in ('positioning_hold', 'holding'):
        c = boards[b]['pos']
        robot = robot_at(c + np.array([0, -0.06, 0.35]), side=0.75, depth=1.25)
    elif rs in ('fetching_tool', 'tool_ready') and env.robot_pose is not None:
        tgt = handover_target(P, px, py, env.robot_pose)
        sgn = 1 if E.POSES[env.robot_pose][0] == 1 else -1
        ticks, strain, _ = E.handover_cost(P, env.robot_pose)
        marker = dict(pos=tgt, rgba=np.array(AWK if E.is_awkward(ticks, strain) else GOOD))
        item = env.robot_item
        t = 1.0 if rs == 'tool_ready' else 1 - env.robot_ticks / max(env.robot_total, 1)
        away = np.array([tgt[0] - px, tgt[1] - py, 0.0])
        away = away / max(np.linalg.norm(away), 1e-6)
        ee_t = home_ee * (1 - t) + (tgt + 0.14 * away) * t
        base_t = S.HOME_XY * (1 - t) + np.array([tgt[0] + sgn * 0.7, tgt[1] - 1.25]) * t
        yaw = float(np.arctan2(tgt[1] - base_t[1], tgt[0] - base_t[0])) if t > 0.3 else np.pi / 2
        robot = dict(base=base_t, yaw=yaw, carriage_z=float(ee_t[2] + 0.3), ee=ee_t, kind='tool', frame_t=1.0,
                     elbow_push=np.zeros(3))
        if t >= 0.3:
            tools['tool_cut' if item == 'cut_tools' else 'tool_gun'] = ee_t.copy()
        if rs == 'tool_ready':      # the worker reaches out for it
            if P.side == E.POSES[env.robot_pose][0]:
                if P.handedness == 'right':
                    hr = tgt.copy()
                else:
                    hl = tgt.copy()
            else:
                hr = tgt.copy() if sgn == 1 else hr
                hl = tgt.copy() if sgn == -1 else hl

    robot = settle(robot)
    hl = np.array([hl[0], hl[1], max(hl[2], z0 + 0.2)])
    hr = np.array([hr[0], hr[1], max(hr[2], z0 + 0.2)])
    for k in ('wtool_cut', 'wtool_gun'):
        if wtools[k] is not None:
            wtools[k] = np.array([wtools[k][0], wtools[k][1], max(wtools[k][2], z0 + 0.2)])
    return {'worker': dict(pos=np.array([px, py, z0]), lean=lean, hand_l=hl, hand_r=hr, walk=0.0,
                           moving=bool(ph and ph.startswith('haul')), carrying=bool(ph and ph.startswith('haul')), waiting=env.worker_state in ('waiting', 'requesting'),
                           stool=on_stool),
            'robot': robot, 'boards': boards, 'tools': {**tools, **wtools}, 'marker': marker}


def lerp_pose(a, b, t):
    if a is None or b is None:
        return b if t >= 0.5 else a
    if isinstance(a, dict):
        out = {}
        for k in a:
            if k == 'quat':
                out[k] = blend(dict(pos=0, quat=a[k], size=0), dict(pos=0, quat=b[k], size=0), t)['quat']
            elif k == 'yaw':
                d = (b[k] - a[k] + np.pi) % (2 * np.pi) - np.pi
                out[k] = a[k] + d * t
            else:
                out[k] = lerp_pose(a[k], b[k], t)
        return out
    if isinstance(a, list):
        return [lerp_pose(x, y, t) for x, y in zip(a, b)]
    if a is None or b is None or isinstance(a, (bool, str, np.bool_)):
        return b if t >= 0.5 else a
    return np.asarray(a) + (np.asarray(b) - np.asarray(a)) * t


# ----------------------------------------------------------------------------- HUD
_FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        for path in ('/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/Supplemental/Arial.ttf',
                     '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
            try:
                _FONTS[key] = ImageFont.truetype(path, size, index=1 if (bold and path.endswith('.ttc')) else 0)
                break
            except Exception:
                continue
        else:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


ROBOT_TEXT = {'free': 'idle', 'bringing': 'carrying a board', 'positioning_steady': 'moving to the cut station',
              'steadying': 'steadying the board (cut)', 'positioning_hold': 'lifting the board onto the studs',
              'holding': 'holding the board', 'fetching_tool': 'fetching a tool', 'tool_ready': 'offering a tool'}


def draw_hud(frame, env, P, label, done_text=None):
    img = Image.fromarray(frame).convert('RGBA')
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    W, H = img.size
    d.rounded_rectangle((10, 10, 330, 150), 10, fill=(255, 255, 255, 225))
    secs = env.time * C.TICK_SECONDS
    ink, muted = (30, 30, 28, 255), (100, 98, 92, 255)
    d.text((22, 16), label, font=font(15, True), fill=ink)
    d.text((22, 38), f'{P.name}', font=font(13), fill=muted)
    d.text((22, 55), f'{P.height_m:.2f} m · reach {P.reach_m:.2f} m · lifts {P.lift_capacity_kg:.0f} kg', font=font(11), fill=muted)
    d.text((22, 68), f'pace x{P.pace:.2f} · {P.handedness}-handed', font=font(11), fill=muted)
    d.text((22, 80), f'{secs // 60:02d}:{secs % 60:02d}', font=font(26, True), fill=ink)
    d.text((100, 88), f'board {min(env.board + 1, C.N_BOARDS)}/{C.N_BOARDS}', font=font(13), fill=ink)
    d.text((22, 112), f'worker: {env.activity}  ·  robot: {ROBOT_TEXT[env.robot_state]}', font=font(12), fill=ink)
    d.text((22, 130), f'waiting {env.wait_ticks * C.TICK_SECONDS / 60:.1f} min · strain {env.strain:.0f} · '
                      f'boards held {env.holds}', font=font(12), fill=ink)
    if done_text:
        d.rounded_rectangle((W - 330, 10, W - 10, 70), 10, fill=(27, 175, 122, 235))
        d.text((W - 318, 22), done_text, font=font(24, True), fill=(255, 255, 255, 255))
    return np.array(Image.alpha_composite(img, ov).convert('RGB'))


# ----------------------------------------------------------------------------- running episodes
class View:
    def __init__(self, strategy_key, profile, seed, width, height):
        self.strategy = make_strategy(strategy_key)
        self.P = profile
        self.env = CobuildDrywallEnv(profile=profile, **self.strategy.env_kwargs())
        self.obs, _ = self.env.reset(seed=seed)
        self.scene = S.Scene(profile, width, height)
        self.checker = Checker(self.scene)
        self.pose = pose_of(self.env, profile)
        self.walk = 0.0
        self.label = LABELS[strategy_key]

    @property
    def done(self):
        return self.env.done or self.env.time >= C.MAX_TICKS

    def step(self):
        if self.done:
            return self.pose, self.pose
        prev = self.pose
        a = self.strategy.act(self.env, self.obs)
        self.obs, _, _, _, _ = self.env.step(a)
        self.pose = pose_of(self.env, self.P)
        return prev, self.pose

    def compose(self, a, b, t):
        """The final pose drawn between two tick poses: interpolation, then the clearance rules (the worker walks
        around obstacles, the robot gives way). audit_3d.py measures exactly this."""
        pose = lerp_pose(a, b, t)
        if pose['worker']['moving'] or np.linalg.norm(a['worker']['pos'][:2] - b['worker']['pos'][:2]) > 0.05:
            self.walk += 0.7
            pose['worker']['moving'] = True
        pose['worker']['walk'] = self.walk
        w = pose['worker']                                   # the straight walk between stations goes around obstacles
        if not w['carrying']:        # while carrying, the worker stays behind the board instead
            w['pos'] = np.array([*push_out(w['pos'][:2], WORKER_RADIUS), w['pos'][2]])
        return yield_robot(self.scene, pose, self.checker)

    def frame(self, a, b, t):
        pose = self.compose(a, b, t)
        img = self.scene.render()
        done_text = None
        if self.done:
            secs = self.env.time * C.TICK_SECONDS
            done_text = f'DONE {secs // 60:02d}:{secs % 60:02d}'
        return draw_hud(img, self.env, self.P, self.label, done_text)


def resolve_profile(name, seed):
    if name == 'random':
        return sample_population(np.random.default_rng(seed), name='random_worker')
    return GENERIC if name == 'generic_average' else PERSONAS[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', default='short_left', choices=['random', 'generic_average', *PERSONAS])
    ap.add_argument('--strategy', default='C_cobuild_rule', choices=STRATEGY_KEYS)
    ap.add_argument('--compare', action='store_true', help='two strategies side by side (same worker, same variability)')
    ap.add_argument('--top', default='B_proactive_generic', choices=STRATEGY_KEYS)
    ap.add_argument('--bottom', default='C_cobuild_rule', choices=STRATEGY_KEYS)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--video', help='write an .mp4 instead of opening a window')
    ap.add_argument('--live', action='store_true', help='show frames in a window (press q to quit, space to pause)')
    ap.add_argument('--snapshot', help='save one PNG and exit')
    ap.add_argument('--at-tick', type=int, default=60)
    ap.add_argument('--width', type=int, default=800)
    ap.add_argument('--height', type=int, default=560)
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed multiplier')
    args = ap.parse_args()
    if not (args.video or args.live or args.snapshot):
        args.live = True
    profile = resolve_profile(args.profile, args.seed)
    keys = [args.top, args.bottom] if args.compare else [args.strategy]
    views = [View(k, profile, args.seed, args.width, args.height) for k in keys]
    fpt = max(1, int(round(FRAMES_PER_TICK / args.speed)))

    writer = None
    if args.video:
        # H.264 ('avc1') plays in QuickTime, VS Code and browsers; 'mp4v' often does not
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*'avc1'), FPS, (args.width * len(views), args.height))

    def emit(frames):
        img = np.concatenate(frames, axis=1)
        if writer is not None:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if args.live:
            cv2.imshow('Cobuild 3D', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(int(1000 / FPS)) & 0xFF
            if key == ord('q'):
                sys.exit(0)
            while key == ord(' '):
                key = cv2.waitKey(100) & 0xFF
        return img

    tick = 0
    while not all(v.done for v in views):
        pairs = [v.step() for v in views]
        tick += 1
        for f in range(1, fpt + 1):
            img = emit([v.frame(a, b, f / fpt) for v, (a, b) in zip(views, pairs)])
        if args.snapshot and tick >= args.at_tick:
            Image.fromarray(img).save(args.snapshot)
            print(f'saved {args.snapshot}')
            return
    for _ in range(FPS * 2):      # hold the last frame for two seconds
        img = emit([v.frame(v.pose, v.pose, 1.0) for v in views])
    if args.snapshot:
        Image.fromarray(img).save(args.snapshot)
        print(f'saved {args.snapshot}')
    if writer is not None:
        writer.release()
        print(f'saved {args.video}: ' + ', '.join(f'{v.label} {v.env.time * C.TICK_SECONDS // 60:.0f}:{v.env.time * C.TICK_SECONDS % 60:02.0f}'
                                                   for v in views))
    if args.live:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
