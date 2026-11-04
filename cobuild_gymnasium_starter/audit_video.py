"""Frame-by-frame check of the three demo videos for the two things that must never happen:

    1. ARM / ROBOT INTO THE WORKER  any robot part overlaps any part of the worker's body, arms or hands
                                    (only a hand touching the gripper, to take a tool, is allowed)
    2. BOARD MOVING ON ITS OWN      a board moves while nobody holds it (the robot's suction frame or both of the
                                    worker's hands), or a board hangs in the air without anything under it

    python audit_video.py              # the three videos of the README, both panels each
    python audit_video.py --verbose    # list every event with its time in the video (mm:ss)

Runs exactly the frames the video shows (View.compose) and reports times in the same clock as the .mp4
(20 frames per second, 8 frames per tick), so an event at 0:41 is what you see at 0:41.
"""
import argparse
from collections import defaultdict

import numpy as np

import visualize_3d as V
from collisions import ROBOT_HOLDERS, ROBOT_REST, WORKER_ARMS, WORKER_BODY, WORKER_HANDS
from profiles import PERSONAS

VIDEOS = [('B_vs_Cobuild_low_capacity_short_left', 'low_capacity_short_left', 'B_proactive_generic', 'C_cobuild_rule'),
          ('NoRobot_vs_CobuildPPO_short_left', 'short_left', 'no_robot', 'C_cobuild_ppo'),
          ('B_vs_Cobuild_expert_tall_right', 'expert_tall_right', 'B_proactive_generic', 'C_cobuild_rule')]
PENETRATION = 0.005      # metres of overlap that count as "the robot is inside the worker"
GRIP = 0.03              # a board this close to the suction frame is held by it
HAND = 0.06              # ... or this close to BOTH hands of the worker
SUPPORT = 0.03           # a still board this close to something solid is resting on it
MOVE = 0.002             # a board that moved more than this between two frames has moved
SOLID = ('floor', 'pallet', 'cart_top', 'backwall', 'plate_lo', 'plate_hi')    # what a resting board may lie on


def clock(frame):
    s = frame / V.FPS
    return f'{int(s // 60)}:{s % 60:04.1f}'


def audit(strategy, profile_name, seed=1000):
    """Events (frame, kind, detail) for one panel of a video."""
    view = V.View(strategy, PERSONAS[profile_name], seed, 64, 48)
    ck = view.checker
    robot = [g for g in (*ROBOT_REST, *ROBOT_HOLDERS) if g in ck.id]
    worker = [*WORKER_BODY, *WORKER_ARMS, *WORKER_HANDS]
    boards = [f'board{b}_g' for b in range(4)]
    studs = [n for n in ck.id if n.startswith('stud')]
    events, last, frame = [], {}, 0
    while not view.done:
        a, b = view.step()
        for f in range(1, V.FRAMES_PER_TICK + 1):
            frame += 1
            view.compose(a, b, f / V.FRAMES_PER_TICK)
            d = view.scene.data
            for w in worker:
                for r in robot:
                    if w in WORKER_HANDS and r == 'gripper_g':
                        continue
                    depth = -ck.dist(w, r)
                    if depth > PENETRATION:
                        events.append((frame, 'robot-into-worker', f'{r} in {w} by {depth * 100:.0f} cm'))
            for g in boards:
                pos, mat = d.geom_xpos[ck.id[g]].copy(), d.geom_xmat[ck.id[g]].copy()
                if pos[2] < -4:
                    continue
                moved = g in last and (np.linalg.norm(pos - last[g][0]) > MOVE or np.abs(mat - last[g][1]).max() > 0.01)
                last[g] = (pos, mat)
                by_robot = ck.dist(g, 'frame_g') < GRIP
                by_worker = ck.dist(g, 'hand_l_g') < HAND and ck.dist(g, 'hand_r_g') < HAND
                if moved and not (by_robot or by_worker):
                    events.append((frame, 'board-moves-alone', f'{g} (robot {view.env.robot_state}, '
                                                               f'worker {view.env.phase or view.env.worker_state})'))
                elif not moved and not (by_robot or by_worker):
                    gap = min(ck.dist(g, s) for s in (*SOLID, *studs, *[o for o in boards if o != g]))
                    if gap > SUPPORT:
                        events.append((frame, 'board-floating', f'{g} {gap * 100:.0f} cm above support'))
    return events


def summary(events):
    out = defaultdict(lambda: [0, None, None])
    for frame, kind, _ in events:
        s = out[kind]
        s[0] += 1
        s[1] = frame if s[1] is None else min(s[1], frame)
        s[2] = frame if s[2] is None else max(s[2], frame)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--only', help='audit only the video whose name contains this text')
    args = ap.parse_args()
    total = 0
    for name, prof, top, bottom in VIDEOS:
        if args.only and args.only not in name:
            continue
        for panel, strat in (('top', top), ('bottom', bottom)):
            ev = audit(strat, prof)
            total += len(ev)
            s = summary(ev)
            line = ', '.join(f'{k}: {n} frames ({clock(a)}-{clock(b)})' for k, (n, a, b) in s.items()) or 'clean'
            print(f'{name:40s} {panel:6s} {strat:22s} {line}')
            if args.verbose:
                seen = set()
                for frame, kind, det in ev:
                    key = (frame // 10, kind)            # one line per half second
                    if key not in seen:
                        seen.add(key)
                        print(f'      {clock(frame):>7s}  {kind:18s} {det}')
    print('\nTOTAL problem frames:', total)
