"""Geometric audit of the 3D choreography: where do things pass through each other?

    python audit_3d.py                                   # every strategy, 3 personas, summary
    python audit_3d.py --profile short_left --strategy C_cobuild_rule --verbose

Runs the same episodes as visualize_3d.py, applies every interpolated frame to the MuJoCo scene (no
rendering) and measures the signed distance between the drawn shapes with mj_geomDistance:

    floor        any robot part, board, tool or worker part below the floor
    worker-robot a worker's body/arms intersect any part of the robot (hands may touch the gripper/tool)
    worker-board the worker's torso, legs or head intersect a board
    robot-board  the robot (except its suction frame, which holds the board) intersects a board
    robot-static the robot intersects the wall, pallet, cart or tool table
    board-static a board intersects the pallet, cart, wall framing or the other boards

These are checks on the DRAWING (the environment itself has no geometry, see docs/MODEL.md). A depth is
the worst overlap in metres; small overlaps (under TOL) are contact, not penetration.
"""
import argparse
from collections import defaultdict

import visualize_3d as V
from collisions import Checker as Audit, TOL
from profiles import PERSONAS
from strategies import STRATEGY_KEYS

def run(strategy_key, profile, seed=1000, fpt=V.FRAMES_PER_TICK, verbose=False):
    view = V.View(strategy_key, profile, seed, 64, 48)
    au = Audit(view.scene)
    stats = defaultdict(lambda: [0, 0.0])         # (kind, a, b) -> [frames, worst depth]
    frames = 0
    while not view.done:
        a, b = view.step()
        for f in range(1, fpt + 1):
            pose = view.compose(a, b, f / fpt)            # exactly what the video shows
            frames += 1
            seen = set()
            for kind, g1, g2, depth in au.check(pose):
                key = (kind, g1, g2)
                stats[key][1] = max(stats[key][1], depth)
                if key not in seen:
                    stats[key][0] += 1
                    seen.add(key)
    return frames, stats


def summarise(stats):
    by_kind = defaultdict(lambda: [0, 0.0])
    for (kind, *_), (n, depth) in stats.items():
        by_kind[kind][0] += n
        by_kind[kind][1] = max(by_kind[kind][1], depth)
    return by_kind


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', nargs='*', default=['short_left', 'expert_tall_right', 'low_capacity_short_left'])
    ap.add_argument('--strategy', nargs='*', default=['no_robot', 'B_proactive_generic', 'C_cobuild_rule'])
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--verbose', action='store_true', help='list every pair of shapes')
    args = ap.parse_args()
    total = defaultdict(lambda: [0, 0.0])
    for p in args.profile:
        for k in args.strategy:
            assert k in STRATEGY_KEYS
            frames, stats = run(k, PERSONAS[p], args.seed)
            kinds = summarise(stats)
            line = ', '.join(f'{kind} {n} (worst {d:.2f} m)' for kind, (n, d) in sorted(kinds.items())) or 'none'
            print(f'{p:26s} {k:22s} {frames:5d} frames | {line}')
            for kind, (n, d) in kinds.items():
                total[kind][0] += n
                total[kind][1] = max(total[kind][1], d)
            if args.verbose:
                for (kind, g1, g2), (n, depth) in sorted(stats.items(), key=lambda kv: -kv[1][1])[:12]:
                    print(f'      {kind:13s} {g1:12s} x {g2:12s} {n:4d} frames, worst {depth:.2f} m')
    print('\nTOTAL pair-events:', {k: f'{n} (worst {d:.2f} m)' for k, (n, d) in total.items()} or 'none')
