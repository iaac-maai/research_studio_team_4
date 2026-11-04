"""How a worker's body turns into time and strain for each drywall task.

Simple, explicit formulas (no learning) shared by the environment and the rule-based planners.
"Strain points" are a relative unit for comparing strategies, not a clinical score
(a later version could map postures to REBA/RULA). All constants live in config.py.
"""
import itertools

import config as C

# 12 tool-handover poses: (side index, height in m, distance in m)
POSES = [(s, h, d) for s, h, d in itertools.product(range(len(C.HANDOVER_SIDES)),
                                                   C.HANDOVER_HEIGHTS_M, C.HANDOVER_DISTANCES_M)]


def pose_label(i):
    side, h, d = POSES[i]
    return (f'{C.HANDOVER_SIDES[side]}-{["low", "mid", "high"][C.HANDOVER_HEIGHTS_M.index(h)]}-'
            f'{["near", "far"][C.HANDOVER_DISTANCES_M.index(d)]}')


# ------------------------------------------------------------------ the wall and the load
def is_upper(b):
    return b in C.UPPER_ROW_BOARDS


def needs_cut(b):
    return b in C.CUT_BOARDS


def board_kg(b):
    return C.BOARD_KG * (C.CUT_FRACTION if needs_cut(b) else 1.0)


def load_ratio(P, b):
    return board_kg(b) / P.lift_capacity_kg


def lift_slow(P, b):
    """Heavy boards slow every lifting phase down once they exceed ~60 % of the worker's capacity."""
    return 1.0 + C.LIFT_LOAD_SLOWDOWN * max(0.0, load_ratio(P, b) - C.LIFT_LOAD_KNEE)


def overhead_deficit(P, b):
    """Metres by which the worker's overhead reach falls short of the wall top (upper row only)."""
    return max(0.0, C.WALL_TOP_M - P.overhead_reach_m) if is_upper(b) else 0.0


def height_factor(P, b):
    return 1.0 + C.OVERHEAD_FACTOR_PER_M * overhead_deficit(P, b)


def needs_stool(P, b):
    return overhead_deficit(P, b) > C.STOOL_DEFICIT_M


# ------------------------------------------------------------------ phase durations and strain
def phase_cost(P, b, name, mode='alone'):
    """Expected (ticks, strain) of one phase for worker P on board b, before noise and fatigue.

    modes: place 'alone'|'aligned' · screw_a 'tack'|'free' · cut 'plain'|'steady' · reload 'alone'|'swap'
    """
    B, pace = C.BASE, P.pace
    lr, slow, hf, deficit = load_ratio(P, b), lift_slow(P, b), height_factor(P, b), overhead_deficit(P, b)
    if name == 'prep':
        t = B['prep'] * pace
        return t, C.STRAIN_PREP * t
    if name == 'tools_fetch':
        t = B['tools_fetch'] * pace
        return t, 0.05 * t
    if name == 'measure':
        t = B['measure'] * pace
        return t, C.STRAIN_MEASURE * t
    if name == 'cut':
        steady = mode == 'steady'
        t = B['cut'] * pace * (C.STEADY_CUT_TIME_FACTOR if steady else 1.0)
        return t, C.STRAIN_CUT * B['cut'] * pace * (C.STEADY_CUT_STRAIN_FACTOR if steady else 1.0)
    if name.startswith('haul'):  # worker lifts the board off the stack / cut station and walks with it
        carry = {'haul_stack_wall': 'carry_stack_wall', 'haul_stack_cut': 'carry_stack_cut',
                 'haul_cut_wall': 'carry_cut_wall'}[name]
        t1, t2 = B['lift'] * pace * slow, B[carry] * pace * slow
        return t1 + t2, C.STRAIN_LIFT * lr * t1 + C.STRAIN_CARRY * lr * t2
    if name == 'place':
        if mode == 'aligned':
            t = C.ALIGN_TICKS * pace
            return t, C.ALIGN_STRAIN * t
        t = B['place'] * pace * slow * hf
        return t, C.STRAIN_PLACE * lr * hf * t
    if name in ('screw_a', 'screw_b'):
        t = B[name] * pace * (1.0 + C.SCREW_OVERHEAD_SLOWDOWN_PER_M * deficit)
        s = (C.STRAIN_SCREW + C.STRAIN_SCREW_OVERHEAD_PER_M * deficit) * t
        if name == 'screw_a' and mode == 'tack':  # no robot: the worker holds the board with one hand
            s += C.STRAIN_TACK_HOLD * lr * hf * t
        return t, s
    if name == 'reload':
        t = (C.SWAP_TICKS if mode == 'swap' else B['reload']) * pace
        return t, C.STRAIN_RELOAD * t
    if name == 'stool':
        t = B['stool'] * pace
        return t, C.STRAIN_STOOL * t
    raise KeyError(name)


CATEGORY = {'prep': 'prep', 'tools_fetch': 'prep', 'stool': 'prep', 'measure': 'measure', 'cut': 'cut',
            'haul_stack_wall': 'haul', 'haul_stack_cut': 'haul', 'haul_cut_wall': 'haul', 'place': 'place',
            'screw_a': 'screw', 'screw_b': 'screw', 'reload_a': 'reload', 'reload_b': 'reload'}
CATEGORIES = ['prep', 'measure', 'cut', 'haul', 'place', 'screw', 'reload', 'waiting']


def is_heavy(name, mode):
    """Does the worker carry the board's weight during this phase?"""
    return name.startswith('haul') or (name == 'place' and mode != 'aligned') or (name == 'screw_a' and mode == 'tack')


def is_overhead(name, b):
    return is_upper(b) and name in ('place', 'screw_a', 'screw_b')


# ------------------------------------------------------------------ tool handovers
def handover_cost(P, pose_idx):
    """Extra worker ticks and strain when taking a tool at this pose. Returns (ticks, strain, details)."""
    side, height, dist = POSES[pose_idx]
    ticks, strain = 0.0, 0.0
    details = {'wrong_side': False, 'height_mismatch_m': 0.0, 'out_of_reach': False}
    if side != P.side:
        ticks += C.WRONG_SIDE_TICKS
        strain += C.WRONG_SIDE_STRAIN
        details['wrong_side'] = True
    dh = abs(height - P.comfort_handover_m)
    details['height_mismatch_m'] = dh
    if dh > C.HEIGHT_TOLERANCE_M:
        strain += (dh - C.HEIGHT_TOLERANCE_M) / 0.10 * C.HEIGHT_STRAIN_PER_10CM
    if dh > C.HEIGHT_MISMATCH_TICKS_THRESHOLD_M:
        ticks += C.HEIGHT_MISMATCH_TICKS
    if dist > P.reach_m:
        ticks += C.OUT_OF_REACH_TICKS
        strain += C.OUT_OF_REACH_STRAIN
        details['out_of_reach'] = True
    return ticks, strain, details


def is_awkward(ticks, strain):
    return ticks > 0 or strain > C.AWKWARD_STRAIN_THRESHOLD


def approach_ticks(pose_idx):
    return C.NEAR_APPROACH_EXTRA_TICKS if POSES[pose_idx][2] == min(C.HANDOVER_DISTANCES_M) else 0


def pose_cost(P, i):
    ticks, strain, _ = handover_cost(P, i)
    return ticks + 0.5 * approach_ticks(i) + (-C.R_STRAIN) * strain


def best_tool_pose(P):
    return min(range(len(POSES)), key=lambda i: pose_cost(P, i))


# ------------------------------------------------------------------ should the robot hold the board?
def hold_is_worth_it(P, b):
    """Compare, in reward units (ticks + weighted strain), the worker placing and holding the board alone
    against the robot holding it. While it holds the board the single arm cannot hand over the screw
    gun, so the worker reloads alone: once if the robot is released after the first screws, twice
    (reload_a and reload_b) if it holds until the board is fastened. That lost help is added to the cost."""
    w = -C.R_STRAIN
    t_p, s_p = phase_cost(P, b, 'place', 'alone')
    t_a, s_a = phase_cost(P, b, 'screw_a', 'tack')
    t_al, s_al = phase_cost(P, b, 'place', 'aligned')
    t_f, s_f = phase_cost(P, b, 'screw_a', 'free')
    cost_alone = (t_p + t_a) + w * (s_p + s_a)
    t_alone_rel, _ = phase_cost(P, b, 'reload', 'alone')
    t_swap_rel, _ = phase_cost(P, b, 'reload', 'swap')
    last = b == C.N_BOARDS - 1
    n_blocked = 1 if (C.HOLD_RELEASE_AFTER == 'screw_a' or last) else 2
    extra_screw = 0.0
    if C.HOLD_RELEASE_AFTER == 'screw_b':   # the robot also holds during the second half of the screws
        extra_screw = 0.0                   # (the worker works at the same speed; holding adds no time)
    cost_held = (t_al + t_f) + w * (s_al + s_f) + n_blocked * (t_alone_rel - t_swap_rel) + extra_screw
    return cost_held < cost_alone
