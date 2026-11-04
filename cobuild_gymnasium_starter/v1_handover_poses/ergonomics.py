"""How a worker's body turns into time and physical strain.

This is the "physics" of the Cobuild simulation: simple, explicit formulas
(no hidden learning) that both the environment and the rule-based planner
use. "Strain points" are a relative ergonomic-load unit for comparing
strategies, not a validated clinical score (a later version could map them
to REBA/RULA assessments).
"""
import itertools

import config as C

# All 12 handover poses as (side index, height in m, distance in m)
POSES = [(s, h, d) for s, h, d in itertools.product(range(len(C.HANDOVER_SIDES)),
                                                   C.HANDOVER_HEIGHTS_M,
                                                   C.HANDOVER_DISTANCES_M)]


def pose_label(pose_idx):
    side, h, d = POSES[pose_idx]
    hname = ['low', 'mid', 'high'][C.HANDOVER_HEIGHTS_M.index(h)]
    dname = ['near', 'far'][C.HANDOVER_DISTANCES_M.index(d)]
    return f'{C.HANDOVER_SIDES[side]}-{hname}-{dname}'


def is_upper(board):
    return board in C.UPPER_ROW_BOARDS


def overhead_deficit(profile, board):
    """How many metres the worker falls short of the top of an upper-row board (0 if lower row)."""
    if not is_upper(board):
        return 0.0
    return max(0.0, C.WALL_TOP_M - profile.overhead_reach_m)


def handover_cost(profile, pose, item):
    """Extra ticks and strain for the worker to take `item` from the robot at `pose`.
    Returns (ticks, strain, details), where details says which mismatches occurred."""
    side, height, dist = pose
    ticks, strain = 0, 0.0
    details = {'wrong_side': False, 'height_mismatch_m': 0.0, 'out_of_reach': False}
    if side != profile.side:
        ticks += C.WRONG_SIDE_TICKS
        strain += C.WRONG_SIDE_STRAIN
        details['wrong_side'] = True
    dh = abs(height - profile.comfort_handover_m)
    details['height_mismatch_m'] = dh
    if dh > C.HEIGHT_TOLERANCE_M:
        strain += (dh - C.HEIGHT_TOLERANCE_M) / 0.10 * C.HEIGHT_STRAIN_PER_10CM
    if dh > C.HEIGHT_TIME_THRESHOLD_M:
        ticks += 1
    if dist > profile.reach_m:
        ticks += C.OUT_OF_REACH_TICKS
        strain += C.OUT_OF_REACH_STRAIN
        details['out_of_reach'] = True
    if item == 'board':  # awkward postures hurt more when holding ~26 kg
        strain *= 1.0 + profile.load_ratio
    return ticks, strain, details


def is_awkward(ticks, strain):
    return ticks > 0 or strain > C.AWKWARD_STRAIN_THRESHOLD


def robot_approach_ticks(pose):
    """Robot travel overhead of a pose: approaching close to a person is slower."""
    return C.NEAR_APPROACH_EXTRA_TICKS if pose[2] == min(C.HANDOVER_DISTANCES_M) else 0


def lift_unsupported(profile, board):
    """(duration multiplier before pace/noise/fatigue, strain) of lifting a board alone."""
    load = profile.load_ratio
    load_factor = 1.0 + C.LIFT_LOAD_SLOWDOWN * max(0.0, load - C.LIFT_LOAD_KNEE)
    height_factor = 1.0 + C.OVERHEAD_SLOWDOWN_PER_M * overhead_deficit(profile, board)
    return C.BASE_TICKS['lift'] * load_factor * height_factor, C.LIFT_STRAIN_BASE * load * height_factor


def step_base(profile, board, step, supported=False):
    """Expected duration (ticks, before noise and fatigue) and intrinsic strain of a step."""
    if step == 'lift':
        if supported:
            return float(C.SUPPORTED_LIFT_TICKS), C.SUPPORTED_LIFT_STRAIN
        ticks, strain = lift_unsupported(profile, board)
        return ticks * profile.pace, strain
    ticks = C.BASE_TICKS[step] * profile.pace
    strain = C.BASE_STRAIN[step]
    if step == 'fasten':
        deficit = overhead_deficit(profile, board)
        ticks *= 1.0 + C.FASTEN_OVERHEAD_SLOWDOWN_PER_M * deficit
        strain += C.FASTEN_OVERHEAD_STRAIN_PER_M * deficit
    return ticks, strain


def pose_cost(profile, pose_idx, item):
    """Single number used to rank poses: worker ticks + robot ticks (half weight) + strain in tick units."""
    pose = POSES[pose_idx]
    ticks, strain, _ = handover_cost(profile, pose, item)
    return ticks + 0.5 * robot_approach_ticks(pose) + (-C.R_STRAIN) * strain


def best_pose(profile, item):
    return min(range(len(POSES)), key=lambda i: pose_cost(profile, i, item))


def support_is_worth_it(profile, board):
    """Should the robot hold this board during the lift?

    Holding saves lift time and strain, but the robot is busy, so the screws
    for that board can only be fetched after the lift (the worker then waits
    for them). Compares both options in reward units (time + weighted strain).
    """
    alone_ticks, alone_strain = step_base(profile, board, 'lift')
    held_ticks, held_strain = step_base(profile, board, 'lift', supported=True)
    pose = POSES[best_pose(profile, 'screws')]
    screw_delay = C.FETCH_TICKS['screws'] + robot_approach_ticks(pose)
    w = -C.R_STRAIN
    wait_cost_per_tick = -C.R_TICK - C.R_WORKER_WAIT  # a waiting tick costs time AND an idle penalty
    cost_alone = alone_ticks + w * alone_strain
    cost_held = held_ticks + wait_cost_per_tick * screw_delay + w * held_strain
    return cost_held < cost_alone
