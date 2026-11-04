"""Every tunable number of the Cobuild drywall-installation simulation (v2).

Sources are in docs/RESEARCH.md. Tags used below:
    [DATA]    calibrated to a published field measurement
    [LIT]     direction or size supported by the literature
    [ASSUME]  our own assumption: nobody has measured it. Change it, then re-run
              analysis.py to see how much the conclusions depend on it.
"""

# --- Time -------------------------------------------------------------------------
TICK_SECONDS = 6          # [DATA] Yuan et al. use a 6 s unit; one tick = 6 s of real work
MAX_TICKS = 900           # episode is truncated after 90 minutes (never reached by sane policies)

# --- The wall -----------------------------------------------------------------------
# 4.88 m x 2.44 m wall, 4 boards of 1.22 x 2.44 m laid horizontally. The upper row (boards 0, 1)
# goes first. The right-hand board of each row is trimmed to fit (needs cutting), matching the
# "50 % whole / 50 % partial sheets" mix observed by Yuan et al.
N_BOARDS = 4
UPPER_ROW_BOARDS = (0, 1)
CUT_BOARDS = (1, 3)
WALL_TOP_M = 2.44
BOARD_KG = 34.0           # [LIT] 5/8" 4x8 sheet: 31-34 kg in tables, ~42 kg implied by Yuan's density
CUT_FRACTION = 0.75       # [ASSUME] a trimmed board weighs 75 % of a whole one
STUD_SPACING_M = 0.406    # [LIT] 16 in
SCREWS_PER_BOARD = 28     # [LIT] ~16 in field / 8 in edge spacing

# --- Worker durations, in ticks, for the AVERAGE worker working alone ----------------------
# Calibrated (see tests/test_calibration.py) so that the average worker alone needs ~4.7 min per board
# (Yuan: ~4.8 min per piece from 100 pieces per 8 h; Dasgupta: 4 min 30 s per panel for a pair) and the
# activity shares are close to Yuan's Table 1: cut/measure 20 %, lift+carry 19 %, hold/place 5 %,
# screw 16 %, reload ('in between') 19 %, other 21 %. [DATA]
BASE = {
    'prep': 7.0,          # "other" (21 % of the time in Yuan): mark, communicate, climb, move things
    'tools_fetch': 2.0,   # fetching the tape and cutter yourself
    'measure': 7.0,
    'cut': 12.0,
    'lift': 3.0,          # lifting the board off the stack
    'carry_stack_wall': 5.0,
    'carry_stack_cut': 2.0,
    'carry_cut_wall': 3.0,
    'place': 2.5,         # position the board on the studs and hold it
    'screw_a': 4.0,       # first half of the screws (the board is only held up until these are in)
    'screw_b': 4.0,
    'reload': 4.5,        # load / adjust the screw gun ("in between", 19 % of the time)
    'stool': 3.0,         # climb a stool and come back down (upper row, short worker)
}
DURATION_NOISE_SIGMA = 0.15   # [ASSUME] lognormal variability of every phase

# --- Intrinsic strain per tick (relative "strain points", NOT a validated score) -----------
# Heavy phases are scaled by the load ratio (board kg / worker lifting capacity) and, for work
# on the upper row, by a height factor. [ASSUME] shape follows Yuan: lift/hold are heavy.
STRAIN_LIFT = 1.6
STRAIN_CARRY = 0.9
STRAIN_PLACE = 1.8
STRAIN_TACK_HOLD = 0.9    # extra strain while holding the board during the first screws
STRAIN_SCREW = 0.25
STRAIN_SCREW_OVERHEAD_PER_M = 1.0  # per metre the worker falls short of the wall top
STRAIN_MEASURE, STRAIN_CUT, STRAIN_RELOAD, STRAIN_PREP, STRAIN_STOOL = 0.15, 0.3, 0.1, 0.1, 0.5

LIFT_LOAD_KNEE = 1.0               # [ASSUME] lifts slow down only once the board exceeds the worker's capacity
LIFT_LOAD_SLOWDOWN = 1.5
OVERHEAD_FACTOR_PER_M = 1.0        # [ASSUME] heavy work on the upper row: x(1 + 1 per metre of deficit)
SCREW_OVERHEAD_SLOWDOWN_PER_M = 1.0
STOOL_DEFICIT_M = 0.15             # a stool is needed on the upper row if the deficit exceeds this
FATIGUE_SCALE = 0.6                # multiplies every worker's fatigue_rate (0 = no fatigue); tuned so the average worker is ~5 % slower by the end of a wall

# Ergonomic anthropometry (Drillis & Contini-style proportions, fractions of stature)
COMFORT_HANDOVER_RATIO = 0.60
OVERHEAD_REACH_RATIO = 1.25

# --- Robot (one arm: it can do only one thing at a time) ----------------------------------
# [ASSUME] nobody has published durations for a robot doing these tasks on a site.
BRING_TICKS = {('stack', 'cutstation'): 4, ('stack', 'staged'): 5, ('cutstation', 'staged'): 4}
STEADY_SETUP_TICKS = 1
STEADY_CUT_TIME_FACTOR = 0.85      # [ASSUME] cutting a steadied board is 15 % faster
STEADY_CUT_STRAIN_FACTOR = 0.7     # [ASSUME] ... and 30 % less strain
LIFT_HOLD_SETUP_TICKS = 3          # lift the board from staging, position and align it
ALIGN_TICKS = 2.0                  # [ASSUME] the worker directs and checks the alignment of the held board
ALIGN_STRAIN = 0.1
HOLD_RELEASE_AFTER = 'screw_b'     # [LIT] Dasgupta: the board is supported until it is fastened ('screw_a' = release after half the screws)
FETCH_TOOL_TICKS = 2
NEAR_APPROACH_EXTRA_TICKS = 1      # approaching close to a person is slower
SWAP_TICKS = 1.5                   # swapping an empty screw gun for a loaded one

# Tool handover poses (side x height x distance = 12 poses)
HANDOVER_SIDES = ('left', 'right')
HANDOVER_HEIGHTS_M = (0.90, 1.05, 1.20)
HANDOVER_DISTANCES_M = (0.45, 0.65)
HEIGHT_TOLERANCE_M = 0.05
HEIGHT_MISMATCH_TICKS_THRESHOLD_M = 0.15
# extra worker time (ticks) / strain of a badly placed tool handover [LIT: direction; ASSUME: size]
WRONG_SIDE_TICKS, WRONG_SIDE_STRAIN = 0.5, 0.4
OUT_OF_REACH_TICKS, OUT_OF_REACH_STRAIN = 0.5, 0.6
HEIGHT_MISMATCH_TICKS = 0.5
HEIGHT_STRAIN_PER_10CM = 0.5
AWKWARD_STRAIN_THRESHOLD = 0.3

# --- Reward (used by PPO and by the planners to compare options) ---------------------------
R_TICK = -1.0
R_WORKER_WAIT = -1.0       # extra cost per tick the worker stands idle
R_STRAIN = -0.5            # per strain point: one point is "worth" half a tick
R_EXPOSURE = -0.1          # per tick the robot hovers with a tool nobody has taken yet
R_UNNECESSARY = -1.0       # command that makes no sense in the current state
R_IGNORED = -0.2           # command sent while the robot is busy
R_COMPLETE = 10.0
