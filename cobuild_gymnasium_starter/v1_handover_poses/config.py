"""Every tunable number of the Cobuild drywall simulation, in one place.

All values are ILLUSTRATIVE ASSUMPTIONS chosen to be plausible, not measured
data. They are collected here so the team can replace them with site
observations or literature values without touching the simulation logic.
See docs/MODEL.md for the reasoning behind each group.
"""

# --- Time ---------------------------------------------------------------------
TICK_SECONDS = 15          # one simulation tick ≈ 15 s of real work (only used to print minutes)
MAX_TICKS = 300            # episode is truncated after this many ticks (never reached by sane policies)

# --- The wall ---------------------------------------------------------------------
# A 2.4 m high wall covered by 4 standard boards (1.2 m x 2.4 m, laid horizontally):
# two on the upper row (installed first, as is common practice), two on the lower row.
N_BOARDS = 4
UPPER_ROW_BOARDS = (0, 1)   # board indices that sit on the upper row
WALL_TOP_M = 2.40           # the worker must push upper boards up to this height
BOARD_WEIGHT_KG = 26.0      # 12.5 mm gypsum board, 1.2 x 2.4 m ≈ 25-28 kg

# --- Installation procedure, per board ------------------------------------------
# step name -> (base duration in ticks for an average worker, item the step needs or None)
STEPS = ('measure', 'cut', 'lift', 'fasten')
BASE_TICKS = {'measure': 3, 'cut': 4, 'lift': 3, 'fasten': 8}
NEEDS = {'measure': None, 'cut': 'board', 'lift': None, 'fasten': 'screws'}
# baseline physical load of the work itself (strain points), independent of the robot
BASE_STRAIN = {'measure': 0.2, 'cut': 0.5, 'lift': 0.0, 'fasten': 1.0}
DURATION_NOISE_SIGMA = 0.15  # each step's duration is multiplied by lognormal(0, sigma)

# --- Robot ----------------------------------------------------------------------
FETCH_TICKS = {'board': 3, 'screws': 2}  # go to the material stack, pick, come back
NEAR_APPROACH_EXTRA_TICKS = 1            # slowing down to approach close to a person
SUPPORT_SETUP_TICKS = 1                  # moving into position to hold a board on the wall
SUPPORTED_LIFT_TICKS = 4                 # lift + position + tack while the robot holds the board

# Handover poses the robot can choose from (side x height x distance = 12 poses)
HANDOVER_SIDES = ('left', 'right')
HANDOVER_HEIGHTS_M = (0.90, 1.05, 1.20)  # low / mid / high
HANDOVER_DISTANCES_M = (0.45, 0.65)      # near / far (measured from the worker's body)

# --- Ergonomics (how the worker's body turns into time and strain) ------------------
# Anthropometric ratios (fractions of stature), approximations based on the
# Drillis & Contini segment proportions: elbow height ≈ 0.63 H, shoulder ≈ 0.82 H,
# arm length ≈ 0.44 H. A comfortable handover sits a bit below the elbow.
COMFORT_HANDOVER_RATIO = 0.60     # comfortable handover height = 0.60 x stature
OVERHEAD_REACH_RATIO = 1.25       # standing vertical grip reach ≈ (0.82 + 0.44) x stature

HEIGHT_TOLERANCE_M = 0.05         # mismatch below this costs nothing
HEIGHT_STRAIN_PER_10CM = 2.0      # strain points per 10 cm of mismatch beyond the tolerance
HEIGHT_TIME_THRESHOLD_M = 0.15    # mismatch above this also costs 1 tick (bend / stretch)
WRONG_SIDE_TICKS, WRONG_SIDE_STRAIN = 1, 1.0       # turn the torso / switch hands
OUT_OF_REACH_TICKS, OUT_OF_REACH_STRAIN = 1, 1.5   # step forward / lean
AWKWARD_STRAIN_THRESHOLD = 0.5   # a handover counts as 'awkward' if it costs a tick or more strain than this

LIFT_STRAIN_BASE = 3.0            # strain of an unsupported lift = 3 x load ratio x height factor
LIFT_LOAD_KNEE = 0.6              # lifts become slower once board weight > 60 % of capacity
LIFT_LOAD_SLOWDOWN = 1.5
OVERHEAD_SLOWDOWN_PER_M = 2.0     # per metre the worker falls short of the wall top
FASTEN_OVERHEAD_SLOWDOWN_PER_M = 1.0
FASTEN_OVERHEAD_STRAIN_PER_M = 2.0
SUPPORTED_LIFT_STRAIN = 0.5
FATIGUE_SCALE = 1.0               # multiplies every worker's fatigue_rate (0 = no fatigue; sensitivity analysis)

# --- Reward weights (used by PPO, and by the rule-based planner to compare options) -----
R_TICK = -1.0               # every tick of work costs 1: minimise total time
R_WORKER_WAIT = -1.0        # extra cost when the worker is standing idle
R_STRAIN = -0.5             # per strain point: 1 strain point "is worth" half a tick
R_EXPOSURE = -0.1           # per tick the robot hovers in the worker's space holding an item
R_UNNECESSARY = -1.0        # delivery requested when nothing is left to deliver / invalid support
R_IGNORED = -0.2            # command sent while the robot is busy (it is ignored)
R_COMPLETE = 10.0           # bonus for finishing the wall
