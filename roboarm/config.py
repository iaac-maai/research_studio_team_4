"""
config.py - every adjustable parameter of the simulator, in one place.

Units: lengths in metres, angles in degrees here (converted to radians in code),
time in seconds. The world frame has its origin on the floor below the robot,
x pointing right and y pointing UP (the screen conversion is done in render.py).
"""

# --------------------------------------------------------------------------- #
# Window and view
# --------------------------------------------------------------------------- #
WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 780
PANEL_WIDTH = 340            # control panel on the right
TOP_BAR_HEIGHT = 40          # read-out strip above the world view
FPS = 60
VIEW_X_RANGE = (-4.8, 4.8)   # metres visible horizontally (sets the zoom)
VIEW_Y_BOTTOM = -0.45        # world y shown at the bottom edge of the view

# --------------------------------------------------------------------------- #
# Robot (a planar serial arm; add entries to the three lists for more joints)
# --------------------------------------------------------------------------- #
ROBOT_BASE = (0.0, 0.45)                 # shoulder joint position
LINK_LENGTHS = [2.1, 1.8, 0.9]           # one entry per link
LINK_RADII = [0.12, 0.10, 0.075]         # links are "capsules" (thick segments)
# Joint 1 is measured from the +x axis; every other joint relative to the previous link.
JOINT_LIMITS_DEG = [(0.0, 180.0), (-150.0, 150.0), (-150.0, 150.0)]
HOME_CONFIG_DEG = [95.0, -55.0, -125.0]  # "parked" pose; must be collision-free
LINK_LENGTH_SLIDER_RANGE = (0.6, 2.6)

# Gripper (attached to the end of the last link = the "flange")
TCP_OFFSET = 0.26        # flange -> tool centre point (where the object is held)
FINGER_LENGTH = 0.36
FINGER_RADIUS = 0.03
PALM_RADIUS = 0.04
FINGER_OPEN = 0.25       # distance from gripper axis to each finger when open
FINGER_CLEARANCE = 0.005 # gap between finger and object when closed

# --------------------------------------------------------------------------- #
# Object and the two stations (A = pick, B = place; the task can run both ways)
# --------------------------------------------------------------------------- #
OBJECT_SIZE = (0.30, 0.30)   # width, height of the box
OBJECT_REST_GAP = 0.02       # tiny gap between object and table (avoids "touching = collision")
STATIONS = {
    "A": {"x": -2.9, "table": (-3.45, -2.35), "top": 0.90},
    "B": {"x": 3.0, "table": (2.45, 3.55), "top": 1.10},
}
GRASP_ANGLE_DEG = -90.0      # gripper points straight down when picking / placing
APPROACH_HEIGHT = 0.35       # vertical approach / retreat distance above the object
CARTESIAN_STEP = 0.02        # resolution of the straight-line approach (m)

# --------------------------------------------------------------------------- #
# Environment (default layout). Coordinates in metres.
# kind: "hanging" (from the ceiling) / "rising" (from the floor) / "free"
# Every obstacle listed here can be dragged with the mouse.
# --------------------------------------------------------------------------- #
CEILING_Y = 6.0
OBSTACLES = [
    {"name": "Left pillar", "kind": "rising", "rect": (-1.95, 0.0, -1.55, 1.75)},
    {"name": "Left beam", "kind": "hanging", "rect": (-2.35, 2.95, -1.35, CEILING_Y)},
    {"name": "Right pillar", "kind": "rising", "rect": (1.45, 0.0, 1.85, 1.65)},
    {"name": "Right beam", "kind": "hanging", "rect": (1.55, 3.10, 2.45, CEILING_Y)},
    {"name": "Centre stalactite", "kind": "hanging",
     "poly": [(-0.75, CEILING_Y), (-0.25, 4.45), (0.25, 4.45), (0.75, CEILING_Y)]},
]
# Random "Regenerate obstacles" jitter (metres)
REGENERATE_JITTER_X = 0.45
REGENERATE_JITTER_Y = 0.45
REGENERATE_ATTEMPTS = 12

# --------------------------------------------------------------------------- #
# Motion planning
# --------------------------------------------------------------------------- #
PLANNER_BACKEND = "rrt_connect"   # "rrt_connect" (built-in) or "ompl" (optional)
OMPL_ALGORITHM = "RRTConnect"     # any OMPL geometric planner: RRTConnect, RRT, PRM, KPIECE1, BITstar...
OMPL_TIME_LIMIT = 10.0            # seconds per query (OMPL stops on time, not iterations)
RRT_STEP_SIZE = 0.30              # max joint-space distance of one tree extension (rad)
MAX_ITERATIONS = 15000            # samples per RRT-Connect query
ITERATION_SLIDER_RANGE = (200, 20000)
COLLISION_STEP = 0.03             # max Cartesian displacement between edge checks (m)
SAFETY_MARGIN = 0.01              # minimum clearance near the tables (approach / grasp) (m)
PLANNING_MARGIN = 0.05            # clearance required for free-space moves (transit / carry) (m)
SHORTCUT_ITERATIONS = 400         # path smoothing attempts (0 = keep raw RRT path)
IK_RESTARTS = 40                  # random seeds for the inverse-kinematics solver
RETURN_HOME = True                # plan a last move back to the home pose
RANDOM_SEED = None                # set an int for repeatable plans

# --------------------------------------------------------------------------- #
# Animation
# --------------------------------------------------------------------------- #
JOINT_SPEED = 0.9            # rad/s along the joint-space path (x the speed slider)
APPROACH_SPEED_FACTOR = 0.4  # approach / retreat moves are slower
GRIPPER_SPEED = 0.35         # m/s finger travel
SPEED_SLIDER_RANGE = (0.1, 4.0)
JOG_SPEED_DEG = 60.0         # manual jogging speed (deg/s, x3 with Shift)
GHOSTS_PER_MOVE = 7          # snapshots of the whole arm drawn along each free-space move

# --------------------------------------------------------------------------- #
# Colours (R, G, B)
# --------------------------------------------------------------------------- #
COLORS = {
    "background": (246, 242, 233),
    "grid": (230, 224, 210),
    "floor": (200, 188, 165),
    "floor_line": (150, 136, 112),
    "obstacle": (96, 112, 140),
    "obstacle_edge": (52, 62, 84),
    "obstacle_hover": (126, 146, 182),
    "boundary": (176, 166, 148),
    "table": (176, 132, 92),
    "table_edge": (110, 78, 50),
    "pedestal": (88, 88, 96),
    "link": (245, 152, 50),
    "link_edge": (120, 66, 18),
    "link_colliding": (226, 60, 60),
    "joint": (60, 60, 72),
    "joint_center": (240, 240, 240),
    "gripper": (80, 84, 98),
    "object": (80, 180, 110),
    "object_edge": (30, 100, 55),
    "station_a": (60, 140, 230),
    "station_b": (220, 80, 140),
    "path_tcp": (200, 40, 160),
    "ghost": (245, 152, 50),
    "tree_start": (70, 120, 220),
    "tree_goal": (230, 140, 40),
    "raw_path": (150, 150, 150),
    "clearance": (20, 170, 90),
    "collision": (226, 50, 50),
    "custom_goal": (140, 90, 200),
    "panel": (36, 40, 52),
    "panel_text": (225, 228, 235),
    "panel_dim": (140, 146, 160),
    "button": (60, 66, 84),
    "button_hover": (78, 86, 108),
    "button_on": (56, 128, 196),
    "accent": (245, 152, 50),
    "ok": (90, 200, 120),
    "warn": (240, 190, 70),
    "error": (240, 90, 90),
    "text": (40, 40, 48),
}
