"""settings.py - parameters that only exist in the physics version.

Robot geometry (link lengths, radii, joint limits, base, gripper sizes, home
pose) and planner settings are NOT repeated here: they are read from
roboarm/config.py so both simulators describe the same robot.

Units: metres, kilograms, seconds, newtons, radians.
"""

# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
TIMESTEP = 0.002          # physics step (s)
CONTROL_DT = 0.02         # one RobotSim.step() = 10 physics steps
GRAVITY = -9.81

# --------------------------------------------------------------------------- #
# Third dimension. The arm moves in the x-z plane (CSV y = MuJoCo z); every
# shape still needs a thickness along MuJoCo y.
# --------------------------------------------------------------------------- #
OBSTACLE_DEPTH = 1.0      # tables and obstacles
OBJECT_DEPTH = 0.30       # the box
FINGER_PAD_DEPTH = 0.12   # finger pads

# --------------------------------------------------------------------------- #
# Masses. Chosen, not measured: a light-weight arm of this size.
# --------------------------------------------------------------------------- #
LINK_MASSES = [4.0, 3.0, 1.5]
PALM_MASS = 0.6
FINGER_MASS = 0.15
OBJECT_MASS = 1.0

# Arm bodies get MuJoCo's per-body gravity compensation (gravcomp="1"): the
# motors do not have to hold the arm's own weight, so they do not sag off target.
# The carried box is NOT compensated.
GRAVITY_COMPENSATION = True

# --------------------------------------------------------------------------- #
# Friction (MuJoCo: sliding, torsional, rolling). The grasp relies on this.
# --------------------------------------------------------------------------- #
FRICTION = (1.0, 0.005, 0.0001)

# Contact stiffness (MuJoCo solref: time constant s, damping ratio), applied to
# every geom. MuJoCo contacts are slightly soft; with its default (0.02 s) the
# box sinks ~1 mm into each finger, rocks a little at every arm motion and
# "walks" ~4 cm along the fingers over one carry. 0.004 s (= 2 x TIMESTEP, the
# stiffest value MuJoCo recommends) cuts that to ~4 mm. More grip force, more
# friction or the no-slip solver did not help (measured 2026-11-28).
CONTACT_SOLREF = (0.004, 1.0)

# --------------------------------------------------------------------------- #
# Joint motors: position servo (stiffness kp) + velocity servo (damping kv).
# step() feeds the velocity servo the speed implied by consecutive targets,
# so the arm follows a moving target without lagging behind it.
# kp, kv per joint; tuned with tests/test_control.py.
# --------------------------------------------------------------------------- #
JOINT_KP = [60000.0, 25000.0, 4000.0]
JOINT_KV = [4000.0, 1500.0, 160.0]
JOINT_ARMATURE = 0.1
MAX_JOINT_SPEED = 2.0     # rad/s, cap on the velocity reference

# --------------------------------------------------------------------------- #
# Gripper: each finger has a position servo whose force is capped at
# GRIP_FORCE. "Closed" commands the fingers inside the box, so they squeeze it
# with GRIP_FORCE each. Holding works only through friction:
#     holds if 2 * friction * GRIP_FORCE > box weight (+ inertial forces).
# --------------------------------------------------------------------------- #
GRIP_FORCE = 40.0         # N per finger
FINGER_KP = 2000.0
FINGER_KV = 40.0
FINGER_MIN_OFFSET = 0.04  # fully closed: finger centre 4 cm from the gripper axis
GRIPPER_SETTLE_TIME = 0.6 # s waited after opening / closing during a task

# --------------------------------------------------------------------------- #
# Task execution
# --------------------------------------------------------------------------- #
PLACE_DROP_GAP = 0.02     # box is released this far above the table (m)
