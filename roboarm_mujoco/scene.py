"""
scene.py - builds the MuJoCo model (MJCF XML) of the arm and its workspace.

Coordinates
-----------
The layout is 2D: x right, y up, metres, origin on the floor below the robot.
In MuJoCo the arm moves in the x-z plane, so

    layout (x, y)  ->  MuJoCo (x, 0, z=y)

Every hinge axis is -y. With that axis a positive angle turns +x towards +z,
so a joint angle means exactly what it means in roboarm/robot.py (q1 from the
+x axis, q2 and q3 relative to the previous link).

Layouts
-------
A layout is a list of dicts with keys name, type, shape, x, y, width, height
(the format returned by roboarm_mujoco.environments.layout.load_layout).
default_layout() converts roboarm/config.py into that format.
"""
from xml.sax.saxutils import quoteattr

import numpy as np

from roboarm_mujoco import paths  # noqa: F401  (puts roboarm/ on sys.path)
from roboarm_mujoco import settings as S
import config as rcfg

ARM_JOINTS = ("joint1", "joint2", "joint3")
FINGER_JOINTS = ("finger_left_slide", "finger_right_slide")
OBJECT_JOINTS = ("object_x", "object_z", "object_rot")
LINK_GEOMS = ("link1", "link2", "link3")
GRIPPER_GEOMS = ("palm", "finger_left", "finger_right")
FINGER_GEOMS = ("finger_left", "finger_right")
ROBOT_GEOMS = LINK_GEOMS + GRIPPER_GEOMS
OBJECT_GEOM = "object"
TABLE_PREFIX = "table:"
OBSTACLE_PREFIX = "obstacle:"
VALID_TYPES = ("obstacle", "table", "box")
VALID_SHAPES = ("box", "circle")


class LayoutError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #
def _row(name, type_, shape, x0, y0, x1, y1):
    return {"name": name, "type": type_, "shape": shape, "x": float(x0 + x1) / 2, "y": float(y0 + y1) / 2,
            "width": float(x1 - x0), "height": float(y1 - y0)}


def default_layout():
    """roboarm/config.py's default scene in the agreed CSV row format.

    The CSV format only has rectangles and circles, so the trapezoid
    "Centre stalactite" becomes a rectangle with the trapezoid's average
    width (same height, same area). The box starts resting on Table A.
    """
    rows = []
    for name, st in rcfg.STATIONS.items():
        x0, x1 = st["table"]
        rows.append(_row(f"Table {name}", "table", "box", x0, 0.0, x1, st["top"]))
    for spec in rcfg.OBSTACLES:
        if "rect" in spec:
            rows.append(_row(spec["name"], "obstacle", "box", *spec["rect"]))
        else:
            pts = np.asarray(spec["poly"], float)
            y0, y1 = pts[:, 1].min(), pts[:, 1].max()
            x, y = pts[:, 0], pts[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))   # shoelace
            half = area / (y1 - y0) / 2                                              # average half-width
            cx = (x.max() + x.min()) / 2
            rows.append(_row(spec["name"], "obstacle", "box", cx - half, y0, cx + half, y1))
    w, h = rcfg.OBJECT_SIZE
    a = rcfg.STATIONS["A"]
    rows.append({"name": "Object", "type": "box", "shape": "box", "x": a["x"], "y": a["top"] + h / 2,
                 "width": w, "height": h})
    return rows


def validate_layout(rows):
    """Check the agreed rules; returns the rows with numbers as floats."""
    clean = []
    for i, r in enumerate(rows):
        missing = {"name", "type", "shape", "x", "y", "width", "height"} - set(r)
        if missing:
            raise LayoutError(f"row {i}: missing {sorted(missing)}")
        row = {"name": str(r["name"]).strip(), "type": str(r["type"]).strip().lower(),
               "shape": str(r["shape"]).strip().lower()}
        for k in ("x", "y", "width", "height"):
            row[k] = float(r[k])
        if row["type"] not in VALID_TYPES:
            raise LayoutError(f"row {i} ({row['name']}): type must be one of {VALID_TYPES}")
        if row["shape"] not in VALID_SHAPES:
            raise LayoutError(f"row {i} ({row['name']}): shape must be one of {VALID_SHAPES}")
        if row["width"] <= 0 or row["height"] <= 0:
            raise LayoutError(f"row {i} ({row['name']}): width and height must be positive")
        clean.append(row)
    boxes = [r for r in clean if r["type"] == "box"]
    if len(boxes) != 1:
        raise LayoutError(f"the layout needs exactly one row of type 'box', found {len(boxes)}")
    tables = [r for r in clean if r["type"] == "table"]
    if len(tables) < 2:
        raise LayoutError("the layout needs two tables ('Table A' and 'Table B')")
    return clean


def load_csv_layout(path):
    """Read an environment CSV with the team's shared reader (environments/layout.py)."""
    try:
        from roboarm_mujoco.environments.layout import load_layout
    except ImportError as exc:
        raise ImportError("roboarm_mujoco/environments/layout.py (load_layout) is needed to read CSV "
                          "layouts; it is maintained by the environments person.") from exc
    return validate_layout(load_layout(path))


def object_row(layout):
    return next(r for r in layout if r["type"] == "box")


def table_rows(layout):
    return [r for r in layout if r["type"] == "table"]


# --------------------------------------------------------------------------- #
# MJCF
# --------------------------------------------------------------------------- #
def _f(*v):
    return " ".join(f"{float(x):.12g}" for x in v)


def _rgba(name, alpha=1.0):
    r, g, b = rcfg.COLORS[name]
    return _f(r / 255, g / 255, b / 255, alpha)


def _shape_geom(name, row, depth, extra=""):
    """A layout row as a MuJoCo geom (box, or cylinder lying along y for circles)."""
    pos = _f(row["x"], 0, row["y"])
    if row["shape"] == "circle":
        return (f'<geom name={quoteattr(name)} type="cylinder" pos="{pos}" zaxis="0 1 0" '
                f'size="{_f(row["width"] / 2, depth / 2)}" {extra}/>')
    return (f'<geom name={quoteattr(name)} type="box" pos="{pos}" '
            f'size="{_f(row["width"] / 2, depth / 2, row["height"] / 2)}" {extra}/>')


def unique_names(layout):
    """Geom name for every row (prefix + CSV name, made unique)."""
    seen, out = {}, []
    for r in layout:
        prefix = {"table": TABLE_PREFIX, "obstacle": OBSTACLE_PREFIX, "box": ""}[r["type"]]
        base = OBJECT_GEOM if r["type"] == "box" else prefix + r["name"]
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(base if n == 0 else f"{base}#{n + 1}")
    return out


def build_xml(layout, robot_margin=0.0, object_margin=0.0):
    """MJCF of the scene. The margins are only used by the planner's collision
    checker (planning.py): a geom with margin m reports contacts whenever it is
    closer than m to something. They must be set here, before compiling;
    changing model.geom_margin afterwards is not seen by MuJoCo's broad phase.
    The physics simulation uses margin 0."""
    layout = validate_layout(layout)
    rm = f' margin="{_f(robot_margin)}"' if robot_margin else ""
    om = f' margin="{_f(object_margin)}"' if object_margin else ""
    L = rcfg.LINK_LENGTHS
    if len(L) != 3:
        raise ValueError("roboarm_mujoco models a 3-joint arm; config.LINK_LENGTHS must have 3 entries")
    R = rcfg.LINK_RADII
    lim = np.radians(rcfg.JOINT_LIMITS_DEG)
    bx, by = rcfg.ROBOT_BASE
    fo, fl, fr = rcfg.FINGER_OPEN, rcfg.FINGER_LENGTH, rcfg.FINGER_RADIUS
    finger_lo = S.FINGER_MIN_OFFSET - fo           # slide joint: 0 = open, negative = closing
    gc = ' gravcomp="1"' if S.GRAVITY_COMPENSATION else ""
    fric = f'friction="{_f(*S.FRICTION)}"'
    x_lo = min(r["x"] - r["width"] / 2 for r in layout) - 1.0
    x_hi = max(r["x"] + r["width"] / 2 for r in layout) + 1.0
    floor_half = max(abs(x_lo), abs(x_hi), 6.0)

    env = []
    for name, r in zip(unique_names(layout), layout):
        if r["type"] == "table":
            env.append(_shape_geom(name, r, S.OBSTACLE_DEPTH, f'rgba="{_rgba("table")}"'))
        elif r["type"] == "obstacle":
            env.append(_shape_geom(name, r, S.OBSTACLE_DEPTH, f'rgba="{_rgba("obstacle")}"'))
    obj = object_row(layout)
    obj_geom = _shape_geom(OBJECT_GEOM, {**obj, "x": 0.0, "y": 0.0}, S.OBJECT_DEPTH,
                           f'mass="{S.OBJECT_MASS}" rgba="{_rgba("object")}" {fric}{om}')
    ped_top = by - 0.05

    def link(i):
        return (f'<joint name="joint{i + 1}" type="hinge" axis="0 -1 0" range="{_f(*lim[i])}" '
                f'armature="{S.JOINT_ARMATURE}"/>\n'
                f'<geom name="link{i + 1}" type="capsule" fromto="{_f(0, 0, 0, L[i], 0, 0)}" size="{R[i]}" '
                f'mass="{S.LINK_MASSES[i]}" rgba="{_rgba("link")}"{rm}/>')

    def finger(side, sign):
        return f"""
            <body name="finger_{side}" pos="{_f(0, 0, sign * fo)}"{gc}>
              <joint name="finger_{side}_slide" type="slide" axis="{_f(0, 0, sign)}" range="{_f(finger_lo, 0)}"/>
              <geom name="finger_{side}" type="box" pos="{_f(fl / 2, 0, 0)}" size="{_f(fl / 2, S.FINGER_PAD_DEPTH / 2, fr)}"
                    mass="{S.FINGER_MASS}" rgba="{_rgba('gripper')}" {fric}{rm}/>
            </body>"""

    actuators = []
    for i, j in enumerate(ARM_JOINTS):
        actuators.append(f'<position name="{j}_pos" joint="{j}" kp="{S.JOINT_KP[i]}" ctrlrange="{_f(*lim[i])}"/>')
    for i, j in enumerate(ARM_JOINTS):
        actuators.append(f'<velocity name="{j}_vel" joint="{j}" kv="{S.JOINT_KV[i]}" '
                         f'ctrlrange="{_f(-S.MAX_JOINT_SPEED, S.MAX_JOINT_SPEED)}"/>')
    for j in FINGER_JOINTS:
        actuators.append(f'<position name="{j}_pos" joint="{j}" kp="{S.FINGER_KP}" kv="{S.FINGER_KV}" '
                         f'ctrlrange="{_f(finger_lo, 0)}" forcerange="{_f(-S.GRIP_FORCE, S.GRIP_FORCE)}"/>')

    nl = "\n    "
    return f"""<mujoco model="roboarm_mujoco">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{S.TIMESTEP}" gravity="0 0 {S.GRAVITY}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <global azimuth="90" elevation="-5" offwidth="1280" offheight="720"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/>
  </visual>
  <default>
    <geom solref="{_f(*S.CONTACT_SOLREF)}"/>
  </default>
  <statistic center="0 0 2.6" extent="6.5"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.78 0.70" rgb2="0.76 0.72 0.64" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 2"/>
  </asset>
  <worldbody>
    <light pos="0 -8 9" dir="0 0.7 -0.7" directional="true"/>
    <geom name="floor" type="plane" size="{_f(floor_half, 2, 0.1)}" material="grid"/>
    {nl.join(env)}
    <body name="pedestal">
      <geom name="pedestal" type="box" pos="{_f(bx, 0, ped_top / 2)}" size="{_f(0.38, 0.3, ped_top / 2)}" rgba="{_rgba('pedestal')}"/>
    </body>
    <body name="link1" pos="{_f(bx, 0, by)}"{gc}>
      {link(0)}
      <body name="link2" pos="{_f(L[0], 0, 0)}"{gc}>
        {link(1)}
        <body name="link3" pos="{_f(L[1], 0, 0)}"{gc}>
          {link(2)}
          <body name="gripper" pos="{_f(L[2], 0, 0)}"{gc}>
            <geom name="palm" type="capsule" fromto="{_f(0, 0, -fo, 0, 0, fo)}" size="{rcfg.PALM_RADIUS}"
                  mass="{S.PALM_MASS}" rgba="{_rgba('gripper')}"{rm}/>
            <site name="tcp" pos="{_f(rcfg.TCP_OFFSET, 0, 0)}" size="0.03" rgba="1 0.2 0.2 0.6"/>{finger("left", 1)}{finger("right", -1)}
          </body>
        </body>
      </body>
    </body>
    <body name="object">
      <joint name="object_x" type="slide" axis="1 0 0"/>
      <joint name="object_z" type="slide" axis="0 0 1"/>
      <joint name="object_rot" type="hinge" axis="0 -1 0"/>
      {obj_geom}
    </body>
  </worldbody>
  <contact>
    <exclude body1="pedestal" body2="link1"/>
    <exclude body1="link3" body2="finger_left"/>
    <exclude body1="link3" body2="finger_right"/>
    <exclude body1="finger_left" body2="finger_right"/>
  </contact>
  <equality>
    <!-- parallel gripper: the two fingers are mechanically coupled (they always
         move by the same amount). This acts only between the fingers; the box
         is held by friction alone. -->
    <joint name="finger_coupling" joint1="finger_left_slide" joint2="finger_right_slide"/>
  </equality>
  <actuator>
    {nl.join(actuators)}
  </actuator>
</mujoco>
"""
