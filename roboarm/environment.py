"""
environment.py - the workspace: obstacles, tables (stations A and B) and the
object to be transported.

All obstacles are CONVEX polygons (rectangles, trapezoids, ...). A concave
shape can be built from several convex pieces. The floor, ceiling, side walls,
tables and the robot's pedestal are fixed; the obstacles listed in
config.OBSTACLES can be dragged with the mouse or regenerated randomly.
"""
import copy
from dataclasses import dataclass, field

import numpy as np

import config as cfg
from geometry import ensure_ccw, point_in_convex_polygon, rect_points


@dataclass
class Obstacle:
    name: str
    points: np.ndarray            # (K, 2) counter-clockwise
    kind: str = "free"            # hanging / rising / free / table / boundary / pedestal
    movable: bool = True
    home_points: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.points = ensure_ccw(self.points)
        if self.home_points is None:
            self.home_points = self.points.copy()

    def contains(self, p):
        return point_in_convex_polygon(p, self.points)

    def translate(self, delta):
        self.points = self.points + np.asarray(delta, float)

    def bbox(self):
        return self.points.min(axis=0), self.points.max(axis=0)


@dataclass
class Station:
    name: str
    x: float
    table: tuple   # (x0, x1)
    top: float

    def object_rest_center(self, object_size):
        return np.array([self.x, self.top + cfg.OBJECT_REST_GAP + object_size[1] / 2])


class Environment:
    def __init__(self, obstacles, stations, object_size=cfg.OBJECT_SIZE):
        self.obstacles = obstacles
        self.stations = stations
        self.object_size = tuple(object_size)
        self.object_center = None
        self.object_angle = 0.0
        self.object_station = None   # "A", "B" or None while carried
        self.version = 0             # increases on every change (invalidates plans)
        self.place_object_at("A")

    # ------------------------------------------------------------- creation
    @classmethod
    def default(cls):
        x0, x1 = cfg.VIEW_X_RANGE
        obstacles = [
            Obstacle("Floor", rect_points(x0 - 2, -2.0, x1 + 2, 0.0), "boundary", False),
            Obstacle("Ceiling", rect_points(x0 - 2, cfg.CEILING_Y, x1 + 2, cfg.CEILING_Y + 3), "boundary", False),
            Obstacle("Left wall", rect_points(x0 - 2, -2.0, x0 + 0.12, cfg.CEILING_Y + 3), "boundary", False),
            Obstacle("Right wall", rect_points(x1 - 0.12, -2.0, x1 + 2, cfg.CEILING_Y + 3), "boundary", False),
            # The robot's mount. Link 1 is bolted onto it, so the collision
            # checker ignores the pedestal for link 1 only (see collision.py).
            Obstacle("Pedestal", rect_points(-0.38, 0.0, 0.38, cfg.ROBOT_BASE[1] - 0.05), "pedestal", False),
        ]
        stations = {}
        for name, s in cfg.STATIONS.items():
            stations[name] = Station(name, s["x"], tuple(s["table"]), s["top"])
            obstacles.append(Obstacle(f"Table {name}", rect_points(s["table"][0], 0.0, s["table"][1], s["top"]),
                                      "table", False))
        for spec in cfg.OBSTACLES:
            pts = rect_points(*spec["rect"]) if "rect" in spec else np.array(spec["poly"], float)
            obstacles.append(Obstacle(spec["name"], pts, spec.get("kind", "free"), True))
        return cls(obstacles, stations)

    def copy(self):
        return copy.deepcopy(self)

    # -------------------------------------------------------------- object
    def place_object_at(self, station_name):
        st = self.stations[station_name]
        self.object_center = st.object_rest_center(self.object_size)
        self.object_angle = 0.0
        self.object_station = station_name
        self.version += 1

    def set_object_pose(self, center, angle, station=None):
        self.object_center = np.asarray(center, float)
        self.object_angle = float(angle)
        self.object_station = station
        self.version += 1

    def object_half_size(self):
        return (self.object_size[0] / 2, self.object_size[1] / 2)

    def object_corners(self):
        from robot import PlanarArm
        return PlanarArm.box_corners(self.object_center, self.object_angle, self.object_half_size())

    # ------------------------------------------------------- collision data
    def collision_polygons(self, include_object=True):
        """List of (name, CCW polygon) used by the collision checker."""
        polys = [(o.name, o.points) for o in self.obstacles]
        if include_object and self.object_station is not None:
            polys.append(("object", self.object_corners()))
        return polys

    # ---------------------------------------------------------- interaction
    def movable_obstacle_at(self, p):
        for i in reversed(range(len(self.obstacles))):
            o = self.obstacles[i]
            if o.movable and o.contains(p):
                return i
        return None

    def move_obstacle(self, index, delta):
        o = self.obstacles[index]
        lo, hi = o.bbox()
        x0, x1 = cfg.VIEW_X_RANGE
        d = np.asarray(delta, float).copy()
        # keep at least part of the obstacle inside the view
        d[0] = np.clip(d[0], x0 + 0.2 - hi[0], x1 - 0.2 - lo[0])
        d[1] = np.clip(d[1], -lo[1] - 1.5 if o.kind == "rising" else -lo[1],
                       cfg.CEILING_Y - lo[1] - 0.2)
        o.translate(d)
        self.version += 1

    def protected_zones(self):
        """Rectangles that random obstacles must stay out of (stations, pedestal)."""
        zones = [rect_points(-0.7, 0.0, 0.7, 1.2)]
        for st in self.stations.values():
            zones.append(rect_points(st.table[0] - 0.05, 0.0, st.table[1] + 0.05,
                                     st.top + self.object_size[1] + cfg.APPROACH_HEIGHT + 0.5))
        return zones

    def randomize(self, rng):
        """Jitter every movable obstacle around its default position."""
        zones = self.protected_zones()
        for o in self.obstacles:
            if not o.movable:
                continue
            for _ in range(50):
                d = np.array([rng.uniform(-1, 1) * cfg.REGENERATE_JITTER_X,
                              rng.uniform(-1, 1) * cfg.REGENERATE_JITTER_Y])
                pts = o.home_points + d
                lo, hi = pts.min(0), pts.max(0)
                if not any(lo[0] < z[:, 0].max() and hi[0] > z[:, 0].min() and
                           lo[1] < z[:, 1].max() and hi[1] > z[:, 1].min() for z in zones):
                    o.points = pts
                    break
        self.version += 1

    def restore_default_obstacles(self):
        for o in self.obstacles:
            o.points = o.home_points.copy()
        self.version += 1
