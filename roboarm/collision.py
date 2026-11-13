"""
collision.py - decides whether a robot configuration (or a whole motion) is
collision-free.

Model
-----
* Every robot part (links, palm, fingers) is a "capsule": a segment with a
  radius. A carried object is its 4 edges (radius 0).
* Every obstacle is a convex polygon.
* A capsule collides with a polygon if the distance between its centre
  segment and the polygon is smaller than its radius (+ a safety margin).
  The segment-polygon distance is 0 if the segment crosses an edge or has an
  endpoint inside the polygon; otherwise it is the smallest segment-edge
  distance.
* Self-collision: parts that are not directly connected must not overlap.
* A motion (edge) between two configurations is checked by testing many
  intermediate configurations, spaced so that no point of the robot moves more
  than COLLISION_STEP metres between two checks.
"""
import numpy as np

import config as cfg
from geometry import closest_points_segments, cross2, segment_segment_distance


class CollisionChecker:
    def __init__(self, robot, env, held=None, gripper_closed=None,
                 include_free_object=True, margin=cfg.SAFETY_MARGIN, step=cfg.COLLISION_STEP):
        """
        held           : HeldObject carried by the gripper (or None)
        gripper_closed : finger opening used for collision (default: closed iff holding)
        include_free_object : the resting object counts as an obstacle
        """
        self.robot = robot
        self.held = held
        closed = (held is not None) if gripper_closed is None else gripper_closed
        self.finger_offset = robot.finger_closed if closed else robot.finger_open
        self.margin = margin
        self.step = step
        polys = env.collision_polygons(include_object=include_free_object and held is None)
        self.poly_names = [name for name, _ in polys]
        self.edge_a = np.concatenate([p for _, p in polys])
        self.edge_b = np.concatenate([np.roll(p, -1, axis=0) for _, p in polys])
        sizes = [len(p) for _, p in polys]
        self.poly_starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
        self.edge_poly = np.repeat(np.arange(len(polys)), sizes)
        self.pairs = robot.self_collision_pairs(held)
        self.part_names = robot.part_names(held)
        # (part, polygon) pairs that are never checked: link 1 is mounted on the pedestal
        self.ignore = np.zeros((len(self.part_names), len(polys)), bool)
        if "Pedestal" in self.poly_names:
            self.ignore[0, self.poly_names.index("Pedestal")] = True
        self.reach = robot.reach_radii(self.finger_offset, held)
        # Broad phase data: bounding box of every polygon, and its edges padded
        # to the same count (repeating the last edge does not change a minimum).
        self.poly_lo = np.array([p.min(axis=0) for _, p in polys])
        self.poly_hi = np.array([p.max(axis=0) for _, p in polys])
        kmax = max(sizes)
        idx = np.array([[start + min(k, n - 1) for k in range(kmax)]
                        for start, n in zip(self.poly_starts, sizes)])
        self.pad_a, self.pad_b = self.edge_a[idx], self.edge_b[idx]      # (P, Kmax, 2)
        self.checks = 0   # number of configurations tested (a planning cost metric)

    # ----------------------------------------------------------- distances
    def _parts(self, Q):
        return self.robot.collision_parts(Q, self.finger_offset, self.held)

    def _env_clearance(self, sa, sb, radii):
        """(B, S, P): clearance of every part from every obstacle polygon."""
        A, B = sa[:, :, None, :], sb[:, :, None, :]
        C, D = self.edge_a[None, None], self.edge_b[None, None]
        d = segment_segment_distance(A, B, C, D)                       # (B, S, E)
        d_poly = np.minimum.reduceat(d, self.poly_starts, axis=2)      # (B, S, P)
        edge_dir = (self.edge_b - self.edge_a)[None, None]
        inside_a = np.minimum.reduceat(cross2(edge_dir, A - C), self.poly_starts, axis=2) > 0
        inside_b = np.minimum.reduceat(cross2(edge_dir, B - C), self.poly_starts, axis=2) > 0
        d_poly = np.where(inside_a | inside_b, 0.0, d_poly)
        d_poly = np.where(self.ignore[None], np.inf, d_poly)
        return d_poly - radii[None, :, None]

    def _self_clearance(self, sa, sb, radii):
        """(B, pairs): clearance between non-adjacent robot parts."""
        if len(self.pairs) == 0:
            return np.full((sa.shape[0], 1), np.inf)
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d = segment_segment_distance(sa[:, i], sb[:, i], sa[:, j], sb[:, j])
        return d - radii[i] - radii[j]

    def clearance_batch(self, Q):
        """Smallest clearance (m) of each configuration; negative = penetration."""
        Q = np.atleast_2d(np.asarray(Q, float))
        sa, sb, radii = self._parts(Q)
        env = self._env_clearance(sa, sb, radii).reshape(len(Q), -1).min(axis=1)
        own = self._self_clearance(sa, sb, radii).min(axis=1)
        return np.minimum(env, own)

    def clearance(self, q):
        return float(self.clearance_batch(q)[0])

    def _collides_fast(self, sa, sb, radii):
        """(B,) bool. Same answer as clearance <= margin, but only computes exact
        distances for part/obstacle pairs whose bounding boxes are close
        ("broad phase" + "narrow phase", as in game physics engines)."""
        reach = (radii + self.margin)[None, :, None]
        lo = np.minimum(sa, sb) - reach                                  # (B, S, 2)
        hi = np.maximum(sa, sb) + reach
        near = np.all((lo[:, :, None] <= self.poly_hi[None, None]) &
                      (hi[:, :, None] >= self.poly_lo[None, None]), axis=-1)   # (B, S, P)
        near &= ~self.ignore[None]
        hit = np.zeros(sa.shape[0], bool)
        b, s, p = np.nonzero(near)
        if len(b):
            A, Bq = sa[b, s][:, None], sb[b, s][:, None]
            C, D = self.pad_a[p], self.pad_b[p]
            d = segment_segment_distance(A, Bq, C, D).min(axis=1)
            edge_dir = D - C
            inside = (cross2(edge_dir, A - C).min(axis=1) > 0) | (cross2(edge_dir, Bq - C).min(axis=1) > 0)
            coll = inside | (d - radii[s] <= self.margin)
            hit[b[coll]] = True
        return hit | (self._self_clearance(sa, sb, radii).min(axis=1) <= self.margin)

    # ------------------------------------------------------------- validity
    def valid_batch(self, Q):
        Q = np.atleast_2d(np.asarray(Q, float))
        self.checks += len(Q)
        ok = self.robot.within_limits(Q)
        if not ok.any():
            return ok
        sa, sb, radii = self._parts(Q)
        return ok & ~self._collides_fast(sa, sb, radii)

    def is_valid(self, q):
        return bool(self.valid_batch(q)[0])

    def interpolate(self, qa, qb, include_start=False):
        """Intermediate configurations of the straight joint-space edge qa -> qb."""
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        displacement = float(np.sum(np.abs(qb - qa) * self.reach))   # upper bound (m)
        n = max(1, int(np.ceil(displacement / self.step)))
        t = np.linspace(0.0, 1.0, n + 1)[0 if include_start else 1:, None]
        return qa + t * (qb - qa)

    def edge_valid(self, qa, qb):
        """True if every intermediate configuration from qa to qb (including qb) is valid.

        Checked coarse-to-fine: first the end point and every 8th sample, so most
        blocked edges are rejected after a few checks.
        """
        Q = self.interpolate(qa, qb)
        if len(Q) > 8:
            coarse = np.zeros(len(Q), bool)
            coarse[7::8] = True
            coarse[-1] = True
            if not np.all(self.valid_batch(Q[coarse])):
                return False
            Q = Q[~coarse]
        return bool(np.all(self.valid_batch(Q)))

    def path_valid(self, path):
        return all(self.edge_valid(a, b) for a, b in zip(path[:-1], path[1:])) and self.is_valid(path[0])

    # -------------------------------------------------------- diagnostics
    def report(self, q):
        """Details for one configuration: colliding parts and the closest obstacle.

        Returns dict with
          'clearance'  : minimum clearance (m)
          'colliding'  : set of part indices in collision
          'pairs'      : list of human-readable collision descriptions
          'closest'    : (part name, obstacle name, point on robot, point on obstacle, distance)
        """
        q = np.asarray(q, float)
        sa, sb, radii = self._parts(q[None])
        env = self._env_clearance(sa, sb, radii)[0]          # (S, P)
        own = self._self_clearance(sa, sb, radii)[0]
        colliding, pairs = set(), []
        for s, p in zip(*np.nonzero(env <= self.margin)):
            colliding.add(int(s))
            pairs.append(f"{self.part_names[s]} hits {self.poly_names[p]}")
        for k in np.nonzero(own <= self.margin)[0] if len(self.pairs) else []:
            i, j = self.pairs[k]
            colliding.update((int(i), int(j)))
            pairs.append(f"{self.part_names[i]} hits {self.part_names[j]}")
        if not self.robot.within_limits(q):
            pairs.append("joint limit exceeded")
        # closest part / obstacle pair, with the actual closest points for drawing
        s, p = np.unravel_index(np.argmin(env), env.shape)
        best = None
        for e in np.nonzero(self.edge_poly == p)[0]:
            dist, pr, po = closest_points_segments(sa[0, s], sb[0, s], self.edge_a[e], self.edge_b[e])
            if best is None or dist < best[0]:
                best = (dist, pr, po)
        dist, pr, po = best
        direction = po - pr
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            pr = pr + direction / norm * min(radii[s], norm)   # move to the capsule surface
        closest = (self.part_names[s], self.poly_names[p], pr, po, float(env[s, p]))
        clearance = float(min(env.min(), own.min()))
        return {"clearance": clearance, "colliding": colliding, "pairs": pairs, "closest": closest}
