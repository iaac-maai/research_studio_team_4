"""
geometry.py - small 2D geometry toolbox used by the collision checker.

Everything is vectorised with NumPy: the functions accept arrays of points /
segments of any (broadcastable) shape whose last axis has size 2. Checking
thousands of segment pairs in one call is what makes Python fast enough for
sampling-based planning.
"""
import numpy as np


def cross2(u, v):
    """z-component of the 3D cross product of two 2D vectors (signed area)."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def point_segment_distance(p, a, b):
    """Distance from point(s) p to segment(s) [a, b]."""
    ab = b - a
    ap = p - a
    denom = np.sum(ab * ab, axis=-1)
    t = np.sum(ap * ab, axis=-1) / np.maximum(denom, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1)


def segments_intersect(a, b, c, d):
    """True where segment [a, b] properly crosses segment [c, d]."""
    d1 = cross2(b - a, c - a)
    d2 = cross2(b - a, d - a)
    d3 = cross2(d - c, a - c)
    d4 = cross2(d - c, b - c)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def segment_segment_distance(a, b, c, d):
    """Shortest distance between segments [a, b] and [c, d] (0 if they cross).

    In 2D, two segments that do not cross are closest at one of their four
    endpoints, so the distance is the smallest of four point-segment distances.
    (Touching / collinear cases also give 0 through those four distances.)
    """
    dist = np.minimum(
        np.minimum(point_segment_distance(a, c, d), point_segment_distance(b, c, d)),
        np.minimum(point_segment_distance(c, a, b), point_segment_distance(d, a, b)),
    )
    return np.where(segments_intersect(a, b, c, d), 0.0, dist)


def closest_points_segments(a, b, c, d):
    """Scalar version returning (distance, point on [a,b], point on [c,d])."""
    a, b, c, d = (np.asarray(x, float) for x in (a, b, c, d))
    if segments_intersect(a, b, c, d):
        # intersection point
        r, s = b - a, d - c
        t = cross2(c - a, s) / cross2(r, s)
        p = a + t * r
        return 0.0, p, p
    best = None
    for p, (s0, s1), first in ((a, (c, d), True), (b, (c, d), True),
                               (c, (a, b), False), (d, (a, b), False)):
        seg = s1 - s0
        t = np.clip(np.dot(p - s0, seg) / max(np.dot(seg, seg), 1e-12), 0.0, 1.0)
        q = s0 + t * seg
        dist = float(np.linalg.norm(p - q))
        if best is None or dist < best[0]:
            best = (dist, p, q) if first else (dist, q, p)
    return best


def polygon_area(points):
    """Signed area (positive when the vertices are counter-clockwise)."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def ensure_ccw(points):
    points = np.asarray(points, float)
    return points if polygon_area(points) > 0 else points[::-1].copy()


def point_in_convex_polygon(p, poly):
    """True if point p is strictly inside convex CCW polygon `poly` (K, 2)."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    return bool(np.all(cross2(b - a, np.asarray(p, float) - a) > 0))


def rect_points(x0, y0, x1, y1):
    """Counter-clockwise rectangle corners."""
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])
