r"""File responsible for detecting collisions between triangles."""

import numpy as np

__all__ = [
    "is_point_inside_triangle",
    "intersection_line_segment",
    "intersect_line_segment_boundary_triangle"
]


def is_point_inside_triangle(t1, t2, t3, pt, interior=False):
    r"""
    Check if a point lies (on boundary and interior) inside the triangle.

    Parameters
    ----------
    t1 : ndarray(3,)
        Coordinate of vertex of triangle.
    t2 : ndarray(3,)
        Coordinate of vertex of triangle.
    t3 : ndarray(3,)
        Coordinate of vertex of triangle.
    pt : ndarray(3,)
        Coordinate of point for test.
    interior : bool
        If True, then only check if it's in the interior.

    Returns
    -------
    bool
        Returns true if point lies inside triangle.

    Notes
    -----
    Obtained from https://blackpawn.com/texts/pointinpoly/default.html
    """
    a = t3 - t1
    b = t2 - t1
    c = pt - t1

    dot00 = np.dot(a, a)
    dot01 = np.dot(a, b)
    dot02 = np.dot(a, c)
    dot11 = np.dot(b, b)
    dot12 = np.dot(b, c)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    if u > -1e-10 and v > -1e-10 and u + v < 1.0:
        return True
    return False


def intersection_line_segment(a0, a1, b0, b1):
    r"""
    Return the intersection between two line segments in three-dimensional space.

    Parameters
    ----------
    a0 : ndarray(3,)
        X, Y, Z coordinate of initial point on line segment (a0, a1).
    a1 : ndarray(3,)
        X, Y, Z coordinate of final point on line segment (a0, a1).
    b0 : ndarray(3,)
        X, Y, Z coordinate of initial point on line segment (a0, a1).
    b1 : ndarray(3,)
        X, Y, Z coordinate of final point on line segment (a0, a1)>

    Returns
    -------
    (a, b) :
        The intersection points such that a0 + a (a1 - a0) = b0 + b (b1 - b0).
        If either a or b is nan, then no intersection is possible.
        If both a or b are in-between 0 and 1, then the line segment intersects.

    Notes
    -----
    - Obtained from Paul Borke's notes: http://paulbourke.net/geometry/pointlineplane/

    """
    x1, y1, z1 = a0
    x2, y2, z2 = a1
    x3, y3, z3 = b0
    x4, y4, z4 = b1
    d1321 = (x1 - x3) * (x2 - x1) + (y1 - y3) * (y2 - y1) + (z1 - z3) * (z2 - z1)
    d2121 = (x2 - x1)**2.0 + (y2 - y1)**2.0 + (z2 - z1)**2.0

    d1343 = (x1 - x3) * (x4 - x3) + (y1 - y3) * (y4 - y3) + (z1 - z3) * (z4 - z3)
    d4321 = (x4 - x3) * (x2 - x1) + (y4 - y3) * (y2 - y1) + (z4 - z3) * (z2 - z1)
    d4343 = (x4 - x3) ** 2.0 + (y4 - y3) ** 2.0 + (z4 - z3) ** 2.0

    a = (d1343 * d4321 - d1321 * d4343) / (d2121 * d4343 - d4321 * d4321)
    b = (d1343 + a * d4321) / d4343
    return a, b


def intersect_line_segment_boundary_triangle(t1, t2, t3, a0, a1):
    r"""
    Check if line segment (a0, a1) intersects boundary of triangle.

    Use this if the line segment lies on the same plane as the triangle.

    Parameters
    ----------
    t1 : ndarray(3,)
        First point of the triangle.
    t2 : ndarray(3,)
        Second point of the triangle
    t3 : ndarray(3,)
        Third point of the triangle.
    a0 : ndarray(3,)
        Initial point of line-segment (a0, a1).
    a1 : ndarray(3,)
        Final point of line-segment (a0, a1).

    Returns
    -------
    bool
        Returns true if the line segment (a0, a1) intersects the three boundary of a triangle.

    """
    # Edge (1, 2)
    a, b = intersection_line_segment(t1, t2, a0, a1)
    dist = np.linalg.norm(t1 + a * (t2 - t1) - a0 - b * (a1 - a0))
    if 1e-10 < a < 1 and 1e-10 < b < 1 and dist < 1e-2:
        print("abdist", a, b, dist)
        return True
    # Edge (2, 3)
    a, b = intersection_line_segment(t2, t3, a0, a1)
    dist = np.linalg.norm(t2 + a * (t3 - t2) - a0 - b * (a1 - a0))
    if 1e-10 < a < 1 and 1e-10 < b < 1 and dist < 1e-2:
        print("abdist2", a, b, dist)
        return True
    # Edge(3, 0)
    a, b = intersection_line_segment(t3, t1, a0, a1)
    dist = np.linalg.norm(t3 + a * (t1 - t3) - a0 - b * (a1 - a0))
    if 1e-10 < a < 1 and 1e-10 < b < 1 and dist < 1e-2:
        print("abdist3", a, b, dist)
        return True
    return False
