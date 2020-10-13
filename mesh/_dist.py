r"""
This file is responsible for calculating distances between triangles, points and line segments.
"""

import numpy as np

__all__ = [
    "distance_vertex_and_edge",
    "distance_between_point_and_hyperplane",
]


def distance_vertex_and_edge(pt, a0, a1):
    r"""
    Calculate smallest distance between point and line segment.
    
    Parameters
    ----------
    pt : ndarray(3,)
        Coordinate of point.
    a0 : ndarray(3,)
        Coordinate of initial point in line segment (a0, a1).
    a1 : ndarray(3,)
        Coordinate of final point in line segment (a0, a1).

    Returns
    -------
    float
        Smallest distance between point and line segment.

    """
    # See section "Distance from a Point to a Ray or Segment (any Dimension n)"
    # in http://geomalgorithms.com/a02-_lines.html
    assert isinstance(pt, np.ndarray)
    assert isinstance(a0, np.ndarray)
    assert isinstance(a1, np.ndarray)
    edge = a1 - a0
    w = pt - a0
    c1 = np.dot(w, edge)
    c2 = np.dot(edge, edge)
    if c1 <= 0.0:
        return np.linalg.norm(pt - a0)
    elif c2 <= c1:
        return np.linalg.norm(pt - a1)
    b = c1 / c2
    pb = a0 + b * edge
    return np.linalg.norm(pt - pb)


def distance_between_point_and_hyperplane(pt, normal, pt_on_plane):
    assert np.abs(np.linalg.norm(normal) - 1.0) < 1e-10
    return np.abs(np.dot(normal, pt - pt_on_plane))


def distance_between_points(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)
