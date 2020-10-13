import numpy as np
from scipy.optimize import minimize, root_scalar

__all__ = [
    "step_equilateral_triangle",
    "project_onto_affine_plane",
    "normal_on_surface",
    "project_onto_surface",
    "get_curvature_info",
    "angle_between_vectors",
    "find_random_point_on_isosurface",
    "calculate_front_angle",
    "step_isosceles_triangle",
]


def step_isosceles_triangle(pt, midpt, normal, dist_edge, length_side):
    # Projecting it on the tangent space of mid-point.
    diff = pt - midpt
    diff = diff + np.dot(normal, diff) * normal
    # height of isosceles.
    height = np.sqrt(np.abs(length_side ** 2.0 - dist_edge ** 2.0 / 4.0))
    assert not np.isnan(height)
    return midpt + height * diff / np.linalg.norm(diff)


def step_equilateral_triangle(midpt, perpendicular_unit, edge_dist):
    # Get the third point so that you have equilateral triangle.
    assert np.abs(np.linalg.norm(perpendicular_unit) - 1.0) < 1e-5
    return midpt + edge_dist * perpendicular_unit * np.sqrt(3.0) / 2.0


def project_onto_affine_plane(pt, pt_plane, normal):
    # Projects a point onto a affine plane with center pt_plane and normal normal.
    # Normal vector is normalized.
    # https://math.stackexchange.com/questions/1664030/computing-the-projection-of-a-point-onto-an-affine-plane
    return pt - np.dot(normal, pt - pt_plane) * normal


def angle_between_vectors(pt1, pt2):
    r"""Gets interior angle between two vectors centered at the origin."""
    return np.arccos(np.dot(pt1, pt2) / (np.linalg.norm(pt1) * np.linalg.norm(pt2)))


def normal_on_surface(fgrad, pt):
    grad = fgrad(pt)
    return grad / np.linalg.norm(grad)


def project_onto_surface(x0, f, iso, fgrad, tol=1e-13, maxiter=500):
    # Hard Coded Newton Algorithm.
    counter = 0
    x_b = x0.copy()  # Before
    x_a = x0.copy()  # After

    distance = np.linalg.norm(x_b - x_a)
    while (distance > tol or counter == 0) and counter < maxiter:
        temp = x_a.copy()
        grad = fgrad(x_a)  # Grad of f(x,y,z) - iso is the gradient of f(x, y, z).
        x_a = x_a - (f(x_a) - iso) * grad / np.linalg.norm(grad) ** 2.0  # Newton Step.
        x_b = temp
        distance = np.linalg.norm(x_b - x_a)
        counter += 1
    if counter == maxiter:
        print("distance tol grad", distance, tol, grad)
        raise ValueError("Newton did not converge.")
    return x_a


def find_random_point_on_isosurface(f, iso_val):
    # First find a sample of point via root-finding fixed with x, y coordinate.
    rand_pt = np.random.random((3,))
    fiso_z = lambda z: f(np.array([rand_pt[0], rand_pt[1], z])) - iso_val
    l_bnd, u_bnd = -10.0, 0.0
    is_same_sign = lambda x, y : np.sign(fiso_z(x)) == np.sign(fiso_z(y))
    counter = 0
    # If it is the same sign, try increaisng the lower bound.
    while is_same_sign(l_bnd, u_bnd) and counter < 10000:
        print(l_bnd, u_bnd, fiso_z(l_bnd), fiso_z(u_bnd))
        l_bnd += 0.01
        counter += 1
    # Try minimizing it using optimizer. Else try it with root-finder.
    if is_same_sign(l_bnd, u_bnd):
        sol = minimize(f, rand_pt)
        print(sol)
        seed = sol["x"]
        assert sol["success"]
    else:
        root_sol = root_scalar(fiso_z, method="brenth", bracket=(l_bnd, u_bnd))
        assert root_sol.converged
        seed = np.array([rand_pt[0], rand_pt[1], root_sol.root])
    assert f(seed) - iso_val < 1e-10
    return seed


def get_curvature_info(pt_coord, grad, fhess, type="max"):
    #  From the paper Adaptive polygonization of implicit surfaces.
    norm = np.linalg.norm(grad)
    hess = fhess(pt_coord)
    actual_hess = np.zeros((3, 3), dtype=np.float)
    for i in range(0, 3):
        for j in range(0, 3):
            actual_hess[i, j] = (hess[i, j] * norm - grad[i] * np.dot(grad, hess[j, :]) / norm) / norm**2.0

    eigenvalues = np.sort(np.linalg.eigvals(actual_hess))[::-1]
    assert np.any(np.abs(eigenvalues) < 1e-10)  # First eigenvalue is zero.
    k1, k2 = eigenvalues[np.abs(eigenvalues) > 1e-10]

    if type == "max":
        eigenvalues = np.sort(np.abs(eigenvalues))
        k1, k2 = eigenvalues[1], eigenvalues[2]
        return max(np.abs(k1), np.abs(k2))
    elif type == "gaussian":
        return k1 * k2
    elif type == "mean":
        return (k1 + k2) / 2.0
    elif type == "shape-index":
        return -2.0 * ((k2 + k1) / (k2 - k1)) / np.pi
    else:
        raise ValueError("DId not recognize type")


def calculate_front_angle(c_pt, b_pt, a_pt, c_normal):
    r"""
    Calculate the front angle of the edge (b, c) and (c, a).

    Note that this must respect the orientation of the boundary ie b->c->a.

    Parameters
    ----------
    c_pt : ndarray(3,)
        Coordinates of point inbetween point 'b' and point 'a'.
    b_pt : ndarray(3,)
        Coordinates of point 'b', meaning before.
    a_pt : ndarray(3,)
        Coordinates of point 'a', meaning after.
    c_normal : ndarray(3,)
        The normal of the isosurface at `c_pt`.

    Returns
    -------
    float :
        The angle between the edges (b, c) and (c, a). It is called front because
        on the boundary of the mesh, it gives the angle on the outside of the mesh,
        assuming the orientation is the same as the one on the boundary.

    """
    pt1 = b_pt - c_pt
    pt2 = a_pt - c_pt
    ang = angle_between_vectors(pt1, pt2)
    t = np.cross(pt1, pt2)
    if t.dot(c_normal) >= 0.0:
        return ang
    else:
        return 2.0 * np.pi - ang
