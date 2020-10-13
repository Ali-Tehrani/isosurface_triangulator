import numpy as np
from mesh._dist import distance_between_point_and_hyperplane
from mesh.utils import angle_between_vectors

__all__ = ["Vertex"]


class Vertex:
    def __init__(self, coord, index, normal, is_original=True):
        self._index = index
        self._coord = coord
        self._normal = normal
        self._neighbors = []
        self._triangle = []

        # Additional Information if it is a gap vertices.
        self.is_boundary = False
        self.before_vert_ind = None
        self.after_vert_ind = None

        # Additional information for dyer's edge-flipping algorithm.
        self.is_original = is_original

    @property
    def index(self):
        return self._index

    @property
    def coord(self):
        return self._coord

    @property
    def normal(self):
        return self._normal

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def triangles(self):
        return self._triangle

    def add_neighbor(self, indices):
        for i in indices:
            self._neighbors.append(i)

    def remove_duplicates_neighbors(self):
        self._neighbors = list(set(self._neighbors))

    def remove_duplicates_triangles(self):
        self._triangle = list(set(self._triangle))

    def add_triangle(self, tri_indices):
        for i in tri_indices:
            self._triangle.append(i)

    def convert_boundary(self):
        self.is_boundary = True

    @staticmethod
    def project_onto_affine_plane(pt, pt_plane, normal):
        # Projects a point onto a affine plane with center pt_plane and normal normal.
        # Normal vector is normalized.
        # https://math.stackexchange.com/questions/1664030/computing-the-projection-of-a-point-onto-an-affine-plane
        return pt - np.dot(normal, pt - pt_plane) * normal

    def compute_planes(self, current_v, before_v, after_v):
        assert isinstance(current_v, Vertex)
        assert isinstance(before_v, Vertex)
        assert before_v.index == self.before_vert_ind
        assert isinstance(after_v, Vertex)
        assert after_v.index == self.after_vert_ind
        pt1 = before_v.coord - current_v.coord
        pt2 = after_v.coord - current_v.coord
        self.before_plane = np.cross(current_v.normal, pt1)
        self.before_plane /= np.linalg.norm(self.before_plane)
        self.after_plane = np.cross(pt2, current_v.normal)
        self.after_plane /= np.linalg.norm(self.after_plane)

    def is_vertex_above(self, current_v, before_v, after_v, other_vertex, which_plane="before"):
        assert isinstance(current_v, Vertex)
        assert isinstance(other_vertex, Vertex)
        assert current_v.index == self.index
        assert which_plane in ["before", "after"]
        if which_plane == "before":
            dist = np.linalg.norm(current_v.coord - before_v.coord)
            normal = self.before_plane
        else:
            dist = np.linalg.norm(current_v.coord - after_v.coord)
            normal = self.after_plane

        # Check if it is contained the half-space in which the normal points, ie the other gaps.
        if np.dot(normal, other_vertex.coord - current_v.coord) >= 0.0:
            # Distance between point and the plane!
            dist_plane = distance_between_point_and_hyperplane(
                other_vertex.coord, normal, current_v.coord
            )
            if dist_plane >= dist * 0.1:
                return True
            return False
        # Wasn't contained in the right half-space, so return False.
        return False
