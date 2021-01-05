import numpy as np
from ._vert import Vertex
from ._collision import (
    is_point_inside_triangle,
    intersect_line_segment_boundary_triangle,
    intersection_line_segment,
)
from ._dist import distance_vertex_and_edge
from .utils import project_onto_affine_plane

__all__ = ["Triangle"]


class Triangle():
    r"""
    Methods
    -------
    get_longest_edge_of_triangle() :
    get_vertices() :
    compute_distance_two_triangle() :
    get_max_over_min_length_ratio() :
    get_angles() :
    normal_to_plane() :
    overlap_with_other_triangle() :

    """
    def __init__(self, index, v1, v2, v3):
        assert isinstance(index, int)
        assert isinstance(v1, Vertex)
        assert isinstance(v2, Vertex)
        assert isinstance(v3, Vertex)
        self.index = index
        self.vertex = [v1, v2, v3]
        self.vindices = [v1.index, v2.index, v3.index]

    def get_longest_edge_of_triangle(self):
        dist_12 = np.linalg.norm(self.vertex[0].coord - self.vertex[1].coord)
        dist_23 = np.linalg.norm(self.vertex[1].coord - self.vertex[2].coord)
        dist_13 = np.linalg.norm(self.vertex[0].coord - self.vertex[2].coord)
        return max([dist_12, dist_23, dist_13])

    def get_vertices(self):
        return self.vertex[0], self.vertex[1], self.vertex[2]

    def compute_distance_two_triangle(self, v1, v2, v3):
        r"""(v1, v2, v3) : Vertex, Vertices that form the other triangle."""

        # Computes the distance between two triangles.
        min_dist = np.inf
        # Enumerate through all vertex combo.
        vertex = [v1, v2, v3]
        edge = [(0, 1), (1, 2), (2, 0)]
        for v in self.vertex:
            for e in edge:
                b1, b2 = vertex[e[0]], vertex[e[1]]
                dist = distance_vertex_and_edge(v.coord, b1.coord, b2.coord)
                if dist <= min_dist:
                    min_dist = dist
        for v in vertex:
            for e in edge:
                b1, b2 = self.vertex[e[0]], self.vertex[e[1]]
                dist = distance_vertex_and_edge(v.coord, b1.coord, b2.coord)
                if dist <= min_dist:
                    min_dist = dist
        # Distance between vertices.
        for i in range(0, 3):
            for j in range(0, 3):
                # Distance between vertex
                dist = np.linalg.norm(self.vertex[i].coord - vertex[j].coord)
                if dist <= min_dist:
                    min_dist = dist

        assert not np.isinf(min_dist), "Minimum distance wasn't found."
        return min_dist

    def get_max_over_min_length_ratio(self):
        v1, v2, v3 = self.get_vertices()
        dist_12 = np.linalg.norm(v1.coord - v2.coord)
        dist_23 = np.linalg.norm(v2.coord - v3.coord)
        dist_31 = np.linalg.norm(v3.coord - v1.coord)
        ratio = max(dist_12, dist_23, dist_31) / min(dist_12, dist_23, dist_31)
        return ratio

    def get_angles(self):
        v1, v2, v3 = self.get_vertices()
        return Triangle.get_angles_from_vertices(v1.coord, v2.coord, v3.coord)

    @staticmethod
    def get_angles_from_vertices(p1, p2, p3):
        angles = []
        # Length between x, y
        length_xy = np.linalg.norm(p1 - p2)
        # Length between x, z
        length_xz = np.linalg.norm(p1 - p3)
        length_yz = np.linalg.norm(p2 - p3)
        # Get Angles Between All Edges.
        # Angle between ix->iy and ix->iz
        c = length_yz**2.0 - length_xy**2.0 - length_xz**2.0
        c /= (-2.0 * length_xy * length_xz)
        angles.append(np.arccos(c))
        # Angle between iy->ix and iy->iz
        c = length_xz** 2.0 - length_xy ** 2.0 - length_yz ** 2.0
        c /= (-2.0 * length_xy * length_yz)
        angles.append(np.arccos(c))
        #Angle between iz->ix and iz->iy
        c = length_xy ** 2.0 - length_xz ** 2.0 - length_yz ** 2.0
        c /= (-2.0 * length_xz * length_yz)
        angles.append(np.arccos(c))
        return angles

    def normal_to_plane(self):
        b1, b2, b3 = self.get_vertices()
        cross = np.cross(b2.coord - b1.coord, b3.coord - b1.coord)
        return cross / np.linalg.norm(cross)

    def is_overlap_with_other_triangle(self, v1, v2, v3, normal):
        indices = {v1.index, v2.index, v3.index}
        b1, b2, b3 = self.get_vertices()
        indices_tri = {b1.index, b2.index, b3.index}
        overlap_indices = indices.intersection(indices_tri)
        numb_overlap = len(overlap_indices)
        normal2 = self.normal_to_plane()

        # Project points onto plane induced by the other.
        proj_b1 = project_onto_affine_plane(b1.coord, v1.coord, normal)
        proj_b2 = project_onto_affine_plane(b2.coord, v1.coord, normal)
        proj_b3 = project_onto_affine_plane(b3.coord, v1.coord, normal)
        proj_v1 = project_onto_affine_plane(v1.coord, b1.coord, normal2)
        proj_v2 = project_onto_affine_plane(v2.coord, b1.coord, normal2)
        proj_v3 = project_onto_affine_plane(v3.coord, b1.coord, normal2)

        assert np.abs(np.dot(proj_v1, normal2) - np.dot(b1.coord, normal2)) < 1e-5
        assert np.abs(np.dot(proj_v2, normal2) - np.dot(b1.coord, normal2)) < 1e-5
        assert np.abs(np.dot(proj_v3, normal2) - np.dot(b1.coord, normal2)) < 1e-5

        # Check if projected vertices that aren't shared is inside the other.
        diff_vert = indices.difference(overlap_indices)
        for i in diff_vert:
            if i == v1.index:
                # Check if proj_v1 is inside this triangle.
                overlap = is_point_inside_triangle(b1.coord, b2.coord, b3.coord, proj_v1)
            elif i == v2.index:
                # Check if proj_v2 is inside this triangle.
                overlap = is_point_inside_triangle(b1.coord, b2.coord, b3.coord, proj_v2)
            elif i == v3.index:
                # Check if proj_v3 is inside this triangle.
                overlap = is_point_inside_triangle(b1.coord, b2.coord, b3.coord, proj_v3)
            if overlap:
                print("Vertex in another. 1")
                return True

        diff_other = indices_tri.difference(overlap_indices)
        for i in diff_other:
            if i == b1.index:
                # Check if proj_b1 is inside tri
                overlap = is_point_inside_triangle(v1.coord, v2.coord, v3.coord, proj_b1)
            elif i == b2.index:
                # Check if proj_b2 is inside tri.
                overlap = is_point_inside_triangle(v1.coord, v2.coord, v3.coord, proj_b2)
            elif i == b3.index:
                # Check if proj_b3 is inside tri.
                overlap = is_point_inside_triangle(v1.coord, v2.coord, v3.coord, proj_b3)
            if overlap:
                print("Vertex in another. 2")
                return True

        # Check Edges.
        if numb_overlap == 1:
            # If they have one vertex in common call it v
            # then check if opposite edge (not containing v) intersects.
            index1 = diff_vert.pop()
            index2 = diff_vert.pop()
            # Find Opposite Point.
            if index1 == v1.index:
                k0 = proj_v1
            elif index1 == v2.index:
                k0 = proj_v2
            elif index1 == v3.index:
                k0 = proj_v3
            # Find the other Opposite POint.
            if index2 == v1.index:
                k1 = proj_v1
            elif index2 == v2.index:
                k1 = proj_v2
            elif index2 == v3.index:
                k1 = proj_v3
            if intersect_line_segment_boundary_triangle(b1.coord, b2.coord, b3.coord, k0, k1):
                return True

            index1 = diff_other.pop()
            index2 = diff_other.pop()
            if index1 == b1.index:
                k0 = proj_b1
            elif index1 == b2.index:
                k0 = proj_b2
            elif index1 == b3.index:
                k0 = proj_b3
            # Fidn the other Opposite POint.
            if index2 == b1.index:
                k1 = proj_b1
            elif index2 == b2.index:
                k1 = proj_b2
            elif index2 == b3.index:
                k1 = proj_b3
            if intersect_line_segment_boundary_triangle(v1.coord, v2.coord, v3.coord, k0, k1):
                return True

        elif numb_overlap == 2:
            # If they have two vertices call it a, b in common
            # then compare opposite edges e.g. (b,c) and (a, d) to each other.
            index = diff_vert.pop()
            if index == v1.index:
                k0 = proj_v1  # Non-shared vertex.
                k1 = proj_v2
                k2 = proj_v3
                k1_ind = v2.index
                k2_ind = v3.index
            elif index == v2.index:
                k0 = proj_v2  # Non-shared vertex.
                k1 = proj_v1
                k2 = proj_v3
                k1_ind = v1.index
                k2_ind = v3.index
            elif index == v3.index:
                k0 = proj_v3  # Non-shared vertex.
                k1 = proj_v1
                k2 = proj_v2
                k1_ind = v1.index
                k2_ind = v2.index

            index_other = diff_other.pop()
            if index_other == b1.index:
                m0 = b1
                if b2.index == k1_ind:
                    m1 = b2
                    m2 = b3
                elif b2.index == k2_ind:
                    m1 = b3
                    m2 = b2
            elif index_other == b2.index:
                m0 = b2
                if b1.index == k1_ind:
                    m1 = b1
                    m2 = b3
                elif b1.index == k2_ind:
                    m1 = b3
                    m2 = b1
            elif index_other == b3.index:
                m0 = b3
                if b1.index == k1_ind:
                    m1 = b1
                    m2 = b2
                elif b1.index == k2_ind:
                    m1 = b2
                    m2 = b1

            lam1, lam2 = intersection_line_segment(k0, k1, m0.coord, m2.coord)
            if -1e-10 <= lam1 <= 1.0 and -1e-10 <= lam2 <= 1.0:
                print("Triangle shared, Line segment intersects 1")
                return True
            lam1, lam2 = intersection_line_segment(k0, k2, m0.coord, m1.coord)
            if -1e-10 <= lam1 <= 1.0 and -1e-10 <= lam2 <= 1.0:
                print("Triangle shared, Line segment intersects 2")
                return True

        else:
            # If They have no vertices in common. (Excluding the same triangle)
            # Check for edge intersection to the boundary of the other triangle.

            # First Check Triangle (b1, b2, b3).
            is_overlap = intersect_line_segment_boundary_triangle(b1.coord, b2.coord, b3.coord, proj_v1, proj_v2)
            if is_overlap:
                return True
            is_overlap2 = intersect_line_segment_boundary_triangle(b1.coord, b2.coord, b3.coord, proj_v2, proj_v3)
            if is_overlap2:
                return True
            is_overlap3 = intersect_line_segment_boundary_triangle(b1.coord, b2.coord, b3.coord, proj_v3, proj_v1)
            if is_overlap3:
                return True

            # Second Check the other triangle.
            is_overlap = intersect_line_segment_boundary_triangle(v1.coord, v2.coord, v3.coord, proj_b1, proj_b2)
            if is_overlap:
                return True
            is_overlap2 = intersect_line_segment_boundary_triangle(v1.coord, v2.coord, v3.coord, proj_b2, proj_b3)
            if is_overlap2:
                return True
            is_overlap3 = intersect_line_segment_boundary_triangle(v1.coord, v2.coord, v3.coord, proj_b3, proj_b1)
            if is_overlap3:
                return True
        return False

    def area(self):
        r"""
        Heron's formula.
        https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
        https://people.eecs.berkeley.edu/~wkahan/Triangle.pdf
        """
        v1, v2, v3 = self.get_vertices()
        p1, p2, p3 = v1.coord, v2.coord, v3.coord
        # a = np.linalg.norm(p1 - p2)
        # b = np.linalg.norm(p2 - p3)
        # c = np.linalg.norm(p3 - p1)
        # prim = a + b + c
        # s = prim / 2.0
        # return np.sqrt(s * (s - a) * (s - b) * (s - c))

        return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))