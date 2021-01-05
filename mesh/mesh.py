import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from octreepy import SimpleOctree
from sortedcollections import ValueSortedDict

from ._vert import Vertex
from ._tri import Triangle
from ._dist import (
    distance_between_point_and_hyperplane,
    distance_vertex_and_edge,
    distance_between_points,
)
from .utils import (
    find_random_point_on_isosurface,
    project_onto_surface,
    normal_on_surface,
    calculate_front_angle,
    step_equilateral_triangle,
    angle_between_vectors,
    step_isosceles_triangle,
)


class _ActiveEdge():
    r"""
    These edges are always oriented correctly.
    """
    def __init__(self, edge, other_vertex):
        assert isinstance(edge, tuple)
        assert isinstance(edge[0], Vertex)
        assert isinstance(edge[1], Vertex)
        assert isinstance(other_vertex, Vertex)
        self._edge = edge
        self._other_vertex = other_vertex
        self.is_gap = False

    @property
    def edge(self):
        return self._edge

    @property
    def other_vertex(self):
        return self._other_vertex

    def get_vertices(self):
        return self.edge[0], self.edge[1]

    def get_vertex_ind(self):
        return self.edge[0].index, self.edge[1].index

    def midpt(self):
        return (self._edge[0].coord + self._edge[1].coord) / 2.0

    def perpendicular_from_midpt(self, midpt, normal):
        r"""Unit vector perpendicular to a edge. THe other vert is apart of the same triangle."""
        edge_vec = self.edge[1].coord - midpt
        perpedin = np.cross(normal, edge_vec)
        mid_pt = (self.edge[0].coord + self.edge[1].coord) / 2.0
        if np.dot(perpedin, self.other_vertex.coord - mid_pt) > 0.:
            return -perpedin / np.linalg.norm(perpedin)
        return perpedin / np.linalg.norm(perpedin)


class ActiveEdgeHolder:
    r"""Container class for holding Boundary/Active Edges for the Growing phase."""
    def __init__(self):
        self._active_edge_list = []
        self._gap_edge_list = []  # Need gap list when points are removed from active_edge_list.
        self.longest_edge_distance = -np.inf  # Used for passing the second test.

    def __len__(self):
        return len(self._active_edge_list)

    def pop(self):
        return self._active_edge_list.pop()

    def get_adjacent_edge(self, edge):
        r"""
        Gets the adjacent active edges (and their indices) that follows the orientation (boundary).
        """
        other_edges = []
        other_index = []
        is_gap = []
        indices = set(edge.get_vertex_ind())
        # Go through all other edges.
        for i, edge_other in enumerate(self._active_edge_list + self._gap_edge_list):
            # Get its indices.
            indices_other = set(edge_other.get_vertex_ind())
            numb_matching = len(indices_other.intersection(indices))
            if numb_matching == 1:
                other_edges.append(edge_other)
                if edge_other.is_gap:
                    index = i - len(self._active_edge_list)
                    is_gap.append(True)
                else:
                    index = i
                    is_gap.append(False)
                other_index.append(index)
                # Stop after finding two edges.
                if len(other_edges) == 2:
                    break
        assert len(other_edges) == 2
        return other_edges, other_index, is_gap

    def get_correct_orientation(self, edge1, edge2):
        r"""
        Returns the vertices (c, b, a) such that (b, c) is one edge and (c, a) is the other edge.
        """
        v1, v2 = edge1.get_vertices()
        b1, b2 = edge2.get_vertices()
        if v1.index == b2.index:
            # Case: (1->2) and (3->1)
            return (v1, b1, v2)
        if v2.index == b1.index:
            # Case (1->2) and (2->3)
            return (v2, v1, b2)
        assert RuntimeError("It couldn't find correct orientation of boundary in growing phase.")

    def add_edge(self, v1, v2, v3):
        r"""Edge (v1, v2) is added, and v3 is other_vertex. """
        self._active_edge_list.append(_ActiveEdge((v1, v2), v3))
        dist = distance_between_points(v1.coord, v2.coord)
        if dist > self.longest_edge_distance:
            self.longest_edge_distance = dist

    def add_first_vertices_and_triangle(self, v1, v2, v3):
        self.add_edge(v1, v2, v3)
        self.add_edge(v2, v3, v1)
        self.add_edge(v3, v1, v2)

    def add_new_point_and_triangle(self, v1, v2, v3):
        r"""
        v3 is a new point and (v1, v2, v3) is a new triangle.
        Edge always move from 1 to 2.
        """
        self.add_edge(v1, v3, v2)
        self.add_edge(v3, v2, v1)

    def add_to_gap(self, edge):
        edge.is_gap = True
        self._gap_edge_list.append(edge)

    def add_ear_cutting(self, c, b, a, edge_index, is_gap):
        r"""
        Do Earcutting on triangle (c, b, a). Remove the edge from edge_index from active_edge_list.
        is_gap tells whether the edge was already looked at, so you can't remove it.
        """
        if not is_gap:
            self._active_edge_list.pop(edge_index)
        else:
            self._gap_edge_list.pop(edge_index)
        self.add_edge(b, a, c)

    def remove_edge_needle_triangle(self, b, a, edge_index):
        r"""Removed vertex a and replaced it with b."""
        self._active_edge_list.pop(edge_index)
        if_found = False
        # Go through active edges and remove all accounts of vertex b.
        for i in range(0, len(self._active_edge_list)):
            edge_ind = self._active_edge_list[i].get_vertex_ind()
            print("edge ", edge_ind, a.index)
            if a.index == edge_ind[0]:
                print("shoulda happened")
                self._active_edge_list[i]._edge = (b, self._active_edge_list[i]._edge[1])
                if_found = True
            if a.index == edge_ind[1]:
                self._active_edge_list[i]._edge = (self._active_edge_list[i]._edge[0], b)
                if_found = True

        if not if_found:
            for i in range(0, len(self._gap_edge_list)):
                edge_ind = self._gap_edge_list[i].get_vertex_ind()
                if a.index == edge_ind[0]:
                    self._gap_edge_list[i]._edge = (b, self._gap_edge_list[i]._edge[1])
                elif a.index == edge_ind[1]:
                    self._gap_edge_list[i]._edge = (self._gap_edge_list[i]._edge[0], b)


class _FrontAngleDict():
    def __init__(self):
        self._front = None  # Keys are "vertex index" and Values are "front angles" for Boundary.

    def __len__(self):
        return len(self._front)

    @property
    def front(self):
        return self._front

    def setup_front_angles(self, front_dict):
        self._front = ValueSortedDict(front_dict)

    def update_front_angle(self, index, angle):
        self._front.update([(index, angle)])

    def add(self):
        pass

    def pop_front_angle(self):
        r"""Return smallest angle alongside its index."""
        index, angle = self._front.popitem(0)
        return index, angle

    def remove_vertices(self, v_list):
        for v in v_list:
            v.is_boundary = False
            v.after_vert_ind = None
            v.before_vert_ind = None
            self._front.pop(v.index)

    def remove_vertex(self, vertex_index):
        self._front.pop(vertex_index)


class VertexTriangles:
    r"""Datastructure on holding vertices of class Vertex and triangles of class Triangle."""
    def __init__(self, bounds, numb_pts, distance):
        # numb_pts : int, number of points you wnat in smallest octant.
        # distance : float, expected distance between each consequent point
        self._vertices = []
        self._triangles = []
        bounds_min = [x[0] for x in bounds]
        bounds_max = [x[1] for x in bounds]
        self.octree = SimpleOctree(bounds_min, bounds_max, numb_pts, distance)

    def length_vertices(self):
        return len(self._vertices)

    def length_triangles(self):
        return len(self._triangles)

    def add_vertex(self, v):
        self._vertices.append(v)
        # self.octree.insert(list(v.coord), v.index)
        print(v.coord)
        self.octree.InsertPoint(v.coord, v.index)

    def add_triangle(self, t):
        self._triangles.append(t)

    def get_triangle(self, index):
        return self._triangles[index]

    def get_vertex(self, index):
        return self._vertices[index]

    def to_array(self):
        vertices = []
        triangles = []
        removed_indices = []
        for i in range(len(self._vertices)):
            # I added None to vertices that are removed.
            if self._vertices[i] is not None:
                vertices.append(self._vertices[i].coord)
            else:
                removed_indices.append(i)
        removed_indices = np.array(removed_indices)
        for tri in self._triangles:
            # Here, it counts the number of integers to decrease by because of removed vertices.
            indices = [tri.vindices[0], tri.vindices[1], tri.vindices[2]]
            for i in range(0, 3):
                numb_to_dec = np.count_nonzero(tri.vindices[i] > removed_indices)
                if numb_to_dec > 0.0:
                    indices[i] -= numb_to_dec
            triangles.append(indices)
        return np.array(vertices), np.array(triangles)

    def add_first_vertices_and_triangle(self, v1, v2, v3):
        assert len(self._vertices) == 0
        assert len(self._triangles) == 0
        # Add vertices.
        [self.add_vertex(v) for v in [v1, v2, v3]]
        # Add Triangle
        tri_index = len(self._triangles)
        self.add_triangle(Triangle(tri_index, v1, v2, v3))
        # Add neighbour info to vertices.
        v1.add_neighbor((v2.index, v3.index))
        v2.add_neighbor((v1.index, v3.index))
        v3.add_neighbor((v1.index, v2.index))
        # Add triangle info to vertices.
        [v.add_triangle([tri_index]) for v in [v1, v2, v3]]

    def add_new_vertex_with_triangle(self, v1, v2, v3):
        # V3 is already added
        self.add_vertex(v3)
        tri_index = len(self._triangles)
        self.add_triangle(Triangle(tri_index, v1, v2, v3))
        # Add neighbour info to vertices.
        [v.add_neighbor([v3.index]) for v in [v1, v2]]
        [v3.add_neighbor([v.index]) for v in [v1, v2]]
        # Add triangle info to vertices.
        [v.add_triangle([tri_index]) for v in [v1, v2, v3]]

    def add_ear_cutting(self, c, b, a):
        # No new vertices are added.
        tri_index = len(self._triangles)
        self.add_triangle(Triangle(tri_index, c, b, a))
        # Add neighbour info to vertices.
        b.add_neighbor([a.index])
        a.add_neighbor([b.index])
        # Add triangle info to vertices.
        [v.add_triangle([tri_index]) for v in [c, b, a]]

    def search_radius(self, pt, radius, return_distance=False):
        r""" When pt is a coord, is used in growing phase.
        When pt is a vertex, is used in filling phase."""
        if isinstance(pt, Vertex):
            neighbors = pt.neighbors
            coord = pt.coord
        else:
            coord = pt
        # result = np.array([
        #     [x[0], x[2]] for x in self.octree.by_distance_from_point(coord, epsilon=radius)
        # ])
        result = self.octree.SearchRadius(coord, radius)
        # from octrees.octrees import Octree
        # octree = Octree(((-15, 15), (-15, 15), (-15, 15)))
        # for v in self._vertices:
        #     print("previous vertices", v.coord)
        #     octree.insert(list(v.coord), v.index)
        # python_result = [x for x in octree.by_distance_from_point(coord, epsilon=radius)]
        # print("Python :",  )
        # print("Coordinate and radius ", coord, radius)
        # for k in python_result:
        #     print(k)
        #     dist, coordinate, index = k
        #     print(np.linalg.norm(np.array(coordinate) - coord))
        # my_result = self.octree.SearchRadius(self._vertices[1].coord, radius)
        # print("My result ", my_result)
        # print("My software", my_result[0].vertex.coordinate, my_result[1].vertex.coordinate, my_result[2].vertex.coordinate)
        # print()
        # print("Result", result)
        indices = np.array([distance_vertex_obj.vertex.index for distance_vertex_obj in result], dtype=np.int)

        if isinstance(pt, Vertex):
            # Reduce indices to those that are on the boundary and not its neighbors.
            distances = np.array([
                result[j].distance for j, i in enumerate(indices) if
                self._vertices[i].is_boundary and i not in neighbors
            ])
            indices = np.array([
                i for i in indices if self._vertices[i].is_boundary and i not in neighbors
            ])

            indices = indices[1:]  # Remove itself as it is included in it.
            if return_distance:
                return indices, distances[1:]
            return indices

        if return_distance:
            return indices, indices[1:, 0]
        return indices


    def remove_needle_triangle(self, b, a):
        # b, a are Vertex.
        # Joining Vertex b and Vertex A together (Replace A by B), by the following procedure.
        # Calculate the mid-point between b and a.
        mid_pt = b.coord + (a.coord - b.coord) / 2.0
        print("before and after index", b.index, a.index)
        print("besfore", self.to_array())
        # Find index of b and a.
        index_b = None
        index_a = None
        for i in range(len(self._vertices) - 1, -1, -1):
            # Go backwards because it might be faster to find (b and a).
            v = self.get_vertex(i)
            if v.index == b.index:
                index_b = i
            elif v.index == a.index:
                index_a = i

            if index_a is not None and index_b is not None:
                break

        # Move b to be the midpoint and add vertex a information to b.
        self._vertices[index_b].coord = mid_pt
        self._vertices[index_b].add_neighbor(a.neighbors)
        self._vertices[index_b].remove_duplicates_neighbors()
        self._vertices[index_b].add_triangle(a.triangles)
        self._vertices[index_b].remove_duplicates_triangles()

        # Update neighbours of a to make it neighbors of b.
        for i in a.neighbors:
            for j, k in enumerate(self._vertices[i].neighbors):
                if k == a.index:
                    self._vertices[i].neighbors[j] = b.index
                    self._vertices[i].remove_duplicates_neighbors()

        # Update triangles of a to point to b.
        for t_ind in a.triangles:
            for i in range(0, 3):
                if a.index == self._triangles[t_ind].vindices[i]:
                    self._triangles[t_ind].vindices[i] = b.index
                    self._triangles[t_ind].vertex[i] = b
        print(self.to_array())
        # Remove vertex a.
        self._vertices[index_a] = None
        # Go Through each vertex past that index_a, and decrease all of its indices.
        # While it does that, go through each triangle and change the triangle too.

    def complete_one_triangle(self, c, b, a):
        tri = Triangle(len(self._triangles), c, a, b)
        self.add_triangle(tri)
        # Update vertices triangle information.
        [v.add_triangle([tri.index]) for v in [c, b, a]]
        # Update neighbor hood
        b.add_neighbor([a.index])
        a.add_neighbor([b.index])

    def search_collision_points_around_current(self, c, b, a, radius):
        r"""
        Finds potential collusion points around c with radius "radius".

        Note that this is different from Karaknis and Stewart, as I only care if it is infront.

        Returns
        -------
        list of indices
        """
        indices = self.search_radius(c, radius)

        # Compute normal plane spanned by the "normal of c" and vector "b - c" (and "a - c")
        pt1 = b.coord - c.coord
        pt2 = a.coord - c.coord
        before_plane = np.cross(c.normal, pt1)
        before_plane /= np.linalg.norm(before_plane)
        after_plane = np.cross(pt2, c.normal)
        after_plane /= np.linalg.norm(after_plane)

        actual_neighbors = []
        for i in indices:
            # Only-select points that are in-front, facing the boundary.
            other_vertex = self.get_vertex(i)

            # Check if it is contained the half-space in which the normal points, ie the other gaps.
            before_is_above = False
            if np.dot(before_plane, other_vertex.coord - c.coord) >= 0.0:
                before_is_above = True

            after_is_above = False
            if np.dot(after_plane, other_vertex.coord - c.coord) >= 0.0:
                after_is_above = True

            # If the point is above both planes.
            if before_is_above and after_is_above:
                actual_neighbors.append(i)

            # # If Convex.
            # if 2.0 * np.pi - front_angle <= np.pi:  # If interior angle is less than pi. gvertex.is_convex:
            #     if before_is_above and after_is_above:
            #         actual_neighbors.append(i)
            # # If Concave.
            # else:
            #     if before_is_above or after_is_above:
            #         actual_neighbors.append(i)
        return actual_neighbors, before_plane, after_plane

    def search_collision_points_around_before_after(self, b, a, before_plane, after_plane, radius):
        indices_b, dist_b = self.search_radius(b, radius, return_distance=True)
        indices_a, dist_a = self.search_radius(a, radius, return_distance=True)

        # Remove vertex b from indices_a and vertex a from indices_b.
        for i in range(len(indices_b)):
            if indices_b[i] == a.index:
                indices_b = np.delete(indices_b, [i])
                dist_b = np.delete(dist_b, [i])
                break
        for j in range(len(indices_a)):
            if indices_a[j] == b.index:
                indices_a = np.delete(indices_a, [j])
                dist_a = np.delete(dist_a, [j])
                break

        # If they both don't have any neighbors.
        print(indices_a, indices_b)
        if len(indices_a) == 0 and len(indices_b) == 0:
            return None

        # Find the best point that is in-front of both of them.
        indices = np.hstack((indices_a, indices_b))
        distances = np.hstack((dist_a, dist_b))
        sort_ind = distances.argsort()
        print(sort_ind)
        print(indices)
        print(distances)
        indices = indices[sort_ind]

        for i in range(len(indices)):
            other_vertex = self.get_vertex(indices[i])

            before_is_above = False
            if np.dot(before_plane, other_vertex.coord - b.coord) >= 0.0:
                before_is_above = True

            after_is_above = False
            if np.dot(after_plane, other_vertex.coord - a.coord) >= 0.0:
                after_is_above = True

            if before_is_above and after_is_above:
                return indices[i]
        return None

    def is_overlap_of_other_triangles(self, v1, v2, v3, tri_indices):
        r"""
        Returns true if the triangle (v1, v2, v3) intersects any triangle.

        Parameters
        ----------
        (v1, v2, v3): Vertex
            Vertices.
        tri_indices : list
            List of triangle indices.

        """
        normal2 = np.cross(v2.coord - v1.coord, v3.coord - v1.coord)
        normal2 /= np.linalg.norm(normal2)
        print("Start triangle overlap")
        print("Triagnle ", v1.index, v2.index, v3.index)
        for t in np.unique(tri_indices):
            tri = self.get_triangle(t)
            print("Vertex indices", tri.vindices)
            overlap = tri.is_overlap_with_other_triangle(v1, v2, v3, normal2)

            if overlap:
                print("Found overlap")
                return True
        return False

    def triangle_collision_list(self, b, a, n, other_vertices_indices=()):
        # Returns potential triangle collisions.
        triangle_collision = [tri_i for tri_i in n.triangles if
                               any([x.is_boundary for x in self.get_triangle(tri_i).vertex])]
        triangle_collision += [tri_i for tri_i in a.triangles if
                               any([x.is_boundary for x in self.get_triangle(tri_i).vertex])]
        triangle_collision += [tri_i for tri_i in b.triangles if
                               any([x.is_boundary for x in self.get_triangle(tri_i).vertex])]
        for i in other_vertices_indices:
            v = self.get_vertex(i)
            triangle_collision += [tri_i for tri_i in v.triangles if
                                   any([x.is_boundary for x in self._triangles[tri_i].vertex])]
        return np.unique(triangle_collision)

    def quality_measure(self, triangle_set):
        # Here is how I define a quality measure of a triangle.
        min_ang = np.inf
        for t in triangle_set:
            min_angles = min(Triangle.get_angles_from_vertices(t[0].coord, t[1].coord, t[2].coord))
            if min_angles < min_ang:
                min_ang = min_angles
        return min_ang

    def best_triangle_set_with_overlap(self, list_tri_set, list_tri_collision):
        r"""Return the best triangle set from a list by maximizing quality and no overlap.
        list_tri_set : list of set of 3 Vertex
            Ie [(v1, v2, v3), (v4, v5, v6), ..] where v_i are class "Vertex".
        list_tri_collsion : list of int.
        """
        quality = [self.quality_measure(triangle_set) for triangle_set in list_tri_set]
        print("Qualtiy ", quality)
        index = [i for i in range(len(list_tri_set))]

        # Sort them in decreasing order
        _, index = zip(*sorted(zip(quality, index), reverse=True))
        print(_)
        best_index = None
        for i in range(len(list_tri_set)):
            print("triangle set i", i)
            # Grab the highest quality triangle set.
            tri_set = list_tri_set[index[i]]

            # Check if that triangle set doesn't overlap with  others.
            is_overlap = False
            for tri in tri_set:
                # Five cycles gives redundant triangles with the same vertex twice, this handles that.
                if len({tri[0].index, tri[1].index, tri[2].index}) != 3:
                    is_overlap = True
                    break
                if self.is_overlap_of_other_triangles(tri[0], tri[1], tri[2], list_tri_collision):
                    print("is oveerlap")
                    is_overlap = True
                    break

            # If it doesn't overlap, then  return this one.
            if not is_overlap:
                best_index = index[i]
                break

        # If they weren't found
        not_found = False
        if best_index is None:
            not_found = True

        return best_index, not_found

    def add_triangle_set(self, tri_set):
        r""" Adds the triangle set.

        For code usability and at the expense of spped, I decided to make one function.
        """
        for tri in tri_set:
            # For each vertex add triangle information
            [v.add_triangle([self.length_triangles()]) for v in tri]

            self.add_triangle(Triangle(self.length_triangles(), tri[0], tri[1], tri[2]))
            # Add neighbourhood information
            tri[0].add_neighbor([tri[1].index, tri[2].index])
            tri[0].remove_duplicates_neighbors()
            tri[1].add_neighbor([tri[0].index, tri[2].index])
            tri[1].remove_duplicates_neighbors()
            tri[2].add_neighbor([tri[0].index, tri[1].index])
            tri[2].remove_duplicates_neighbors()

    def remove_boundary_info_vertex(self, v):
        v.is_boundary = False
        v.after_vert_ind = None
        v.before_vert_ind = None


class IsoSurfaceTriangulator:
    r"""
    Class for triangulation an implicit surface, inspired by Karkanis and Stewart's algorithm.

    Methods
    -------
    __call__(delta=0.1, tol_root=1e-10, max_length=1.5) : Run the triangulation algorithm.
    to_array : Returns vertices and triangle indices into numpy array format.
    plot_surface(extra_vertex=[]) : Plot the mesh at its current state with additional vertices.

    """
    def __init__(self, func, fgrad, isoval, bounds, numb_pts, distance, fhess=None):
        # These are manupliated by the algorithm,
        self._vert_tris = VertexTriangles(bounds, numb_pts, distance)
        # These are needed for the growing phase of Karkanis and Stewart's.
        self._active_edges = ActiveEdgeHolder()
        # This are static and will not change.
        self.func = func
        self.fgrad = fgrad
        self.fhess = fhess
        self.isoval = isoval
        self.bounds = bounds
        # Filling Phase Information
        self._front = _FrontAngleDict()
        # Dyer's Edge flipping information.
        self._nld = []  # Stores the NonDelaunay edges.

    def to_array(self):
        return self._vert_tris.to_array()

    def plot_surface(self, extra_verts=[]):
        r"""extra_verts =[(v, "r)] will plot the vertex v with color "red". Useful for debugging."""
        vert, triangles = self.to_array()
        # Test that it's unique vertices
        new_vert = np.unique(vert.round(decimals=5), axis=1)
        assert len(vert) == len(new_vert), "Duplicate vertices are added, upto fifth decimal place."

        # from scipy.spatial.distance import pdist
        # dist = pdist(vert)
        # mini = np.min(dist)
        # print("Minimum distance is ", np.min(dist))
        # if mini < 0.05:
        #     print("Minimum distance is verrry SMALL.")
        #     plt.title("Distance between vertices less than 0.1")
        #     plt.plot(dist[dist < 0.1], "ro")
        #     plt.show()

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        if len(extra_verts) != 0:
            for v in extra_verts:
                if isinstance(v, tuple):
                    if isinstance(v[0], Vertex):
                        a = v[0].coord
                    else:
                        a = v[0]
                    color = v[1]
                else:
                    a = v.coord
                    color = "g"
                ax.scatter(a[0], a[1], a[2], marker="o", c=color, zorder=2, s=80)
        X, Y, Z = vert[:, 0], vert[:, 1], vert[:, 2]
        ax.plot_trisurf(X, Y, Z, triangles=triangles, cmap=plt.cm.Spectral, zorder=1)
        plt.show()

    def plot_mesh_angles_ratio(self, bins=50):
        angles = []
        length_ratio = []

        lengths = np.zeros((len(self._vert_tris._vertices), len(self._vert_tris._vertices)))
        for t in self._vert_tris._triangles:
            ix, iy, iz = t.vindices
            if np.abs(lengths[ix, iy]) < 1e-10:
                # Length between x, y
                length_xy = np.linalg.norm(self._vert_tris._vertices[ix].coord -
                                           self._vert_tris._vertices[iy].coord)
                lengths[ix, iy] = length_xy
                lengths[iy, ix] = length_xy
            if np.abs(lengths[ix, iz]) < 1e-10:
                # Length between x, z
                length_xz = np.linalg.norm(self._vert_tris._vertices[ix].coord -
                                           self._vert_tris._vertices[iz].coord)
                lengths[ix, iz] = length_xz
            if np.abs(lengths[iy, iz]) < 1e-10:
                # Length between y, z
                length_yz = np.linalg.norm(self._vert_tris._vertices[iy].coord -
                                           self._vert_tris._vertices[iz].coord)
                lengths[iy, iz] = length_yz
            length_ratio.append(max(length_xz, length_xz, length_yz) /
                                min(length_xz, length_xz, length_yz))

            # Get Angles Between All Edges.
            # Angle between ix->iy and ix->iz
            c = lengths[iy, iz]**2.0 - lengths[ix, iy]**2.0 - lengths[ix, iz]**2.0
            c /= (-2.0 * lengths[ix, iy] * lengths[ix, iz])
            angles.append(np.arccos(c))
            # Angle between iy->ix and iy->iz
            c = lengths[ix, iz] ** 2.0 - lengths[ix, iy] ** 2.0 - lengths[iy, iz] ** 2.0
            c /= (-2.0 * lengths[ix, iy] * lengths[iy, iz])
            angles.append(np.arccos(c))
            #Angle between iz->ix and iz->iy
            c = lengths[ix, iy] ** 2.0 - lengths[ix, iz] ** 2.0 - lengths[iy, iz] ** 2.0
            c /= (-2.0 * lengths[ix, iz] * lengths[iy, iz])
            angles.append(np.arccos(c))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        angles = np.array(angles) * 180.0 / np.pi
        fig.suptitle("Histogram of angles of triangle mesh and Length Ratios "
                     "(Closer to One More Equalateral), #Vertices = %d" %
                     len(self._vert_tris._vertices))
        ax1.hist(angles, bins=bins)
        ax1.axvline(x=np.mean(angles),color="r", label="Mean")
        ax1.axvline(x=np.mean(angles) + np.std(angles), color="y", label="Mean + STD")

        ax1.text(2, 500, 'Minimum angle %f' % np.min(angles), style='italic',
                 bbox={'facecolor': 'red', 'alpha': 0.5})
        ax1.text(2, 400, 'Maximum angle %f' % np.max(angles), style='italic',
                 bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        ax1.set(xlabel="Angles (Degrees)", ylabel="Frequency")

        ax2.hist(length_ratio, bins=bins)
        ax2.set(xlabel="Max(Length of Edges)/Min(Length of Edges)")
        plt.legend()
        plt.show()

    def _analytic_curvature(self, pt, grad, type="max"):
        #  From the paper Adaptive polygonization of implicit surfaces.
        norm = np.linalg.norm(grad)
        hess = self.fhess(pt)
        actual_hess = np.zeros((3, 3), dtype=np.float)
        for i in range(0, 3):
            for j in range(0, 3):
                actual_hess[i, j] = (
                    (hess[i, j] * norm - grad[i] * np.dot(grad, hess[j, :]) / norm) / norm ** 2.0
                )

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

    def _calculate_curvature(self, point, grad, type="max"):
        r"""type is either "max", "gaussian", "mean", "shape-index"."""
        if self.fhess is None:
            # Approximate the curvature.
            pass
        # Calculate curvature.
        return self._analytic_curvature(point, grad, type=type)

    def _initial_triangle_point(self, p0, delta):
        r"""
        Given a point on the isosurface, construct the first initial triangle.

        Returns
        -------
        (Vertex, Vertex, Vertex) : The three vertices of the triangle.
        """
        normal = normal_on_surface(self.fgrad, p0)

        # Construct orthonormal basis on tangent space at p0.
        if normal[0] > 0.5 or normal[1] > 0.5:
            grad1 = np.array([normal[1], normal[0], 0.])
        else:
            grad1 = np.array([-normal[2], 0., normal[0]])
        grad2 = np.cross(normal, grad1)
        grad1, grad2 = grad1 / np.linalg.norm(grad1), grad2 / np.linalg.norm(grad2)
        # Rotate teh gradient by pi/3
        grad_rot = grad1 * np.cos(np.pi / 3.0) + grad2 * np.sin(np.pi / 3.0)

        # Calculate the size to step out.
        if self.fhess is None:
            # If hessian is not provided, use move based on self.delta.
            curvature = 1.0
        else:
            # If hessian is provided, calculate the curvature.
            curvature = self._calculate_curvature(p0, self.fgrad(p0), type="max")
            curvature = 1.0 / curvature
            # curvature = delta * min(curvature, 1.5)
        # Take a step
        v1 = p0 + delta * curvature * grad1.copy()
        v2 = p0 + grad_rot * delta * curvature / np.linalg.norm(grad_rot)
        v2 = project_onto_surface(v2, self.func, self.isoval, self.fgrad)

        v1 = project_onto_surface(v1, self.func, self.isoval, self.fgrad)
        print("v1, v2projected",  v1, v2)
        seed_vertex = Vertex(p0, 0, normal)
        v1_vertex = Vertex(v1, 1, normal_on_surface(self.fgrad, v1))
        v2_vertex = Vertex(v2, 2, normal_on_surface(self.fgrad, v2))
        return seed_vertex, v1_vertex, v2_vertex

    def is_cap_triangle(self, angle):
        if angle <= 15.0 * np.pi / 180.:
            return True
        return False

    def is_needle_triangle(self, c_vert, b_vert, a_vert, needle_ratio=3):
        r"""
        cap_ratio : float
            If the max length / min length of triangle is greater than cap_ratio.
            Then it is a needle_triangle.

        """
        # See paper "A Robust Procedure to Eliminate Degenerate Faces from Triangle Meshes".
        dist_12 = distance_between_points(c_vert.coord, b_vert.coord)
        dist_23 = distance_between_points(b_vert.coord, a_vert.coord)
        dist_13 = distance_between_points(c_vert.coord, a_vert.coord)
        print("Is Needle?")
        print(dist_12, dist_23, dist_13)
        print("Distance", distance_between_points(b_vert.coord, a_vert.coord))
        print("Ratio", max(dist_12, dist_23, dist_13) / min(dist_12, dist_23, dist_13))
        ratio = max(dist_12, dist_23, dist_13) / min(dist_12, dist_23, dist_13)
        if ratio > needle_ratio:
            return True
        return False

    @staticmethod
    def radius_of_curvature(v1, v2, pt, normal):
        r"""Obtained from Karkanis paper. Edge from 1->2."""
        # Find closest Vertex.
        dist_13 = np.linalg.norm(v1.coord - pt)
        dist_23 = np.linalg.norm(v2.coord - pt)
        if dist_13 < dist_23:
            y = v1
            dist = dist_13
        else:
            y = v2
            dist = dist_23
        normaly = y.normal
        # get normal
        ang = angle_between_vectors(normal, normaly)
        return dist / (2.0 * np.sin(ang / 2.0))

    def attempt_create_isosceles(self, edge, delta):
        # edge is ActiveEdge
        # delta is float.
        midpt = (edge.edge[0].coord + edge.edge[1].coord) / 2.0
        normal = self.fgrad(midpt)
        perpendicular = edge.perpendicular_from_midpt(midpt, normal)

        if self.fhess is None:
            # Attempt Karkanis and Stewart's approach of estimating curvature.

            # Take a step so that it is a equilateral triangle
            dist_edge = distance_between_points(edge.edge[0].coord, edge.edge[1].coord)
            pt = step_equilateral_triangle(midpt, perpendicular, dist_edge)
            pt = project_onto_surface(pt, self.func, self.isoval, self.fgrad)

            radius = self.radius_of_curvature(edge.edge[0], edge.edge[1], pt,
                                              normal_on_surface(self.fgrad, pt))
            length_side = delta * min(radius, 1.5)
            pt = step_isosceles_triangle(pt, midpt, normal, dist_edge, length_side)
        else:
            # If fhess is present then analytically calculate the hessian.
            curv = self._analytic_curvature(
                edge.midpt(), self.fgrad(edge.midpt()), type="max")
            radius = 1.0 / curv
            length_side = delta * min(radius, 1.5)
            pt = step_equilateral_triangle(midpt, perpendicular, length_side)

        # Need to limit the length of this using some heuristic!!!!
        pt = project_onto_surface(pt, self.func, self.isoval, self.fgrad)
        vertex = Vertex(pt, self._vert_tris.length_vertices(), normal_on_surface(self.fgrad, pt))
        return vertex

    def pass_first_test(self, v1, v2, v3):
        # Check if new edges angles greater than 45 degrees. v3 is the new point and so
        # the new edges are (2->3) and (3->1). Original edge is (1->2)
        edge1 = v2.coord - v1.coord
        edge2 = v3.coord - v2.coord
        ang1 = angle_between_vectors(edge2, -edge1)
        ang2 = angle_between_vectors(v3.coord - v1.coord, edge1)

        if ang1 >= 0.785398 and ang2 >= 0.785398:
            return True
        return False

    def radius_second_test(self, longest_edge, delta):
        # Obtain radius from Appendix for the second test.
        l_prime = self._active_edges.longest_edge_distance
        q = delta
        return np.sqrt((longest_edge * 2.0 / 3.0 + q)**2.0 + (l_prime / 2.0)**2.0)

    def pass_second_test(self, v1, v2, v3, delta, if_earcutting=False):
        assert isinstance(v1, Vertex)
        assert isinstance(v2, Vertex)
        assert isinstance(v3, Vertex)
        # Get the longest edge of the triangle (v1, v2, v3).
        dist_12 = distance_between_points(v1.coord, v2.coord)
        dist_23 = distance_between_points(v2.coord, v3.coord)
        dist_31 = distance_between_points(v3.coord, v1.coord)
        l = max(dist_12, dist_23, dist_31)

        rad = self.radius_second_test(l, delta)
        indices = self._vert_tris.search_radius((v1.coord + v2.coord + v3.coord) / 3.0, rad)
        nbhr_triangle = v1.triangles + v2.triangles

        for i in indices:
            nbhr_triangle += self._vert_tris.get_vertex(i).triangles
        is_overlap = self._vert_tris.is_overlap_of_other_triangles(v1, v2, v3, nbhr_triangle)
        print("Is overlap ? ", is_overlap)

        if not if_earcutting:
            if not is_overlap:
                for t in np.unique(nbhr_triangle):
                    tri = self._vert_tris.get_triangle(t)
                    # Since this computes the distance between two triangles.
                    # Then it should ignore when they are sharing two indices together.
                    if len(set(tri.vindices).intersection({v1.index, v2.index, v3.index})) == 0:
                        dist = tri.compute_distance_two_triangle(v1, v2, v3)
                        # Added dist becauase of it finding the ti
                        if dist <= l / 3.0 and dist <= tri.get_longest_edge_of_triangle() / 3.0:
                            return False

        return not is_overlap

    def growing_phase(self, delta):
        r"""Based on Karkanis and Stewart's algorithm."""
        global count
        count = 0
        while len(self._active_edges) != 0:
            print("count ", count)
            edge = self._active_edges.pop()
            assert isinstance(edge, _ActiveEdge)
            v1, v2 = edge.get_vertices()
            print("edge index", v1.index, v2.index)

            # Get the other two, adjacent edges and calculate their respective front angle.
            other_edges, other_indices, is_gap = self._active_edges.get_adjacent_edge(edge)
            print([e.get_vertex_ind() for e in other_edges], other_indices, len(self._active_edges))
            (c, b, a) = self._active_edges.get_correct_orientation(edge, other_edges.pop())
            angle1 = calculate_front_angle(c.coord, b.coord, a.coord, c.normal)
            (c1, b1, a1) = self._active_edges.get_correct_orientation(edge, other_edges.pop())
            angle2 = calculate_front_angle(c1.coord, b1.coord, a1.coord, c1.normal)

            if angle1 <= 75.0 * np.pi / 180:
                # If front angle is less than 80 degrees then run Earcutting.
                if self.pass_second_test(c, b, a, delta, if_earcutting=True):
                    print("EARCUTTING 1")
                    self._active_edges.add_ear_cutting(c, b, a, other_indices[1], is_gap[1])
                    self._vert_tris.add_ear_cutting(c, b, a)
                else:
                    print("EARCUTTING 1 FAILED: No point Added")
                    # Add to the list of gap boundaries.
                    self._active_edges.add_to_gap(edge)

            elif angle2 <= 75.0 * np.pi / 180:
                # If front angle is less than 80 degrees then run Earcutting.
                if self.pass_second_test(c1, b1, a1, delta, if_earcutting=True):
                    print("EARCUTTING 2")
                    self._active_edges.add_ear_cutting(c1, b1, a1, other_indices[0], is_gap[0])
                    self._vert_tris.add_ear_cutting(c1, b1, a1)
                else:
                    print("EARCUTTING 2 FAILED: No point Added")
                    # Add to the list of gap boundaries.
                    self._active_edges.add_to_gap(edge)
            else:
                v3 = self.attempt_create_isosceles(edge, delta)
                if self.pass_first_test(v1, v2, v3):
                    test2 = self.pass_second_test(v1, v2, v3, delta)
                    if test2:
                        print("TEST 2 SUCCEED: Add Point")
                        self._vert_tris.add_new_vertex_with_triangle(v1, v2, v3)
                        self._active_edges.add_new_point_and_triangle(v1, v2, v3)
                    else:
                        print("TEST 2 FAILED: No point Added")
                        # Add to the list of gap boundaries.
                        self._active_edges.add_to_gap(edge)
                else:
                    print("TEST 1 FAILED: No point Added")
                    # Add to the list of gap boundaries.
                    self._active_edges.add_to_gap(edge)

            count += 1
            # if count >= 80:
            if count % 20000 == 0:
                self.plot_surface(extra_verts=[edge.get_vertices()[0], edge.get_vertices()[1]])
            print("")

        # mesh.test_triangle_information_in_vertices()

    def setup_front_points_angles_and_edge_information_to_vertices(self):
        r"""
        Setup up "_front" attribute for the filling phase.
        Algorithm is based on the assumption that each vertex on boundary is only in two edges.
        """
        front_dict = {}  # Gets inserted into Value Sorted at the index Dict.
        for g in self._active_edges._gap_edge_list:
            v1, v2 = g.get_vertices()

            # Add these new gap vertices
            self._vert_tris.get_vertex(v1.index).after_vert_ind = v2.index
            assert v1.after_vert_ind is not None
            if v1.index not in front_dict:
                self._vert_tris.get_vertex(v1.index).convert_boundary()
                front_dict[v1.index] = None
            else:
                # If it is in it, then it must have been a previous edge such that its v2, then!
                assert v1.before_vert_ind is not None
                ang = calculate_front_angle(
                    self._vert_tris.get_vertex(v1.index).coord,
                    self._vert_tris.get_vertex(v1.before_vert_ind).coord,
                    self._vert_tris.get_vertex(v1.after_vert_ind).coord,
                    self._vert_tris.get_vertex(v1.index).normal
                )
                front_dict[v1.index] = ang

            self._vert_tris.get_vertex(v2.index).before_vert_ind = v1.index
            assert v2.before_vert_ind is not None
            if v2.index not in front_dict:
                self._vert_tris.get_vertex(v2.index).convert_boundary()
                front_dict[v2.index] = None
            else:
                # If it is in it, then there must have been a edge such that this v2 was a v1 before.
                assert v2.after_vert_ind is not None
                ang = calculate_front_angle(
                    self._vert_tris.get_vertex(v2.index).coord,
                    self._vert_tris.get_vertex(v2.before_vert_ind).coord,
                    self._vert_tris.get_vertex(v2.after_vert_ind).coord,
                    self._vert_tris.get_vertex(v2.index).normal
                )
                front_dict[v2.index] = ang
        print(front_dict)
        self._front.setup_front_angles(front_dict)

    def is_cycle(self, b, a, length):
        r"""Checks if it is a cycle of a prescribed length00."""
        v = a
        for i in range(length - 2):
            v = self._vert_tris.get_vertex(v.after_vert_ind)
        if v.index == b.index:
            return True
        return False

    def update_front_angles(self, vert_list):
        for v in vert_list:
            ang = calculate_front_angle(
                v.coord,
                self._vert_tris.get_vertex(v.before_vert_ind).coord,
                self._vert_tris.get_vertex(v.after_vert_ind).coord,
                v.normal
            )
            self._front.update_front_angle(v.index, ang)

    def attempt_filling_algorithm(self, c, b, a, n, is_four_cycle=False, is_five_cycle=False):
        r"""
        The main algorithm to fill in triangles to connect, c, b, a and new_vertex n.
        if_cycle
        """
        if is_four_cycle:
            assert n.index == b.before_vert_ind
        if is_five_cycle:
            assert n.index == b.before_vert_ind

        # If the New point is next-door neighbor to b or a.
        if (n.index == b.before_vert_ind or n.index == a.after_vert_ind) and not is_five_cycle:
            # Compare two triangle sets [(c, n, b) and (c, n, a)] and [(c, b, a), (n, b, a)]
            triangle_set1 = [(c, n, b), (c, n, a)]
            triangle_set2 = [(c, b, a), (n, b, a)]
            triangle_sets = [triangle_set1, triangle_set2]

            tri_collision_list = self._vert_tris.triangle_collision_list(b, a, n)
            index, not_found = self._vert_tris.best_triangle_set_with_overlap(
                triangle_sets, tri_collision_list
            )

            if not_found:
                # All triangle sets considered overlaped with collisioned triangles.
                raise AllTriangleSetsOverlapError()
            tri_set = triangle_sets[index]
            self._vert_tris.add_triangle_set(tri_set)

            if is_four_cycle:
                self._vert_tris.remove_boundary_info_vertex(a)
                self._vert_tris.remove_boundary_info_vertex(b)
                self._vert_tris.remove_boundary_info_vertex(n)
            elif n.index == b.before_vert_ind:
                # Remove before vertex from boundarry
                print("n is bb", b.before_vert_ind)
                self._vert_tris.remove_boundary_info_vertex(b)
                self._front.remove_vertex(b.index)
                # Update boundary info on new_vert and after_vert
                n.after_vert_ind = a.index
                a.before_vert_ind = n.index
                self.update_front_angles([n, a])
            elif n.index == a.after_vert_ind:
                print("n is aa", a.after_vert_ind)
                # Remove after vertex from boundary.
                self._vert_tris.remove_boundary_info_vertex(a)
                self._front.remove_vertex(a.index)
                n.before_vert_ind = b.index
                b.after_vert_ind = n.index
                # Update front angle
                self.update_front_angles([n, b])
        else:
            tri_collision_list = self._vert_tris.triangle_collision_list(
                b, a, n, other_vertices_indices=(n.after_vert_ind, n.before_vert_ind))
            nb = self._vert_tris.get_vertex(n.before_vert_ind)
            na = self._vert_tris.get_vertex(n.after_vert_ind)
            print("nb na", nb.index, na.index)

            triangle_set1 = [(c, b, a), (b, a, n), (a, n, nb)]
            triangle_set2 = [(c, b, a), (b, a, n), (b, n, na)]
            triangle_set3 = [(c, n, b), (c, n, a), (n, a, nb)]
            triangle_set4 = [(c, n, b), (c, n, a), (n, b, na)]
            triangle_sets = [triangle_set1, triangle_set2, triangle_set3, triangle_set4]

            index, not_found = self._vert_tris.best_triangle_set_with_overlap(
                triangle_sets, tri_collision_list
            )
            if not_found:
                # All triangle sets considered overlaped with collisioned triangles.
                raise AllTriangleSetsOverlapError()
            tri_set = triangle_sets[index]

            print("Index won", index)
            self._vert_tris.add_triangle_set(tri_set)

            if is_five_cycle:
                [self._vert_tris.remove_boundary_info_vertex(v) for v in [b, a, n, nb]]
            else:
                if index == 0 or index == 2:
                    nb.after_vert_ind = a.index
                    a.before_vert_ind = nb.index
                    b.after_vert_ind = n.index
                    n.before_vert_ind = b.index

                    if a.after_vert_ind == nb.index:
                        self._vert_tris.remove_boundary_info_vertex(a)
                        self._vert_tris.remove_boundary_info_vertex(nb)
                        self._front.remove_vertex(a.index)
                        self._front.remove_vertex(nb.index)
                        self.update_front_angles([b, n])
                    else:
                        self.update_front_angles([nb, a, b, n])
                elif index == 1 or index == 3:
                    na.before_vert_ind = b.index
                    a.before_vert_ind = n.index
                    b.after_vert_ind = na.index
                    n.after_vert_ind = a.index

                    if na.index == b.before_vert_ind:
                        self._vert_tris.remove_boundary_info_vertex(na)
                        self._vert_tris.remove_boundary_info_vertex(b)
                        self._front.remove_vertex(b.index)
                        self._front.remove_vertex(na.index)
                        self.update_front_angles([a, n])
                    else:
                        self.update_front_angles([na, a, b, n])

    def attempt_complete_one_triangle(self, c, b, a):
        triangle_collision = self._vert_tris.triangle_collision_list(c, b, a)
        overlap = self._vert_tris.is_overlap_of_other_triangles(c, b, a, triangle_collision)

        if overlap:
            print("##### ERRROR ######")
            self.plot_surface([(c, "r"), (b, "g"), (a, "m")])
            raise RuntimeError("Couldn't find any triangle set including completing (c, b, a).")
        else:
            self._vert_tris.complete_one_triangle(c, b, a)
            a.before_vert_ind = b.index
            b.after_vert_ind = a.index
            self.update_front_angles([b, a])

    def filling_phase(self):
        count = 0
        while len(self._front) != 0:
            vindex, angle = self._front.pop_front_angle()
            print("index", vindex, "angle", angle, "count", count)
            c = self._vert_tris.get_vertex(vindex)
            b = self._vert_tris.get_vertex(c.before_vert_ind)
            a = self._vert_tris.get_vertex(c.after_vert_ind)
            print("c, b, a index", c.index, b.index, a.index)
            # self._front.remove_vertex(c.index)

            # if count >= 201:
            #     self.plot_surface([(c, "r"), (b, "g"), (a, "m")])

            if self.is_cycle(b, a, 3):
                # If 3 Cycle, then complete triangle.
                self._vert_tris.complete_one_triangle(c, b, a)
                self._front.remove_vertices([b, a])
            elif self.is_cycle(b, a, 4):
                # If 4 Cycle, then find best two triangles.
                print("Four Cycle")
                bb = self._vert_tris.get_vertex(b.before_vert_ind)
                self.attempt_filling_algorithm(
                    c, b, a, bb,
                    is_four_cycle=True
                )
                self._front.remove_vertices([b, a, bb])
            elif self.is_cycle(b, a, 5):
                # If 5 Cycle, then find best three triangles.
                print("Five cycle")
                bb = self._vert_tris.get_vertex(b.before_vert_ind)
                aa = self._vert_tris.get_vertex(a.after_vert_ind)
                self.attempt_filling_algorithm(
                    c, b, a, bb,
                    is_five_cycle=True
                )
                self._front.remove_vertices([b, a, aa, bb])
            else:
                # Get closest neighbors of current
                dist_cb = distance_between_points(c.coord, b.coord)
                dist_ca = distance_between_points(c.coord, a.coord)
                radius = max(dist_ca, dist_cb)
                indices, before_plane, after_plane = \
                    self._vert_tris.search_collision_points_around_current(c, b, a, radius)
                print("Current Neighbors :", indices)
                # If there are nearby vertices of current.
                if len(indices) != 0:
                    not_found = False
                    for i in range(len(indices)):
                        print("try Neighbor ", indices[i])
                        try:
                            n = self._vert_tris.get_vertex(indices[i])
                            self.attempt_filling_algorithm(c, b, a, n)
                            break
                        except AllTriangleSetsOverlapError:
                            # Try the next closest vertex.
                            if i == len(indices) - 1:
                                not_found = True
                    if not_found:
                        # Try completing the (c, b, a) triangle. If fail, algorithm raises Error.
                        # self.attempt_complete_one_triangle(c, b, a)
                        index = self._vert_tris.search_collision_points_around_before_after(
                            b, a, before_plane, after_plane, radius)
                        print("Before/After Neighbor found: ", index)
                        if index is not None:
                            try:
                                n = self._vert_tris.get_vertex(index)
                                self.attempt_filling_algorithm(c, b, a, n)
                            except AllTriangleSetsOverlapError:
                                # Try completing the (c, b, a) triangle. If fail, algorithm raises Error.
                                print("Failed: Try completing one triangle.")
                                self.attempt_complete_one_triangle(c, b, a)
                        else:
                            self.attempt_complete_one_triangle(c, b, a)
                else:
                    # Get nearby vertices of Before and After.
                    index = self._vert_tris.search_collision_points_around_before_after(
                        b, a, before_plane, after_plane, radius)
                    print("Before/After Neighbor found: ", index)
                    if index is not None:
                        try:
                            n = self._vert_tris.get_vertex(index)
                            self.attempt_filling_algorithm(c, b, a, n)
                        except AllTriangleSetsOverlapError:
                            # Try completing the (c, b, a) triangle. If fail, algorithm raises Error.
                            print("Failed: Try completing one triangle.")
                            self.attempt_complete_one_triangle(c, b, a)
                    else:
                        self.attempt_complete_one_triangle(c, b, a)

            self._vert_tris.remove_boundary_info_vertex(c)
            # if count >= 201:
            #     self.plot_surface([(c, "r"), (b, "g"), (a, "m")])
            count += 1
            print("")

    def __call__(self, delta=0.1, tol_root=1e-10, max_length=1.5):
        r"""

        Parameters
        ----------
        delta : float
            Ratio of triangle size. Between zero and 1.
        tol_root : float
            Tolerance for solving the root equation
        max_length : float
            Maximum height of a edge in a triangle.

        Returns
        -------
        (ndarray(float), ndarray(int)):
            Returns the coordinates of vertices and triangle indices
        Notes
        -----
        If hess is not included, then it is approximated.

        """
        # Add the first initial Point and Triangle..
        seed = find_random_point_on_isosurface(self.func, self.isoval)
        v1, v2, v3 = self._initial_triangle_point(seed, delta)
        self._vert_tris.add_first_vertices_and_triangle(v1, v2, v3)
        self._active_edges.add_first_vertices_and_triangle(v1, v2, v3)
        self.plot_surface()

        # Start the growing phase.
        self.growing_phase(delta)
        print("Finish Growing Phase.")
        # self.plot_surface()
        # self.plot_mesh_angles_ratio()

        # Start the Filling Phase.
        self.setup_front_points_angles_and_edge_information_to_vertices()
        self.filling_phase()
        print("Finish Filling Phase")
        # self.plot_surface()
        # self.plot_mesh_angles_ratio()
        return self.to_array()

    def laplacian(self):
        pass


class AllTriangleSetsOverlapError(RuntimeError):
    """
    Raised when all possible triangle sets (collection of triangles) overlap with neighbor
        triangles.
    """
    pass
