from functools import reduce
from itertools import product

import numpy as np
from functions import dictionary, flatten, get_combinations
from scipy.spatial.distance import cdist

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class SimpleGraph:
    def __init__(self, vertices_count, edges, labels=[], coordinates=[]):

        self.vertices_count = vertices_count
        self.vertices = np.array(range(vertices_count))
        self.edges = np.array(
            [(v1, v2) if v1 < v2 else (v2, v1) for v1, v2 in edges]
        )
        self.labels = np.array(labels)
        self.coordinates = np.array(coordinates)
        self.vertex_degrees = None
        self.adjacency_list = None
        self.distmatrix = None

    def calc_dist_matrix(self):

        if len(self.coordinates) == 0:
            raise AttributeError("The graph geometry is not specified!")
        self.distmatrix = cdist(
            self.coordinates, self.coordinates, metric="euclidean"
        )
        return self.distmatrix

    def calc_adjacency_list(self):

        self.adjacency_list = [[] for x in range(self.vertices_count)]
        for v1, v2 in self.edges:
            self.adjacency_list[v1].append(v2)
            self.adjacency_list[v2].append(v1)
        return self.adjacency_list

    def calc_vertex_degrees(self):

        if self.adjacency_list is None:
            self.calc_adjacency_list()
        self.vertex_degrees = np.array(
            [len(neighbors) for neighbors in self.adjacency_list]
        )
        return self.vertex_degrees

    def copy(self):

        copy = SimpleGraph(
            self.vertices_count,
            [],
            labels=self.labels,
            coordinates=self.coordinates,
        )
        copy.edges = np.array(self.edges)
        if self.adjacency_list is not None:
            copy.adjacency_list = np.array(
                [neighbors[:] for neighbors in self.adjacency_list]
            )
        return copy

    def find_connected_parts(self):

        connected_parts = []
        traversed = [0] * self.vertices_count
        for vertex in self.vertices:
            if traversed[vertex] == 0:
                connected_vertices = [vertex]
                for v in connected_vertices:
                    traversed[v] = 1
                    for neighbor in self.adjacency_list[v]:
                        if traversed[neighbor] == 0:
                            connected_vertices += [neighbor]
                            traversed[neighbor] = 1
                connected_parts += [connected_vertices]
        return connected_parts

    def remove_edges(self, edges):

        d = {tuple(edge): i for i, edge in enumerate(self.edges)}
        indexes = [d[tuple(edge)] for edge in edges]
        self.edges = np.delete(self.edges, indexes, axis=0)
        self.calc_adjacency_list()
        self.calc_vertex_degrees()
        return None

    def is_chain(self):

        vertex_degrees = list(set(self.vertex_degrees))
        types_count = len(vertex_degrees)
        if (types_count == 1 and vertex_degrees[0] == 1) or (
            types_count == 2
            and (
                (vertex_degrees[0] == 1 and vertex_degrees[1] == 2)
                or (vertex_degrees[1] == 1 and vertex_degrees[0] == 2)
            )
        ):
            return True
        return False

    def draw(self):

        import pylab as pl
        import mpl_toolkits.mplot3d as a3
        import matplotlib.pyplot as plt

        ax = a3.Axes3D(pl.figure())
        # Plot vertices
        for i, c in enumerate(self.coordinates):
            ax.scatter(*c, color="b", s=200, zorder=50)
            ax.text(*c, self.labels[i] + str(i), fontsize=10, zorder=100)
        # Plot edges
        if len(self.edges) != 0:
            n1 = np.array([self.coordinates[i_1] for i_1, i_2 in self.edges])
            n2 = np.array([self.coordinates[i_2] for i_1, i_2 in self.edges])
            x = [[n1[i][0], n2[i][0]] for i in range(len(n1))]
            y = [[n1[i][1], n2[i][1]] for i in range(len(n1))]
            z = [[n1[i][2], n2[i][2]] for i in range(len(n1))]
            for i in range(len(x)):
                ax.plot(x[i], y[i], z[i], linewidth=3, color="b", zorder=1)
        # plt.title(self.name)
        plt.tight_layout()
        plt.axis("off")
        plt.show()
        return None


class GraphComparator:
    def __init__(self, graph_1, graph_2):

        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.bijections = None
        self.delta_lengths = []
        if graph_1.vertex_degrees is None:
            graph_1.calc_vertex_degrees()
        if graph_2.vertex_degrees is None:
            graph_2.calc_vertex_degrees()
        self.invariants_1 = np.array(graph_1.vertex_degrees)
        self.invariants_2 = np.array(graph_2.vertex_degrees)
        self.invariants_dict_1 = dictionary(
            self.invariants_1, graph_1.vertices
        )
        self.invariants_dict_2 = dictionary(
            self.invariants_2, graph_2.vertices
        )

    def maps_vertices(self, set_1, set_2):

        mapping = []
        d_1 = dictionary(self.invariants_1[set_1], set_1)
        d_2 = dictionary(self.invariants_2[set_2], set_2)
        for k, v in d_1.items():
            mapping += [[v, d_2.get(k)]]
        return mapping

    def find_isomorphic_bijections(self, all=True):
        def maps_coordination_shells(
            pairs, shells_mapping, traversed_1, traversed_2
        ):

            shells_mappings = []
            for pair in shells_mapping:
                neighbors_pairs = []
                neighbors_1 = np.array(
                    [
                        n
                        for n in self.graph_1.adjacency_list[pair[0]]
                        if traversed_1[n] == 0
                    ]
                )
                if len(neighbors_1) != 0:
                    traversed_1[neighbors_1] = 1
                neighbors_2 = np.array(
                    [
                        n
                        for n in self.graph_2.adjacency_list[pair[1]]
                        if traversed_2[n] == 0
                    ]
                )
                if len(neighbors_2) != 0:
                    traversed_2[neighbors_2] = 1
                if len(neighbors_1) != len(neighbors_2):
                    return False
                if len(neighbors_1) != 0 and len(neighbors_2) != 0:
                    for vertices, equiv_vertices in self.maps_vertices(
                        neighbors_1, neighbors_2
                    ):
                        if equiv_vertices is None or len(vertices) != len(
                            equiv_vertices
                        ):
                            return False
                        elif len(vertices) == 1:
                            neighbors_pairs.append(
                                [list(product(vertices, equiv_vertices))]
                            )
                        else:
                            neighbors_pairs.append(
                                get_combinations(vertices, equiv_vertices)
                            )
                    neighbors_pairs = reduce(
                        (lambda x, y: list(product(x, y))), neighbors_pairs
                    )
                    shells_mappings.append(
                        [flatten(c) for c in neighbors_pairs]
                    )
            if len(shells_mappings) == 0:
                pairs.sort()
                edges_2 = {tuple(e): True for e in self.graph_2.edges}
                for v1, v2 in self.graph_1.edges:
                    if (
                        edges_2.get((pairs[v1][1], pairs[v2][1])) is None
                        and edges_2.get((pairs[v2][1], pairs[v1][1])) is None
                    ):
                        return False
                isomorphic_bijections.append(pairs)
                return True
            shells_mappings = reduce(
                (lambda x, y: [flatten(pr) for pr in list(product(x, y))]),
                shells_mappings,
            )
            for shells_mapping in shells_mappings:
                new_pairs = pairs[:]
                new_pairs.extend(shells_mapping)
                maps_coordination_shells(
                    new_pairs,
                    shells_mapping,
                    np.array(traversed_1),
                    np.array(traversed_2),
                )
            return False

        isomorphic_bijections = []
        if (
            self.graph_1.vertices_count == self.graph_2.vertices_count
            and len(self.graph_1.edges) == len(self.graph_2.edges)
            and np.array_equal(
                np.sort(self.invariants_1), np.sort(self.invariants_2)
            )
        ):
            pairs = list(
                product([0], self.invariants_dict_2[self.invariants_1[0]])
            )
            for pair in pairs:
                traversed_1 = np.zeros(
                    self.graph_1.vertices_count, dtype="int"
                )
                traversed_2 = np.zeros(
                    self.graph_1.vertices_count, dtype="int"
                )
                traversed_1[pair[0]] = 1
                traversed_2[pair[1]] = 1
                if (
                    maps_coordination_shells(
                        [pair], [pair], traversed_1, traversed_2
                    )
                    and not all
                ):
                    self.bijections = isomorphic_bijections
                    [b.sort() for b in self.bijections]
                    self.bijections = np.array(self.bijections)
                    return self.bijections
        self.bijections = isomorphic_bijections
        [b.sort() for b in self.bijections]
        self.bijections = np.array(self.bijections)
        return self.bijections

    def compare_geometry(self):

        if self.graph_1.distmatrix is None:
            self.graph_1.calc_dist_matrix()
        if self.graph_2.distmatrix is None:
            self.graph_2.calc_dist_matrix()
        if self.bijections is None:
            self.find_isomorphic_bijections()
        if len(self.bijections) == 0:
            return False
        else:
            n = (
                self.graph_1.vertices_count
                * (self.graph_1.vertices_count - 1)
                / 2.0
            )
            for i, b in enumerate(self.bijections):
                delta = 0.0
                n12 = b[:, 1]
                for j in range(len(n12) - 1):
                    for k in range(j + 1, len(n12)):
                        delta += abs(
                            self.graph_1.distmatrix[j][k]
                            - self.graph_2.distmatrix[n12[j]][n12[k]]
                        )
                self.delta_lengths.append(delta / n)
            return True
