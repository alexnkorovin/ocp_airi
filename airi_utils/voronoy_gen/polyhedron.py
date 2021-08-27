import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np
import pylab as pl
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

import geometry as gm
from graph import GraphComparator, SimpleGraph


class Polygon:
    def __init__(self, vertices, labels, find_order=True, name="Polygon"):

        self.name = name
        self.n_v = len(vertices)
        self.symbol = str(self.n_v)
        if find_order:
            order = self.find_order(vertices)
        else:
            order = range(self.n_v)
        self.labels = np.array(labels)[order]
        self.vertices = np.array(vertices)[order]
        self.edges = np.array(
            [[i, (i + 1) % self.n_v] for i in range(self.n_v)]
        )
        self.centroid = gm.calc_centroid(vertices)
        self.simplexes = self.find_simplexes()
        self.area = self.calc_area()
        self.norm = self.calc_norm()
        self.color = [0.2365083, 0.78865635, 0.66769619]
        self.graph = None
        self.distmatrix = None

    @staticmethod
    def find_order(vertices):

        vs = vertices
        n = len(vertices)
        vs = np.array(
            [
                [p[0], p[2]]
                for p in gm.orient(vs, gm.calc_centroid(vs[:-1]), vs[0], vs[1])
            ]
        )
        angles = gm.calc_angles([[0, 2]] * n, [[0, 0]] * n, vs)
        angles[vs[:, 0] < 0] = 2 * np.pi - angles[vs[:, 0] < 0]
        order = list(enumerate(angles))
        order.sort(key=lambda x: x[1])
        order = [o for o, a in order]
        return order

    def calc_dist_matrix(self):

        self.distmatrix = cdist(
            self.vertices, self.vertices, metric="euclidean"
        )
        return self.distmatrix

    def build_graph(self, mod=0):

        """ 0 convex hull, 1 - convex hull and centroid, 2 - full graph, 3 - full graph with centroid"""
        vertices_count = len(self.vertices)
        if mod == 0:
            self.graph = SimpleGraph(
                vertices_count, self.edges, coordinates=self.vertices
            )
        elif mod == 1:
            coordinates = self.vertices[:]
            coordinates.append(self.centroid)
            edges = self.edges[:]
            edges.extend(
                [[vertices_count, i] for i in range(vertices_count)],
                coordinates=coordinates,
            )
            self.graph = SimpleGraph(vertices_count + 1, self.edges)
        elif mod == 2:
            edges = [
                [i, j]
                for i in range(vertices_count)
                for j in range(i + 1, vertices_count)
            ]
            self.graph = SimpleGraph(vertices_count, edges)
        elif mod == 3:
            vertices_count += 1
            coordinates = self.vertices[:]
            coordinates.append(self.centroid)
            edges = [
                [i, j]
                for i in range(vertices_count)
                for j in range(i + 1, vertices_count)
            ]
            self.graph = SimpleGraph(
                vertices_count, edges, coordinates=coordinates
            )
        else:
            raise AttributeError("Unrecognized mod!")
        return self.graph

    def calc_norm(self):

        norm = np.cross(
            self.vertices[0] - self.centroid, self.vertices[1] - self.centroid
        )
        norm /= np.linalg.norm(norm)
        norm += self.centroid
        self.norm = norm
        return self.norm

    def reverse(self):

        self.vertices = self.vertices[::-1]
        self.labels = self.labels[::-1]
        self.edges = np.fliplr(self.edges[::-1])
        self.norm = self.calc_norm()
        return None

    def find_simplexes(self):

        self.simplexes = [
            (self.vertices[v_ind - 1], self.vertices[v_ind], self.centroid)
            for v_ind in range(self.n_v)
        ]
        return self.simplexes

    def calc_area(self):

        self.area = 0.5 * sum(
            [abs(np.dot(v1 - v2, v3 - v2)) for v1, v2, v3 in self.simplexes]
        )
        return self.area

    def is_inside(self, pt, tol=0.001):

        inside = False
        v1 = pt - self.centroid
        v2, v3 = self.vertices[:2] - self.centroid
        if not gm.are_coplanar([v1, v2, v3], tol):
            return False
        plg = np.vstack((self.vertices, pt))
        plg = [
            [p[0], p[2]]
            for p in gm.orient(plg, gm.calc_centroid(plg[:-1]), plg[0], plg[1])
        ]
        pt, plg = plg[-1], np.array(plg[:-1])
        x, y = pt
        for i in range(1, self.n_v + 1):
            x1, y1 = plg[i - 1]
            x2, y2 = plg[i % self.n_v]
            if y1 != y2 and min(y1, y2) <= y < max(y1, y2) and x < max(x1, x2):
                x_ints = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                if abs(x_ints - x) < tol:
                    return False
                if x1 == x2 or x < x_ints:
                    inside = not inside
        return inside

    def draw(self):

        ax = a3.Axes3D(pl.figure())
        # Plot vertices of the polygon
        for i, v in enumerate(self.vertices):
            ax.scatter(v[0], v[1], v[2], color="b")
        # Plot the polygon
        f = a3.art3d.Poly3DCollection([self.vertices], linewidths=2, alpha=0.4)
        f.set_color(self.color)
        f.set_edgecolor("k")
        ax.add_collection3d(f)
        plt.axis("off")
        plt.show()
        return None

    def __str__(self):

        data = "###################################################\n"
        data += "Name : " + self.name + "\n"
        data += "Vertices :\n"
        data += str(len(self.vertices)) + "\n"
        for i, v in enumerate(self.vertices):
            data += (
                str(self.labels[i])
                + "\t"
                + "%14.8f" % v[0]
                + "%14.8f" % v[1]
                + "%14.8f" % v[2]
                + "\n"
            )
        data += "Edges :\n"
        data += str(len(self.edges)) + "\n"
        for i, edge in enumerate(self.edges):
            data += (
                str(i + 1)
                + "\t"
                + str(self.labels[edge[0]])
                + "\t"
                + str(self.labels[edge[1]])
                + "\n"
            )
        data += "Surface area : " + "%14.8f" % self.area + "\n"
        data += "\n###################################################"
        return data


class Polyhedron:
    def __init__(
        self, vertices, labels, faces, find_order=True, name="Polyhedron"
    ):

        self.name = name
        self.vertices = np.array(vertices)
        self.labels = np.array(labels)
        faces = [f[Polygon.find_order(self.vertices[f])] for f in faces]
        self.faces = np.array(
            [
                Polygon(self.vertices[f], self.labels[f], find_order=False)
                for f in faces
            ]
        )
        self.edges = [
            [faces[i][i1], faces[i][i2]]
            for i, f in enumerate(self.faces)
            for i1, i2 in f.edges
        ]
        self.edges = list(
            set([(i1, i2) if i1 < i2 else (i2, i1) for i1, i2 in self.edges])
        )
        self.edges.sort()
        self.symbol = self.calc_symbol()
        self.area = sum([f.area for f in self.faces])
        self.centroid = gm.calc_centroid(vertices)
        self.volume = self.calc_volume()
        self.face_reindex()
        self.graph = None
        self.distmatrix = None

    def calc_dist_matrix(self):

        self.distmatrix = cdist(
            self.vertices, self.vertices, metric="euclidean"
        )
        return self.distmatrix

    def build_graph(self, mod=0):

        """ 0 convex hull, 1 - convex hull and centroid, 2 - full graph, 3 - full graph with centroid"""
        vertices_count = len(self.vertices)
        if mod == 0:
            self.graph = SimpleGraph(
                vertices_count, self.edges, coordinates=self.vertices
            )
        elif mod == 1:
            coordinates = self.vertices[:]
            coordinates.append(self.centroid)
            edges = self.edges[:]
            edges.extend(
                [[vertices_count, i] for i in range(vertices_count)],
                coordinates=coordinates,
            )
            self.graph = SimpleGraph(vertices_count + 1, self.edges)
        elif mod == 2:
            edges = [
                [i, j]
                for i in range(vertices_count)
                for j in range(i + 1, vertices_count)
            ]
            self.graph = SimpleGraph(vertices_count, edges)
        elif mod == 3:
            vertices_count += 1
            coordinates = self.vertices[:]
            coordinates.append(self.centroid)
            edges = [
                [i, j]
                for i in range(vertices_count)
                for j in range(i + 1, vertices_count)
            ]
            self.graph = SimpleGraph(
                vertices_count, edges, coordinates=coordinates
            )
        else:
            raise AttributeError("Unrecognized mod!")
        return self.graph

    def face_reindex(self):

        for f in self.faces:
            if np.linalg.norm(f.norm - self.centroid) < np.linalg.norm(
                2 * f.centroid - f.norm - self.centroid
            ):
                f.reverse()
        return None

    def calc_volume(self):

        c = self.centroid
        pyramids_vs = []
        for f in self.faces:
            simplex_vs = []
            for s in f.simplexes:
                v1, v2, v3 = s - c
                simplex_vs += [abs(np.dot(v1, np.cross(v2, v3))) / 6]
            pyramids_vs += [sum(simplex_vs)]
        self.volume = sum(pyramids_vs)
        return self.volume

    def calc_symbol(self):

        rings = np.zeros(len(self.faces), dtype=np.int)
        for face in self.faces:
            rings[len(face.vertices)] += 1
        self.symbol = " ".join(
            [str(i) + "^" + str(r) for i, r in enumerate(rings) if r > 0]
        )
        return self.symbol

    def draw(self):

        ax = a3.Axes3D(pl.figure())
        # Plot polyhedron's vertices
        for i, v in enumerate(self.vertices):
            ax.scatter(*v, color="b")
            ax.text(*v, str(i + 1), fontsize=20, zorder=100)
        # Plot polyhedron's faces
        for face in self.faces:
            # ax.scatter(*face.norm, color='r')
            f = a3.art3d.Poly3DCollection(
                [face.vertices], linewidths=2, alpha=0.5
            )
            f.set_color(face.color)
            f.set_edgecolor("k")
            ax.add_collection3d(f)
        plt.axis("off")
        plt.show()
        return None

    def __str__(self):

        data = "###################################################\n\n"
        data += "Name : " + self.name + "\n"
        data += "Symbol : " + self.symbol + "\n"
        data += "Vertices : " + str(len(self.vertices)) + "\n"
        for i, v in enumerate(self.vertices):
            data += (
                str(self.labels[i])
                + "\t"
                + "%14.8f" % v[0]
                + "%14.8f" % v[1]
                + "%14.8f" % v[2]
                + "\n"
            )
        data += "Edges : " + str(len(self.edges)) + "\n"
        for i, edge in enumerate(self.edges):
            data += (
                str(i + 1)
                + "\t"
                + str(self.labels[edge[0]])
                + "\t"
                + str(self.labels[edge[1]])
                + "\n"
            )
        data += "Faces : " + str(len(self.faces)) + "\n"
        for i, face in enumerate(self.faces):
            data += (
                str(i + 1)
                + "\t"
                + "\t".join([str(le) for le in face.labels])
                + "\n"
            )
        data += "Surface area : " + "%14.8f" % self.area + "\n"
        data += "Volume : " + "%14.8f" % self.volume + "\n"
        data += "\n###################################################\n"
        return data


class FigureConstructor:
    def __init__(self, points, labels=None):

        self.points = np.array(points, dtype=np.float)
        if labels is None:
            self.labels = np.array(
                [str(i) for i in range(1, len(self.points) + 1)]
            )
        else:
            self.labels = np.empty((len(labels)), dtype=object)
            for i, l in enumerate(labels):
                self.labels[i] = l
            if len(set(self.labels)) != len(self.labels):
                raise ValueError("There are duplicates in labels!")
            if len(points) != len(set(labels)):
                raise ValueError(
                    "Number of labels is not fit numbers of points!"
                )
        self.hull = None
        self.figure = None

    def construct_convex_figure(self, name="figure", tol=0.01):

        self.hull = ConvexHull(self.points)
        if len(self.points) > len(self.hull.vertices):
            raise ValueError("The figure is not convex!")
        if gm.are_coplanar(self.points - gm.calc_centroid(self.points), 0.1):
            figure = Polygon(self.points, self.labels, name=name)
        else:
            faces = []
            for eq in self.hull.equations:
                indexes = list(
                    set(
                        [
                            i
                            for i, p in enumerate(self.points)
                            if abs(
                                eq[0] * p[0]
                                + eq[1] * p[1]
                                + eq[2] * p[2]
                                + eq[3]
                            )
                            < tol
                        ]
                    )
                )
                indexes.sort()
                faces += [tuple(indexes)]
            faces = np.array([np.array(f) for f in set(faces)])
            figure = Polyhedron(
                self.points[self.hull.vertices],
                self.labels[self.hull.vertices],
                faces,
                name=name,
            )
            figure.build_graph()
        return figure


class FigureComparator:
    def __init__(self, figure_1, figure_2):

        self.figure_1 = figure_1
        self.figure_2 = figure_2
        self.bijections = None
        self.delta_lengths = []

    def compare(self):

        if self.figure_1.graph is None:
            self.figure_1.build_graph()
        if self.figure_2.graph is None:
            self.figure_2.build_graph()
        gc = GraphComparator(self.figure_1.graph, self.figure_2.graph)
        is_isomorphic = gc.compare_geometry()
        self.bijections = gc.bijections
        self.delta_lengths = gc.delta_lengths
        return is_isomorphic
