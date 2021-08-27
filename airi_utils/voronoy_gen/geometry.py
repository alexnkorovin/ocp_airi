from functools import reduce

import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist


def calc_dist(ps1, ps2):

    return np.sqrt(np.sum((ps1 - ps2) ** 2, axis=1))


def calc_angles(ps1, ps2, ps3):

    vs1 = np.array(ps1) - ps2
    vs2 = np.array(ps3) - ps2
    a = np.array(list(map(np.linalg.norm, vs1))) * list(
        map(np.linalg.norm, vs2)
    )
    a[a == 0] = None
    with np.warnings.catch_warnings():
        cos = np.array([np.dot(v1, v2) / a for v1, v2, a in zip(vs1, vs2, a)])
        cos[cos > 1] = 1.0
        cos[cos < -1] = -1.0
        angles = np.arccos(cos)
        angles[angles > np.pi] = np.pi
    return angles


def calc_dihedrals(ps1, ps2, ps3, ps4):

    vs1 = -1 * (ps2 - ps1)
    vs2 = ps3 - ps2
    vs3 = ps4 - ps3
    v1 = [
        v1 - np.dot(v1, v2) / np.dot(v2, v2) * v2 for v1, v2 in zip(vs1, vs2)
    ]
    v2 = [
        v3 - np.dot(v3, v2) / np.dot(v2, v2) * v2 for v3, v2 in zip(vs3, vs2)
    ]
    a = np.array(map(np.linalg.norm, v1)) * map(np.linalg.norm, v2)
    a[a == 0] = None
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings("ignore", r".*encountered")
        cos = np.array([np.dot(v1, v2) / a for v1, v2, a in zip(v1, v2, a)])
        cos[cos > 1] = 0.0
        cos[cos < -1] = -1.0
        dihedrals = np.arccos(cos)
        dihedrals[dihedrals > np.pi] = np.pi

    return dihedrals


def calc_dihedrals2(ps1, ps2, ps3, ps4):

    b0 = -1.0 * (ps2 - ps1)
    b1 = ps3 - ps2
    b2 = ps4 - ps3
    b1 = np.array(list(map(lambda x: x / np.linalg.norm(x), b1)))
    v = b0 - list(map(lambda x: np.dot(x[0], x[1]) * x[1], zip(b0, b1)))
    w = b2 - list(map(lambda x: np.dot(x[0], x[1]) * x[1], zip(b2, b1)))
    x = list(map(lambda x: np.dot(x[0], x[1]), zip(v, w)))
    c = list(map(lambda x: np.cross(x[0], x[1]), zip(b1, v)))
    y = list(map(lambda x: np.dot(x[0], x[1]), zip(c, w)))
    dihedrals = np.array(
        list(map(lambda x: np.arctan2(x[0], x[1]), zip(y, x)))
    )
    return dihedrals


def calc_centroid(ps):

    centroid = reduce(lambda x, y: x + y, np.array(ps)) / len(ps)
    return centroid


def are_coplanar(vs, prec=0.01):

    if len(vs) < 3:
        return True
    else:
        for i in range(len(vs) - 2):
            for j in range(i + 1, len(vs) - 1):
                ab = np.cross(vs[i], vs[j])
                for li in range(j + 1, len(vs)):
                    if abs(np.dot(ab, vs[li])) > prec:
                        return False
    return True


def get_non_coplanar(vs, tol=0.01):

    n = len(vs)
    if n < 3:
        return None
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            for k in range(j + 1, len(vs)):
                if not are_coplanar([vs[i], vs[j], vs[k]], tol):
                    print("X", 0.0, 0.0, 0.0)
                    print("C", *vs[i])
                    print("C", *vs[j])
                    print("C", *vs[k])
                    return [i, j, k]
    return None


# Turns coordinate system around axe
def rotate(a, ps, axe=0):

    sin = np.sin(a)
    cos = np.cos(a)
    rm = np.array(
        [
            [[1, 0, 0], [0, cos, -sin], [0, sin, cos]],
            [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]],
            [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]],
        ]
    )
    m = np.full((len(ps), 3, 3), rm[axe])
    ps = list(map(lambda x, y: np.dot(x, y), m, ps))
    return ps


def orient(ps, v0, v1, v2):

    ps = np.vstack((v1, v2, ps))
    ps -= v0
    if ps[0][1] == 0:
        a = 0
    else:
        a = np.arcsin(
            np.fabs(ps[0][1]) / np.sqrt(ps[0][1] ** 2 + ps[0][2] ** 2)
        )
    if (ps[0][1] < 0 <= ps[0][2]) or (ps[0][1] > 0 > ps[0][2]):
        a = 2 * np.pi - a
    if (ps[0][1] * np.sin(a) + ps[0][2] * np.cos(a)) < 0:
        a = np.pi + a
    ps = rotate(a, ps, 0)
    if ps[0][0] == 0:
        b = 0
    else:
        b = np.arcsin(
            np.fabs(ps[0][0]) / np.sqrt(ps[0][0] ** 2 + ps[0][2] ** 2)
        )
    if (ps[0][0] < 0 and ps[0][2] < 0) or (ps[0][0] > 0 and ps[0][2] > 0):
        b = 2 * np.pi - b
    if (ps[0][2] * np.cos(b) - ps[0][0] * np.sin(b)) < 0:
        b = np.pi + b
    ps = rotate(b, ps, 1)
    if ps[1][1] == 0:
        c = 0
    else:
        c = np.arcsin(
            np.fabs(ps[1][1]) / np.sqrt(ps[1][0] ** 2 + ps[1][1] ** 2)
        )
    if (ps[1][0] < 0 and ps[1][1] < 0) or (ps[1][0] > 0 and ps[1][1] > 0):
        c = 2 * np.pi - c
    if (ps[1][0] * np.cos(c) - ps[1][1] * np.sin(c)) < 0:
        c = np.pi + c
    ps = rotate(c, ps, 2)
    return ps[2:]


def calc_vectors(a, b, c, alpha, betta, gamma):

    alpha = np.radians(alpha % 180)
    betta = np.radians(betta % 180)
    gamma = np.radians(gamma % 180)
    c1 = c * np.cos(betta)
    c2 = c * (np.cos(alpha) - np.cos(gamma) * np.cos(betta)) / np.sin(gamma)
    c3 = np.sqrt(c * c - c1 * c1 - c2 * c2)
    m = np.array(
        [
            [a, 0.0, 0.0],
            [b * np.cos(gamma), b * np.sin(gamma), 0.0],
            [c1, c2, c3],
        ]
    )
    return m


def is_inside_poly(pt, poly, tol=0.001):

    n = len(poly)
    inside = False
    x, y = pt
    for i in range(1, n + 1):
        x1, y1 = poly[i - 1]
        x2, y2 = poly[i % n]
        if y1 != y2 and min(y1, y2) <= y < max(y1, y2) and x < max(x1, x2):
            x_ints = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if abs(x_ints - x) < tol:
                return False
            if x1 == x2 or x < x_ints:
                inside = not inside
    return inside


def calc_solid_angle(origin, vs):

    if len(vs) != 3:
        raise ValueError("The number of vectors should be equal to 3!")
    v1, v2, v3 = vs - origin
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    l3 = np.linalg.norm(v3)
    a = np.dot(v1, np.cross(v2, v3))
    b = (
        l1 * l2 * l3
        + np.dot(v1, v2) * l3
        + np.dot(v2, v3) * l1
        + np.dot(v3, v1) * l2
    )
    solid_angle = 2 * np.arctan(a / b)
    return solid_angle


def find_v_neighbors(points, central_points, key=1, direct=True):

    neighbors = {i: None for i in central_points}
    solid_angles = {i: None for i in central_points}
    vor = Voronoi(points)
    for i in central_points:
        region = vor.regions[vor.point_region[i]]
        if -1 in region:
            raise RuntimeError(
                'The domain for "' + str(i) + '" point is not closed!'
            )
        local_neighbors = []
        loc_angles = []
        for j in range(len(points)):
            face = vor.vertices[
                np.intersect1d(region, vor.regions[vor.point_region[j]])
            ]
            n = len(face)
            if i != j and n >= key:
                poly = np.vstack((face, calc_centroid([points[i], points[j]])))
                poly = [
                    [p[0], p[2]]
                    for p in orient(
                        poly, calc_centroid(poly[:-1]), poly[0], poly[1]
                    )
                ]
                pt, poly = poly[-1], np.array(poly[:-1])
                angles = calc_angles([[0, 2]] * n, [[0, 0]] * n, poly)
                angles[poly[:, 0] < 0] = 2 * np.pi - angles[poly[:, 0] < 0]
                order = list(enumerate(angles))
                order.sort(key=lambda x: x[1])
                order = [o for o, a in order]
                face = face[order]
                fc = calc_centroid(face)
                simps = [
                    (face[v_ind - 1], face[v_ind], fc) for v_ind in range(n)
                ]
                loc_angles += [
                    abs(sum([calc_solid_angle(points[i], s) for s in simps]))
                ]
                if (not direct) or is_inside_poly(pt, poly[order]):
                    local_neighbors.append(j)
        neighbors[i] = local_neighbors
        solid_angles[i] = loc_angles / sum(loc_angles) * 100
    return neighbors, solid_angles


def find_closest_ps_bijections(ps_1, ps_2):

    n_1, n_2 = len(ps_1), len(ps_2)
    if n_1 != n_2:
        raise AttributeError("The lengths of point sets are not equivalent!")
    closest_pairs = []
    cns_1 = [n_1 for _ in ps_1]
    cns_2 = [n_2 for _ in ps_2]
    dist_matr = cdist(ps_1, ps_2, metric="euclidean")
    pairs = [
        ([i, j], dist)
        for i, dists in enumerate(dist_matr)
        for j, dist in enumerate(dists)
    ]
    pairs.sort(key=lambda x: x[1])
    for i in range(1, len(pairs) + 1):
        i_1, i_2 = pairs[-i][0][0], pairs[-i][0][1]
        if cns_1[i_1] > 1 and cns_2[i_2] > 1:
            cns_1[i_1] -= 1
            cns_2[i_2] -= 1
        else:
            closest_pairs.append(pairs[-i])
    return [pair for pair, dist in closest_pairs], [
        dist for pair, dist in closest_pairs
    ]
