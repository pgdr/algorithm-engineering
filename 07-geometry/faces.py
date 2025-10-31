"""Compute the faces of a planar graph."""

import cmath


class Vec:
    def __init__(v, x, y):
        v.c = complex(x, y)

    @property
    def x(v):
        return v.c.real

    @property
    def y(v):
        return v.c.imag

    @staticmethod
    def _from_real(c):
        return Vec(c.real, c.imag)

    def __sub__(v1, v2):
        return v1._from_real(v1.c - v2.c)

    @property
    def length(v):
        return abs(v.c)

    @property
    def theta(v):
        return cmath.phase(v.c)

    def __repr__(v):
        return f"Vec({v.x}, {v.y})"


def is_outer(face):
    global left, right, top, bottom
    for v in left, right, top, bottom:
        if v[0] not in face:
            return False
    return True


def rel_theta(origo):
    return lambda p: (p.point - origo).theta


def sort_neighborhood(v, nv):
    pv = G.vertices[v]
    neighbors = sorted([Vertex(u, G.vertices[u]) for u in nv], key=rel_theta(pv))
    return [u.name for u in neighbors]


def sort_all_neighborhoods(G):
    for v, nv in G.edges.items():
        G.edges[v] = sort_neighborhood(v, nv)


sort_all_neighborhoods(G)


def cycle(s):
    m = min(s)
    i = s.index(m)
    return s[i:] + s[:i]


def get_face(edge, G):
    face = [edge[0]]
    while True:
        u, v = edge
        if v in face:
            break
        face.append(v)
        nb = G.edges[v]
        ui = nb.index(u)
        nxt = nb[(ui + 1) % len(nb)]
        edge = v, nxt
    return face


faces = {}
for v, nv in G.edges.items():
    for u in nv:
        edge = v, u
        faces[edge] = tuple(cycle(get_face(edge)))


def get_polygon(edge, G):
    f = faces[edge]
    poly = []
    for v in f:
        poly.append(G.vertices[v])
    return poly


def face_to_polygon(face, G):
    poly = []
    for v in face:
        poly.append(G.vertices[v])
    return poly


G = None
