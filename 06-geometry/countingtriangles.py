class Vec:
    def __init__(vec, x, y):
        vec.c = complex(x, y)

    def cross(vec, other):
        return (vec.c.conjugate() * other.c).imag

    def __sub__(vec, other):
        v = vec.c - other.c
        return Vec(v.real, v.imag)

    @property
    def x(vec):
        return vec.real

    @property
    def y(vec):
        return vec.imag


def orient(a, b, p):
    return (b - a).cross(p - a) > 0


def intersect(line1, line2):
    a, b = line1
    c, d = line2
    return (orient(a, b, c) != orient(a, b, d)) and (orient(c, d, a) != orient(c, d, b))
