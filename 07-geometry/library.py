"""Some Geometry functions"""

import cmath
import math


class Vec:
    def __init__(self, x=None, y=None, c=None):
        if c is None:
            if x is None or y is None:
                raise ValueError("Either (x or y) or c")
            self.c = complex(x, y)
        else:
            if x is not None or y is not None:
                raise ValueError("When c, not x y")
            self.c = c

    @property
    def x(self):
        return self.c.real

    @property
    def y(self):
        return self.c.imag

    @property
    def phi(self):
        return cmath.phase(self.c)

    @property
    def r(self):
        return abs(self.c)

    def __mul__(self, o):
        if isinstance(o, Vec):
            o = o.c
        return Vec(c=self.c * o)

    def __truediv__(self, s):
        return Vec(self.x / s, self.y / s)

    def __add__(self, o):
        if isinstance(o, Vec):
            o = o.c
        return Vec(c=self.c + o)

    def __sub__(self, o):
        if isinstance(o, Vec):
            o = o.c
        return Vec(c=self.c - o)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"C({self.x}, {self.y})"

    @property
    def conj(self):
        return Vec(c=self.c.conjugate())

    @property
    def rad(self):
        return cmath.phase(self.c)

    @property
    def deg(self):
        return cmath.phase(self.c) * 180 / math.pi

    def translate(v1, v2):
        return v1 + v2

    def scale(v, scalar, c):
        return c + (v - c) * scalar

    def rotate(v, angle):
        return v * cmath.polar(1, angle)

    def perp(v):
        return Vec(-v.y, v.x)

    def dot(v1, v2):
        return (v1.conj * v2).x  # x

    def dot_(v1, v2):
        return (v1.x * v2.x) + (v1.y * v2.y)

    def is_perp(v1, v2):
        return v1.dot(v2) == 0

    def cross(v1, v2):
        return (v1.conj * v2).y  # y

    def cross_(v1, v2):
        return v1.x * v2.y - v1.y * v2.x

    def __hash__(v):
        return hash(v.c)

    def __eq__(v1, v2):
        return v1.c == v2.c

    def __getitem__(v, idx):
        if idx == 0:
            return v.x
        if idx == 1:
            return v.y
        raise IndexError(str(v) + " " + str(idx))


def cis(r, phi):
    return Vec(r * math.cos(phi), (r * math.sin(phi)))


def sgn(a):
    return (Vec(0) < a) - (a < Vec(0))


def orient(a, b, c):
    return (b - a).cross(c - a)


def in_angle(a, b, c, p):
    """p inside angle created from the lines from a to b and a to c."""
    assert orient(a, b, c) != 0
    if orient(a, b, c) < 0:
        b, c = c, b
    return orient(a, b, p) >= 0 and orient(a, c, p) <= 0


def is_convex(polygon):
    has_pos = has_neg = False
    for idx in range(len(polygon)):
        p1, p2, p3 = polygon[idx - 2], polygon[idx - 1], polygon[idx]
        o = orient(p1, p2, p3)
        has_pos |= o > 0
        has_neg |= o < 0

    return not (has_pos and has_neg)
