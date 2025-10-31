"""The Shoelace algorithm for computing the area of a polygon."""

from collections import namedtuple as T

P = T("P", "x y")


def shoelace(poly):
    fw = sum(poly[i - 1].x * poly[i].y for i in range(len(poly)))
    poly = list(reversed(poly))
    bw = sum(poly[i - 1].x * poly[i].y for i in range(len(poly)))
    return abs(fw - bw) / 2.0


def assert_eq(p, exp):
    act = shoelace(p)
    assert act == exp, str((act, exp, p))
    print(act, "=", exp)


assert_eq([P(1, 1), P(2, 1), P(2, 0), P(1, 0)], 1.0)
assert_eq([P(1, 1), P(2, 2), P(3, 1), P(2, 0)], 2.0)
assert_eq([P(2, 7), P(10, 1), P(8, 6), P(11, 7), P(7, 10)], 32.0)
assert_eq([P(3, 4), P(5, 11), P(12, 8), P(9, 5), P(5, 6)], 30.0)
assert_eq([P(5, 11), P(12, 8), P(9, 5), P(5, 6), P(3, 4)], 30.0)
