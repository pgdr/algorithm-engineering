"""The Jarvis march (gift wrapping) algorithm for computing the convex hull of
a set of $n$ points in $O(n \\cdot h)$ time where $h$ is the number of points
on the hull.
"""

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Point:
    x: float
    y: float
    idx: float

    @property
    def tuple(self):
        return self.x, self.y

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, -1)


def leftmost(points):
    return sorted(points)[0].idx


def cross(a, b):
    return a.x * b.y - a.y * b.x


def orient(a, b, c):
    res = cross(b - a, c - a)
    if res == 0:
        return 0  # on line
    elif res > 0:
        return 1  # cw
    else:
        return 2  # ccw


def jarvis(points):
    N = len(points)
    l = leftmost(points)
    hull = []
    b = 0
    a = None
    while a != l:
        if a is None:
            a = l  # start
        hull.append(a)
        b = (a + 1) % N
        for i in range(N):
            if orient(points[a], points[i], points[b]) == 2:
                b = i
        a = b
    return [points[i] for i in hull]
