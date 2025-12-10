"""A divide and conquer algorithm for computing a closest pair of points in a
set of $n$ points in the plane in time $O(n \\log n)$.  It outputs both their
distance and the pair itself.
"""

import itertools as IT
import math
from collections import namedtuple as T

Point = T("Point", "x y")
Sol = T("Sol", "delta pair")


def bruteforce(points):
    return min(Sol(math.hypot(a.x - b.x, a.y - b.y), (a, b)) for (a, b) in IT.combinations(points, 2))


def closest_pair(X, Y):
    if len(X) <= 5:
        return bruteforce(X)
    pivot = X[len(X) // 2]
    Xl = [p for p in X if p <= pivot]
    Yl = [p for p in Y if p <= pivot]
    Xr = [p for p in X if p > pivot]
    Yr = [p for p in Y if p > pivot]
    OPT = min(closest_pair(Xl, Yl), closest_pair(Xr, Yr))
    S = [p for p in Y if abs(p.x - pivot.x) < OPT.delta]
    if len(S) > 1:
        OPT = min(OPT, min(bruteforce(IT.islice(S[i:], 6)) for i in range(len(S) - 1)))
    return OPT


def closest(points):
    X = sorted(points)
    Y = sorted(points, key=lambda p: (p.y, p.x))
    return closest_pair(X, Y)


# Driver
import random

N = 100
P = [Point(random.random() * 1000, random.random() * 1000) for _ in range(N)]
delta, (p1, p2) = closest(P)
print("delta:", round(delta, 3))
