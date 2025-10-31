"""Given a set of $n$ $(x, y)$ points, compute the least squares for the data
set.  In addition, create a data structure that can answer query
`least_squares(i, j)` of all points in `data[i:j]`.  Using dynamic programming
we can compute the entire data structure in time $O(n^2)$.
"""

from collections import namedtuple

RegressionState = namedtuple("RegressionState", "a b n sum_x sum_y sum_xy sum_x2 a_nom denom b_nom")


def regression(x, y, state=None):
    if state is None or state.n == 0:
        n = 1
        sum_x, sum_y = x, y
        sum_xy, sum_x2 = x * y, x * x
    else:
        n = state.n + 1
        sum_x = state.sum_x + x
        sum_y = state.sum_y + y
        sum_xy = state.sum_xy + x * y
        sum_x2 = state.sum_x2 + x * x

    a_nom = n * sum_xy - sum_x * sum_y
    denom = n * sum_x2 - sum_x * sum_x
    b_nom = sum_y * sum_x2 - sum_x * sum_xy

    if denom == 0:
        a = 0.0
        b = sum_y / n
    else:
        a = a_nom / denom
        b = b_nom / denom

    return RegressionState(a, b, n, sum_x, sum_y, sum_xy, sum_x2, a_nom, denom, b_nom)


def build_dp(points):
    N = len(points)
    DP = {}
    for i in range(N):
        DP[(i, i)] = regression(*points[i])

    for i in range(N):
        for j in range(i + 1, N):
            DP[(i, j)] = regression(*points[j], DP[(i, j - 1)])
    return DP


# RUNNING THE CODE
import random


def generate_points(N):
    points = []
    for x in range(N):
        y = round(2 + x + (random.randint(-200, 200) * 0.12), 3)
        points.append((1 + x, y))
    return points


def axb(state, r=5):
    """Print s.a and s.b as "a·x+b"."""
    a = round(state.a, r)
    b = round(state.b, r)
    if b >= 0:
        return f"{a}·x + {b} ({state.n} points)"
    return f"{a}·x - {-b} ({state.n} points)"


RegressionState.__str__ = axb

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        exit("Usage: ./regression.py <N>")
    N = int(sys.argv[1])
    random.seed(1000 * N % 2**20 - 1)
    points = generate_points(N)

    DP = build_dp(points)
    line1 = DP[(0, N - 1)]
    print(line1)
