"""Travelling Salesman dynamic programming routine."""

import itertools as IT
import math
from collections import defaultdict as D
from collections import namedtuple as T

Set = frozenset
Place = T("Place", "name x y")
Graph = T("Graph", "V E")
INFTY = float("inf")
Sol = T("Sol", "cost path")


def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def tsp(G, start, end):
    DP = D(lambda: Sol(INFTY, None))
    DP[(Set([start]), start)] = Sol(0, [start])
    V = Set(G.V) - Set([start, end])

    for k in range(len(V) + 2):
        for S in IT.combinations(G.V, k):
            set_s = Set(S) | Set([start])
            for v in S:
                v_sol = DP[(set_s, v)]  # cost for v, parent for v
                for u in G.V:
                    if u in S:
                        continue
                    set_su = set_s | Set([u])
                    c_sol = DP[(set_su, u)]
                    n_cost = v_sol.cost + dist(v, u)
                    if n_cost < c_sol.cost:
                        DP[(set_su, u)] = Sol(n_cost, v_sol.path + [u])
    return DP[Set(G.V), end]
