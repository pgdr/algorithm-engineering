"""Layered Graph algorithm for search with $k$ free edges.

Suppose you have a directed graph with edge weights and we want to find a
shortest path from $s$ to $t$.  Suppose furthermore that we can use $k$ edges
for free.

The Layered Graph algorithm takes $k+1$ copies of $G$, $G_0, G_1, ..., G_k$,
and creates additional edges from $u_i$ to $v_{i+1}$ with weight 0, for all
edges $uv$ in $G$.

A path from $s_0$ to some $t_i$ corresponds to an $s$-$t$ path in the original
graph using exactly $i$ free edges.  Thus, the shortest path from $s_0$ to any
layered copy of $t$ gives the optimal path using at most $k$ free edges.
"""

from collections import namedtuple as T
from heapq import heappop, heappush

Graph = T("Graph", "V E w")


def create_graph(V, E, w):
    return Graph(V=tuple(V), E=tuple(E), w=w)


def create_path(parent, t):
    path = [t]
    while t in parent:
        t = parent[t]
        path.append(t)
    return tuple(reversed(path))


def dijkstra(graph, s, t):
    dist = {s: 0}
    parent = {}
    q = [(0, s)]

    while q:
        d, v = heappop(q)
        if d != dist[v]:
            continue
        if v == t:
            return create_path(parent, t), d
        for u in graph.V:
            if (v, u) not in graph.w:
                continue
            alt = d + graph.w[(v, u)]
            if alt >= dist.get(u, float("inf")):
                continue
            dist[u] = alt
            parent[u] = v
            heappush(q, (alt, u))


def layered_graph(graph, k):
    V = tuple((v, i) for i in range(k + 1) for v in graph.V)
    E = []
    w = {}

    for i in range(k + 1):
        for v, u in graph.E:
            E.append(((v, i), (u, i)))
            w[((v, i), (u, i))] = graph.w[(v, u)]

    for i in range(k):
        for v, u in graph.E:
            E.append(((v, i), (u, i + 1)))
            w[((v, i), (u, i + 1))] = 0

    return create_graph(V, E, w)


def unwrap(path):
    return tuple(v for (v, _) in path)


def shortest_path(graph, s, t, k):
    H = layered_graph(graph, k)
    best = None

    for i in range(k + 1):
        P, d = dijkstra(H, (s, 0), (t, i))
        if P is None:
            continue
        if best is None or d < best[1]:
            best = (P, d)

    if best is None:
        return None, float("inf")

    P, d = best
    return unwrap(P), d
