"""The Edmonds–Karp variant of the Ford–Fulkerson algorithm for computing
maximum flow.
"""

from collections import namedtuple as T

Graph = T("Graph", "V E c F")


def create_graph(V, E, c):
    n = len(V)
    F = [[0] * n for _ in range(n)]
    for v, u in c:
        F[v][u] = c[(v, u)]
    return Graph(V=V, E=E, c=c, F=F)


def create_path(parent, t):
    path = [t]
    v = t
    while v:
        v = parent[v]
        path.append(v)
    return tuple(reversed(path))


def bfs(graph, s, t):
    q = [s]
    parent = {}
    while q:
        v = q.pop(0)
        for u in graph.V:
            if u in parent:
                continue  # seen it before
            if graph.F[v][u] <= 0:
                continue  # vu saturated
            parent[u] = v
            if u == t:
                return create_path(parent, t)
            q.append(u)


edges = lambda p: zip(p, p[1:])


def maxflow(graph, s, t):
    flow = 0
    while P := bfs(graph, s, t):
        bottleneck = min(graph.F[v][u] for (v, u) in edges(P))
        print(P, bottleneck)
        if bottleneck == 0:
            return flow
        flow += bottleneck
        for i in range(1, len(P)):
            v, u = P[i - 1], P[i]
            graph.F[v][u] -= bottleneck
            graph.F[u][v] += bottleneck
    return flow
