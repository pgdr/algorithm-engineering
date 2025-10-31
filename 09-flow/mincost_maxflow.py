"""Successive shortest augmenting path algorithm for Min Cost Max Flow."""

from collections import namedtuple as T

Graph = T("Graph", "V E c w F")


def create_graph(V, E, c, w):
    n = len(V)
    F = [[0] * n for _ in range(n)]
    for v, u in E:
        F[v][u] = c[(v, u)]
    return Graph(V=V, E=E, c=c, w=w, F=F)


def create_path(parent, s, t):
    path = [t]
    v = t
    while v != s:
        v = parent[v]
        path.append(v)
    return tuple(reversed(path))


edges = lambda p: zip(p, p[1:])


def residual_cost(graph, v, u):
    if (v, u) in graph.w:
        return graph.w[(v, u)]  # forward residual edge
    return -graph.w[(u, v)]  # reverse residual edge


def shortest_path(graph, s, t):
    dist = {v: float("inf") for v in graph.V}
    parent = {}
    dist[s] = 0

    n = len(graph.V)
    for _ in range(n - 1):
        changed = False
        for v in graph.V:
            if dist[v] == float("inf"):
                continue
            for u in graph.V:
                if graph.F[v][u] <= 0:
                    continue  # no residual capacity
                cand = dist[v] + residual_cost(graph, v, u)
                if cand < dist[u]:
                    dist[u] = cand
                    parent[u] = v
                    changed = True
        if not changed:
            break

    if t not in parent:
        return None, None

    return create_path(parent, s, t), dist[t]


def min_cost_max_flow(graph, s, t):
    flow = 0
    cost = 0

    while True:
        P, path_cost = shortest_path(graph, s, t)
        if P is None:
            return flow, cost

        bottleneck = min(graph.F[v][u] for (v, u) in edges(P))
        print(P, bottleneck, path_cost)

        flow += bottleneck
        cost += bottleneck * path_cost

        for v, u in edges(P):
            graph.F[v][u] -= bottleneck
            graph.F[u][v] += bottleneck
