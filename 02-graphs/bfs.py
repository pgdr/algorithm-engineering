"""Breadth-first search (BFS) for shortest paths in an unweighted graph.

BFS explores vertices in layers of increasing distance from a source vertex.
It computes shortest-path distances measured in number of edges, and the
parent pointers define a BFS tree that can be used to reconstruct paths.

Runs in O(n + m) time.
"""

from collections import namedtuple as T

Graph = T("Graph", "V E")


def create_graph(V, E):
    return Graph(V=tuple(V), E=tuple(E))


def create_path(parent, t):
    path = [t]
    while t in parent:
        t = parent[t]
        path.append(t)
    return tuple(reversed(path))


def bfs(graph, s, t=None):
    dist = {s: 0}
    parent = {}
    q = [s]

    while q:
        v = q.pop(0)
        if v == t:
            return create_path(parent, t), dist[t]
        for x, u in graph.E:
            if x != v or u in dist:
                continue
            dist[u] = dist[v] + 1
            parent[u] = v
            q.append(u)

    if t is None:
        return dist, parent
    return None, float("inf")
