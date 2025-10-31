"""The A* algorithm for shortest-path search with a heuristic."""

import math
import heapq
from collections import namedtuple

Node = namedtuple("Node", "name x y")


def euclid(p, q):
    return math.hypot(p.x - q.x, p.y - q.y)


def create_path(parent, t):
    path = [t]
    while t in parent:
        t = parent[t]
        path.append(t)
    return tuple(reversed(path))


def astar(G, source, target, h):
    dist = {v: float("inf") for v in G.nodes}
    parent = {}
    dist[source] = 0
    pq = [(h(source, target), source)]

    while pq:
        d, v = heapq.heappop(pq)
        if v == target:
            return create_path(parent, target), d

        for u in G[v]:
            this = dist[v] + G[v][u]["weight"]
            if this < dist[u]:
                dist[u] = this
                parent[u] = v
                heapq.heappush(pq, (this + h(u, target), u))

    return None, float("inf")
