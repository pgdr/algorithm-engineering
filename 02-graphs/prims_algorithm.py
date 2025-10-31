"""Prim's algorithm for minimum spanning tree."""

from heapq import heappush, heappop


def prim(n, edges):
    adj = [[] for _ in range(n)]
    for w, u, v in edges:
        adj[u].append((w, v))
        adj[v].append((w, u))

    seen = [False] * n
    pq = [(0, 0, -1)]
    mst = []

    while pq and len(mst) < n - 1:
        w, v, p = heappop(pq)
        if seen[v]:
            continue
        seen[v] = True
        if p != -1:
            mst.append((p, v))
        for w2, u in adj[v]:
            if not seen[u]:
                heappush(pq, (w2, u, v))

    return mst
