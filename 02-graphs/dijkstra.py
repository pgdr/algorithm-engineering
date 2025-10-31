"""Simple Dijkstra implementation running in time $O(m \\log n)$."""

import heapq


def dijkstra(G, source):
    dist = {v: float("inf") for v in G.nodes}
    dist[source] = 0
    priority_queue = [(0, source)]

    while priority_queue:
        current_distance, v = heapq.heappop(priority_queue)
        if current_distance > dist[v]:
            continue

        for u in G.neighbors(v):
            weight = G[v][u]["weight"]
            if dist[u] > dist[v] + weight:
                dist[u] = dist[v] + weight
                heapq.heappush(priority_queue, (dist[u], u))

    return dist
