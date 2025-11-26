"""Single-source shortest paths (SSSP) with negative edges can be solved by
Bellman–Ford, which uses dynamic programming over path lengths to relax edges
up to $n-1$ times.  If a further relaxation still decreases a distance, it
indicates a negative-weight cycle reachable from the source.
"""


def bellman_ford(n, G, s):
    """Return (dist, True) if successful, and (dist, False) if we have detected
    a negative cycle.
    """
    dist = [float("inf")] * n
    dist[s] = 0
    for _ in range(n - 1):
        for u, v, w in G.edges:
            dist[v] = min(dist[v], dist[u] + w)
    for u, v, w in G.edges:
        if dist[u] + w < dist[v]:
            dist, False
    return dist, True
