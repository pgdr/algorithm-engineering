"""APSP (All-Pairs Shortest Paths) is the problem of finding the shortest path
distances between every pair of vertices in a weighted graph.

Floyd–Warshall is a classic dynamic programming algorithm for APSP.

By updating `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])` for each
intermediate vertex `k`, it computes shortest paths for all pairs in $O(n^3)$
time.

"""


def floyd_warshall(dist):
    n = len(dist)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
