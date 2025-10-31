"""Maximum-spacing k-clustering via Kruskal's algorithm.

Given an undirected weighted graph on vertices {0, ..., n-1}, the goal of
maximum-spacing k-clustering is to partition the vertices into exactly k
clusters so that the minimum-weight edge joining two different clusters is as
large as possible.

Runs in $O(m \\log m)$ time, dominated by sorting the edges.

"""

from collections import defaultdict as D


def maximum_spacing_clustering(n, edges, k):
    uf = Unionfind(n)
    merges = 0

    for w, u, v in sorted(edges):
        if uf.union(u, v):
            merges += 1
            if merges == n - k:
                break

    clusters = D(set)
    for v in range(n):
        clusters[uf[v]].add(v)

    return set(clusters.values())
