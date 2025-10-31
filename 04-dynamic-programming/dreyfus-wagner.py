"""Steiner Tree is the problem of finding a minimum-weight connected subgraph
that spans a given set of terminal vertices in a weighted graph.

Dreyfus--Wagner is a classic dynamic programming algorithm for Steiner Tree
parameterized by the number of terminals.

By storing `dp[mask][v]` as the minimum cost of a Steiner tree connecting the
terminals in `mask` and ending at vertex `v`, it combines smaller terminal
subsets and then propagates costs through shortest-path distances.

Using a precomputed all-pairs shortest-path matrix `dist`, it runs in $O(3^k n
+ 2^k n^2)$ time, where $k$ is the number of terminals and $n$ is the number of
vertices.

"""

from collections import defaultdict
from math import inf


def dreyfus_wagner(dist, terminals):
    n = len(dist)
    k = len(terminals)
    dp = defaultdict(lambda: [inf] * n)

    for i, t in enumerate(terminals):
        mask = 1 << i
        for v in range(n):
            dp[mask][v] = dist[v][t]

    for mask in range(1, 1 << k):
        if mask & (mask - 1) == 0:
            continue

        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub < other:
                for v in range(n):
                    cand = dp[sub][v] + dp[other][v]
                    if cand < dp[mask][v]:
                        dp[mask][v] = cand
            sub = (sub - 1) & mask

        for v in range(n):
            dp[mask][v] = min(dp[mask][u] + dist[u][v] for u in range(n))

    return min(dp[(1 << k) - 1])
