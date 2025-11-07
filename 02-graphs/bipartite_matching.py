"""Compute a maximum matching in an unweighted bipartite graph using DFS to
find augmenting paths.  Implements the classic alternating-path approach for
bipartite matching.
"""


def max_bipartite_matching(G, A, B):
    matchA = {u: None for u in A}
    matchB = {v: None for v in B}

    def dfs(u, seen):
        for v in G[u]:
            if v in seen:
                continue
            seen.add(v)
            if matchB[v] is None or dfs(matchB[v], seen):
                matchA[u] = v
                matchB[v] = u
                return True
        return False

    for u in A:
        dfs(u, set())

    return matchA, matchB
