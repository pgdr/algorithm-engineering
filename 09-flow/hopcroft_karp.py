"""Hopcroft-Karp algorithm for maximum matching in a bipartite graph."""

from collections import deque
from math import inf


def bfs(G):
    Q = deque()
    for u in range(1, G.m + 1):
        if G.RHS[u] is None:
            G.dist[u] = 0
            Q.append(u)
        else:
            G.dist[u] = inf
    G.dist[None] = inf
    while Q:
        u = Q.popleft()
        if G.dist[u] < G.dist[None]:
            for v in G.adj[u]:
                if G.dist[G.LHS[v]] == inf:
                    G.dist[G.LHS[v]] = G.dist[u] + 1
                    Q.append(G.LHS[v])
    return G.dist[None] != inf


def dfs(G, u):
    if u is None:
        return True
    for v in G.adj[u]:
        if G.dist[G.LHS[v]] == G.dist[u] + 1:
            if dfs(G, G.LHS[v]):
                G.LHS[v] = u
                G.RHS[u] = v
                return True
    G.dist[u] = inf
    return False


def hopcroft_karp(G):
    G.RHS = [None for _ in range(G.m + 1)]
    G.LHS = [None for _ in range(G.n + 1)]
    G.dist = {u: inf for u in range(1, G.m + 1)}
    G.dist[None] = inf

    result = 0
    while bfs(G):
        for u in range(1, G.m + 1):
            if G.RHS[u] is None and dfs(G, u):
                result += 1
    return result
