"""Connected components via BFS."""

from collections import deque


def bfs(adj, s):
    seen = {s}
    q = deque([s])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if u not in seen:
                seen.add(u)
                q.append(u)
    return frozenset(seen)


def connected_components(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    seen = set()
    for v in range(n):
        if v not in seen:
            comp = bfs(adj, v)
            yield comp
            seen |= comp
