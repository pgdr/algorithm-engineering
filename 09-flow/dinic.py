"""Dinic's algorithm for maximum flow using BFS levels and DFS blocking flows."""

from collections import deque


def add_edge(E, adj, v, u, cap):
    i = len(E)
    E.append([v, u, cap, 0])
    E.append([u, v, 0, 0])
    adj[v].append(i)
    adj[u].append(i + 1)


def bfs(n, adj, E, s):
    level = [-1] * n
    level[s] = 0
    q = deque([s])

    while q:
        v = q.popleft()
        for i in adj[v]:
            _, u, cap, flow = E[i]
            if flow == cap or level[u] != -1:
                continue
            level[u] = level[v] + 1
            q.append(u)

    return level


def dfs(v, t, pushed, adj, E, level, ptr):
    if pushed == 0 or v == t:
        return pushed

    while ptr[v] < len(adj[v]):
        i = adj[v][ptr[v]]
        _, u, cap, flow = E[i]

        if level[u] != level[v] + 1:
            ptr[v] += 1
            continue

        tr = dfs(u, t, min(pushed, cap - flow), adj, E, level, ptr)
        if tr == 0:
            ptr[v] += 1
            continue

        E[i][3] += tr
        E[i ^ 1][3] -= tr
        return tr

    return 0


def dinic(n, edges, s, t):
    E = []
    adj = [[] for _ in range(n)]

    for v, u, cap in edges:
        add_edge(E, adj, v, u, cap)

    flow = 0
    INF = 10**18

    while True:
        level = bfs(n, adj, E, s)
        if level[t] == -1:
            return flow

        ptr = [0] * n
        pushed = dfs(s, t, INF, adj, E, level, ptr)
        while pushed:
            flow += pushed
            pushed = dfs(s, t, INF, adj, E, level, ptr)
