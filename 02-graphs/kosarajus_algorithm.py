"""A strongly connected component (SCC) is a maximal set of vertices where
every vertex can reach every other by directed paths.  Kosaraju's algorithm
finds all SCCs by doing a DFS to record finish times, reversing all edges, then
running DFS again in reverse finish order.
"""


def kosaraju(G):
    vis, order = set(), []

    def dfs(u, graph, collect):
        vis.add(u)
        for v in graph[u]:
            if v not in vis:
                dfs(v, graph, collect)
        collect.append(u)

    for u in G:
        if u not in vis:
            dfs(u, G, order)

    R = {u: [] for u in G}
    for u in G:
        for v in G[u]:
            R[v].append(u)

    vis.clear()
    while order:
        u = order.pop()
        if u not in vis:
            comp = []
            dfs(u, R, comp)
            yield comp
