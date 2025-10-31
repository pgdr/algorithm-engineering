"""Use postorder DFS to get a reversed topological ordering."""


def dfs_toposort(G):
    visited = set()
    order = []

    def dfs(u):
        if u not in visited:
            for v in G[u]:
                dfs(v)
            visited.add(u)
            order.append(u)

    for u in G.nodes:
        if u not in visited:
            dfs(u)

    yield from reversed(order)
