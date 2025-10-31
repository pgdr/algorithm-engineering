"""Run DFS and get a DFS tree"""

from collections import defaultdict as D
from collections import namedtuple as T

DiGraph = T("DiGraph", "V E")


def outneighbors(G, v):
    # Yes we use edge list, the worst one
    for x, y in G.E:
        if v == x:
            yield y


pre = D(int)
post = {}
time = 0
tree_edges = set()


def dfs(G, node):
    global time
    time += 1
    pre[node] = time
    for v in outneighbors(G, node):
        if not pre[v]:
            tree_edges.add((node, v))
            dfs(G, v)
    time += 1
    post[node] = time


def dfs_tree(G, pre, post, tree_edges):
    edge_type = {}
    for edge in G.E:
        u, v = edge
        if edge in tree_edges:
            edge_type[edge] = "tree"
        elif pre[u] < pre[v] < post[v] < post[u]:
            edge_type[edge] = "forward"
        elif pre[v] < pre[u] < post[u] < post[v]:
            edge_type[edge] = "back"
        elif pre[v] < post[v] < pre[u] < post[u]:
            edge_type[edge] = "cross"
    return edge_type


def topological_sort(G, post):
    toposort = [-1] * (2 * len(G.V) + 1)
    for v in G.V:
        toposort[post[v]] = v
    return [x for x in reversed(toposort) if x >= 0]
