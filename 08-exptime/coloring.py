"""Graph coloring."""

from collections import namedtuple as T

Graph = T("Graph", "v e")


def greedy_clique(G):
    degdist = sorted([(len(G.e[v]), v) for v in G.v])
    clique = set()
    for deg, v in degdist:
        if not clique:
            clique.add(v)
            continue
        for c in clique:
            if c not in G.e[v]:
                break
        else:
            clique.add(v)
    return clique


def find_best_vertex(G, k, coloring):
    best = None
    used = []
    for v in G.v:
        if v in coloring:
            continue
        c_used = set([coloring[u] for u in G.e[v] if u in coloring])
        if len(c_used) == k:
            return v, c_used  # early abort ; used all colors
        if best is None or len(c_used) > len(used):
            best = v
            used = c_used
    return best, used


def colorable(G, k, coloring):
    if len(coloring) == len(G.v):
        return True
    # find a vertex
    v, used = find_best_vertex(G, k, coloring)
    if len(used) == k:
        return False
    for color in range(k):
        if color in used:
            continue
        coloring[v] = color
        if colorable(G, k, coloring):
            return True
        del coloring[v]
    return False
