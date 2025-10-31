r"""Exact dynamic programming for Feedback Arc Set in a directed graph.

Given a directed graph, the \textsc{Feedback Arc Set} problem asks for an
ordering of the vertices minimizing the number of backward edges, that is,
edges $uv$ with $u$ appearing after $v$ in the ordering.  Equivalently, one
seeks a minimum set of edges whose removal makes the graph acyclic.

This implementation uses subset DP.  Let $\text{dp}(S, v)$ be the minimum
number of backward edges in an ordering of the vertex set $S$ that ends with
$v$.  If $v$ is placed last, then every edge from $v$ to a vertex still in $S$
contributes one backward edge, and the remaining vertices are solved
recursively.

Runs in $O(n^2 2^n)$ time.

"""

from functools import cache

Z = frozenset
INF = 10**10


def create_graph(n, edges):
    G = [set() for _ in range(n)]
    for u, v in edges:
        G[u].add(v)
    return tuple(map(Z, G))


def backward_edges(G, v, S):
    return len(G[v] & S)


def feedback_arc_set(G):
    V = Z(range(len(G)))

    @cache
    def dp(S, v):
        if not S:
            return 0, None
        if v not in S:
            return INF, None
        if len(S) == 1:
            return 0, None

        T = S - {v}
        best = INF
        pred = None
        for u in T:
            cost, _ = dp(T, u)
            if cost < best:
                best = cost
                pred = u
        return best + backward_edges(G, v, T), pred

    cost, last = min((dp(V, v)[0], v) for v in V)

    order = []
    S = V
    v = last
    while v is not None:
        order.append(v)
        _, v = dp(S, v)
        S = S - {order[-1]}
    order.reverse()

    return cost, tuple(order)
