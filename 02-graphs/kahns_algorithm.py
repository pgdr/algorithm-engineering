"""Kahn's algorithm produces a topological order of a DAG by repeatedly
removing vertices with zero in-degree and updating their neighbors' in-degrees
until none remain.
"""

from collections import deque


def kahns(G):
    indegree = {u: 0 for u in G.nodes}
    for u in G.nodes:
        for v in G[u]:
            indegree[v] += 1

    queue = deque([u for u in G.nodes if indegree[u] == 0])
    count = 0

    while queue:
        u = queue.popleft()
        yield u
        count += 1
        for v in G[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    if count != len(G.nodes):
        raise ValueError("Graph is not a DAG")
