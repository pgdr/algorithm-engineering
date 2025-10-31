"""Kruskal's algorithm for minimum spanning tree (MST).

Given a weighted undirected graph, Kruskal's algorithm computes a minimum
spanning tree by considering the edges in nondecreasing order of weight and
adding an edge whenever it connects two previously disconnected components.

The graph is given as a list of edges of the form $(w, u, v)$, where $w$ is the
weight of the edge $uv$.  Sorting this list yields the order in which edges are
considered.  A Union Find data structure maintains the connected components.

Returns the edges of the MST together with its total weight.

Runs in $O(m \\log m)$ time, dominated by sorting the m edges.

"""


class Unionfind:
    def __init__(self, size):
        self._roots = list(range(size))
        self._rank = [0] * size

    def find(self, elt):
        root = elt
        while root != self._roots[root]:
            root = self._roots[root]

        while elt != root:  # path compression
            parent = self._roots[elt]
            self._roots[elt] = root
            elt = parent

        return root

    def __getitem__(self, elt):
        return self.find(elt)

    def is_connected(self, a, b):
        return self.find(a) == self.find(b)

    def union(self, a, b):
        r1 = self.find(a)
        r2 = self.find(b)
        if r1 == r2:
            return False

        if self._rank[r1] < self._rank[r2]:
            self._roots[r1] = r2
        elif self._rank[r1] > self._rank[r2]:
            self._roots[r2] = r1
        else:
            self._roots[r2] = r1
            self._rank[r1] += 1
        return True


def kruskal(n, edges):
    uf = Unionfind(n)
    mst = []
    weight = 0

    for w, u, v in sorted(edges):
        if uf.union(u, v):
            mst.append((u, v))
            weight += w
            if len(mst) == n - 1:
                break

    return mst, weight
