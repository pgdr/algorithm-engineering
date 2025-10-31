"""Simple Union Find (with path compression) datastructure implementation for
use in MST.
"""


class Unionfind:
    def __init__(self, size):
        self._roots = list(range(size))
        self._rank = [0 for _ in range(size)]

    def __getitem__(self, elt):
        root = elt
        while root != self._roots[root]:
            root = self._roots[root]

        # path compression
        while elt != root:
            parent = self._roots[elt]
            self._roots[elt] = root
            elt = parent

        return root

    def is_connected(self, a, b):
        return self[a] == self[b]

    def union(self, p, q):
        r1 = self[p]
        r2 = self[q]
        if r1 == r2:
            return False

        self._roots[r1] = r2
        return True
