"""Basic implementation of a segment tree."""

left = lambda x: 2 * x
right = lambda x: 2 * x + 1
parent = lambda x: x // 2
index = lambda T, i: len(T) // 2 + i


def fill(tree, op):
    internal = range(1, len(tree) // 2)
    for idx in reversed(internal):
        tree[idx] = op((tree[left(idx)], tree[right(idx)]))


def query_(t, l, r):
    yield t[l]  # yield t[r] in incl case
    while True:
        pl = parent(l)
        pr = parent(r)
        if pl == pr:
            return
        if l % 2 == 0:
            yield t[right(pl)]
        if r % 2 == 1:
            yield t[left(pr)]
        l = pl
        r = pr


def query(t, l, r, op=sum):
    return op(query_(t, l, r))


def update(tree, idx, value, op=sum):
    tree[idx] = value
    idx = parent(idx)
    while idx > 0:
        tree[idx] = op((tree[left(idx)], tree[right(idx)]))
        idx = parent(idx)


def create_tree(data, op=sum):
    t = [0] * len(data) + data
    fill(t, op)
    return t


class ArrayTree:
    def __init__(self, data, op=sum):
        d = [e for e in data]
        self.tree = [0] * len(data) + d
        self.op = op
        fill(self.tree, self.op)

    def __getitem__(self, q):
        if isinstance(q, int):
            return self.tree[index(self.tree, q)]
        l, r = q.start, q.stop
        l_idx = index(self.tree, l)
        r_idx = index(self.tree, r)
        return query(self.tree, l_idx, r_idx, op=self.op)

    def __setitem__(self, idx, val):
        i = index(self.tree, idx)
        update(self.tree, i, val, op=self.op)
