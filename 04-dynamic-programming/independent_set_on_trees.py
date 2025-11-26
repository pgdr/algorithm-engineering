"""Weighted Independent Set on trees."""

from collections import defaultdict as D
from collections import namedtuple as T

Tree = T("Tree", "r v e w")
Dp = T("Dp", "pick skip")


def dfs(tree, r):
    children = [dfs(tree, child) for child in tree.e[r]]
    return Dp(
        tree.w[r] + sum(c.skip for c in children),
        sum(max(c.skip, c.pick) for c in children),
    )


children = D(
    list,
    {
        1: [2, 10],
        2: [3, 4],
        3: [5, 6],
        4: [7, 8, 9],
        10: [11, 12],
    },
)

weights = {
    1: 15,
    2: 19,
    10: 5,
    3: 10,
    4: 3,
    5: 7,
    6: 2,
    7: 5,
    8: 9,
    9: 2,
    11: 10,
    12: 1,
}

tree = Tree(r=1, v=list(range(1, 13)), e=children, w=weights)
print(max(dfs(tree, tree.r)))
