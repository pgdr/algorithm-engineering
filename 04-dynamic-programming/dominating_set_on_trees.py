"""Weighted Dominating Set on Trees.  Dynamic Programming."""

from collections import defaultdict as D
from collections import namedtuple as T

Tree = T("Tree", "r v e w")
Dp = T("Dp", "pick skip_below skip_above")


DP = {}


def dfs(tree, r):
    pick = tree.w[r]
    skip_above = 0

    # leaf
    if not tree.e[r]:
        skip_below = 10**10  # impossible, so infty
        DP[r] = Dp(pick, skip_below, skip_above)
        return

    for c in tree.e[r]:
        dfs(tree, c)
        pick += min(DP[c].pick, DP[c].skip_above)  # we dom c
        skip_above += min(DP[c].pick, DP[c].skip_below)  # we don't dom c

    skip_below = 10**10  # infty
    for dom in tree.e[r]:  # guess dominator below
        cost = DP[dom].pick
        for c in tree.e[r]:
            if c == dom:
                continue
            cost += min(DP[c].pick, DP[c].skip_below)
        skip_below = min(skip_below, cost)
    DP[r] = Dp(pick, skip_below, skip_above)


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
dfs(tree, tree.r)
print(DP)
