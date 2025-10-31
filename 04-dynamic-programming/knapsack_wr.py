"""Knapsack with repetitions.  Dynamic Programming."""

from collections import defaultdict as D
from collections import namedtuple as T

vals = [16, 14, 13, 14, 15, 16, 4, 17, 10, 6, 5, 13, 10, 12, 20, 6, 3, 17, 19, 6, 6, 19, 12, 1, 12, 8, 13, 9, 4, 7]

weis = [15, 4, 7, 7, 16, 20, 4, 3, 20, 4, 15, 18, 2, 1, 8, 6, 18, 20, 3, 1, 1, 11, 11, 14, 9, 2, 5, 11, 19, 17]

I = T("I", "v w")
items = [I(*e) for e in zip(vals, weis)]
dp = D(int)


def knapsack_wr(W):
    for w in range(W + 1):
        for item in items:
            if item.w <= w:
                dp[w] = max(dp[w], dp[w - item.w] + item.v)
    return dp[W]


print(knapsack_wr(53))
