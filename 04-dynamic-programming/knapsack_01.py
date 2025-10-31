"""0-1-Knapsack dynamic programming."""

from collections import defaultdict as D
from collections import namedtuple as T

vals = [2, 13, 5, 9]
weis = [2, 7, 3, 5]

I = T("I", "v w")
items = [I(*e) for e in zip(vals, weis)]
N = len(items)
dp = D(int)


def knapsack_01(W):
    for idx, it in enumerate(items):
        for w in range(W + 1):
            skip = dp[(w, idx - 1)]
            pick = 0
            if it.w <= w:
                pick = it.v + dp[(w - it.w, idx - 1)]
            dp[(w, idx)] = max(pick, skip)

    return dp[(W, N - 1)]


print(knapsack_01(16))
