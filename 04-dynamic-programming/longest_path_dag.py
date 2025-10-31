"""Longest Path in a DAG.  Dynamic programming."""

from collections import defaultdict as D

dp = D(int)

dag = [
    (0, 1),
    (1, 2),
    (1, 6),
    (1, 7),
    (4, 0),
    (4, 6),
    (5, 1),
    (5, 3),
    (6, 7),
    (7, 3),
]


def neighbors(v):
    for e in dag:
        if e[0] == v:
            yield e[1]


def lp(v):
    if v in dp:
        return dp[v]
    children = [lp(u) for u in neighbors(v)]
    val = 1 + max(children) if children else 0
    dp[v] = val
    return val


for i in range(8):
    print(i, lp(i))

print(dp)
d = {}
for i in range(8):
    d[i] = dp[i]
print(d)
# {2: 0, 3: 0, 7: 1, 6: 2, 1: 3, 0: 4, 4: 5, 5: 4}
