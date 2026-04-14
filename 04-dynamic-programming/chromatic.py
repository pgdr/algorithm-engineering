"""Compute the chromatic number of a graph using subset DP.

The chromatic number of a graph is the minimum number of independent
sets whose union is the vertex set.

"""


def chromatic_number(N):
    n = 1 << len(N)
    indep = [False] * n
    indep[0] = True

    for S in range(1, n):
        v = (S & -S).bit_length() - 1
        T = S ^ (1 << v)
        indep[S] = indep[T] and ((N[v] & T) == 0)

    INF = n + 1
    dp = [INF] * n  # chromatic number of G[S]
    dp[0] = 0

    for S in range(1, n):
        T = S
        while T:
            if indep[T]:
                dp[S] = min(dp[S], dp[S ^ T] + 1)
            T = (T - 1) & S
    return dp[n - 1]


def read():
    n = int(input())
    N = [0] * n
    for u in range(n):
        Nu = [int(v) for v in input().split()]
        for v in Nu:
            N[u] |= 1 << v
            N[v] |= 1 << u
    return N


N = read()
print(chromatic_number(N))
