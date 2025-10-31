"""Exact $2^n$ algorithm for Feedback Arc Set (with memoization)."""

from functools import cache

Z = frozenset
INF = float("inf")


@cache
def W(G, v, S):
    return len(S.intersection(G[v]))


@cache
def fas(G, S, v):
    assert isinstance(S, Z), "wrong type " + str(type(S))
    if not S:
        return 0, None
    if len(S) == 1:
        if v in S:
            return 0, v
        return INF, None
    Sv = Z(S - set([v]))
    opt = INF
    opt_u = None
    for u in Sv:
        res, _ = fas(G, Sv, u)
        if res < opt:
            opt = res
            opt_u = u
    retval = opt + W(G, v, S), opt_u
    return retval
