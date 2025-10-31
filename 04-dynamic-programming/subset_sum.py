"""Dynamic programming routine (recursion with memoization) for Subset Sum."""

from functools import cache

S = [267, 493, 869, 961, 1000, 1153, 1246, 1598, 1766, 1922]
x = 5842


@cache
def sss(i=len(S) - 1, t=x):
    if t == 0:
        return True
    if i == 0:
        return S[0] == t

    return sss(i - 1, t) or sss(i - 1, t - S[i])


print(sss())
print(sss.keys())
