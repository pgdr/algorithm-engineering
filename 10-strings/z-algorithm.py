r"""Z algorithm for linear-time prefix matching.

For each position $i$ in a string $s$, let $Z[i]$ be the length of the longest
prefix of $s$ matching the substring starting at $i$.  The algorithm maintains
a rightmost matching interval $[l, r)$ and reuses it to avoid redundant
comparisons.

Runs in $O(n)$ time.

"""


def z_algorithm(s):
    n = len(s)
    Z = [0] * n
    l = r = 0

    for i in range(1, n):
        if i < r:
            Z[i] = min(r - i, Z[i - l])
        while i + Z[i] < n and s[Z[i]] == s[i + Z[i]]:
            Z[i] += 1
        if i + Z[i] > r:
            l, r = i, i + Z[i]

    return Z
