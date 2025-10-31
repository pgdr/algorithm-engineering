r"""Zeta $\zeta$ transform over subsets (submasks).

The subset zeta transform (a.k.a. SOS DP) is: given an array $f$ indexed by
bitmasks in $\{0 \dots 2^{n-1}\}$, the subset zeta transform produces $f'$
where:

$$f'[mask] = \sum_{\text{sub} \subseteq \text{mask}} f[\text{sub}].$$

This is useful when you want, for every mask, the sum/aggregation over all its
submasks, but you want all answers at once in $O(n \cdot 2^n)$ rather than
doing an $O(3^n)$ total amount of submask iteration.

"""


def zeta_submasks(f, n):
    N = 1 << n
    for i in range(n):
        bit = 1 << i
        for mask in range(N):
            if mask & bit:
                f[mask] += f[mask ^ bit]
    return f


if __name__ == "__main__":
    n = 3
    w = [0] * (1 << n)
    w[0b001] = 5  # {0}
    w[0b101] = 2  # {0,2}
    w[0b110] = 7  # {1,2}
    g = w[:]
    zeta_submasks(g, n)
    print("g[0b101] =", g[0b101])
