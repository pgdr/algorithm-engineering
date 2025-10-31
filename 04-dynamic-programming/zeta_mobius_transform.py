r"""Zeta and Möbius transforms on the subset lattice.

The zeta transform, also known as Yates DP, takes a function $f$ on subsets and
computes

$$F[S] = \sum_{T \subseteq S} f[T].$$

The Möbius transform inverts this relation:

$$f[S] = \sum_{T \subseteq S} (-1)^{|S \setminus T|} F[T].$$

This is the algebraic form of inclusion--exclusion, so inclusion--exclusion is
precisely Möbius inversion on the subset lattice.

These transforms are basic tools in subset DP.  They are also used in fast
subset-sum convolution: transform two functions, combine them appropriately,
and invert to recover the convolution.

"""


def zeta_transform(f):
    r"""Subset zeta transform (Yates DP): F[S] = \sum_{T \subseteq S} f[T]."""
    n = len(f).bit_length() - 1
    f = f[:]
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                f[mask] += f[mask ^ (1 << i)]
    return f


def mobius_transform(f):
    """Subset Möbius transform: inverse of the zeta transform."""
    n = len(f).bit_length() - 1
    f = f[:]
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                f[mask] -= f[mask ^ (1 << i)]
    return f


inclusion_exclusion = mobius_transform
