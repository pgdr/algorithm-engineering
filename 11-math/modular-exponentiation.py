r"""Modular exponentiation by repeated squaring.

Compute $a^b \mod m$ by scanning the bits of $b$ from low to high.  Repeatedly
square the base, and whenever the current bit is set, multiply it into the
answer modulo $m$.

Runs in $O(\log b)$ time.

"""


def modexp(a, b, m):
    assert b >= 0
    assert m > 0
    x = a % m
    y = 1

    while b:
        if b & 1:
            y = (y * x) % m
        x = (x * x) % m
        b >>= 1

    return y
