r"""Karatsuba multiplication for large integers represented as strings.

This implementation multiplies two nonnegative integers given as decimal
strings using a divide-and-conquer strategy.  The inputs are padded to equal
lengths that are powers of two, then recursively split into high and low
halves:

$$x = x1 \cdot 10^{n/2} + x0$$

$$y = y1 \cdot 10^{n/2} + y0$$

The product is computed using three recursive multiplications:

$$A = x1 \cdot y1$$

$$B = x0 \cdot y0$$

$$P = (x1 + x0)(y1 + y0)$$

and combined via

$$x \cdot y = A \cdot 10^n + (P - A - B) \cdot 10^{n/2} + B$$

This reduces the number of recursive multiplications from four to three,
yielding a time complexity of

$$O(n^{\log_2 3}) \approx O(n^{1.585}),$$

which improves over the quadratic elementary school method.

The implementation uses string arithmetic with Python integers for base cases
and addition/subtraction, and handles arbitrary-length inputs.

"""

import math


def add(x, y, z="0"):
    n = len(x)
    return str(int(x) + int(y) + int(z))


def sub(x, y, z="0"):
    n = len(x)
    return str(int(x) - int(y) - int(z))


def shift(x, n):
    """Left shift x with n positions"""
    return x + "0" * n


def align(x, y):
    """Pad with 0s to get x and y same length and power of two"""
    n = max([len(x), len(y)])
    N = 2 ** math.ceil(math.log2(n))
    x = x.zfill(N)
    y = y.zfill(N)
    return x, y


def M(x, y):
    x, y = align(x, y)
    n = len(x)
    if n <= 1:
        return str(int(x) * int(y))  # single digit multiplication
    x1, x0 = x[: n // 2], x[n // 2 :]
    y1, y0 = y[: n // 2], y[n // 2 :]
    A = M(x1, y1)
    B = M(x0, y0)
    P = M(add(x1, x0), add(y1, y0))
    S = sub(P, A, B)
    R = add(shift(A, n), shift(S, n // 2), B)
    return R


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        exit("usage: karatsuba A B")
    A, B = sys.argv[1:]
    al = align(A, B)
    n = len(al[0]) + 1
    P = M(A, B)
    print(A, "*", B, "=", P.lstrip("0") or "0")
