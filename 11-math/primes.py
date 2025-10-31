"""Factorize a number into prime factors."""

from math import sqrt


def factoring(n):
    while n % 2 == 0:
        yield 2
        n //= 2
    for i in range(3, int(sqrt(n)) + 1, 2):
        while n % i == 0:
            yield i
            n //= i
    if n > 1:
        yield n


N = 2310 * 510510
print(f"factors({N}) =", *factoring(N))
