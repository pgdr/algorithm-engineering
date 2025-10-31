"""Compute the prefix sum of a given list as input."""

from sys import stdin as inp

a = [int(x) for x in inp.readline().split()]
b = []

for idx, val in enumerate(a):
    b.append(val if idx == 0 else val + b[idx - 1])

print(b)
