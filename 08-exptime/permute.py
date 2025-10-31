"""Enumerate all permutations of a list"""


def permute(seq, i, n):
    if i == n:
        yield tuple(seq)
    for j in range(i, n):
        seq[j], seq[i] = seq[i], seq[j]
        yield from permute(seq, i + 1, n)
        seq[j], seq[i] = seq[i], seq[j]


def permutations(seq):
    yield from permute(list(seq), 0, len(seq))
