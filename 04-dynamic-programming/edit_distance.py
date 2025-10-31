"""Compute the edit distance (Levenshtein distance) with dynamic programming.
With applications in spell checking.
"""

from collections import defaultdict


def edit_dp(a, b):
    opt = defaultdict(lambda: 10**10)
    for la in range(len(a)):
        for lb in range(len(b)):
            if la == 0 or lb == 0:
                opt[(la, lb)] = max(lb, la)
            elif a[la] == b[lb]:
                opt[(la, lb)] = opt[la - 1, lb - 1]
            else:
                op_i = opt[(la, lb - 1)]
                op_d = opt[(la - 1, lb)]
                op_c = opt[(la - 1, lb - 1)]
                opt[(la, lb)] = min(op_i, op_d, op_c) + 1
    return opt[(len(a) - 1, len(b) - 1)]
