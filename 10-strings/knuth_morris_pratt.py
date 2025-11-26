"""Knuth–Morris–Pratt (KMP) algorithm for pattern matching."""


def match(text, pattern):
    """Yield indices i where text[i:len(P)] == P"""
    for i in range(len(text)):
        if text[i : i + len(pattern)] == pattern:
            yield i


def bf(text, pattern):
    """Return the indices i where text[i:len(P)] == P"""
    if len(text) <= len(pattern):
        return [0] if text == pattern else []
    indices = []
    for i in range(len(text) - len(pattern) + 1):
        matches = True
        for j in range(len(pattern)):
            if text[i + j] != pattern[j]:
                matches = False
                break
        if matches:
            indices.append(i)
    return indices


def longest_prefix_suffix(pattern):
    """Return lps table.

    `lps[i]` is where to start matching if first mismatch at i+1, or
    length of longest suffix that is also a prefix of the string from 0
    to i.

    """
    n, k = 0, len(pattern)
    lps = [0] * k
    idx = 1

    while idx < k:
        print(idx, n, pattern[idx], pattern[n])
        if pattern[idx] == pattern[n]:
            n += 1
            lps[idx] = n
        elif n != 0:
            n = lps[n - 1]
        else:
            lps[idx] = 0
        idx += 1
    return lps


def kmp(text, pattern, lps=None):
    """Yield indices idx where text[idx:idx+N] == P"""
    N, M = len(text), len(pattern)

    if lps is None:
        lps = longest_prefix_suffix(pattern)

    txt_idx = 0
    pat_idx = 0
    while txt_idx < N:
        if pattern[pat_idx] == text[txt_idx]:
            txt_idx += 1
            pat_idx += 1

        if pat_idx == M:
            yield txt_idx - pat_idx
            pat_idx = lps[pat_idx - 1]

        # mismatch after pat_idx matches
        elif txt_idx < N and pattern[pat_idx] != text[txt_idx]:
            # Do not match lps[0..lps[pat_idx-1]] characters,
            # they will match anyway
            if pat_idx != 0:
                pat_idx = lps[pat_idx - 1]
            else:
                txt_idx += 1
