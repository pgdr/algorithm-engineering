"""Find the length of the longest contiguous subarray with a non-negative sum.
Uses prefix sums and two-pointer scanning over forward minima and reverse
maxima.
"""


def longest_non_negative(data):
    neginf = sum(abs(e) for e in data) * -1000  # -infty
    data = [neginf] + data + [neginf]
    presum = [0]
    for idx, val in enumerate(data):
        presum.append(val + presum[idx])

    max_rev = [0 for _ in presum]
    max_rev[-1] = presum[-1]
    for i in range(len(presum) - 2, -1, -1):
        max_rev[i] = max(presum[i], max_rev[i + 1])

    min_fwd = [0 for _ in presum]
    for i in range(1, len(presum)):
        min_fwd[i] = min(presum[i], min_fwd[i - 1])

    lo = 0
    hi = 0
    mx = 0
    while True:
        lv = min_fwd[lo]
        rv = max_rev[hi]
        mx = max(mx, (hi - lo - 1))
        if rv - lv >= 0:
            hi += 1
            if hi >= len(max_rev):
                break
        else:
            lo += 1
            if lo >= len(min_fwd):
                break
    return mx
