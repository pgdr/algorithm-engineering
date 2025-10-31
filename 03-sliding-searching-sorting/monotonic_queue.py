"""Monotonic queue for $O(1)$-amortized sliding-window min/max queries.

Maintains a deque of candidate indices in monotone order as the window moves.
Use it to compute all window minima/maxima in overall $O(n)$ time.
"""

from collections import deque


def sliding_window_min(arr, k):
    """Return list of minima of each length-k window of arr."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    n = len(arr)
    if k > n:
        return []

    q = deque()  # stores indices; arr[q[0]] is current min
    out = []

    for i, x in enumerate(arr):
        # drop indices that left the window
        while q and q[0] <= i - k:
            q.popleft()

        # maintain increasing values in deque
        while q and arr[q[-1]] >= x:
            q.pop()

        q.append(i)

        # window is formed
        if i >= k - 1:
            out.append(arr[q[0]])

    return out


def sliding_window_max(arr, k):
    """Return list of maxima of each length-k window of arr."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    n = len(arr)
    if k > n:
        return []

    q = deque()  # stores indices; arr[q[0]] is current max
    out = []

    for i, x in enumerate(arr):
        while q and q[0] <= i - k:
            q.popleft()

        # maintain decreasing values in deque
        while q and arr[q[-1]] <= x:
            q.pop()

        q.append(i)

        if i >= k - 1:
            out.append(arr[q[0]])

    return out


if __name__ == "__main__":
    arr = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3

    mins = sliding_window_min(arr, k)
    maxs = sliding_window_max(arr, k)

    print("arr :", arr)
    print("k   :", k)
    print("mins:", mins)  # [-1, -3, -3, -3, 3, 3]
    print("maxs:", maxs)  # [3, 3, 5, 5, 6, 7]
