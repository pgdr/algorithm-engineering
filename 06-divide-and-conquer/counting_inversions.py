r"""Counting inversions via merge sort (divide and conquer).

An inversion in a list $L$ is a pair of indices $(i, j)$ with $i < j$ and $L[i]
> L[j]$.  This algorithm counts all inversions while simultaneously sorting the
list.

The list is recursively split into two halves.  After counting inversions in
each half, the merge step counts *cross inversions*: whenever an element from
the right half is placed before remaining elements of the left half, it forms
inversions with all those remaining elements.

Returns a pair `(inv, sorted_L)`, where `inv` is the number of inversions and
`sorted_L` is the sorted version of $L$.

Runs in $O(n \log n)$ time.

"""


def count_and_sort(L):
    if len(L) <= 1:
        return 0, L
    n2 = len(L) // 2
    inv1, L1 = count_and_sort(L[:n2])
    inv2, L2 = count_and_sort(L[n2:])
    count = 0
    L = []
    p1 = p2 = 0
    while p1 < len(L1) and p2 < len(L2):
        if L1[p1] < L2[p2]:
            L.append(L1[p1])
            p1 += 1
        else:
            L.append(L2[p2])
            p2 += 1
            count += len(L1) - p1
    return inv1 + inv2 + count, L + L1[p1:] + L2[p2:]
