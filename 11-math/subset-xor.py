r"""Subset XOR via Gaussian elimination over $\mathbb{F}_2$.

Given a multiset of nonnegative integers, we may form the XOR of any subset.
Viewing each integer as a bit vector, XOR is exactly vector addition over
$\mathbb{F}_2$.  Thus the set of all subset XOR values is the linear span of
the input vectors.

The core step is Gaussian elimination over $\mathbb{F}_2$: repeatedly pick a
pivot bit, use it to eliminate that bit from all other vectors, and thereby
obtain a basis of the span.  From this basis we can

- enumerate all subset XOR values,
- count how many distinct values are reachable,
- test whether a target value is reachable, and
- compute the maximum obtainable XOR.

The elimination runs in $O(n B)$ time for $n$ integers of $B$ bits, up to
constant factors depending on the representation.

"""


def xor_basis(nums):
    """Return a reduced XOR basis for the span of nums."""
    basis = []
    for x in nums:
        y = x
        for b in basis:
            y = min(y, y ^ b)
        if y:
            basis.append(y)
            basis.sort(reverse=True)
    return basis


def subset_xor_values(nums):
    """Return all distinct subset XOR values."""
    values = [0]
    for b in xor_basis(nums):
        values += [v ^ b for v in values]
    return values


def count_distinct_subset_xors(nums):
    """Return the number of distinct subset XOR values."""
    return 1 << len(xor_basis(nums))


def max_subset_xor(nums):
    """Return the maximum XOR obtainable from a subset of nums."""
    ans = 0
    for b in sorted(xor_basis(nums), reverse=True):
        ans = max(ans, ans ^ b)
    return ans


def subset_xor_reachable(nums, target):
    """Return whether target is the XOR of some subset of nums."""
    x = target
    for b in sorted(xor_basis(nums), reverse=True):
        x = min(x, x ^ b)
    return x == 0


def subset_xor_sum(nums):
    """Return the sum of all subset XOR values, counted with multiplicity."""
    nums = list(nums)
    if not nums:
        return 0
    bit_or = 0
    for x in nums:
        bit_or |= x
    return bit_or << (len(nums) - 1)
