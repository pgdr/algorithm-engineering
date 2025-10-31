## 01-greedy
### Huffman
```python
"""Compute the Huffman encoding of a provided text as input.
Outputs the encoding for each symbol.
"""

import sys
import heapq
from collections import namedtuple
from collections import Counter

Node = namedtuple("Node", "f s ascii children")


def huffman(freq):
    pq = [Node(f, s, ord(s), None) for (s, f) in freq.items()]
    heapq.heapify(pq)
    while len(pq) > 1:
        v1 = heapq.heappop(pq)
        v2 = heapq.heappop(pq)
        v = Node(v1.f + v2.f, "", -1, children=(v1, v2))
        heapq.heappush(pq, v)
    return heapq.heappop(pq)


def dfs(root, prefix=""):
    if not root.children:
        yield (root, prefix)
        return
    yield from dfs(root.children[0], prefix + "0")
    yield from dfs(root.children[1], prefix + "1")


def get_codes(text):
    freq = Counter(text)
    tree = huffman(freq)
    return sorted(dfs(tree), key=lambda e: len(e[1]))


if __name__ == "__main__":
    text = sys.stdin.read().strip()
    codes = get_codes(text)
    for node, code in codes:
        symbol = node.s
        if symbol == "\n":
            symbol = "\\n"
        if symbol == "\t":
            symbol = "\\t"
        if node.ascii == 13:
            symbol = "␍"
        if node.ascii == 65279:
            symbol = "‗"
        lhs = f"{symbol.ljust(2)} ({str(node.ascii).rjust(4)})"
        print(lhs.ljust(10), "\t", len(code), "\t", code)

```

### Stable Matching
```python
"""The Gale–Shapley algorithm for Stable Matching.  The algorithm runs
in linear time in the total size of preferences, or, $O(n^2)$ where $n$
is the number of "hospitals".

"""


def stable_matching(hospital_preferences, student_preferences):
    students = [s for s in student_preferences]
    hospitals = [h for h in hospital_preferences]
    proposals = {h: 0 for h in hospitals}
    unmatched_hospitals = [h for h in hospitals]
    student = {h: None for h in hospitals}
    hospital = {s: None for s in students}
    inrank = {s: {} for s in students}  # maps s to each hospital's s-ranking
    for s in students:
        for idx, h in enumerate(student_preferences[s]):
            inrank[s][h] = idx

    while unmatched_hospitals:
        h = unmatched_hospitals.pop()
        nxt = proposals[h]  # we could pop here instead
        s = hospital_preferences[h][nxt]
        proposals[h] += 1
        # h proposes to its best student not yet proposed to
        if not hospital[s]:
            # s is available
            hospital[s] = h
            student[h] = s
        else:
            sh = hospital[s]
            rank_sh = inrank[s][sh]
            rank_h = inrank[s][h]
            if rank_h < rank_sh:
                # s dumps sh for h
                hospital[s] = h
                student[sh] = None
                student[h] = s
                unmatched_hospitals.append(sh)
            else:
                # s rejects
                unmatched_hospitals.append(h)
    return student


def _generate_instance(n):
    from random import sample as shuffle

    hospitals = [f"h{i}" for i in range(n)]
    students = [f"s{i}" for i in range(n)]

    hospital_preferences = {h: students[:n] for h in hospitals[:n]}
    student_preferences = {s: hospitals[:n] for s in students[:n]}

    for h in hospitals[:n]:
        hospital_preferences[h] = shuffle(hospital_preferences[h], n)

    for s in students[:n]:
        student_preferences[s] = shuffle(student_preferences[s], n)

    return hospital_preferences, student_preferences


if __name__ == "__main__":
    hospital_preferences, student_preferences = _generate_instance(20)
    M = stable_matching(hospital_preferences, student_preferences)
    for h in M:
        print(f"Hospital {h} + Student {M[h]}")

```

## 02-graphs
### Bipartite Matching
```python
"""Compute a maximum matching in an unweighted bipartite graph using DFS to
find augmenting paths.  Implements the classic alternating-path approach for
bipartite matching.
"""


def max_bipartite_matching(G, A, B):
    matchA = {u: None for u in A}
    matchB = {v: None for v in B}

    def dfs(u, seen):
        for v in G[u]:
            if v in seen:
                continue
            seen.add(v)
            if matchB[v] is None or dfs(matchB[v], seen):
                matchA[u] = v
                matchB[v] = u
                return True
        return False

    for u in A:
        dfs(u, set())

    return matchA, matchB

```

### Dfs Tree
```python
"""Run DFS and get a DFS tree"""

from collections import defaultdict as D
from collections import namedtuple as T

DiGraph = T("DiGraph", "V E")


def outneighbors(G, v):
    # Yes we use edge list, the worst one
    for x, y in G.E:
        if v == x:
            yield y


pre = D(int)
post = {}
time = 0
tree_edges = set()


def dfs(G, node):
    global time
    time += 1
    pre[node] = time
    for v in outneighbors(G, node):
        if not pre[v]:
            tree_edges.add((node, v))
            dfs(G, v)
    time += 1
    post[node] = time


def dfs_tree(G, pre, post, tree_edges):
    edge_type = {}
    for edge in G.E:
        u, v = edge
        if edge in tree_edges:
            edge_type[edge] = "tree"
        elif pre[u] < pre[v] < post[v] < post[u]:
            edge_type[edge] = "forward"
        elif pre[v] < pre[u] < post[u] < post[v]:
            edge_type[edge] = "back"
        elif pre[v] < post[v] < pre[u] < post[u]:
            edge_type[edge] = "cross"
    return edge_type


def topological_sort(G, post):
    toposort = [-1] * (2 * len(G.V) + 1)
    for v in G.V:
        toposort[post[v]] = v
    return [x for x in reversed(toposort) if x >= 0]

```

### Dijkstra
```python
"""Simple Dijkstra implementation running in time $O(m \\log n)$."""

import heapq


def dijkstra(G, source):
    dist = {v: float("inf") for v in G.nodes}
    dist[source] = 0
    priority_queue = [(0, source)]

    while priority_queue:
        current_distance, v = heapq.heappop(priority_queue)
        if current_distance > dist[v]:
            continue

        for u in G.neighbors(v):
            weight = G[v][u]["weight"]
            if dist[u] > dist[v] + weight:
                dist[u] = dist[v] + weight
                heapq.heappush(priority_queue, (dist[u], u))

    return dist

```

### Kahns Algorithm
```python
"""Kahn's algorithm produces a topological order of a DAG by repeatedly
removing vertices with zero in-degree and updating their neighbors' in-degrees
until none remain.
"""

from collections import deque


def kahns(G):
    indegree = {u: 0 for u in G.nodes}
    for u in G.nodes:
        for v in G[u]:
            indegree[v] += 1

    queue = deque([u for u in G.nodes if indegree[u] == 0])
    count = 0

    while queue:
        u = queue.popleft()
        yield u
        count += 1
        for v in G[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    if count != len(G.nodes):
        raise ValueError("Graph is not a DAG")

```

### Kosarajus Algorithm
```python
"""A strongly connected component (SCC) is a maximal set of vertices where
every vertex can reach every other by directed paths.  Kosaraju's algorithm
finds all SCCs by doing a DFS to record finish times, reversing all edges, then
running DFS again in reverse finish order.
"""


def kosaraju(G):
    vis, order = set(), []

    def dfs(u, graph, collect):
        vis.add(u)
        for v in graph[u]:
            if v not in vis:
                dfs(v, graph, collect)
        collect.append(u)

    for u in G:
        if u not in vis:
            dfs(u, G, order)

    R = {u: [] for u in G}
    for u in G:
        for v in G[u]:
            R[v].append(u)

    vis.clear()
    while order:
        u = order.pop()
        if u not in vis:
            comp = []
            dfs(u, R, comp)
            yield comp

```

### Rrt
```python
"""Grow a rapidly-exploring random tree (RRT) toward a target while avoiding
obstacles.  Expands the tree incrementally by sampling feasible edges until the
goal region is reached or iterations are exhausted.
"""


def rrt(start, target, obstacles, n=1, tree=None):
    if tree is None:
        tree = Tree(start, set([start]), set())  # rapidly exploring random tree (RRT)
    for i in range(n):
        edge = sample_tree_node(tree, obstacles)
        if edge:
            q, v = edge
            tree.V.add(q)
            tree.E.add((q, v))
            if dist(q, target) < CLOSE:
                return tree, True
    return tree, False

```

### Topological Sort With Dfs
```python
"""Use postorder DFS to get a reversed topological ordering."""


def dfs_toposort(G):
    visited = set()
    order = []

    def dfs(u):
        if u not in visited:
            for v in G[u]:
                dfs(v)
            visited.add(u)
            order.append(u)

    for u in G.nodes:
        if u not in visited:
            dfs(u)

    yield from reversed(order)

```

### Union Find
```python
"""Simple Union Find (with path compression) datastructure implementation for
use in MST.
"""


class Unionfind:
    def __init__(self, size):
        self._roots = list(range(size))
        self._rank = [0 for _ in range(size)]

    def __getitem__(self, elt):
        root = elt
        while root != self._roots[root]:
            root = self._roots[root]

        # path compression
        while elt != root:
            parent = self._roots[elt]
            self._roots[elt] = root
            elt = parent

        return root

    def is_connected(self, a, b):
        return self[a] == self[b]

    def union(self, p, q):
        r1 = self[p]
        r2 = self[q]
        if r1 == r2:
            return False

        self._roots[r1] = r2
        return True

```

## 03-sliding-searching-sorting
### Consecutive Sum
```python
"""Maximum consecutive sum in an array with possibly negative values."""


def consecutive_sum(array, idx=0, con_sum=0, global_max=0):
    if idx >= len(array):
        return global_max
    copt = max(array[idx], array[idx] + con_sum)
    gopt = max(global_max, copt)
    return consecutive_sum(array, idx + 1, copt, gopt)


print(consecutive_sum([4, 0, 3, -8, 1, 4, -2, -10]))

```

### Longest Nonnegative
```python
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

```

### Monotonic Queue
```python
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

```

### Prefix Sum
```python
"""Compute the prefix sum of a given list as input."""

from sys import stdin as inp

a = [int(x) for x in inp.readline().split()]
b = []

for idx, val in enumerate(a):
    b.append(val if idx == 0 else val + b[idx - 1])

print(b)

```

### Sliding Window
```python
"""Sliding window to find best $k$ sized window in linear time."""

data = [3.5, 7.5, 8.0, 5.7, 3.1, 4.2, 7.2, 0.1, 3.4, 1.2, -4]
k = 3
acc = sum(data[:k])
best = acc
winstart = 0

for winstart in range(1, len(data) - k + 1):
    acc += -data[winstart - 1] + data[winstart + k - 1]
    if acc > best:
        best = acc

print(best)

```

## 04-dynamic-programming
### Bellman Ford
```python
"""Single-source shortest paths (SSSP) with negative edges can be solved by
Bellman–Ford, which uses dynamic programming over path lengths to relax edges
up to $n-1$ times.  If a further relaxation still decreases a distance, it
indicates a negative-weight cycle reachable from the source.
"""


def bellman_ford(n, G, s):
    """Return (dist, True) if successful, and (dist, False) if we have detected
    a negative cycle.
    """
    dist = [float("inf")] * n
    dist[s] = 0
    for _ in range(n - 1):
        for u, v, w in G.edges:
            dist[v] = min(dist[v], dist[u] + w)
    for u, v, w in G.edges:
        if dist[u] + w < dist[v]:
            dist, False
    return dist, True

```

### Dominating Set On Trees
```python
"""Weighted Dominating Set on Trees.  Dynamic Programming."""

from collections import defaultdict as D
from collections import namedtuple as T

Tree = T("Tree", "r v e w")
Dp = T("Dp", "pick skip_below skip_above")


DP = {}


def dfs(tree, r):
    pick = tree.w[r]
    skip_above = 0

    # leaf
    if not tree.e[r]:
        skip_below = 10**10  # impossible, so infty
        DP[r] = Dp(pick, skip_below, skip_above)
        return

    for c in tree.e[r]:
        dfs(tree, c)
        pick += min(DP[c].pick, DP[c].skip_above)  # we dom c
        skip_above += min(DP[c].pick, DP[c].skip_below)  # we don't dom c

    skip_below = 10**10  # infty
    for dom in tree.e[r]:  # guess dominator below
        cost = DP[dom].pick
        for c in tree.e[r]:
            if c == dom:
                continue
            cost += min(DP[c].pick, DP[c].skip_below)
        skip_below = min(skip_below, cost)
    DP[r] = Dp(pick, skip_below, skip_above)


children = D(
    list,
    {
        1: [2, 10],
        2: [3, 4],
        3: [5, 6],
        4: [7, 8, 9],
        10: [11, 12],
    },
)

weights = {
    1: 15,
    2: 19,
    10: 5,
    3: 10,
    4: 3,
    5: 7,
    6: 2,
    7: 5,
    8: 9,
    9: 2,
    11: 10,
    12: 1,
}

tree = Tree(r=1, v=list(range(1, 13)), e=children, w=weights)
dfs(tree, tree.r)
print(DP)

```

### Edit Distance
```python
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

```

### Fast Zeta Transform
```python
"""SOS DP / Fast Zeta Transform over subsets (submasks).

This module provides the classic *subset zeta transform* (a.k.a. SOS DP step).

Given an array f indexed by bitmasks in {0..2^n-1}, the subset zeta transform
produces f' where:

    f'[mask] = sum_{sub ⊆ mask} f[sub].

This is useful when you want, for every mask, the sum/aggregation over all its
submasks, but you want all answers at once in O(n·2^n) rather than doing an
O(3^n) total amount of submask iteration.
"""


def zeta_submasks(f, n):
    N = 1 << n
    for i in range(n):
        bit = 1 << i
        for mask in range(N):
            if mask & bit:
                f[mask] += f[mask ^ bit]
    return f


if __name__ == "__main__":
    n = 3
    w = [0] * (1 << n)
    w[0b001] = 5  # {0}
    w[0b101] = 2  # {0,2}
    w[0b110] = 7  # {1,2}
    g = w[:]
    zeta_submasks(g, n)
    print("g[0b101] =", g[0b101])

```

### Feedback Arc Set
```python
"""Exact $2^n$ algorithm for Feedback Arc Set (with memoization)."""

from functools import cache

Z = frozenset
INF = float("inf")


@cache
def W(G, v, S):
    return len(S.intersection(G[v]))


@cache
def fas(G, S, v):
    assert isinstance(S, Z), "wrong type " + str(type(S))
    if not S:
        return 0, None
    if len(S) == 1:
        if v in S:
            return 0, v
        return INF, None
    Sv = Z(S - set([v]))
    opt = INF
    opt_u = None
    for u in Sv:
        res, _ = fas(G, Sv, u)
        if res < opt:
            opt = res
            opt_u = u
    retval = opt + W(G, v, S), opt_u
    return retval

```

### Floyd-warshall
```python
"""APSP (All-Pairs Shortest Paths) is the problem of finding the shortest path
distances between every pair of vertices in a weighted graph.

Floyd–Warshall is a classic dynamic programming algorithm for APSP.

By updating `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])` for each
intermediate vertex `k`, it computes shortest paths for all pairs in $O(n^3)$
time.

"""


def floyd_warshall(dist):
    n = len(dist)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

```

### Independent Set On Trees
```python
"""Weighted Independent Set on trees."""

from collections import defaultdict as D
from collections import namedtuple as T

Tree = T("Tree", "r v e w")
Dp = T("Dp", "pick skip")


def dfs(tree, r):
    children = [dfs(tree, child) for child in tree.e[r]]
    return Dp(
        tree.w[r] + sum(c.skip for c in children),
        sum(max(c.skip, c.pick) for c in children),
    )


children = D(
    list,
    {
        1: [2, 10],
        2: [3, 4],
        3: [5, 6],
        4: [7, 8, 9],
        10: [11, 12],
    },
)

weights = {
    1: 15,
    2: 19,
    10: 5,
    3: 10,
    4: 3,
    5: 7,
    6: 2,
    7: 5,
    8: 9,
    9: 2,
    11: 10,
    12: 1,
}

tree = Tree(r=1, v=list(range(1, 13)), e=children, w=weights)
print(max(dfs(tree, tree.r)))

```

### Knapsack 01
```python
"""0-1-Knapsack dynamic programming."""

from collections import defaultdict as D
from collections import namedtuple as T

vals = [2, 13, 5, 9]
weis = [2, 7, 3, 5]

I = T("I", "v w")
items = [I(*e) for e in zip(vals, weis)]
N = len(items)
dp = D(int)


def knapsack_01(W):
    for idx, it in enumerate(items):
        for w in range(W + 1):
            skip = dp[(w, idx - 1)]
            pick = 0
            if it.w <= w:
                pick = it.v + dp[(w - it.w, idx - 1)]
            dp[(w, idx)] = max(pick, skip)

    return dp[(W, N - 1)]


print(knapsack_01(16))

```

### Knapsack Wr
```python
"""Knapsack with repetitions.  Dynamic Programming."""

from collections import defaultdict as D
from collections import namedtuple as T

vals = [16, 14, 13, 14, 15, 16, 4, 17, 10, 6, 5, 13, 10, 12, 20, 6, 3, 17, 19, 6, 6, 19, 12, 1, 12, 8, 13, 9, 4, 7]

weis = [15, 4, 7, 7, 16, 20, 4, 3, 20, 4, 15, 18, 2, 1, 8, 6, 18, 20, 3, 1, 1, 11, 11, 14, 9, 2, 5, 11, 19, 17]

I = T("I", "v w")
items = [I(*e) for e in zip(vals, weis)]
dp = D(int)


def knapsack_wr(W):
    for w in range(W + 1):
        for item in items:
            if item.w <= w:
                dp[w] = max(dp[w], dp[w - item.w] + item.v)
    return dp[W]


print(knapsack_wr(53))

```

### Line Breaking
```python
"""Use DP to minimize the raggedness of a paragraph of text."""


def break_lines(words, column=72):
    n = len(words)
    lens = [len(w) for w in words]
    dp = [(0, None)] * (n + 1)
    dp[n] = (0, None)
    for i in range(n - 1, -1, -1):
        best = (float("inf"), None)
        line_len = 0
        for j in range(i, n):
            line_len += lens[j] + (0 if j == i else 1)
            if line_len > column:
                break
            cost = 0 if j == n - 1 else (column - line_len) ** 3
            total = cost + dp[j + 1][0]
            if total < best[0]:
                best = (total, j + 1)
                dp[i] = best
                lines = []
                i = 0
    while i < n:
        j = dp[i][1]
        lines.append(" ".join(words[i:j]))
        i = j
    return lines


def justify(paragraph, column=72):
    words = paragraph.strip().split(" ")
    lines = break_lines(words, column)
    filled = []
    for line in lines[:-1]:
        chars = list(line)
        idx = 0
        while len(chars) < column:
            try:
                idx = chars.index(".", idx) + 1
                chars.insert(idx, " ")
                idx += 2
            except ValueError:
                chars.append(" ")
                idx = 0
        filled.append("".join(chars[:column]))
    filled.append(lines[-1])
    return "\n".join(filled)

```

### Linear Regression Queries
```python
"""Given a set of $n$ $(x, y)$ points, compute the least squares for the data
set.  In addition, create a data structure that can answer query
`least_squares(i, j)` of all points in `data[i:j]`.  Using dynamic programming
we can compute the entire data structure in time $O(n^2)$.
"""

from collections import namedtuple

RegressionState = namedtuple("RegressionState", "a b n sum_x sum_y sum_xy sum_x2 a_nom denom b_nom")


def regression(x, y, state=None):
    if state is None or state.n == 0:
        n = 1
        sum_x, sum_y = x, y
        sum_xy, sum_x2 = x * y, x * x
    else:
        n = state.n + 1
        sum_x = state.sum_x + x
        sum_y = state.sum_y + y
        sum_xy = state.sum_xy + x * y
        sum_x2 = state.sum_x2 + x * x

    a_nom = n * sum_xy - sum_x * sum_y
    denom = n * sum_x2 - sum_x * sum_x
    b_nom = sum_y * sum_x2 - sum_x * sum_xy

    if denom == 0:
        a = 0.0
        b = sum_y / n
    else:
        a = a_nom / denom
        b = b_nom / denom

    return RegressionState(a, b, n, sum_x, sum_y, sum_xy, sum_x2, a_nom, denom, b_nom)


def build_dp(points):
    N = len(points)
    DP = {}
    for i in range(N):
        DP[(i, i)] = regression(*points[i])

    for i in range(N):
        for j in range(i + 1, N):
            DP[(i, j)] = regression(*points[j], DP[(i, j - 1)])
    return DP


# RUNNING THE CODE
import random


def generate_points(N):
    points = []
    for x in range(N):
        y = round(2 + x + (random.randint(-200, 200) * 0.12), 3)
        points.append((1 + x, y))
    return points


def axb(state, r=5):
    """Print s.a and s.b as "a·x+b"."""
    a = round(state.a, r)
    b = round(state.b, r)
    if b >= 0:
        return f"{a}·x + {b} ({state.n} points)"
    return f"{a}·x - {-b} ({state.n} points)"


RegressionState.__str__ = axb

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        exit("Usage: ./regression.py <N>")
    N = int(sys.argv[1])
    random.seed(1000 * N % 2**20 - 1)
    points = generate_points(N)

    DP = build_dp(points)
    line1 = DP[(0, N - 1)]
    print(line1)

```

### Longest Path Dag
```python
"""Longest Path in a DAG.  Dynamic programming."""

from collections import defaultdict as D

dp = D(int)

dag = [
    (0, 1),
    (1, 2),
    (1, 6),
    (1, 7),
    (4, 0),
    (4, 6),
    (5, 1),
    (5, 3),
    (6, 7),
    (7, 3),
]


def neighbors(v):
    for e in dag:
        if e[0] == v:
            yield e[1]


def lp(v):
    if v in dp:
        return dp[v]
    children = [lp(u) for u in neighbors(v)]
    val = 1 + max(children) if children else 0
    dp[v] = val
    return val


for i in range(8):
    print(i, lp(i))

print(dp)
d = {}
for i in range(8):
    d[i] = dp[i]
print(d)
# {2: 0, 3: 0, 7: 1, 6: 2, 1: 3, 0: 4, 4: 5, 5: 4}

```

### Subset Sum
```python
"""Dynamic programming routine (recursion with memoization) for Subset Sum."""

from functools import cache

S = [267, 493, 869, 961, 1000, 1153, 1246, 1598, 1766, 1922]
x = 5842


@cache
def sss(i=len(S) - 1, t=x):
    if t == 0:
        return True
    if i == 0:
        return S[0] == t

    return sss(i - 1, t) or sss(i - 1, t - S[i])


print(sss())
print(sss.keys())

```

## 05-segment-trees
### Segment Tree
```python
"""Basic implementation of a segment tree."""

left = lambda x: 2 * x
right = lambda x: 2 * x + 1
parent = lambda x: x // 2
index = lambda T, i: len(T) // 2 + i


def fill(tree, op):
    internal = range(1, len(tree) // 2)
    for idx in reversed(internal):
        tree[idx] = op((tree[left(idx)], tree[right(idx)]))


def query_(t, l, r):
    yield t[l]  # yield t[r] in incl case
    while True:
        pl = parent(l)
        pr = parent(r)
        if pl == pr:
            return
        if l % 2 == 0:
            yield t[right(pl)]
        if r % 2 == 1:
            yield t[left(pr)]
        l = pl
        r = pr


def query(t, l, r, op=sum):
    return op(query_(t, l, r))


def update(tree, idx, value, op=sum):
    tree[idx] = value
    idx = parent(idx)
    while idx > 0:
        tree[idx] = op((tree[left(idx)], tree[right(idx)]))
        idx = parent(idx)


def create_tree(data, op=sum):
    t = [0] * len(data) + data
    fill(t, op)
    return t


class ArrayTree:
    def __init__(self, data, op=sum):
        d = [e for e in data]
        self.tree = [0] * len(data) + d
        self.op = op
        fill(self.tree, self.op)

    def __getitem__(self, q):
        if isinstance(q, int):
            return self.tree[index(self.tree, q)]
        l, r = q.start, q.stop
        l_idx = index(self.tree, l)
        r_idx = index(self.tree, r)
        return query(self.tree, l_idx, r_idx, op=self.op)

    def __setitem__(self, idx, val):
        i = index(self.tree, idx)
        update(self.tree, i, val, op=self.op)

```

### Sqrt Decomposition
```python
"""Basic implementation for SQRT decomposition.  The SQRT data structure can
answer range queries such as sum, max, min in time $O(\\sqrt n)$ and can do update
element in $O(\\sqrt n)$ as well.

In theory, this data structure can take any [associative
operation](https://en.wikipedia.org/wiki/Associative_property) (e.g., gcd,
lcm), and the elements can even be higher order structures such as matrices
with the operation being matrix multiplication.  Indeed, you can take elements
over $\\text{GL}_n(\\mathbb{C})$, and answer such queries.
"""

import math


class SQRT:
    """The SQRT data structure.

    The SQRT data structure can answer range queries such as sum, max,
    min in time $O(\\sqrt n)$ and can do update element in $O(\\sqrt n)$ as well.

    The update element functionality makes this a data structure that
    outperforms prefix sum in some cases.

    """

    def _update(self, block):
        if block < 0:
            return
        self._block_data[block] = {
            "sum": sum(self._blocks[block]),
            "min": min(self._blocks[block]),
            "max": max(self._blocks[block]),
        }

    def _initialize(self, data):
        block = -1
        for idx, e in enumerate(data):
            if (idx % self._block_size) == 0:
                self._update(block)
                block += 1
                self._blocks.append([])
            self._blocks[block].append(e)
        if block not in self._block_data:
            # the trailing block
            self._update(block)

    def __init__(self, data):
        """Create a SQRT decomposition of iterable data.

        Complexity: O(n).

        """
        d = [e for e in data]
        if not d:
            raise ValueError("SQRT undefined for empty set")
        self._glob_min = min(d) - 1
        self._glob_max = max(d) + 1

        self._len = len(d)
        self._blocks = []
        self._block_size = int(math.sqrt(len(d)))
        self._block_data = {}
        self._initialize(d)

    def __len__(self):
        return self._len

    def _pos(self, idx):
        return idx // self._block_size, idx % self._block_size

    def __getitem__(self, coord):
        """Get a statistics dictionary for either an index or a slice.

        SQRT(lst)[l:r] returns a dictionary with keys "sum", "min", and
        "max", mapping to their respective values for the given range.

        Complexity: $O(\\sqrt n)$.

        """
        if isinstance(coord, int):
            b, l = self._pos(coord)
            return self._blocks[b][l]
        if isinstance(coord, slice):
            if coord.step is not None:
                raise ValueError("SQRT cannot stride")
            coord = (coord.start, coord.stop)
        if not isinstance(coord, tuple):
            raise ValueError(f"cannot unpack {coord} in sqrt[(l, r)]")
        if len(coord) != 2:
            raise ValueError(f"size mismatch for {coord} in sqrt[(l, r)]")
        l, r = coord
        l_block, l_loc = self._pos(l)
        r_block, r_loc = self._pos(r)

        if l_block == r_block:
            # special case when l and r within same block
            view = self._blocks[l_block][l_loc:r_loc]
            return {
                "sum": sum(view),
                "min": min(view),
                "max": max(view),
            }

        stats = {
            "sum": 0,
            "min": self._glob_max,
            "max": self._glob_min,
        }
        for l_idx in range(l_loc, self._block_size):
            e = self._blocks[l_block][l_idx]
            stats["sum"] += e
            stats["min"] = min(stats["min"], e)
            stats["max"] = max(stats["max"], e)

        for b in range(l_block + 1, r_block):
            stats["sum"] += self._block_data[b]["sum"]
            stats["min"] = min(stats["min"], self._block_data[b]["min"])
            stats["max"] = max(stats["max"], self._block_data[b]["max"])

        for r_idx in range(r_loc):
            e = self._blocks[r_block][r_idx]
            stats["sum"] += e
            stats["min"] = min(stats["min"], e)
            stats["max"] = max(stats["max"], e)

        return stats

    def __setitem__(self, idx, val):
        """Update index idx to be val.

        Complexity: $O(\\sqrt n)$.

        """
        self._glob_min = min(self._glob_min, val - 1)
        self._glob_max = max(self._glob_max, val + 1)
        block, loc = self._pos(idx)
        self._blocks[block][loc] = val
        self._update(block)

    def __str__(self):
        """Return a verbose string representation of the SQRT decomposition."""
        retval = "SQRT(\n"
        for idx, elt in enumerate(self._blocks):
            s_i = str(idx).ljust(3)
            s_o = str(self._block_data[idx]).ljust(4 * 12 + 4)
            s_d = "[" + ", ".join(str(x).rjust(5) for x in elt) + "]"
            retval += f"     {s_i} {s_o} {s_d}\n"
        return retval + ")"

    def __repr__(self):
        data = [str(self[i]) for i in range(len(self))]
        return f"SQRT([{', '.join(data)}])"

```

## 06-divide-and-conquer
### Closest Pair
```python
"""A divide and conquer algorithm for computing a closest pair of points in a
set of $n$ points in the plane in time $O(n \\log n)$.  It outputs both their
distance and the pair itself.
"""

import itertools as IT
import math
from collections import namedtuple as T

Point = T("Point", "x y")
Sol = T("Sol", "delta pair")


def bruteforce(points):
    return min(Sol(math.hypot(a.x - b.x, a.y - b.y), (a, b)) for (a, b) in IT.combinations(points, 2))


def closest_pair(X, Y):
    if len(X) <= 5:
        return bruteforce(X)
    pivot = X[len(X) // 2]
    Xl = [p for p in X if p <= pivot]
    Yl = [p for p in Y if p <= pivot]
    Xr = [p for p in X if p > pivot]
    Yr = [p for p in Y if p > pivot]
    OPT = min(closest_pair(Xl, Yl), closest_pair(Xr, Yr))
    S = [p for p in Y if abs(p.x - pivot.x) < OPT.delta]
    if len(S) > 1:
        OPT = min(OPT, min(bruteforce(IT.islice(S[i:], 6)) for i in range(len(S) - 1)))
    return OPT


def closest(points):
    X = sorted(points)
    Y = sorted(points, key=lambda p: (p.y, p.x))
    return closest_pair(X, Y)


# Driver
import random

N = 100
P = [Point(random.random() * 1000, random.random() * 1000) for _ in range(N)]
delta, (p1, p2) = closest(P)
print("delta:", round(delta, 3))

```

### Fast Fourier Transform
```python
"""Fast Fourier Transform using divide and conquer in time $O(n \\log n)$."""

import cmath


def fft(data):
    N = len(data)
    if N <= 1:
        return data
    data0 = data[0::2]  # even data
    data1 = data[1::2]  # odd data
    res0 = fft(data0)
    res1 = fft(data1)
    X = [0 + 0j] * N
    for k in range(N // 2):
        angle = -2 * cmath.pi * k / N
        twiddle = cmath.exp(1j * angle)
        X[k] = res0[k] + twiddle * res1[k]
        X[k + N // 2] = res0[k] - twiddle * res1[k]
    return X


if __name__ == "__main__":
    import math

    N = 2**20
    R = lambda x: (math.sin(x) + math.sin(2 * x) - math.sin(3 * x) + math.sin(x / 5) + math.cos(x)) / (0.1 * (1 + x))
    data = [R(i) for i in range(N)]
    from datetime import datetime as dt

    start = dt.now()
    fft_result = [cround(e) for e in fft(data)]
    t_fft = round((dt.now() - start).total_seconds(), 3)

    print(t_fft, "s")

```

## 07-geometry
### Bentley Ottmann
```python
"""Find all intersection points among a set of line segments using a
simplified Bentley–Ottmann sweep.  Maintains an active list of segments and
schedules intersection checks through an event queue.

"""

import heapq


def intersect(a, b):
    (x1, y1), (x2, y2) = a
    (x3, y3), (x4, y4) = b
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    if (
        min(x1, x2) <= px <= max(x1, x2)
        and min(y1, y2) <= py <= max(y1, y2)
        and min(x3, x4) <= px <= max(x3, x4)
        and min(y3, y4) <= py <= max(y3, y4)
    ):
        return px, py


def bentley_ottmann(segments):
    events = []
    [heapq.heappush(events, (min(p[0], q[0]), p, q)) for p, q in segments]
    res = set()
    active = []
    while events:
        x, p, q = heapq.heappop(events)
        seg = [s for s in segments if s == (p, q) or s == (q, p)][0]
        if seg not in active:
            active.append(seg)
        else:
            active.remove(seg)
        for s in active:
            if s != seg:
                r = intersect(s, seg)
                if r:
                    res.add((round(r[0], 6), round(r[1], 6)))
    return res

```

### Faces
```python
"""Compute the faces of a planar graph."""

import cmath


class Vec:
    def __init__(v, x, y):
        v.c = complex(x, y)

    @property
    def x(v):
        return v.c.real

    @property
    def y(v):
        return v.c.imag

    @staticmethod
    def _from_real(c):
        return Vec(c.real, c.imag)

    def __sub__(v1, v2):
        return v1._from_real(v1.c - v2.c)

    @property
    def length(v):
        return abs(v.c)

    @property
    def theta(v):
        return cmath.phase(v.c)

    def __repr__(v):
        return f"Vec({v.x}, {v.y})"


def is_outer(face):
    global left, right, top, bottom
    for v in left, right, top, bottom:
        if v[0] not in face:
            return False
    return True


def rel_theta(origo):
    return lambda p: (p.point - origo).theta


def sort_neighborhood(v, nv):
    pv = G.vertices[v]
    neighbors = sorted([Vertex(u, G.vertices[u]) for u in nv], key=rel_theta(pv))
    return [u.name for u in neighbors]


def sort_all_neighborhoods(G):
    for v, nv in G.edges.items():
        G.edges[v] = sort_neighborhood(v, nv)


sort_all_neighborhoods(G)


def cycle(s):
    m = min(s)
    i = s.index(m)
    return s[i:] + s[:i]


def get_face(edge, G):
    face = [edge[0]]
    while True:
        u, v = edge
        if v in face:
            break
        face.append(v)
        nb = G.edges[v]
        ui = nb.index(u)
        nxt = nb[(ui + 1) % len(nb)]
        edge = v, nxt
    return face


faces = {}
for v, nv in G.edges.items():
    for u in nv:
        edge = v, u
        faces[edge] = tuple(cycle(get_face(edge)))


def get_polygon(edge, G):
    f = faces[edge]
    poly = []
    for v in f:
        poly.append(G.vertices[v])
    return poly


def face_to_polygon(face, G):
    poly = []
    for v in face:
        poly.append(G.vertices[v])
    return poly


G = None

```

### Graham Scan
```python
"""The Graham scan algorithm for computing the convex hull of a set of $n$
points in the plane.  It runs in time $O(n \\log n)$, which is optimal.
"""

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Point:
    x: float
    y: float
    idx: float

    @property
    def tuple(self):
        return self.x, self.y

    def __lt__(self, other):
        return (self.x, self.y) < (
            other.x,
            other.y,
        )

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, -1)

    @classmethod
    def from_str(c, s):
        return Point(*map(float, s.split()))


def cross(o, a, b):
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def right(a, b, c):
    return cross(a, b, c) <= 0


def graham(points):
    points = sorted(set(points))
    S, hull = [], []  # S is a stack of points

    for p in points:
        while len(S) >= 2 and right(S[-2], S[-1], p):
            S.pop()
        S.append(p)
    hull += S

    S = []
    for p in reversed(points):
        while len(S) >= 2 and right(S[-2], S[-1], p):
            S.pop()
        S.append(p)
    hull += S[1:-1]  # ignore endpoints
    return hull

```

### Jarvis March
```python
"""The Jarvis march (gift wrapping) algorithm for computing the convex hull of
a set of $n$ points in $O(n \\cdot h)$ time where $h$ is the number of points
on the hull.
"""

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Point:
    x: float
    y: float
    idx: float

    @property
    def tuple(self):
        return self.x, self.y

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, -1)


def leftmost(points):
    return sorted(points)[0].idx


def cross(a, b):
    return a.x * b.y - a.y * b.x


def orient(a, b, c):
    res = cross(b - a, c - a)
    if res == 0:
        return 0  # on line
    elif res > 0:
        return 1  # cw
    else:
        return 2  # ccw


def jarvis(points):
    N = len(points)
    l = leftmost(points)
    hull = []
    b = 0
    a = None
    while a != l:
        if a is None:
            a = l  # start
        hull.append(a)
        b = (a + 1) % N
        for i in range(N):
            if orient(points[a], points[i], points[b]) == 2:
                b = i
        a = b
    return [points[i] for i in hull]

```

### Library
```python
"""Some Geometry functions"""

import cmath
import math


class Vec:
    def __init__(self, x=None, y=None, c=None):
        if c is None:
            if x is None or y is None:
                raise ValueError("Either (x or y) or c")
            self.c = complex(x, y)
        else:
            if x is not None or y is not None:
                raise ValueError("When c, not x y")
            self.c = c

    @property
    def x(self):
        return self.c.real

    @property
    def y(self):
        return self.c.imag

    @property
    def phi(self):
        return cmath.phase(self.c)

    @property
    def r(self):
        return abs(self.c)

    def __mul__(self, o):
        if isinstance(o, Vec):
            o = o.c
        return Vec(c=self.c * o)

    def __truediv__(self, s):
        return Vec(self.x / s, self.y / s)

    def __add__(self, o):
        if isinstance(o, Vec):
            o = o.c
        return Vec(c=self.c + o)

    def __sub__(self, o):
        if isinstance(o, Vec):
            o = o.c
        return Vec(c=self.c - o)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"C({self.x}, {self.y})"

    @property
    def conj(self):
        return Vec(c=self.c.conjugate())

    @property
    def rad(self):
        return cmath.phase(self.c)

    @property
    def deg(self):
        return cmath.phase(self.c) * 180 / math.pi

    def translate(v1, v2):
        return v1 + v2

    def scale(v, scalar, c):
        return c + (v - c) * scalar

    def rotate(v, angle):
        return v * cmath.polar(1, angle)

    def perp(v):
        return Vec(-v.y, v.x)

    def dot(v1, v2):
        return (v1.conj * v2).x  # x

    def dot_(v1, v2):
        return (v1.x * v2.x) + (v1.y * v2.y)

    def is_perp(v1, v2):
        return v1.dot(v2) == 0

    def cross(v1, v2):
        return (v1.conj * v2).y  # y

    def cross_(v1, v2):
        return v1.x * v2.y - v1.y * v2.x

    def __hash__(v):
        return hash(v.c)

    def __eq__(v1, v2):
        return v1.c == v2.c

    def __getitem__(v, idx):
        if idx == 0:
            return v.x
        if idx == 1:
            return v.y
        raise IndexError(str(v) + " " + str(idx))


def cis(r, phi):
    return Vec(r * math.cos(phi), (r * math.sin(phi)))


def sgn(a):
    return (Vec(0) < a) - (a < Vec(0))


def orient(a, b, c):
    return (b - a).cross(c - a)


def in_angle(a, b, c, p):
    """p inside angle created from the lines from a to b and a to c."""
    assert orient(a, b, c) != 0
    if orient(a, b, c) < 0:
        b, c = c, b
    return orient(a, b, p) >= 0 and orient(a, c, p) <= 0


def is_convex(polygon):
    has_pos = has_neg = False
    for idx in range(len(polygon)):
        p1, p2, p3 = polygon[idx - 2], polygon[idx - 1], polygon[idx]
        o = orient(p1, p2, p3)
        has_pos |= o > 0
        has_neg |= o < 0

    return not (has_pos and has_neg)

```

### Shoelace Formula
```python
"""The Shoelace algorithm for computing the area of a polygon."""

from collections import namedtuple as T

P = T("P", "x y")


def shoelace(poly):
    fw = sum(poly[i - 1].x * poly[i].y for i in range(len(poly)))
    poly = list(reversed(poly))
    bw = sum(poly[i - 1].x * poly[i].y for i in range(len(poly)))
    return abs(fw - bw) / 2.0


def assert_eq(p, exp):
    act = shoelace(p)
    assert act == exp, str((act, exp, p))
    print(act, "=", exp)


assert_eq([P(1, 1), P(2, 1), P(2, 0), P(1, 0)], 1.0)
assert_eq([P(1, 1), P(2, 2), P(3, 1), P(2, 0)], 2.0)
assert_eq([P(2, 7), P(10, 1), P(8, 6), P(11, 7), P(7, 10)], 32.0)
assert_eq([P(3, 4), P(5, 11), P(12, 8), P(9, 5), P(5, 6)], 30.0)
assert_eq([P(5, 11), P(12, 8), P(9, 5), P(5, 6), P(3, 4)], 30.0)

```

## 08-exptime
### Coloring
```python
"""Graph coloring."""

from collections import namedtuple as T

Graph = T("Graph", "v e")


def greedy_clique(G):
    degdist = sorted([(len(G.e[v]), v) for v in G.v])
    clique = set()
    for deg, v in degdist:
        if not clique:
            clique.add(v)
            continue
        for c in clique:
            if c not in G.e[v]:
                break
        else:
            clique.add(v)
    return clique


def find_best_vertex(G, k, coloring):
    best = None
    used = []
    for v in G.v:
        if v in coloring:
            continue
        c_used = set([coloring[u] for u in G.e[v] if u in coloring])
        if len(c_used) == k:
            return v, c_used  # early abort ; used all colors
        if best is None or len(c_used) > len(used):
            best = v
            used = c_used
    return best, used


def colorable(G, k, coloring):
    if len(coloring) == len(G.v):
        return True
    # find a vertex
    v, used = find_best_vertex(G, k, coloring)
    if len(used) == k:
        return False
    for color in range(k):
        if color in used:
            continue
        coloring[v] = color
        if colorable(G, k, coloring):
            return True
        del coloring[v]
    return False

```

### Permute
```python
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

```

### Travelling Salesman
```python
"""Travelling Salesman dynamic programming routine."""

import itertools as IT
import math
from collections import defaultdict as D
from collections import namedtuple as T

Set = frozenset
Place = T("Place", "name x y")
Graph = T("Graph", "V E")
INFTY = float("inf")
Sol = T("Sol", "cost path")


def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def tsp(G, start, end):
    DP = D(lambda: Sol(INFTY, None))
    DP[(Set([start]), start)] = Sol(0, [start])
    V = Set(G.V) - Set([start, end])

    for k in range(len(V) + 2):
        for S in IT.combinations(G.V, k):
            set_s = Set(S) | Set([start])
            for v in S:
                v_sol = DP[(set_s, v)]  # cost for v, parent for v
                for u in G.V:
                    if u in S:
                        continue
                    set_su = set_s | Set([u])
                    c_sol = DP[(set_su, u)]
                    n_cost = v_sol.cost + dist(v, u)
                    if n_cost < c_sol.cost:
                        DP[(set_su, u)] = Sol(n_cost, v_sol.path + [u])
    return DP[Set(G.V), end]

```

## 09-flow
### Edmonds Karp
```python
"""The Edmonds–Karp variant of the Ford–Fulkerson algorithm for computing
maximum flow.
"""

from collections import namedtuple as T

Graph = T("Graph", "V E c F")


def create_graph(V, E, c):
    n = len(V)
    F = [[0] * n for _ in range(n)]
    for v, u in c:
        F[v][u] = c[(v, u)]
    return Graph(V=V, E=E, c=c, F=F)


def create_path(parent, t):
    path = [t]
    v = t
    while v:
        v = parent[v]
        path.append(v)
    return tuple(reversed(path))


def bfs(graph, s, t):
    q = [s]
    parent = {}
    while q:
        v = q.pop(0)
        for u in graph.V:
            if u in parent:
                continue  # seen it before
            if graph.F[v][u] <= 0:
                continue  # vu saturated
            parent[u] = v
            if u == t:
                return create_path(parent, t)
            q.append(u)


edges = lambda p: zip(p, p[1:])


def maxflow(graph, s, t):
    flow = 0
    while P := bfs(graph, s, t):
        bottleneck = min(graph.F[v][u] for (v, u) in edges(P))
        print(P, bottleneck)
        if bottleneck == 0:
            return flow
        flow += bottleneck
        for i in range(1, len(P)):
            v, u = P[i - 1], P[i]
            graph.F[v][u] -= bottleneck
            graph.F[u][v] += bottleneck
    return flow

```

## 10-strings
### Hashing
```python
"""Rolling hash."""

A = 3
B = 97


def phash(word, A=3, B=97):
    n = len(word)
    return sum(ord(e) * (A ** (n - i - 1)) for i, e in enumerate(word)) % B


def window(text, k, idx, cur_hash, A, B):
    if idx == 0:
        return phash(text[:k], A, B)

    cur_hash -= ord(text[idx - 1]) * (A ** (k - 1))
    cur_hash *= A
    cur_hash += ord(text[idx + k - 1])

    return cur_hash % B


def rolling_hash(text, k):
    A = 3
    B = 97
    h = [0] * len(text)
    for i in range(len(text) - 5):
        h[i] = window(text, k, i, h[i - 1], A, B)
    return h


def find_all(h, the_hash, text, word):
    for i, e in enumerate(h):
        if e == the_hash:
            if text[i : i + len(word)] == word:
                yield i


print("hash(ALLEY) =", phash("ALLEY"))

text = """\
IN THE ALLEY BELOW THE VALLEY \
THERE'S AN ALLY CALLED MALLEY\
"""

k = 5
h = [phash(text[i : i + 5]) for i in range(len(text))]
print("h(text) =", h)
word = "ALLEY"
retval = list(find_all(h, phash(word), text, word))
print(f"find({word}) =", retval)

print(h)
print(rolling_hash(text, k))


print()
print()


tst = 0
word = "VALLEYMALLEYCALLEY"
for i in range(5):
    tst += ord(word[i]) * (A ** (k - i - 1))
    print(i, tst)
    tst %= B
    print(i, tst)
print("exp", phash(word[0:5]), word[0:5])
print("act", tst)

i = 1
print("slide")
tst -= (ord(word[0]) * A ** (k - 1)) % B
tst *= A
tst += ord(word[i + 5 - 1])
tst %= B
print("exp", phash(word[i : i + 5]), word[i : i + 5])
print(tst)

```

### Knuth Morris Pratt
```python
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

```

### Trie
```python
"""A basic implementation of a trie."""


def trie(words):
    root = {}
    for word in words:
        current = root
        for letter in word:
            current = current.setdefault(letter, {})
        current["*"] = "*"
    return root


def find(T, w):
    current = T
    for letter in w:
        if letter not in current:
            return None  # Not found
        current = current[letter]
    return current  # Returns node in tree


def _dfs(T):
    for k in T:
        if k == "*":
            yield ""
        else:
            for v in _dfs(T[k]):
                yield k + v


def complete(T, w):
    C = find(T, w)  # continuations
    if not C:
        return []
    for c in _dfs(C):
        yield w + c


if __name__ == "__main__":
    import sys
    import termios
    import tty

    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    DICT = "/usr/share/dict/words"
    words = []
    with open(DICT, "r") as fin:
        for line in fin:
            w = line.strip().lower()
            if w.isalnum():
                words.append(w)
    T = trie(words)
    w = ""
    inp = ""
    entered = []
    print("Enter word: ")
    while True:
        inp = getch()
        if inp == "Q":
            break
        elif ord(inp) == 13:
            entered.append(w)
            w = ""
            inp = ""
        elif ord(inp) == 127:
            inp = ""
            if w:
                w = w[: len(w) - 1]
        w = w + inp
        if w:
            for idx, x in enumerate(list(complete(T, w))):
                print(1 + idx, x)
                if idx >= 29:
                    break
        print("=" * 20)
        print(f'"{w}" (Type Q to exit, backspace to erase, enter to accept)')
    print()
    for w in entered:
        print(w)

```

## 11-math
### Gcd
```python
"""Extended Euclidean algorithm: compute the coefficients of Bézout's identity, i.e., $ax + by = \\gcd(a, b)$."""


def gcdx(a, b):
    if a == 0:
        return b, 0, 1
    g, x1, y1 = gcdx(b % a, a)
    x = y1 - (b // a) * x1
    y = x1

    return g, x, y


A = 450
B = 1080
g, x, y = gcdx(450, 1080)
if y >= 0:
    print(f"{A}·{x} + {B}·{y} = gcd({A}, {B}) = {g}")
else:
    print(f"{A}·{x} - {B}·{-y} = gcd({A}, {B}) = {g}")

```

### Primes
```python
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

```

### Sieve
```python
"""The Sieve of Eratosthenes — an ancient algorithm for finding all prime numbers up to any given limit."""


def sieve(n):
    if n < 2:
        return
    prime = [True] * (n + 1)
    prime[0] = False
    prime[1] = False
    for i in range(2, n + 1):
        if not prime[i]:
            continue
        yield i
        for j in range(i + i, n, i):
            prime[j] = False


print(*sieve(101))

```
