# Algorithm engineering

A repository of Python snippets for different algorithms.

See the [appendix for source code](APPENDIX.md) for all of the algorithms.


## Greedy

* [Huffman](01-greedy/huffman.py)
  - Compute the Huffman encoding of a provided text as input.
    Outputs the encoding for each symbol.
* [Stable Matching](01-greedy/stable_matching.py)
  - The Gale–Shapley algorithm for Stable Matching.  The algorithm runs
    in linear time in the total size of preferences, or, $O(n^2)$ where $n$
    is the number of "hospitals".

## Graphs

* [Bipartite Matching](02-graphs/bipartite_matching.py)
  - Compute a maximum matching in an unweighted bipartite graph using DFS to
    find augmenting paths.  Implements the classic alternating-path approach for
    bipartite matching.
* [Dfs Tree](02-graphs/dfs_tree.py)
  - Run DFS and get a DFS tree
* [Dijkstra](02-graphs/dijkstra.py)
  - Simple Dijkstra implementation running in time $O(m \log n)$.
* [Kahns Algorithm](02-graphs/kahns_algorithm.py)
  - Kahn's algorithm produces a topological order of a DAG by repeatedly
    removing vertices with zero in-degree and updating their neighbors' in-degrees
    until none remain.
* [Kosarajus Algorithm](02-graphs/kosarajus_algorithm.py)
  - A strongly connected component (SCC) is a maximal set of vertices where
    every vertex can reach every other by directed paths.  Kosaraju's algorithm
    finds all SCCs by doing a DFS to record finish times, reversing all edges, then
    running DFS again in reverse finish order.
* [Rrt](02-graphs/rrt.py)
  - Grow a rapidly-exploring random tree (RRT) toward a target while avoiding
    obstacles.  Expands the tree incrementally by sampling feasible edges until the
    goal region is reached or iterations are exhausted.
* [Topological Sort With Dfs](02-graphs/topological_sort_with_dfs.py)
  - Use postorder DFS to get a reversed topological ordering.
* [Union Find](02-graphs/union_find.py)
  - Simple Union Find (with path compression) datastructure implementation for
    use in MST.

## Sliding searching sorting

* [Consecutive Sum](03-sliding-searching-sorting/consecutive_sum.py)
  - Maximum consecutive sum in an array with possibly negative values.
* [Longest Nonnegative](03-sliding-searching-sorting/longest_nonnegative.py)
  - Find the length of the longest contiguous subarray with a non-negative sum.
    Uses prefix sums and two-pointer scanning over forward minima and reverse
    maxima.
* [Monotonic Queue](03-sliding-searching-sorting/monotonic_queue.py)
  - Monotonic queue for $O(1)$-amortized sliding-window min/max queries.
    
    Maintains a deque of candidate indices in monotone order as the window moves.
    Use it to compute all window minima/maxima in overall $O(n)$ time.
* [Prefix Sum](03-sliding-searching-sorting/prefix_sum.py)
  - Compute the prefix sum of a given list as input.
* [Sliding Window](03-sliding-searching-sorting/sliding_window.py)
  - Sliding window to find best $k$ sized window in linear time.

## Dynamic programming

* [Bellman Ford](04-dynamic-programming/bellman_ford.py)
  - Single-source shortest paths (SSSP) with negative edges can be solved by
    Bellman–Ford, which uses dynamic programming over path lengths to relax edges
    up to $n-1$ times.  If a further relaxation still decreases a distance, it
    indicates a negative-weight cycle reachable from the source.
* [Dominating Set On Trees](04-dynamic-programming/dominating_set_on_trees.py)
  - Weighted Dominating Set on Trees.  Dynamic Programming.
* [Edit Distance](04-dynamic-programming/edit_distance.py)
  - Compute the edit distance (Levenshtein distance) with dynamic programming.
    With applications in spell checking.
* [Fast Zeta Transform](04-dynamic-programming/fast_zeta_transform.py)
  - SOS DP / Fast Zeta Transform over subsets (submasks).
    
    This module provides the classic *subset zeta transform* (a.k.a. SOS DP step).
    
    Given an array f indexed by bitmasks in {0..2^n-1}, the subset zeta transform
    produces f' where:
    
        f'[mask] = sum_{sub ⊆ mask} f[sub].
    
    This is useful when you want, for every mask, the sum/aggregation over all its
    submasks, but you want all answers at once in O(n·2^n) rather than doing an
    O(3^n) total amount of submask iteration.
* [Feedback Arc Set](04-dynamic-programming/feedback_arc_set.py)
  - Exact $2^n$ algorithm for Feedback Arc Set (with memoization).
* [Floyd-warshall](04-dynamic-programming/floyd-warshall.py)
  - APSP (All-Pairs Shortest Paths) is the problem of finding the shortest path
    distances between every pair of vertices in a weighted graph.
    
    Floyd–Warshall is a classic dynamic programming algorithm for APSP.
    
    By updating `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])` for each
    intermediate vertex `k`, it computes shortest paths for all pairs in $O(n^3)$
    time.
* [Independent Set On Trees](04-dynamic-programming/independent_set_on_trees.py)
  - Weighted Independent Set on trees.
* [Knapsack 01](04-dynamic-programming/knapsack_01.py)
  - 0-1-Knapsack dynamic programming.
* [Knapsack Wr](04-dynamic-programming/knapsack_wr.py)
  - Knapsack with repetitions.  Dynamic Programming.
* [Line Breaking](04-dynamic-programming/line_breaking.py)
  - Use DP to minimize the raggedness of a paragraph of text.
* [Linear Regression Queries](04-dynamic-programming/linear_regression_queries.py)
  - Given a set of $n$ $(x, y)$ points, compute the least squares for the data
    set.  In addition, create a data structure that can answer query
    `least_squares(i, j)` of all points in `data[i:j]`.  Using dynamic programming
    we can compute the entire data structure in time $O(n^2)$.
* [Longest Path Dag](04-dynamic-programming/longest_path_dag.py)
  - Longest Path in a DAG.  Dynamic programming.
* [Subset Sum](04-dynamic-programming/subset_sum.py)
  - Dynamic programming routine (recursion with memoization) for Subset Sum.

## Segment trees

* [Segment Tree](05-segment-trees/segment_tree.py)
  - Basic implementation of a segment tree.
* [Sqrt Decomposition](05-segment-trees/sqrt_decomposition.py)
  - Basic implementation for SQRT decomposition.  The SQRT data structure can
    answer range queries such as sum, max, min in time $O(\sqrt n)$ and can do update
    element in $O(\sqrt n)$ as well.
    
    In theory, this data structure can take any [associative
    operation](https://en.wikipedia.org/wiki/Associative_property) (e.g., gcd,
    lcm), and the elements can even be higher order structures such as matrices
    with the operation being matrix multiplication.  Indeed, you can take elements
    over $\text{GL}_n(\mathbb{C})$, and answer such queries.

## Divide and conquer

* [Closest Pair](06-divide-and-conquer/closest_pair.py)
  - A divide and conquer algorithm for computing a closest pair of points in a
    set of $n$ points in the plane in time $O(n \log n)$.  It outputs both their
    distance and the pair itself.
* [Fast Fourier Transform](06-divide-and-conquer/fast_fourier_transform.py)
  - Fast Fourier Transform using divide and conquer in time $O(n \log n)$.

## Geometry

* [Bentley Ottmann](07-geometry/bentley_ottmann.py)
  - Find all intersection points among a set of line segments using a
    simplified Bentley–Ottmann sweep.  Maintains an active list of segments and
    schedules intersection checks through an event queue.
* [Faces](07-geometry/faces.py)
  - Compute the faces of a planar graph.
* [Graham Scan](07-geometry/graham_scan.py)
  - The Graham scan algorithm for computing the convex hull of a set of $n$
    points in the plane.  It runs in time $O(n \log n)$, which is optimal.
* [Jarvis March](07-geometry/jarvis_march.py)
  - The Jarvis march (gift wrapping) algorithm for computing the convex hull of
    a set of $n$ points in $O(n \cdot h)$ time where $h$ is the number of points
    on the hull.
* [Library](07-geometry/library.py)
  - Some Geometry functions
* [Shoelace Formula](07-geometry/shoelace_formula.py)
  - The Shoelace algorithm for computing the area of a polygon.

## Exptime

* [Coloring](08-exptime/coloring.py)
  - Graph coloring.
* [Permute](08-exptime/permute.py)
  - Enumerate all permutations of a list
* [Travelling Salesman](08-exptime/travelling_salesman.py)
  - Travelling Salesman dynamic programming routine.

## Flow

* [Edmonds Karp](09-flow/edmonds_karp.py)
  - The Edmonds–Karp variant of the Ford–Fulkerson algorithm for computing
    maximum flow.

## Strings

* [Hashing](10-strings/hashing.py)
  - Rolling hash.
* [Knuth Morris Pratt](10-strings/knuth_morris_pratt.py)
  - Knuth–Morris–Pratt (KMP) algorithm for pattern matching.
* [Trie](10-strings/trie.py)
  - A basic implementation of a trie.

## Math

* [Gcd](11-math/gcd.py)
  - Extended Euclidean algorithm: compute the coefficients of Bézout's identity, i.e., $ax + by = \gcd(a, b)$.
* [Primes](11-math/primes.py)
  - Factorize a number into prime factors.
* [Sieve](11-math/sieve.py)
  - The Sieve of Eratosthenes — an ancient algorithm for finding all prime numbers up to any given limit.
