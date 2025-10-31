# Algorithm engineering

A repository of Python snippets for different algorithms.


## Graphs

* [Dfs Tree](02-graphs/dfs_tree.py)
  - Run DFS and get a DFS tree
* [Dijkstra](02-graphs/dijkstra.py)
  - Simple Dijkstra implementation running in time $O(m \log n)$.
* [Kahns Algorithm](02-graphs/kahns_algorithm.py)
  - Kahn's algorithm produces a topological order of a DAG by repeatedly
    removing vertices with zero in-degree and updating their neighbors' in-degrees
    until none remain.
* [Topological Sort With Dfs](02-graphs/topological_sort_with_dfs.py)
  - Use postorder DFS to get a reversed topological ordering.
* [Union Find](02-graphs/union_find.py)
  - Simple Union Find (with path compression) datastructure implementation for
    use in MST.

## Sliding searching sorting

* [Consecutive Sum](03-sliding-searching-sorting/consecutive_sum.py)
  - Maximum consecutive sum in an array with possibly negative values.
* [Prefix Sum](03-sliding-searching-sorting/prefix_sum.py)
  - Compute the prefix sum of a given list as input.
* [Sliding Window](03-sliding-searching-sorting/sliding_window.py)
  - Sliding window to find best $k$ sized window in linear time.

## Dynamic programming

* [Dominating Set On Trees](04-dynamic-programming/dominating_set_on_trees.py)
  - Weighted Dominating Set on Trees.  Dynamic Programming.
* [Edit Distance](04-dynamic-programming/edit_distance.py)
  - Compute the edit distance (Levenshtein distance) with dynamic programming.
    With applications in spell checking.
* [Feedback Arc Set](04-dynamic-programming/feedback_arc_set.py)
  - Exact $2^n$ algorithm for Feedback Arc Set (with memoization).
* [Independent Set On Trees](04-dynamic-programming/independent_set_on_trees.py)
  - Weighted Independent Set on trees.
* [Knapsack 01](04-dynamic-programming/knapsack_01.py)
  - 0-1-Knapsack dynamic programming.
* [Knapsack Wr](04-dynamic-programming/knapsack_wr.py)
  - Knapsack with repetitions.  Dynamic Programming.
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
    answer range queries such as sum, max, min in time O(√n) and can do update
    element in O(√n) as well.
    
    In theory, this data structure can take any [associative
    operation](https://en.wikipedia.org/wiki/Associative_property) (e.g., gcd,
    lcm), and the elements can even be higher order structures such as matrices
    with the operation being matrix multiplication.  Indeed, you can take elements
    over $\text{GL}_n(\mathbb{C})$, and answer such queries.

## Geometry

* [Closest Pair](06-geometry/closest_pair.py)
  - A divide and conquer algorithm for computing a closest pair of points in a
    set of $n$ points in the plane in time $O(n \log n)$.  It outputs both their
    distance and the pair itself.
* [Faces](06-geometry/faces.py)
  - Compute the faces of a planar graph.
* [Graham Scan](06-geometry/graham_scan.py)
  - The Graham scan algorithm for computing the convex hull of a set of $n$
    points in the plane.  It runs in time $O(n \log n)$, which is optimal.
* [Jarvis March](06-geometry/jarvis_march.py)
  - The Jarvis march (gift wrapping) algorithm for computing the convex hull of
    a set of $n$ points in $O(n \cdot h)$ time where $h$ is the number of points
    on the hull.
* [Library](06-geometry/library.py)
  - Some Geometry functions
* [Shoelace Formula](06-geometry/shoelace_formula.py)
  - The Shoelace algorithm for computing the area of a polygon.

## Exptime

* [Coloring](07-exptime/coloring.py)
  - Graph coloring.
* [Permute](07-exptime/permute.py)
  - Enumerate all permutations of a list
* [Travelling Salesman](07-exptime/travelling_salesman.py)
  - Travelling Salesman dynamic programming routine.

## Flow

* [Edmonds Karp](08-flow/edmonds_karp.py)
  - The Edmonds–Karp variant of the Ford–Fulkerson algorithm for computing
    maximum flow.

## Strings

* [Hashing](09-strings/hashing.py)
  - Rolling hash.
* [Knuth Morris Pratt](09-strings/knuth_morris_pratt.py)
  - Knuth–Morris–Pratt (KMP) algorithm for pattern matching.
* [Trie](09-strings/trie.py)
  - A basic implementation of a trie.

## Math

* [Gcd](10-math/gcd.py)
  - Extended Euclidean algorithm: compute the coefficients of Bézout's identity, i.e., $ax + by = \gcd(a, b)$.
* [Primes](10-math/primes.py)
  - Factorize a number into prime factors.
* [Sieve](10-math/sieve.py)
  - The Sieve of Eratosthenes — an ancient algorithm for finding all prime numbers up to any given limit.
