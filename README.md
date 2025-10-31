# Algorithm engineering

A repository of Python snippets for different algorithms that I have
been teaching in the courses
[INF234 --- Algorithms](https://www4.uib.no/en/studies/courses/inf234)
and
[INF237 --- Algorithm Engineering](https://www4.uib.no/en/studies/courses/inf237).

See the [appendix for source code](APPENDIX.md) for all of the algorithms.


## Greedy

* [Huffman](01-greedy/huffman.py)
  - Compute the Huffman encoding of a provided text as input.
    Outputs the encoding for each symbol.
* [Interval-partitioning](01-greedy/interval-partitioning.py)
  - Greedy interval partitioning (coloring an interval graph).
    
    Given a set of intervals with start time s and finish time f, assign each
    interval to a "color" (or resource) so that no two overlapping intervals
    share the same color.  Equivalently, partition the intervals into the minimum
    number of non-overlapping subsets.
    
    The algorithm processes all start/end events in increasing time order.  When
    an interval ends, its color becomes available again; when an interval starts,
    it is assigned any available color.  This is optimal and uses exactly the
    maximum number of simultaneously active intervals.
    
    Runs in O(n log n) time due to sorting of event times.
    Returns a mapping: color -> list of intervals.
* [Stable Matching](01-greedy/stable_matching.py)
  - The Gale–Shapley algorithm for Stable Matching.  The algorithm runs
    in linear time in the total size of preferences, or, $O(n^2)$ where $n$
    is the number of "hospitals".

## Graphs

* [Astar](02-graphs/astar.py)
  - The A* algorithm for shortest-path search with a heuristic.
* [Bfs](02-graphs/bfs.py)
  - Breadth-first search (BFS) for shortest paths in an unweighted graph.
    
    BFS explores vertices in layers of increasing distance from a source vertex.
    It computes shortest-path distances measured in number of edges, and the
    parent pointers define a BFS tree that can be used to reconstruct paths.
    
    Runs in O(n + m) time.
* [Bipartite Matching](02-graphs/bipartite_matching.py)
  - Compute a maximum matching in an unweighted bipartite graph using DFS to
    find augmenting paths.  Implements the classic alternating-path approach for
    bipartite matching.
* [Connected Components](02-graphs/connected_components.py)
  - Connected components via BFS.
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
* [Kruskals Algorithm](02-graphs/kruskals_algorithm.py)
  - Kruskal's algorithm for minimum spanning tree (MST).
    
    Given a weighted undirected graph, Kruskal's algorithm computes a minimum
    spanning tree by considering the edges in nondecreasing order of weight and
    adding an edge whenever it connects two previously disconnected components.
    
    The graph is given as a list of edges of the form $(w, u, v)$, where $w$ is the
    weight of the edge $uv$.  Sorting this list yields the order in which edges are
    considered.  A Union Find data structure maintains the connected components.
    
    Returns the edges of the MST together with its total weight.
    
    Runs in $O(m \log m)$ time, dominated by sorting the m edges.
* [Layered-graph](02-graphs/layered-graph.py)
  - Layered Graph algorithm for search with $k$ free edges.
    
    Suppose you have a directed graph with edge weights and we want to find a
    shortest path from $s$ to $t$.  Suppose furthermore that we can use $k$ edges
    for free.
    
    The Layered Graph algorithm takes $k+1$ copies of $G$, $G_0, G_1, ..., G_k$,
    and creates additional edges from $u_i$ to $v_{i+1}$ with weight 0, for all
    edges $uv$ in $G$.
    
    A path from $s_0$ to some $t_i$ corresponds to an $s$-$t$ path in the original
    graph using exactly $i$ free edges.  Thus, the shortest path from $s_0$ to any
    layered copy of $t$ gives the optimal path using at most $k$ free edges.
* [Maximum Spacing Clustering](02-graphs/maximum_spacing_clustering.py)
  - Maximum-spacing k-clustering via Kruskal's algorithm.
    
    Given an undirected weighted graph on vertices {0, ..., n-1}, the goal of
    maximum-spacing k-clustering is to partition the vertices into exactly k
    clusters so that the minimum-weight edge joining two different clusters is as
    large as possible.
    
    Runs in $O(m \log m)$ time, dominated by sorting the edges.
* [Prims Algorithm](02-graphs/prims_algorithm.py)
  - Prim's algorithm for minimum spanning tree.
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
* [Dreyfus-wagner](04-dynamic-programming/dreyfus-wagner.py)
  - Steiner Tree is the problem of finding a minimum-weight connected subgraph
    that spans a given set of terminal vertices in a weighted graph.
    
    Dreyfus--Wagner is a classic dynamic programming algorithm for Steiner Tree
    parameterized by the number of terminals.
    
    By storing `dp[mask][v]` as the minimum cost of a Steiner tree connecting the
    terminals in `mask` and ending at vertex `v`, it combines smaller terminal
    subsets and then propagates costs through shortest-path distances.
    
    Using a precomputed all-pairs shortest-path matrix `dist`, it runs in $O(3^k n
    + 2^k n^2)$ time, where $k$ is the number of terminals and $n$ is the number of
    vertices.
* [Edit Distance](04-dynamic-programming/edit_distance.py)
  - Compute the edit distance (Levenshtein distance) with dynamic programming.
    With applications in spell checking.
* [Fast Zeta Transform](04-dynamic-programming/fast_zeta_transform.py)
  - Zeta $\zeta$ transform over subsets (submasks).
    
    The subset zeta transform (a.k.a. SOS DP) is: given an array $f$ indexed by
    bitmasks in $\{0 \dots 2^{n-1}\}$, the subset zeta transform produces $f'$
    where:
    
    $$f'[mask] = \sum_{\text{sub} \subseteq \text{mask}} f[\text{sub}].$$
    
    This is useful when you want, for every mask, the sum/aggregation over all its
    submasks, but you want all answers at once in $O(n \cdot 2^n)$ rather than
    doing an $O(3^n)$ total amount of submask iteration.
* [Feedback Arc Set](04-dynamic-programming/feedback_arc_set.py)
  - Exact dynamic programming for Feedback Arc Set in a directed graph.
    
    Given a directed graph, the \textsc{Feedback Arc Set} problem asks for an
    ordering of the vertices minimizing the number of backward edges, that is,
    edges $uv$ with $u$ appearing after $v$ in the ordering.  Equivalently, one
    seeks a minimum set of edges whose removal makes the graph acyclic.
    
    This implementation uses subset DP.  Let $\text{dp}(S, v)$ be the minimum
    number of backward edges in an ordering of the vertex set $S$ that ends with
    $v$.  If $v$ is placed last, then every edge from $v$ to a vertex still in $S$
    contributes one backward edge, and the remaining vertices are solved
    recursively.
    
    Runs in $O(n^2 2^n)$ time.
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
* [Zeta Mobius Transform](04-dynamic-programming/zeta_mobius_transform.py)
  - Zeta and Möbius transforms on the subset lattice.
    
    The zeta transform, also known as Yates DP, takes a function $f$ on subsets and
    computes
    
    $$F[S] = \sum_{T \subseteq S} f[T].$$
    
    The Möbius transform inverts this relation:
    
    $$f[S] = \sum_{T \subseteq S} (-1)^{|S \setminus T|} F[T].$$
    
    This is the algebraic form of inclusion--exclusion, so inclusion--exclusion is
    precisely Möbius inversion on the subset lattice.
    
    These transforms are basic tools in subset DP.  They are also used in fast
    subset-sum convolution: transform two functions, combine them appropriately,
    and invert to recover the convolution.

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
* [Counting Inversions](06-divide-and-conquer/counting_inversions.py)
  - Counting inversions via merge sort (divide and conquer).
    
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
* [Fast Fourier Transform](06-divide-and-conquer/fast_fourier_transform.py)
  - Fast Fourier Transform.
    
    Compute FFT using divide and conquer in time $O(n \log n)$.
* [Karatsuba](06-divide-and-conquer/karatsuba.py)
  - Karatsuba multiplication for large integers represented as strings.
    
    This implementation multiplies two nonnegative integers given as decimal
    strings using a divide-and-conquer strategy.  The inputs are padded to equal
    lengths that are powers of two, then recursively split into high and low
    halves:
    
    $$x = x1 \cdot 10^{n/2} + x0$$
    
    $$y = y1 \cdot 10^{n/2} + y0$$
    
    The product is computed using three recursive multiplications:
    
    $$A = x1 \cdot y1$$
    
    $$B = x0 \cdot y0$$
    
    $$P = (x1 + x0)(y1 + y0)$$
    
    and combined via
    
    $$x \cdot y = A \cdot 10^n + (P - A - B) \cdot 10^{n/2} + B$$
    
    This reduces the number of recursive multiplications from four to three,
    yielding a time complexity of
    
    $$O(n^{\log_2 3}) \approx O(n^{1.585}),$$
    
    which improves over the quadratic elementary school method.
    
    The implementation uses string arithmetic with Python integers for base cases
    and addition/subtraction, and handles arbitrary-length inputs.

## Geometry

* [Bentley Ottmann](07-geometry/bentley_ottmann.py)
  - Bentley--Ottmann sweep to find all intersection points.
    
    Find all intersection points among a set of line segments using a simplified
    version of the Bentley--Ottmann sweep.  Maintains an active list of segments
    and schedules intersection checks through an event queue.
* [Faces](07-geometry/faces.py)
  - Compute the faces of a planar graph.
* [Graham Scan](07-geometry/graham_scan.py)
  - The Graham scan algorithm for computing the convex hull.
    
    Computes the convex hull of a set of $n$ points in the plane.  It runs in time
    $O(n \log n)$, which is optimal.
* [Jarvis March](07-geometry/jarvis_march.py)
  - The Jarvis march (gift wrapping) algorithm.
    
    Computing the convex hull of a set of $n$ points in $O(n \cdot h)$ time where
    $h$ is the number of points on the hull.
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

* [Dinic](09-flow/dinic.py)
  - Dinic's algorithm for maximum flow using BFS levels and DFS blocking flows.
* [Edmonds Karp](09-flow/edmonds_karp.py)
  - The Edmonds–Karp variant of the Ford–Fulkerson algorithm for computing
    maximum flow.
* [Hopcroft Karp](09-flow/hopcroft_karp.py)
  - Hopcroft-Karp algorithm for maximum matching in a bipartite graph.
* [Hungarian](09-flow/hungarian.py)
  - Hungarian algorithm for the assignment problem.
    
    Given an $n \times n$ cost matrix cost, the assignment problem asks for a
    bijection between rows and columns minimizing the total cost
    
        sum(cost[i][ass[i]] for i in range(n)).
    
    Equivalently, this is a minimum-cost perfect matching problem in a complete
    bipartite graph with rows on one side, columns on the other, and edge weight
    `cost[i][j]`.
    
    The Hungarian algorithm maintains dual potentials on rows and columns and
    repeatedly augments a partial matching along zero reduced-cost edges.  In the
    implementation below, this appears as the classical $O(n^3)$ shortest
    augmenting-path formulation.
    
    Runs in $O(n^3)$ time and $O(n^2)$ space for the input matrix.
* [Mincost Maxflow](09-flow/mincost_maxflow.py)
  - Successive shortest augmenting path algorithm for Min Cost Max Flow.

## Strings

* [Hashing](10-strings/hashing.py)
  - Rolling hash.
* [Knuth Morris Pratt](10-strings/knuth_morris_pratt.py)
  - Knuth–Morris–Pratt (KMP) algorithm for pattern matching.
* [Trie](10-strings/trie.py)
  - A basic implementation of a trie.
* [Z-algorithm](10-strings/z-algorithm.py)
  - Z algorithm for linear-time prefix matching.
    
    For each position $i$ in a string $s$, let $Z[i]$ be the length of the longest
    prefix of $s$ matching the substring starting at $i$.  The algorithm maintains
    a rightmost matching interval $[l, r)$ and reuses it to avoid redundant
    comparisons.
    
    Runs in $O(n)$ time.

## Math

* [Gcd](11-math/gcd.py)
  - Extended Euclidean algorithm.
    
    Compute the coefficients of Bézout's identity, i.e., $ax + by = \gcd(a, b)$.
* [Modular-exponentiation](11-math/modular-exponentiation.py)
  - Modular exponentiation by repeated squaring.
    
    Compute $a^b \mod m$ by scanning the bits of $b$ from low to high.  Repeatedly
    square the base, and whenever the current bit is set, multiply it into the
    answer modulo $m$.
    
    Runs in $O(\log b)$ time.
* [Primes](11-math/primes.py)
  - Factorize a number into prime factors.
* [Sieve](11-math/sieve.py)
  - The Sieve of Eratosthenes --- an ancient algorithm for finding all prime numbers up to any given limit.
* [Subset-xor](11-math/subset-xor.py)
  - Subset XOR via Gaussian elimination over $\mathbb{F}_2$.
    
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
