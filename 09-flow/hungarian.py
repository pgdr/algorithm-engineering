r"""Hungarian algorithm for the assignment problem.

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

"""


def check_square(cost):
    n = len(cost)
    if any(len(row) != n for row in cost):
        raise ValueError("cost matrix must be square")
    return n


def assignment_cost(cost, ass):
    return sum(cost[i][ass[i]] for i in range(len(cost)))


def hungarian(cost):
    n = check_square(cost)
    if n == 0:
        return 0, []

    # 1-indexed internally, following the standard compact formulation.
    u = [0] * (n + 1)  # row potentials
    v = [0] * (n + 1)  # column potentials
    p = [0] * (n + 1)  # matched row for each column
    way = [0] * (n + 1)  # predecessor columns for augmentation

    for i in range(1, n + 1):
        p[0] = i
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        j0 = 0

        while True:
            used[j0] = True
            i0 = p[j0]
            bottleneck = float("inf")
            j1 = 0

            for j in range(1, n + 1):
                if used[j]:
                    continue
                this = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if this < minv[j]:
                    minv[j] = this
                    way[j] = j0
                if minv[j] < bottleneck:
                    bottleneck = minv[j]
                    j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += bottleneck
                    v[j] -= bottleneck
                else:
                    minv[j] -= bottleneck

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    ass = [-1] * n
    for j in range(1, n + 1):
        ass[p[j] - 1] = j - 1

    value = assignment_cost(cost, ass)
    return value, ass
