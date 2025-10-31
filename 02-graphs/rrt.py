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
