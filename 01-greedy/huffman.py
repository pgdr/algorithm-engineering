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
