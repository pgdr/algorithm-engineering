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
