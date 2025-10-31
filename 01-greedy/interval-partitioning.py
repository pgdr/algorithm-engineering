"""Greedy interval partitioning (coloring an interval graph).

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
"""

from random import randint as R
from collections import namedtuple as T
from collections import defaultdict as D

Interval = T("Interval", "s f")


def rand_interval():
    a = R(1, 50)
    b = R(3, 30)
    return Interval(a, a + b)


def length(iv):
    return iv.f - iv.s


def schedule_all(intervals):
    start = D(list)
    end = D(list)
    timesteps = []
    for iv in intervals:
        start[iv.s].append(iv)
        end[iv.f].append(iv)
        timesteps += [iv.s, iv.f]
    timesteps = sorted(set(timesteps))  # makes unique
    colors = list(reversed(range(len(intervals))))
    coloring = D(list)
    interval_color = {}
    for t in timesteps:
        for iv in end[t]:
            color = interval_color[iv]
            colors.append(color)
        for iv in start[t]:
            color = colors.pop()
            coloring[color].append(iv)
            interval_color[iv] = color
    return coloring


I = [rand_interval() for _ in range(20)]
print(I)
coloring = schedule_all(I)
for x in sorted(coloring):
    ivs = sorted(coloring[x])
    s = [" " * ivs[0].s]
    for i in range(len(ivs) - 1):
        s += "(" + "-" * (length(ivs[i]) - 2) + ")"
        s += " " * (ivs[i + 1].s - ivs[i].f)
    s += "(" + "-" * (length(ivs[-1]) - 2) + ")"
    print(x, " ".join(s))
