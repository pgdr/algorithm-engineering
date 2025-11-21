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
