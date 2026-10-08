"""Stage 1b: turn 1-pixel line masks into ordered pen strokes.

A plotter wants polylines, and it wants as few pen lifts as possible. So:
trace the skeleton into paths, simplify them, stitch paths whose ends meet,
then order them to keep pen-up travel short.
"""

import numpy as np
import cv2

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_paths(mask):
    """Walk a 1px skeleton into a list of (N, 2) arrays of (x, y) points."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    on = set(zip(ys.tolist(), xs.tolist()))

    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy, dx in NEIGHBOURS if (y + dy, x + dx) in on]

    degree = {p: len(nbrs(p)) for p in on}
    used = set()  # undirected pixel-pair edges already drawn

    def walk(start, nxt):
        path = [start, nxt]
        used.add(frozenset((start, nxt)))
        prev, cur = start, nxt
        while degree[cur] == 2:
            step = [n for n in nbrs(cur) if n != prev and frozenset((cur, n)) not in used]
            if not step:
                break
            prev, cur = cur, step[0]
            used.add(frozenset((prev, cur)))
            path.append(cur)
        return path

    paths = []
    # Open paths: start from endpoints and junctions.
    for p in sorted(on):
        if degree[p] != 2:
            for n in nbrs(p):
                if frozenset((p, n)) not in used:
                    paths.append(walk(p, n))
    # Whatever is left are closed loops.
    for p in sorted(on):
        for n in nbrs(p):
            if frozenset((p, n)) not in used:
                paths.append(walk(p, n))

    return [np.array([(x, y) for y, x in pth], dtype=np.float32) for pth in paths]


def path_length(pts):
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def simplify(pts, eps):
    if len(pts) < 3 or eps <= 0:
        return pts
    return cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, False).reshape(-1, 2)


def stitch(paths, gap, reach=8, max_turn=40.0, bridge=0.0):
    """Join paths end-to-end where their ends lie within `gap` px and continue
    each other's direction (good continuation). Skeleton tracing breaks a line
    at every junction, so an arch crossed by fifteen bars arrives as sixteen
    pieces; this puts it back together.

    All candidate joins are ranked and taken best-first, so at a junction the
    straightest continuation wins rather than whichever piece came first.
    Directions are measured `reach` points back from the end, because the
    skeleton bends right at a junction.

    Ends further apart than `gap` but within `bridge` are joined only when
    they line up: each lies on the other's line of travel. That closes the
    gaps left where a crossing bar interrupted an edge."""
    paths = [p for p in paths if len(p) >= 2]
    n = len(paths)
    if n < 2:
        return paths

    def end_dir(p, e):
        k = min(reach, len(p) - 1)
        d = (p[-1] - p[-1 - k]) if e else (p[0] - p[k])
        return d / (np.linalg.norm(d) + 1e-9)

    pts = np.array([[p[0], p[-1]] for p in paths], float).reshape(-1, 2)
    dirs = np.array([[end_dir(p, 0), end_dir(p, 1)] for p in paths]).reshape(-1, 2)
    cands = []
    lim = -np.cos(np.radians(max_turn))
    for k in range(2 * n):
        dist = np.linalg.norm(pts - pts[k], axis=1)
        for m in np.flatnonzero(dist <= max(gap, bridge)):
            if m <= k or m // 2 == k // 2:
                continue
            dot = dirs[k] @ dirs[m]
            if dot >= lim:  # outward directions must oppose: one continues the other
                continue
            if dist[m] > gap:
                v = pts[m] - pts[k]
                ahead = v @ dirs[k]
                perp = lambda d: np.array([-d[1], d[0]])
                side = max(abs(v @ perp(dirs[k])), abs(v @ perp(dirs[m])))
                if ahead <= 0 or side > max(2.0, 0.2 * dist[m]) or dot > -np.cos(np.radians(20)):
                    continue
            cands.append((dist[m] / max(gap, bridge) + (1 + dot) * 4, k, m))
    cands.sort()

    link = {}                       # end id -> end id it joins
    parent = list(range(n))         # no cycles: a closed loop stays open

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for _, k, m in cands:
        if k in link or m in link or find(k // 2) == find(m // 2):
            continue
        link[k], link[m] = m, k
        parent[find(k // 2)] = find(m // 2)

    # Walk each chain from a free end.
    out, used = [], set()
    for start in range(n):
        if start in used:
            continue
        # Find a chain end: move along links until an end with no partner.
        i, e = start, 0
        seen = set()
        while 2 * i + e in link and i not in seen:
            seen.add(i)
            j = link[2 * i + e]
            i, e = j // 2, 1 - j % 2
        # (i, e) is a free end; build the chain from it.
        chain = []
        while True:
            used.add(i)
            p = paths[i] if e == 0 else paths[i][::-1]
            chain.append(p)
            far = 2 * i + (1 - e)
            if far not in link:
                break
            j = link[far]
            i, e = j // 2, j % 2
        out.append(np.vstack(chain))
    return out


def order(paths):
    """Greedy nearest-neighbour ordering, flipping paths when the far end is closer."""
    remaining = list(range(len(paths)))
    out, pos = [], np.zeros(2)
    while remaining:
        starts = np.array([paths[i][0] for i in remaining])
        ends = np.array([paths[i][-1] for i in remaining])
        ds, de = np.linalg.norm(starts - pos, axis=1), np.linalg.norm(ends - pos, axis=1)
        k = int(np.argmin(np.minimum(ds, de)))
        p = paths[remaining.pop(k)]
        if de[k] < ds[k]:
            p = p[::-1]
        out.append(p)
        pos = p[-1]
    return out


def trace(mask, min_len=8.0, eps=0.9, gap=4.0, scale=1, bridge=10.0):
    """Trace a mask drawn at `scale` x photo size; lengths and the returned
    coordinates are in photo pixels. Dense paths keep ~1 point per photo px."""
    paths = skeleton_paths(mask)
    paths = stitch(paths, gap * scale, reach=6 * scale, bridge=bridge * scale)
    paths = [np.vstack([p[:-1:scale], p[-1:]]) / scale for p in paths]
    paths = [simplify(p.astype(np.float32), eps) for p in paths]
    paths = [p for p in paths if path_length(p) >= min_len]
    return paths


def travel_stats(layers):
    """Pen-down length, pen-up travel and lift count for a plot in layer order."""
    down = up = 0.0
    lifts = 0
    pos = np.zeros(2)
    for paths in layers:
        for p in paths:
            up += float(np.linalg.norm(p[0] - pos))
            down += path_length(p)
            pos = p[-1]
            lifts += 1
    return {"strokes": lifts, "pen_down_px": round(down), "pen_up_px": round(up)}
