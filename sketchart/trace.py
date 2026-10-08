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
    if len(pts) < 3:
        return pts
    return cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, False).reshape(-1, 2)


def stitch(paths, gap):
    """Join paths end-to-end where their ends lie within `gap` px and roughly
    continue each other's direction. Fewer, longer strokes read more like a
    confident hand and save pen lifts."""
    paths = [p for p in paths if len(p) >= 2]

    def end_dir(p, at_end):
        a, b = (p[-1], p[max(0, len(p) - 4)]) if at_end else (p[0], p[min(len(p) - 1, 3)])
        d = a - b
        n = np.linalg.norm(d)
        return d / n if n else d

    changed = True
    while changed:
        changed = False
        ends = np.array([[p[0], p[-1]] for p in paths]).reshape(-1, 2)  # 2 ends per path
        alive = [True] * len(paths)
        for i in range(len(paths)):
            if not alive[i]:
                continue
            for i_end in (1, 0):
                pt = paths[i][-1] if i_end else paths[i][0]
                d = np.linalg.norm(ends - pt, axis=1)
                d[2 * i:2 * i + 2] = np.inf
                for k in np.argsort(d)[:4]:
                    if d[k] > gap:
                        break
                    j, j_end = divmod(int(k), 2)
                    if not alive[j]:
                        continue
                    # Outward direction at i's end should oppose j's outward direction.
                    if np.dot(end_dir(paths[i], i_end), end_dir(paths[j], j_end)) > -0.5:
                        continue
                    a = paths[i] if i_end else paths[i][::-1]
                    b = paths[j][::-1] if j_end else paths[j]
                    paths[i] = np.vstack([a, b])
                    alive[j] = False
                    ends[2 * j:2 * j + 2] = np.inf
                    ends[2 * i], ends[2 * i + 1] = paths[i][0], paths[i][-1]
                    changed = True
                    break
        paths = [p for p, a in zip(paths, alive) if a]
    return paths


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


def trace(mask, min_len=8.0, eps=0.9, gap=4.0):
    paths = skeleton_paths(mask)
    paths = stitch(paths, gap)
    paths = [simplify(p, eps) for p in paths]
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
