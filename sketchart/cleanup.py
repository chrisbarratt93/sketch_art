"""Clean-up passes on traced strokes (photo-pixel coordinates).

collapse_pairs: an edge detector outlines both sides of a thin feature (a
glazing bar, a wire, a lamp post), which reads as a tube with rounded ends -
the clearest tell of a traced edge map. Where two stretches of line run
parallel and facing each other no more than `max_sep` apart, draw their
midline once instead. Wider features (a 6px mullion) keep both edges, as an
architect would draw them.
"""

import numpy as np
from scipy.spatial import cKDTree

from .trace import path_length


def _resample(p, step=1.0):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 2 * step:
        return p.astype(float), s
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1), t


def _runs(mask):
    """(start, stop) of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _nearest_partner(P, S, A, max_sep):
    """Fallback partner for gap-bridged samples: the nearest sample on another
    stroke (or far along this one) within 1.5 x max_sep."""
    dist, idx = cKDTree(P).query(P, k=min(8, len(P)), distance_upper_bound=1.5 * max_sep)
    n = len(P)
    out = np.full(n, -1)
    for c in range(1, idx.shape[1]):
        j = idx[:, c]
        jj = np.where(j < n, j, 0)
        ok = (j < n) & ((S != S[jj]) | (np.abs(A - A[jj]) > 3 * max_sep))
        out = np.where((out < 0) & ok, jj, out)
    return out


def collapse_pairs(strokes, max_sep=3.5, min_sep=0.8, cos_par=0.9, k=12):
    if max_sep <= 0 or not strokes:
        return strokes
    pts, tan, sid, arc = [], [], [], []
    for i, s in enumerate(strokes):
        r, t = _resample(np.asarray(s, float))
        g = np.gradient(r, axis=0) if len(r) > 1 else np.zeros_like(r)
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
        pts.append(r), tan.append(g), sid.append(np.full(len(r), i)), arc.append(t)
    offs = np.cumsum([0] + [len(r) for r in pts])
    P, T, S, A = np.vstack(pts), np.vstack(tan), np.concatenate(sid), np.concatenate(arc)

    # Best facing partner for every sample: parallel, beside (not ahead), within max_sep,
    # and on the same stroke only if far along it (the other side of a closed loop).
    dist, idx = cKDTree(P).query(P, k=min(k, len(P)), distance_upper_bound=max_sep)
    n = len(P)
    partner = np.full(n, -1)
    for c in range(1, idx.shape[1]):
        j = idx[:, c]
        ok = j < n
        jj = np.where(ok, j, 0)
        off = P[jj] - P
        d = dist[:, c]
        ok &= (d >= min_sep) & (np.abs(np.sum(T * T[jj], 1)) > cos_par)
        ok &= np.abs(np.sum(off * T, 1)) < 0.5 * np.maximum(d, 1e-9)
        ok &= (S != S[jj]) | (np.abs(A - A[jj]) > 3 * max_sep)
        partner = np.where((partner < 0) & ok, jj, partner)
    paired = partner >= 0
    # Pairing flickers sample to sample along a stroke, which would chop it into
    # dashes. Per stroke: bridge short unpaired gaps, then ignore short paired runs.
    min_run, max_gap = int(3 * max_sep), int(2 * max_sep)
    for i in range(len(strokes)):
        a, b = offs[i], offs[i + 1]
        m = paired[a:b]
        for s0, s1 in _runs(~m):
            if 0 < s0 and s1 < len(m) and s1 - s0 <= max_gap:
                m[s0:s1] = True
        for s0, s1 in _runs(m):
            if s1 - s0 < min_run:
                m[s0:s1] = False
        paired[a:b] = m
    partner = np.where(paired & (partner < 0), _nearest_partner(P, S, A, max_sep), partner)
    paired &= partner >= 0
    # Keep each pair once: the sample with the smaller index carries the midline.
    keep_mid = paired & (np.arange(n) < partner)
    drop = paired & ~keep_mid

    out = []
    for i in range(len(strokes)):
        a, b = offs[i], offs[i + 1]
        q = P[a:b].copy()
        mid = keep_mid[a:b]
        q[mid] = 0.5 * (q[mid] + P[partner[a:b][mid]])
        gone = drop[a:b].copy()
        # A short unpaired stretch between a dropped run and the stroke's end is
        # the rounded cap of a "sausage": drop it too.
        for s0, s1 in _runs(~gone & ~mid):
            touches = (s0 > 0 and (gone[s0 - 1] or mid[s0 - 1])) or (s1 < len(gone) and (gone[s1] or mid[s1]))
            at_end = s0 == 0 or s1 == len(gone)
            if touches and at_end and (s1 - s0) < 3 * max_sep:
                gone[s0:s1] = True
        for s0, s1 in _runs(~gone):
            piece = q[s0:s1]
            if len(piece) >= 2 and path_length(piece) >= 2:
                out.append(piece)
    return out
