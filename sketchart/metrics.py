"""Numbers for the things that make a traced drawing look machine-made.

All lengths are in photo pixels. Lower is better for every metric except
`strokes`/`length`, which are context (how much is drawn).

* doubled:  fraction of drawn length that runs alongside a parallel line from
            another stroke 1.5-6px away, i.e. both outlines of a thin feature.
* wobble:   median RMS deviation (px) of near-straight strokes from their
            best-fit line. A ruled or confident line is close to 0.
* short:    fraction of strokes shorter than 6px (specks, ticks).
* loops:    small closed strokes (< 30px round): blobs, mushy lettering.

Fidelity, against a reference line mask (the uncleaned trace), so cleanup
can't score well by drawing something else:
* invented: fraction of drawn length more than `tol` px from any reference line.
* missed:   fraction of reference line length more than `tol` px from any drawn line.
"""

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .trace import path_length


def _resample(p, step):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step:
        return None
    t = np.arange(0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def doubled_fraction(strokes, step=1.5, near=(1.5, 6.0), cos_par=0.94):
    pts, tans, ids = [], [], []
    for i, p in enumerate(strokes):
        r = _resample(np.asarray(p, float), step)
        if r is None or len(r) < 3:
            continue
        t = np.gradient(r, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        pts.append(r), tans.append(t), ids.append(np.full(len(r), i))
    if not pts:
        return 0.0
    pts, tans, ids = np.vstack(pts), np.vstack(tans), np.concatenate(ids)
    tree = cKDTree(pts)
    hit = np.zeros(len(pts), bool)
    for k, nbrs in enumerate(tree.query_ball_point(pts, near[1])):
        nb = np.asarray(nbrs)
        nb = nb[ids[nb] != ids[k]]
        if not len(nb):
            continue
        off = pts[nb] - pts[k]
        d = np.linalg.norm(off, axis=1)
        par = np.abs(tans[nb] @ tans[k]) > cos_par
        across = np.abs(off @ tans[k]) < 0.5 * d   # partner is beside, not ahead
        hit[k] = np.any((d >= near[0]) & par & across)
    return float(hit.mean())


def wobble(strokes, min_len=20.0, max_dev=3.0):
    rms = []
    for p in strokes:
        p = np.asarray(p, float)
        if path_length(p) < min_len:
            continue
        r = _resample(p, 1.0)
        c = r.mean(0)
        _, _, vt = np.linalg.svd(r - c, full_matrices=False)
        dev = (r - c) @ vt[1]
        if np.abs(dev).max() <= max_dev:          # meant to be straight
            rms.append(np.sqrt(np.mean(dev ** 2)))
    return float(np.median(rms)) if rms else 0.0


def _distance_to(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)


def fidelity(strokes, ref_mask, scale, tol=3.0):
    """`ref_mask` is on a grid `scale` x photo pixels."""
    h, w = ref_mask.shape
    drawn = np.zeros_like(ref_mask, np.uint8)
    for s in strokes:
        cv2.polylines(drawn, [np.round(np.asarray(s) * scale).astype(np.int32)], False, 1, 1)
    drawn = drawn > 0
    t = tol * scale
    invented = float((_distance_to(ref_mask)[drawn] > t).mean()) if drawn.any() else 0.0
    missed = float((_distance_to(drawn)[ref_mask] > t).mean()) if ref_mask.any() else 0.0
    return round(invented, 3), round(missed, 3)


def summary(strokes, ref_mask=None, scale=1):
    strokes = [np.asarray(s, float) for s in strokes if len(s) >= 2]
    lengths = np.array([path_length(s) for s in strokes]) if strokes else np.zeros(0)
    closed = [s for s, l in zip(strokes, lengths)
              if l < 30 and np.linalg.norm(s[0] - s[-1]) < 2 and len(s) > 3]
    out = {
        "strokes": len(strokes),
        "length": int(lengths.sum()),
        "doubled": round(doubled_fraction(strokes), 3),
        "wobble": round(wobble(strokes), 3),
        "short": round(float((lengths < 6).mean()) if len(lengths) else 0.0, 3),
        "loops": len(closed),
    }
    if ref_mask is not None:
        out["invented"], out["missed"] = fidelity(strokes, ref_mask, scale)
    return out
