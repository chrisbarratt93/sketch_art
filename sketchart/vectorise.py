"""Raster ink drawing -> pen strokes for the plotter.

Ink is separated from paper, thinned to centrelines and traced into strokes
with the same tracer as ink mode. Each stroke's pen comes from how wide the
drawn line was. Areas too wide to be a line (solid fills) are hatched with
parallel strokes instead, so the plotter never scribbles a skeleton through a
blob. Line weights are not yet calibrated against real pens.
"""

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize

from .stylise import sample
from .trace import order, trace


def ink_mask(gray, f):
    big = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    # Paper is rarely pure white in generated images; threshold against the local paper level.
    paper = cv2.dilate(big, np.ones((31, 31), np.uint8))
    ink = big < paper.astype(np.float32) * 0.72
    return remove_small_objects(ink, max_size=6 * f * f), big


def hatch(mask, spacing, angle=45.0):
    """Parallel strokes clipped to `mask` (pixel coordinates of `mask`)."""
    h, w = mask.shape
    a = np.deg2rad(angle)
    d, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.array([w / 2, h / 2])
    r = np.hypot(w, h) / 2
    out = []
    for off in np.arange(-r, r, spacing):
        t = np.arange(-r, r, 0.5)
        pts = c + off * n + t[:, None] * d
        ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        inside = np.zeros(len(t), bool)
        inside[ok] = mask[pts[ok, 1].astype(int), pts[ok, 0].astype(int)]
        edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(np.int8), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s > 4:
                out.append(pts[[s, e - 1]].astype(np.float32))
    return out


def vectorise(gray, f=2, fill_px=3.0, hatch_px=2.0):
    """Returns ((w, h), [(layer, pen width, strokes)], filled fraction)."""
    h, w = gray.shape
    ink, _ = ink_mask(gray, f)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    # Solid fills: anything more than 2*fill_px photo px across.
    core = dist > fill_px * f
    fill = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * fill_px * f) + 1,) * 2)) > 0
    fill &= ink
    lines = ink & ~fill
    strokes = trace(skeletonize(lines), min_len=2.0, eps=0.3, scale=f, gap=1.5, bridge=0)
    layers = {"heavy": [], "medium": [], "fine": []}
    for s in strokes:
        width = 2 * float(np.median(sample(dist, s * f))) / f  # drawn line width, photo px
        layers["heavy" if width >= 2.6 else "fine" if width < 1.4 else "medium"].append(s)
    fills = [p / f for p in hatch(fill, hatch_px * f)] if fill.any() else []
    out = [("heavy", 1.5, order(layers["heavy"])), ("medium", 1.0, order(layers["medium"])),
           ("fine", 0.6, order(layers["fine"])), ("fill", 0.6, order(fills))]
    return (w, h), out, float(fill.mean())
