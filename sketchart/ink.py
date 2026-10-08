"""Faithful ink tracing of a learned edge map.

TEED's edge map already looks like a pen drawing: it keeps the lines a person
would draw, at the density a person would draw them, and drops texture. So
instead of rebuilding the drawing from geometry, this traces that map as
directly as possible:

* the confident part of the map is skeletonised to 1px centrelines and traced
  into strokes, keeping short marks (they carry much of the hand-drawn feel);
* strokes are smoothed only enough to remove pixel stair-steps;
* line weight comes from the map itself: bold, confident edges go to a heavier
  pen layer and faint ones to a fine pen, the way ink weight varies by hand.

No straightening, merging, perspective snapping or corner joining.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .edges import EdgeParams, load_gray, teed_edges
from .stylise import sample, smooth_path
from .trace import path_length, simplify, trace


@dataclass
class InkParams:
    max_side: int = 1400        # working resolution (longest side, px)
    detail: int = 2             # trace on this multiple of the photo grid
    teed_scale: float = 2.0     # run TEED at this multiple of the photo size
    hi: float = 0.45            # hysteresis thresholds on the edge map (0..1)
    lo: float = 0.2
    min_len: float = 3.0        # drop marks shorter than this (photo px)
    smooth: float = 0.8         # stair-step smoothing (photo px)
    heavy_pct: float = 15.0     # boldest % of strokes -> heavy pen
    fine_pct: float = 35.0      # faintest % of strokes -> fine pen


def ink(path, p=InkParams()):
    """Returns {"heavy", "medium", "fine": [strokes], "gray", "prob"}."""
    color, gray = load_gray(path, p.max_side)
    f = p.detail
    mask, prob = teed_edges(color, f, EdgeParams(teed_scale=p.teed_scale, teed_hi=p.hi, teed_lo=p.lo))
    # How bold the network drew each line: local width of its confident band.
    band = (prob >= p.lo).astype(np.uint8)
    width = cv2.distanceTransform(band, cv2.DIST_L2, 3)

    strokes, bold = [], []
    for pts in trace(mask, min_len=p.min_len, eps=0, scale=f, gap=1.5, bridge=0):
        on_grid = pts * f
        bold.append(float(np.mean(sample(prob, on_grid) * np.minimum(sample(width, on_grid) / f, 2.0))))
        s = smooth_path(pts, p.smooth) if path_length(pts) > 4 * p.smooth else pts
        strokes.append(simplify(s.astype(np.float32), 0.2))

    out = {"heavy": [], "medium": [], "fine": [], "gray": gray, "prob": prob}
    if strokes:
        hi_cut = np.percentile(bold, 100 - p.heavy_pct)
        lo_cut = np.percentile(bold, p.fine_pct)
        for s, b in zip(strokes, bold):
            out["heavy" if b >= hi_cut else "fine" if b < lo_cut else "medium"].append(s)
    return out
