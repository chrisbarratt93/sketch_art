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

Optionally (`scene`, `declutter`, see sketchart/scene.py) the photo is first
cleaned of clutter and split into scene layers, and each layer gets its own
thresholds, minimum mark length, smoothing and heaviest pen (REGION_STYLE):
buildings are traced as before, foliage and hills only by their boldest
lines, the ground lightly, and the sky stays paper.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .edges import load_gray, teed_centrelines, teed_probability
from .stylise import sample, smooth_path
from .cleanup import collapse_pairs
from .trace import path_length, simplify, stitch, trace


@dataclass
class InkParams:
    max_side: int = 1400        # working resolution (longest side, px)
    detail: int = 2             # trace on this multiple of the photo grid
    teed_scale: float = 2.0     # run TEED at this multiple of the photo size
    hi: float = 0.45            # hysteresis thresholds on the edge map (0..1)
    lo: float = 0.2
    min_len: float = 3.0        # drop marks shorter than this (photo px)
    smooth: float = 0.8         # stair-step smoothing (photo px)
    collapse: float = 3.0       # draw facing parallel outlines closer than this as one midline (photo px)
    heavy_pct: float = 15.0     # boldest % of strokes -> heavy pen
    fine_pct: float = 35.0      # faintest % of strokes -> fine pen
    scene: bool = False         # per-region styles from semantic segmentation (REGION_STYLE)
    declutter: str = "off"      # "inpaint": fill clutter before tracing; "drop": leave it as paper
    clutter: tuple | None = None  # text prompts for declutter (None = scene.CLUTTER)
    clutter_grow: int = 6       # grow clutter masks by this (photo px) to catch halos and shadows
    sky_margin: int = 4         # sky closer than this to anything else still draws (photo px)


PENS = ["fine", "medium", "heavy"]
# Per scene layer: hysteresis thresholds, shortest mark and smoothing (photo px),
# and the heaviest pen allowed. Unset keys fall back to InkParams.
REGION_STYLE = {
    "sky": dict(lo=np.inf, hi=np.inf),
    "building": dict(),
    "vegetation": dict(lo=0.35, hi=0.6, min_len=8.0, smooth=1.5, top="medium"),
    "hills": dict(lo=0.35, hi=0.6, min_len=10.0, smooth=2.0, top="fine"),
    "ground": dict(lo=0.3, hi=0.55, min_len=6.0, top="fine"),
    "other": dict(),
}


def ink(path, p=InkParams(), prob=None):
    """Returns {"heavy", "medium", "fine": [strokes], "gray", "prob"}, plus
    "photo" (the photo actually traced), "layers" and "clutter" when used.
    Pass a previously returned `prob` to skip running the network again
    (only valid for the same photo, size and declutter setting)."""
    color, gray = load_gray(path, p.max_side)
    f = p.detail
    out = {"heavy": [], "medium": [], "fine": []}

    clutter, layers = None, None
    if p.scene or p.declutter != "off":
        from . import scene
        # Segmented before inpainting: removed objects fall in "other", which
        # is drawn like a building, so the fill behind them is traced normally.
        layers = scene.segment(color)
    if p.declutter != "off":
        clutter, out["clutter_labels"] = scene.clutter_mask(color, list(p.clutter or scene.CLUTTER), layers)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.clutter_grow + 1,) * 2)
        clutter = cv2.dilate(clutter.astype(np.uint8), k) > 0
        out["clutter"] = clutter
        if p.declutter == "inpaint":
            color = scene.inpaint(color, clutter)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)[:, :, 0]

    lo, hi = p.lo, p.hi
    if not p.scene:
        layers = None
    else:
        out["layers"] = layers
        lo = np.array([REGION_STYLE[n].get("lo", p.lo) for n in scene.LAYERS], np.float32)[layers]
        hi = np.array([REGION_STYLE[n].get("hi", p.hi) for n in scene.LAYERS], np.float32)[layers]
        # Silhouettes sit on the sky boundary and the segmentation is a few px
        # out, so only sky well away from everything else is kept blank.
        sky = layers == scene.LAYERS.index("sky")
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.sky_margin + 1,) * 2)
        edge = sky & ~(cv2.erode(sky.astype(np.uint8), k) > 0)
        lo[edge], hi[edge] = p.lo, p.hi
    if p.declutter == "drop":
        lo, hi = np.broadcast_to(lo, gray.shape).copy(), np.broadcast_to(hi, gray.shape).copy()
        lo[clutter] = hi[clutter] = np.inf
    if np.ndim(lo):
        lo, hi = (cv2.resize(t, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for t in (lo, hi))

    if prob is None:
        prob = teed_probability(color, f, p.teed_scale)
    mask = teed_centrelines(prob, lo, hi)
    # How bold the network drew each line: local width of its confident band.
    band = (prob >= p.lo).astype(np.uint8)
    width = cv2.distanceTransform(band, cv2.DIST_L2, 3)

    strokes, bold, styles = [], [], []
    traced = trace(mask, min_len=p.min_len, eps=0, scale=f, gap=1.5, bridge=0)
    if p.collapse > 0:
        # Collapsing leaves midlines and the unpaired remainders as separate pieces
        # that end where they used to meet; join them back up.
        traced = stitch(collapse_pairs(traced, p.collapse), gap=p.collapse, reach=4)
    for pts in traced:
        style = {}
        if layers is not None:
            ij = np.clip(np.round(pts).astype(int), 0, [layers.shape[1] - 1, layers.shape[0] - 1])
            style = REGION_STYLE[scene.LAYERS[np.bincount(layers[ij[:, 1], ij[:, 0]]).argmax()]]
        if path_length(pts) < style.get("min_len", p.min_len):
            continue
        on_grid = pts * f
        bold.append(float(np.mean(sample(prob, on_grid) * np.minimum(sample(width, on_grid) / f, 2.0))))
        sm = style.get("smooth", p.smooth)
        s = smooth_path(pts, sm) if path_length(pts) > 4 * sm else pts
        strokes.append(simplify(s.astype(np.float32), 0.2))
        styles.append(style)

    out.update(gray=gray, prob=prob, photo=color)
    if strokes:
        hi_cut = np.percentile(bold, 100 - p.heavy_pct)
        lo_cut = np.percentile(bold, p.fine_pct)
        for s, b, style in zip(strokes, bold, styles):
            pen = PENS.index("heavy" if b >= hi_cut else "fine" if b < lo_cut else "medium")
            out[PENS[min(pen, PENS.index(style.get("top", "heavy")))]].append(s)
    return out
