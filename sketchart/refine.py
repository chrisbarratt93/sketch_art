"""Refine a generated ink drawing so its marks look deliberate.

A generated drawing (klein) has the right content at the right density, but
up close its marks give it away: ragged edges, speckle, strokes that break
and restart, lines that jitter, and line widths that swell and pinch at
random. A pen drawing is made of strokes: each one starts, travels smoothly
at a steady width and stops. So:

1. Vectorise: ink is thinned to centrelines and traced into strokes, keeping
   the drawn width along each one.
2. Clean the stroke set: drop specks and stray ticks, re-join strokes that
   were broken where they should continue.
3. Make each stroke deliberate: split at real corners, smooth the jitter out
   of each run, and make near-straight runs straight.
4. Re-draw: every stroke at one steady width (its median drawn width), with
   ends that taper as a pen lifts, so it renders as ink rather than a trace.

Everything works in drawing pixels. Areas too wide to be a line (solid
blacks) are kept as filled shapes rather than skeletonised.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

from .stylise import fit_line, sample, split_at_corners
from .trace import path_length, simplify, stitch, trace


@dataclass
class RefineParams:
    threshold: int = 180        # ink is darker than this (0-255); klein's thin lines are grey
    detail: int = 2             # vectorise on this multiple of the drawing grid
    speck_area: float = 10.0    # ink blobs smaller than this (px^2) are dropped
    hole_area: float = 6.0      # paper specks inside ink smaller than this (px^2) are filled
    fill_width: float = 7.0     # ink wider than this (px) is a solid fill, kept as a shape
    min_len: float = 0.0        # drop strokes shorter than this (px), 0 = keep all
    bridge: float = 0.0         # re-join aligned stroke ends up to this far apart (px)
    smooth: float = 0.0         # smoothing along each stroke (sigma, px); 0 = off
    corner_deg: float = 40.0    # turns sharper than this are kept as corners
    straight_tol: float = 0.0   # runs within this of a straight line become straight (px)
    steady: bool = False        # draw each stroke at its median width with tapered ends
    taper: float = 6.0          # taper length at a stroke's free ends (px)


# Starting points found with review_klein.py.
PRESETS = {
    "plain": dict(),
    "tidy": dict(smooth=2.5, steady=True, min_len=5.0, bridge=6.0, straight_tol=0.7),
}


def vectorise(gray, p):
    """-> (strokes in drawing px, distance-to-paper map in drawing px on the
    `detail` grid, fill mask at drawing size)."""
    f = p.detail
    big = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) if f > 1 else gray
    ink = big < p.threshold
    if p.speck_area:
        ink = remove_small_objects(ink, max_size=int(p.speck_area * f * f))
    if p.hole_area:
        ink = remove_small_holes(ink, max_size=int(p.hole_area * f * f))
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5) / f
    core = dist > p.fill_width / 2
    k = int(p.fill_width * f) | 1
    fill = (cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0) & ink
    lines = ink & ~fill
    strokes = trace(skeletonize(lines), min_len=1.0, eps=0, scale=f, gap=1.5, bridge=0)
    small_fill = cv2.resize(fill.astype(np.uint8), gray.shape[::-1], interpolation=cv2.INTER_AREA) > 0
    # Effective width of thin lines from the ink they carry: a faint grey 1px line
    # holds less ink than a black one. A disk of radius r over a line of width w
    # holds about 2*r*w of ink. Thick lines use their shape (neighbours would inflate it).
    r = 2.0 * f
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * r) | 1,) * 2).astype(np.float32)
    dark = (255 - big.astype(np.float32)) / 255
    # Normalise by the kernel's actual diameter in px (odd), not 2r.
    ink_w = cv2.filter2D(dark, -1, disk) / disk.shape[0] / f
    width = np.where(dist < 1.5, np.minimum(ink_w, 2 * dist + 0.5), 2 * dist)
    return strokes, width, small_fill


def _resample(pts, step=1.0):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 2 * step:
        return pts.astype(float)
    t = np.linspace(0, s[-1], int(np.ceil(s[-1] / step)) + 1)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)


def deliberate(pts, p):
    """Split at corners, smooth each run, straighten near-straight runs, and
    rejoin the runs at their shared corners."""
    pts = _resample(np.asarray(pts, float))
    if p.smooth <= 0 and p.straight_tol <= 0:
        return pts
    runs = split_at_corners(pts, p.corner_deg) if len(pts) > 12 else [pts]
    out = []
    for r in runs:
        if p.smooth > 0 and len(r) > 4:
            # Pin the ends so corners stay where they were.
            a, b = r[0].copy(), r[-1].copy()
            r = np.stack([gaussian_filter1d(r[:, i], p.smooth, mode="nearest") for i in (0, 1)], 1)
            n = len(r)
            w = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / max(2 * p.smooth, 1), 0, 1)[:, None]
            r[0], r[-1] = a, b
            r = w * r + (1 - w) * np.where(np.arange(n)[:, None] < n / 2, a, b)
        if p.straight_tol > 0 and len(r) >= 3:
            c, u, dev, _ = fit_line(r)
            if dev <= p.straight_tol:
                t = (r - c) @ u
                r = np.array([c + t[0] * u, c + t[-1] * u])
        out.append(r if not out else r[1:])
    return np.vstack(out)


def refine(gray, p=RefineParams()):
    """-> {"strokes": [(pts, width)], "fill": mask, "size": (w, h)}."""
    strokes, dist, fill = vectorise(gray, p)
    if p.bridge > 0:
        strokes = stitch(strokes, gap=1.5, reach=6, bridge=p.bridge)
    out = []
    for s in strokes:
        if path_length(s) < max(p.min_len, 1.0):
            continue
        widths = np.maximum(sample(dist, s * p.detail), 0.4)
        d = deliberate(s, p)
        out.append((simplify(d.astype(np.float32), 0.15), widths))
    h, w = gray.shape
    return {"strokes": out, "fill": fill, "size": (w, h)}


def _draw_stroke(canvas, d, wid):
    """Exact-width stroke: one quad per segment (cv2's thick lines run ~2px wide)
    plus round joins, with 1/16 px precision."""
    S = 16
    n = len(d)
    if n < 2:
        cv2.circle(canvas, tuple(np.round(d[0] * S).astype(int)), int(wid[0] / 2 * S), 255, -1, cv2.LINE_8, 4)
        return
    seg = np.diff(d, axis=0)
    nrm = np.stack([-seg[:, 1], seg[:, 0]], 1) / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9)
    for i in range(n - 1):
        a, b = d[i], d[i + 1]
        ha, hb = nrm[i] * wid[i] / 2, nrm[i] * wid[i + 1] / 2
        quad = np.array([a + ha, b + hb, b - hb, a - ha])
        cv2.fillConvexPoly(canvas, np.round(quad * S).astype(np.int32), 255, cv2.LINE_8, 4)
        if 0 < i:
            r = wid[i] / 2
            if r * S >= 8:
                cv2.circle(canvas, tuple(np.round(a * S).astype(int)), int(r * S), 255, -1, cv2.LINE_8, 4)


def render(result, p=RefineParams(), ss=4, source=None):
    """Draw refined strokes as ink: grayscale, white paper. Lines are drawn
    unantialiased at `ss` x and area-averaged down, so a stroke of width w
    carries w px of ink. Fill areas (dense dark texture) keep `source`'s own
    pixels when given, otherwise they are drawn solid."""
    w, h = result["size"]
    canvas = np.zeros((h * ss, w * ss), np.uint8)
    for pts, widths in result["strokes"]:
        d = _resample(np.asarray(pts, float), 0.5)
        n = len(d)
        if p.steady:
            base = float(np.median(widths))
            s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(d, axis=0), axis=1))])
            L = s[-1]
            t = min(p.taper, L / 3) if L > 0 else 1
            wid = base * np.clip(np.minimum(s, L - s) / max(t, 1e-6), 0.35, 1.0)
        else:
            wid = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(widths)), widths)
        # Polygon fill covers boundary pixels too, adding ~1 px at ss: take it off.
        _draw_stroke(canvas, d * ss, np.maximum(wid * ss - 1, 0.5))
    ink = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    out = 255 - ink
    fill = result["fill"]
    if fill.any():
        out[fill] = np.minimum(out[fill], source[fill] if source is not None else 0)
    return out
