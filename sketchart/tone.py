"""Stage 3: tone by hatching.

How an ink artist builds value, and what this module copies:

* Lights stay white paper. Only areas darker than a threshold get any ink,
  and the threshold sits fairly dark: sketches exaggerate contrast.
* Value is built in passes. Mid tones get one set of parallel diagonals; darker
  areas get a second set crossing it; the deepest shadows get the first set
  again, between the existing lines. Each pass is its own layer.
* Hatching is done in patches: a block of short parallel strokes of similar
  length, then the next block, with little gaps and ragged ends. Long,
  ruler-perfect lines across a whole region look like a machine fill.
* The sky is left empty, and hatching stops sooner than line work at the
  vignette, so the drawing fades from tone to line to paper.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label
from skimage.morphology import remove_small_objects

from .stylise import hand_line, sample


@dataclass
class ToneParams:
    levels: tuple = (0.50, 0.66, 0.80)   # darkness (0..1) at which each pass starts
    angles: tuple = (52.0, -20.0, 52.0)  # hatch angle per pass, degrees anticlockwise from horizontal
    spacing: float = 6.5                 # px between hatch lines within a pass
    patch_len: tuple = (16.0, 40.0)      # length range of a patch of strokes (px)
    patch_lines: tuple = (5, 10)         # how many adjacent lines share patch breaks
    gap: tuple = (1.5, 4.0)              # gap between patches along a line (px)
    jitter_deg: float = 1.5              # per-stroke angle jitter
    bridge: int = 7                      # light bars thinner than ~this (px) get toned over
    min_area: int = 300                  # ignore tone regions smaller than this (px)
    reach: float = 0.72                  # hatch only this far into the vignette (0..1)
    keep_sky: bool = True                # leave the sky as white paper
    seed: int = 2


def darkness(gray):
    """0 = paper white, 1 = darkest ink, stretched to the photo's own range
    and smoothed so regions are blocky, not pixel noise."""
    g = gray
    for _ in range(2):
        g = cv2.bilateralFilter(g, d=9, sigmaColor=25, sigmaSpace=9)
    g = cv2.GaussianBlur(g, (0, 0), 2.5).astype(np.float32)
    lo, hi = np.percentile(g, (2, 98))
    return 1 - np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)


def sky_mask(gray):
    """Smooth, low-texture area connected to the top of the frame."""
    g = cv2.GaussianBlur(gray, (0, 0), 2).astype(np.float32)
    mean = cv2.GaussianBlur(g, (0, 0), 6)
    std = np.sqrt(np.maximum(cv2.GaussianBlur(g * g, (0, 0), 6) - mean * mean, 0))
    smooth = std < max(4.0, np.percentile(std, 30))
    smooth = cv2.morphologyEx(smooth.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    lab, _ = label(smooth)
    top = set(np.unique(lab[:3])) - {0}
    sky = np.isin(lab, list(top))
    # Swallow little islands (birds, wires, cloud edges) inside the sky.
    sky = binary_fill_holes(sky)
    return cv2.dilate(sky.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0


def region(dark, level, exclude, min_area, bridge):
    m = (dark >= level) & ~exclude
    # Close first: bridges thin light bars, so a glazed wall becomes one toned
    # area (as an artist sees it) while wide light panels stay white. Then
    # open: removes slivers too thin to hatch convincingly.
    ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, ell(bridge))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ell(9)) > 0
    return remove_small_objects(m, max_size=min_area)


def hatch(mask, angle_deg, spacing, p, rng, phase=0.0):
    """Patchy parallel strokes filling `mask` at `angle_deg`."""
    h, w = mask.shape
    a = np.radians(angle_deg)
    u = np.array([np.cos(a), -np.sin(a)])      # along the stroke (image y points down)
    n = np.array([-u[1], u[0]])                # across the strokes
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    o_min, o_max = (corners @ n).min(), (corners @ n).max()
    t_min, t_max = (corners @ u).min(), (corners @ u).max()
    t = np.arange(t_min, t_max, 1.0)

    strokes = []
    offsets = np.arange(o_min + phase * spacing, o_max, spacing)
    i = 0
    while i < len(offsets):
        # A band of adjacent lines shares where its patches break.
        band = offsets[i:i + rng.integers(*p.patch_lines, endpoint=True)]
        i += len(band)
        breaks, pos = [], t_min + rng.uniform(0, p.patch_len[1])
        while pos < t_max:
            breaks.append(pos)
            pos += rng.uniform(*p.patch_len)
        breaks = np.array(breaks)
        for o in band:
            o = o + rng.normal(0, 0.12 * spacing)
            pts = o * n + t[:, None] * u
            inside = sample(mask, pts).astype(bool)
            inside &= (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            if not inside.any():
                continue
            # Cut at the band's patch breaks (each line a little off), leaving a gap.
            cut = np.zeros(len(t), bool)
            for b in breaks + rng.normal(0, 1.5, len(breaks)):
                cut |= np.abs(t - b) < rng.uniform(*p.gap) / 2
            on = inside & ~cut
            edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
            for s, e in zip(edges[::2], edges[1::2]):
                if e - s < 4:
                    continue
                # Ragged ends: hands don't stop exactly at a boundary.
                s = max(0, s + int(rng.normal(0, 1.2)))
                e = min(len(t), e + int(rng.normal(0, 1.2)))
                seg = np.array([pts[s], pts[e - 1]])
                strokes.append(_tilt(seg, rng, p.jitter_deg))
    return [hand_line(s, rng, 0.25, bow=False) for s in strokes if np.linalg.norm(s[1] - s[0]) >= 3]


def _tilt(seg, rng, deg):
    ang = np.radians(rng.normal(0, deg))
    c, s = np.cos(ang), np.sin(ang)
    m = seg.mean(0)
    return (seg - m) @ np.array([[c, s], [-s, c]]) + m


def tone(gray, field, p=ToneParams()):
    """Returns {"dark": darkness map, "regions": [masks], "passes": [[strokes], ...]}."""
    rng = np.random.default_rng(p.seed)
    dark = darkness(gray)
    exclude = field > p.reach
    if p.keep_sky:
        exclude |= sky_mask(gray)
    regions, passes = [], []
    for k, (level, angle) in enumerate(zip(p.levels, p.angles)):
        m = region(dark, level, exclude, p.min_area, p.bridge)
        regions.append(m)
        # A pass that repeats an earlier angle goes between its lines.
        phase = 0.5 if angle in p.angles[:k] else 0.0
        passes.append(hatch(m, angle, p.spacing, p, rng, phase))
    return {"dark": dark, "regions": regions, "passes": passes}
