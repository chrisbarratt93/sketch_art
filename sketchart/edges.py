"""Stage 1a: pull a clean line map out of a photo.

An ink artist uses one line for two different things, and we need to find
both:

* contours - boundaries between areas of different tone (a roof against the
  sky, the edge of a column). These are *step edges*: found with Canny.
* lines - things that are themselves thin (glazing bars, mullions, railings,
  mortar joints). Canny would outline both sides of these and give a
  "sausage"; an artist draws a single stroke down the middle. These are
  *ridges/valleys*: found from the Hessian, which gives the centreline.

Contours that just run along the flank of a detected line are dropped, so a
thin bar becomes one stroke, not three.

Both come out as 1-pixel-wide binary masks, ready to trace into pen strokes.
Detection runs on an upsampled copy of the photo (`detail`, default 2x): the
masks are that much finer than the photo's pixel grid, so traced strokes land
to half a pixel and bars only a pixel or two wide are still resolved. All
lengths below are in photo pixels and are scaled internally.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, skeletonize


@dataclass
class EdgeParams:
    max_side: int = 1400              # working resolution (longest side, px)
    contour_sigma: float = 2.0        # blur before Canny
    contour_keep: float = 10.0        # % of pixels counted as strong gradient
    ridge_scales: tuple = (1.0, 1.5, 2.5)  # Hessian scales ~ half-width of bars (px)
    ridge_keep: float = 7.0           # % of pixels counted as strong ridge
    min_component: int = 25           # drop specks smaller than this (px)
    margin: int = 4                   # ignore this many px at the border
    detail: int = 2                   # run detection at this multiple of the photo size
    method: str = "classic"           # "classic" (Canny + ridges) or "teed" (learned edges)
    teed_scale: float = 2.0           # run TEED at this multiple of the photo size
    teed_ridges: bool = False         # with "teed": also add ridge centrelines for thin bars
    teed_hi: float = 0.45             # TEED hysteresis thresholds (normalised probability)
    teed_lo: float = 0.2


def load_gray(path, max_side):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = max_side / max(h, w) if max_side else 1.0
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    # Lab lightness tracks perceived brightness better than a plain RGB average.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    return img, gray


def flatten(gray):
    """Edge-preserving smoothing: kills paving/cloud texture, keeps hard edges,
    then local contrast normalisation so shadowed areas still yield lines."""
    out = gray
    for _ in range(3):
        out = cv2.bilateralFilter(out, d=9, sigmaColor=30, sigmaSpace=7)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)


def contours(gray, sigma, keep):
    # Float throughout: on an upsampled image, gentle gradients are under one
    # grey level per pixel and an 8-bit blur would quantise them away.
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1)
    # Thresholds from the gradient distribution, not fixed numbers, so the
    # same settings work on bright and dull photos.
    mag = np.hypot(dx, dy)
    hi = np.percentile(mag, 100 - keep)
    k = 30000 / (mag.max() + 1e-6)  # Canny wants int16 gradients
    to16 = lambda g: np.clip(g * k, -32767, 32767).astype(np.int16)
    return cv2.Canny(to16(dx), to16(dy), 0.4 * hi / 4 * k, hi / 4 * k, L2gradient=True) > 0


def ridge_response(gray, scales):
    """Scale-normalised line response, positive for thin bright bars and thin
    dark bars alike and near zero on edges and flat areas, plus the angle of
    the line's normal (across the bar) at each pixel."""
    img = gray.astype(np.float32) / 255
    best = np.zeros_like(img)
    normal = np.zeros_like(img)
    for s in scales:
        Hrr, Hrc, Hcc = hessian_matrix(img, sigma=s, order="rc", use_gaussian_derivatives=False)
        l1, l2 = hessian_matrix_eigvals([Hrr, Hrc, Hcc])  # l1 >= l2
        # Bright ridge: strongly negative l2. Dark valley: strongly positive l1.
        # Require the cross-curvature to dominate so blobs/corners don't count.
        bright = np.where(np.abs(l2) > 2 * np.abs(l1), -l2, 0)
        dark = np.where(np.abs(l1) > 2 * np.abs(l2), l1, 0)
        resp = s * s * np.maximum(bright, dark)
        # The centre of a real bar is flat across (no gradient); the shoulder
        # of something wider than this scale (a panel, a wall) is steep.
        # Reject shoulders, or wide light areas get mistaken for bars.
        g = cv2.GaussianBlur(img, (0, 0), s)
        grad = s * np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)) / 2
        resp *= np.clip(1 - grad / (resp + 1e-6), 0, 1)
        # Angle (row, col frame) of l1's eigenvector; l2's is perpendicular.
        theta1 = 0.5 * np.arctan2(2 * Hrc, Hrr - Hcc)
        theta = np.where(dark >= bright, theta1, theta1 + np.pi / 2)
        better = resp > best
        best = np.where(better, resp, best)
        normal = np.where(better, theta, normal)
    return best, normal


def suppress(strength, angle):
    """Non-maximum suppression across the line: keep a pixel only if it beats
    both neighbours along its normal. Gives centrelines directly, where
    skeletonising a thresholded band leaves ladders and loops."""
    dr = np.rint(np.cos(angle)).astype(int)
    dc = np.rint(np.sin(angle)).astype(int)
    h, w = strength.shape
    rr, cc = np.mgrid[0:h, 0:w]
    a = strength[np.clip(rr + dr, 0, h - 1), np.clip(cc + dc, 0, w - 1)]
    b = strength[np.clip(rr - dr, 0, h - 1), np.clip(cc - dc, 0, w - 1)]
    return (strength >= a) & (strength >= b)


def ridges(gray, scales, keep):
    """Returns (centrelines, band): the 1px line mask and the full width of
    the bars it came from."""
    r, angle = ridge_response(gray, scales)
    hi = np.percentile(r, 100 - keep)
    band = apply_hysteresis_threshold(r, 0.5 * hi, hi)
    centre = suppress(r, angle) & band
    return centre, band


def teed_probability(bgr, f, scale):
    """TEED edge probability (0..1) on the `f` x grid, running the network at
    `scale` x the photo size."""
    from . import teed
    big = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    # Return on the f x grid directly: resampling the thin output lines down
    # and up again aliases them.
    prob = teed.edge_probability(big, scale / f)
    # The network's "no edge" output is sigmoid(min smish) ~ 0.438, not 0.
    return np.clip((prob - 0.438) / (1 - 0.438), 0, 1)


def teed_centrelines(prob, lo, hi):
    """1px centrelines of the confident part of a TEED map."""
    strong = apply_hysteresis_threshold(prob, lo, hi)
    # TEED's lines are already narrow, so the skeleton of the confident band is
    # its centreline. (Non-max suppression with 45-degree steps leaves it dotted.)
    return skeletonize(strong)


def teed_edges(bgr, f, p):
    """Learned edges as 1px centrelines on the `f` x grid. Returns
    (mask, probability)."""
    prob = teed_probability(bgr, f, p.teed_scale)
    return teed_centrelines(prob, p.teed_lo, p.teed_hi), prob


def thin(mask, min_size):
    mask = remove_small_objects(mask, max_size=min_size, connectivity=2)
    return skeletonize(mask)


def extract(path, p=EdgeParams()):
    """Masks "contours" and "lines" are at `detail` x the photo size (divide
    traced coordinates by p.detail); "gray" and "flat" are at photo size."""
    color, gray = load_gray(path, p.max_side)
    f = p.detail
    flat = flatten(gray)
    # Smooth at photo size, then upsample: smoothing at the larger size averages
    # over 4x the pixels and washes out low-contrast detail.
    work = cv2.resize(flat, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) if f > 1 else flat

    centre, line_band = ridges(work, [s * f for s in p.ridge_scales], p.ridge_keep)
    lines = thin(centre, p.min_component * f)
    k = 4 * f + 1   # ~2 photo px either side
    band = cv2.dilate(line_band.astype(np.uint8), np.ones((k, k), np.uint8)) > 0

    if p.method == "teed":
        edge, prob = teed_edges(color, f, p)
        if not p.teed_ridges:
            lines = np.zeros_like(lines)
            band = np.zeros_like(band)
    else:
        edge = contours(work, p.contour_sigma * f, p.contour_keep)
    # A contour inside (a slightly grown) ridge band is just the flank of a bar.
    contour = thin(edge & ~band, p.min_component * f)

    # Filters misbehave at the image border; nothing worth drawing lives there.
    m = p.margin * f
    for k in (lines, contour):
        k[:m], k[-m:], k[:, :m], k[:, -m:] = False, False, False, False

    return {"color": color, "gray": gray, "flat": flat, "detail": f,
            "contours": contour, "lines": lines, "ridge_band": line_band}
