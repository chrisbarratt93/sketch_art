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
    ridge_scales: tuple = (1.5, 2.5)  # Hessian scales ~ half-width of bars (px)
    ridge_keep: float = 6.0           # % of pixels counted as strong ridge
    min_component: int = 25           # drop specks smaller than this (px)
    margin: int = 4                   # ignore this many px at the border


def load_gray(path, max_side):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
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
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    # Thresholds from the gradient distribution, not fixed numbers, so the
    # same settings work on bright and dull photos.
    mag = np.hypot(cv2.Sobel(blurred, cv2.CV_32F, 1, 0), cv2.Sobel(blurred, cv2.CV_32F, 0, 1))
    hi = np.percentile(mag, 100 - keep)
    return cv2.Canny(blurred, 0.4 * hi / 4, hi / 4, L2gradient=True) > 0


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


def thin(mask, min_size):
    mask = remove_small_objects(mask, max_size=min_size, connectivity=2)
    return skeletonize(mask)


def extract(path, p=EdgeParams()):
    color, gray = load_gray(path, p.max_side)
    flat = flatten(gray)

    centre, line_band = ridges(flat, p.ridge_scales, p.ridge_keep)
    lines = thin(centre, p.min_component)

    edge = contours(flat, p.contour_sigma, p.contour_keep)
    # A contour inside (a slightly grown) ridge band is just the flank of a bar.
    band = cv2.dilate(line_band.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    contour = thin(edge & ~band, p.min_component)

    # Filters misbehave at the image border; nothing worth drawing lives there.
    m = p.margin
    for k in (lines, contour):
        k[:m], k[-m:], k[:, :m], k[:, -m:] = False, False, False, False

    return {"color": color, "gray": gray, "flat": flat,
            "contours": contour, "lines": lines, "ridge_band": line_band}
