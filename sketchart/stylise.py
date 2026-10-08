"""Stage 2: turn traced edges into strokes that read as drawn by hand.

What separates a confident ink sketch from an edge map:

* Lines are *decisions*. A building edge is one straight stroke, not a chain
  of pixel steps broken wherever the photo was noisy. So paths are split at
  corners, near-straight runs become true straight strokes, and collinear
  fragments are merged across gaps.
* Not everything gets drawn. Strokes are scored by contrast, length and
  closeness to the focal point; weak ones are dropped, and the drawing fades
  out towards its edges through an irregular vignette instead of stopping at
  a rectangle.
* Weight carries hierarchy: silhouettes heavy, structure medium, detail fine.
  Each weight is its own layer (its own pen). Heavy strokes are also gone over
  twice, slightly offset, the way an artist reinforces a line. That works with
  a single pen too.
* The hand shows: straight strokes overshoot or stop short of corners and sit
  a fraction of a degree off true, long lines bow slightly, and every line has
  a low-frequency wobble.

All randomness comes from one seed, so a drawing is reproducible.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .edges import ridge_response
from .trace import path_length, simplify


@dataclass
class StyleParams:
    corner_deg: float = 35.0      # turn sharper than this splits a path
    straight_tol: float = 1.3     # max deviation (px) for a run to count as straight
    merge_deg: float = 3.0        # collinear merge: max angle between strokes
    merge_offset: float = 2.0     # collinear merge: max sideways offset (px)
    merge_gap: float = 14.0       # collinear merge: max gap between ends (px)
    min_len: float = 9.0          # drop strokes shorter than this after merging
    drop_pct: float = 30.0        # drop this % of strokes, lowest scores first
    vignette: float = 1.0         # size of the drawn area (bigger = less fade)
    ground_fade: float = 0.72     # below the focal point the drawn area is this much shorter
    sky_reach: float = 1.0        # above the focal point it is this much taller
    heavy_reach: float = 0.75     # heavy lines only this far into the vignette (0..1)
    heavy_pct: float = 8.0        # top % of contours drawn heavy
    heavy_min_len: float = 40.0   # heavy strokes must be at least this long
    indicate: float = 0.35        # chance of skipping a bar inside a repeating run
    fine_len: float = 40.0        # thin-bar strokes shorter than this are fine
    overshoot: float = 6.0        # max overshoot at a free line end (px)
    wobble: float = 0.55          # hand wobble amplitude (px)
    angle_jitter: float = 0.35    # sd of a straight stroke's tilt off true (degrees)
    curve_smooth: float = 1.5     # extra smoothing of curved strokes (px)
    vp_count: int = 3             # perspective directions to find (0 = off)
    vp_snap_deg: float = 2.0      # snap lines this close to a vanishing direction onto it
    join_reach: float = 12.0      # extend/trim a line end up to this far to meet a corner (0 = off)
    join_deg: float = 25.0        # lines must differ by this much to form a corner
    corner_over: tuple = (1.5, 5.0)  # how far lines cross past a joined corner (px)
    hand: bool = True             # False: geometry only, no selection/weight/hand effects
    seed: int = 1


PRESETS = {
    # Geometry only: every detected line, cleaned up (straightened, merged,
    # perspective-snapped, corners joined) but not selected, weighted or
    # hand-styled. For judging detection and geometry on their own.
    "plain": dict(hand=False, curve_smooth=2.5, join_reach=16.0, min_len=5.0),
    # Loose urban sketch: fades, indicates repeats, visible hand.
    "loose": dict(),
    # Architect's sketch: accurate perspective, corners that meet and cross,
    # more of the detail kept, steadier hand.
    "architect": dict(drop_pct=15.0, indicate=0.0, wobble=0.22, angle_jitter=0.08,
                      curve_smooth=2.5, ground_fade=0.65, sky_reach=1.4, heavy_pct=6.0, min_len=7.0,
                      join_reach=16.0),
}


def preset(name, **overrides):
    return StyleParams(**{**PRESETS[name], **overrides})


# ---------------------------------------------------------------- geometry

def smooth_path(pts, sigma=1.5):
    if len(pts) < 5:
        return pts.astype(np.float32)
    return np.stack([gaussian_filter1d(pts[:, i].astype(np.float32), sigma, mode="nearest")
                     for i in (0, 1)], axis=1)


def split_at_corners(pts, corner_deg, k=5):
    """Split a dense path where it turns sharply (local maxima of turn angle)."""
    n = len(pts)
    if n < 2 * k + 3:
        return [pts]
    a = pts[k:-k] - pts[:-2 * k]
    b = pts[2 * k:] - pts[k:-k]
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    cuts, last = [], 0
    for i in np.argsort(-turn):
        if turn[i] < corner_deg:
            break
        idx = i + k
        if all(abs(idx - c) > k for c in cuts):
            cuts.append(idx)
    pieces = []
    for c in sorted(cuts):
        pieces.append(pts[last:c + 1])
        last = c
    pieces.append(pts[last:])
    return [p for p in pieces if len(p) >= 2]


def fit_line(pts):
    """Least-squares line: returns (centre, unit direction, max deviation, t range)."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    u = vt[0]
    t = (pts - c) @ u
    dev = np.abs((pts - c) @ np.array([-u[1], u[0]]))
    return c, u, dev.max(), (t.min(), t.max())


def classify(paths, p):
    """Split traced paths into straight segments [(p0, p1)] and curves [pts]."""
    straights, curves = [], []
    for path in paths:
        for piece in split_at_corners(smooth_path(path), p.corner_deg):
            length = path_length(piece)
            if length < 3:
                continue
            c, u, dev, (t0, t1) = fit_line(piece)
            if dev <= p.straight_tol + 0.01 * length:
                straights.append(np.array([c + t0 * u, c + t1 * u]))
                continue
            core = straight_core(piece, p)
            poly = as_polygon(piece, p) if core is None else [core]
            if poly is not None:
                straights += poly
            else:
                curves.append(piece)   # dense: smoothing comes later, simplifying last
    return straights, curves


def straight_core(pts, p, min_core=0.6):
    """A straight edge whose ends curl into rounded corners (the top of a
    panel) -> one straight stroke spanning it. Corner joining later squares
    it up with its neighbours."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = s[-1]
    if total < 8:
        return None
    for trim in np.arange(0.05, (1 - min_core) / 2 + 1e-9, 0.05):
        mid = pts[(s >= trim * total) & (s <= (1 - trim) * total)]
        if len(mid) < 3:
            return None
        c, u, dev, _ = fit_line(mid)
        if dev <= p.straight_tol:
            # The trimmed ends must turn sharply away (a rounded corner), not
            # drift gently (an arch, whose middle can also look straight).
            for end, inner in ((pts[0], mid[0]), (pts[-1], mid[-1])):
                d = end - inner
                n = np.linalg.norm(d)
                if n > 1.5 and abs(d @ u) / n > np.cos(np.radians(35)):
                    return None
            t = (pts - c) @ u
            return np.array([c + t.min() * u, c + t.max() * u])
    return None


def as_polygon(pts, p, min_side=4.0, min_turn=50.0):
    """A curve that is really a few straight sides with rounded corners (a
    window panel, a door) -> its straight sides, squared off. Real curves
    (arches) need many small turns to approximate and are left alone."""
    v = simplify(pts.astype(np.float32), max(1.2, p.straight_tol)).astype(float)
    # A rounded corner simplifies to a short bevel: replace it with the
    # sharp corner where the neighbouring sides meet.
    k = 1
    while 1 <= k < len(v) - 2:
        if np.linalg.norm(v[k + 1] - v[k]) < min_side:
            a0, a1, b0, b1 = v[k - 1], v[k], v[k + 1], v[k + 2]
            da, db = a1 - a0, b1 - b0
            den = da[0] * db[1] - da[1] * db[0]
            if abs(den) > 1e-6:
                t = ((b0 - a0)[0] * db[1] - (b0 - a0)[1] * db[0]) / den
                x = a0 + t * da
                if np.linalg.norm(x - (a1 + b0) / 2) < 2 * min_side:
                    v = np.vstack([v[:k], x, v[k + 2:]])
                    continue
        k += 1
    if len(v) < 3 or len(v) > 6:
        return None
    d = np.diff(v, axis=0)
    sides = np.linalg.norm(d, axis=1)
    if sides.min() < min_side:
        return None
    cos = np.sum(d[:-1] * d[1:], 1) / (sides[:-1] * sides[1:])
    if np.degrees(np.arccos(np.clip(cos, -1, 1))).min() < min_turn:
        return None
    return [np.array([v[k], v[k + 1]]) for k in range(len(v) - 1)]


def merge_collinear(segs, kinds, p, rounds=3):
    """Merge straight segments that lie on the same line with small gaps or
    overlaps. Rebuilds broken building edges into single strokes and folds
    near-duplicate parallel strokes together. A merged stroke is a contour if
    any of its parts was."""
    for _ in range(rounds):
        if len(segs) < 2:
            break
        S = np.array(segs)                                # (n, 2, 2)
        d = S[:, 1] - S[:, 0]
        L = np.linalg.norm(d, axis=1) + 1e-9
        u = d / L[:, None]
        nrm = np.stack([-u[:, 1], u[:, 0]], 1)
        # Pairwise: angle, sideways offset of j's ends from i's line, gap along i.
        same_dir = np.abs(u @ u.T) > np.cos(np.radians(p.merge_deg))
        rel0 = S[None, :, 0] - S[:, None, 0]              # [i, j] = j.start - i.start
        rel1 = S[None, :, 1] - S[:, None, 0]
        off = np.maximum(np.abs(np.einsum("ijk,ik->ij", rel0, nrm)),
                         np.abs(np.einsum("ijk,ik->ij", rel1, nrm)))
        t0 = np.einsum("ijk,ik->ij", rel0, u)
        t1 = np.einsum("ijk,ik->ij", rel1, u)
        lo, hi = np.minimum(t0, t1), np.maximum(t0, t1)
        gap = np.maximum(lo - L[:, None], -hi)            # <0 means overlap
        ok = same_dir & (off < p.merge_offset) & (gap < p.merge_gap)
        ok &= ok.T
        np.fill_diagonal(ok, False)
        if not ok.any():
            break
        # Union-find over mergeable pairs.
        parent = list(range(len(segs)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, j in zip(*np.nonzero(np.triu(ok))):
            parent[find(i)] = find(j)
        groups = {}
        for i in range(len(segs)):
            groups.setdefault(find(i), []).append(i)
        merged, merged_kinds = [], []
        for idx in groups.values():
            merged_kinds.append("contour" if any(kinds[i] == "contour" for i in idx) else "line")
            if len(idx) == 1:
                merged.append(S[idx[0]])
                continue
            # Length-weighted fit through all endpoints, extent = full span.
            pts = S[idx].reshape(-1, 2)
            w = np.repeat(L[idx], 2)
            c = (pts * w[:, None]).sum(0) / w.sum()
            _, _, vt = np.linalg.svd((pts - c) * np.sqrt(w)[:, None], full_matrices=False)
            uu = vt[0]
            t = (pts - c) @ uu
            merged.append(np.array([c + t.min() * uu, c + t.max() * uu]))
        segs, kinds = merged, merged_kinds
    return list(segs), list(kinds)


def _homog_lines(S):
    """Homogeneous line through each segment's endpoints."""
    a = np.hstack([S[:, 0], np.ones((len(S), 1))])
    b = np.hstack([S[:, 1], np.ones((len(S), 1))])
    l = np.cross(a, b)
    return l / (np.linalg.norm(l[:, :2], axis=1, keepdims=True) + 1e-12)


def _vp_error(v, mids, dirs):
    """Angle (deg) between each segment and the direction from its midpoint
    to vanishing point v (homogeneous, so v at infinity is a direction)."""
    to = v[None, :2] - mids * v[2]
    cos = np.abs(np.sum(to * dirs, 1)) / (np.linalg.norm(to, axis=1) * np.linalg.norm(dirs, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def vanishing_points(segs, p, rng, min_len=25.0, inlier_deg=1.5, iters=600):
    """RANSAC the dominant perspective directions: points where many long
    lines meet (or directions, for lines that stay parallel)."""
    S = np.array(segs, float)
    d = S[:, 1] - S[:, 0]
    L = np.linalg.norm(d, axis=1)
    mids = S.mean(1)
    lines = _homog_lines(S)
    remaining = L >= min_len
    centre = mids.mean(0)
    vps = []
    for _ in range(p.vp_count):
        idx = np.flatnonzero(remaining)
        if len(idx) < 4:
            break
        w = L[idx] / L[idx].sum()
        best, best_score = None, 0.0
        for _ in range(iters):
            a, b = rng.choice(idx, 2, replace=False, p=w)
            v = np.cross(lines[a], lines[b])
            if not np.any(v):
                continue
            v = v / np.linalg.norm(v)
            # Must be a different direction (seen from the image centre) from
            # the ones already found, or the dominant one gets found twice.
            if any(_vp_error(v, centre[None], (u[:2] - centre * u[2])[None])[0] < 8 for u in vps):
                continue
            score = L[idx][_vp_error(v, mids[idx], d[idx]) < inlier_deg].sum()
            if score > best_score:
                best, best_score = v, score
        if best is None:
            break
        vps.append(best)
        remaining[idx[_vp_error(best, mids[idx], d[idx]) < 3 * inlier_deg]] = False
    return vps


def snap_to_vps(segs, vps, snap_deg, min_len=10.0):
    """Rotate each line about its midpoint to aim exactly at the vanishing
    point it nearly aims at already. Parallel edges then converge properly."""
    if not vps:
        return segs
    out = []
    for seg in segs:
        a, b = seg
        d = b - a
        L = np.linalg.norm(d)
        m = (a + b) / 2
        errs = [(_vp_error(v, m[None], d[None])[0], v) for v in vps]
        err, v = min(errs, key=lambda e: e[0])
        if L < min_len or err > snap_deg:
            out.append(seg)
            continue
        to = v[:2] - m * v[2]
        u = to / np.linalg.norm(to)
        if u @ d < 0:
            u = -u
        out.append(np.array([m - u * L / 2, m + u * L / 2]))
    return out


def join_corners(segs, p):
    """Move each line end to its intersection with a crossing line nearby, so
    corners meet exactly. Returns the segments and, per end, "L" (a corner
    where both lines end), "T" (ends against the other line's side) or None."""
    S = np.array(segs, float)
    n = len(S)
    ends = [[None, None] for _ in range(n)]
    if n < 2 or p.join_reach <= 0:
        return list(S), ends
    d = S[:, 1] - S[:, 0]
    L = np.linalg.norm(d, axis=1) + 1e-9
    u = d / L[:, None]
    reaches = np.minimum(np.clip(0.35 * L, p.join_reach, 2.5 * p.join_reach), 0.5 * L)
    new = S.copy()
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    cos_lim = np.cos(np.radians(p.join_deg))
    for i in range(n):
        # Long lines (roof slopes) may reach further: a finial or a gutter
        # often hides the last bit of a long edge.
        reach = reaches[i]
        for e in (0, 1):
            P = S[i, e]
            out = u[i] if e else -u[i]
            # Candidates: steep enough to make a corner, and near this end.
            rel = P - S[:, 0]
            along = np.clip(np.sum(rel * u, 1), 0, L)
            near = np.linalg.norm(rel - along[:, None] * u, axis=1) < reach * 1.5
            cand = np.flatnonzero(near & (np.abs(u @ u[i]) < cos_lim))
            cand = cand[cand != i]
            if not len(cand):
                continue
            denom = cross(out, u[cand])
            q = S[cand, 0] - P
            t = cross(q, u[cand]) / denom           # along this line, past the end
            s = cross(q, out) / denom               # along the other line
            past = np.maximum(0, np.maximum(-s, s - L[cand]))
            # Reaching far is only believable where both lines stop short of a
            # shared corner (a gable apex behind a finial). Running into the
            # side of another line, only close a small gap.
            at_end_j = (s < reaches[cand]) | (s > L[cand] - reaches[cand])
            # ...and for that, the two lines' ends must actually be close.
            j_end = np.where((s < L[cand] / 2)[:, None], S[cand, 0], S[cand, 1])
            at_end_j &= np.linalg.norm(j_end - P, axis=1) <= reach
            lim = np.where(at_end_j, reach, min(reach, 0.6 * p.join_reach))
            ok = (t <= lim) & (t >= -0.6 * reach) & (past <= np.maximum(reach, reaches[cand]))
            if not ok.any():
                continue
            cost = np.where(ok, np.abs(t) + past, np.inf)
            k = int(np.argmin(cost))
            j = cand[k]
            new[i, e] = P + t[k] * out
            at_end = s[k] < reaches[j] or s[k] > L[j] - reaches[j]
            ends[i][e] = "L" if at_end else "T"
    return list(new), ends


# ---------------------------------------------------------------- importance

def contrast_map(flat):
    """0..1 map of how strongly something is happening at each pixel: the
    larger of edge gradient and thin-bar response."""
    g = cv2.GaussianBlur(flat, (0, 0), 1.5).astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    ridge, _ = ridge_response(flat, (1.5, 2.5))
    c = np.maximum(grad / np.percentile(grad, 99), ridge / np.percentile(ridge, 99))
    # Strokes sit a pixel or two off the true peak, so sample a local max.
    return cv2.dilate(np.clip(c, 0, 1), np.ones((5, 5), np.uint8))


def focal_point(contrast):
    """Centre of mass of strong detail: where the drawing should be densest."""
    w = gaussian_filter(contrast, 15) ** 2
    ys, xs = np.mgrid[0:w.shape[0], 0:w.shape[1]]
    return np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])


def vignette_field(shape, centre, scale, ground, rng, sky=1.0):
    """>1 outside the drawn area. Elliptical around the focal point, with a
    soft random edge so strokes peter out raggedly, as when a sketcher stops.
    Sketchers lose interest in the ground long before the sky edge of the
    subject, so the ellipse is shorter below the focus (`ground` < 1)."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    # Reach to whichever image edge is further, so an off-centre focus
    # doesn't leave one side of the subject undrawn.
    rx = scale * 1.05 * max(centre[0], w - centre[0])
    ry = scale * 1.0 * max(centre[1], h - centre[1])
    ry = np.where(ys > centre[1], ry * ground, ry * sky)
    r = np.sqrt(((xs - centre[0]) / rx) ** 2 + ((ys - centre[1]) / ry) ** 2)
    noise = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 25)
    noise /= np.abs(noise).max() + 1e-9
    return r + 0.18 * noise


def sample(field, pts):
    h, w = field.shape
    x = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    return field[y, x]


def tone_separation(blurred, pts, reach=10.0):
    """How different the tone is on the two sides of a stroke (0..1). High
    for silhouettes (roof against sky), low for a bar inside a panel."""
    if len(pts) < 2:
        return 0.0
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    a = sample(blurred, pts + nrm * reach)
    b = sample(blurred, pts - nrm * reach)
    return float(np.abs(a - b).mean())


def indicate(strokes, rng, chance, spacing=14.0):
    """Suggest repeating patterns instead of drawing every element: inside a
    run of closely spaced parallel strokes, skip some at random, never the
    outermost ones, so the run still reads as complete."""
    straight = [i for i, x in enumerate(strokes) if x["straight"]]
    if chance <= 0 or len(straight) < 3:
        return strokes
    S = np.array([strokes[i]["geom"] for i in straight])
    mid = S.mean(1)
    d = S[:, 1] - S[:, 0]
    u = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    nrm = np.stack([-u[:, 1], u[:, 0]], 1)
    drop = set()
    for k, i in enumerate(straight):
        rel = mid - mid[k]
        parallel = np.abs(u @ u[k]) > np.cos(np.radians(8))
        side = rel @ nrm[k]
        along = np.abs(rel @ u[k])
        near = parallel & (along < 0.5 * np.linalg.norm(d[k])) & (np.abs(side) < spacing)
        near[k] = False
        # Neighbours on both sides = inside a run, not at its edge.
        if (near & (side > 0)).any() and (near & (side < 0)).any() and rng.random() < chance:
            drop.add(i)
    return [x for i, x in enumerate(strokes) if i not in drop]


def resample(pts, step):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return pts
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)


def clip_to_field(pts, field, limit=1.0):
    """Cut a stroke wherever it leaves the drawn area; returns the inside runs
    (a run that reaches an original end keeps that end's exact position)."""
    dense = resample(pts, 2.0)
    inside = sample(field, dense) < limit
    runs, cur = [], []
    for q, ok in zip(dense, inside):
        if ok:
            cur.append(q)
        elif cur:
            runs.append(np.array(cur))
            cur = []
    if cur:
        runs.append(np.array(cur))
    return runs


# ---------------------------------------------------------------- the hand

def hand_line(pts, rng, wobble, bow=True):
    """Resample a stroke and add a gentle bow plus low-frequency wobble
    perpendicular to it."""
    dense = resample(pts, 3.0)
    n = len(dense)
    if n < 3:
        return dense
    tang = np.gradient(dense, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    length = path_length(dense)
    s = np.linspace(0, 1, n)
    offset = np.zeros(n)
    if bow:
        offset += rng.normal(0, 0.004) * length * 4 * s * (1 - s)
    for _ in range(2):
        cycles = rng.uniform(0.5, 2.0) * max(1.0, length / 120)
        offset += wobble * rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * cycles * s + rng.uniform(0, 2 * np.pi))
    # Pin the ends a little so wobble doesn't flick the tips.
    offset *= np.minimum(1, np.minimum(s, 1 - s) * 8 + 0.3)
    return simplify((dense + normal * offset[:, None]).astype(np.float32), 0.25)


def hand_straight(seg, rng, p, ends=(None, None)):
    a, b = seg
    d = b - a
    L = np.linalg.norm(d)
    u = d / L

    def ext(kind):
        if kind == "L":   # joined corner: cross through it, architect-style
            return rng.uniform(*p.corner_over) * min(1.0, L / 30)
        if kind == "T":   # ends on another line's side: just touch it
            return rng.uniform(0, 0.4 * p.corner_over[0])
        # Free end: overshoot a little, or now and then stop short.
        return rng.uniform(-0.3, 1.0) * min(p.overshoot, 0.12 * L)
    a, b = a - u * ext(ends[0]), b + u * ext(ends[1])
    # A fraction of a degree off true, about the midpoint.
    ang = np.radians(rng.normal(0, p.angle_jitter))
    c, s = np.cos(ang), np.sin(ang)
    m = (a + b) / 2
    R = np.array([[c, -s], [s, c]])
    a, b = m + R @ (a - m), m + R @ (b - m)
    return hand_line(np.array([a, b]), rng, p.wobble)


def reinforce(stroke, rng, offset=0.8):
    """A second pass over a heavy line: shifted sideways a touch, starting a
    little late and finishing a little early."""
    n = len(stroke)
    if n < 2:
        return None
    i0, i1 = int(n * rng.uniform(0, 0.08)), n - int(n * rng.uniform(0, 0.08))
    part = stroke[i0:max(i1, i0 + 2)]
    d = part[-1] - part[0]
    nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    return hand_line(part + nrm * offset * rng.choice([-1, 1]), rng, 0.4, bow=False)


# ---------------------------------------------------------------- pipeline

def stylise(contour_paths, line_paths, flat, p=StyleParams()):
    """Returns {"heavy": [...], "medium": [...], "fine": [...]} stroke lists,
    plus "field": the vignette (>1 = outside the drawing), for later stages."""
    rng = np.random.default_rng(p.seed)
    contrast = contrast_map(flat)
    focus = focal_point(contrast)
    field = vignette_field(flat.shape, focus, p.vignette, p.ground_fade, rng, p.sky_reach)
    diag = np.hypot(*flat.shape)
    blurred = cv2.GaussianBlur(flat, (0, 0), 5).astype(np.float32) / 255

    strokes = []  # dicts: geom, straight, kind
    for kind, paths in (("contour", contour_paths), ("line", line_paths)):
        s, c = classify(paths, p)
        strokes += [{"geom": g, "straight": True, "kind": kind} for g in s]
        strokes += [{"geom": g, "straight": False, "kind": kind} for g in c]

    # Merge across both kinds: a bar and an edge on the same line are one stroke.
    straight = [x for x in strokes if x["straight"]]
    segs, kinds = merge_collinear([x["geom"] for x in straight], [x["kind"] for x in straight], p)
    # Perspective: aim lines at their vanishing points, merge again now that
    # they agree, then make corners meet.
    if p.vp_count and segs:
        segs = snap_to_vps(segs, vanishing_points(segs, p, rng), p.vp_snap_deg)
        segs, kinds = merge_collinear(segs, kinds, p, rounds=1)
    segs, ends = join_corners(segs, p)
    strokes = [x for x in strokes if not x["straight"]] + [
        {"geom": g, "straight": True, "kind": k, "ends": e} for g, k, e in zip(segs, kinds, ends)]

    if not p.hand:
        out = {"heavy": [], "fine": [], "field": field, "medium": [
            x["geom"] if x["straight"] else simplify(smooth_path(x["geom"], p.curve_smooth), 0.3)
            for x in strokes if path_length(x["geom"]) >= p.min_len]}
        return out

    # Clip to the vignette, then score what's left.
    clipped = []
    for x in strokes:
        for run in clip_to_field(x["geom"], field):
            if path_length(run) < p.min_len:
                continue
            y = {**x}
            if x["straight"]:
                g = x["geom"]
                # Ends cut by the vignette are free; untouched ends keep their joins.
                same = [np.linalg.norm(run[0] - g[0]) < 2.5, np.linalg.norm(run[-1] - g[1]) < 2.5]
                y["geom"] = np.array([g[0] if same[0] else run[0], g[1] if same[1] else run[-1]])
                y["ends"] = [x["ends"][0] if same[0] else None, x["ends"][1] if same[1] else None]
            else:
                y["geom"] = run
            clipped.append(y)
    for x in clipped:
        pts = resample(x["geom"], 2.0)
        L = path_length(pts)
        x["len"] = L
        x["contrast"] = float(sample(contrast, pts).mean())
        x["reach"] = float(sample(field, pts).max())
        off = pts.mean(0) - focus
        off[1] /= p.ground_fade if off[1] > 0 else 1   # the ground matters less
        dist = np.linalg.norm(off) / diag
        x["focus"] = 0.4 + 0.6 * np.exp(-(dist / 0.35) ** 2)
        x["score"] = x["contrast"] * np.sqrt(L) * x["focus"]
        x["sep"] = tone_separation(blurred, pts)
    if clipped:
        cut = np.percentile([x["score"] for x in clipped], p.drop_pct)
        clipped = [x for x in clipped if x["score"] >= cut]
    clipped = indicate(clipped, rng, p.indicate)

    # Weight hierarchy: heavy goes to long lines with very different tone on
    # either side, i.e. silhouettes and major shadow boundaries.
    heavy_key = lambda x: x["sep"] * np.sqrt(x["len"]) * x["focus"]
    can_be_heavy = lambda x: x["len"] >= p.heavy_min_len and x["reach"] <= p.heavy_reach
    cands = [x for x in clipped if can_be_heavy(x)]
    share = min(100.0, p.heavy_pct * len(clipped) / max(len(cands), 1))
    heavy_cut = np.percentile([heavy_key(x) for x in cands], 100 - share) if cands else np.inf
    out = {"heavy": [], "medium": [], "fine": [], "field": field}
    for x in clipped:
        if can_be_heavy(x) and heavy_key(x) >= heavy_cut:
            w = "heavy"
        elif x["kind"] == "line" and (x["len"] < p.fine_len or x["contrast"] < 0.5):
            w = "fine"
        else:
            w = "medium"
        if x["straight"]:
            stroke = hand_straight(x["geom"], rng, p, x["ends"])
        else:
            stroke = hand_line(smooth_path(x["geom"], p.curve_smooth), rng, p.wobble * 0.7)
        out[w].append(stroke)
        if w == "heavy":
            second = reinforce(stroke, rng)
            if second is not None:
                out[w].append(second)
    return out
