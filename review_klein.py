"""Review harness for refining generated (klein) drawings.

Runs refine variants over the fixed drawings in examples/drawings, writes
crop sheets (original | variant | ...) at 1:1 drawing pixels to out/review_klein,
and prints metrics.

    uv run review_klein.py
    uv run review_klein.py -v smooth:smooth=2.5,steady=1 -v all:smooth=2.5,bridge=6,min_len=4,steady=1
"""

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path

import cv2
import numpy as np

from sketchart.metrics import summary
from sketchart.refine import RefineParams, refine, render

# Crops as fractions of the drawing (x, y); each crop is SIZE px square.
CROPS = {
    "albert_hall": {"tracery": (0.37, 0.42), "awning": (0.62, 0.78)},
    "market_hall": {"glazing": (0.55, 0.22), "base": (0.38, 0.62)},
    "cotswold_street": {"foliage": (0.02, 0.55), "timber": (0.58, 0.40), "roof": (0.16, 0.16)},
    "mill_pond": {"cottages": (0.55, 0.38), "reeds": (0.62, 0.62)},
    "park": {"tree": (0.12, 0.22), "portico": (0.50, 0.42)},
}
SIZE = 420
OUT = Path("out/review_klein")


def parse_variant(text):
    name, _, kv = text.partition(":")
    types = {f.name: f.type for f in fields(RefineParams)}
    over = {}
    for item in filter(None, kv.split(",")):
        k, v = item.split("=")
        t = types[k] if isinstance(types[k], type) else {"bool": bool, "int": int, "float": float}[types[k]]
        t = (lambda x: x not in ("0", "false", "False")) if t is bool else t
        over[k] = t(v)
    return name, over


def label(im, text):
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im.copy()
    cv2.rectangle(im, (0, 0), (len(text) * 10 + 10, 24), (255, 255, 255), -1)
    cv2.putText(im, text, (5, 17), 0, 0.5, (0, 0, 220), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--variant", action="append", default=[])
    ap.add_argument("--drawings", default=",".join(CROPS))
    args = ap.parse_args()
    variants = [("plain", {})] + [parse_variant(v) for v in args.variant]
    OUT.mkdir(parents=True, exist_ok=True)
    table = {}
    for key in args.drawings.split(","):
        gray = cv2.imread(f"examples/drawings/{key}_klein.png", cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        ref = gray < 128
        renders = []
        for name, over in variants:
            p = replace(RefineParams(), **over)
            r = refine(gray, p)
            strokes = [s for s, _ in r["strokes"]]
            im = render(r, p, source=gray)
            table[f"{key}/{name}"] = {**summary(strokes), "jitter": jitter(strokes), "diff": diff(im, gray)}
            renders.append((name, im))
        for crop, (fx, fy) in CROPS[key].items():
            x0, y0 = int(fx * w), int(fy * h)
            cut = lambda im: im[y0:y0 + SIZE, x0:x0 + SIZE]
            row = [label(cut(gray), "klein")] + [label(cut(im), n) for n, im in renders]
            cv2.imwrite(str(OUT / f"{key}_{crop}.png"), np.hstack(row))
    cols = ["strokes", "length", "short", "loops", "wobble", "jitter", "diff"]
    print(f"{'':28s}" + "".join(f"{c:>9s}" for c in cols))
    for k, m in table.items():
        print(f"{k:28s}" + "".join(f"{m[c]:>9}" for c in cols))
    (OUT / "metrics.json").write_text(json.dumps(table, indent=1))


def diff(a, b, sigma=1.5):
    """How different two drawings look (0 = same), comparing ink after a slight
    blur so a stroke moved by a pixel barely counts. % of full ink-vs-paper."""
    ga, gb = (cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma) for x in (a, b))
    return round(float(np.mean(np.abs(ga - gb))) / 255 * 100, 2)


def jitter(strokes, sigma=4.0):
    """Median high-frequency wander (px RMS) of strokes longer than 30px: how far
    each point sits from a smoothed copy of its stroke. A confident line is ~0."""
    from scipy.ndimage import gaussian_filter1d
    from sketchart.refine import _resample
    from sketchart.trace import path_length
    vals = []
    for s in strokes:
        if path_length(s) < 30:
            continue
        r = _resample(np.asarray(s, float))
        sm = np.stack([gaussian_filter1d(r[:, i], sigma, mode="nearest") for i in (0, 1)], 1)
        vals.append(np.sqrt(np.mean(np.sum((r - sm)[5:-5] ** 2, 1))))
    return round(float(np.median(vals)), 3) if vals else 0.0


if __name__ == "__main__":
    main()
