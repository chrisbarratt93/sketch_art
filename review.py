"""Review harness: run ink-mode variants on the test photos, print metrics and
write contact sheets of fixed crops (photo | variant | variant ...) at print
zoom, so every change is judged on the same evidence.

    uv run review.py                                   # baseline only
    uv run review.py -v collapse4:collapse=4 -v c5:collapse=5,hi=0.5

Variants are `name:param=value,...` overriding InkParams. TEED maps are cached
in out/cache, so after the first run each variant takes seconds.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from sketchart.edges import load_gray, teed_centrelines, teed_probability
from sketchart.ink import InkParams, ink
from sketchart.metrics import summary
from sketchart.output import render

# Crops in working-image coordinates (max_side 1400): y0, x0, size.
PHOTOS = {
    "albert_hall": ("examples/albert_hall.jpg", {
        "tracery": (490, 350, 200), "lettering": (170, 180, 200),
        "stonework": (640, 330, 200), "skyline": (330, 690, 200)}),
    "market_hall": ("examples/market_hall.jpg", {
        "panels": (440, 400, 200), "gable": (60, 420, 200)}),
    "cotswold": ("examples/cotswold_street.webp", {
        "timber_gable": (330, 830, 200), "foliage": (420, 0, 200)}),
}
ZOOM = 2.5
OUT = Path("out/review")


def parse_variant(text):
    name, _, kv = text.partition(":")
    overrides = {}
    for item in filter(None, kv.split(",")):
        k, v = item.split("=")
        overrides[k] = type(getattr(InkParams(), k))(v)
    return name, overrides


def cached_prob(path, p):
    cache = Path("out/cache") / f"{Path(path).stem}_{p.max_side}_{p.detail}_{p.teed_scale}.npy"
    if cache.exists():
        return np.load(cache)
    color, _ = load_gray(path, p.max_side)
    prob = teed_probability(color, p.detail, p.teed_scale)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, prob)
    return prob


def label(im, text):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (len(text) * 10 + 10, 24), (255, 255, 255), -1)
    cv2.putText(im, text, (5, 17), 0, 0.5, (0, 0, 220), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--variant", action="append", default=[], help="name:param=value,...")
    ap.add_argument("--photos", default=",".join(PHOTOS), help="comma-separated subset")
    args = ap.parse_args()
    variants = [("baseline", {})] + [parse_variant(v) for v in args.variant]
    OUT.mkdir(parents=True, exist_ok=True)

    table = {}
    for key in args.photos.split(","):
        path, crops = PHOTOS[key]
        base = InkParams()
        prob = cached_prob(path, base)
        color, gray = load_gray(path, base.max_side)
        h, w = gray.shape
        ref = teed_centrelines(prob, base.lo, base.hi)   # the uncleaned trace
        renders = []
        for name, over in variants:
            p = replace(base, **over)
            s = ink(path, p, prob=prob)
            strokes = s["heavy"] + s["medium"] + s["fine"]
            table[f"{key}/{name}"] = summary(strokes, ref, base.detail)
            layers = [("heavy", 1.5, s["heavy"]), ("medium", 1.0, s["medium"]), ("fine", 0.6, s["fine"])]
            renders.append((name, cv2.cvtColor(render((w, h), layers, scale=3), cv2.COLOR_GRAY2BGR)))
        for crop, (y0, x0, n) in crops.items():
            zoom = lambda im: cv2.resize(im[y0:y0 + n, x0:x0 + n], None, fx=ZOOM, fy=ZOOM,
                                         interpolation=cv2.INTER_AREA)
            row = [label(zoom(color), "photo")] + [label(zoom(im), name) for name, im in renders]
            cv2.imwrite(str(OUT / f"{key}_{crop}.png"), np.hstack(row))

    cols = ["strokes", "length", "doubled", "wobble", "short", "loops", "invented", "missed"]
    print(f"{'':28s}" + "".join(f"{c:>9s}" for c in cols))
    for k, m in table.items():
        print(f"{k:28s}" + "".join(f"{m[c]:>9}" for c in cols))
    (OUT / "metrics.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
