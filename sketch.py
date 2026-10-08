"""Photo -> pen-plotter line drawing.

Stage 1: extract contours and thin lines and trace them into strokes.
Stage 2: stylise those strokes so they read as hand-drawn ink.

    python sketch.py examples/market_hall.jpg -o out/market_hall
    python sketch.py examples/market_hall.jpg --stage 1    # raw foundation only
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sketchart.edges import EdgeParams, extract
from sketchart.output import render, write_svg
from sketchart.stylise import StyleParams, stylise
from sketchart.trace import order, trace, travel_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=None, help="output prefix (default: out/<image name>)")
    ap.add_argument("--stage", type=int, choices=(1, 2), default=2)
    ap.add_argument("--seed", type=int, default=1, help="hand-wobble seed (stage 2)")
    ap.add_argument("--debug", action="store_true", help="also write intermediate masks")
    args = ap.parse_args()

    prefix = Path(args.out or Path("out") / Path(args.image).stem)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    e = extract(args.image, EdgeParams())
    h, w = e["gray"].shape
    if args.stage == 1:
        layers = [
            ("contours", 1.4, order(trace(e["contours"], min_len=12))),
            ("lines", 1.0, order(trace(e["lines"], min_len=10))),
        ]
    else:
        # Dense, unsimplified paths: stylise does its own fitting.
        s = stylise(trace(e["contours"], min_len=4, eps=0), trace(e["lines"], min_len=4, eps=0),
                    e["flat"], StyleParams(seed=args.seed))
        layers = [("heavy", 1.6, order(s["heavy"])),
                  ("medium", 1.1, order(s["medium"])),
                  ("fine", 0.7, order(s["fine"]))]

    write_svg(f"{prefix}.svg", (w, h), layers)
    cv2.imwrite(f"{prefix}_preview.png", render((w, h), layers))
    if args.debug and args.stage == 2:
        weights = np.full((h, w, 3), 255, np.uint8)
        for (_, _, paths), col in zip(layers, [(0, 0, 220), (40, 40, 40), (200, 140, 0)]):
            for pth in paths:
                cv2.polylines(weights, [np.round(pth).astype(np.int32)], False, col, 1, cv2.LINE_AA)
        cv2.imwrite(f"{prefix}_weights.png", weights)
    if args.debug:
        cv2.imwrite(f"{prefix}_flat.png", e["flat"])
        masks = np.full((h, w, 3), 255, np.uint8)
        masks[e["lines"]] = (200, 120, 0)
        masks[e["contours"]] = (0, 0, 220)
        cv2.imwrite(f"{prefix}_masks.png", masks)

    stats = travel_stats([p for _, _, p in layers])
    print(json.dumps({"size": [w, h], **stats}))


if __name__ == "__main__":
    main()
