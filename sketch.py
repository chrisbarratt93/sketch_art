"""Photo -> pen-plotter line drawing.

Stage 1: extract contours and thin lines and trace them into strokes.

    python sketch.py examples/market_hall.jpg -o out/market_hall
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sketchart.edges import EdgeParams, extract
from sketchart.output import render, write_svg
from sketchart.trace import order, trace, travel_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=None, help="output prefix (default: out/<image name>)")
    ap.add_argument("--debug", action="store_true", help="also write intermediate masks")
    args = ap.parse_args()

    prefix = Path(args.out or Path("out") / Path(args.image).stem)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    e = extract(args.image, EdgeParams())
    h, w = e["gray"].shape
    layers = [
        ("contours", 1.4, order(trace(e["contours"], min_len=12))),
        ("lines", 1.0, order(trace(e["lines"], min_len=10))),
    ]

    write_svg(f"{prefix}.svg", (w, h), layers)
    cv2.imwrite(f"{prefix}_preview.png", render((w, h), layers))
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
