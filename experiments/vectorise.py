"""Raster ink drawing (e.g. from comfy_ink.py) -> plotter SVG. See sketchart/vectorise.py.

    uv run experiments/vectorise.py out/gen/albert_hall_klein_plotter_0.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sketchart.output import render, write_svg  # noqa: E402
from sketchart.trace import travel_stats  # noqa: E402
from sketchart.vectorise import vectorise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default=None, help="output folder (default: next to each image)")
    args = ap.parse_args()
    for p in map(Path, args.images):
        size, layers, fill = vectorise(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY))
        dst = Path(args.out or p.parent) / f"{p.stem}_plot"
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_svg(f"{dst}.svg", size, layers)
        cv2.imwrite(f"{dst}_preview.png", render(size, layers))
        print(json.dumps({"file": p.name, "filled": round(fill, 4), **travel_stats([l for _, _, l in layers])}))


if __name__ == "__main__":
    main()
