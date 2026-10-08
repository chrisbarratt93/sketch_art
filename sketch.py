"""Photo -> pen-plotter line drawing.

Stage 1: extract contours and thin lines and trace them into strokes.
Stage 2: stylise those strokes so they read as hand-drawn ink.
Stage 3: add tone with hatching and cross-hatching.

    python sketch.py examples/market_hall.jpg -o out/market_hall
    python sketch.py examples/market_hall.jpg --stage 1    # raw foundation only
    python sketch.py examples/market_hall.jpg --stage 2    # line only, no tone
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sketchart import hub
from sketchart.edges import EdgeParams, extract
from sketchart.output import render, write_svg
from sketchart.ink import InkParams, ink
from sketchart.segment import GROUPS, preview as seg_preview
from sketchart.stylise import PRESETS, preset, stylise
from sketchart.tone import ToneParams, tone
from sketchart.trace import order, trace, travel_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=None, help="output prefix (default: out/<image name>)")
    ap.add_argument("--stage", type=int, choices=(1, 2, 3), default=2)
    ap.add_argument("--edges", choices=("classic", "teed", "teed+ridges"), default="classic",
                    help="edge source: classic (Canny + ridges) or learned TEED (needs `uv sync --extra learned`)")
    ap.add_argument("--style", choices=["ink"] + sorted(PRESETS), default="ink",
                    help="ink (default): trace the learned TEED edge map faithfully (needs `uv sync --extra learned`)")
    ap.add_argument("--max-side", type=int, default=1400,
                    help="downscale so the longest side is at most this (0 = full photo resolution)")
    hf = ap.add_argument_group("Hugging Face models (ink style; needs `uv sync --extra hub`, see MODELS.md)")
    hf.add_argument("--edge-model", choices=hub.NAMES, default="teed", help="edge map to trace (default: teed)")
    hf.add_argument("--model-scale", type=float, default=None,
                    help="run the edge model at this multiple of the photo size (default: per model)")
    hf.add_argument("--hi", type=float, default=None, help="strong-line threshold on the edge map, 0..1")
    hf.add_argument("--lo", type=float, default=None, help="weak-line threshold on the edge map, 0..1")
    hf.add_argument("--remove", default="",
                    help="comma list of scene layers to leave blank: " + ",".join(GROUPS))
    hf.add_argument("--seg-model", choices=("tiny", "large"), default="tiny", help="OneFormer size for --remove")
    hf.add_argument("--depth-fade", type=float, default=0.0,
                    help="0..1: lighten line weight with distance (Depth Anything V2)")
    ap.add_argument("--seed", type=int, default=1, help="hand-wobble seed (stages 2-3)")
    ap.add_argument("--debug", action="store_true", help="also write intermediate masks")
    args = ap.parse_args()

    prefix = Path(args.out or Path("out") / Path(args.image).stem)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.style == "ink":
        remove = tuple(r for r in args.remove.split(",") if r)
        if bad := set(remove) - set(GROUPS):
            ap.error(f"--remove: unknown layer(s) {', '.join(sorted(bad))}")
        s = ink(args.image, InkParams(max_side=args.max_side, edge_model=args.edge_model,
                                      model_scale=args.model_scale, hi=args.hi, lo=args.lo,
                                      remove=remove, seg_model=args.seg_model, depth_fade=args.depth_fade))
        h, w = s["gray"].shape
        layers = [("heavy", 1.5, order(s["heavy"])),
                  ("medium", 1.0, order(s["medium"])),
                  ("fine", 0.6, order(s["fine"]))]
        if args.stage == 3:
            t = tone(s["gray"], np.zeros((h, w), np.float32), ToneParams(seed=args.seed + 1))
            layers += [(f"hatch{i}", 0.6, order(strokes)) for i, strokes in enumerate(t["passes"], 1)]
        write_svg(f"{prefix}.svg", (w, h), layers)
        cv2.imwrite(f"{prefix}_preview.png", render((w, h), layers))
        if args.debug:
            prob = cv2.resize(s["prob"], (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"{prefix}_{args.edge_model}.png", (255 * (1 - prob)).astype(np.uint8))
            if s["layers"] is not None:
                cv2.imwrite(f"{prefix}_layers.png", seg_preview(s["color"], s["layers"]))
            if s["near"] is not None:
                cv2.imwrite(f"{prefix}_depth.png", (255 * s["near"]).astype(np.uint8))
        print(json.dumps({"size": [w, h], **travel_stats([p for _, _, p in layers])}))
        return

    e = extract(args.image, EdgeParams(method="teed" if args.edges.startswith("teed") else "classic",
                                       teed_ridges=args.edges == "teed+ridges", max_side=args.max_side))
    h, w = e["gray"].shape
    f = e["detail"]
    if args.stage == 1:
        layers = [
            ("contours", 1.4, order(trace(e["contours"], min_len=12, scale=f))),
            ("lines", 1.0, order(trace(e["lines"], min_len=10, scale=f))),
        ]
    else:
        # Dense, unsimplified paths: stylise does its own fitting.
        s = stylise(trace(e["contours"], min_len=4, eps=0, scale=f),
                    trace(e["lines"], min_len=4, eps=0, scale=f),
                    e["flat"], preset(args.style, seed=args.seed))
        layers = [("heavy", 1.6, order(s["heavy"])),
                  ("medium", 1.1, order(s["medium"])),
                  ("fine", 0.7, order(s["fine"]))]
        if args.stage == 3:
            t = tone(e["gray"], s["field"], ToneParams(seed=args.seed + 1))
            layers += [(f"hatch{i}", 0.6, order(strokes)) for i, strokes in enumerate(t["passes"], 1)]

    write_svg(f"{prefix}.svg", (w, h), layers)
    cv2.imwrite(f"{prefix}_preview.png", render((w, h), layers))
    if args.debug and args.stage == 3:
        shade = np.full((h, w), 255, np.uint8)
        for k, m in enumerate(t["regions"], 1):
            shade[m] = 255 - 70 * k
        cv2.imwrite(f"{prefix}_tone.png", np.hstack([(255 * (1 - t["dark"])).astype(np.uint8), shade]))
    if args.debug and args.stage >= 2:
        weights = np.full((h, w, 3), 255, np.uint8)
        for (_, _, paths), col in zip(layers, [(0, 0, 220), (40, 40, 40), (200, 140, 0)]):
            for pth in paths:
                cv2.polylines(weights, [np.round(pth).astype(np.int32)], False, col, 1, cv2.LINE_AA)
        cv2.imwrite(f"{prefix}_weights.png", weights)
    if args.debug:
        cv2.imwrite(f"{prefix}_flat.png", e["flat"])
        masks = np.full((h * f, w * f, 3), 255, np.uint8)
        masks[e["lines"]] = (200, 120, 0)
        masks[e["contours"]] = (0, 0, 220)
        cv2.imwrite(f"{prefix}_masks.png", masks)

    stats = travel_stats([p for _, _, p in layers])
    print(json.dumps({"size": [w, h], **stats}))


if __name__ == "__main__":
    main()
