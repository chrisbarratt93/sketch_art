"""Photo -> pen-plotter ink drawing.

Default (--style klein): the photo is cleaned of clutter, FLUX.2 klein (in
ComfyUI) redraws it as a pen-and-ink urban sketch, the result is forced to
black and white, and traced into plotter strokes. Makes --count drawings, one
per seed, because results vary.

    uv run sketch.py examples/albert_hall.jpg            # -> out/albert_hall_s1.png/.svg, _s2, _s3
    uv run sketch.py examples/albert_hall.jpg --count 1 --mp 1 --prompt plotter
    uv run sketch.py examples/albert_hall.jpg --model nanobanana --prompt architect        # paid
    uv run sketch.py examples/cotswold_street.webp --model nanobanana --prompt watercolour_ink --resolution 4K

The classic tracer pipelines are still available:
Stage 1: extract contours and thin lines and trace them into strokes.
Stage 2: stylise those strokes so they read as hand-drawn ink.
Stage 3: add tone with hatching and cross-hatching.

    uv run sketch.py examples/market_hall.jpg --style ink
    uv run sketch.py examples/market_hall.jpg --style plain --stage 1    # raw foundation only
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sketchart.edges import EdgeParams, extract
from sketchart.output import render, write_svg
from sketchart.ink import InkParams, ink
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
    ap.add_argument("--style", choices=["klein", "ink"] + sorted(PRESETS), default="klein",
                    help="klein (default): FLUX.2 klein redraws the photo as ink in ComfyUI, then it is traced "
                         "for the plotter; ink: trace the learned TEED edge map faithfully "
                         "(both need `uv sync --extra learned --extra segment`)")
    ap.add_argument("--max-side", type=int, default=None,
                    help="downscale so the longest side is at most this (0 = full photo resolution; "
                         "default 1536 for klein's reference photo, 1400 otherwise)")
    ap.add_argument("--model", choices=("klein", "nanobanana"), default="klein",
                    help="klein style: who draws it. klein (default): FLUX.2 klein, local and free. nanobanana: "
                         "Nano Banana Pro, paid Comfy credits (~$0.13 per 2K image), follows styles much better; "
                         "needs ~/.config/sketch_art/comfy_api_key")
    ap.add_argument("--count", type=int, default=None,
                    help="klein style: drawings to make, one per seed from --seed (default 3 with klein, 1 with "
                         "nanobanana)")
    ap.add_argument("--prompt", default="urban_rich",
                    help="klein: pen styles urban_rich (default), architect, urban, plotter, engraving; painted styles "
                         "(colour PNG only, no plotter SVG) watercolour_ink, pen_and_wash, pencil; or your own text")
    ap.add_argument("--mp", type=float, default=4.0,
                    help="klein: drawing size in megapixels (4 = about 2400 x 1600, ~50 s each)")
    ap.add_argument("--resolution", choices=("1K", "2K", "4K"), default="2K",
                    help="nanobanana: output size (4K for large prints, ~$0.24 each)")
    ap.add_argument("--attempts", type=int, default=3,
                    help="nanobanana: most images per drawing while the fault review still finds problems "
                         "(1 = never retry; each retry is a paid image)")
    ap.add_argument("--no-review", action="store_true", help="nanobanana: skip the Gemini fault review")
    ap.add_argument("--scene", action="store_true",
                    help="ink: draw each scene layer (building, foliage, ground, sky) in its own way; "
                         "needs `uv sync --extra learned --extra segment`")
    ap.add_argument("--declutter", choices=("off", "inpaint", "drop"), default=None,
                    help="remove cranes, people, cars, street furniture: inpaint what's behind, or (ink only) "
                         "drop to paper. Default: inpaint for klein, off for ink (needs the segment extra)")
    ap.add_argument("--clutter", default=None,
                    help="comma-separated things to remove (default: sketchart.scene.CLUTTER)")
    ap.add_argument("--seed", type=int, default=1, help="hand-wobble seed (stages 2-3); first klein seed")
    ap.add_argument("--debug", action="store_true", help="also write intermediate masks")
    args = ap.parse_args()
    if args.max_side is None:
        args.max_side = 1536 if args.style == "klein" else 1400
    if args.count is None:
        args.count = 1 if args.model == "nanobanana" else 3
    if args.declutter is None:
        args.declutter = "inpaint" if args.style == "klein" else "off"
    if args.style == "klein" and args.declutter == "drop":
        ap.error("--declutter drop only applies to ink style; klein needs a whole photo (use inpaint or off)")
    clutter = tuple(c.strip() for c in args.clutter.split(",")) if args.clutter else None

    prefix = Path(args.out or Path("out") / Path(args.image).stem)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.style == "klein":
        klein_style(args, prefix, clutter)
        return

    if args.style == "ink":
        s = ink(args.image, InkParams(max_side=args.max_side, scene=args.scene,
                                      declutter=args.declutter, clutter=clutter))
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
            cv2.imwrite(f"{prefix}_teed.png", (255 * (1 - prob)).astype(np.uint8))
            if "layers" in s or "clutter" in s:
                from sketchart.scene import overlay
                regions = s.get("layers", np.full((h, w), 5, np.uint8))
                cv2.imwrite(f"{prefix}_scene.jpg", overlay(s["photo"], regions, s.get("clutter")))
            if args.declutter == "inpaint":
                cv2.imwrite(f"{prefix}_clean.jpg", s["photo"])
        extra = {"removed": sorted(set(s["clutter_labels"]))} if "clutter_labels" in s else {}
        print(json.dumps({"size": [w, h], **travel_stats([p for _, _, p in layers]), **extra}))
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


def klein_style(args, prefix, clutter):
    from sketchart import klein
    from sketchart.edges import load_gray
    from sketchart.vectorise import vectorise

    photo, _ = load_gray(args.image, args.max_side)
    removed = []
    if args.declutter == "inpaint":
        from sketchart import scene
        photo, mask, removed, regions = scene.declutter(photo, clutter)
        if args.debug:
            cv2.imwrite(f"{prefix}_scene.jpg", scene.overlay(photo, regions, mask))
    # The (cleaned) photo is klein's reference image; ComfyUI needs it as a file.
    ref = Path(f"{prefix}_clean.png")
    cv2.imwrite(str(ref), photo)
    params = klein.KleinParams(prompt=args.prompt, megapixels=args.mp)
    if args.model == "nanobanana":
        from sketchart import nanobanana
        nb = nanobanana.NanoBananaParams(prompt=args.prompt, resolution=args.resolution,
                                         attempts=args.attempts, review=not args.no_review)
    try:
        for seed in range(args.seed, args.seed + args.count):
            name = f"{prefix}_s{seed}"
            raw = Path(f"{name}_raw.png")
            info = {}
            if args.model == "nanobanana":
                # Seeds step by --attempts so retries never reuse another drawing's seed.
                r = nanobanana.draw(ref, args.seed + (seed - args.seed) * args.attempts, raw, nb)
                Path(f"{name}_review.json").write_text(json.dumps(r, indent=1))
                info = {"kept": r["kept"], "faults": len(next(a for a in r["attempts"] if a["file"] == r["kept"])["problems"])}
            else:
                klein.draw(ref, seed, raw, params)
            if args.prompt in klein.PAINTED:
                # A painting for print: keep its colour and tone, no plotter strokes.
                raw.replace(f"{name}.png")
                print(json.dumps({"painting": f"{name}.png", **info}), flush=True)
                continue
            bw = klein.black_and_white(cv2.imread(str(raw)))
            cv2.imwrite(f"{name}.png", bw)
            if not args.debug:
                raw.unlink()
            size, layers, _ = vectorise(bw)
            write_svg(f"{name}.svg", size, layers)
            cv2.imwrite(f"{name}_preview.png", render(size, layers))
            print(json.dumps({"drawing": f"{name}.png", "size": list(size), **info,
                              **travel_stats([p for _, _, p in layers]),
                              **({"removed": sorted(set(removed))} if removed else {})}), flush=True)
    finally:
        if not args.debug:
            ref.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
