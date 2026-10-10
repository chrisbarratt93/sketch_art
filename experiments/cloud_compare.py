"""Nano Banana Pro experiments: model comparisons and painting over klein's lines (--lines).

For the plain styles, prefer `sketch.py --model nanobanana`, which also cleans
the photo first. Generation, review and retries live in sketchart/nanobanana.py.

Same reference photo and prompt as klein's default, sent to Nano Banana Pro
(Gemini 3 Pro Image) through the local ComfyUI, billed to your Comfy account
credits. The API key is read from ~/.config/sketch_art/comfy_api_key and sent
to ComfyUI in the request body only.

    uv run --all-extras experiments/cloud_compare.py out/gen/inputs/cotswold_street_clean.jpg

Every result is checked by Gemini 3 Pro against the photo (REVIEW) for invented
text or objects, missing scenery and stray blots. If it finds any, the image is
regenerated with a new seed and told what to avoid, up to --attempts times, and
the cleanest attempt is kept. Each review costs a few cents; each repaint costs
a full image.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sketchart import klein, nanobanana  # noqa: E402
from sketchart.output import render, write_svg  # noqa: E402
from sketchart.trace import travel_stats  # noqa: E402
from sketchart.vectorise import vectorise  # noqa: E402


# With --lines: paint over klein's ink drawing (image 1), colours from the photo (image 2).
OVER_LINES = (
    "Image 1 is a detailed pen and ink drawing. Image 2 is the photo it was drawn from. Turn image 1 "
    "into a finished pen, ink and watercolour painting by painting transparent watercolour washes "
    "over it. Keep every ink line of image 1 exactly as drawn: do not redraw, simplify, move or "
    "remove any line, and keep its exact composition and framing. Take the colours from image 2. "
    "{notes}"
    "Washes loose and transparent with soft wet-in-wet edges, gentle granulation and a little colour "
    "straying past the lines, so every ink line stays clearly visible. Leave the sky as bare white "
    "paper or the faintest wash, and let the colour fade out to white watercolour paper towards the "
    "edges of the picture. " + klein.FAITHFUL)
# Scene notes used for the Cotswold street test (2026-10-08).
COTSWOLD_NOTES = (
    "Warm honey Cotswold stone, grey stone roof tiles, green lawns, varied greens for the trees and "
    "hedges, the colours of the flowers. Keep the wooded hillside behind the village and paint it in "
    "soft, slightly blue-green washes that recede into the distance; it must stay in the picture. ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--resolution", default="2K", choices=("1K", "2K", "4K"))
    ap.add_argument("--prompt", default="urban_rich")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lines", default=None, help="klein ink drawing to paint over (uses OVER_LINES)")
    ap.add_argument("--notes", default=COTSWOLD_NOTES, help="with --lines: scene-specific colour notes")
    ap.add_argument("--ink-opacity", type=float, default=0.3,
                    help="with --lines: strength of klein's ink laid back over the paint (1 = full black)")
    ap.add_argument("--attempts", type=int, default=3,
                    help="most images to generate while the review still finds faults (1 = no retries)")
    ap.add_argument("--no-review", action="store_true", help="skip the Gemini fault check")
    ap.add_argument("-o", "--out", default="out/cloud")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.lines:
        prompt = OVER_LINES.format(notes=args.notes.strip() + " ")
        tag = "over_klein_lines"
    else:
        prompt = klein.PROMPTS.get(args.prompt, args.prompt) + " " + klein.FAITHFUL
        tag = args.prompt
    painted = args.prompt in klein.PAINTED or args.lines
    name = out / f"{Path(args.photo).stem.replace('_clean', '')}_nanobananapro_{tag}_{args.resolution}"

    t0 = time.time()
    params = nanobanana.NanoBananaParams(resolution=args.resolution, attempts=args.attempts,
                                         review=not args.no_review)
    # Drawing first (image 1), photo second.
    images = [args.lines, args.photo] if args.lines else [args.photo]
    result = nanobanana.draw(args.photo, args.seed, Path(f"{name}_raw.png"), params, images=images, prompt_text=prompt)
    for a in result["attempts"]:
        print(json.dumps(a), flush=True)
    Path(f"{name}_review.json").write_text(json.dumps(result, indent=1))
    best = next(a for a in result["attempts"] if a["file"] == result["kept"])
    raw = Path(f"{name}_raw.png")

    if painted:
        # A painting for print: keep its colour, no plotter strokes.
        paint = cv2.imread(str(raw))
        raw.replace(f"{name}.png")
        if args.lines:
            # The model repaints its own, lighter line work, so lay klein's ink back over the
            # washes (multiply, like ink over watercolour). The two line up within a few pixels.
            ink = cv2.resize(cv2.imread(args.lines, cv2.IMREAD_GRAYSCALE), paint.shape[1::-1],
                             interpolation=cv2.INTER_CUBIC)
            ink = 1 - args.ink_opacity * (1 - ink[..., None] / 255.0)
            cv2.imwrite(f"{name}_inked.png", (paint * ink).clip(0, 255).astype("uint8"))
        print(json.dumps({"painting": f"{name}.png", "kept": result["kept"], "faults": len(best["problems"]),
                          "seconds": round(time.time() - t0, 1)}))
        return
    bw = klein.black_and_white(cv2.imread(str(raw)))
    cv2.imwrite(f"{name}.png", bw)
    size, layers, _ = vectorise(bw)
    write_svg(f"{name}.svg", size, layers)
    cv2.imwrite(f"{name}_preview.png", render(size, layers))
    print(json.dumps({"drawing": f"{name}.png", "kept": result["kept"], "faults": len(best["problems"]),
                      "seconds": round(time.time() - t0, 1), "size": list(size),
                      **travel_stats([p for _, _, p in layers])}))


if __name__ == "__main__":
    main()
