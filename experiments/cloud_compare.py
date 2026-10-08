"""One-off comparison: a cloud model (via ComfyUI's paid partner nodes) vs local klein.

Same reference photo and prompt as klein's default, sent to Nano Banana Pro
(Gemini 3 Pro Image) through the local ComfyUI, billed to your Comfy account
credits. The API key is read from ~/.config/sketch_art/comfy_api_key and sent
to ComfyUI in the request body only.

    uv run --all-extras experiments/cloud_compare.py out/gen/inputs/cotswold_street_clean.jpg
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sketchart import klein  # noqa: E402
from sketchart.output import render, write_svg  # noqa: E402
from sketchart.trace import travel_stats  # noqa: E402
from sketchart.vectorise import vectorise  # noqa: E402

KEY = Path.home() / ".config/sketch_art/comfy_api_key"

# With --lines: paint over klein's ink drawing (image 1), colours from the photo (image 2).
OVER_LINES = (
    "Image 1 is a detailed pen and ink drawing. Image 2 is the photo it was drawn from. Turn image 1 "
    "into a finished pen, ink and watercolour painting by painting transparent watercolour washes "
    "over it. Keep every ink line of image 1 exactly as drawn: do not redraw, simplify, move or "
    "remove any line, and keep its exact composition and framing. Take the colours from image 2: warm "
    "honey Cotswold stone, grey stone roof tiles, green lawns, varied greens for the trees and "
    "hedges, the colours of the flowers. Keep the wooded hillside behind the village and paint it in "
    "soft, slightly blue-green washes that recede into the distance; it must stay in the picture. "
    "Washes loose and transparent with soft wet-in-wet edges, gentle granulation and a little colour "
    "straying past the lines, so every ink line stays clearly visible. Leave the sky as bare white "
    "paper or the faintest wash, and let the colour fade out to white watercolour paper towards the "
    "edges of the picture.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--model", default="gemini-3-pro-image-preview")
    ap.add_argument("--resolution", default="2K", choices=("1K", "2K", "4K"))
    ap.add_argument("--prompt", default="urban_rich")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lines", default=None, help="klein ink drawing to paint over (uses OVER_LINES)")
    ap.add_argument("-o", "--out", default="out/cloud")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompt = OVER_LINES if args.lines else klein.PROMPTS.get(args.prompt, args.prompt)
    graph = {
        "load": {"class_type": "LoadImage", "inputs": {"image": klein._upload(args.photo)}},
        "gen": {"class_type": "GeminiImage2Node", "inputs": {
            "prompt": prompt, "model": args.model, "seed": args.seed,
            "aspect_ratio": "auto", "resolution": args.resolution, "response_modalities": "IMAGE",
            "images": ["load", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["gen", 0], "filename_prefix": "sketch_art/cloud"}},
    }
    if args.lines:
        # Drawing first (image 1), photo second; ImageBatch resizes the photo to match.
        graph["lines"] = {"class_type": "LoadImage", "inputs": {"image": klein._upload(args.lines)}}
        graph["batch"] = {"class_type": "ImageBatch", "inputs": {"image1": ["lines", 0], "image2": ["load", 0]}}
        graph["gen"]["inputs"]["images"] = ["batch", 0]
        args.prompt = "over_klein_lines"
    t0 = time.time()
    pid = klein._call("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4()),
                                  "extra_data": {"api_key_comfy_org": KEY.read_text().strip()}})["prompt_id"]
    while True:
        h = klein._call(f"/history/{pid}").get(pid)
        if h and h.get("status", {}).get("status_str") == "error":
            sys.exit("ComfyUI error: " + json.dumps(h["status"].get("messages"))[:2000])
        if h and h.get("status", {}).get("completed"):
            break
        time.sleep(3)
    img = next(i for o in h["outputs"].values() for i in o.get("images", []))
    name = out / f"{Path(args.photo).stem.replace('_clean', '')}_nanobananapro_{args.prompt}_{args.resolution}"
    raw = Path(f"{name}_raw.png")
    klein._call(f"/view?filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}", out=raw)
    if args.prompt in klein.PAINTED or args.lines:
        # A painting for print: keep its colour, no plotter strokes.
        raw.replace(f"{name}.png")
        if args.lines:
            # The model repaints its own, lighter line work, so lay klein's ink back over the
            # washes (multiply, like ink over watercolour). The two line up within a few pixels.
            paint = cv2.imread(f"{name}.png")
            ink = cv2.resize(cv2.imread(args.lines, cv2.IMREAD_GRAYSCALE), paint.shape[1::-1],
                             interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{name}_inked.png", (paint * (ink[..., None] / 255.0)).astype("uint8"))
        print(json.dumps({"painting": f"{name}.png", "seconds": round(time.time() - t0, 1)}))
        return
    bw = klein.black_and_white(cv2.imread(str(raw)))
    cv2.imwrite(f"{name}.png", bw)
    size, layers, _ = vectorise(bw)
    write_svg(f"{name}.svg", size, layers)
    cv2.imwrite(f"{name}_preview.png", render(size, layers))
    print(json.dumps({"drawing": f"{name}.png", "seconds": round(time.time() - t0, 1), "size": list(size),
                      **travel_stats([p for _, _, p in layers])}))


if __name__ == "__main__":
    main()
