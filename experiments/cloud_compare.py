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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--model", default="gemini-3-pro-image-preview")
    ap.add_argument("--resolution", default="2K", choices=("1K", "2K", "4K"))
    ap.add_argument("--prompt", default="urban_rich")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out", default="out/cloud")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    graph = {
        "load": {"class_type": "LoadImage", "inputs": {"image": klein._upload(args.photo)}},
        "gen": {"class_type": "GeminiImage2Node", "inputs": {
            "prompt": klein.PROMPTS.get(args.prompt, args.prompt), "model": args.model, "seed": args.seed,
            "aspect_ratio": "auto", "resolution": args.resolution, "response_modalities": "IMAGE",
            "images": ["load", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["gen", 0], "filename_prefix": "sketch_art/cloud"}},
    }
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
    if args.prompt in klein.PAINTED:
        # A painting for print: keep its colour, no plotter strokes.
        raw.replace(f"{name}.png")
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
