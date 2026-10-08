"""One-off comparison: a cloud model (via ComfyUI's paid partner nodes) vs local klein.

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


REVIEW = (
    "Each image shows a reference photo on the left and an artwork made from it on the right. Image 1 "
    "is the whole scene; images 2 to 5 are close-ups of its top-left, top-right, bottom-left and "
    "bottom-right quarters, for checking small details. Simplification, drawing and painting style, "
    "colour changes, white unpainted sky and colour fading out to white paper at the edges are all "
    "intended: do not report them. Look carefully over walls, roofs, signs and water in every "
    "close-up, and report only these faults in the artwork: "
    "\"text\": letters, numbers, symbols, scribbled writing or lettering that is not clearly in the "
    "photo, or a sign or plaque whose content differs from the photo; "
    "\"object\": an object, figure or structure that is not in the photo; "
    "\"missing\": a building, tree, hill, river, reflection or other major element of the photo "
    "that is absent; "
    "\"blot\": a stray blot, smudge or splash of paint that does not depict anything in the photo. "
    "Reply with JSON only, no other text: "
    "{\"problems\": [{\"type\": \"text|object|missing|blot\", \"where\": \"...\", \"what\": \"...\"}]} "
    "with an empty list if there are none.")


def _panels(photo, art, width=2048):
    """Photo | artwork side by side: the whole scene, then four overlapping quarters."""
    art = cv2.imread(str(art))
    photo = cv2.resize(cv2.imread(str(photo)), art.shape[1::-1], interpolation=cv2.INTER_CUBIC)
    h, w = art.shape[:2]
    views = [(0, 0, w, h)] + [(x, y, int(w * 0.55), int(h * 0.55))
                              for y in (0, int(h * 0.45)) for x in (0, int(w * 0.45))]
    out = []
    for x, y, cw, ch in views:
        pair = cv2.hconcat([photo[y:y + ch, x:x + cw], art[y:y + ch, x:x + cw]])
        s = width / pair.shape[1]
        out.append(cv2.resize(pair, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    return out


def _run(graph, key):
    pid = klein._call("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4()),
                                  "extra_data": {"api_key_comfy_org": key}})["prompt_id"]
    while True:
        h = klein._call(f"/history/{pid}").get(pid)
        if h and h.get("status", {}).get("status_str") == "error":
            msgs = [m[1] for m in h["status"].get("messages", []) if m[0] == "execution_error"]
            raise RuntimeError("ComfyUI error: " + (msgs[0].get("exception_message", "") if msgs else "unknown"))
        if h and h.get("status", {}).get("completed"):
            return h["outputs"]
        time.sleep(3)


def review(photo, painting, key, model="gemini-3-pro-preview"):
    """Gemini's list of faults in `painting` compared with `photo` (both local files)."""
    graph, prev = {}, None
    for i, panel in enumerate(_panels(photo, painting)):
        f = Path(painting).with_suffix(f".review{i}.jpg")
        cv2.imwrite(str(f), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        graph[f"p{i}"] = {"class_type": "LoadImage", "inputs": {"image": klein._upload(f)}}
        f.unlink()
        if prev is None:
            prev = [f"p{i}", 0]
        else:
            graph[f"b{i}"] = {"class_type": "ImageBatch", "inputs": {"image1": prev, "image2": [f"p{i}", 0]}}
            prev = [f"b{i}", 0]
    graph["ask"] = {"class_type": "GeminiNode", "inputs": {"prompt": REVIEW, "model": model, "seed": 1,
                                                           "images": prev}}
    graph["show"] = {"class_type": "PreviewAny", "inputs": {"source": ["ask", 0]}}
    text = "".join(str(t) for o in _run(graph, key).values() for t in o.get("text", []))
    try:
        # Usually {"problems": [...]}, but Gemini sometimes answers with the bare list.
        start = min(i for i in (text.find("{"), text.find("[")) if i >= 0)
        end = max(text.rfind("}"), text.rfind("]")) + 1
        found = json.loads(text[start:end])
        return found["problems"] if isinstance(found, dict) else found
    except (ValueError, KeyError, TypeError):
        return [{"type": "review", "where": "", "what": f"unreadable review: {text[:200]}"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--model", default="gemini-3-pro-image-preview")
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
    key = KEY.read_text().strip()

    if args.lines:
        prompt = OVER_LINES.format(notes=args.notes.strip() + " ")
        tag = "over_klein_lines"
    else:
        prompt = klein.PROMPTS.get(args.prompt, args.prompt) + " " + klein.FAITHFUL
        tag = args.prompt
    painted = args.prompt in klein.PAINTED or args.lines
    name = out / f"{Path(args.photo).stem.replace('_clean', '')}_nanobananapro_{tag}_{args.resolution}"
    photo_name = klein._upload(args.photo)
    lines_name = klein._upload(args.lines) if args.lines else None

    t0 = time.time()
    attempts, avoid = [], ""
    for k in range(args.attempts):
        graph = {
            "load": {"class_type": "LoadImage", "inputs": {"image": photo_name}},
            "gen": {"class_type": "GeminiImage2Node", "inputs": {
                "prompt": prompt + avoid, "model": args.model, "seed": args.seed + k,
                "aspect_ratio": "auto", "resolution": args.resolution, "response_modalities": "IMAGE",
                "images": ["load", 0]}},
            "save": {"class_type": "SaveImage", "inputs": {"images": ["gen", 0], "filename_prefix": "sketch_art/cloud"}},
        }
        if lines_name:
            # Drawing first (image 1), photo second; ImageBatch resizes the photo to match.
            graph["lines"] = {"class_type": "LoadImage", "inputs": {"image": lines_name}}
            graph["batch"] = {"class_type": "ImageBatch", "inputs": {"image1": ["lines", 0], "image2": ["load", 0]}}
            graph["gen"]["inputs"]["images"] = ["batch", 0]
        img = next(i for o in _run(graph, key).values() for i in o.get("images", []))
        raw = Path(f"{name}_a{k + 1}.png")
        klein._call(f"/view?filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}", out=raw)
        problems = [] if args.no_review else review(args.photo, raw, key)
        attempts.append({"file": raw.name, "seed": args.seed + k, "problems": problems})
        print(json.dumps(attempts[-1]), flush=True)
        if not problems:
            break
        avoid = (" A previous attempt had these faults; make sure none of them appear: "
                 + "; ".join(f"{p.get('what', '')} ({p.get('where', '')})" for p in problems) + ".")
    best = min(attempts, key=lambda a: len(a["problems"]))
    Path(f"{name}_review.json").write_text(json.dumps({"kept": best["file"], "attempts": attempts}, indent=1))
    raw = out / best["file"]

    if painted:
        # A painting for print: keep its colour, no plotter strokes.
        paint = cv2.imread(str(raw))
        cv2.imwrite(f"{name}.png", paint)
        if args.lines:
            # The model repaints its own, lighter line work, so lay klein's ink back over the
            # washes (multiply, like ink over watercolour). The two line up within a few pixels.
            ink = cv2.resize(cv2.imread(args.lines, cv2.IMREAD_GRAYSCALE), paint.shape[1::-1],
                             interpolation=cv2.INTER_CUBIC)
            ink = 1 - args.ink_opacity * (1 - ink[..., None] / 255.0)
            cv2.imwrite(f"{name}_inked.png", (paint * ink).clip(0, 255).astype("uint8"))
        print(json.dumps({"painting": f"{name}.png", "kept": best["file"], "faults": len(best["problems"]),
                          "seconds": round(time.time() - t0, 1)}))
        return
    bw = klein.black_and_white(cv2.imread(str(raw)))
    cv2.imwrite(f"{name}.png", bw)
    size, layers, _ = vectorise(bw)
    write_svg(f"{name}.svg", size, layers)
    cv2.imwrite(f"{name}_preview.png", render(size, layers))
    print(json.dumps({"drawing": f"{name}.png", "kept": best["file"], "faults": len(best["problems"]),
                      "seconds": round(time.time() - t0, 1), "size": list(size),
                      **travel_stats([p for _, _, p in layers])}))


if __name__ == "__main__":
    main()
