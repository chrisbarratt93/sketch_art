"""Photo -> drawing or painting with Nano Banana Pro (Gemini 3 Pro Image), via ComfyUI.

The same prompts as klein (klein.PROMPTS), sent to Google's model through
ComfyUI's paid partner node and billed to your Comfy account credits (roughly
$0.13 per 2K image, $0.24 at 4K). It follows style instructions much better
than klein, which is what the `architect` and `watercolour_ink` styles need.

Every result is checked by Gemini 3 Pro against the photo (REVIEW) for invented
text or objects, missing scenery and stray blots. If it finds any, the image is
regenerated with the next seed and told what to avoid, up to `attempts` times,
and the cleanest attempt is kept. Each review costs a few cents; each retry
costs a full image.

Needs a running ComfyUI (see klein.py) and a Comfy API key from
platform.comfy.org in ~/.config/sketch_art/comfy_api_key. The key is sent to
ComfyUI in the request body only.
"""

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2

from . import klein

KEY = Path.home() / ".config/sketch_art/comfy_api_key"
MODEL = "gemini-3-pro-image-preview"
REVIEWER = "gemini-3-pro-preview"

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


@dataclass
class NanoBananaParams:
    prompt: str = "architect"   # key of klein.PROMPTS, or a prompt of your own
    resolution: str = "2K"      # 1K, 2K or 4K
    attempts: int = 3           # most images to generate while the review still finds faults
    review: bool = True


def _key():
    if not KEY.exists():
        raise FileNotFoundError(f"No Comfy API key: create one at platform.comfy.org and save it to {KEY}")
    return KEY.read_text().strip()


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


def review(photo, art, key=None):
    """Gemini's list of faults in `art` compared with `photo` (both local files)."""
    key = key or _key()
    graph, prev = {}, None
    for i, panel in enumerate(_panels(photo, art)):
        f = Path(art).with_suffix(f".review{i}.jpg")
        cv2.imwrite(str(f), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        graph[f"p{i}"] = {"class_type": "LoadImage", "inputs": {"image": klein._upload(f)}}
        f.unlink()
        if prev is None:
            prev = [f"p{i}", 0]
        else:
            graph[f"b{i}"] = {"class_type": "ImageBatch", "inputs": {"image1": prev, "image2": [f"p{i}", 0]}}
            prev = [f"b{i}", 0]
    graph["ask"] = {"class_type": "GeminiNode", "inputs": {"prompt": REVIEW, "model": REVIEWER, "seed": 1,
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


def draw(photo_path, seed, dst, p=NanoBananaParams(), images=None, prompt_text=None):
    """Generate from the photo at `photo_path` and write the cleanest attempt to `dst`.

    `images` (default: just the photo) are the reference images sent, in order;
    `prompt_text` overrides the prompt (default: PROMPTS[p.prompt] + FAITHFUL).
    The review always compares against `photo_path`. Returns {"kept", "attempts"};
    every attempt is saved next to `dst` as <stem>_a<n>.png.
    """
    key = _key()
    prompt = prompt_text or klein.PROMPTS.get(p.prompt, p.prompt) + " " + klein.FAITHFUL
    names = [klein._upload(f) for f in (images or [photo_path])]
    dst = Path(dst)
    attempts, avoid = [], ""
    for k in range(max(1, p.attempts)):
        graph = {f"load{i}": {"class_type": "LoadImage", "inputs": {"image": n}} for i, n in enumerate(names)}
        refs = ["load0", 0]
        for i in range(1, len(names)):
            # ImageBatch resizes later images to match the first.
            graph[f"batch{i}"] = {"class_type": "ImageBatch", "inputs": {"image1": refs, "image2": [f"load{i}", 0]}}
            refs = [f"batch{i}", 0]
        graph["gen"] = {"class_type": "GeminiImage2Node", "inputs": {
            "prompt": prompt + avoid, "model": MODEL, "seed": seed + k, "aspect_ratio": "auto",
            "resolution": p.resolution, "response_modalities": "IMAGE", "images": refs}}
        graph["save"] = {"class_type": "SaveImage", "inputs": {"images": ["gen", 0], "filename_prefix": "sketch_art/nanobanana"}}
        img = next(i for o in _run(graph, key).values() for i in o.get("images", []))
        raw = dst.with_name(f"{dst.stem}_a{k + 1}.png")
        klein._call(f"/view?filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}", out=raw)
        problems = review(photo_path, raw, key) if p.review else []
        attempts.append({"file": raw.name, "seed": seed + k, "problems": problems})
        if not problems:
            break
        avoid = (" A previous attempt had these faults; make sure none of them appear: "
                 + "; ".join(f"{q.get('what', '')} ({q.get('where', '')})" for q in problems) + ".")
    best = min(attempts, key=lambda a: len(a["problems"]))
    cv2.imwrite(str(dst), cv2.imread(str(dst.with_name(best["file"]))))
    return {"kept": best["file"], "attempts": attempts}
