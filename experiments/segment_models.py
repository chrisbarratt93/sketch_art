"""Compare semantic segmentation models for scene-layer separation (see IDEAS.md).

Each model's classes are folded into the scene layers ink mode would care about:
sky / building / vegetation / hills / ground / clutter / other. Writes a colour
overlay per model and photo, plus timing and layer coverage as JSON.

    uv run --all-extras experiments/segment_models.py
    uv run --all-extras experiments/segment_models.py --models segformer-ade --images examples/market_hall.jpg
"""

import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, Mask2FormerForUniversalSegmentation

MODELS = {
    "segformer-ade": ("nvidia/segformer-b5-finetuned-ade-640-640", "segformer"),
    "segformer-city": ("nvidia/segformer-b5-finetuned-cityscapes-1024-1024", "segformer"),
    "mask2former-ade": ("facebook/mask2former-swin-large-ade-semantic", "mask2former"),
    "mask2former-city": ("facebook/mask2former-swin-large-cityscapes-semantic", "mask2former"),
}

LAYERS = {
    "sky": ["sky"],
    "building": ["building", "house", "skyscraper", "wall", "tower", "door", "windowpane", "column",
                 "roof", "balcony", "railing", "stairs", "stairway", "step", "bannister", "arcade"],
    "vegetation": ["tree", "grass", "plant", "palm", "flower", "vegetation", "terrain", "field", "bush"],
    "hills": ["mountain", "hill"],
    "ground": ["road", "sidewalk", "floor", "earth", "path", "sand", "dirt track", "runway", "land"],
    "clutter": ["person", "rider", "car", "bus", "truck", "van", "bicycle", "motorcycle", "minibike",
                "signboard", "traffic sign", "traffic light", "streetlight", "pole", "crane", "boat",
                "awning", "fence", "bench", "trade name", "poster", "flag", "ashcan", "booth", "tent",
                "train", "chair", "table", "umbrella", "pot", "sculpture", "fountain", "bulletin board"],
}
COLOURS = {  # BGR
    "sky": (235, 206, 135), "building": (60, 60, 200), "vegetation": (60, 170, 60),
    "hills": (40, 120, 160), "ground": (150, 150, 150), "clutter": (0, 200, 255), "other": (200, 0, 200),
}
NAMES = list(COLOURS)


def label_to_layer(id2label):
    lut = np.full(len(id2label), NAMES.index("other"), np.uint8)
    for i, name in id2label.items():
        names = [n.strip() for n in name.lower().replace(";", ",").split(",")]
        for layer, keys in LAYERS.items():
            if any(n in keys for n in names):
                lut[int(i)] = NAMES.index(layer)
                break
    return lut


def load(key):
    repo, kind = MODELS[key]
    proc = AutoImageProcessor.from_pretrained(repo)
    cls = Mask2FormerForUniversalSegmentation if kind == "mask2former" else AutoModelForSemanticSegmentation
    model = cls.from_pretrained(repo).to("cuda").eval()
    return proc, model, kind


@torch.inference_mode()
def segment(proc, model, kind, img):
    w, h = img.size
    inputs = proc(images=img, return_tensors="pt").to("cuda")
    out = model(**inputs)
    if kind == "mask2former":
        return proc.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0].cpu().numpy()
    logits = torch.nn.functional.interpolate(out.logits, size=(h, w), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--images", nargs="*", default=sorted(str(p) for p in Path("examples").iterdir()))
    ap.add_argument("--max-side", type=int, default=1400)
    ap.add_argument("-o", "--out", default="out/segment")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imgs = {}
    for p in args.images:
        img = Image.open(p).convert("RGB")
        s = args.max_side / max(img.size)
        if args.max_side and s < 1:
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        imgs[Path(p).stem] = img

    report = {}
    for key in args.models:
        t0 = time.time()
        proc, model, kind = load(key)
        load_s = time.time() - t0
        lut = label_to_layer(model.config.id2label)
        torch.cuda.reset_peak_memory_stats()
        for stem, img in imgs.items():
            segment(proc, model, kind, img)  # warm-up so timings exclude cudnn autotune
            torch.cuda.synchronize()
            t0 = time.time()
            seg = segment(proc, model, kind, img)
            torch.cuda.synchronize()
            dt = time.time() - t0
            layer = lut[seg]
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            colour = np.array([COLOURS[n] for n in NAMES], np.uint8)[layer]
            overlay = cv2.addWeighted(bgr, 0.45, colour, 0.55, 0)
            cv2.imwrite(str(out / f"{stem}_{key}.jpg"), np.hstack([bgr, overlay]), [cv2.IMWRITE_JPEG_QUALITY, 88])
            np.save(out / f"{stem}_{key}_layers.npy", layer)
            classes = {model.config.id2label[int(c)]: round(float((seg == c).mean()), 3)
                       for c in np.unique(seg) if (seg == c).mean() > 0.01}
            report.setdefault(stem, {})[key] = {
                "seconds": round(dt, 2),
                "layers": {n: round(float((layer == i).mean()), 3) for i, n in enumerate(NAMES) if (layer == i).any()},
                "classes": dict(sorted(classes.items(), key=lambda kv: -kv[1])),
            }
            print(f"{key:18s} {stem:16s} {dt:5.2f}s", report[stem][key]["layers"], flush=True)
        report.setdefault("_models", {})[key] = {
            "repo": MODELS[key][0], "load_s": round(load_s, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()

    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
