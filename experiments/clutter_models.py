"""Clutter removal test (see IDEAS.md): text prompt -> boxes -> masks -> inpaint.

Grounding DINO finds boxes for the prompts, SAM 2.1 turns each box into a mask,
and LaMa (big-lama TorchScript) inpaints the dilated union, so edge detection
sees the place rather than the moment. Writes photo | detections | mask | inpainted.

    uv run --all-extras experiments/clutter_models.py
    uv run --all-extras experiments/clutter_models.py --images examples/albert_hall.jpg --prompts "crane." "person."
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor)

DINO = "IDEA-Research/grounding-dino-base"
SAM2 = "facebook/sam2.1-hiera-large"
LAMA = ("fashn-ai/LaMa", "big-lama.pt")
PROMPTS = ["crane", "person", "car", "van", "bicycle", "bollard", "street lamp", "sign",
           "umbrella", "table", "chair", "scaffolding", "traffic cone", "bin"]


def detect(proc, model, img, prompts, box_thr, text_thr, batch=4):
    # Grounding DINO returns nothing for long prompt lists, so ask a few labels at a time.
    rs = [detect_once(proc, model, img, prompts[i:i + batch], box_thr, text_thr)
          for i in range(0, len(prompts), batch)]
    return np.concatenate([r[0] for r in rs]).reshape(-1, 4), sum((r[1] for r in rs), []), np.concatenate([r[2] for r in rs])


@torch.inference_mode()
def detect_once(proc, model, img, prompts, box_thr, text_thr):
    text = ". ".join(prompts) + "."
    inputs = proc(images=img, text=text, return_tensors="pt").to("cuda")
    out = model(**inputs)
    r = proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=box_thr, text_threshold=text_thr, target_sizes=[img.size[::-1]])[0]
    keep = [i for i, b in enumerate(r["boxes"]) if (b[2] - b[0]) * (b[3] - b[1]) < 0.5 * img.width * img.height]
    return r["boxes"][keep].cpu().numpy(), [r["text_labels"][i] for i in keep], r["scores"][keep].cpu().numpy()


@torch.inference_mode()
def masks_for(proc, model, img, boxes):
    if len(boxes) == 0:
        return np.zeros((0, img.height, img.width), bool)
    inputs = proc(images=img, input_boxes=[boxes.tolist()], return_tensors="pt").to("cuda")
    out = model(**inputs, multimask_output=False)
    m = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
    return m[:, 0].numpy().astype(bool)


@torch.inference_mode()
def inpaint(lama, rgb, mask):
    h, w = mask.shape
    ph, pw = -h % 8, -w % 8
    im = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mk = np.pad(mask, ((0, ph), (0, pw)))
    x = torch.from_numpy(im).permute(2, 0, 1)[None].float().cuda() / 255
    m = torch.from_numpy(mk)[None, None].float().cuda()
    y = lama(x, m)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (y[:h, :w] * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=sorted(str(p) for p in Path("examples").iterdir()))
    ap.add_argument("--prompts", nargs="*", default=PROMPTS)
    ap.add_argument("--box-thr", type=float, default=0.3)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--dilate", type=int, default=9, help="mask growth in px before inpainting")
    ap.add_argument("--max-side", type=int, default=1400)
    ap.add_argument("-o", "--out", default="out/clutter")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    dproc = AutoProcessor.from_pretrained(DINO)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).cuda().eval()
    sproc = Sam2Processor.from_pretrained(SAM2)
    sam = Sam2Model.from_pretrained(SAM2).cuda().eval()
    lama = torch.jit.load(hf_hub_download(*LAMA), map_location="cuda").eval()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    report = {}
    for p in args.images:
        img = Image.open(p).convert("RGB")
        s = args.max_side / max(img.size)
        if args.max_side and s < 1:
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        rgb = np.asarray(img)
        times = {}

        t0 = time.time()
        boxes, labels, scores = detect(dproc, dino, img, args.prompts, args.box_thr, args.text_thr)
        times["dino"] = time.time() - t0
        t0 = time.time()
        masks = masks_for(sproc, sam, img, boxes)
        times["sam2"] = time.time() - t0
        mask = masks.any(0) if len(masks) else np.zeros(rgb.shape[:2], bool)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.dilate + 1,) * 2)
        mask = cv2.dilate(mask.astype(np.uint8), k) > 0
        t0 = time.time()
        filled = inpaint(lama, rgb, mask)
        times["lama"] = time.time() - t0

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        det = bgr.copy()
        for b, lab, sc in zip(boxes.astype(int), labels, scores):
            cv2.rectangle(det, tuple(b[:2]), tuple(b[2:]), (0, 200, 255), 2)
            cv2.putText(det, f"{lab} {sc:.2f}", (b[0], max(b[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(det, f"{lab} {sc:.2f}", (b[0], max(b[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        shown = bgr.copy()
        shown[mask] = (0.4 * shown[mask] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
        top, bottom = np.hstack([bgr, det]), np.hstack([shown, cv2.cvtColor(filled, cv2.COLOR_RGB2BGR)])
        stem = Path(p).stem
        cv2.imwrite(str(out / f"{stem}_grid.jpg"), np.vstack([top, bottom]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(out / f"{stem}_clean.png"), cv2.cvtColor(filled, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{stem}_mask.png"), mask.astype(np.uint8) * 255)
        report[stem] = {"seconds": {k: round(v, 2) for k, v in times.items()},
                        "masked": round(float(mask.mean()), 3),
                        "detections": [f"{l} {s:.2f}" for l, s in zip(labels, scores)]}
        print(stem, report[stem], flush=True)
    report["_vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
