"""Scene understanding for ink mode: what each pixel is, and what to leave out.

* `segment` labels every pixel with a scene layer (sky / building / vegetation /
  hills / ground / other) using Mask2Former trained on ADE20K, so ink mode can
  draw each layer in its own way.
* `clutter_mask` finds things that belong to the moment rather than the place
  (cranes, people, cars, street furniture) from text prompts: Grounding DINO
  finds boxes and SAM 2.1 turns each box into a mask. Semantic segmentation
  alone misses thin things like cranes against the sky.
* `inpaint` fills a mask with LaMa, so edge detection sees what was behind.

Models come from the Hugging Face hub (cached in ~/.cache/huggingface) and are
loaded per call and released afterwards, so they never share the GPU with each
other or with TEED. Needs `uv sync --extra learned --extra segment`.
"""

import gc

import cv2
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEG_MODEL = "facebook/mask2former-swin-large-ade-semantic"
DINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM_MODEL = "facebook/sam2.1-hiera-large"
LAMA_WEIGHTS = ("fashn-ai/LaMa", "big-lama.pt")  # TorchScript export of big-lama (Apache-2.0)

LAYERS = ["sky", "building", "vegetation", "hills", "ground", "other"]
# ADE20K class names folded into scene layers; anything unlisted is "other".
ADE_LAYERS = {
    "sky": ["sky"],
    "building": ["building", "house", "skyscraper", "wall", "tower", "door", "windowpane", "column",
                 "roof", "balcony", "railing", "stairs", "stairway", "step", "bannister", "arcade",
                 "awning"],
    "vegetation": ["tree", "grass", "plant", "palm", "flower", "field", "bush"],
    "hills": ["mountain", "hill"],
    "ground": ["road", "sidewalk", "floor", "earth", "path", "sand", "dirt track", "runway", "land"],
}
# Labels Grounding DINO also gives to parts of buildings (columns, lettering);
# these must not be mostly building in the segmentation to count as clutter.
LOOKALIKES = {"street lamp", "sign"}
# "road sign" and "traffic sign" are not LOOKALIKES, so a sign on a pole is removed even when the
# segmentation paints it as part of the building behind (the mill pond's no-cycling sign).
CLUTTER = ["crane", "person", "dog", "car", "van", "bicycle", "street lamp", "sign", "road sign",
           "traffic sign", "umbrella", "table", "chair", "scaffolding", "traffic cone", "bin"]


def _release(*models):
    del models
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def _rgb(bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


@torch.inference_mode()
def segment(bgr):
    """Scene layer per pixel, as indices into LAYERS (uint8, photo size)."""
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    proc = AutoImageProcessor.from_pretrained(SEG_MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_MODEL).to(DEVICE).eval()
    h, w = bgr.shape[:2]
    out = model(**proc(images=_rgb(bgr), return_tensors="pt").to(DEVICE))
    seg = proc.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0].cpu().numpy()

    lut = np.full(len(model.config.id2label), LAYERS.index("other"), np.uint8)
    for i, label in model.config.id2label.items():
        names = {n.strip() for n in label.lower().replace(";", ",").split(",")}
        for layer, keys in ADE_LAYERS.items():
            if names & set(keys):
                lut[int(i)] = LAYERS.index(layer)
                break
    _release(model)
    return lut[seg]


@torch.inference_mode()
def _detect(proc, model, img, prompts, box_thr, text_thr):
    inputs = proc(images=img, text=". ".join(prompts) + ".", return_tensors="pt").to(DEVICE)
    r = proc.post_process_grounded_object_detection(
        model(**inputs), inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[img.size[::-1]])[0]
    boxes = r["boxes"].cpu().numpy()
    # With no detections the label list still holds one empty string.
    return boxes, list(r["text_labels"])[:len(boxes)]


@torch.inference_mode()
def clutter_mask(bgr, prompts=CLUTTER, layers=None, box_thr=0.25, text_thr=0.25, batch=4, max_building=0.6):
    """Pixels covered by anything matching `prompts`. Returns (mask, labels).

    With `layers` (from `segment`), LOOKALIKES the segmentation mostly calls
    building are kept: at low confidence Grounding DINO takes cast-iron columns
    for street lamps, while real street furniture has its own ADE20K classes.
    (People and cars are left out of the check: under awnings and in doorways
    the segmentation often paints them as building.)"""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
    img = _rgb(bgr)
    h, w = bgr.shape[:2]

    proc = AutoProcessor.from_pretrained(DINO_MODEL)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(DEVICE).eval()
    boxes, labels = [], []
    # Grounding DINO returns nothing at all for long prompt lists, so ask a few at a time.
    for i in range(0, len(prompts), batch):
        b, l = _detect(proc, dino, img, prompts[i:i + batch], box_thr, text_thr)
        boxes.append(b)
        labels += l
    _release(dino)
    boxes = np.concatenate(boxes).reshape(-1, 4)
    # A box over half the photo is the model labelling the whole scene, not an
    # object. Labels can come back truncated ("street" for "street lamp"); those are guesses.
    keep = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) < 0.5 * w * h) & \
        np.isin(labels, prompts)
    boxes, labels = boxes[keep], [l for l, k in zip(labels, keep) if k]
    if not len(boxes):
        return np.zeros((h, w), bool), []

    proc = Sam2Processor.from_pretrained(SAM_MODEL)
    sam = Sam2Model.from_pretrained(SAM_MODEL).to(DEVICE).eval()
    inputs = proc(images=img, input_boxes=[boxes.tolist()], return_tensors="pt").to(DEVICE)
    out = sam(**inputs, multimask_output=False)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][:, 0].numpy() > 0
    _release(sam)
    if layers is not None:
        building = layers == LAYERS.index("building")
        keep = [l not in LOOKALIKES or (m & building).sum() <= max_building * max(m.sum(), 1)
                for m, l in zip(masks, labels)]
        masks, labels = masks[keep], [l for l, k in zip(labels, keep) if k]
    return masks.any(0) if len(masks) else np.zeros((h, w), bool), labels


@torch.inference_mode()
def inpaint(bgr, mask, max_side=2048):
    """Fill `mask` with LaMa. Large photos are filled at `max_side` and only the
    masked pixels are pasted back, so the rest keeps full resolution."""
    if not mask.any():
        return bgr
    from huggingface_hub import hf_hub_download
    lama = torch.jit.load(hf_hub_download(*LAMA_WEIGHTS), map_location=DEVICE).eval()
    h, w = mask.shape
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    m = cv2.resize(mask.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    sh, sw = m.shape
    ph, pw = -sh % 8, -sw % 8
    rgb = np.pad(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(DEVICE) / 255
    mt = torch.from_numpy(np.pad(m, ((0, ph), (0, pw))))[None, None].float().to(DEVICE)
    y = lama(x, mt)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    _release(lama)
    filled = cv2.cvtColor((y[:sh, :sw] * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    if s < 1:
        filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    out = bgr.copy()
    out[mask] = filled[mask]
    return out


def overlay(bgr, layers, clutter=None):
    """Debug view: scene layers tinted over the photo, clutter in magenta."""
    colours = np.array([(235, 206, 135), (60, 60, 200), (60, 170, 60),
                        (40, 120, 160), (150, 150, 150), (200, 200, 200)], np.uint8)
    out = cv2.addWeighted(bgr, 0.45, colours[layers], 0.55, 0)
    if clutter is not None:
        out[clutter] = (0.4 * out[clutter] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
    return out


def declutter(bgr, prompts=None, grow=6):
    """Clutter found, checked against the segmentation, grown and inpainted.
    Returns (clean photo, mask, labels, scene layers)."""
    layers = segment(bgr)
    mask, labels = clutter_mask(bgr, list(prompts or CLUTTER), layers)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
    mask = cv2.dilate(mask.astype(np.uint8), k) > 0
    return inpaint(bgr, mask), mask, labels, layers
