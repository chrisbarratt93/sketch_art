"""Scene layers and depth from Hugging Face models (needs `uv sync --extra hub`).

* `segment()` labels every pixel with an ADE20K class using OneFormer, then
  groups the 150 classes into the layers a drawing cares about: sky,
  buildings, vegetation, terrain, ground, people, vehicles and street clutter.
* `nearness()` is relative depth from Depth Anything V2 (Small: Apache-2.0),
  scaled to 0 (farthest) .. 1 (nearest).

Weights download on first use to ~/.cache/huggingface.
"""

import warnings

import cv2
import numpy as np

SEG_MODELS = {
    "tiny": "shi-labs/oneformer_ade20k_swin_tiny",
    "large": "shi-labs/oneformer_ade20k_swin_large",
}
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# ADE20K class names (first synonym, as in the model's id2label) -> layer.
GROUPS = {
    "sky": {"sky"},
    "buildings": {"wall", "building", "house", "skyscraper", "tower", "windowpane", "door", "column",
                  "awning", "railing", "stairs", "stairway", "step", "fence", "hovel", "canopy",
                  "bannister", "bridge", "booth", "fountain", "sculpture", "pier"},
    "vegetation": {"tree", "grass", "plant", "flower", "palm", "field"},
    "terrain": {"mountain", "hill", "earth", "land", "rock", "sand", "water", "sea", "river", "lake",
                "waterfall"},
    "ground": {"road", "sidewalk", "floor", "path", "runway", "dirt track"},
    "people": {"person"},
    "vehicles": {"car", "bus", "truck", "van", "bicycle", "minibike", "boat", "ship", "airplane"},
    "clutter": {"signboard", "streetlight", "traffic light", "pole", "flag", "trade name", "poster",
                "ashcan", "bench", "bulletin board", "tent", "barrel", "pot", "plaything"},
}
COLOURS = {  # BGR, for previews
    "sky": (235, 206, 135), "buildings": (60, 80, 200), "vegetation": (60, 170, 60),
    "terrain": (40, 120, 160), "ground": (150, 150, 150), "people": (200, 60, 200),
    "vehicles": (0, 200, 255), "clutter": (30, 30, 220), "other": (230, 230, 230),
}

_cache = {}


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _quiet_import():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import transformers  # noqa: F401


def segment(bgr, model="tiny"):
    """Returns ({layer: bool mask}, label map, id2label). Masks cover every
    layer in GROUPS; unlisted classes are left out ("other")."""
    import torch
    from PIL import Image
    _quiet_import()
    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

    repo = SEG_MODELS.get(model, model)
    if repo not in _cache:
        _cache[repo] = (OneFormerProcessor.from_pretrained(repo),
                        OneFormerForUniversalSegmentation.from_pretrained(repo).to(_device()).eval())
    proc, net = _cache[repo]
    h, w = bgr.shape[:2]
    inputs = proc(images=Image.fromarray(bgr[:, :, ::-1]), task_inputs=["semantic"], return_tensors="pt")
    inputs = {k: v.to(_device()) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = net(**inputs)
    labels = proc.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0].cpu().numpy()

    id2label = net.config.id2label
    name = {int(i): s.split(",")[0].strip() for i, s in id2label.items()}
    masks = {}
    for layer, classes in GROUPS.items():
        ids = [i for i, s in name.items() if s in classes]
        masks[layer] = np.isin(labels, ids)
    return masks, labels, name


def preview(bgr, masks, alpha=0.55):
    """Photo with each layer tinted, for checking the segmentation."""
    tint = np.zeros_like(bgr)
    tint[:] = COLOURS["other"]
    for layer, m in masks.items():
        tint[m] = COLOURS[layer]
    out = cv2.addWeighted(bgr, 1 - alpha, tint, alpha, 0)
    y = 24
    for layer in masks:
        if masks[layer].any():
            cv2.rectangle(out, (8, y - 14), (24, y + 2), COLOURS[layer], -1)
            cv2.putText(out, layer, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, layer, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
    return out


def nearness(bgr):
    """Relative depth, 0 = farthest .. 1 = nearest, at the size of `bgr`."""
    import torch
    from PIL import Image
    _quiet_import()
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if DEPTH_MODEL not in _cache:
        _cache[DEPTH_MODEL] = (AutoImageProcessor.from_pretrained(DEPTH_MODEL),
                               AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(_device()).eval())
    proc, net = _cache[DEPTH_MODEL]
    h, w = bgr.shape[:2]
    inputs = proc(images=Image.fromarray(bgr[:, :, ::-1]), return_tensors="pt").to(_device())
    with torch.no_grad():
        d = net(**inputs).predicted_depth  # relative inverse depth: larger = nearer
    d = torch.nn.functional.interpolate(d[:, None], size=(h, w), mode="bicubic", align_corners=False)[0, 0]
    d = d.cpu().numpy().astype(np.float32)
    lo, hi = np.percentile(d, [1, 99])
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
