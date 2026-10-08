"""Edge maps from Hugging Face models, as drop-in alternatives to TEED.

Each model returns an edge probability map (0..1, bright = line) at the size
of the input photo, so ink mode can trace any of them the same way. Weights
come from the Hugging Face Hub on first use and are cached in
~/.cache/huggingface. Needs `uv sync --extra hub`.

    teed            built-in TEED (sketchart/teed.py), BIPED urban edges
    lineart         Informative Drawings, "realistic" line drawings of photos
    lineart_coarse  the same network trained for bolder, simpler drawings
    lineart_anime   anime-style line art (thin, crisp, ignores shading)
    hed             HED soft edges (ControlNet's retrained version)
    pidinet         PiDiNet, a pixel-difference edge network
    mteed           TEED weights fine-tuned by MistoLine for line art
    anyline         MTEED plus a fine-detail lineart layer (Anyline)

See MODELS.md for what each is, its licence and the knobs worth trying.
"""

import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class EdgeModel:
    repo: str                   # Hugging Face repo id
    loader: str                 # controlnet_aux class name
    load_kw: tuple = ()         # extra from_pretrained kwargs
    call_kw: tuple = ()         # extra __call__ kwargs
    scale: float = 1.0          # default run size, x photo size
    hi: float = 0.5             # default hysteresis thresholds
    lo: float = 0.25


EDGE_MODELS = {
    "lineart": EdgeModel("lllyasviel/Annotators", "LineartDetector", call_kw=(("coarse", False),)),
    "lineart_coarse": EdgeModel("lllyasviel/Annotators", "LineartDetector", call_kw=(("coarse", True),)),
    "lineart_anime": EdgeModel("lllyasviel/Annotators", "LineartAnimeDetector"),
    "hed": EdgeModel("lllyasviel/Annotators", "HEDdetector", hi=0.6, lo=0.35),
    "pidinet": EdgeModel("lllyasviel/Annotators", "PidiNetDetector", hi=0.6, lo=0.35),
    "mteed": EdgeModel("TheMistoAI/MistoLine", "TEEDdetector",
                       load_kw=(("filename", "MTEED.pth"), ("subfolder", "Anyline")),
                       call_kw=(("safe_steps", 0),), scale=1.5),
    "anyline": EdgeModel("TheMistoAI/MistoLine", "AnylineDetector",
                         load_kw=(("filename", "MTEED.pth"), ("subfolder", "Anyline")), scale=1.5),
}
NAMES = ["teed", *EDGE_MODELS]

_loaded = {}


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(name):
    if name not in _loaded:
        m = EDGE_MODELS[name]
        with warnings.catch_warnings():
            # controlnet_aux's package import is noisy (mediapipe, timm deprecations).
            warnings.simplefilter("ignore")
            import controlnet_aux
        _loaded[name] = getattr(controlnet_aux, m.loader).from_pretrained(m.repo, **dict(m.load_kw)).to(device())
    return _loaded[name]


def edge_probability(name, bgr, scale=None):
    """Edge probability (0..1, line = 1) at the size of `bgr`. The model runs
    on a copy whose short side is `scale` x the photo's (default per model)."""
    h, w = bgr.shape[:2]
    m = EDGE_MODELS[name]
    res = int(round(min(h, w) * (scale or m.scale)))
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    kw = dict(m.call_kw, detect_resolution=res)
    if m.loader not in ("AnylineDetector", "TEEDdetector"):  # these return the input size
        kw["image_resolution"] = res
    import torch
    with torch.no_grad():
        out = load(name)(rgb, output_type="np", **kw)
    out = np.asarray(out)
    if out.ndim == 3:
        out = out[:, :, 0]
    out = cv2.resize(out.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA
                     if out.shape[0] > h else cv2.INTER_LINEAR)
    return np.clip(out, 0, 1)


def thresholds(name, hi=None, lo=None):
    d_hi, d_lo = (0.45, 0.2) if name == "teed" else (EDGE_MODELS[name].hi, EDGE_MODELS[name].lo)
    return d_hi if hi is None else hi, d_lo if lo is None else lo


def model_edges(name, bgr, f, scale=None, hi=None, lo=None):
    """Like edges.teed_edges for a Hub model: 1px centrelines on the `f` x
    grid and the probability map there. Returns (mask, probability)."""
    h, w = bgr.shape[:2]
    prob = edge_probability(name, bgr, scale)
    if f > 1:
        prob = cv2.resize(prob, (w * f, h * f), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    hi, lo = thresholds(name, hi, lo)
    strong = apply_hysteresis_threshold(prob, lo, hi)
    # Most of these draw lines several px wide: the skeleton is their centreline.
    return skeletonize(strong), prob
