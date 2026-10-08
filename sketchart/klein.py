"""Photo -> hand-drawn ink drawing with FLUX.2 [klein] 4B, run in ComfyUI.

The tracer (ink.py) can only follow edges. klein is an image-editing model
that has learned how illustrators draw: it simplifies, chooses where to put
detail, and draws each material with its own marks. Given the (decluttered)
photo and an instruction prompt, it redraws the scene as a pen drawing in the
same composition.

The output is forced to pure black ink on white paper (`black_and_white`),
because the model sometimes lets a little colour or grey wash through. The
drawing is then traced into pen strokes for the plotter by vectorise.py.

Needs a running ComfyUI (default http://127.0.0.1:8188, or $COMFY_URL) with
these files in its models folder:
    diffusion_models/flux-2-klein-4b.safetensors
    text_encoders/qwen_3_4b.safetensors
    vae/flux2-vae.safetensors
From WSL, requests go through Windows' curl.exe, because WSL can't reach a
Windows app listening on localhost.
"""

import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SERVER = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
_CURL = shutil.which("curl.exe") or "curl"
_WIN = _CURL.endswith(".exe")

PROMPTS = {
    # Default ("E2" in the 2026-10-08 comparison): between a loose urban sketch and an engraving.
    "urban_rich": (
        "Redraw this photo as a lively pen and ink urban sketch, black fineliner on pure white paper, "
        "with a medium density of detail: richer than a quick sketch, lighter than an engraving. Keep "
        "the exact composition, framing and perspective of the photo, and draw only what is in it: add "
        "nothing. Confident hand-drawn outlines with some looseness. Every material that is in the "
        "photo keeps its texture in ink, for example stripes on awnings, tile rows on roofs, coursing "
        "on stone walls, short flicked strokes for grass, scalloped clusters for leaves. Shadows with "
        "parallel hatching. Leave white paper in the sky and lit areas. Black ink only: no pencil, no "
        "grey tones, no wash, no colour, no solid black fills."),
    "urban": (
        "Turn this photo into a loose urban sketch in black fineliner on white paper, as a skilled urban "
        "sketcher would draw it on location. Confident, slightly wobbly hand-drawn lines, detail "
        "concentrated on the main building, the edges of the picture left unfinished and fading out to "
        "white paper, sparse hatching for shadows. Black ink only: no colour, no grey tones, no fills."),
    "plotter": (
        "Redraw this photo as a clean pen and ink line drawing by a skilled architectural illustrator. "
        "Every mark is a crisp, solid black line on pure white paper. Confident outlines, simplified "
        "detail, shadows shown only by sparse parallel hatching lines. No grey, no pencil, no wash, no "
        "shading, no solid black fills, no paper texture, no colour."),
    "engraving": (
        "Redraw this photo as a detailed pen and ink illustration in the style of a Victorian "
        "architectural engraving: crisp outlines, tone built only from fine parallel hatching and "
        "cross-hatching, black ink on white paper. No grey wash, no colour, no solid fills."),
    # Painted styles (PAINTED): for prints, not the plotter. Kept in colour and tone.
    "watercolour_ink": (
        "Turn this photo into a loose watercolour painting with pen and ink line work, as a skilled "
        "urban sketcher would paint it on location. Keep the exact composition, framing and "
        "perspective of the photo, and paint only what is in it: add nothing. Confident, lively black "
        "fineliner lines for the buildings and key details, with texture marks for stonework, tiles "
        "and foliage. Transparent watercolour washes laid loosely over the lines, soft wet-in-wet "
        "edges, colour that strays a little past the lines, gentle granulation and blooms. Natural, "
        "slightly muted colours true to the photo. Leave white paper in the sky and highlights, and "
        "let the washes fade out to bare white watercolour paper towards the edges of the picture."),
    # watercolour_ink with livelier colour (its result on 2026-10-08 was accurate but a little dull).
    "watercolour_ink_vibrant": (
        "Turn this photo into a loose watercolour painting with pen and ink line work, as a skilled "
        "urban sketcher would paint it on location. Keep the exact composition, framing and "
        "perspective of the photo, and paint only what is in it: add nothing. Confident, lively black "
        "fineliner lines for the buildings and key details, with texture marks for stonework, tiles "
        "and foliage. Transparent watercolour washes laid loosely over the lines, soft wet-in-wet "
        "edges, colour that strays a little past the lines, gentle granulation and blooms. Fresh, "
        "clear and vibrant colour: saturated, luminous washes in the photo's own hues, a little "
        "richer than the photo, with warm sunlit stone, rich varied greens and bright flowers, never "
        "muddy or grey, while every ink line and detail stays crisp and visible. Leave white paper in "
        "the sky and highlights, and let the washes fade out to bare white watercolour paper towards "
        "the edges of the picture."),
    # watercolour_ink_vibrant, with the ink pushed to look deliberately hand-drawn rather than generated.
    "watercolour_ink_hand": (
        "Turn this photo into a loose watercolour painting with pen and ink line work, made by hand by "
        "a skilled urban sketcher on location. Keep the exact composition, framing and perspective of "
        "the photo, and paint only what is in it: add nothing. The ink must look unmistakably hand "
        "drawn: confident, deliberate, continuous pen strokes with natural variation in pressure and "
        "line weight, each line placed on purpose by the artist's hand. No speckle, no stippled noise, "
        "no fuzzy broken fragments, no scratchy over-rendered texture, and no uniform machine-like "
        "detail spread evenly everywhere. Every mark is a choice: the artist draws detail only where it "
        "matters, at the focal point and on key edges, windows, columns and tree trunks, and elsewhere "
        "suggests stonework, tiles, foliage and grass with a few economical, well-chosen strokes or "
        "leaves them to the wash. Transparent watercolour washes laid loosely over the lines, soft "
        "wet-in-wet edges, colour that strays a little past the lines, gentle granulation and blooms. "
        "Fresh, clear and vibrant colour: saturated, luminous washes in the photo's own hues, a little "
        "richer than the photo, never muddy or grey. Leave white paper in the sky and highlights, and "
        "let the washes fade out to bare white watercolour paper towards the edges of the picture."),
    "watercolour_ink_detailed": (
        "Turn this photo into a detailed pen and ink drawing with watercolour, as a skilled "
        "architectural illustrator would make it. Keep the exact composition, framing and perspective "
        "of the photo, and draw only what is in it: add nothing, remove nothing, simplify nothing. "
        "First a careful, detailed black fineliner drawing faithful to the photo: every window and "
        "glazing bar, the rows of roof tiles, the courses of stonework, chimneys, gutters, lamps and "
        "railings, the individual shrubs, flowers and leaf clusters, the texture of the grass. Then "
        "transparent watercolour washes over it, kept light and controlled so every ink line stays "
        "clearly visible: soft wet-in-wet edges, gentle granulation, a little colour straying past the "
        "lines. Natural colours true to the photo. Leave white paper in the sky and highlights, and "
        "let the washes fade out to bare white watercolour paper towards the edges of the picture."),
    "pen_and_wash": (
        "Redraw this photo as a traditional pen and wash drawing: confident dark sepia ink lines from "
        "a dip pen, with transparent sepia and warm grey ink washes brushed in loose layers for shadow "
        "and tone. Keep the exact composition, framing and perspective of the photo, and draw only "
        "what is in it: add nothing. Lively hand-drawn line with texture marks for stonework, tiles "
        "and foliage. Lights left as bare white paper, washes fading out towards the edges of the "
        "picture. Monochrome sepia only, no other colours."),
    "pencil": (
        "Redraw this photo as a skilled architectural pencil sketch in graphite on white cartridge "
        "paper. Keep the exact composition, framing and perspective of the photo, and draw only what "
        "is in it: add nothing. Confident construction lines and outlines, tone built with directional "
        "hatching and soft shading that follows each surface, textures for stonework, tiles, grass "
        "and foliage, darkest accents in the shadows. Lit areas and sky left as white paper, the "
        "sketch fading out towards the edges of the picture. Graphite only, no colour."),
}
# Appended to every prompt, built-in or custom: image models like to invent lettering
# (a "25" on a no-cycling sign, writing on a wall) and stray marks.
FAITHFUL = (
    "Do not add any text, letters, numbers or symbols: any sign, plaque or lettering that is in the "
    "photo is reproduced exactly as it is or left blank. Do not add any object, figure, mark, blot or "
    "splash that is not in the photo, and do not leave out any building, tree, hill or water.")

# Styles that are paintings, not pen drawings: kept in colour/tone and not traced for the plotter.
PAINTED = {"watercolour_ink", "watercolour_ink_vibrant", "watercolour_ink_hand", "watercolour_ink_detailed", "pen_and_wash", "pencil"}


@dataclass
class KleinParams:
    prompt: str = "urban_rich"  # key of PROMPTS, or a prompt of your own
    megapixels: float = 4.0     # output size; ~50 s per drawing at 4 MP on an RTX 3080
    steps: int = 4              # the distilled model is trained for 4


def _path(p):
    p = str(Path(p).resolve())
    return subprocess.check_output(["wslpath", "-w", p], text=True).strip() if _WIN else p


def _call(path, data=None, out=None):
    cmd = [_CURL, "-s", "-f", "-m", "900", f"{SERVER}{path}"]
    if data is not None:
        cmd += ["-H", "Content-Type: application/json", "--data-binary", "@-"]
    if out is not None:
        cmd += ["-o", _path(out)]
    r = subprocess.run(cmd, input=json.dumps(data) if data is not None else None, capture_output=True, text=True)
    if r.returncode == 7:
        raise ConnectionError(f"ComfyUI isn't reachable at {SERVER}. Start ComfyUI Desktop first.")
    if r.returncode:
        raise RuntimeError(f"ComfyUI {path}: curl exit {r.returncode} {r.stdout[:500]}")
    return json.loads(r.stdout) if out is None and r.stdout else None


def _upload(path):
    r = subprocess.run([_CURL, "-s", "-f", "-F", f"image=@{_path(path)}", "-F", "overwrite=true",
                        f"{SERVER}/upload/image"], capture_output=True, text=True)
    if r.returncode:
        raise ConnectionError(f"Couldn't upload to ComfyUI at {SERVER} (curl exit {r.returncode}).")
    return json.loads(r.stdout)["name"]


def _graph(image, prompt, seed, p):
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "load": {"class_type": "LoadImage", "inputs": {"image": image}},
        "scale": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["load", 0], "upscale_method": "lanczos", "megapixels": p.megapixels, "resolution_steps": 16}},
        "size": {"class_type": "GetImageSize", "inputs": {"image": ["scale", 0]}},
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["text", 0]}},
        # The photo goes in as a reference image for both branches, so the
        # drawing keeps its composition; the text says how to draw it.
        "pos": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["text", 0], "latent": ["enc", 0]}},
        "neg": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["zero", 0], "latent": ["enc", 0]}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": ["unet", 0], "positive": ["pos", 0], "negative": ["neg", 0], "cfg": 1.0}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "Flux2Scheduler", "inputs": {"steps": p.steps, "width": ["size", 0], "height": ["size", 1]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["size", 0], "height": ["size", 1], "batch_size": 1}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "sketch_art/klein"}},
    }


def draw(photo_path, seed, dst, p=KleinParams()):
    """Redraw the photo at `photo_path` as ink; writes the raw model output to `dst`."""
    prompt = PROMPTS.get(p.prompt, p.prompt) + " " + FAITHFUL
    pid = _call("/prompt", {"prompt": _graph(_upload(photo_path), prompt, seed, p),
                            "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        h = _call(f"/history/{pid}").get(pid)
        if h and h.get("status", {}).get("status_str") == "error":
            raise RuntimeError("ComfyUI error: " + json.dumps(h["status"].get("messages"))[:2000])
        if h and h.get("status", {}).get("completed"):
            break
        time.sleep(2)
    img = next(i for o in h["outputs"].values() for i in o.get("images", []))
    _call(f"/view?filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}", out=dst)


def black_and_white(bgr, paper=0.85, ink=0.3):
    """Pure black ink on white paper. Colour is dropped, the paper level is
    measured locally (generated paper is rarely even) and pushed to white, and
    the ink is darkened to black, keeping anti-aliased stroke edges."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    level = cv2.GaussianBlur(cv2.dilate(gray, np.ones((25, 25), np.uint8)), (0, 0), 15)
    t = np.clip((gray / np.maximum(level, 1) - ink) / (paper - ink), 0, 1)
    return (255 * t * t * (3 - 2 * t)).astype(np.uint8)
