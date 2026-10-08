"""Photo -> artist-quality ink drawing with local image-editing models in ComfyUI.

Sends each (decluttered) photo through an instruction-following image editor
with a pen-and-ink prompt, and saves what comes back:

* klein:  FLUX.2 [klein] 4B distilled (Apache-2.0), 4 steps.
* qwen:   Qwen-Image-Edit-2511 Q4_K_M GGUF + 4-step Lightning LoRA (Apache-2.0).
          Tested 2026-10-08, lost to klein; models deleted, needs re-download + ComfyUI-GGUF.
* sdxl:   Juggernaut XL + ControlNet Union (lineart), steered by our TEED line map.

ComfyUI runs on Windows; WSL can't reach its localhost, so requests go through
curl.exe. Start ComfyUI Desktop first.

    uv run experiments/comfy_ink.py --models klein qwen --styles urban architect
"""

import argparse
import itertools
import json
import random
import subprocess
import time
import uuid
from pathlib import Path

SERVER = "http://127.0.0.1:8188"

STYLES = {
    "architect": (
        "Redraw this photo as a hand-drawn pen and ink architectural illustration. Black ink lines on "
        "plain white paper, drawn with a fine technical pen by a skilled illustrator. Confident clean "
        "outlines, simplified detail, light hatching for shadows. Black ink only: no colour, no grey "
        "wash, no solid black areas, no text."),
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
    "urban_texture": (
        "Turn this photo into a loose urban sketch in black fineliner on pure white paper, as a skilled "
        "urban sketcher would draw it on location. Keep the exact composition, framing and perspective "
        "of the photo, and draw only what is in the photo: do not add or remove anything. Confident, "
        "slightly wobbly hand-drawn lines, detail concentrated on the main building. Give each surface "
        "its texture in pen marks, wherever it appears in the photo: stripes stay stripes, tiles, "
        "stonework and brick get a light indication, grass and foliage get short strokes and leaf "
        "marks. Use short hatching, stippling and texture marks, lighter and looser than an engraving. "
        "Black ink only: no pencil, no grey tones, no wash, no paper texture, no colour, no solid fills."),
    "urban_rich": (
        "Redraw this photo as a lively pen and ink urban sketch, black fineliner on pure white paper, "
        "with a medium density of detail: richer than a quick sketch, lighter than an engraving. Keep "
        "the exact composition, framing and perspective of the photo, and draw only what is in it: add "
        "nothing. Confident hand-drawn outlines with some looseness. Every material that is in the "
        "photo keeps its texture in ink, for example stripes on awnings, tile rows on roofs, coursing "
        "on stone walls, short flicked strokes for grass, scalloped clusters for leaves. Shadows with "
        "parallel hatching. Leave white paper in the sky and lit areas. Black ink only: no pencil, no "
        "grey tones, no wash, no colour, no solid black fills."),
    "engraving": (
        "Redraw this photo as a detailed pen and ink illustration in the style of a Victorian "
        "architectural engraving: crisp outlines, tone built only from fine parallel hatching and "
        "cross-hatching, black ink on white paper. No grey wash, no colour, no solid fills."),
}


def call(path, data=None, out=None):
    cmd = ["curl.exe", "-s", "-f", "-m", "600", f"{SERVER}{path}"]
    if data is not None:
        cmd += ["-H", "Content-Type: application/json", "--data-binary", "@-"]
    if out is not None:
        cmd += ["-o", subprocess.check_output(["wslpath", "-w", str(out)], text=True).strip()]
    r = subprocess.run(cmd, input=json.dumps(data) if data is not None else None,
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{path}: curl exit {r.returncode} {r.stdout[:500]}")
    return json.loads(r.stdout) if out is None and r.stdout else None


def upload(path):
    win = subprocess.check_output(["wslpath", "-w", str(Path(path).resolve())], text=True).strip()
    r = subprocess.run(["curl.exe", "-s", "-f", "-F", f"image=@{win}", "-F", "overwrite=true",
                        f"{SERVER}/upload/image"], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["name"]


def run(graph, out_png):
    pid = call("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        h = call(f"/history/{pid}").get(pid)
        if h and h.get("status", {}).get("completed"):
            break
        if h and h.get("status", {}).get("status_str") == "error":
            raise RuntimeError(json.dumps(h["status"])[:2000])
        time.sleep(2)
    img = next(i for o in h["outputs"].values() for i in o.get("images", []))
    call(f"/view?filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}", out=out_png)


def klein(image, prompt, seed, mp=1.0):
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "load": {"class_type": "LoadImage", "inputs": {"image": image}},
        "scale": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["load", 0], "upscale_method": "lanczos", "megapixels": mp, "resolution_steps": 16}},
        "size": {"class_type": "GetImageSize", "inputs": {"image": ["scale", 0]}},
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["text", 0]}},
        "pos": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["text", 0], "latent": ["enc", 0]}},
        "neg": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["zero", 0], "latent": ["enc", 0]}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": ["unet", 0], "positive": ["pos", 0], "negative": ["neg", 0], "cfg": 1.0}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": ["size", 0], "height": ["size", 1]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["size", 0], "height": ["size", 1], "batch_size": 1}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "sketch_art/klein"}},
    }


def qwen(image, prompt, seed):
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_M.gguf"}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.1}},
        "norm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["norm", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "load": {"class_type": "LoadImage", "inputs": {"image": image}},
        "scale": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["load", 0]}},
        "pos0": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["scale", 0], "prompt": prompt}},
        "neg0": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["scale", 0], "prompt": ""}},
        "pos": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["pos0", 0], "reference_latents_method": "index_timestep_zero"}},
        "neg": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["neg0", 0], "reference_latents_method": "index_timestep_zero"}},
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["lora", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["enc", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "sketch_art/qwen"}},
    }


def sdxl(image, prompt, seed, lines, size):
    w, h = size
    s = (1024 * 1024 / (w * h)) ** 0.5
    w, h = round(w * s / 64) * 64, round(h * s / 64) * 64
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": "pen and ink drawing, black ink line art on white paper, hand-drawn architectural illustration, hatching, " + prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": "photo, photorealistic, colour, grey wash, gradient, shading, solid black, blurry, text, watermark"}},
        "cn": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "controlnet-union-sdxl-promax.safetensors"}},
        "cntype": {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["cn", 0], "type": "canny/lineart/anime_lineart/mlsd"}},
        "lines": {"class_type": "LoadImage", "inputs": {"image": lines}},
        "inv": {"class_type": "ImageInvert", "inputs": {"image": ["lines", 0]}},
        "apply": {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "control_net": ["cntype", 0], "image": ["inv", 0], "vae": ["ckpt", 2], "strength": 0.8, "start_percent": 0.0, "end_percent": 0.8}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["ckpt", 0], "positive": ["apply", 0], "negative": ["apply", 1], "latent_image": ["latent", 0], "seed": seed, "steps": 30, "cfg": 5.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "sketch_art/sdxl"}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="out/gen/inputs", help="from sketch.py --declutter inpaint --debug")
    ap.add_argument("--photos", nargs="*", default=["albert_hall", "market_hall", "cotswold_street"])
    ap.add_argument("--models", nargs="*", default=["klein"])
    ap.add_argument("--styles", nargs="*", default=list(STYLES))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--mp", type=float, default=1.0, help="klein output size in megapixels")
    ap.add_argument("-o", "--out", default="out/gen")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = Path(args.inputs)

    import cv2
    log = []
    for photo, model, style, k in itertools.product(args.photos, args.models, args.styles, range(args.seeds)):
        tag = f"_{args.mp:g}mp" if model == "klein" and args.mp != 1 else ""
        dst = out / f"{photo}_{model}_{style}{tag}_{k}.png"
        if dst.exists():
            continue
        clean = inputs / f"{photo}_clean.jpg"
        name = upload(clean)
        seed = random.Random(f"{photo}{style}{k}").randrange(2**32)
        if model == "klein":
            g = klein(name, STYLES[style], seed, args.mp)
        elif model == "qwen":
            g = qwen(name, STYLES[style], seed)
        else:
            teed = inputs / f"{photo}_teed.png"
            h, w = cv2.imread(str(teed), cv2.IMREAD_GRAYSCALE).shape
            g = sdxl(name, STYLES[style], seed, upload(teed), (w, h))
        t0 = time.time()
        run(g, dst)
        log.append({"file": dst.name, "seconds": round(time.time() - t0, 1)})
        print(log[-1], flush=True)
    (out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
