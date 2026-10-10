"""Write the ComfyUI workflows the scripts send, as JSON you can drag onto the ComfyUI canvas.

They are in ComfyUI's API format (what the scripts submit); ComfyUI lays the nodes out when
loaded. The reference images are uploaded so the Load Image nodes find them.

    uv run --all-extras experiments/export_workflows.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sketchart import klein, nanobanana  # noqa: E402
import cloud_compare  # noqa: E402

OUT = Path("out/workflows")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Ink drawing with FLUX.2 klein (local, free): the default style.
    park = klein._upload("out/park/park_clean.png")
    g = klein._graph(park, klein.PROMPTS["urban_rich"] + " " + klein.FAITHFUL, 2, klein.KleinParams())
    (OUT / "klein_ink_sketch.json").write_text(json.dumps(g, indent=1))

    # 2. Watercolour over klein's lines with Nano Banana Pro (paid, Comfy credits).
    photo = klein._upload("out/mill2/mill_pond_clean.png")
    lines = klein._upload("out/mill2/mill_pond_s1.png")
    notes = ("Warm honey Cotswold stone walls, grey stone roof tiles with patches of lichen, the tall "
             "red-brick mill chimney, bright spring greens for the lawn and reeds, white blossom on the "
             "tree by the cottages, bare branches on the tree at the left. The still river in the "
             "foreground mirrors the cottages, chimney and blue sky: paint the reflections in soft, "
             "slightly darker and bluer washes with gentle horizontal ripples, and keep them in the picture.")
    g = {
        "lines": {"class_type": "LoadImage", "inputs": {"image": lines}},
        "load": {"class_type": "LoadImage", "inputs": {"image": photo}},
        "batch": {"class_type": "ImageBatch", "inputs": {"image1": ["lines", 0], "image2": ["load", 0]}},
        "gen": {"class_type": "GeminiImage2Node", "inputs": {
            "prompt": cloud_compare.OVER_LINES.format(notes=notes + " "), "model": "gemini-3-pro-image-preview",
            "seed": 2, "aspect_ratio": "auto", "resolution": "4K", "response_modalities": "IMAGE",
            "images": ["batch", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["gen", 0], "filename_prefix": "sketch_art/cloud"}},
    }
    (OUT / "nanobanana_watercolour_over_klein.json").write_text(json.dumps(g, indent=1))

    # 3. The Gemini fault review (paid, a few cents): photo | painting, whole scene only here.
    g = {
        "photo": {"class_type": "LoadImage", "inputs": {"image": photo}},
        "art": {"class_type": "LoadImage", "inputs": {"image": klein._upload("out/mill2/mill_pond_watercolour_4K.jpg")}},
        "batch": {"class_type": "ImageBatch", "inputs": {"image1": ["photo", 0], "image2": ["art", 0]}},
        "ask": {"class_type": "GeminiNode", "inputs": {"prompt": nanobanana.REVIEW, "model": nanobanana.REVIEWER,
                                                       "seed": 1, "images": ["batch", 0]}},
        "show": {"class_type": "PreviewAny", "inputs": {"source": ["ask", 0]}},
    }
    (OUT / "gemini_review.json").write_text(json.dumps(g, indent=1))
    for f in sorted(OUT.glob("*.json")):
        print(f)


if __name__ == "__main__":
    main()
