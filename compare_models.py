"""Run several edge models on one photo and lay the results side by side.

For each model this writes the raw edge map and the ink drawing traced from
it, then a labelled contact sheet of all of them. Optionally adds the scene
segmentation and depth map. Needs `uv sync --extra hub` (see MODELS.md).

    uv run compare_models.py examples/market_hall.jpg
    uv run compare_models.py examples/cotswold_street.webp --models teed,lineart,anyline --scene
    uv run compare_models.py photo.jpg --max-side 0          # full resolution (GPU advised)
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from sketchart import hub
from sketchart.ink import InkParams, ink
from sketchart.output import render, write_svg
from sketchart.trace import order, travel_stats


def label(img, text, sub=""):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    bar = np.full((44, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def sheet(tiles, cols, width=900):
    tiles = [cv2.resize(t, (width, round(t.shape[0] * width / t.shape[1])), interpolation=cv2.INTER_AREA)
             for t in tiles]
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((h - t.shape[0], width, 3), 255, np.uint8)]) for t in tiles]
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--models", default=",".join(hub.NAMES), help="comma list from: " + ",".join(hub.NAMES))
    ap.add_argument("--max-side", type=int, default=1400, help="0 = full photo resolution")
    ap.add_argument("--scene", action="store_true", help="also show segmentation layers and depth")
    ap.add_argument("--seg-model", choices=("tiny", "large"), default="tiny")
    ap.add_argument("-o", "--out", default=None, help="output folder (default: out/compare/<image name>)")
    args = ap.parse_args()

    names = [n for n in args.models.split(",") if n]
    if bad := set(names) - set(hub.NAMES):
        ap.error(f"unknown model(s): {', '.join(sorted(bad))}")
    out = Path(args.out or Path("out") / "compare" / Path(args.image).stem)
    out.mkdir(parents=True, exist_ok=True)

    print(f"device: {hub.device()}")
    maps, drawings, stats = [], [], {}
    for name in names:
        t = time.time()
        try:
            s = ink(args.image, InkParams(max_side=args.max_side, edge_model=name))
        except Exception as e:  # one bad download shouldn't sink the whole comparison
            print(f"{name}: failed: {e}")
            continue
        secs = time.time() - t
        h, w = s["gray"].shape
        layers = [("heavy", 1.5, order(s["heavy"])), ("medium", 1.0, order(s["medium"])),
                  ("fine", 0.6, order(s["fine"]))]
        st = travel_stats([p for _, _, p in layers])
        n = sum(len(p) for _, _, p in layers)
        stats[name] = {"seconds": round(secs, 1), "strokes": n, **st}
        edge = (255 * (1 - cv2.resize(s["prob"], (w, h), interpolation=cv2.INTER_AREA))).astype(np.uint8)
        draw = render((w, h), layers)
        cv2.imwrite(str(out / f"{name}_edges.png"), edge)
        cv2.imwrite(str(out / f"{name}_ink.png"), draw)
        write_svg(str(out / f"{name}.svg"), (w, h), layers)
        sub = f"{n} strokes, {secs:.0f}s"
        maps.append(label(edge, f"{name}: edge map", sub))
        drawings.append(label(draw, f"{name}: ink", sub))
        print(f"{name}: {json.dumps(stats[name])}")

    if maps:
        cv2.imwrite(str(out / "sheet_edges.jpg"), sheet(maps, 2), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(out / "sheet_ink.jpg"), sheet(drawings, 2), [cv2.IMWRITE_JPEG_QUALITY, 88])

    if args.scene:
        from sketchart.edges import load_gray
        from sketchart.segment import nearness, preview, segment
        color, _ = load_gray(args.image, args.max_side)
        masks, _, _ = segment(color, args.seg_model)
        near = nearness(color)
        layers_img = preview(color, masks)
        depth_img = cv2.applyColorMap((255 * near).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(out / "scene_layers.png"), layers_img)
        cv2.imwrite(str(out / "scene_depth.png"), depth_img)
        cv2.imwrite(str(out / "sheet_scene.jpg"),
                    sheet([label(color, "photo"), label(layers_img, f"OneFormer {args.seg_model}: layers"),
                           label(depth_img, "Depth Anything V2 Small", "bright = near")], 3))

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
