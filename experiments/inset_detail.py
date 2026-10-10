"""Draw a small detail separately and set it into a full drawing (e.g. rooftop statues).

The detail is cropped from the photo, drawn by itself (so the model sees it large), and then
placed back into the full drawing: its silhouette is cut from the photo with Grounding DINO +
SAM 2.1, the crop is aligned to the full drawing locally (the drawing models shift things by a
few pixels), and inside the silhouette the full drawing is replaced by the detail sketch.

    uv run --all-extras experiments/inset_detail.py DRAWING PHOTO --detail SKETCH X0 Y0 X1 Y1 [--detail ...] -o OUT

X0 Y0 X1 Y1 is the crop box in photo pixels that SKETCH was drawn from.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sketchart import scene  # noqa: E402


def edges(gray):
    return cv2.GaussianBlur(cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120).astype(np.float32), (0, 0), 2)


def local_shift(drawing, photo_crop, box, reach=25):
    """Offset (dx, dy), within +-reach px, that best lines the photo crop's edges up with the
    drawing at `box`. A bounded search: an open-ended correlation latches onto repeated features
    (windows, columns) far away."""
    x0, y0, _, _ = box
    h, w = drawing.shape
    pe = edges(photo_crop)
    ch, cw = pe.shape
    X0, Y0 = max(0, x0 - reach), max(0, y0 - reach)
    region = cv2.GaussianBlur((255 - drawing[Y0:min(h, y0 + ch + reach), X0:min(w, x0 + cw + reach)])
                              .astype(np.float32), (0, 0), 2)
    score = cv2.matchTemplate(region, pe, cv2.TM_CCOEFF_NORMED)
    _, _, _, (bx, by) = cv2.minMaxLoc(score)
    return X0 + bx - x0, Y0 + by - y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drawing")
    ap.add_argument("photo")
    ap.add_argument("--detail", nargs=5, action="append", metavar=("SKETCH", "X0", "Y0", "X1", "Y1"), required=True)
    ap.add_argument("--prompt", default="statue", help="what to cut out of each crop")
    ap.add_argument("--grow", type=int, default=3, help="widen the silhouette by this (drawing px)")
    ap.add_argument("--weight", type=float, default=0.6,
                    help="detail line weight, in drawing px of extra pen width (0 = as drawn)")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    drawing = cv2.imread(args.drawing, cv2.IMREAD_GRAYSCALE)
    photo = cv2.imread(args.photo)
    h, w = drawing.shape
    sx, sy = w / photo.shape[1], h / photo.shape[0]
    out = drawing.astype(np.float32)
    debug = cv2.cvtColor(drawing, cv2.COLOR_GRAY2BGR)

    for sketch_path, *b in args.detail:
        x0, y0, x1, y1 = map(int, b)
        crop = photo[y0:y1, x0:x1]
        # Silhouette from the photo crop, at 3x so SAM sees enough pixels.
        big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        mask, labels = scene.clutter_mask(big, [args.prompt], keep_features=False)
        if not mask.any():
            print(f"{sketch_path}: no '{args.prompt}' found in the crop, skipped")
            continue
        # Everything at the drawing's scale.
        X0, Y0 = round(x0 * sx), round(y0 * sy)
        cw, ch = round((x1 - x0) * sx), round((y1 - y0) * sy)
        mask = cv2.resize(mask.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_AREA) > 0
        sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
        # The detail was drawn several times larger than it ends up, so its pen lines would shrink to
        # grey hairlines. Thicken them first (in proportion to the shrink), then restore black.
        shrink = sketch.shape[1] / cw
        k = max(1, round(shrink * args.weight))
        sketch = cv2.erode(sketch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        sketch = cv2.resize(sketch, (cw, ch), interpolation=cv2.INTER_AREA)
        sketch = (255 * np.clip((sketch.astype(np.float32) - 40) / 170, 0, 1)).astype(np.uint8)
        crop_g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (cw, ch), interpolation=cv2.INTER_AREA)
        # The sketch keeps the crop's framing only roughly: align it to the crop, then the crop to the drawing.
        win = cv2.createHanningWindow((cw, ch), cv2.CV_32F)
        (ax, ay), _ = cv2.phaseCorrelate(edges(crop_g), cv2.GaussianBlur((255 - sketch).astype(np.float32), (0, 0), 2), win)
        sketch = cv2.warpAffine(sketch, np.float32([[1, 0, -ax], [0, 1, -ay]]), (cw, ch), borderValue=255)
        dx, dy = local_shift(drawing, crop_g, (X0, Y0, X0 + cw, Y0 + ch))
        X0, Y0 = X0 + dx, Y0 + dy
        print(f"{Path(sketch_path).name}: sketch offset ({ax:.1f}, {ay:.1f}), drawing offset ({dx}, {dy}), "
              f"silhouette {mask.mean():.0%} of crop")

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.grow + 1,) * 2)
        m = cv2.dilate(mask.astype(np.uint8), k).astype(np.float32)
        m = cv2.GaussianBlur(m, (0, 0), 1.0)  # soft edge so no hard seam
        region = out[Y0:Y0 + ch, X0:X0 + cw]
        # Inside the silhouette: blank paper, then the detail's ink.
        cleared = region * (1 - m) + 255 * m
        out[Y0:Y0 + ch, X0:X0 + cw] = np.minimum(cleared, sketch * m + 255 * (1 - m))
        cnts, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug[Y0:Y0 + ch, X0:X0 + cw], cnts, -1, (0, 0, 255), 2)

    out = out.clip(0, 255).astype(np.uint8)
    cv2.imwrite(args.out, out)
    cv2.imwrite(str(Path(args.out).with_suffix("")) + "_mask.png", debug)
    print(args.out)


if __name__ == "__main__":
    main()
