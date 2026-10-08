"""Write strokes as plotter-ready SVG and as a raster preview."""

import cv2
import numpy as np


def _d(p):
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p)


def write_svg(path, size, layers):
    """`layers` is a list of (name, stroke_width_px, paths). Each becomes an
    Inkscape layer, so a plotter driver (e.g. AxiDraw) can plot them as
    separate passes or with different pens."""
    w, h = size
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
           f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" fill="white"/>']
    for i, (name, width, paths) in enumerate(layers, 1):
        out.append(f'<g inkscape:groupmode="layer" inkscape:label="{i}-{name}" '
                   f'fill="none" stroke="black" stroke-width="{width}" '
                   f'stroke-linecap="round" stroke-linejoin="round">')
        out += [f'<path d="{_d(p)}"/>' for p in paths]
        out.append("</g>")
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def render(size, layers, scale=2):
    """Anti-aliased raster preview of what the pen would draw."""
    w, h = size
    canvas = np.full((h * scale, w * scale), 255, np.uint8)
    for _, width, paths in layers:
        t = max(1, round(width * scale))
        for p in paths:
            pts = np.round(p * scale * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, 0, t, cv2.LINE_AA, shift=4)
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
