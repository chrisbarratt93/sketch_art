"""Labelled comparison sheet per photo: every stage and style side by side.

    uv run --all-extras experiments/compare_sheet.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GEN = Path("out/gen")
OUT = Path("out/compare")
PHOTOS = {"albert_hall": "examples/albert_hall.jpg", "market_hall": "examples/market_hall.jpg",
          "cotswold_street": "examples/cotswold_street.webp"}
FONT = "/usr/share/fonts/truetype/lato/Lato-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"
COLS, PANEL, GAP = 4, 640, 24
INK, MUTED, RULE = (25, 25, 25), (105, 105, 105), (210, 210, 210)


def sections(p):
    g = lambda f: GEN / f  # noqa: E731
    return [
        ("A. Preparing the photo", "Shared by every route below.", [
            ("A1", "Original photo", "As taken.", PHOTOS[p]),
            ("A2", "Clutter removed", "Cranes, people, lamp posts found by Grounding DINO + SAM 2.1, painted out by LaMa.", g(f"inputs/{p}_clean.jpg")),
            ("A3", "Scene layers", "Mask2Former. Red building, green foliage, blue sky, grey ground; magenta = removed.", g(f"inputs/{p}_scene.jpg")),
            ("A4", "TEED edge map", "Learned edge detector. Feeds the tracer (B1) and SDXL (B2).", g(f"inputs/{p}_teed.png")),
        ]),
        ("B. Which method draws it", "Same cleaned photo (A2). B2-B4 use the 'architect' prompt.", [
            ("B1", "Tracer (current sketch.py)", "Ink mode, --scene --declutter inpaint. No AI drawing.", g(f"inputs/{p}_preview.png")),
            ("B2", "SDXL + ControlNet", "Juggernaut XL guided by the TEED lines (A4). ~17 s.", g(f"{p}_sdxl_architect_0.png")),
            ("B3", "Qwen-Image-Edit 2511", "Instruction editing model. ~65 s. Now deleted.", g(f"{p}_qwen_architect_0.png")),
            ("B4", "FLUX.2 klein 4B", "Instruction editing model. ~13 s. Chosen.", g(f"{p}_klein_architect_0.png")),
        ]),
        ("C. FLUX.2 klein prompt styles", "Same model, different instructions. 1 megapixel unless noted.", [
            ("C1", "Urban sketch", "Loose fineliner, fading edges. Can add grey wash.", g(f"{p}_klein_urban_0.png")),
            ("C2", "Engraving", "Dense hatching and cross-hatching.", g(f"{p}_klein_engraving_0.png")),
            ("C3", "Plotter", "Crisp black lines, sparse hatching, no grey.", g(f"{p}_klein_plotter_0.png")),
            ("C4", "Plotter, 2 megapixels", "Same prompt at twice the pixels: finer lines. ~19 s.", g(f"{p}_klein_plotter_2mp_0.png")),
        ]),
        ("D. Back to pen strokes", "What the plotter would draw.", [
            ("D1", "Tracer plot", "B1 as plotted (vector SVG).", g(f"inputs/{p}_preview.png")),
            ("D2", "klein plotter 2MP, vectorised", "C4 traced into pen strokes. Line weights not yet tuned.", g(f"plot/{p}_klein_plotter_2mp_0_plot_preview.png")),
        ]),
    ]


def wrap(draw, text, font, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def sheet(p):
    f_title, f_sec, f_sub = ImageFont.truetype(BOLD, 44), ImageFont.truetype(BOLD, 30), ImageFont.truetype(FONT, 22)
    f_tag, f_lab, f_cap = ImageFont.truetype(BOLD, 24), ImageFont.truetype(BOLD, 24), ImageFont.truetype(FONT, 19)
    secs = sections(p)
    aspect = Image.open(PHOTOS[p]).size
    ph = round(PANEL * aspect[1] / aspect[0])
    W = COLS * PANEL + (COLS + 1) * GAP
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    cap_h = 34 + 24 * max(len(wrap(probe, c, f_cap, PANEL)) for s in secs for _, _, c, _ in s[2])
    sec_h = 90 + ph + cap_h + GAP
    H = 110 + len(secs) * sec_h + GAP
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((GAP, 30), p.replace("_", " ").title(), font=f_title, fill=INK)
    y = 110
    for title, sub, panels in secs:
        d.line([(GAP, y), (W - GAP, y)], fill=RULE, width=2)
        d.text((GAP, y + 14), title, font=f_sec, fill=INK)
        d.text((GAP + d.textlength(title, font=f_sec) + 18, y + 22), sub, font=f_sub, fill=MUTED)
        y0 = y + 66
        for i, (tag, label, cap, path) in enumerate(panels):
            x = GAP + i * (PANEL + GAP)
            d.text((x, y0), tag, font=f_tag, fill=(180, 60, 40))
            d.text((x + d.textlength(tag, font=f_tag) + 10, y0), label, font=f_lab, fill=INK)
            im = Image.open(path).convert("RGB")
            im.thumbnail((PANEL, ph), Image.LANCZOS)
            box = (x, y0 + 34)
            img.paste(im, (box[0] + (PANEL - im.width) // 2, box[1] + (ph - im.height) // 2))
            d.rectangle([box[0] - 1, box[1] - 1, box[0] + PANEL, box[1] + ph], outline=RULE)
            for k, line in enumerate(wrap(d, cap, f_cap, PANEL)):
                d.text((x, box[1] + ph + 8 + 24 * k), line, font=f_cap, fill=MUTED)
        y += sec_h
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{p}_comparison.jpg"
    img.save(dst, quality=90)
    return dst



def texture_sheet():
    """Round 2: prompts between urban sketch (C1) and engraving (C2), at 4 megapixels."""
    cols = [("C1", "Urban sketch", "1 MP. Previous favourite.", "{p}_klein_urban_0.png"),
            ("E1", "Urban texture", "4 MP. Texture wherever it appears in the photo.", "{p}_klein_urban_texture_4mp_0.png"),
            ("E2", "Urban rich, seed 1", "4 MP. 'Medium density', material-by-material texture.", "{p}_klein_urban_rich_4mp_0.png"),
            ("E3", "Urban rich, seed 2", "4 MP. Same prompt, different random seed.", "{p}_klein_urban_rich_4mp_1.png"),
            ("C2", "Engraving", "1 MP. Previous dense option.", "{p}_klein_engraving_0.png")]
    f_title, f_sec, f_sub = ImageFont.truetype(BOLD, 44), ImageFont.truetype(BOLD, 30), ImageFont.truetype(FONT, 22)
    f_tag, f_cap = ImageFont.truetype(BOLD, 24), ImageFont.truetype(FONT, 19)
    panel, ph = 560, 400
    W = len(cols) * panel + (len(cols) + 1) * GAP
    row_h = 60 + ph + GAP
    H = 110 + 110 + len(PHOTOS) * row_h + GAP
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((GAP, 30), "Between urban sketch and engraving", font=f_title, fill=INK)
    y = 110
    for i, (tag, label, cap, _) in enumerate(cols):
        x = GAP + i * (panel + GAP)
        d.text((x, y), tag, font=f_tag, fill=(180, 60, 40))
        d.text((x + d.textlength(tag, font=f_tag) + 10, y), label, font=f_tag, fill=INK)
        for k, line in enumerate(wrap(d, cap, f_cap, panel)):
            d.text((x, y + 34 + 24 * k), line, font=f_cap, fill=MUTED)
    y += 110
    for p in PHOTOS:
        d.line([(GAP, y), (W - GAP, y)], fill=RULE, width=2)
        d.text((GAP, y + 12), p.replace("_", " ").title(), font=f_sec, fill=INK)
        for i, (_, _, _, pat) in enumerate(cols):
            x = GAP + i * (panel + GAP)
            im = Image.open(GEN / pat.format(p=p)).convert("RGB")
            im.thumbnail((panel, ph), Image.LANCZOS)
            img.paste(im, (x + (panel - im.width) // 2, y + 60 + (ph - im.height) // 2))
        y += row_h
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / "texture_round2.jpg", quality=90)
    return OUT / "texture_round2.jpg"



def grid_sheet(title, cols, dst, panel=760, ph=520):
    """Rows = photos, columns = (tag, label, caption, path pattern with {p})."""
    f_title, f_sec = ImageFont.truetype(BOLD, 44), ImageFont.truetype(BOLD, 30)
    f_tag, f_cap = ImageFont.truetype(BOLD, 24), ImageFont.truetype(FONT, 19)
    W = len(cols) * panel + (len(cols) + 1) * GAP
    row_h = 60 + ph + GAP
    H = 220 + len(PHOTOS) * row_h + GAP
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((GAP, 30), title, font=f_title, fill=INK)
    for i, (tag, label, cap, _) in enumerate(cols):
        x = GAP + i * (panel + GAP)
        d.text((x, 110), tag, font=f_tag, fill=(180, 60, 40))
        d.text((x + d.textlength(tag, font=f_tag) + 10, 110), label, font=f_tag, fill=INK)
        for k, line in enumerate(wrap(d, cap, f_cap, panel)):
            d.text((x, 144 + 24 * k), line, font=f_cap, fill=MUTED)
    y = 220
    for p in PHOTOS:
        d.line([(GAP, y), (W - GAP, y)], fill=RULE, width=2)
        d.text((GAP, y + 12), p.replace("_", " ").title(), font=f_sec, fill=INK)
        for i, (_, _, _, pat) in enumerate(cols):
            path = Path(pat.format(p=p))
            if not path.exists():
                continue
            x = GAP + i * (panel + GAP)
            im = Image.open(path).convert("RGB")
            im.thumbnail((panel, ph), Image.LANCZOS)
            img.paste(im, (x + (panel - im.width) // 2, y + 60 + (ph - im.height) // 2))
        y += row_h
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / dst, quality=90)
    return OUT / dst


def default_sheet():
    return grid_sheet("Default style (klein, urban rich, black and white): pick a seed", [
        (f"S{k}", f"Seed {k}", "4 MP, black and white. Plotter SVG alongside.", f"out/klein/{{p}}_s{k}.png")
        for k in (1, 2, 3)], "default_seeds.jpg")


def painted_sheet():
    return grid_sheet("Painted styles (klein, colour, for print only)", [
        ("P1", "Watercolour and ink", "Fineliner line work with loose transparent washes.", "out/painted/{p}_watercolour_ink_s1.png"),
        ("P2", "Pen and wash", "Sepia dip-pen line with brushed sepia and grey washes.", "out/painted/{p}_pen_and_wash_s1.png"),
        ("P3", "Pencil", "Graphite architectural sketch with hatched tone.", "out/painted/{p}_pencil_s1.png"),
        ("S1", "Default pen (for reference)", "Urban rich, black and white.", "out/klein/{p}_s1.png")], "painted_styles.jpg")


if __name__ == "__main__":
    for p in PHOTOS:
        print(sheet(p))
    print(texture_sheet())
    print(default_sheet())
    print(painted_sheet())
