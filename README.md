# sketch_art

Photo → hand-drawn-looking ink line art, built for a **pen plotter** (AxiDraw,
iDraw, GRBL/Klipper DIY machines). Output is vector SVG strokes, one
Inkscape layer per pen pass. The PNG is only a preview of what the pen draws.

```
uv sync                    # core install
uv sync --extra learned    # + PyTorch for the learned edge detector (needed by the default ink style)
uv sync --extra hub        # + Hugging Face models: other edge detectors, segmentation, depth (MODELS.md)
```

On Linux, PyPI's PyTorch wheel bundles CUDA (several GB). For a CPU-only install, add this to
`pyproject.toml` before `uv sync --extra learned`:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

```
uv run sketch.py examples/albert_hall.jpg                       # ink: faithful trace of learned edges (default)
uv run sketch.py examples/albert_hall.jpg --stage 3             # ink + hatched tone
uv run sketch.py examples/market_hall.jpg --style plain         # classic edges, geometry only
uv run sketch.py examples/market_hall.jpg --style architect     # + selection, weights, hand
uv run sketch.py examples/market_hall.jpg --style loose --stage 3   # urban-sketch look with tone
uv run sketch.py examples/market_hall.jpg --stage 1             # raw foundation
uv run sketch.py examples/market_hall.jpg --seed 7 --style loose    # same drawing, different "hand"
```

## Ink mode (current best)

`--style ink` (`sketchart/ink.py`) runs TEED, a small learned edge detector trained on
human-annotated edges in urban photos (`sketchart/teed.py`, MIT, weights fetched on first use).
Its edge map already reads like a pen drawing, so ink mode traces it as faithfully as possible:
- The confident part of the map is skeletonised to centrelines and traced into strokes, short
  marks included.
- Strokes get stair-step smoothing only. There is no straightening, merging or snapping.
- Line weight comes from how boldly the network drew each line: the top 15% go to the `heavy`
  pen layer and the bottom 35% to `fine`.

Takes ~20–35 s per photo on CPU. TEED uses an NVIDIA GPU automatically when PyTorch can see
one. On Windows, PyPI's PyTorch is CPU-only, so point uv at a CUDA build (the same
`[tool.uv.sources]` pattern as the CPU-only snippet above, with
`url = "https://download.pytorch.org/whl/cu128"` and `marker = "sys_platform == 'win32'"`).
Check with `uv run python -c "from sketchart import teed; print(teed.DEVICE)"`.
For print-quality output use the full photo: `--max-side 0`. The classic pipeline below is still available via `--style`.

## Stages

1. **Foundation**: extract the lines an artist would draw.
   - Detection runs at 2× the photo size (smoothing at 1×, then upsampled), so strokes land to
     half a pixel and 1–2px bars are resolved. Canny runs on float gradients, because 8-bit blurs
     quantise away gentle gradients at 2×.
   - The ridge detector rejects the *shoulders* of things wider than a bar (panels, walls), so wide
     light areas aren't mistaken for bars.
   - Tracing re-links skeleton pieces through junctions by good continuation (best match first),
     and bridges gaps where a crossing bar interrupted an edge, as long as the two ends line up.
   - *Contours* (step edges between tones) come from Canny on a bilateral-flattened, CLAHE image.
   - *Lines* (thin bars such as glazing bars, mullions and railings) come from Hessian ridge/valley
     detection with non-max suppression. You get one stroke down the middle of each bar, not the
     two-sided "sausage" plain edge detection produces.
   - Contours that only flank a detected bar are dropped. Masks are traced into polylines,
     stitched end-to-end, simplified, and ordered to cut pen-up travel.
2. **Geometry + stylise line** in `sketchart/stylise.py`. `--style plain` (the current default)
   runs only the geometry steps and draws every line at one weight. That makes it possible to
   judge detection and geometry without selection or hand effects getting in the way.
   - *Geometry*:
     - Collinear fragments are merged.
     - Straight edges whose ends curl into rounded corners are squared off.
     - Vanishing points are found by RANSAC, and lines within 2° of one are snapped onto it.
     - Line ends are extended or trimmed to meet at real corners. Long reaches are only allowed
       where both lines stop short of a shared corner, e.g. a gable apex hidden by a finial.
   - `--style architect`: steady hand, joined corners cross by 1.5–5px, light pruning, sky kept.
   - `--style loose`: the original urban-sketch look.

   Styled output is three layers (`heavy` / `medium` / `fine`), one per pen.
   - *Decisive strokes*: paths are split at corners. Near-straight runs become true straight
     strokes, and collinear fragments are merged across gaps, so broken building edges become
     one line.
   - *Selection*: strokes are scored by contrast × √length × closeness to an automatically found
     focal point, and the weakest 30% are dropped. An irregular, noisy vignette lets the drawing
     peter out towards the edges.
   - *Indication*: inside runs of closely spaced parallel strokes (glazing bars), about a third
     are skipped at random. The edges of each run are always kept, so the pattern still reads.
   - *Weight*: heavy goes to long strokes with very different tone on either side (silhouettes,
     major shadow lines), weighted by focus. Heavy strokes also get a second, slightly offset
     pass, so the effect works with a single pen.
   - *The hand*: corners overshoot or stop short, lines sit up to about ±0.35° off true, long
     lines bow slightly, and every line has a low-frequency wobble. All of this is seeded and
     reproducible.
   - *Ground fade*: the drawn area is shorter below the focal point than above it, and heavy
     lines aren't allowed near the fading edge, so the foreground dissolves into a few loose
     lines.
   - Known gap: the small curves in arched windows can look scribbly.
3. **Tone (done, first pass)** in `sketchart/tone.py`. Output is three extra layers,
   `hatch1`–`hatch3`.
   - Darkness comes from a bilateral-smoothed image stretched to the photo's own range. Lights
     stay white paper: hatching starts at 50% darkness, cross-hatching at 66%, and a third pass
     between the first set's lines starts at 80%.
   - Regions are closed and then opened. Closing bridges thin light bars, so a glazed wall reads
     as one toned area while wide light panels stay white. Opening removes slivers too thin to
     hatch.
   - Strokes are laid in *patches*: bands of 5–10 adjacent lines share their break points, giving
     blocks of short parallel strokes with ragged ends and small gaps. Each stroke gets angle
     jitter and a slight wobble.
   - The sky (smooth area connected to the top edge) stays empty. Hatching stops sooner than the
     line work at the vignette, so the drawing fades from tone to line to paper.
   - Known gaps: all hatching uses one direction. Artists often turn it to follow each plane, e.g.
     along the perspective lines of a wall. Pen-up travel is high (~0.6× pen-down), which a
     boustrophedon order within each patch would fix.
4. **Plot**: page sizing in mm, pen-width-aware hatch spacing, layer per pen, vpype/AxiDraw export.

## Hugging Face models

[MODELS.md](MODELS.md) covers the optional Hub models: alternative edge maps for ink mode
(`--edge-model lineart|anyline|hed|…`), clutter removal by segmentation (`--remove people,vehicles`),
depth-based line weight (`--depth-fade`), and `compare_models.py`, which lays every model's result
side by side for one photo.

## Ideas

See [IDEAS.md](IDEAS.md) for planned work: layer separation, clutter removal, foliage and
landscape handling.

## Reference / assessment

Style targets: confident single strokes, simplified shapes, detail concentrated at the focal
point, tone by hatching, and plenty of white paper. Arthur L. Guptill's *Rendering in Pen and Ink*
is the classic reference for architectural ink technique. Urban Sketchers work (e.g. Toby Haseler,
Zandro Tumaliuan) is the target look for loose, on-site drawings.

Plotter tool references: vpype, `hatched` (vpype hatching plugin), DrawingBotV3.

Metrics to track per stage (printed by `sketch.py`): stroke count, pen-down length, pen-up travel.
