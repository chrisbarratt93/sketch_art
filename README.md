# sketch_art

Photo → hand-drawn-looking ink line art, built for a **pen plotter** (AxiDraw,
iDraw, GRBL/Klipper DIY machines). Output is vector SVG strokes, one
Inkscape layer per pen pass. The PNG is only a preview of what the pen draws.

```
uv sync                    # core install
uv sync --extra learned    # + PyTorch for the learned edge detector (experimental, not wired in yet)
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
uv run sketch.py examples/market_hall.jpg                       # line geometry only (current default)
uv run sketch.py examples/market_hall.jpg --style architect     # + selection, weights, hand
uv run sketch.py examples/market_hall.jpg --style loose --stage 3   # urban-sketch look with tone
uv run sketch.py examples/market_hall.jpg --stage 1             # raw foundation
uv run sketch.py examples/market_hall.jpg --seed 7 --style loose    # same drawing, different "hand"
```

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

## Reference / assessment

Style targets: confident single strokes, simplified shapes, detail concentrated at the focal
point, tone by hatching, and plenty of white paper. Arthur L. Guptill's *Rendering in Pen and Ink*
is the classic reference for architectural ink technique. Urban Sketchers work (e.g. Toby Haseler,
Zandro Tumaliuan) is the target look for loose, on-site drawings.

Plotter tool references: vpype, `hatched` (vpype hatching plugin), DrawingBotV3.

Metrics to track per stage (printed by `sketch.py`): stroke count, pen-down length, pen-up travel.
