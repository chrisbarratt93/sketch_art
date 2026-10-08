# sketch_art

Photo → hand-drawn-looking ink line art, built for a **pen plotter** (AxiDraw,
iDraw, GRBL/Klipper DIY machines). Output is vector SVG strokes, one
Inkscape layer per pen pass. The PNG is only a preview of what the pen draws.

```
uv sync                                    # core install
uv sync --extra learned --extra segment    # + PyTorch and the scene/clutter models (needed by klein and ink)
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
uv run sketch.py examples/albert_hall.jpg                       # klein: AI-drawn urban sketch (default)
uv run sketch.py examples/albert_hall.jpg --count 1 --mp 1      # one quick draft (~15 s)
uv run sketch.py examples/albert_hall.jpg --style ink           # ink: faithful trace of learned edges
uv run sketch.py examples/albert_hall.jpg --style ink --stage 3 # ink + hatched tone
uv run sketch.py examples/market_hall.jpg --style plain         # classic edges, geometry only
uv run sketch.py examples/market_hall.jpg --style architect     # + selection, weights, hand
uv run sketch.py examples/market_hall.jpg --style loose --stage 3   # urban-sketch look with tone
uv run sketch.py examples/market_hall.jpg --stage 1             # raw foundation
uv run sketch.py examples/market_hall.jpg --seed 7 --style loose    # same drawing, different "hand"
```

## Klein mode (current best, default)

`--style klein` (`sketchart/klein.py`) hands the drawing itself to FLUX.2 [klein] 4B (Apache-2.0),
an image-editing model run locally in ComfyUI. The tracer can only follow edges; klein has learned
how illustrators draw, so it simplifies, puts detail where it matters, and draws each material with
its own marks.
1. The photo is decluttered (see below) and becomes klein's reference image.
2. klein redraws it from the `urban_rich` prompt: a pen-and-ink urban sketch with medium detail,
   keeping each material's texture (awning stripes, tiles, stonework, grass) and the exact
   composition. Other pen prompts: `--prompt urban | plotter | engraving`, or your own text.
   Painted styles for prints (kept in colour, no plotter SVG): `--prompt watercolour_ink |
   pen_and_wash | pencil`.
3. The drawing is forced to pure black ink on white paper, because klein sometimes lets colour or
   grey wash through.
4. It is traced into plotter strokes (`sketchart/vectorise.py`): centrelines, with the pen chosen by
   line width, and solid areas hatched. Line weights are not yet calibrated against real pens.

Results vary by seed, so it makes `--count` drawings (default 3), one per seed from `--seed`:
`out/<photo>_s1.png` (the drawing), `_s1.svg` (plotter strokes) and `_s1_preview.png`. At the
default 4 megapixels (about 2400 x 1600) each takes ~50 s on an RTX 3080. 4 MP is about the limit
for this model; upscale afterwards for bigger prints.

Setup: ComfyUI must be running (default `http://127.0.0.1:8188`, or set `COMFY_URL`), with
`diffusion_models/flux-2-klein-4b.safetensors`, `text_encoders/qwen_3_4b.safetensors` and
`vae/flux2-vae.safetensors` (from Hugging Face `Comfy-Org/vae-text-encorder-for-flux-klein-4b`) in its
models folder. From WSL, requests go through Windows' `curl.exe`.
`experiments/comfy_ink.py` and `experiments/compare_sheet.py` hold the model and prompt comparisons
(2026-10-08) that led here.

## Ink mode

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

### Scene layers and clutter removal (klein and ink modes)

```
uv sync --extra learned --extra segment
uv run sketch.py examples/albert_hall.jpg --style ink --scene --declutter inpaint
uv run sketch.py examples/albert_hall.jpg --style ink --declutter drop --clutter "crane,person"
uv run sketch.py examples/cotswold_street.webp --style ink --scene --debug    # + _scene.jpg overlay
uv run sketch.py examples/albert_hall.jpg --clutter "crane,person,car"        # klein: what to remove
```

klein mode always declutters with `inpaint` unless you pass `--declutter off`. `--scene` applies to
ink mode only.

`sketchart/scene.py` uses models from the Hugging Face hub (~3GB, downloaded on first use). Adds about
10–15 s per photo on an RTX 3080.
- `--scene`: Mask2Former (Swin-L, ADE20K) labels each pixel sky / building / vegetation / hills /
  ground / other, and `REGION_STYLE` in `ink.py` sets thresholds, shortest mark, smoothing and
  heaviest pen per layer. Buildings are traced as before. Foliage and hills keep only their bolder
  lines, and never go to the heavy pen. The ground is drawn lightly, and open sky stays paper.
- `--declutter inpaint|drop`: Grounding DINO finds `--clutter` prompts (default: cranes, people,
  vehicles, street lamps, signs, tables, chairs and so on), and SAM 2.1 masks them. `inpaint` fills
  them with LaMa before tracing, so the building behind is drawn. `drop` leaves them as paper.
  Detections of "street lamp" or "sign" that the segmentation calls building are kept, because
  cast-iron columns get mistaken for lamps.
- Known gaps: black bollards aren't detected. A thin lamp post right against a facade can be kept
  as building. Check model licences before selling prints: Grounding DINO and SAM 2.1 are
  Apache-2.0 and LaMa's code is Apache-2.0, but the Mask2Former checkpoint is listed as "other" and
  the `fashn-ai/LaMa` repackaging has no licence tag.

## Refining generated drawings (`sketchart/refine.py`)

Makes the marks in a generated ink drawing look deliberate. It vectorises the drawing into
strokes, keeping each one's drawn width. Width is measured from the ink a line carries, so faint
lines stay light. Cleanup passes then work on the strokes, and the strokes are redrawn at exact
widths. Dense dark texture (foliage masses, black beams) is kept as fill using the drawing's own
pixels. `PRESETS["tidy"]`:
- drop fragments under 5px
- re-join aligned breaks up to 6px
- smooth jitter between real corners
- straighten runs within 0.7px of a line
- draw each stroke at a steady width with tapered ends

`uv run review_klein.py -v name:param=value,...` compares variants on fixed 1:1 crops of
`examples/drawings/*_klein.png` (sheets in `out/review_klein/`, snapshot in `docs/refine/`).
Metrics: strokes, short marks, small loops, wobble, jitter, and `diff`, the % difference between
the re-rendered drawing and klein's.

| (Albert Hall) | strokes | short marks | loops | jitter | diff vs klein |
|---|---|---|---|---|---|
| plain re-render | 20,295 | 58% | 51 | 0.23 | 2.2% |
| tidy | 9,137 | 16% | 1 | 0.16 | 3.2% |

Known gaps:
- Foliage is mostly dense fill, which stroke cleanup doesn't touch. It needs redrawing as
  deliberate leaf-cluster marks.
- `min_len` also removes genuine short marks in grass.
- The faintest texture (fine stone hatching) is lost at the 180 ink threshold.

## Review harness

`uv run review.py` renders fixed close-up crops (tracery, lettering, stonework, skyline, panels,
gable, timber gable, foliage) from the three test photos at print zoom. It writes them to
`out/review/` as photo | baseline | variants sheets and prints metrics (`sketchart/metrics.py`):
- `doubled`: share of line length that runs beside a parallel line, i.e. both outlines of one
  feature.
- `wobble`: how far near-straight strokes stray from a true straight line.
- `short`: share of tiny marks.
- `loops`: small closed blobs.
- `invented` / `missed`: fidelity against the uncleaned trace.

Try a variant with `-v name:param=value,...`, e.g. `-v c4:collapse=4`. TEED maps are cached, so
variants take seconds after the first run.

Cleanup passes (`sketchart/cleanup.py`):
- `collapse_pairs` (ink default 3px): where two outlines run parallel and face each other
  within 3px, draw their midline once and drop the rounded end caps.

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
2. **Geometry + stylise line** in `sketchart/stylise.py`. `--style plain` runs only the geometry steps and draws every line at one weight. That makes it possible to
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
