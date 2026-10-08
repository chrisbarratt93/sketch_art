# Example line drawings (for tracing and clean-up work)

Fixed inputs for working on the tracing/clean-up stage, so it can be developed and compared without
generating new drawings. All were made on 2026-10-08 with the pipeline in `sketch.py` and
`experiments/cloud_compare.py`.

| File | What it is |
|---|---|
| `*_clean.jpg` | The photo after clutter removal (`--declutter inpaint`): what the drawing models saw. Reference for checking that clean-up keeps real detail. |
| `albert_hall_klein.png`, `market_hall_klein.png`, `cotswold_street_klein.png` | FLUX.2 klein 4B, `urban_rich` prompt, seed 1, 4 MP, after `klein.black_and_white`. |
| `mill_pond_klein.png` | Same, mill pond photo, seed 1 (with the no-invented-text rule in the prompt). |
| `park_klein.png` | Same, park photo, seed 2. The photo is small (1060×700), so klein upscaled it ~2.4×. |
| `mill_pond_klein_raw.png`, `park_klein_raw.png` | klein's output before `black_and_white`, stored as greyscale. Keeps the tonal information the threshold throws away. |
| `cotswold_street_nanobananapro.png` | Nano Banana Pro (cloud), same photo and prompt, 2K, after `black_and_white`. A useful contrast: cleaner, more economical lines (~17k pen strokes vs ~44k for klein). |

## The problem to work on

The drawings read well overall, but the detailed areas look AI-made rather than hand-drawn:
- Dense foliage turns into noisy speckle and broken fragments instead of deliberate leaf-cluster
  marks.
- Stonework, tiles and glazing get jittery, wobbly short marks with no consistent stroke direction.
- Many tiny disconnected fragments and dots, which also make the plotter SVG huge
  (40,000–60,000 strokes per drawing; `sketchart/vectorise.py`).

Directions to try: fragment removal and stroke merging in vector space, smoothing and straightening
strokes, replacing noisy regions with regular hatching, or using a cloud model (Comfy credits) to
redraw or clean the drawing given the existing line work.
