# sketch_art

Photo → hand-drawn-looking ink line art, built for a **pen plotter** (AxiDraw,
iDraw, GRBL/Klipper DIY machines). Output is vector SVG strokes, one
Inkscape layer per pen pass. The PNG is only a preview of what the pen draws.

```
pip install -r requirements.txt
python sketch.py examples/market_hall.jpg --debug           # stage 2 -> out/market_hall.svg, _preview.png
python sketch.py examples/market_hall.jpg --stage 1         # foundation only
python sketch.py examples/market_hall.jpg --seed 7          # same drawing, different "hand"
```

## Stages

1. **Foundation (done, first pass)**: extract the lines an artist would draw.
   - *Contours* (step edges between tones) come from Canny on a bilateral-flattened, CLAHE image.
   - *Lines* (thin bars such as glazing bars, mullions and railings) come from Hessian ridge/valley
     detection with non-max suppression. You get one stroke down the middle of each bar, not the
     two-sided "sausage" plain edge detection produces.
   - Contours that only flank a detected bar are dropped. Masks are traced into polylines,
     stitched end-to-end, simplified, and ordered to cut pen-up travel.
2. **Stylise line (done, first pass)** in `sketchart/stylise.py`. Output is three layers
   (`heavy` / `medium` / `fine`), one per pen.
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
   - Known gaps: the foreground (pavement, bollards) is still busier than an artist would draw
     it, and the small curves in arched windows can look scribbly.
3. **Tone**: directional hatching and cross-hatching driven by a tone map, following surface
   orientation where possible. Keep the sky mostly empty.
4. **Plot**: page sizing in mm, pen-width-aware hatch spacing, layer per pen, vpype/AxiDraw export.

## Reference / assessment

Style targets: confident single strokes, simplified shapes, detail concentrated at the focal
point, tone by hatching, and plenty of white paper. Arthur L. Guptill's *Rendering in Pen and Ink*
is the classic reference for architectural ink technique. Urban Sketchers work (e.g. Toby Haseler,
Zandro Tumaliuan) is the target look for loose, on-site drawings.

Plotter tool references: vpype, `hatched` (vpype hatching plugin), DrawingBotV3.

Metrics to track per stage (printed by `sketch.py`): stroke count, pen-down length, pen-up travel.
