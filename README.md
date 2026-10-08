# sketch_art

Photo → hand-drawn-looking ink line art, built for a **pen plotter** (AxiDraw,
iDraw, GRBL/Klipper DIY machines). Output is vector SVG strokes, one
Inkscape layer per pen pass. The PNG is only a preview of what the pen draws.

```
pip install -r requirements.txt
python sketch.py examples/market_hall.jpg --debug           # stage 3 -> out/market_hall.svg, _preview.png
python sketch.py examples/market_hall.jpg --stage 1         # foundation only
python sketch.py examples/market_hall.jpg --stage 2         # line only, no tone
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
