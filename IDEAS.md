# Ideas for later

Ideas noted during development. Each has a first thought on approach; none are started.

## Separate buildings from foreground and background
Split the photo into layers (sky, buildings, vegetation, ground, foreground objects), so each can
be drawn in its own style and at its own level of detail. For example: crisp, detailed lines on
buildings, loose ones on trees, little or nothing on the ground.
- Approach: semantic segmentation (e.g. a SegFormer or Mask2Former model trained on ADE20K or
  Cityscapes, which have building/sky/tree/road/person/car classes), or Segment Anything with
  prompts. Run locally on the GPU.
- The segmentation masks then drive per-region settings in ink mode (thresholds, weight,
  smoothing).

## Clean up unwanted clutter
Remove pedestrians, vehicles, signs, building work and cranes, plant, wires and so on, so the
print shows the place rather than the moment.
- Approach: use the same segmentation (person/car/sign classes) plus a user-supplied mask for
  one-off items like the cranes. Then either inpaint the photo before edge detection (e.g. LaMa),
  so what's behind is drawn plausibly, or just drop the strokes inside the mask and leave paper.
- Inpainting gives cleaner results; dropping strokes is simpler and never invents detail.
- Check licences on any model before selling prints.

## Non-architectural subjects: hills, trees, foliage
The Cotswold street test (`docs/ink/cotswold_*`) shows the current gap: TEED outlines every leaf
cluster, so foliage turns to noise.
- Quick win: weight strokes by TEED confidence. Foliage is faint in TEED's map but is currently
  drawn as bold as building edges.
- Proper fix: draw vegetation regions (from segmentation) the way an illustrator does, with a
  scalloped outline per mass, a few interior texture marks, and shading by hatching or stippling.
  Don't trace every clump.
- Hills: a silhouette line plus sparse tree-mass indications, getting lighter with distance
  (atmospheric perspective).
