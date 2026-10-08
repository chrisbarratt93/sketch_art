# Hugging Face models

Optional models from the Hugging Face Hub, wired into ink mode for testing. None of them is the
default yet: TEED stays the default until one of these proves better on real photos.

```
uv sync --extra hub        # torch, controlnet-aux, transformers (weights download on first use)
```

Weights are cached in `~/.cache/huggingface` (sizes below). Once they're cached you can work
offline with `HF_HUB_OFFLINE=1`. Every model uses an NVIDIA GPU automatically when PyTorch can
see one. Under WSL2 the normal Linux PyTorch wheel from PyPI includes CUDA and uses the Windows
driver, so nothing extra is needed. Check with `uv run python -c "from sketchart import hub; print(hub.device())"`.

## Quick start: compare everything on one photo

```
uv run compare_models.py examples/market_hall.jpg
uv run compare_models.py examples/cotswold_street.webp --scene        # + segmentation and depth
uv run compare_models.py examples/albert_hall.jpg --models teed,lineart,anyline --max-side 0
```

Output goes to `out/compare/<photo>/`:
- `sheet_edges.jpg`: every model's raw edge map, side by side.
- `sheet_ink.jpg`: the ink drawing traced from each map.
- `sheet_scene.jpg`: segmentation layers and depth, with `--scene`.
- Per model: `<model>_edges.png`, `<model>_ink.png`, `<model>.svg`.
- `stats.json`: time, stroke count and pen travel per model.

The script only skips a model if its download fails, so the first run fetches everything.

## Edge models: `--edge-model`

These replace TEED as the map that ink mode traces. The rest of the pipeline is unchanged:
skeletonise, trace, then weight by how boldly the model drew each line.

```
uv run sketch.py examples/market_hall.jpg --edge-model lineart
uv run sketch.py examples/market_hall.jpg --edge-model anyline --hi 0.6 --lo 0.3 --debug
```

| name | model | Hub repo / file | why try it |
|---|---|---|---|
| `teed` | TEED (default) | GitHub `xavysp/TEED` | baseline: urban edges, ignores texture |
| `lineart` | Informative Drawings, "realistic" | `lllyasviel/Annotators` `sk_model.pth` | trained to turn photos into *line drawings*, not edge maps; closest to an artist's choices |
| `lineart_coarse` | same, "coarse" weights | `lllyasviel/Annotators` `sk_model2.pth` | bolder, simpler drawing; may suit the loose style |
| `lineart_anime` | anime line-art U-Net | `lllyasviel/Annotators` `netG.pth` | very clean thin lines; may flatten buildings |
| `hed` | HED (ControlNet retrain) | `lllyasviel/Annotators` `ControlNetHED.pth` | soft, wide strokes; good silhouettes, weak on fine bars |
| `pidinet` | PiDiNet | `lllyasviel/Annotators` `table5_pidinet.pth` | fast, crisp, a bit more texture than TEED |
| `mteed` | TEED fine-tuned for line art (MistoLine) | `TheMistoAI/MistoLine` `Anyline/MTEED.pth` | same network as ours, retrained on line art |
| `anyline` | MTEED + a fine-detail lineart layer | as above | keeps small detail (text, ornament) that TEED drops |

Knobs:
- `--model-scale`: run the model on a copy at this multiple of the photo size. The defaults are
  1.0, or 1.5 for `mteed`/`anyline`. Higher gives finer lines and costs more time and memory.
- `--hi` / `--lo`: hysteresis thresholds on the 0..1 map. Each model has a default (see
  `EDGE_MODELS` in `sketchart/hub.py`), but these were set without seeing real outputs, so tune
  them. Look at `--debug`'s `<name>_<model>.png` and raise `--lo` if there's too much texture.

The edge models run through [`controlnet-aux`](https://pypi.org/project/controlnet-aux/), Hugging
Face's packaging of the ControlNet preprocessors. It depends on `opencv-python-headless`, which
`pyproject.toml` overrides away so it can't replace our `opencv-contrib-python-headless`.

## Scene understanding: `--remove`, `--depth-fade`

These are the first steps towards the layer ideas in [IDEAS.md](IDEAS.md).

- **Segmentation**: OneFormer trained on ADE20K (`shi-labs/oneformer_ade20k_swin_tiny`, or the
  slower, more accurate `--seg-model large`). Its 150 classes are grouped into `sky`, `buildings`,
  `vegetation`, `terrain`, `ground`, `people`, `vehicles` and `clutter` (signs, street lights,
  bins, bollards…); see `GROUPS` in `sketchart/segment.py`.
  - `--remove people,vehicles,clutter` drops every stroke lying mostly inside those layers and
    leaves paper. This is the "just drop the strokes" version of clutter removal; inpainting is
    future work.
  - With `--debug`, `<name>_layers.png` shows the tinted layers.
- **Depth**: Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`).
  - `--depth-fade 0.6` scales each stroke's boldness by nearness, so distant detail drops to the
    fine pen and near edges go heavy. This is a first pass at atmospheric perspective.
  - With `--debug`, `<name>_depth.png` shows the map (bright = near).

```
uv run sketch.py examples/cotswold_street.webp --remove people,vehicles,clutter --depth-fade 0.5 --debug
```

## Licences (matters before selling prints)

Checked from search results, not from every model card. Confirm on each repo before commercial use.

| model | weights licence |
|---|---|
| TEED | MIT |
| Depth Anything V2 **Small** | Apache-2.0. The Base and Large models are CC-BY-NC-4.0 (non-commercial), which is why Small is used. |
| OneFormer ADE20K | MIT per the Hub metadata; a third-party listing disagrees for `swin_tiny`, so confirm. |
| HED (ControlNet retrain) | Released by lllyasviel as an Apache-2.0 reimplementation. Confirm. |
| Informative Drawings (`lineart`) | Check the upstream `carolineec/informative-drawings` repo. |
| PiDiNet, anime lineart, MTEED/Anyline | Not verified. Check each upstream repo / model card. |

NVIDIA's SegFormer ADE20K weights were avoided on purpose: they're licensed for non-commercial
research and evaluation only.

## Not wired in yet (candidates)

- **Inpainting for clutter removal**: LaMa (e.g. an ONNX export on the Hub), so the wall behind a
  removed van is drawn rather than left blank.
- **Segment Anything 2** (`facebook/sam2.1-hiera-*`, Apache-2.0): click-to-mask for one-off items
  like cranes, which ADE20K has no class for.
- **Diffusion restyle**: SDXL with a line-art ControlNet (e.g. MistoLine), turning the photo into
  a pen-drawing image, then tracing that. This could look the most hand-drawn, but needs ~8 GB+
  of VRAM and invents detail, and the licences need checking.
